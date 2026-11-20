# SPDX-License-Identifier: Apache-2.0
# 10-28 (Qwen3.8-Flash stack): DFlash2 + GPU prompt lookup. The engine verifies up to K = num_speculative_tokens
# (e.g. 27); DFlash2 keeps its trained 8-block (7 drafts, QWEN4_HYB_DK). Per request the n-gram continuation's columns
# DK..K-1 are appended when a match exists and its first DK tokens equal DFlash2's drafts, and all K columns come from
# the n-gram when its key matched >= QWEN4_HYB_TRUST tokens; otherwise DFlash2's drafts are padded with the last one.
# How many of the K columns the target verifies is chosen by the scheduler per step (QWEN4_HYBRID in
# v1/core/sched/scheduler.py). Lossless: every column is checked by the target.
# Two kernels per step: _match (longest suffix match of the last tokens over the window, all key lengths in one pass)
# and _combine (n-gram copy + DFlash2 agreement + output, resets the match key for the next step). With
# draft_sample_method=probabilistic, q of deterministic columns is kept one-hot (_onehot_reset / _onehot_write).
import copy
import os

import torch

from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

_DK = int(os.environ.get("QWEN4_HYB_DK", "7"))
_TRUST = int(os.environ.get("QWEN4_HYB_TRUST", "8"))  # key length from which the n-gram alone is used
_NMAX = int(os.environ.get("QWEN4_NGRAM_MAX", "10"))
_NMIN = int(os.environ.get("QWEN4_NGRAM_MIN", "3"))
_WINDOW = 16384  # search the last 16K tokens
_CHUNK = 1024


@triton.jit
def _match_kernel(idx_ptr, tok_ptr, tok_stride, len_ptr, key_ptr, NMAX: tl.constexpr, WINDOW: tl.constexpr,
                  CHUNK: tl.constexpr):
    # candidate copy start e: match length l(e) = largest t <= NMAX with tok[e-s] == tok[L-s] for all s <= t;
    # key = l << 32 | e, atomic max -> longest match, then the latest occurrence
    r = tl.program_id(0)
    c = tl.program_id(1)
    req = tl.load(idx_ptr + r)
    L = tl.load(len_ptr + req)
    base = tok_ptr + req.to(tl.int64) * tok_stride
    e = tl.maximum(L - WINDOW, 0) + c * CHUNK + tl.arange(0, CHUNK)
    ok = (e < L) & (e >= 1)
    ln = tl.zeros((CHUNK,), dtype=tl.int64)
    for t in tl.static_range(1, NMAX + 1):
        key = tl.load(base + L - t, mask=L - t >= 0, other=-2)
        v = tl.load(base + e - t, mask=ok & (e - t >= 0), other=-1)
        ok = ok & (v == key)
        ln += ok.to(tl.int64)
    packed = tl.where(ln > 0, (ln << 32) | e.to(tl.int64), -1)
    best = tl.max(packed, axis=0)
    if best >= 0:
        tl.atomic_max(key_ptr + r, best)


@triton.jit
def _combine_kernel(idx_ptr, tok_ptr, tok_stride, len_ptr, key_ptr, d_ptr, d_stride, out_ptr, out_stride, det_ptr,
                    DK: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, NMIN: tl.constexpr, TRUST: tl.constexpr):
    r = tl.program_id(0)
    req = tl.load(idx_ptr + r)
    L = tl.load(len_ptr + req)
    packed = tl.load(key_ptr + r)
    tl.store(key_ptr + r, -1)  # ready for the next step
    ln = tl.where(packed >= 0, packed >> 32, 0)
    e = (packed & 0xFFFFFFFF).to(tl.int32)
    matched = ln >= NMIN
    k = tl.arange(0, KP)
    pos = e + k
    have = matched & (pos < L) & (k < K)
    base = tok_ptr + req.to(tl.int64) * tok_stride
    ng = tl.load(base + pos, mask=have, other=-1)
    d = tl.load(d_ptr + r * d_stride + k, mask=k < DK, other=-1)
    d_last = tl.load(d_ptr + r * d_stride + DK - 1)
    agree = tl.sum(((ng == d) & (k < DK)).to(tl.int32), axis=0) == DK
    trust = ln >= TRUST
    use = trust | (matched & agree)
    head = tl.where(trust & (ng >= 0), ng, d)
    tail = tl.where(use & (ng >= 0), ng, d_last)
    o = tl.where(k < DK, head, tail)
    tl.store(out_ptr + r * out_stride + k, o, mask=k < K)
    # columns whose draft is not a DFlash2 sample (n-gram head of a trusted row, every tail column): bit k
    det = ((k >= DK) | (trust & (ng >= 0))) & (k < K)
    tl.store(det_ptr + r, tl.sum(tl.where(det, 1 << k, 0), axis=0))


@triton.jit
def _onehot_reset_kernel(idx_ptr, dl_ptr, dl_s0, dl_s1, prev_ptr, K: tl.constexpr, KP: tl.constexpr):
    # before DFlash2 caches this step's candidates: drop last step's one-hot cells of these requests
    r = tl.program_id(0)
    req = tl.load(idx_ptr + r).to(tl.int64)
    k = tl.arange(0, KP)
    prev = tl.load(prev_ptr + req * K + k, mask=k < K, other=-1)
    tl.store(dl_ptr + req * dl_s0 + k * dl_s1 + prev, float("-inf"), mask=(k < K) & (prev >= 0))
    tl.store(prev_ptr + req * K + k, -1, mask=k < K)


@triton.jit
def _onehot_write_kernel(idx_ptr, out_ptr, out_stride, det_ptr, dl_ptr, dl_s0, dl_s1, prev_ptr, cand_ptr,
                         DK: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, TOPK: tl.constexpr,
                         TOPKP: tl.constexpr):
    # deterministic columns get q = one-hot at their draft token (exact for prompt-lookup proposals); in DFlash2's
    # columns that means clearing the candidates it cached for this request first
    r = tl.program_id(0)
    req = tl.load(idx_ptr + r).to(tl.int64)
    det = tl.load(det_ptr + r)
    for c in tl.static_range(DK):
        if (det >> c) & 1:
            j = tl.arange(0, TOPKP)
            cand = tl.load(cand_ptr + (req * DK + c) * TOPK + j, mask=j < TOPK, other=0)
            tl.store(dl_ptr + req * dl_s0 + c * dl_s1 + cand, float("-inf"), mask=j < TOPK)
    k = tl.arange(0, KP)
    isdet = ((det >> k) & 1) == 1
    tok = tl.load(out_ptr + r * out_stride + k, mask=k < K, other=0)
    m = isdet & (k < K)
    tl.store(dl_ptr + req * dl_s0 + k * dl_s1 + tok, 0.0, mask=m)
    tl.store(prev_ptr + req * K + k, tok.to(tl.int32), mask=m)


class DFlash2NgramSpeculator(DFlash2Speculator):
    _speculator_name = "DFlash2+ngram"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        sc = vllm_config.speculative_config
        self.hyb_k = sc.num_speculative_tokens
        assert self.hyb_k > _DK, f"hybrid needs num_speculative_tokens > {_DK}"
        sc_d = copy.copy(sc)
        object.__setattr__(sc_d, "num_speculative_tokens", _DK)
        object.__setattr__(sc_d, "num_speculative_tokens_per_batch_size", None)
        cfg_d = copy.copy(vllm_config)
        object.__setattr__(cfg_d, "speculative_config", sc_d)
        super().__init__(cfg_d, device)
        self.hyb_tokens = torch.zeros(self.max_num_reqs, self.hyb_k, dtype=torch.int64, device=device)
        self.hyb_key = torch.full((self.max_num_reqs,), -1, dtype=torch.int64, device=device)
        self.req_states = None  # the runner binds lazily when this attribute is None
        self.hyb_det = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self._dl_big = None
        if self.draft_logits is not None:
            # probabilistic drafts: the rejection sampler reads q for all K columns -> [reqs, K, V], DFlash2 caches its
            # candidate scores into the first DK columns (view), deterministic columns are one-hot (sparse cells)
            shape = (self.max_num_reqs, self.hyb_k, self.draft_logits.size(-1))
            dtype, fill = self.draft_logits.dtype, self.draft_logits[0, 0, 0].item()
            self.draft_logits = None
            torch.cuda.empty_cache()
            self._dl_big = torch.full(shape, fill, dtype=dtype, device=device)
            self._dl_small = self._dl_big[:, :_DK]
            self.draft_logits = self._dl_big
            self.hyb_prev = torch.full((self.max_num_reqs, self.hyb_k), -1, dtype=torch.int32, device=device)

    def _cache_draft_logits(self, candidate_ids: torch.Tensor, num_sample: int) -> None:
        self.draft_logits = self._dl_small
        try:
            super()._cache_draft_logits(candidate_ids, num_sample)
        finally:
            self.draft_logits = self._dl_big

    def bind_req_states(self, req_states) -> None:
        self.req_states = req_states

    def propose(self, input_batch, *args, **kwargs) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        live = not kwargs.get("dummy_run", False) and self.req_states is not None and num_reqs > 0
        KP = triton.next_power_of_2(self.hyb_k)
        dl = self._dl_big
        if live and dl is not None:
            _onehot_reset_kernel[(num_reqs,)](input_batch.idx_mapping, dl, dl.stride(0), dl.stride(1), self.hyb_prev,
                                              K=self.hyb_k, KP=KP)
        d = super().propose(input_batch, *args, **kwargs)
        out = self.hyb_tokens[:num_reqs]
        if not live:
            out[:, :_DK].copy_(d)
            out[:, _DK:].copy_(d[:, -1:])
            return out
        rs = self.req_states
        tok = rs.all_token_ids.gpu
        idx = input_batch.idx_mapping
        lens = rs.total_len.gpu
        _match_kernel[(num_reqs, triton.cdiv(_WINDOW, _CHUNK))](
            idx, tok, tok.stride(0), lens, self.hyb_key, NMAX=_NMAX, WINDOW=_WINDOW, CHUNK=_CHUNK)
        _combine_kernel[(num_reqs,)](
            idx, tok, tok.stride(0), lens, self.hyb_key, d, d.stride(0), out, out.stride(0), self.hyb_det,
            DK=_DK, K=self.hyb_k, KP=KP, NMIN=_NMIN, TRUST=_TRUST)
        if dl is not None:
            _onehot_write_kernel[(num_reqs,)](
                idx, out, out.stride(0), self.hyb_det, dl, dl.stride(0), dl.stride(1), self.hyb_prev,
                self._cached_candidate_ids, DK=_DK, K=self.hyb_k, KP=KP, TOPK=self.selector_top_k,
                TOPKP=triton.next_power_of_2(self.selector_top_k))
        return out
