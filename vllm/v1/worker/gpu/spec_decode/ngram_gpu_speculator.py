# SPDX-License-Identifier: Apache-2.0
# 10-28 (Qwen3.8-Flash stack): GPU prompt-lookup (n-gram) speculator for the V2 model runner.
# Drafts the K tokens that followed the most recent earlier occurrence of the request's last N tokens
# (N = prompt_lookup_max down to prompt_lookup_min). No draft model; lossless (target verifies).
import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

_WINDOW = 16384  # search the last 16K tokens (all_token_ids lives in pinned host memory, read via UVA)
_CHUNK = 1024


@triton.jit
def _ngram_match_kernel(idx_ptr, tok_ptr, tok_stride, len_ptr, best_ptr, N: tl.constexpr,
                        WINDOW: tl.constexpr, CHUNK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1)
    req = tl.load(idx_ptr + r)
    if req < 0:
        return
    L = tl.load(len_ptr + req)
    base = tok_ptr + req.to(tl.int64) * tok_stride
    lo = tl.maximum(L - WINDOW, 0)
    j = lo + c * CHUNK + tl.arange(0, CHUNK)   # candidate start of an earlier occurrence
    ok = (j + N < L) & (j >= 0)                # occurrence must end before the key and have a successor
    for t in tl.static_range(N):
        key = tl.load(base + L - N + t)
        v = tl.load(base + j + t, mask=ok, other=-1)
        ok = ok & (v == key)
    best = tl.max(tl.where(ok, j + N, -1), axis=0)   # end of the match = first copied position
    if best >= 0:
        tl.atomic_max(best_ptr + r, best)


@triton.jit
def _ngram_copy_kernel(idx_ptr, tok_ptr, tok_stride, len_ptr, best_ptr, out_ptr, out_stride, last_ptr,
                       K: tl.constexpr, KP: tl.constexpr):
    r = tl.program_id(0)
    req = tl.load(idx_ptr + r)
    k = tl.arange(0, KP)
    if req < 0:
        return
    L = tl.load(len_ptr + req)
    best = tl.load(best_ptr + r)
    base = tok_ptr + req.to(tl.int64) * tok_stride
    pos = best + k
    have = (best >= 0) & (pos < L)
    fill = tl.load(last_ptr + req)
    v = tl.load(base + pos, mask=have & (k < K), other=0)
    v = tl.where(have, v, fill)
    tl.store(out_ptr + r * out_stride + k, v, mask=k < K)


class NgramGpuSpeculator(BaseSpeculator):
    supports_mm_inputs = False
    draft_logits = None
    draft_token_confidence_probs = None
    pcp_manager = None

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        sc = vllm_config.speculative_config
        self.num_speculative_steps = sc.num_speculative_tokens
        self.nmax = int(os.environ.get("QWEN4_NGRAM_MAX", 0) or getattr(sc, "prompt_lookup_max", None) or 4)
        self.nmin = int(os.environ.get("QWEN4_NGRAM_MIN", 0) or getattr(sc, "prompt_lookup_min", None) or 2)
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.device = device
        self.draft_tokens = torch.zeros(self.max_num_reqs, self.num_speculative_steps, dtype=torch.int64,
                                        device=device)
        self.best = torch.full((self.max_num_reqs,), -1, dtype=torch.int32, device=device)
        self.best_n = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)  # matched key length, 0 = none
        self.req_states = None
        self.draft_attn_layer_names: set[str] = set()

    def bind_req_states(self, req_states) -> None:
        self.req_states = req_states

    def load_model(self, target_model) -> None:
        pass

    def set_attn(self, *args, **kwargs) -> None:
        pass

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    def capture(self) -> None:
        pass

    def observe_verification(self, *args, **kwargs) -> None:
        pass

    def prepare_watermarking(self, *args, **kwargs) -> None:
        pass

    def propose(self, input_batch, attn_metadata: dict[str, Any], slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature,
                seeds, dp_sync=None, dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None,
                is_profile=False) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        out = self.draft_tokens[:num_reqs]
        if dummy_run or self.req_states is None or num_reqs == 0:
            out.zero_()
            return out
        rs = self.req_states
        tok = rs.all_token_ids.gpu
        idx = input_batch.idx_mapping
        lens = rs.total_len.gpu
        best = self.best[:num_reqs]
        best.fill_(-1)
        best_n = self.best_n[:num_reqs]
        grid = (num_reqs, triton.cdiv(_WINDOW, _CHUNK))
        for n in range(self.nmax, self.nmin - 1, -1):  # longest n first; later n only fill rows still unmatched
            if n == self.nmax:
                _ngram_match_kernel[grid](idx, tok, tok.stride(0), lens, best, N=n, WINDOW=_WINDOW, CHUNK=_CHUNK)
                best_n.zero_().masked_fill_(best >= 0, n)
            else:
                tmp = torch.full_like(best, -1)
                _ngram_match_kernel[grid](idx, tok, tok.stride(0), lens, tmp, N=n, WINDOW=_WINDOW, CHUNK=_CHUNK)
                best_n.masked_fill_((best < 0) & (tmp >= 0), n)
                torch.where(best >= 0, best, tmp, out=best)
        _ngram_copy_kernel[(num_reqs,)](idx, tok, tok.stride(0), lens, best, out, out.stride(0),
                                        rs.last_sampled_tokens, K=self.num_speculative_steps,
                                        KP=triton.next_power_of_2(self.num_speculative_steps))
        return out
