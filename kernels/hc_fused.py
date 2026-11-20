"""Fused HyperConnection mix for small batches (Qwen3.8-Flash decode / spec verify), 10-28.

Stock GatedResidual.mix per call (M <= 16 tokens):  down GEMM (cuBLAS split-K) -> splitK reduce -> hc_silu ->
up GEMM -> hc_gate_mix  = 5 kernels, ~17 us + gaps, 96 calls per step.  Here:
  K1 hc_down_silu   : split-K down projection, the last CTA of each N tile reduces the partials in a fixed order
                      (deterministic), rounds to bf16 like the stock GEMM output, applies hc_silu to the low-rank
                      columns and returns the injection columns.
  K2 hc_up_gatemix  : up projection of the 4 HC slices of one output tile in one CTA, rounded to bf16, followed
                      by the hc_gate_mix epilogue (sum over streams of sigmoid(gate) * xn, / HC).
Numerics: fp32 accumulation, different reduction order than cuBLAS -> tier B (checked against fp64).
"""
import torch
from vllm.triton_utils import tl, triton

_BM = 16
import os as _os
_SPLITS = int(_os.environ.get("HCF_S", "20"))
_BK1 = int(_os.environ.get("HCF_BK1", "256"))
_W1 = int(_os.environ.get("HCF_W1", "8"))
_BN2 = int(_os.environ.get("HCF_BN2", "8"))
_W2 = int(_os.environ.get("HCF_W2", "4"))


@triton.jit
def _rne_bf16(x):
    b = x.to(tl.uint32, bitcast=True)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _down_silu_kernel(x_ptr, w_ptr, part_ptr, cnt_ptr, lora_ptr, inj_ptr, M, N, K, R, NI,
                      stride_x, stride_w, stride_l, stride_i,
                      S: tl.constexpr, KS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      BK: tl.constexpr, NPAD: tl.constexpr, HC: tl.constexpr):
    s = tl.program_id(0)
    nb = tl.program_id(1)
    offs_m = tl.arange(0, BM)
    offs_n = nb * BN + tl.arange(0, BN)
    mm = offs_m < M
    mn = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in tl.static_range(0, KS, BK):
        offs_k = s * KS + kk + tl.arange(0, BK)
        x = tl.load(x_ptr + offs_m[:, None] * stride_x + offs_k[None, :], mask=mm[:, None], other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * stride_w + offs_k[None, :], mask=mn[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    pp = part_ptr + (s * BM + offs_m[:, None]) * NPAD + offs_n[None, :]
    tl.store(pp, acc)
    done = tl.atomic_add(cnt_ptr + nb, 1, sem="acq_rel")
    if done == S - 1:
        tot = tl.zeros((BM, BN), dtype=tl.float32)
        for ss in range(S):  # fixed order -> run-to-run identical
            tot += tl.load(part_ptr + (ss * BM + offs_m[:, None]) * NPAD + offs_n[None, :])
        v = _rne_bf16(tot)  # stock GEMM output is bf16
        t = v / HC
        y = t * tl.sigmoid(t)
        is_l = offs_n < R
        tl.store(lora_ptr + offs_m[:, None] * stride_l + offs_n[None, :], y.to(lora_ptr.dtype.element_ty),
                 mask=mm[:, None] & is_l[None, :])
        is_i = (offs_n >= R) & (offs_n < R + NI)
        tl.store(inj_ptr + offs_m[:, None] * stride_i + (offs_n - R)[None, :], v.to(inj_ptr.dtype.element_ty),
                 mask=mm[:, None] & is_i[None, :])
        tl.atomic_xchg(cnt_ptr + nb, 0)


@triton.jit
def _up_gatemix_kernel(l_ptr, w_ptr, xn_ptr, o_ptr, M, R, stride_l, stride_w, stride_xn, stride_o,
                       HC_DIM: tl.constexpr, HC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr, RP: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BM)
    mm = offs_m < M
    offs_c = tl.arange(0, HC * BN)            # column c -> stream c // BN, inner pid*BN + c % BN
    rows = (offs_c // BN) * HC_DIM + pid * BN + (offs_c % BN)
    offs_k = tl.arange(0, RP)
    mk = offs_k < R
    a = tl.load(l_ptr + offs_m[:, None] * stride_l + offs_k[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
    w = tl.load(w_ptr + rows[:, None] * stride_w + offs_k[None, :], mask=mk[None, :], other=0.0)
    g = _rne_bf16(tl.dot(a, tl.trans(w)))      # [BM, HC*BN], stock up GEMM output is bf16
    x = tl.load(xn_ptr + offs_m[:, None] * stride_xn + rows[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
    p = tl.reshape(tl.sigmoid(g) * x, (BM, HC, BN))
    mix = tl.sum(p, axis=1) / HC
    offs_i = pid * BN + tl.arange(0, BN)
    tl.store(o_ptr + offs_m[:, None] * stride_o + offs_i[None, :], mix.to(o_ptr.dtype.element_ty),
             mask=mm[:, None])


class _Work:
    def __init__(self):
        self.bufs = {}

    def get(self, n, s, device):
        key = (n, device)
        if key not in self.bufs:
            npad = triton.cdiv(n, 64) * 64
            self.bufs[key] = (torch.empty(s * _BM * npad, dtype=torch.float32, device=device),
                              torch.zeros(npad // 64, dtype=torch.int32, device=device), npad)
        return self.bufs[key]


_WORK = _Work()


def hc_down_silu(xn, w, lora_rank, hc):
    """xn [M<=16, K] bf16, w [N, K] (N = lora_rank [+ hc + pad]) -> (silu'd lora [M, R], injection [M, N-R] or None)."""
    M, K = xn.shape
    N = w.shape[0]
    S = _SPLITS
    assert M <= _BM and K % (S * _BK1) == 0 and xn.stride(1) == 1 and w.stride(1) == 1
    part, cnt, npad = _WORK.get(N, S, xn.device)
    lora = torch.empty((M, lora_rank), dtype=xn.dtype, device=xn.device)
    ni = N - lora_rank
    inj = torch.empty((M, max(ni, 1)), dtype=xn.dtype, device=xn.device)
    _down_silu_kernel[(S, npad // 64)](
        xn, w, part, cnt, lora, inj, M, N, K, lora_rank, ni,
        xn.stride(0), w.stride(0), lora.stride(0), inj.stride(0),
        S=S, KS=K // S, BM=_BM, BN=64, BK=_BK1, NPAD=npad, HC=hc, num_warps=_W1)
    return lora, (inj if ni > 0 else None)


def hc_up_gatemix(lora, w_up, xn, hc):
    """lora [M<=16, R] bf16 (already silu'd), w_up [D, R], xn [M, D] -> block_input [M, D // hc]."""
    M, R = lora.shape
    D = w_up.shape[0]
    hc_dim = D // hc
    assert M <= _BM and lora.stride(1) == 1 and w_up.stride(1) == 1 and xn.stride(1) == 1
    out = torch.empty((M, hc_dim), dtype=xn.dtype, device=xn.device)
    BN = _BN2
    _up_gatemix_kernel[(triton.cdiv(hc_dim, BN),)](
        lora, w_up, xn, out, M, R, lora.stride(0), w_up.stride(0), xn.stride(0), out.stride(0),
        HC_DIM=hc_dim, HC=hc, BM=_BM, BN=BN, BK=64, RP=triton.next_power_of_2(R), num_warps=_W2)
    return out
