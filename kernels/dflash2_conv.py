"""Fused DFlash2 grouped dynamic convolution (one Triton kernel instead of ~10 aten ops per call), 10-27.

Bit-identical to vllm.model_executor.models.qwen3_dflash2._grouped_conv for bf16: every intermediate is rounded
to bf16 exactly where the eager torch ops round (coefficients = base + delta; tap-0 product; each later tap's
product, its mask multiply and the running sum). The rounding is done with integer round-to-nearest-even on the
fp32 bits, because the compiler otherwise contracts bf16 round trips into fused multiply-adds.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _rne_bf16(x):
    # fp32 -> nearest-even bf16 value, kept in fp32 (finite inputs)
    b = x.to(tl.uint32, bitcast=True)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _grouped_conv_kernel(x_ptr, d_ptr, b_ptr, o_ptr, H, GS: tl.constexpr,
                         stride_x, stride_dt, stride_dtap, stride_o, stride_btap,
                         BLOCK: tl.constexpr, TAPS: tl.constexpr, BLOCK_H: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    m = offs < H
    g = offs // GS
    pos = t % BLOCK
    x0 = tl.load(x_ptr + t * stride_x + offs, mask=m, other=0.0).to(tl.float32)
    c = _rne_bf16(tl.load(b_ptr + offs, mask=m, other=0.0).to(tl.float32)
                  + tl.load(d_ptr + t * stride_dt + g, mask=m, other=0.0).to(tl.float32))
    out = _rne_bf16(c * x0)
    for tap in tl.static_range(1, TAPS):
        c = _rne_bf16(tl.load(b_ptr + tap * stride_btap + offs, mask=m, other=0.0).to(tl.float32)
                      + tl.load(d_ptr + t * stride_dt + tap * stride_dtap + g, mask=m, other=0.0).to(tl.float32))
        xs = tl.load(x_ptr + (t - tap) * stride_x + offs, mask=m & (t >= tap), other=0.0).to(tl.float32)
        p = _rne_bf16(c * xs) * (pos >= tap).to(tl.float32)
        out = _rne_bf16(out + p)
    tl.store(o_ptr + t * stride_o + offs, out.to(o_ptr.dtype.element_ty), mask=m)


def grouped_conv(hidden_states, delta, base, block_size, num_groups, group_size, taps):
    """hidden_states [T, H] bf16; delta [T, taps, G] (last dim contiguous); base [taps, H]."""
    assert hidden_states.dtype == torch.bfloat16
    T, H = hidden_states.shape
    x = hidden_states if hidden_states.stride(-1) == 1 else hidden_states.contiguous()
    assert delta.stride(-1) == 1 and base.stride(-1) == 1
    out = torch.empty((T, H), dtype=x.dtype, device=x.device)
    if T == 0:
        return out
    BLOCK_H = 1024
    _grouped_conv_kernel[(T, triton.cdiv(H, BLOCK_H))](
        x, delta, base, out, H, group_size,
        x.stride(0), delta.stride(0), delta.stride(1), out.stride(0), base.stride(0),
        BLOCK=block_size, TAPS=taps, BLOCK_H=BLOCK_H, num_warps=4)
    return out
