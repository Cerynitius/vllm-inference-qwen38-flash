"""FP8 (per-output-channel weight, dynamic per-token activation) for the DFlash2 drafter's dense projections, 10-27.

The drafter only proposes tokens; the target verifies with its own weights, so the served output distribution is
unchanged (acceptance may move slightly). Conversion happens on the first eager call (vLLM's profiling run), after
which the BF16 weight is released; a call inside a CUDA-graph capture before that falls back to BF16.
"""
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


class LazyFp8LinearMethod(UnquantizedLinearMethod):
    def apply(self, layer, x, bias=None):
        if not hasattr(layer, "_w8"):
            if torch.cuda.is_current_stream_capturing():
                return super().apply(layer, x, bias)
            w = layer.weight.data
            scale = (w.float().abs().amax(dim=1) / _FP8_MAX).clamp(min=1e-12)
            layer._w8 = (w.float() / scale[:, None]).to(torch.float8_e4m3fn).t()
            layer._ws = scale.view(1, -1).contiguous()
            layer.weight.data = torch.empty(0, dtype=w.dtype, device=w.device)
        x2 = x.reshape(-1, x.shape[-1])
        xq, xs = ops.scaled_fp8_quant(x2.contiguous(), use_per_token_if_dynamic=True)
        y = ops.cutlass_scaled_mm(xq, layer._w8, xs, layer._ws, x.dtype, bias=bias)
        return y.view(*x.shape[:-1], y.shape[-1])


def install(module, suffixes=("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "fc")) -> int:
    n = 0
    for name, m in module.named_modules():
        if name.rsplit(".", 1)[-1] in suffixes and type(getattr(m, "quant_method", None)) is UnquantizedLinearMethod:
            m.quant_method = LazyFp8LinearMethod()
            n += 1
    return n
