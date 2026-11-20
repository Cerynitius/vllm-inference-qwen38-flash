# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import maybe_prefix


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


_FUSED_CONV = None
_DRAFT_VOCAB = os.environ.get("DFLASH2_DRAFT_VOCAB", "")
if os.environ.get("DFLASH2_FUSED_CONV", "0") == "1":
    from dflash2_conv import grouped_conv as _FUSED_CONV


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        if _FUSED_CONV is not None and hidden_states.dtype == torch.bfloat16:
            # one Triton kernel, bit-identical to _grouped_conv (kernels/ on PYTHONPATH)
            return _FUSED_CONV(
                hidden_states,
                delta,
                self.base_kernel[side],
                self.block_size,
                self.num_groups,
                self.group_size,
                self.taps,
            )
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            # Query tokens per request: the bonus token plus the mask tokens.
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


@support_torch_compile
class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        # Without its own tag the selector shares the draft head's compile cache.
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlash2Qwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        draft_config = self.config.dflash_config
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.candidate_logits_processor = LogitsProcessor(
            vllm_config.model_config.get_vocab_size(),
            scale=float(draft_config.get("output_multiplier", 1.0)),
            soft_cap=softcap if softcap > 0 else None,
        )
        if os.environ.get("DFLASH2_FP8_LINEAR", "0") == "1":
            # drafter-only FP8 projections (lossless for the served output; ~/qwen38flash-stack/kernels, 10-27)
            from drafter_fp8 import install

            print(f"dflash2: {install(self.model)} drafter projections -> FP8", flush=True)

    def _fp8_head(self, lm_head, hidden_states, embedding_bias=None):
        # 10-27 (DFLASH2_FP8_HEAD=1): the drafter only proposes candidates (the target verifies with its own
        # BF16 head, so output stays lossless); an FP8 per-row copy of the shared head halves the 1.27 GB read.
        from vllm import _custom_ops as ops

        if self._head_fp8 is None:
            if torch.cuda.is_current_stream_capturing():
                # built on the first eager (profiling) call; never allocate inside a graph capture
                return lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
            w = lm_head.weight
            if _DRAFT_VOCAB:
                # 10-28 DFLASH2_DRAFT_VOCAB: head rows of the most frequent tokens only (lossless for the output)
                self._vocab = torch.load(_DRAFT_VOCAB).to(device=w.device, dtype=torch.int64)
                w = w[self._vocab]
            fmax = torch.finfo(torch.float8_e4m3fn).max
            w8 = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
            scale = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
            for r in range(0, w.shape[0], 8192):  # chunked: no full-size fp32 temporary in the profiled peak
                wc = w[r : r + 8192].float()
                sc = (wc.abs().amax(dim=1) / fmax).clamp(min=1e-12)
                w8[r : r + 8192] = (wc / sc[:, None]).to(torch.float8_e4m3fn)
                scale[r : r + 8192] = sc
            self._head_fp8 = (w8.t(), scale.view(1, -1).contiguous())
        w8t, ws = self._head_fp8
        xq, xs = ops.scaled_fp8_quant(
            hidden_states.contiguous(), use_per_token_if_dynamic=True
        )
        return ops.cutlass_scaled_mm(xq, w8t, xs, ws, hidden_states.dtype)

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if os.environ.get("DFLASH2_FP8_HEAD", "0") == "1":
            if not hasattr(self, "_head_fp8"):
                self._head_fp8 = None
                self.candidate_logits_processor._apply_head = self._fp8_head
            if _DRAFT_VOCAB:
                lp = self.candidate_logits_processor
                logits = self._fp8_head(self.lm_head, hidden_states)
                if not hasattr(self, "_vocab"):  # graph-capture fallback before the copy exists
                    return lp.get_top_k_tokens(self.lm_head, hidden_states, self.model.candidate_selector.top_k)
                values, idx = torch.topk(logits, self.model.candidate_selector.top_k, dim=-1)
                ids = self._vocab[idx]
                values = values.float()
                if lp.scale != 1.0:
                    values = values * lp.scale
                if lp.soft_cap is not None:
                    values = torch.tanh(values / lp.soft_cap) * lp.soft_cap
                return ids, values
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k
        )


EntryClass = DFlash2Qwen3ForCausalLM
