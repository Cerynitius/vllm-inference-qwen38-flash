# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import VllmConfig


def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "ngram_gpu":
        # 10-28 Qwen3.8-Flash stack: GPU prompt lookup for the V2 runner
        from vllm.v1.worker.gpu.spec_decode.ngram_gpu_speculator import NgramGpuSpeculator

        return NgramGpuSpeculator(vllm_config, device)
    if speculative_config.method == "extract_hidden_states":
        from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
            ExtractHiddenStatesSpeculator,
        )

        return ExtractHiddenStatesSpeculator(vllm_config, device)
    elif speculative_config.method == "dflash":
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
            if __import__("os").environ.get("QWEN4_HYBRID", "0") == "1":  # 10-28: DFlash2 + n-gram
                from vllm.v1.worker.gpu.spec_decode.dflash2.hybrid import DFlash2NgramSpeculator

                return DFlash2NgramSpeculator(vllm_config, device)
            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
                DFlash2Speculator,
            )

            return DFlash2Speculator(vllm_config, device)
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        return DFlashSpeculator(vllm_config, device)
    elif speculative_config.method == "dspark":
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        return DSparkSpeculator(vllm_config, device)
    elif speculative_config.use_gemma4_mtp():
        from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import (
            Gemma4Speculator,
        )

        return Gemma4Speculator(vllm_config, device)
    elif speculative_config.use_multi_module_mtp():
        from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
            MultiModuleMTPSpeculator,
        )

        return MultiModuleMTPSpeculator(vllm_config, device)
    elif speculative_config.method == "mtp":
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        return MTPSpeculator(vllm_config, device)
    elif speculative_config.use_eagle():
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import (
            EagleSpeculator,
        )

        return EagleSpeculator(vllm_config, device)
    else:
        raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
