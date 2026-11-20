#!/bin/bash
# Qwen3.8-Flash-Next (NVFP4) on one RTX PRO 6000 Blackwell with vLLM 0.30.0 + the overlay in ../vllm.
#   GPU:  routed experts NVFP4 + dense layers (BF16, or FP8 in the -dfp8 checkpoint) + vision + drafter
#   host: PLE n-gram table (pinned, looked up through UVA, prefetched while the previous layer runs)
# All switches are environment variables; scripts/final.env holds the measured configuration (see docs/USAGE.md).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-$HERE/.venv}; V=$VENV/bin
HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-8000}"
SPEC="${SPEC:-0}"; PREFIX="${PREFIX:-1}"; GMU="${GMU:-0.92}"
MODEL="${MODEL:?MODEL=<path to Qwen3.8-Flash-Next-NVFP4 or its -dfp8 copy>}"

# JIT toolchain (FlashInfer fused MoE, gdn_defer, moe_w4a8): CUDA 13.x nvcc + the cuBLAS/cuRAND/NVRTC headers and
# libs of the venv's nvidia/cu13 wheels (linked into third_party/cuda-extra by install.sh)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
CX=$HERE/third_party/cuda-extra
export FLASHINFER_EXTRA_LDFLAGS="-L$CUDA_HOME/lib -L$CUDA_HOME/lib64 -L$CX/lib"
export FLASHINFER_EXTRA_CUDAFLAGS="-I$CX/include"
export FLASHINFER_EXTRA_CFLAGS="-I$CX/include -I$CUDA_HOME/targets/x86_64-linux/include"
export MAX_JOBS=${MAX_JOBS:-8}
export PATH="$V:$CUDA_HOME/bin:$PATH"
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$HERE/.cache/vllm}
# compile cache off unless COMPILE_CACHE=1 (a stale cached graph survives code changes)
[[ ${COMPILE_CACHE:-0} != 1 ]] && export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_PLE_CPU_OFFLOAD=1
export QWEN27B_KERNELS=$HERE/kernels
export PYTHONPATH=$HERE/kernels${PYTHONPATH:+:$PYTHONPATH}

# SPEC0=1: no per-draft-token GDN/PLE state slots. gdn_defer verifies from the request's single state slot and
#   writes the accepted state back in place. Without SPEC0_ALIGN it needs prefix caching off (forces PREFIX=0).
# SPEC0_ALIGN=1: keeps prefix caching (mamba align mode); block-boundary states are stored straight into the
#   aligned block.  SPEC0_MIXED=1: draft rows of mixed prefill/decode steps also go through the deferred kernel.
if [[ ${SPEC0:-0} == 1 ]]; then
  export QWEN4_SPEC0=1 QWEN4_SPEC0_K=${SPEC:-7} QWEN38_GDN_DEFER=1
  if [[ ${SPEC0_ALIGN:-0} == 1 ]]; then export QWEN4_SPEC0_ALIGN=1; else PREFIX=0; fi
fi
[[ ${SPEC0_MIXED:-0} == 1 ]] && export QWEN4_SPEC0_MIXED=1
# HYBRID=1 (DRAFTER=dflash2, SPEC0=1): DFlash2 drafts 7 tokens, a GPU n-gram matcher extends them up to SPEC;
#   the scheduler steps the verify width through QWEN4_HYB_TIERS (last = SPEC).
if [[ ${HYBRID:-0} == 1 ]]; then
  export QWEN4_HYBRID=1 QWEN4_GDN_MTP_MAX=$((SPEC + 1)) GDN_DEFER_NVCC=-DGDN_MAX_MTP=$((SPEC + 1))
fi
[[ ${HC_FUSED:-0} == 1 ]] && export QWEN4_HC_FUSED=1   # fused small-batch hyper-connection mix (kernels/hc_fused.py)
# SCHED=1: latency-aware prefill scheduler (vllm/v1/core/sched/qwen38_sched.py); cost per chunk of n tokens at
#   offset s = n*(C_LIN + C_ATT*(s+n/2)), fitted on the PRO 6000
if [[ ${SCHED:-0} == 1 ]]; then
  export QWEN38_SCHED=1 QWEN38_SCHED_TADD_MS=${QWEN38_SCHED_TADD_MS:-40} QWEN38_SCHED_TSOLO_MS=${QWEN38_SCHED_TSOLO_MS:-500}
  if [[ ${MOE:-} == marlin ]]; then : "${QWEN38_SCHED_CLIN_US:=77.6}" "${QWEN38_SCHED_CATT_NS:=0.033}"
  else : "${QWEN38_SCHED_CLIN_US:=64.8}" "${QWEN38_SCHED_CATT_NS:=0.05}"; fi
  export QWEN38_SCHED_CLIN_US QWEN38_SCHED_CATT_NS
fi
# QWEN4_LLG_BA=1: GDN in_proj_ba GEMM through sm120-lowlat-gemm (third_party/sm120-lowlat-gemm)
if [[ ${QWEN4_LLG_BA:-0} == 1 ]]; then
  export PYTHONPATH=$HERE/third_party/sm120-lowlat-gemm:$PYTHONPATH LLG_PREFETCH=0
fi
EXTRA=()
[[ -n ${KVGIB:-} ]] && EXTRA+=(--kv-cache-memory-bytes "$(python3 -c "print(int($KVGIB*2**30))")")
# DRAFTER=dflash2: block-diffusion drafter (DRAFTER_MODEL, block 8); SPEC_SCHED='[[lo,hi,k],...]' sets the verify
#   width per running-batch-size range; DRAFT_SAMPLE=probabilistic / REJECT=block pick draft sampling / rejection
if [[ ${DRAFTER:-mtp} == dflash2 ]]; then
  EXTRA+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFTER_MODEL:?DRAFTER_MODEL=<path to Qwen3.8-Flash-Next-DFlash2>}\",\"num_speculative_tokens\":${SPEC/#0/7},\"disable_eagle_block_drop\":true${DRAFT_SAMPLE:+,\"draft_sample_method\":\"$DRAFT_SAMPLE\"}${REJECT:+,\"rejection_sample_method\":\"$REJECT\"}${SPEC_SCHED:+,\"num_speculative_tokens_per_batch_size\":$SPEC_SCHED}}")
elif [[ ${DRAFTER:-mtp} == ngram ]]; then
  # GPU prompt lookup only, up to SPEC tokens (SPEC>7 needs GDN_DEFER_NVCC=-DGDN_MAX_MTP=16)
  EXTRA+=(--speculative-config "{\"method\":\"ngram_gpu\",\"num_speculative_tokens\":$SPEC,\"prompt_lookup_max\":${NGMAX:-4},\"prompt_lookup_min\":${NGMIN:-2}${SPEC_SCHED:+,\"num_speculative_tokens_per_batch_size\":$SPEC_SCHED}}")
elif [[ $SPEC != 0 ]]; then
  EXTRA+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC}")
fi
if [[ $PREFIX == 1 ]]; then EXTRA+=(--enable-prefix-caching); else EXTRA+=(--no-enable-prefix-caching); fi
# MOE=b12x|cutlass|marlin|... overrides the NVFP4 MoE backend (auto = FLASHINFER_CUTLASS on sm120)
[[ -n ${MOE:-} ]] && EXTRA+=(--moe-backend "$MOE")
[[ ${MOE:-} == b12x ]] && export PYTHONPATH=$HERE/third_party/b12x:$PYTHONPATH
[[ -n ${B12X_TILE:-} ]] && export B12X_DYNAMIC_TILE_MN=$B12X_TILE
# SSMDT=bfloat16: GDN state dtype (checkpoint: float32; bf16 is not equivalent but halves the per-sequence state)
[[ -n ${SSMDT:-} ]] && EXTRA+=(--mamba-ssm-cache-dtype "$SSMDT")
# MMKW='{"device":"cuda"}': HF image processor on the GPU (faster image TTFT, ~2 GiB less KV pool)
[[ -n ${MMKW:-} ]] && EXTRA+=(--mm-processor-kwargs "$MMKW") && export QWEN4_MMPROC_GPU=1
[[ ${EAGER:-0} == 1 ]] && EXTRA+=(--enforce-eager)
[[ ${PROF:-0} == 1 ]] && EXTRA+=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"'"${PROFDIR:-$HERE/prof}"'","torch_profiler_with_stack":false}')
PIN=(); [[ -n ${CPUS:-} ]] && PIN=(taskset -c "$CPUS")
# PREBUILD=1: only JIT-compile the sm120 fused-MoE op (toolchain errors show in a minute, not after the model load)
if [[ ${PREBUILD:-0} == 1 ]]; then
  exec "${PIN[@]}" "$V/python" -c "from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module as g; g().build_and_load(); print('PREBUILD OK')"
fi
exec "${PIN[@]}" "$V/vllm" serve "$MODEL" \
  --served-model-name "${SERVED_NAME:-qwen3.8-flash}" \
  --host "$HOST" --port "$PORT" \
  --quantization modelopt \
  --engram-config '{"cpu_offload": true}' \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --seed 0 \
  --gpu-memory-utilization "$GMU" \
  --max-num-seqs "${MAXSEQS:-2}" \
  --max-num-batched-tokens "${MNBT:-16384}" \
  --enable-chunked-prefill --no-enable-flashinfer-autotune \
  "${EXTRA[@]}"
