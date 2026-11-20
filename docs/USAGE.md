# Usage

## Requirements

- One RTX PRO 6000 Blackwell (96 GB, sm_120)
- Host RAM: about 50 GB free for the pinned PLE n-gram table
- CUDA toolkit 13.x (`nvcc`) for the JIT-built kernels; set `CUDA_HOME`
- Python 3.12, Linux x86_64

## Install

```bash
./install.sh                    # VENV=.venv  PYTHON=python3.12
```

The script installs vLLM 0.30.0 and overlays `vllm/`. It also installs b12x 1.5.1 and sm120-lowlat-gemm into `third_party/`.

## Weights

| model | source |
|---|---|
| target | `nvidia/Qwen3.8-Flash-Next-NVFP4` (revision `fc694b54`) |
| drafter | `incoai/Qwen3.8-Flash-Next-DFlash2` |

Convert the dense projections to FP8. The output directory holds the rewritten shards; every other file is symlinked to the source:

```bash
.venv/bin/python tools/make_dense_fp8.py /path/Qwen3.8-Flash-Next-NVFP4 /path/Qwen3.8-Flash-Next-NVFP4-dfp8
```

## Run

```bash
MODEL=/path/Qwen3.8-Flash-Next-NVFP4-dfp8 \
DRAFTER_MODEL=/path/Qwen3.8-Flash-Next-DFlash2 \
scripts/serve_final.sh
```

`scripts/serve_final.sh` loads `scripts/final.env` and then starts `scripts/serve.sh`. Variables that are already set in the environment override the file.

The first start JIT-builds the FlashInfer fused MoE, `gdn_defer` and `moe_w4a8`. `PREBUILD=1 scripts/serve_final.sh` builds only the FlashInfer MoE and then exits.

## Request

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 262144
}'
```

The server speaks the OpenAI chat/completions API, with reasoning (`reasoning_content`), tool calls (`qwen3_coder` parser) and image input.

## Variables

### Server

| variable | default | meaning |
|---|---|---|
| `MODEL` | required | target checkpoint |
| `DRAFTER_MODEL` | required with `DRAFTER=dflash2` | DFlash2 drafter checkpoint |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | listen address |
| `SERVED_NAME` | `qwen3.8-flash` | model name in the API |
| `VENV` | `.venv` | venv with vLLM |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA toolkit for JIT builds |
| `CPUS` | unset | `taskset -c` list for the server |
| `GMU` | `0.92` | `--gpu-memory-utilization` |
| `KVGIB` | unset | fixed KV cache size in GiB (skips profiling) |
| `MAXSEQS` | `2` | `--max-num-seqs` |
| `MNBT` | `16384` | `--max-num-batched-tokens` |
| `PREFIX` | `1` | prefix caching (forced to 0 by `SPEC0=1` without `SPEC0_ALIGN=1`) |
| `MOE` | auto | NVFP4 MoE backend: `b12x`, `cutlass`, `marlin`, … |
| `B12X_TILE` | unset | b12x dynamic MoE tile override |
| `SSMDT` | checkpoint (float32) | GDN state dtype, e.g. `bfloat16` |
| `MMKW` | unset | `'{"device":"cuda"}'` runs the image processor on the GPU |
| `COMPILE_CACHE` | `0` | `1` keeps the vLLM compile cache |
| `EAGER` | `0` | `1` = `--enforce-eager` |
| `PROF` / `PROFDIR` | `0` / `prof/` | torch profiler via `POST /start_profile`, `/stop_profile` |
| `PREBUILD` | `0` | `1` builds the FlashInfer fused MoE and exits |

### Speculative decoding

| variable | default | meaning |
|---|---|---|
| `DRAFTER` | `mtp` | `dflash2`, `ngram` (GPU prompt lookup) or `mtp` |
| `SPEC` | `0` | draft tokens per step (`0` = off for `mtp`, 7 for `dflash2`) |
| `SPEC_SCHED` | unset | `[[lo,hi,k],...]`: verify width k for running-batch sizes lo..hi |
| `DRAFT_SAMPLE` | greedy | `probabilistic` = sampled drafts with standard speculative sampling |
| `REJECT` | unset | rejection method, e.g. `block` |
| `HYBRID` | `0` | `1` = DFlash2 + n-gram extension up to `SPEC` (needs `SPEC0=1`) |
| `QWEN4_HYB_TIERS` | `7,15,23` | verify widths the scheduler steps through (last = `SPEC`) |
| `QWEN4_NGRAM_MAX` / `QWEN4_NGRAM_MIN` | `10` / `3` | n-gram key length range for `HYBRID=1` |
| `NGMAX` / `NGMIN` | `4` / `2` | prompt-lookup key lengths for `DRAFTER=ngram` |
| `DFLASH2_DRAFT_VOCAB` | unset | reduced drafter vocabulary (`assets/draft_vocab_98304.pt`) |
| `DFLASH2_FP8_HEAD` | `0` | FP8 drafter head for candidate top-k |
| `DFLASH2_FP8_LINEAR` | `0` | FP8 drafter projections |
| `DFLASH2_FUSED_CONV` | `0` | fused drafter dynamic convolution |

### State and kernels

| variable | default | meaning |
|---|---|---|
| `SPEC0` | `0` | one GDN/PLE state slot per request (gdn_defer) |
| `SPEC0_ALIGN` | `0` | keep prefix caching with `SPEC0` (mamba align mode) |
| `SPEC0_MIXED` | `0` | draft rows of mixed prefill/decode steps through the deferred kernel |
| `QWEN4_MOE_W4A8_DECODE` | `0` | `moe_w4a8` kernel for MoE calls of 1–64 tokens (needs `MOE=b12x`) |
| `HC_FUSED` | `0` | fused hyper-connection mix |
| `QWEN4_LLG_BA` | `0` | GDN `in_proj_ba` through sm120-lowlat-gemm |

### Prefill scheduler

| variable | default | meaning |
|---|---|---|
| `SCHED` | `0` | `1` = latency-aware prefill scheduler |
| `QWEN38_SCHED_TADD_MS` | `40` | prefill time added to a step that has running decodes |
| `QWEN38_SCHED_TSOLO_MS` | `500` | step time budget when no decode is running |
| `QWEN38_SCHED_CLIN_US` / `QWEN38_SCHED_CATT_NS` | `64.8` / `0.05` (`marlin`: `77.6` / `0.033`) | per-token linear and attention cost |
