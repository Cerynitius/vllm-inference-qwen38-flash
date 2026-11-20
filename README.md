# vllm-inference-qwen38-flash

Qwen3.8-Flash-Next (NVFP4) served on a single RTX PRO 6000 Blackwell with vLLM 0.30.0. The stack adds speculative decoding with a block-diffusion drafter and an n-gram extender, single-slot GDN state for draft tokens, hand-written sm_120a kernels and a latency-aware prefill scheduler. Expert weights stay on the GPU; the PLE n-gram table sits in pinned host memory.

Usage: [docs/USAGE.md](docs/USAGE.md)

## Performance

All numbers were measured on an RTX PRO 6000 Blackwell Workstation Edition with a 400 W power limit, using the `scripts/final.env` configuration unless a row says otherwise.

**Single stream** (temperature 1.0, official sampling parameters, 3 prompts × 3 runs per class)

| content | tok/s |
|---|---|
| math | 411.5 |
| chat | 222.7 |
| code from scratch | 418.8 |
| code edit with context (whole module, add type hints) | 640 |

**Concurrency** (~2K-token prompts, 512 output tokens each)

| concurrent requests | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| aggregate output tok/s | 229 | 344 | 505 | 724 | 957 | 1244 | 1130 | 1074 |

The KV pool holds 519K tokens. At c ≥ 48 the pool caps the number of running requests, and the remaining requests queue.

**Long context** (single stream, drafter fixed at k = 7)

| context | 1K | 32K | 128K | 240K |
|---|---|---|---|---|
| decode step | 13.0 ms | 13.1 ms | 13.1 ms | 12.8 ms |
| cold prefill | 0.1 s | 2.3 s | 9.7 s | 18.7 s |

Decode step time does not grow with context length because QSA sparse attention reads a fixed budget. A follow-up turn on a cached 128K conversation starts in 0.33 s. A 2048 px image reaches its first token in 320 ms.

**Prefill next to running chats**

| scenario | stock vLLM scheduler | this scheduler |
|---|---|---|
| 4 chat streams during a 128K prefill | 6–10 tok/s, longest stall 0.49 s | 16–19 tok/s, longest stall 0.15 s |
| 32K prompt arriving with a 128K prompt (32K TTFT) | 11.1 s | 3.0 s |
| short request during a 256K prefill (TTFT) | 1.42 s | 0.83 s |

**Output quality**

| check | result |
|---|---|
| decode path vs prefill recompute, tokens that are not the prefill argmax | 0.98 % (99 / 10,140) |
| same check, b12x W4A4 decode MoE (reference noise level) | 1.04 % |
| final stack vs same-precision baseline, teacher-forced ΔNLL | −0.0010 ± 0.0026 |
| single-slot GDN state (SPEC0) vs stock, greedy decode | 6,949 / 6,949 tokens identical |

The dense projections are converted to FP8 (GDN and attention projections, shared experts) by `tools/make_dense_fp8.py`. This conversion is lossy. It is not part of the equivalence rows above, which compare against a baseline that uses the same precision.

## What is inside

- **Speculative decoding.** A DFlash2 block-diffusion drafter proposes 7 tokens, and a GPU n-gram matcher extends them up to 27 when the text repeats. The verify width steps 7 → 15 → 27 while every draft is accepted. Drafts are sampled probabilistically, so the output distribution is unchanged. The draft count per step shrinks as the batch grows (7 / 5 / 3 / 2).
- **SPEC0.** GDN and PLE state keep one slot per request instead of one per draft token. `gdn_defer` verifies against that slot and writes the accepted state back in place. Per-request fixed KV drops from 11.9 % of the pool to 2.1 %, and prefix caching still works.
- **Kernels (sm_120a).**
  - `moe_w4a8`: decode MoE for 1–64 tokens, NVFP4 weights × MXFP8 activations. Persistent producer/consumer kernel with TMA bulk copies; deterministic.
  - `gdn_defer`: GDN verify and materialize kernels for 8- and 28-token steps.
  - `hc_fused`: hyper-connection mix in two kernels instead of five.
  - Drafter: fused dynamic convolution (bit-identical) and FP8 projections, plus a 98,304-token draft vocabulary.
  - The GDN `in_proj_ba` GEMM runs through [sm120-lowlat-gemm](https://github.com/Cerynitius/sm120-lowlat-gemm).
- **MoE.** Prefill uses b12x W4A4. Decode uses `moe_w4a8`.
- **Scheduler.** The latency-aware prefill scheduler sizes prefill chunks from a cost model fitted on this card, so running chats keep a minimum speed while long prompts are processed.

## License

Apache-2.0. The files under `vllm/` are modified copies of vLLM 0.30.0 files (Apache-2.0); see [NOTICE](NOTICE).
