#!/usr/bin/env python3
"""Build Qwen3.8-Flash-Next-NVFP4-dfp8: the official NVFP4 checkpoint with part of the BF16 dense layers
re-quantized to FP8 per-channel weights + dynamic per-token activations (ModelOpt FP8_PER_CHANNEL_PER_TOKEN).

Quantized: GDN in_proj_qkv / in_proj_z / out_proj, attention q/k/v/o,
shared-expert gate/up/down.  Kept BF16: HC mixers, router gate, shared_expert_gate, GDN in_proj_a/b + conv1d,
QSA indexer, PLE, lm_head, embed_tokens, vision.  Weight scale = row absmax / 448 (e4m3 max), fp32.
Shards holding a target tensor are rewritten (targets replaced); all other files are symlinked.

  make_dense_fp8.py SRC DST
"""
import json, os, re, sys
import torch
from safetensors.torch import load_file, save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
T = re.compile(r"^(model\.language_model\.layers\.\d+\.(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
               r"|self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.shared_expert\.(gate_proj|up_proj|down_proj)))\.weight$")
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
wmap = idx["weight_map"]
shards = sorted({f for k, f in wmap.items() if T.match(k)})
modules = sorted(T.match(k).group(1) for k in wmap if T.match(k))

for name in os.listdir(src):
    if name in shards or name in ("model.safetensors.index.json", "hf_quant_config.json", "config.json"):
        continue
    p = os.path.join(dst, name)
    if not os.path.lexists(p):
        os.symlink(os.path.join(src, name), p)

for sh in shards:
    out = os.path.join(dst, sh)
    if os.path.exists(out) and not os.path.islink(out):
        print("skip (done)", sh, flush=True)
        continue
    t = load_file(os.path.join(src, sh))
    for k in [k for k in t if T.match(k)]:
        w = t[k].float()
        scale = (w.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
        t[k] = (w / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        t[k[: -len(".weight")] + ".weight_scale"] = scale.to(torch.float32)
        wmap[k[: -len(".weight")] + ".weight_scale"] = sh
    save_file(t, out + ".part", metadata={"format": "pt"})
    os.rename(out + ".part", out)
    fd = os.open(out, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    fd = os.open(os.path.join(src, sh), os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    print("rewrote", sh, flush=True)
    del t

json.dump(idx, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)


def fix_excludes(lst):
    out = []
    for x in lst:
        m = re.match(r"^(model\.language_model\.layers\.\d+\.)(linear_attn|self_attn|mlp\.shared_expert)\*$", x)
        if not m:
            out.append(x)
        elif m.group(2) == "linear_attn":
            out += [m.group(1) + "linear_attn." + s for s in ("in_proj_a", "in_proj_b", "in_proj_ba", "conv1d")]
        elif m.group(2) == "self_attn":
            out.append(m.group(1) + "self_attn.indexer*")
        # mlp.shared_expert* dropped (shared_expert_gate keeps its own entry)
    return out


algo = {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"}
hq = json.load(open(os.path.join(src, "hf_quant_config.json")))
hq["quantization"]["exclude_modules"] = fix_excludes(hq["quantization"]["exclude_modules"])
hq["quantization"]["quantized_layers"].update({m: dict(algo) for m in modules})
json.dump(hq, open(os.path.join(dst, "hf_quant_config.json"), "w"), indent=2)
cfg = json.load(open(os.path.join(src, "config.json")))
qc = cfg["quantization_config"]
if "ignore" in qc:
    qc["ignore"] = fix_excludes(qc["ignore"])
qc.setdefault("quantized_layers", {}).update({m: dict(algo) for m in modules})
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
print("DONE", len(modules), "modules ->", algo["quant_algo"], flush=True)
