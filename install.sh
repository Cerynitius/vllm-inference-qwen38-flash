#!/bin/bash
# Creates the venv, installs vLLM 0.30.0, overlays the modified vLLM files from ./vllm, and puts the two
# kernel packages that are kept outside the venv into third_party/:
#   b12x 1.5.1 (pins nvidia-cutlass-dsl 4.8.0, vLLM 0.30.0 ships 4.7.1)
#   sm120-lowlat-gemm (GitHub, commit d08aab6)
#   VENV    venv dir (.venv)          PYTHON  interpreter for the venv (python3.12)
set -eu
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=${VENV:-$HERE/.venv}
PYTHON=${PYTHON:-python3.12}
[[ -x $VENV/bin/python ]] || "$PYTHON" -m venv "$VENV"
P=$VENV/bin/python
"$P" -m pip install -U pip
"$P" -m pip install "vllm==0.30.0" "flashinfer-python==0.6.18.post1" cuda-python ninja
SP=$("$P" -c "import os, vllm; print(os.path.dirname(vllm.__file__))")
[[ $("$P" -c "import vllm; print(vllm.__version__)") == 0.30.0* ]] || { echo "vllm 0.30.0 expected"; exit 1; }
(cd "$HERE/vllm" && find . -name "*.py") | while read -r f; do
  install -D -m 644 "$HERE/vllm/$f" "$SP/$f"
done
"$P" -m pip install --no-deps --target "$HERE/third_party/b12x" "b12x==1.5.1" \
  "nvidia-cutlass-dsl==4.8.0" "nvidia-cutlass-dsl-libs-base==4.8.0" "nvidia-cutlass-dsl-libs-core==4.8.0" \
  "nvidia-cutlass-dsl-libs-cu13==4.8.0"
LL=$HERE/third_party/sm120-lowlat-gemm
[[ -d $LL/.git ]] || git clone https://github.com/Cerynitius/sm120-lowlat-gemm "$LL"
git -C "$LL" checkout -q d08aab6
# cuBLAS / cuRAND / NVRTC headers and libs from the venv's CUDA 13 wheels, for the FlashInfer JIT
NV=$("$P" -c "import os, nvidia; print(os.path.join(list(nvidia.__path__)[0], 'cu13'))")
CX=$HERE/third_party/cuda-extra
mkdir -p "$CX/include" "$CX/lib"
for h in "$NV"/include/cublas*.h "$NV"/include/curand*.h "$NV"/include/nvrtc.h; do ln -sf "$h" "$CX/include/"; done
ln -sf "$NV/lib/libcublas.so.13" "$CX/lib/libcublas.so"
ln -sf "$NV/lib/libcublasLt.so.13" "$CX/lib/libcublasLt.so"
ln -sf "$NV/lib/libnvrtc.so.13" "$CX/lib/libnvrtc.so"
echo "installed: $VENV  (vllm overlay -> $SP)"
