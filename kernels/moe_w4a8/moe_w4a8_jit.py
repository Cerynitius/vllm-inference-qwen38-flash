"""JIT-builds moe_w4a8.cu (sm_120a); `from moe_w4a8_jit import ext` -> ext.moe_w4a8(...)."""
import os
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
_bdir = os.path.join(HERE, "build")
os.makedirs(_bdir, exist_ok=True)
_flags = os.environ.get("MOE_W4A8_NVCC", "").split()   # testing only
if _flags:
    _bdir = os.path.join(HERE, "build_" + str(abs(hash(" ".join(_flags))) % 10**8)); os.makedirs(_bdir, exist_ok=True)
ext = load(name="moe_w4a8_ext" + ("_" + _bdir.rsplit("_", 1)[-1] if _flags else ""), sources=[os.path.join(HERE, "moe_w4a8.cu")],
           extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a", "-lineinfo"] + _flags,
           build_directory=_bdir, verbose=False)
