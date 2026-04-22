"""Build script for DREAMPlace C++/CUDA ops (electric_potential).

Compiles a minimal subset of DREAMPlace — just the FFT-based density op —
as a PyTorch C++ extension. No CMake, no flex/bison, no Boost. Uses
torch.utils.cpp_extension to pick up the active torch install's compiler
and CUDA config.

Usage (from this directory):
    uv run python setup.py build_ext --inplace

Prerequisites on Windows:
    - PyTorch with CUDA 12.6 (configured in pyproject.toml)
    - CUDA Toolkit 12.6 installed, nvcc on PATH
    - Visual Studio 2022 Build Tools (MSVC cl.exe on PATH via x64 Native Tools prompt)
    - DREAMPlace submodule cloned under external/DREAMPlace

After build, `import cpp_ops.electric_potential` from Python returns a
callable that matches the DREAMPlace signature.
"""

import sys
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

REPO_ROOT = Path(__file__).resolve().parents[3]
DREAMPLACE = REPO_ROOT / "external" / "DREAMPlace"
OPS_ROOT = DREAMPLACE / "dreamplace" / "ops"
OP_SRC = OPS_ROOT / "electric_potential" / "src"
UTILITY_SRC = OPS_ROOT / "utility" / "src"

if not OP_SRC.exists():
    raise RuntimeError(
        f"DREAMPlace sources not found at {OP_SRC}.\n"
        "Add the submodule first:\n"
        "    git submodule add https://github.com/limbo018/DREAMPlace external/DREAMPlace\n"
        "    git submodule update --init --recursive external/DREAMPlace"
    )

# Discover sources — electric_potential's .cpp/.cu files live in OP_SRC.
# DREAMPlace reshuffles source layouts occasionally; glob keeps us robust.
sources = [str(p) for p in OP_SRC.glob("*.cpp")] + [str(p) for p in OP_SRC.glob("*.cu")]
if not sources:
    raise RuntimeError(f"No .cpp/.cu sources found in {OP_SRC}")

setup(
    name="dreamplace_cpp_ops",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="cpp_ops._electric_potential",
            sources=sources,
            # OPS_ROOT is required so `#include "utility/src/torch.h"` resolves.
            include_dirs=[str(OPS_ROOT), str(OP_SRC), str(UTILITY_SRC)],
            libraries=["cufft"],
            define_macros=[("ENABLE_CUDA", "1")],
            extra_compile_args={
                # /Zc:preprocessor forces MSVC into standards-conforming preprocessor
                # mode — required for DREAMPlace's variadic macros (msg.h etc.).
                "cxx": (["/O2", "/std:c++17", "/Zc:preprocessor", "/DENABLE_CUDA"]
                        if sys.platform == "win32"
                        else ["-O3", "-std=c++17", "-DENABLE_CUDA"]),
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-DENABLE_CUDA",
                    # RTX A4000 = Ampere GA104 = sm_86 (change if retargeting — see README)
                    "-gencode=arch=compute_86,code=sm_86",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["cpp_ops"],
)
