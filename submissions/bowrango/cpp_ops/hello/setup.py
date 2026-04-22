"""Build a minimal CUDA extension to validate the Windows + cu126 workflow."""

import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="hello_cuda",
    ext_modules=[
        CUDAExtension(
            name="hello_cuda",
            sources=["hello.cu"],
            extra_compile_args={
                "cxx": (["/O2", "/std:c++17"] if sys.platform == "win32"
                        else ["-O3", "-std=c++17"]),
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-gencode=arch=compute_86,code=sm_86",  # RTX A4000
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
