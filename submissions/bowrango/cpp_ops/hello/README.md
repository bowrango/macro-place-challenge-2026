# Hello CUDA — Minimal Workflow Sanity Check

A single-kernel PyTorch CUDA extension that adds two tensors. Tests the
Windows + CUDA 12.6 + MSVC 14.40 + PyTorch C++ extension pipeline without
any DREAMPlace entanglement.

## Build & test

```cmd
cd submissions\bowrango\cpp_ops\hello
build.bat
```

`build.bat` pins MSVC/SDK, builds `hello_cuda.pyd`, then runs `test.py`
which imports it and runs the kernel on GPU. Success print:

```
hello_cuda.add works: max |err| = 0.00e+00
device=cuda:0  shape=(1024,)  dtype=torch.float32
GPU: NVIDIA RTX A4000
```

## What this tells us

- **If this builds and `test.py` passes** → the toolchain works end-to-end.
  DREAMPlace's build failures were DREAMPlace-specific (shadow headers,
  missing CMake-defined macros like `DREAMPLACE_TENSOR_SCALARTYPE`),
  not the workflow itself.
- **If this fails** → the failure message is now small enough to actually
  diagnose (no more 2000-line DREAMPlace output).

## Next step

Once this works, we have two options for DREAMPlace's FFT density:
1. Replicate the specific `#define`s DREAMPlace's CMake sets (e.g.,
   `DREAMPLACE_TENSOR_SCALARTYPE=float`) and keep going.
2. Port the FFT electrostatic density to pure PyTorch using `torch.fft`
   — same algorithm, no build system.
