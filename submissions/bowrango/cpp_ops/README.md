# DREAMPlace C++/CUDA Ops — Minimal Build

Compiles just the pieces of DREAMPlace we need as a PyTorch C++ extension,
against the project's cu126 torch install. No CMake, flex/bison, or Boost.

## What gets built

- **electric_potential** — FFT-based density loss (this pass)
- **macro_legalize** — Tetris + Abacus macro legalizer (to add next)

## One-time setup

From the repo root:

```powershell
git submodule add https://github.com/limbo018/DREAMPlace external/DREAMPlace
git submodule update --init --recursive external/DREAMPlace
```

Commit the `.gitmodules` change and the submodule pointer so your Mac
clone picks it up too.

## Windows build (CUDA 12.6)

Prerequisites:

1. **CUDA Toolkit 12.6** installed — `nvcc --version` should work in your shell.
2. **Visual Studio 2022 Build Tools** with the "Desktop development with C++" workload.
3. **PyTorch cu126** — already configured via `pyproject.toml`; `uv sync` handles it.

Build from the `x64 Native Tools Command Prompt for VS 2022` (so `cl.exe` is on PATH):

```powershell
cd submissions\bowrango\cpp_ops
uv run python setup.py build_ext --inplace
```

First build takes 5–15 min depending on the machine. After it completes,
a `_electric_potential*.pyd` file appears in this directory and
`from cpp_ops.electric_potential import electric_potential` works.

## GPU architecture

`setup.py` currently targets `sm_86` (Ampere, 30xx) and `sm_89` (Ada, 40xx).
If your GPU is different, edit the `-gencode` lines:

| GPU family | arch/code |
|------------|-----------|
| Turing (20xx, T4) | `compute_75,code=sm_75` |
| Ampere (30xx, A100) | `compute_80,code=sm_80` or `86,sm_86` |
| Ada (40xx, L4) | `compute_89,code=sm_89` |
| Hopper (H100) | `compute_90,code=sm_90` |

Check your GPU with `nvidia-smi --query-gpu=compute_cap --format=csv`.

## Troubleshooting

- **`nvcc not found`** — CUDA Toolkit path not on PATH. From CUDA's install dir,
  add `bin\` to PATH or launch from the `Developer Command Prompt for VS 2022`
  with CUDA integration enabled.
- **`LINK : fatal error LNK1181: cannot open input file 'cufft.lib'`** —
  the linker can't find cuFFT. Add `%CUDA_PATH%\lib\x64` to `LIB` env var.
- **`torch.h not found`** — torch isn't in the active venv. Verify
  `uv run python -c "import torch; print(torch.__file__)"`.
- **Source-file mismatch** — if DREAMPlace has moved files around, adjust
  the glob in `setup.py`. The glob pattern `*.cpp` / `*.cu` in
  `dreamplace/ops/electric_potential/src/` should cover most layouts.

## Mac behavior

Building on Mac is unsupported in this minimal config — DREAMPlace's
CUDA kernels require CUDA. On Mac, `import cpp_ops.electric_potential`
will raise `ImportError`; `dreamplacer.py` detects this and falls back
to the pure-PyTorch `smooth_density`.
