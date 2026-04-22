# DREAMPlace C++/CUDA Ops — Minimal Build

Compiles just the pieces of DREAMPlace we need as a PyTorch C++ extension,
against the project's cu126 torch install. No CMake, no flex/bison, no Boost.

## What gets built

- **electric_potential** — FFT-based density loss (this pass)
- **macro_legalize** — Tetris + Abacus macro legalizer (to add next)

## One-time: add the submodule

From the repo root on either machine:

```bash
git submodule add https://github.com/limbo018/DREAMPlace external/DREAMPlace
git submodule update --init --recursive external/DREAMPlace
```

Commit `.gitmodules` and the submodule pointer so the other machine picks it up
after `git pull && git submodule update --init --recursive`.

## Windows build (CUDA 12.6)

### Prerequisites

1. **CUDA Toolkit 12.6** installed. Verify: `nvcc --version` prints `release 12.6`.

2. **Visual Studio 2022 Community** with the "Desktop development with C++" workload.

3. **MSVC v14.40 toolset (VS 17.10)** — this is the critical one. CUDA 12.6 only
   supports up to MSVC 14.40; the default 14.44 (VS 17.14) breaks with `<cmath>` errors
   like `'acosf' is not a member of global namespace` even on pure C++ files.

   To install: open **Visual Studio Installer** → Modify your VS 2022 install →
   **Individual components** tab → search `14.40` → check:

   > **MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.40-17.10)**

   Click Modify. Both toolsets coexist; we pin to 14.40 via `-vcvars_ver` below.

4. **PyTorch cu126** — already configured via project `pyproject.toml`; `uv sync`
   picks the CUDA wheel on Windows automatically.

5. **Ninja** (optional, faster builds): `uv pip install ninja`. Without it the build
   falls back to distutils (serial, slower) but still works.

### Build steps (one-shot)

Just run `build.bat` in this directory — double-click from Explorer or
`.\build.bat` from any cmd window. It pins MSVC to 14.40, verifies the compiler
version, sets `DISTUTILS_USE_SDK=1`, and runs `setup.py build_ext --inplace`.

If VS is installed to a non-Community edition (Professional / Enterprise /
BuildTools), edit the `VCVARS` path near the top of `build.bat`.

### Build steps (manual)

If you want to see each step or the batch file breaks, do it by hand. Open a
**plain `cmd.exe`** (not the Start-menu x64 Native Tools prompt — that picks
up the newest toolset). Pin to 14.40 manually:

```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.40
```

Verify with `cl` — the banner must say `Version 19.40.x for x64` (not 19.44).

Then set the env var PyTorch's build_ext requires and kick off the build:

```cmd
set DISTUTILS_USE_SDK=1
cd submissions\bowrango\cpp_ops
uv run python setup.py build_ext --inplace
```

First build takes 5–15 min (serial without ninja, ~2 min with). After success a
`_electric_potential*.pyd` appears next to `setup.py`, and
`from cpp_ops.electric_potential import electric_potential` works in Python.

### GPU architecture

`setup.py` targets `sm_86` (Ampere, 30xx) and `sm_89` (Ada, 40xx) by default.
If your GPU is different, edit the `-gencode` lines in `setup.py`:

| GPU family | `arch/code` |
|------------|-------------|
| Turing (20xx, T4) | `compute_75,code=sm_75` |
| Ampere (30xx, A100) | `compute_80` or `compute_86` |
| Ada (40xx, L4) | `compute_89,code=sm_89` |
| Hopper (H100) | `compute_90,code=sm_90` |

Check your card's compute capability:
```cmd
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Troubleshooting

**`UserWarning: ... DISTUTILS_USE_SDK is not set`** — you forgot step 3 of the build.
Run `set DISTUTILS_USE_SDK=1` (mind the typo trap: `DIS`, not `DES`).

**`error C2039: 'acosf' is not a member of global namespace`** (and similar `<cmath>`
errors) — MSVC toolset too new. Confirm `cl` reports 19.40.x, not 19.44. If it says
19.44, re-run `vcvarsall.bat x64 -vcvars_ver=14.40` in the current shell.

**`HostX86\x64\cl.exe` in the compile command** — wrong host compiler. You launched
from a shell where VS env wasn't set up. Use the `vcvarsall.bat` line above; host
should be `HostX64\x64`.

**`fatal error: cannot open include file: utility/src/torch.h`** — `include_dirs` is
missing `dreamplace/ops/`. Already fixed in this `setup.py`; pull latest.

**`nvcc not found`** — CUDA Toolkit `bin\` not on PATH. `vcvarsall.bat` doesn't add
CUDA; add it yourself or use the CUDA-integrated prompt. Check with `where nvcc`.

**`LINK : fatal error LNK1181: cannot open input file 'cufft.lib'`** — linker can't
find cuFFT. Add `%CUDA_PATH%\lib\x64` to `LIB` env var, or confirm CUDA 12.6 is fully
installed (not just the runtime).

**`torch.h not found` / `ATen/ATen.h not found`** — PyTorch missing from the active
venv. Verify with `uv run python -c "import torch; print(torch.__file__)"`.

## Mac behavior

Building is unsupported on Mac in this config — DREAMPlace's CUDA kernels require
CUDA. On Mac, `import cpp_ops.electric_potential` raises `ImportError`;
`dreamplacer.py` detects this and falls back to the pure-PyTorch `smooth_density`.
