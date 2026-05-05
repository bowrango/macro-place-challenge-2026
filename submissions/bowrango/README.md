# DREAMPlace + Spiral Cleanup

A two-stage macro placer:

1. **DREAMPlace** (Lin et al., DAC '19) runs as a black-box solver inside
   a Docker image. The `Benchmark` is converted to UCLA Bookshelf,
   DREAMPlace does Nesterov global placement plus its built-in macro
   legalizer, and the resulting `.pl` is read back as a tensor of macro
   centers.
2. **Spiral cleanup** runs a minimum-displacement spiral search over hard
   macros to clear residual overlaps left by DREAMPlace's legalizer.
   Already-legal macros keep their position via a fast-path; only
   overlapping ones move. On by default; pass `--no-spiral-cleanup` to
   disable.

TODO Make the objective in PlaceObj.py match macro_place.objective.compute_proxy_cost
TODO Adapt gridsearch.py example to optimize dreamplacer hyperparameters

## Notes
Maybe just replace the gated Gaussian noise with Levy
Running GD without legalization blows up congestion/density and makes proxy worse
Group refinement
Hessian negative-eigenvalue saddle escape branch

Bayesian optimization (Optuna):
iterations
target_density
density_weight

## Files

- [Files](#files)
- [Dependency](#dependency)
- [How to Build](#how-to-build)
  - [Build with Docker (CPU, macOS)](#build-with-docker-cpu-macos)
  - [Build with Docker (GPU, Linux / Windows)](#build-with-docker-gpu-linux--windows)
- [How to Run](#how-to-run)
- [Configurations](#configurations)
- [Visualization](#visualization)
- [Troubleshooting](#troubleshooting)

| File | Role |
|---|---|
| `dreamplace_adapter.py` | `DreamPlaceAdapter` placer + CLI. The evaluator imports this. Builds the docker invocation, bind-mounts the work dir / DREAMPlace tree / this directory into the container, and reads the `.pl` back. |
| `dreamplace_io.py` | `Benchmark` ↔ UCLA Bookshelf (`.aux`/`.nodes`/`.nets`/`.pl`/`.scl`/`.wts`) and the JSON config that drives `dreamplace/Placer.py`. Pin offsets come from `plc.modules_w_pins`; coordinates scale µm by `SCALE = 10_000`. |
| `dreamplace_runner.py` | In-container shim. Monkey-patches `BasicPlace.build_legalization` to skip DREAMPlace's greedy std-cell legalizer (which re-shuffles already-legal macros when there are no real std cells), hot-loads editable `external/DREAMPlace/dreamplace/LevyPlacer.py`, then forwards to installed `dreamplace/Placer.py` via `runpy`. |
| `placer.py` | Spiral search (used as cleanup). |
| `make_mp4.py` | Stitch DREAMPlace plot PNGs into an MP4. |

The runner ships next to the adapter, not in the image, so iterating on
the legalizer patch doesn't require a rebuild — the adapter bind-mounts
this directory at `/adapter` inside the container and invokes
`python3 /adapter/dreamplace_runner.py <config.json>`.

`LevyPlacer.py` is loaded from the bind-mounted DREAMPlace source tree
(`/dreamplace/dreamplace`, override with `DREAMPLACE_SOURCE_DIR`). The
installed tree is still used for compiled ops and the rest of DREAMPlace, so
Python-only edits to `external/DREAMPlace/dreamplace/LevyPlacer.py` take
effect on the next run without rebuilding or reinstalling DREAMPlace. Keep
the Levy tuning constants in `LevyPlacer.py`; the adapter does not pass them
through as environment options.

## Dependency

- [Docker](https://docs.docker.com/get-docker/) — Desktop on macOS /
  Windows, daemon on Linux.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  — only for `--gpu 1`. Required at *build* time of the CUDA image too,
  not just runtime: `cmake/TorchExtension.cmake` calls
  `torch.cuda.is_available()` at configure time, and skips CUDA kernels
  if no GPU is attached.
- The `external/DREAMPlace` git submodule:
  ```
  git submodule update --init --recursive external/DREAMPlace
  ```

## How to Build

### Build with Docker (CPU, macOS)

Pull the upstream image and build DREAMPlace into `external/DREAMPlace/install/`.

```bash
docker pull --platform linux/amd64 limbo018/dreamplace:cuda

cd external/DREAMPlace
docker run --rm \
  -v $(pwd):/dreamplace \
  -w /dreamplace \
  limbo018/dreamplace:cuda \
  bash -c "mkdir -p build install && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/dreamplace/install \
             -DPython_EXECUTABLE=\$(which python) && \
    make -j2 && make install"
```

The adapter selects this image automatically on macOS for CPU prototyping.

### Build with Docker (GPU, Linux / Windows)

The upstream image targets older CUDA + sm_60. RTX 30/A-series cards
(sm_86, Ampere) and Ada cards need a newer CUDA toolchain.
`external/DREAMPlace/Dockerfile` provides a CUDA 11.8 + PyTorch 2.0.1+cu118 base;
`build.sh` builds DREAMPlace inside it.  By default the script
lets DREAMPlace's CMake choose CUDA architectures; set
`DREAMPLACE_CUDA_ARCHITECTURES` if you want to force a specific list.

```bash
cd external/DREAMPlace

# 1. Build the CUDA 11.8 base image once.
docker build -t bowrango/dreamplace:cuda118 .

# 2. Build DREAMPlace inside it. --gpus all is REQUIRED
docker run --rm --gpus all \
  -v $(pwd):/dreamplace \
  -w /dreamplace \
  bowrango/dreamplace:cuda118 \
  bash build.sh

# Optional: force both local A4000/Ampere and eval Ada targets.
docker run --rm --gpus all \
  -e DREAMPLACE_CUDA_ARCHITECTURES='8.6;8.9' \
  -v $(pwd):/dreamplace \
  -w /dreamplace \
  bowrango/dreamplace:cuda118 \
  bash build.sh
```

PowerShell equivalent on Windows:

```powershell
cd external/DREAMPlace
docker build -t bowrango/dreamplace:cuda118 .

docker run --rm --gpus all `
  -e "DREAMPLACE_CUDA_ARCHITECTURES=8.6;8.9" `
  --mount "type=bind,source=$($PWD.Path),target=/dreamplace" `
  -w /dreamplace `
  bowrango/dreamplace:cuda118 `
  bash build.sh
```

## Run

```bash
# Through the evaluator harness.
uv run evaluate submissions/bowrango/dreamplace_adapter.py -b ibm01

# Direct CLI.
uv run python submissions/bowrango/dreamplace_adapter.py -b ibm01

# GPU (after building the DREAMPlace Dockerfile).
uv run python submissions/bowrango/dreamplace_adapter.py -b ibm01 --gpu 1
```

Bookshelf files and DREAMPlace outputs land under
`dreamplace_work/<benchmark>/`. Pass `--no-keep-work` (or
`keep_work=False` to `DreamPlaceConfig`) to clean up after each run.

## Configurations

| Flag | Default | Effect |
|---|---|---|
| `--target-density` | `0.80` | DREAMPlace bin-occupancy ceiling, in (0, 1]. Lower spreads more. |
| `--stop-overflow` | `0.005` | Global-placement convergence threshold. Tighter forces more iterations of spreading. |
| `--iterations` | `1000` | Global-placement iteration cap. |
| `--density-weight` | `4e-5` | Scales the spreading force vs wirelength force at iteration 0 (sets λ_0). Pass `auto` to scale by `utilization × cell density`. |
| `--num-bins` | `256` | Density-grid resolution per axis. |
| `--legalize` / `--no-legalize` | on | Run DREAMPlace's macro legalizer. |
| `--spiral-cleanup` / `--no-spiral-cleanup` | on | Run min-displacement spiral search over hard macros after DREAMPlace. |
| `--detailed` | off | Run DREAMPlace's ABCDPlace detailed placement after legalization. |
| `--fillers` | off | Filler-cell padding to reach `target_density`. |
| `--gpu` | `0` | `0` = CPU, `1` = GPU. |
| `--image` | OS-dependent | `limbo018/dreamplace:cuda` on macOS, `bowrango/dreamplace:cuda118` elsewhere. |

## Visualization

```bash
# Per-iteration PNGs from inside the container.
uv run python submissions/bowrango/dreamplace_adapter.py -b ibm01 \
    --plot --plot-every 1

# Stitch them into an MP4.
uv run python submissions/bowrango/make_mp4.py -b ibm01
```

PNGs land at `dreamplace_work/<bench>/<bench>/plot/iter*.png`. The MP4
drops next to that directory as `animation.mp4`. Requires `ffmpeg` on
PATH.
