# DREAMPlace + Spiral Cleanup

A two-stage macro placer:

1. **DREAMPlace** (Lin et al., DAC '19) runs as a black-box solver inside
   the upstream `limbo018/dreamplace:cuda` Docker image. The `Benchmark`
   is converted to UCLA Bookshelf, DREAMPlace does Nesterov global
   placement plus its built-in macro legalizer, and the resulting `.pl`
   is read back as a tensor of macro centers.
2. **Spiral cleanup** runs a minimum-displacement spiral search over
   hard macros to clear residual overlaps from DREAMPlace's legalizer.
   Already-legal macros keep their position via a fast-path; only
   overlapping ones move. On by default; pass `--no-spiral-cleanup` to
   disable.

## Files

| File | Role |
|---|---|
| `dreamplace_adapter.py` | `DreamPlaceAdapter` placer + CLI |
| `dreamplace_io.py` | Bookshelf write/read and DREAMPlace JSON config |
| `dreamplace_runner.py` | In-container shim that swaps DREAMPlace's legalizer for a macro-only one |
| `placer.py` | Spiral search (used standalone or as cleanup) |
| `make_mp4.py` | Stitch DREAMPlace plot PNGs into an MP4 |

## One-time setup

```bash
docker pull --platform linux/amd64 limbo018/dreamplace:cuda
git submodule update --init --recursive external/DREAMPlace
uv run python submissions/bowrango/dreamplace_adapter.py --build
```

The base image only ships build dependencies — DREAMPlace itself compiles
from the submodule into `external/DREAMPlace/install/`. To force a
rebuild, pass `--force-rebuild`.

## Run

```bash
# Through the evaluator harness.
uv run evaluate submissions/bowrango/dreamplace_adapter.py -b ibm01

# Direct CLI.
uv run python submissions/bowrango/dreamplace_adapter.py -b ibm01
```

Per-benchmark Bookshelf files and DREAMPlace outputs land under
`dreamplace_work/<benchmark>/`. Pass `keep_work=False` to
`DreamPlaceConfig` to clean up after each run.

### Tunables

| Flag | Default | Effect |
|---|---|---|
| `--auto-tune` / `--no-auto-tune` | on | Pick `target_density`/`stop_overflow`/`iterations` from canvas size at runtime. Small canvases (< 1000 µm²) get safer values to avoid gradient instability; larger benchmarks get the aggressive ibm10-tuned values. Disable to honor explicit flags below. |
| `--target-density` | `0.80` | DREAMPlace bin-occupancy ceiling (used when `--no-auto-tune`). Lower spreads more. |
| `--stop-overflow` | `0.005` | Global-placement convergence threshold (used when `--no-auto-tune`). Tighter forces more iterations of spreading. |
| `--iterations` | `3000` | Global-placement iteration cap (used when `--no-auto-tune`). |
| `--num-bins` | `256` | Density-grid resolution per axis. |
| `--legalize` / `--no-legalize` | on | Run DREAMPlace's macro legalizer. |
| `--spiral-cleanup` / `--no-spiral-cleanup` | on | Run min-displacement spiral pass over hard macros after DREAMPlace. |
| `--detailed` | off | Run DREAMPlace's ABCDPlace detailed placement after legalization. |
| `--fillers` | off | Filler-cell padding to reach `target_density`. |
| `--gpu` | `0` | `0` = CPU, `1` = GPU (requires NVIDIA Container Toolkit). |

## Visualization

```bash
# Generate per-iteration PNGs from inside the container.
uv run python submissions/bowrango/dreamplace_adapter.py -b ibm01 \
    --plot --plot-every 1

# Stitch them into an MP4.
uv run python submissions/bowrango/make_mp4.py -b ibm01
```

PNGs land at `dreamplace_work/<bench>/<bench>/plot/iter*.png`. The MP4
output drops next to that directory as `animation.mp4` by default.
Requires `ffmpeg` on PATH.

## Platform notes

- **macOS / Linux**: works as-is.
- **Windows**: works on Docker Desktop with the WSL2 backend. NVIDIA
  Container Toolkit is required for `--gpu 1`. The adapter detects the
  platform and skips POSIX-only flags automatically.
