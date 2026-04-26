# Macro Placement

## Problem

The netlist is a weighted hypergraph H = (V, E). Nodes are hard macros, soft macros, and I/O ports. Hyperedges are nets connecting nodes via pins. The objective is to minimize `proxy = HPWL + 0.5*density + 0.5*congestion` subject to hard macro non-overlap constraints. Density and congestion costs are driven by the worst 5-10% of grid cells, not averages.

## Method: Spiral Search + Interleaved FD

Starting from the benchmark's initial placement, the algorithm:

1. **Legalize hard macros** via spiral search (largest-area first). Each macro spirals outward from its initial position, scored by `displacement^2 + 0.15 * connectivity * canvas_diag`. Displacement keeps macros near initial positions; connectivity pulls connected macros together.

2. **Interleave FD on soft macros** every 50 hard macro placements. One force-directed step repositions standard cell clusters while hard macros are being placed, similar to the C++ SA baseline.

3. **Read back** final soft macro positions after all hard macros are legalized.

**Key insight**: the initial positions encode a high-quality global placement. The spiral search should fix overlaps with minimal disruption, not re-optimize from scratch.

## Results

```
    Benchmark     Proxy   vs SA  vs RePlAce
        ibm01    1.2222   +7.2%      -22.5%
        ibm02    1.6815  +11.8%       +8.5%
        ibm03    1.3548  +22.1%       -2.5%
        ibm04    1.3968   +7.1%       -7.3%
        ibm06    1.7082  +31.8%       -5.5%
        ibm07    1.4581  +27.9%       +0.4%
        ibm08    1.5500  +19.4%       -8.5%
        ibm09    1.1000  +20.7%       +1.7%
        ibm10    1.4507  +31.3%       +3.3%
        ibm11    1.1917  +30.4%       -1.2%
        ibm12    1.6538  +41.5%       +4.2%
        ibm13    1.3996  +26.9%       -4.8%
        ibm14    1.5554  +31.6%       -0.8%
        ibm15    1.5624  +32.1%       -3.1%
        ibm16    1.5505  +30.6%       -4.9%
        ibm17    1.6815  +54.2%       -2.2%
        ibm18    1.7470  +37.1%       +1.4%
          AVG    1.4861  +30.1%       -1.9%
```

## What Didn't Work

- **BFS/center-seed ordering**: disrupts the initial layout's spread
- **Density penalty in scoring**: fights connectivity, worse proxy
- **Re-legalization pass**: wirelength improves but density/congestion worsen
- **Soft macro centroid placement**: creates density hotspots from stacking
- **C++ SA from spiral init**: starts from scratch, can't reach our 1.18 in 500 iters

## Other Files

- `train_placer.py` — PPO + GNN for RL-based placement (experimental)
- `cpp_placer.py` — wrapper for C++ SA binary
- `sa_runner.cpp` — single-threaded C++ SA runner (macOS compatible)
- `dreamplace_adapter.py` — DREAMPlace solver backend via Docker (see below)
- `dreamplace_io.py` — Bookshelf format converters used by the adapter

## DREAMPlace Backend (Docker)

`dreamplace_adapter.DreamPlaceAdapter` runs the upstream
[`limbo018/dreamplace:cuda`](https://hub.docker.com/r/limbo018/dreamplace)
image as a black-box solver. The adapter:

1. Converts the Benchmark to UCLA Bookshelf
   (`.aux/.nodes/.nets/.pl/.scl/.wts`).
2. Writes a DREAMPlace JSON config.
3. Bind-mounts the work dir into the container and runs
   `python dreamplace/Placer.py <config.json>`.
4. Parses the resulting `.gp.pl` back into a `(num_macros, 2)` tensor of
   centers in microns.

### One-time setup

```bash
# 1. Pull the image (~5 GB) and the DREAMPlace source.
docker pull --platform linux/amd64 limbo018/dreamplace:cuda
git submodule update --init --recursive external/DREAMPlace

# 2. Build DREAMPlace inside the container. Output streams live so you
#    can watch cmake/make. Takes ~30-60 min under Rosetta on Apple Silicon.
uv run python submissions/bowrango/dreamplace_adapter.py --build
```

The base image only ships build dependencies — DREAMPlace itself is built
from the submodule. Build artifacts are written to
`external/DREAMPlace/{build,install}/` on the host (bind-mounted) and reused
on every subsequent placement.

On Apple Silicon the image runs under Rosetta (the official build is
x86_64-only). CPU mode is the only option on Docker Desktop / macOS — there
is no nvidia-docker for Mac.

To force a rebuild, delete `external/DREAMPlace/install/` or pass
`--force-rebuild`.

### Run

```bash
# Single benchmark, via the evaluator harness
uv run evaluate submissions/bowrango/dreamplace_adapter.py -b ibm01

# Or directly, with knobs
uv run python submissions/bowrango/dreamplace_adapter.py -b ibm01 \
    --iterations 1000 --target-density 1.0
```

### Notes

- Pin offsets are zero-ed (every pin attached at macro center). The proxy
  HPWL in `macro_place.objective` is also macro-center based, so this matches.
- Coordinates are scaled by `SCALE = 10_000` since Bookshelf uses integer
  units. The benchmark canvases are tiny (~23 µm) and DREAMPlace's defaults
  assume ISPD-scale (mm), so without this scale the bin grid collapses.
- I/O ports referenced by `plc.nets` are emitted as fixed terminals so
  DREAMPlace sees the boundary anchors that pull macros toward their
  natural positions — without them macros would collapse to the centroid.
- Per-benchmark Bookshelf files are kept under `dreamplace_work/<name>/`
  for inspection. Pass `keep_work=False` to clean up.
