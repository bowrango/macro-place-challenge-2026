"""
DREAMPlace solver-backend adapter.

Wraps the upstream `limbo018/dreamplace:cuda` Docker image so it can be used
as a placer in this challenge's evaluation harness:

    uv run evaluate submissions/bowrango/dreamplace_adapter.py -b ibm01

Flow:
  1. Convert the Benchmark to UCLA Bookshelf (.aux/.nodes/.nets/.pl/.scl/.wts)
     under `submissions/bowrango/dreamplace_work/<benchmark>/`.
  2. Write a DREAMPlace JSON config pointing at those files.
  3. Run the Docker image, mounting the work directory and invoking
     `python dreamplace/Placer.py <config.json>` inside the container.
  4. Parse `<benchmark>.gp.pl` and return a (num_macros, 2) tensor of centers.

The Docker container is invoked with `--platform linux/amd64` so the x86_64
image runs under Rosetta on Apple Silicon. CPU-only by default — the official
image needs nvidia-docker for GPU mode, which Docker Desktop on macOS lacks.

First call will pull the image (~5 GB); subsequent calls reuse the local copy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dreamplace_io import (  # noqa: E402
    write_bookshelf,
    read_pl,
    write_dreamplace_config,
)
from placer import _load_plc  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(__file__).resolve().parent / "dreamplace_work"
DREAMPLACE_SRC = REPO_ROOT / "external/DREAMPlace"
DEFAULT_IMAGE = "limbo018/dreamplace:cuda"
CONTAINER_WORK = "/work"
CONTAINER_DREAMPLACE = "/DREAMPlace"
CONTAINER_INSTALL = f"{CONTAINER_DREAMPLACE}/install"
CONTAINER_PYDEPS = f"{CONTAINER_INSTALL}/python_deps"
# Marker that proves a complete build is on disk. The install step only
# creates this once everything compiled and copied successfully.
BUILD_MARKER_REL = "install/dreamplace/Placer.py"
# DREAMPlace's NonLinearPlace.py imports torch_optimizer, which the base
# pytorch:1.7.1 image doesn't ship. Installed once into the bind-mounted
# install dir so it persists alongside the C++ build.
#
# IMPORTANT: --no-deps. Without it, pip --target resolves torch>=1.5 and
# downloads a fresh torch 2.x (~800 MB + CUDA wheels) into python_deps/,
# which then *shadows* the container's torch 1.7.1 on sys.path. The C++
# extensions were compiled against 1.7.1's caffe2/c10 ABI and fail to
# load against 2.x with `undefined symbol: …caffe2…TypeMeta…`. With
# --no-deps we install only the pure-Python wheels and let `import torch`
# fall through to the container's built-in 1.7.1.
# Determined by diffing DREAMPlace's requirements.txt and a static `import`
# scan of dreamplace/*.py against the base image's `pip list`. The base image
# already has pyunpack, patool, matplotlib, pkgconfig, scipy, numpy, torch,
# and shapely — only these few are missing. `pytorch_ranger` is included
# because it's torch_optimizer's runtime dep that --no-deps would drop.
# `cairocffi` is omitted: only used when plot_flag=1, which our config sets to 0.
PYDEPS_MARKER_REL = "install/python_deps/ncg_optimizer/__init__.py"
PYDEPS_PACKAGES = ("torch_optimizer==0.3.0", "pytorch_ranger", "ncg_optimizer==0.2.2")


class DreamPlaceAdapter:
    """Run DREAMPlace inside Docker as a black-box solver.

    Args:
        image: Docker image to use. Default `limbo018/dreamplace:cuda`.
        platform: Docker platform string. `linux/amd64` runs under Rosetta on
            Apple Silicon — required since the official image is x86_64-only.
        gpu: 0 for CPU mode (default on Docker Desktop / macOS), 1 to ask
            DREAMPlace to use CUDA inside the container (needs nvidia-docker).
        iterations: global-placement iteration count.
        target_density: density target passed to DREAMPlace (1.0 = pack tight).
        legalize / detailed: forwarded to the corresponding DREAMPlace flags.
        keep_work: leave the per-benchmark work directory on disk for debugging.
        timeout: subprocess timeout (seconds) for each Docker run.
    """

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        platform: str = "linux/amd64",
        gpu: int = 0,
        iterations: int = 1000,
        target_density: float = 1.0,
        legalize: bool = True,
        detailed: bool = False,
        keep_work: bool = True,
        timeout: int = 1800,
        build_timeout: int = 7200,
        dreamplace_src: Path = DREAMPLACE_SRC,
    ):
        self.image = image
        self.platform = platform
        self.gpu = gpu
        self.iterations = iterations
        self.target_density = target_density
        self.legalize = legalize
        self.detailed = detailed
        self.keep_work = keep_work
        self.timeout = timeout
        self.build_timeout = build_timeout
        self.dreamplace_src = dreamplace_src

        self._check_docker()
        self._require_built()

    # ─────────────────────────────────────────────────────────────────────
    # Public placer interface (matches the convention used by evaluate.py).
    # ─────────────────────────────────────────────────────────────────────

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = _load_plc(benchmark.name)

        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        work = WORK_ROOT / benchmark.name
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)

        self._ensure_python_deps()

        sys.stderr.write(f"  dreamplace: writing bookshelf to {work}\n")
        index = write_bookshelf(benchmark, work, plc=plc)

        # The container sees the work dir at /work via the bind mount.
        config_in = f"{CONTAINER_WORK}/{benchmark.name}.json"
        write_dreamplace_config(
            work_dir_in_container=CONTAINER_WORK,
            name=benchmark.name,
            config_path=work / f"{benchmark.name}.json",
            iterations=self.iterations,
            target_density=self.target_density,
            gpu=self.gpu,
            legalize=self.legalize,
            detailed=self.detailed,
        )

        # ── invoke DREAMPlace inside the container ──────────────────────
        t0 = time.time()
        self._run_docker(work, config_in, benchmark.name)
        sys.stderr.write(f"  dreamplace: docker run took {time.time() - t0:.1f}s\n")

        # ── locate the output .pl ───────────────────────────────────────
        pl_path = self._find_output_pl(work, benchmark.name)
        if pl_path is None:
            raise RuntimeError(
                f"DREAMPlace did not produce a .gp.pl under {work} — "
                "check stderr above. Set keep_work=True to inspect."
            )

        out = read_pl(pl_path, benchmark, index)

        # Fixed macros must remain at their input positions.
        fixed = benchmark.macro_fixed.bool()
        out[fixed] = benchmark.macro_positions[fixed]

        if not self.keep_work:
            shutil.rmtree(work, ignore_errors=True)

        return out

    # ─────────────────────────────────────────────────────────────────────
    # Internals.
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_docker() -> None:
        if shutil.which("docker") is None:
            raise RuntimeError(
                "`docker` not found on PATH. Install Docker Desktop and ensure "
                "it is running before using DreamPlaceAdapter."
            )
        try:
            subprocess.run(
                ["docker", "info"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(
                "Docker daemon is not reachable. Start Docker Desktop and try "
                "again."
            ) from e

    def _run_docker(self, work: Path, config_in_container: str, name: str) -> None:
        # Bind-mount: host work dir → /work, DREAMPlace submodule → /DREAMPlace.
        # The submodule is mounted so we can use the previously-built install
        # under /DREAMPlace/install (built by `--build` on first call).
        # PYTHONPATH adds the persistent python_deps dir so torch_optimizer
        # imports cleanly. MPLCONFIGDIR silences matplotlib's HOME-not-writable
        # warning when running as a non-root user without a HOME dir.
        # PYTHONUNBUFFERED forces python's stdout/stderr to flush per line so
        # progress streams live to our terminal instead of dumping after exit.
        run_cmd = (
            f"export PYTHONPATH={CONTAINER_PYDEPS}:${{PYTHONPATH:-}} && "
            f"export MPLCONFIGDIR=/tmp/mpl && mkdir -p $MPLCONFIGDIR && "
            f"export PYTHONUNBUFFERED=1 && "
            f"cd {CONTAINER_INSTALL} && python -u dreamplace/Placer.py {config_in_container}"
        )
        cmd = [
            "docker", "run", "--rm",
            "--platform", self.platform,
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{work}:{CONTAINER_WORK}",
            "-v", f"{self.dreamplace_src}:{CONTAINER_DREAMPLACE}",
            "-w", CONTAINER_INSTALL,
            self.image,
            "bash", "-c", run_cmd,
        ]

        sys.stderr.write("  dreamplace: " + " ".join(cmd) + "\n")
        proc = subprocess.run(
            cmd,
            timeout=self.timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Mirror DREAMPlace's stdout/stderr to ours so progress is visible.
        sys.stderr.write(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"DREAMPlace exited with code {proc.returncode}. "
                f"Work dir preserved at {work}."
            )

    def _require_built(self) -> None:
        """Fail fast if DREAMPlace hasn't been built yet. The build is an
        explicit, separate step (`--build` on this script) so the user can
        watch its 30-60 min output stream live."""
        marker = self.dreamplace_src / BUILD_MARKER_REL
        if marker.exists():
            return
        raise RuntimeError(
            f"DREAMPlace not built yet (missing {marker}).\n"
            f"Run the build first (output streams live, ~30-60 min on Apple Silicon):\n"
            f"  uv run python submissions/bowrango/dreamplace_adapter.py --build"
        )

    def _ensure_python_deps(self) -> None:
        """Install DREAMPlace's pure-Python runtime deps into a persistent
        directory under the install tree. Runs once, idempotent thereafter.

        We can't reuse the host's uv venv: the C++ extensions were compiled
        against the container's Python 3.8 + PyTorch 1.7.1 ABI, so DREAMPlace
        has to run inside the container. The packages here are pure-Python so
        installing them with the container's pip is fast (~10s)."""
        marker = self.dreamplace_src / PYDEPS_MARKER_REL
        if marker.exists():
            return

        sys.stderr.write(
            f"  dreamplace: installing runtime Python deps "
            f"({', '.join(PYDEPS_PACKAGES)}) into {marker.parent.parent}\n"
        )
        pip_cmd = (
            f"pip install --no-cache-dir --no-deps "
            f"--target={CONTAINER_PYDEPS} "
            f"{' '.join(PYDEPS_PACKAGES)}"
        )
        cmd = [
            "docker", "run", "--rm",
            "--platform", self.platform,
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{self.dreamplace_src}:{CONTAINER_DREAMPLACE}",
            self.image,
            "bash", "-c", pip_cmd,
        ]
        # Inherit stdout/stderr — pip's progress streams live, no silent hangs.
        proc = subprocess.run(cmd, timeout=300)
        if proc.returncode != 0 or not marker.exists():
            raise RuntimeError(
                f"Failed to install Python runtime deps "
                f"(exit {proc.returncode}, marker {marker} missing)."
            )

    @staticmethod
    def build(
        image: str = DEFAULT_IMAGE,
        platform: str = "linux/amd64",
        dreamplace_src: Path = DREAMPLACE_SRC,
        build_timeout: int = 7200,
        force: bool = False,
        jobs: int = 2,
    ) -> None:
        """Build DREAMPlace inside the container. Output streams live to the
        terminal so the user can follow cmake/make progress.

        Run as a separate step before any placement:
            uv run python submissions/bowrango/dreamplace_adapter.py --build
        """
        if shutil.which("docker") is None:
            raise RuntimeError("`docker` not found on PATH.")
        if not dreamplace_src.exists():
            raise RuntimeError(
                f"DREAMPlace source not found at {dreamplace_src}. "
                "Run: git submodule update --init --recursive external/DREAMPlace"
            )

        marker = dreamplace_src / BUILD_MARKER_REL
        if marker.exists() and not force:
            sys.stderr.write(
                f"  dreamplace: already built at {marker.parent.parent}\n"
                f"  (pass --force-rebuild or delete external/DREAMPlace/install/ "
                f"to rebuild)\n"
            )
            return

        # Low parallelism by default: each cc1plus instance with -flto and the
        # full PyTorch headers can easily eat 3-4 GB. -j$(nproc) under Docker
        # Desktop's VM (often 4-8 GB total) is a one-way ticket to OOM kills.
        build_script = (
            f"set -euo pipefail && "
            f"cd {CONTAINER_DREAMPLACE} && "
            f"mkdir -p build install && cd build && "
            f"cmake .. "
            f"-DCMAKE_INSTALL_PREFIX={CONTAINER_INSTALL} "
            f"-DPython_EXECUTABLE=$(which python) && "
            f"make -j{jobs} && make install"
        )
        cmd = [
            "docker", "run", "--rm",
            "--platform", platform,
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{dreamplace_src}:{CONTAINER_DREAMPLACE}",
            "-w", CONTAINER_DREAMPLACE,
            image,
            "bash", "-c", build_script,
        ]

        sys.stderr.write(
            "  dreamplace: building DREAMPlace inside container "
            "(one-time, ~30-60 min under Rosetta). Streaming build output…\n"
        )
        sys.stderr.write("  dreamplace: " + " ".join(cmd) + "\n")
        sys.stderr.flush()

        t0 = time.time()
        # Inherit stdout/stderr so cmake/make output streams live to the user.
        proc = subprocess.run(cmd, timeout=build_timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"DREAMPlace build failed (exit {proc.returncode}). "
                f"Partial artifacts in {dreamplace_src}/build."
            )
        if not marker.exists():
            raise RuntimeError(
                f"Build reported success but marker {marker} is missing."
            )
        sys.stderr.write(
            f"  dreamplace: build complete in {time.time() - t0:.0f}s — "
            f"installed under {marker.parent.parent}\n"
        )

    @staticmethod
    def _find_output_pl(work: Path, name: str) -> Path | None:
        # DREAMPlace writes to `<result_dir>/<design_name>/<design_name>.gp.pl`.
        # `design_name` is derived from the .aux basename, so it equals `name`.
        candidates = [
            work / name / f"{name}.gp.pl",
            work / f"{name}.gp.pl",
        ]
        for c in candidates:
            if c.exists():
                return c
        # Fall back to a recursive search.
        hits = list(work.rglob("*.gp.pl"))
        return hits[0] if hits else None


if __name__ == "__main__":
    import argparse
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    parser = argparse.ArgumentParser(description="DREAMPlace via Docker")
    parser.add_argument("--build", action="store_true",
                        help="Build DREAMPlace inside the container and exit. "
                             "Output streams live; do this once before placement.")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="With --build, rebuild even if install/ already exists.")
    parser.add_argument("--jobs", "-j", type=int, default=2,
                        help="With --build, parallel make jobs (default 2). "
                             "Increase if Docker Desktop has plenty of memory "
                             "(~4 GB per job under Rosetta + LTO).")
    parser.add_argument("--benchmark", "-b", default="ibm01")
    parser.add_argument("--iterations", "-n", type=int, default=1000)
    parser.add_argument("--target-density", type=float, default=1.0)
    parser.add_argument("--no-legalize", action="store_true")
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.build:
        DreamPlaceAdapter.build(
            image=args.image,
            platform=args.platform,
            force=args.force_rebuild,
            jobs=args.jobs,
        )
        sys.exit(0)

    root = REPO_ROOT / "external/MacroPlacement/Testcases/ICCAD04" / args.benchmark
    benchmark, plc = load_benchmark_from_dir(str(root))

    placer = DreamPlaceAdapter(
        image=args.image,
        platform=args.platform,
        gpu=args.gpu,
        iterations=args.iterations,
        target_density=args.target_density,
        legalize=not args.no_legalize,
        detailed=args.detailed,
    )
    placement = placer.place(benchmark)

    result = compute_proxy_cost(placement, benchmark, plc)
    print(
        f"proxy={result['proxy_cost']:.4f}  "
        f"(wl={result['wirelength_cost']:.3f} "
        f"den={result['density_cost']:.3f} "
        f"cong={result['congestion_cost']:.3f})  "
        f"overlaps={result['overlap_count']}"
    )
