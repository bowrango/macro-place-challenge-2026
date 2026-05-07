"""
DREAMPlace adapter.

Wraps a DREAMPlace Docker image as a placer:
    uv run evaluate submissions/bowrango/dreamplace_adapter.py -b ibm01

Per-call flow: Bookshelf write → JSON config → docker run → read .pl →
optional spiral cleanup. The image must already contain a built DREAMPlace
under /dreamplace/install — build it manually inside `external/DREAMPlace`
before invoking the adapter.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from macro_place.benchmark import Benchmark

CONTAINER_WORK = "/work"
CONTAINER_DREAMPLACE = "/dreamplace"
CONTAINER_INSTALL = f"{CONTAINER_DREAMPLACE}/install"
CONTAINER_PYDEPS = f"{CONTAINER_INSTALL}/python_deps"

BUILD_MARKER_REL = "install/dreamplace/Placer.py"
PYDEPS_MARKER_REL = "install/python_deps/ncg_optimizer/__init__.py"

# --no-deps so pip doesn't pull a torch wheel as a transitive dep and
# shadow the container's torch (1.7.1 in limbo018/dreamplace:cuda; 2.0.1+cu118
# in bowrango/dreamplace:cuda118) — DREAMPlace's C++ extensions are linked
# against that exact build.
PYDEPS_PACKAGES = (
    "torch_optimizer==0.3.0",
    "pytorch_ranger",
    "ncg_optimizer==0.2.2",
)

# Images that bake PYDEPS_PACKAGES into site-packages at build time, so the
# bind-mounted python_deps install can be skipped.
PYDEPS_BAKED_IN_IMAGES = frozenset({"bowrango/dreamplace:cuda118"})


def _load_sibling(name: str):
    """Load `<this dir>/<name>.py` as a top-level module. The evaluator loads
    us via `spec_from_file_location` outside any package, so relative imports
    don't work; this avoids touching `sys.path`."""
    p = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load sibling module {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_base = _load_sibling("dreamplace_base")
_io = _load_sibling("dreamplace_io")
_placer_mod = _load_sibling("placer")

AdapterConfigBase = _base.AdapterConfigBase
DEFAULT_IMAGE = _base.DEFAULT_IMAGE
DEFAULT_PLATFORM = _base.DEFAULT_PLATFORM
DREAMPLACE_SRC = _base.DREAMPLACE_SRC
DEFAULT_WORK_ROOT = _base.DEFAULT_WORK_ROOT
REPO_ROOT = _base.REPO_ROOT
ensure_default_logging = _base.ensure_default_logging
logger = _base.logger
write_bookshelf = _io.write_bookshelf
read_pl = _io.read_pl
write_dreamplace_config = _io.write_dreamplace_config
_load_plc = _placer_mod._load_plc
_spiralsearch = _placer_mod.spiralsearch
_extract_hard_edges = _placer_mod._extract_hard_edges


def _user_args() -> list[str]:
    """`--user $UID:$GID` on POSIX. Skipped on Windows: `os.getuid` doesn't
    exist there, and Docker Desktop on Windows handles bind-mount ownership."""
    if os.name == "nt":
        return []
    return ["--user", f"{os.getuid()}:{os.getgid()}"]


def _gpu_args(gpu: int) -> list[str]:
    """`--gpus all` when GPU is requested. Without this flag the NVIDIA
    runtime won't attach devices regardless of `params.gpu` in the JSON."""
    return ["--gpus", "all"] if gpu >= 1 else []


class DreamPlaceConfig(AdapterConfigBase):
    """Tunables for `DreamPlaceAdapter`. Plain class instead of `@dataclass`
    because the evaluator loads us outside `sys.modules`, which Python 3.9's
    dataclass introspection requires."""

    __slots__ = (
        "iterations", "target_density", "density_weight",
        "legalize", "detailed", "enable_fillers",
        "stop_overflow", "num_bins", "spiral_cleanup",
        "plot", "plot_every",
    )

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        platform: str = DEFAULT_PLATFORM,
        gpu: int = 0,
        iterations: int = 1000,
        target_density: float = 0.80,
        density_weight: Union[float, str] = 4e-5,
        legalize: bool = True,
        detailed: bool = False,
        enable_fillers: bool = False,
        stop_overflow: float = 0.005,
        num_bins: int = 256,
        spiral_cleanup: bool = True,
        plot: bool = False,
        plot_every: int = 1,
        keep_work: bool = True,
        timeout: int = 1800,
        dreamplace_src: Path = DREAMPLACE_SRC,
        work_root: Path = DEFAULT_WORK_ROOT,
        run_id: Optional[str] = None,
    ):
        super().__init__(
            image=image,
            platform=platform,
            gpu=gpu,
            keep_work=keep_work,
            timeout=timeout,
            dreamplace_src=dreamplace_src,
            work_root=work_root,
            run_id=run_id,
        )
        self.iterations = iterations
        self.target_density = target_density
        self.density_weight = density_weight
        self.legalize = legalize
        self.detailed = detailed
        self.enable_fillers = enable_fillers
        self.stop_overflow = stop_overflow
        self.num_bins = num_bins
        self.spiral_cleanup = spiral_cleanup
        self.plot = plot
        self.plot_every = plot_every

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.as_dict().items())
        return f"DreamPlaceConfig({fields})"


def _auto_density_weight(benchmark: Benchmark) -> float:
    """`density_weight ∝ utilization × num_macros / canvas_area`. Constant
    calibrated so ibm01 (util≈0.43, 1140 macros / 527 µm²) → 8e-5."""
    canvas_area = benchmark.canvas_width * benchmark.canvas_height
    macro_area = float(
        (benchmark.macro_sizes[:, 0] * benchmark.macro_sizes[:, 1]).sum()
    )
    utilization = macro_area / canvas_area
    cell_density = benchmark.num_macros / canvas_area
    return 8.6e-5 * utilization * cell_density


class DreamPlaceAdapter:
    """Run DREAMPlace inside Docker as a black-box solver. Environment
    checks run lazily on first `place()` — call `validate_environment()`
    explicitly to fail fast."""

    def __init__(self, config: Optional[DreamPlaceConfig] = None):
        self.config = config if config is not None else DreamPlaceConfig()
        self._validated = False

    def validate_environment(self) -> None:
        if self._validated:
            return
        self.config.ensure_logging()
        self._check_docker()
        self._require_built()
        self._validated = True

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        self.validate_environment()
        cfg = self.config
        plc = _load_plc(benchmark.name)

        if cfg.density_weight == "auto":
            density_weight = _auto_density_weight(benchmark)
            canvas_area = benchmark.canvas_width * benchmark.canvas_height
            macro_area = float(
                (benchmark.macro_sizes[:, 0] * benchmark.macro_sizes[:, 1]).sum()
            )
            logger.info(
                "density_weight=auto → %.2e "
                "(util=%.3f, num_macros=%d, canvas=%.1f µm²)",
                density_weight, macro_area / canvas_area,
                benchmark.num_macros, canvas_area,
            )
        else:
            density_weight = cfg.density_weight

        cfg.work_root.mkdir(parents=True, exist_ok=True)
        work = cfg.work_dir_for(benchmark.name)
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)

        self._ensure_python_deps()

        logger.info("writing bookshelf to %s", work)
        index = write_bookshelf(benchmark, work, plc=plc)
        write_dreamplace_config(
            work_dir_in_container=CONTAINER_WORK,
            name=benchmark.name,
            config_path=work / f"{benchmark.name}.json",
            iterations=cfg.iterations,
            target_density=cfg.target_density,
            gpu=cfg.gpu,
            legalize=cfg.legalize,
            detailed=cfg.detailed,
            enable_fillers=cfg.enable_fillers,
            stop_overflow=cfg.stop_overflow,
            plot=cfg.plot,
            num_bins=cfg.num_bins,
            density_weight=density_weight,
        )

        config_in = f"{CONTAINER_WORK}/{benchmark.name}.json"
        t0 = time.time()
        self._run_docker(work, config_in)
        logger.info("docker run took %.1fs", time.time() - t0)

        if cfg.plot:
            plot_dir = work / benchmark.name / "plot"
            pngs = sorted(plot_dir.glob("iter*.png")) if plot_dir.exists() else []
            if pngs:
                logger.info("wrote %d plot(s) to %s", len(pngs), plot_dir)
            else:
                logger.warning("plot=True but no PNGs found under %s", plot_dir)

        pl_path = self._find_output_pl(work, benchmark.name)
        if pl_path is None:
            raise RuntimeError(f"DREAMPlace produced no .pl under {work}")
        logger.info("reading %s", pl_path.name)
        out = read_pl(pl_path, benchmark, index)

        # Restore fixed macros to their exact input position; rounding
        # through Bookshelf integer coords can shift them by ≤ 1 unit.
        fixed = benchmark.macro_fixed
        out[fixed] = benchmark.macro_positions[fixed]

        if cfg.spiral_cleanup:
            t1 = time.time()
            out = self._spiral_cleanup(benchmark, out, plc)
            logger.info("spiral cleanup took %.2fs", time.time() - t1)

        if not cfg.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        return out

    @staticmethod
    def _spiral_cleanup(
        benchmark: Benchmark, positions: torch.Tensor, plc
    ) -> torch.Tensor:
        """Resolve residual hard-macro overlaps with `placer.spiralsearch`,
        seeded from DREAMPlace's output. Already-legal macros hit the
        fast-path; only overlapping ones move. Soft macros are not touched."""
        n_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        movable = benchmark.get_movable_mask()[:n_hard].numpy()

        if plc is not None:
            edges, edge_weights = _extract_hard_edges(plc)
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
            edge_weights = np.zeros(0, dtype=np.float64)

        pos_in = positions[:n_hard].numpy().copy().astype(np.float64)
        pos_out = _spiralsearch(
            pos_in, movable, sizes, sizes[:, 0] / 2, sizes[:, 1] / 2,
            cw, ch, n_hard, edges=edges, edge_weights=edge_weights,
        )

        # spiralsearch returns a torch.Tensor (placer.py); just dtype-coerce.
        result = positions.clone()
        result[:n_hard] = pos_out.to(torch.float32)
        return result

    def _run_docker(self, work: Path, config_in_container: str) -> None:
        """Invoke `dreamplace_runner.py` inside the container. Output is
        unbuffered so progress streams live and Ctrl+C propagates."""
        cfg = self.config
        container_adapter = "/adapter"
        cmd = [
            "docker", "run", "--rm",
            "--platform", cfg.platform,
            *_user_args(),
            *_gpu_args(cfg.gpu),
            "-e", f"PYTHONPATH={CONTAINER_PYDEPS}:{CONTAINER_INSTALL}",
            "-e", "PYTHONUNBUFFERED=1",
            "-e", "MPLCONFIGDIR=/tmp/mpl",
            "-e", f"DREAMPLACE_PLOT_EVERY={cfg.plot_every}",
            "-v", f"{work}:{CONTAINER_WORK}",
            "-v", f"{cfg.dreamplace_src}:{CONTAINER_DREAMPLACE}",
            "-v", f"{cfg.adapter_dir}:{container_adapter}",
            "-w", CONTAINER_INSTALL,
            cfg.image,
            "bash", "-c",
            f"mkdir -p $MPLCONFIGDIR && "
            f"python3 -u {container_adapter}/dreamplace_runner.py "
            f"{config_in_container}",
        ]
        logger.info("%s", " ".join(cmd))
        proc = subprocess.run(cmd, timeout=cfg.timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"DREAMPlace exited {proc.returncode}. Work dir: {work}."
            )

    @staticmethod
    def _find_output_pl(work: Path, name: str) -> Optional[Path]:
        """Pick the latest stage's `.pl` available — detailed > legal > global."""
        for suffix in ("dp.pl", "lg.pl", "gp.pl"):
            for candidate in (work / name / f"{name}.{suffix}",
                              work / f"{name}.{suffix}"):
                if candidate.exists():
                    return candidate
        hits = list(work.rglob("*.pl"))
        return next((p for p in hits if p.name != f"{name}.pl"), None)

    @staticmethod
    def _check_docker() -> None:
        if shutil.which("docker") is None:
            raise RuntimeError(
                "`docker` not on PATH. Install Docker Desktop and start it."
            )
        try:
            subprocess.run(
                ["docker", "info"], check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise RuntimeError("Docker daemon unreachable.") from e

    def _require_built(self) -> None:
        marker = self.config.dreamplace_src / BUILD_MARKER_REL
        if marker.exists():
            return
        raise RuntimeError(
            f"DREAMPlace not built (missing {marker}). From "
            f"external/DREAMPlace (wipe build/ install/ first if you've "
            f"previously built against another image):\n"
            f"  docker run --rm -v <pwd>:{CONTAINER_DREAMPLACE} "
            f"-w {CONTAINER_DREAMPLACE} {self.config.image} "
            f"bash build.sh"
        )

    def _ensure_python_deps(self) -> None:
        """Idempotent install of pure-Python runtime deps into the
        bind-mounted install dir. Skipped on images that already have
        them in site-packages."""
        if self.config.image in PYDEPS_BAKED_IN_IMAGES:
            return
        marker = self.config.dreamplace_src / PYDEPS_MARKER_REL
        if marker.exists():
            return

        logger.info("installing python deps (%s)", ", ".join(PYDEPS_PACKAGES))
        cmd = [
            "docker", "run", "--rm",
            "--platform", self.config.platform,
            *_user_args(),
            "-v", f"{self.config.dreamplace_src}:{CONTAINER_DREAMPLACE}",
            self.config.image,
            "bash", "-c",
            f"pip install --no-cache-dir --no-deps "
            f"--target={CONTAINER_PYDEPS} {' '.join(PYDEPS_PACKAGES)}",
        ]
        proc = subprocess.run(cmd, timeout=300)
        if proc.returncode != 0 or not marker.exists():
            raise RuntimeError(f"pip install failed (exit {proc.returncode})")

def _config_defaults() -> dict:
    """All `DreamPlaceConfig` field defaults, sourced via introspection.
    Single source of truth for both programmatic and CLI defaults."""
    import inspect
    return {
        name: param.default
        for name, param in inspect.signature(DreamPlaceConfig.__init__).parameters.items()
        if name != "self" and param.default is not inspect.Parameter.empty
    }


def main() -> None:
    import argparse
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    cfg = _config_defaults()
    bool_optional = argparse.BooleanOptionalAction

    def _density_weight_arg(s: str):
        return "auto" if s.lower() == "auto" else float(s)

    parser = argparse.ArgumentParser(description="DREAMPlace via Docker")
    parser.add_argument("--benchmark", "-b", default="ibm01")

    # DreamPlaceConfig-bound options. All defaults come from cfg[...] so
    # editing DreamPlaceConfig.__init__ is the only place to change them.
    parser.add_argument("--iterations", "-n", type=int,
                        default=cfg["iterations"])
    parser.add_argument("--target-density", type=float,
                        default=cfg["target_density"],
                        help="DREAMPlace bin-occupancy ceiling, in (0, 1]. "
                             "Lower spreads more.")
    parser.add_argument("--density-weight", type=_density_weight_arg,
                        default=cfg["density_weight"],
                        help="Scales the spreading force vs wirelength force "
                             "at iteration 0 (sets λ_0). 'auto' (default) "
                             "picks from utilization × cell density; or pass "
                             "a positive float to fix it.")
    parser.add_argument("--stop-overflow", type=float,
                        default=cfg["stop_overflow"],
                        help="Convergence threshold for global placement.")
    parser.add_argument("--num-bins", type=int,
                        default=cfg["num_bins"],
                        help="Density-grid resolution per axis.")
    parser.add_argument("--legalize", action=bool_optional,
                        default=cfg["legalize"],
                        help="Run DREAMPlace's macro legalizer.")
    parser.add_argument("--detailed", action=bool_optional,
                        default=cfg["detailed"],
                        help="Run DREAMPlace's ABCDPlace detailed placement.")
    parser.add_argument("--fillers", action=bool_optional,
                        default=cfg["enable_fillers"],
                        help="Filler-cell padding to reach target_density.")
    parser.add_argument("--spiral-cleanup", action=bool_optional,
                        default=cfg["spiral_cleanup"],
                        help="Min-displacement spiral search over hard macros "
                             "to clear residual overlaps after DREAMPlace.")
    parser.add_argument("--plot", action=bool_optional,
                        default=cfg["plot"],
                        help="Write PNG snapshots of global placement to "
                             "dreamplace_work/<bench>/<bench>/plot/.")
    parser.add_argument("--plot-every", type=int,
                        default=cfg["plot_every"],
                        help="With --plot, snapshot every N global-placement iterations.")
    parser.add_argument("--keep-work", action=bool_optional,
                        default=cfg["keep_work"],
                        help="Preserve Bookshelf inputs and DREAMPlace "
                             "outputs under dreamplace_work/<bench>/.")
    parser.add_argument("--image", default=cfg["image"])
    parser.add_argument("--platform", default=cfg["platform"])
    parser.add_argument("--gpu", type=int, default=cfg["gpu"])

    args = parser.parse_args()
    ensure_default_logging()

    config = DreamPlaceConfig(
        image=args.image,
        platform=args.platform,
        gpu=args.gpu,
        iterations=args.iterations,
        target_density=args.target_density,
        density_weight=args.density_weight,
        legalize=args.legalize,
        detailed=args.detailed,
        enable_fillers=args.fillers,
        stop_overflow=args.stop_overflow,
        num_bins=args.num_bins,
        spiral_cleanup=args.spiral_cleanup,
        plot=args.plot,
        plot_every=args.plot_every,
        keep_work=args.keep_work,
    )
    placer = DreamPlaceAdapter(config)

    root = REPO_ROOT / "external/MacroPlacement/Testcases/ICCAD04" / args.benchmark
    benchmark, plc = load_benchmark_from_dir(str(root))
    placement = placer.place(benchmark)

    result = compute_proxy_cost(placement, benchmark, plc)
    print(
        f"proxy={result['proxy_cost']:.4f}  "
        f"(wl={result['wirelength_cost']:.3f} "
        f"den={result['density_cost']:.3f} "
        f"cong={result['congestion_cost']:.3f})  "
        f"overlaps={result['overlap_count']}"
    )


if __name__ == "__main__":
    main()
