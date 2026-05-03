"""
Stitch DREAMPlace's per-iteration PNGs into an MP4.

Run after a placement with `--plot --plot-every 1`. Default input is
`dreamplace_work/<bench>/<bench>/plot/iter*.png`; default output drops next
to that dir as `animation.mp4`.

    uv run python submissions/bowrango/make_mp4.py -b ibm01
    uv run python submissions/bowrango/make_mp4.py -b ibm01 --fps 60
    uv run python submissions/bowrango/make_mp4.py --dir path/to/plots -o out.mp4

Requires ffmpeg on PATH (`brew install ffmpeg` on macOS).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    for candidate in [script_dir, *script_dir.parents, Path.cwd()]:
        if (candidate / "macro_place").exists():
            return candidate
    return Path.cwd()


WORK_ROOT = Path(
    os.environ.get("DREAMPLACE_WORK_ROOT", str(_find_repo_root() / "dreamplace_work"))
)


def _plot_dir_for(benchmark: str) -> Path:
    return WORK_ROOT / benchmark / benchmark / "plot"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render DREAMPlace plot PNGs to MP4."
    )
    parser.add_argument(
        "--benchmark", "-b",
        help="Benchmark name; reads from dreamplace_work/<b>/<b>/plot/.",
    )
    parser.add_argument(
        "--dir", type=Path,
        help="Explicit PNG directory (overrides --benchmark).",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        help="Output MP4 path. Default: <plot_dir>/../animation.mp4.",
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.dir is not None:
        plot_dir = args.dir
    elif args.benchmark:
        plot_dir = _plot_dir_for(args.benchmark)
    else:
        parser.error("pass --benchmark/-b or --dir")

    if not plot_dir.is_dir():
        sys.exit(f"no such directory: {plot_dir}")

    frames = sorted(plot_dir.glob("iter*.png"))
    if not frames:
        sys.exit(f"no iter*.png under {plot_dir}")

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not on PATH. Install with `brew install ffmpeg`.")

    out = args.output if args.output is not None else plot_dir.parent / "animation.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Glob pattern preserves iteration ordering so legalization snapshots
    # slot in alongside global-placement frames. The scale filter rounds
    # to even dimensions, which libx264 requires.
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-pattern_type", "glob",
        "-i", str(plot_dir / "iter*.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    subprocess.check_call(cmd)
    print(f"wrote {out}  ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
