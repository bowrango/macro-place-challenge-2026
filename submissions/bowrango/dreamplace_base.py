"""Shared DREAMPlace adapter paths and logging."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


ADAPTER_DIR = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    for candidate in [ADAPTER_DIR, *ADAPTER_DIR.parents, Path.cwd()]:
        if (candidate / "macro_place").exists():
            return candidate
    return Path.cwd()


REPO_ROOT = find_repo_root()
DEFAULT_WORK_ROOT = Path(
    os.environ.get("DREAMPLACE_WORK_ROOT", str(REPO_ROOT / "dreamplace_work"))
)


def find_dreamplace_src() -> Path:
    override = os.environ.get("DREAMPLACE_SRC")
    if override:
        return Path(override)

    candidates = (
        REPO_ROOT / "external/DREAMPlace",
        ADAPTER_DIR / "DREAMPlace",
        Path.cwd() / "external/DREAMPlace",
        Path("/dreamplace"),
        Path("/DREAMPlace"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DREAMPLACE_SRC = find_dreamplace_src()
DEFAULT_IMAGE = (
    "limbo018/dreamplace:cuda"
    if sys.platform == "darwin"
    else "bowrango/dreamplace:cuda118"
)
DEFAULT_PLATFORM = "linux/amd64"
LOGGER_NAME = "dreamplace_adapter"
logger = logging.getLogger(LOGGER_NAME)


def ensure_default_logging() -> logging.Logger:
    current = logger
    while current is not None:
        if current.handlers:
            return logger
        current = current.parent
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("  dreamplace: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class AdapterConfigBase:
    """Base config for host paths, Docker runtime, and adapter logging."""

    __slots__ = (
        "image", "platform", "gpu", "keep_work", "timeout",
        "dreamplace_src", "work_root", "adapter_dir", "repo_root", "run_id",
    )

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        platform: str = DEFAULT_PLATFORM,
        gpu: int = 0,
        keep_work: bool = True,
        timeout: int = 1800,
        dreamplace_src: Path = DREAMPLACE_SRC,
        work_root: Path = DEFAULT_WORK_ROOT,
        adapter_dir: Path = ADAPTER_DIR,
        repo_root: Path = REPO_ROOT,
        run_id: Optional[str] = None,
    ):
        self.image = image
        self.platform = platform
        self.gpu = gpu
        self.keep_work = keep_work
        self.timeout = timeout
        self.dreamplace_src = Path(dreamplace_src)
        self.work_root = Path(work_root)
        self.adapter_dir = Path(adapter_dir)
        self.repo_root = Path(repo_root)
        self.run_id = run_id

    def work_dir_for(self, benchmark_name: str) -> Path:
        suffix = f"_{self.run_id}" if self.run_id else ""
        return self.work_root / f"{benchmark_name}{suffix}"

    def ensure_logging(self) -> logging.Logger:
        return ensure_default_logging()

    def as_dict(self) -> dict:
        result = {}
        for cls in reversed(type(self).mro()):
            for slot in getattr(cls, "__slots__", ()):
                if hasattr(self, slot):
                    value = getattr(self, slot)
                    result[slot] = str(value) if isinstance(value, Path) else value
        return result
