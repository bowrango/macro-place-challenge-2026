"""
In-container shim. Patches `BasicPlace.build_legalization` to skip
DREAMPlace's greedy standard-cell legalizer (which, in the absence of real
std cells, re-shuffles already-legal macros and returns illegal positions),
hot-loads the editable `LevyPlacer.py` experiment, then forwards to
`dreamplace/Placer.py` via `runpy`.

The bare `import BasicPlace` is intentional: DREAMPlace's top-level modules do
the same, and `import dreamplace.BasicPlace` would patch a different
`sys.modules` entry than the one the flow uses.
"""

import importlib.util
import logging
import os
import sys

PLACER_PATH = "/dreamplace/install/dreamplace/Placer.py"
INSTALL_DREAMPLACE_DIR = os.path.dirname(PLACER_PATH)
SOURCE_DREAMPLACE_DIR = os.environ.get(
    "DREAMPLACE_SOURCE_DIR", "/dreamplace/dreamplace"
)

# Bare imports such as `import BasicPlace` should resolve to the installed
# Python package that matches the compiled ops.  We selectively override
# editable source files below.
sys.path.insert(0, INSTALL_DREAMPLACE_DIR)

import BasicPlace as BasicPlace_mod  # noqa: E402
import dreamplace.ops.macro_legalize.macro_legalize as macro_legalize  # noqa: E402


def _build_legalization_macro_only(self, params, placedb, data_collections, device):
    """Run only macro legalization; skip the row-based std-cell passes."""
    ml = macro_legalize.MacroLegalize(
        node_size_x=data_collections.node_size_x,
        node_size_y=data_collections.node_size_y,
        node_weights=data_collections.num_pins_in_nodes,
        flat_region_boxes=data_collections.flat_region_boxes,
        flat_region_boxes_start=data_collections.flat_region_boxes_start,
        node2fence_region_map=data_collections.node2fence_region_map,
        xl=placedb.xl,
        yl=placedb.yl,
        xh=placedb.xh,
        yh=placedb.yh,
        site_width=placedb.site_width,
        row_height=placedb.row_height,
        num_bins_x=placedb.num_bins_x,
        num_bins_y=placedb.num_bins_y,
        num_movable_nodes=placedb.num_movable_nodes,
        num_terminal_NIs=placedb.num_terminal_NIs,
        num_filler_nodes=placedb.num_filler_nodes,
    )

    def legalize_op(pos):
        logging.info("Start legalization (macro-only, greedy skipped)")
        return ml(pos, pos)

    return legalize_op


BasicPlace_mod.BasicPlace.build_legalization = _build_legalization_macro_only


def _load_source_override(module_name, source_name=None):
    """Load an editable DREAMPlace Python module without reinstalling.

    DREAMPlace's compiled ops live under /dreamplace/install and are tied to
    the container's torch build.  Putting the source tree ahead of install on
    PYTHONPATH would make `dreamplace.ops.*` resolve to source directories that
    do not contain the built extensions.  Instead, override only the top-level
    Python module we are actively experimenting with.
    """
    source_name = source_name or module_name
    path = os.path.join(SOURCE_DREAMPLACE_DIR, f"{source_name}.py")
    if not os.path.exists(path):
        return

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load DREAMPlace override {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    logging.info("Loaded editable DREAMPlace override: %s", path)


# _load_source_override("NonLinearPlace", source_name="LevyPlacer")
# logging.info("Set DREAMPlace engine to LevyPlacer")

import runpy  # noqa: E402

if __name__ == "__main__":
    # argv passes through; Placer.py reads sys.argv[1] as the config path.
    runpy.run_path(PLACER_PATH, run_name="__main__")
