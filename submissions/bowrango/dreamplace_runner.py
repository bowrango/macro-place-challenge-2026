"""
In-container shim. Patches `BasicPlace.build_legalization` to skip
DREAMPlace's greedy standard-cell legalizer (which, in the absence of real
std cells, re-shuffles already-legal macros and returns illegal positions),
then forwards to `dreamplace/Placer.py` via `runpy`.

The bare `import BasicPlace` is intentional: `NonLinearPlace.py` does the
same, and `import dreamplace.BasicPlace` would patch a different
`sys.modules` entry than the one the flow uses.
"""

import logging
import os
import sys

PLACER_PATH = "/DREAMPlace/install/dreamplace/Placer.py"
sys.path.insert(0, os.path.dirname(PLACER_PATH))

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

import runpy  # noqa: E402

if __name__ == "__main__":
    # argv passes through; Placer.py reads sys.argv[1] as the config path.
    runpy.run_path(PLACER_PATH, run_name="__main__")
