"""
Connectivity-aware spiral-search macro placer.

Legalizes initial placement using greedy largest-first spiral search
with connectivity-weighted scoring.

Usage:
    uv run evaluate submissions/bowrango/placer.py -b ibm01
    uv run evaluate submissions/bowrango/placer.py --all
"""

import math
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# PlacementCost loading (needed for net extraction)
# ---------------------------------------------------------------------------

def _load_plc(name: str):
    from macro_place.loader import load_benchmark_from_dir, load_benchmark

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc

    ng45_map = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    d = ng45_map.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


# ---------------------------------------------------------------------------
# Net edge extraction (hard-macro-to-hard-macro weighted edges from hypergraph)
# ---------------------------------------------------------------------------

def _extract_edges(benchmark: Benchmark, plc):
    """Build pairwise weighted edge list from the net hypergraph."""
    name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx

    edge_dict: dict[tuple[int, int], float] = {}
    for driver, sinks in plc.nets.items():
        macros = set()
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])
        if len(macros) >= 2:
            ml = sorted(macros)
            w = 1.0 / (len(ml) - 1)
            for i in range(len(ml)):
                for j in range(i + 1, len(ml)):
                    pair = (ml[i], ml[j])
                    edge_dict[pair] = edge_dict.get(pair, 0.0) + w

    if not edge_dict:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64)

    edges = np.array(list(edge_dict.keys()), dtype=np.int64)
    weights = np.array([edge_dict[tuple(e)] for e in edges], dtype=np.float64)
    return edges, weights


# ---------------------------------------------------------------------------
# Legalization: spiral search, largest-first
# ---------------------------------------------------------------------------

def spiralsearch(pos, movable, sizes, half_w, half_h, cw, ch, n,
              edges=None, edge_weights=None):
    """Greedy legalization: place macros one-by-one, largest-area first.

    For each macro, search outward from its initial position in expanding
    square rings (spiral search). Each candidate is scored by:
        score = displacement² + alpha * connectivity_cost * canvas_diag
    where displacement² keeps macros near their original position, and
    connectivity_cost (weighted Manhattan distance to already-placed
    neighbours) pulls connected macros together to reduce wirelength.

    We don't stop at the first valid ring — we search EXTRA_RINGS beyond it
    to find better connectivity trade-offs at slightly larger displacement.
    """

    # Pairwise minimum separation: two macros i,j need center-to-center
    # distance >= (w_i + w_j)/2 in x and (h_i + h_j)/2 in y to not overlap.
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

    # Place largest macros first — they're hardest to fit so give them
    # first pick of positions.
    order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])

    # Build weighted adjacency list from the net hypergraph edges so we
    # can compute connectivity cost during the spiral search.
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    if edges is not None and len(edges) > 0:
        for k, (a, b) in enumerate(edges):
            w = edge_weights[k] if edge_weights is not None else 1.0
            adj[a].append((b, w))
            adj[b].append((a, w))

    placed = np.zeros(n, dtype=bool)
    any_placed = False  # fast flag to skip overlap checks when nothing placed yet
    legal = pos.copy()

    def _conn_cost(idx, cx, cy):
        """Weighted Manhattan distance to already-placed neighbours."""
        cost = 0.0
        for nb, w in adj[idx]:
            if placed[nb]:
                cost += w * (abs(cx - legal[nb, 0]) + abs(cy - legal[nb, 1]))
        return cost

    # Displacement is L2² (units²), connectivity is weighted L1 (units).
    # Multiply connectivity by canvas_diag to get units², then scale by
    # conn_alpha to control the displacement-vs-connectivity trade-off.
    canvas_diag = math.sqrt(cw ** 2 + ch ** 2)
    conn_alpha = 0.3

    # After finding the first valid ring, search this many extra rings
    # to see if a slightly farther position has much better connectivity.
    EXTRA_RINGS = 3

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            any_placed = True
            continue

        # Fast path: if current position doesn't overlap anything, keep it.
        if any_placed:
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            overlap = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
            overlap[idx] = False
            if not overlap.any():
                placed[idx] = True
                continue

        # Spiral search: expand outward in square rings from the original
        # position. Step size is proportional to macro size — finer for
        # small macros so they can slot into tighter gaps.
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.15
        best_p = legal[idx].copy()
        best_d = float("inf")
        first_valid_ring = -1

        for r in range(1, 300):
            # If we've searched enough extra rings past the first hit, stop.
            if first_valid_ring >= 0 and r > first_valid_ring + EXTRA_RINGS:
                break

            # Only visit cells on the perimeter of ring r (skip interior —
            # those were checked in earlier rings).
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue

                    # Candidate position, clamped to keep macro within canvas
                    cx = np.clip(pos[idx, 0] + dxm * step, half_w[idx], cw - half_w[idx])
                    cy = np.clip(pos[idx, 1] + dym * step, half_h[idx], ch - half_h[idx])

                    # Check overlap against all already-placed macros
                    if any_placed:
                        dx = np.abs(cx - legal[:, 0])
                        dy = np.abs(cy - legal[:, 1])
                        overlap = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
                        overlap[idx] = False
                        if overlap.any():
                            continue

                    # Score: displacement² + weighted connectivity
                    disp = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                    conn = _conn_cost(idx, cx, cy)
                    d = disp + conn_alpha * conn * canvas_diag

                    if d < best_d:
                        best_d = d
                        best_p = np.array([cx, cy])
                        if first_valid_ring < 0:
                            first_valid_ring = r

        legal[idx] = best_p
        placed[idx] = True
        any_placed = True

    return torch.tensor(legal, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main placer class
# ---------------------------------------------------------------------------

class SAPlacer:
    """Connectivity-aware spiral-search macro placer."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        n_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        movable = benchmark.get_movable_mask()[:n_hard].numpy()

        # Load PlacementCost for net extraction
        plc = _load_plc(benchmark.name)

        # Extract hard-macro net edges
        if plc is not None:
            edges, edge_weights = _extract_edges(benchmark, plc)
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
            edge_weights = np.zeros(0, dtype=np.float64)

        # Legalize initial placement
        pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        pos = spiralsearch(pos, movable, sizes, half_w, half_h, cw, ch, n_hard,
                        edges=edges, edge_weights=edge_weights)

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = pos
        return full_pos
