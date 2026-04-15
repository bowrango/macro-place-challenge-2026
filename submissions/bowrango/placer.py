"""
Connectivity-aware spiral-search macro placer.

Legalizes initial placement by resolving overlaps with minimal displacement.
Each macro spirals outward from its initial position, scored by:
    displacement² + alpha * connectivity_cost * canvas_diag

Usage:
    uv run evaluate submissions/bowrango/placer.py -b ibm01
    uv run evaluate submissions/bowrango/placer.py --all
    uv run python submissions/bowrango/placer.py -b ibm01 --visualize 10
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _progress(current, total, label="", width=30):
    """Print a simple inline progress bar."""
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    sys.stderr.write(f"\r  {label} [{bar}] {current}/{total}")
    if current >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


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
# Net edge extraction (hard-macro-to-hard-macro weighted edges)
# ---------------------------------------------------------------------------

def _extract_hard_edges(plc):
    """Build pairwise weighted edges between hard macros only."""
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
# Force-directed placement for soft macros
# ---------------------------------------------------------------------------

def _fd_step(plc):
    """Run one FD step on soft macros. Fast — called interleaved with placement."""
    plc.optimize_stdcells(
        use_current_loc=True,
        move_stdcells=True,
        move_macros=False,
        log_scale_conns=False,
        use_sizes=False,
        io_factor=1.0,
        num_steps=[1],
        max_move_distance=[1.0],
        attract_factor=[1.0],
        repel_factor=[1.0e4],
    )


def _read_soft_positions(full_pos, benchmark, plc):
    """Read soft macro positions from plc back into full_pos tensor."""
    n_hard = benchmark.num_hard_macros
    for i, macro_idx in enumerate(benchmark.soft_macro_indices):
        node = plc.modules_w_pins[macro_idx]
        x, y = node.get_pos()
        full_pos[n_hard + i, 0] = float(x)
        full_pos[n_hard + i, 1] = float(y)
    return full_pos


# ---------------------------------------------------------------------------
# Spiral search legalization
# ---------------------------------------------------------------------------

def spiralsearch(pos, movable, sizes, half_w, half_h, cw, ch, n,
              edges=None, edge_weights=None, snapshot_fn=None,
              fd_callback=None, fd_every=50):
    """Legalize overlapping macros with minimal displacement.

    Places macros one-by-one (largest-area first). Each macro spirals
    outward from its initial position. Candidates are scored by:
        displacement² + conn_alpha * weighted_manhattan_to_neighbors * canvas_diag
    """

    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

    order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    if edges is not None and len(edges) > 0:
        for k, (a, b) in enumerate(edges):
            w = edge_weights[k] if edge_weights is not None else 1.0
            adj[a].append((b, w))
            adj[b].append((a, w))

    placed = np.zeros(n, dtype=bool)
    any_placed = False
    legal = pos.copy()

    def _conn_cost(idx, cx, cy):
        cost = 0.0
        for nb, w in adj[idx]:
            if placed[nb]:
                cost += w * (abs(cx - legal[nb, 0]) + abs(cy - legal[nb, 1]))
        return cost

    canvas_diag = math.sqrt(cw ** 2 + ch ** 2)
    conn_alpha = 0.15
    EXTRA_RINGS = 5

    for step_i, idx in enumerate(order):
        _progress(step_i + 1, n, "legalize")
        if not movable[idx]:
            placed[idx] = True
            any_placed = True
            continue

        # Fast path: keep initial position if no overlap
        if any_placed:
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            overlap = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
            overlap[idx] = False
            if not overlap.any():
                placed[idx] = True
                if snapshot_fn is not None:
                    snapshot_fn(step_i, legal, placed)
                continue

        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.10
        best_p = legal[idx].copy()
        best_d = float("inf")
        first_valid_ring = -1

        for r in range(1, 300):
            if first_valid_ring >= 0 and r > first_valid_ring + EXTRA_RINGS:
                break

            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue

                    cx = np.clip(pos[idx, 0] + dxm * step, half_w[idx], cw - half_w[idx])
                    cy = np.clip(pos[idx, 1] + dym * step, half_h[idx], ch - half_h[idx])

                    if any_placed:
                        dx = np.abs(cx - legal[:, 0])
                        dy = np.abs(cy - legal[:, 1])
                        overlap = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
                        overlap[idx] = False
                        if overlap.any():
                            continue

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

        if snapshot_fn is not None:
            snapshot_fn(step_i, legal, placed)

        # Interleaved FD on soft macros every fd_every hard placements
        if fd_callback is not None and (step_i + 1) % fd_every == 0:
            fd_callback(legal)

    return torch.tensor(legal, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main placer class
# ---------------------------------------------------------------------------

class SAPlacer:
    """Connectivity-aware spiral-search macro placer."""

    def __init__(self, seed: int = 42, visualize_every: int = 0):
        self.seed = seed
        self.visualize_every = visualize_every

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

        plc = _load_plc(benchmark.name)

        if plc is not None:
            edges, edge_weights = _extract_hard_edges(plc)
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
            edge_weights = np.zeros(0, dtype=np.float64)

        snapshot_fn = None
        if self.visualize_every > 0:
            snapshot_fn = self._make_snapshot_fn(benchmark, plc)

        # Build FD callback: pushes hard positions into plc, runs 1 FD step
        fd_callback = None
        if plc is not None and benchmark.num_soft_macros > 0:
            from macro_place.objective import _set_placement
            def _fd_cb(legal_positions):
                fp = benchmark.macro_positions.clone()
                fp[:n_hard] = torch.tensor(legal_positions, dtype=torch.float32)
                _set_placement(plc, fp, benchmark)
                _fd_step(plc)
            fd_callback = _fd_cb

        pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        pos = spiralsearch(pos, movable, sizes, half_w, half_h, cw, ch, n_hard,
                        edges=edges, edge_weights=edge_weights,
                        snapshot_fn=snapshot_fn,
                        fd_callback=fd_callback, fd_every=50)

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = pos

        # Read final soft macro positions from plc
        if plc is not None and benchmark.num_soft_macros > 0:
            full_pos = _read_soft_positions(full_pos, benchmark, plc)

        return full_pos

    def _make_snapshot_fn(self, benchmark, plc):
        """Create a callback that saves placement snapshots with proxy cost overlay."""
        from macro_place.utils import visualize_placement
        from macro_place.objective import compute_proxy_cost
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vis_dir = Path("vis") / benchmark.name
        vis_dir.mkdir(parents=True, exist_ok=True)
        k = self.visualize_every
        n_hard = benchmark.num_hard_macros
        n_movable = int(benchmark.get_movable_mask()[:n_hard].sum())

        def snapshot(step_i, legal, placed):
            if step_i % k != 0 and step_i != n_movable - 1:
                return
            full_pos = benchmark.macro_positions.clone()
            full_pos[:n_hard] = torch.tensor(legal, dtype=torch.float32)

            costs = compute_proxy_cost(full_pos, benchmark, plc)

            save_path = str(vis_dir / f"step_{step_i:04d}.png")
            visualize_placement(full_pos, benchmark, save_path=save_path, plc=plc)

            img = plt.imread(save_path)
            fig, ax = plt.subplots(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=100)
            ax.imshow(img)
            ax.axis("off")

            label = (f"Step {step_i+1}/{n_movable}  |  "
                     f"proxy={costs['proxy_cost']:.4f}  "
                     f"wl={costs['wirelength_cost']:.3f}  "
                     f"den={costs['density_cost']:.3f}  "
                     f"cong={costs['congestion_cost']:.3f}  "
                     f"overlaps={costs['overlap_count']}")
            ax.text(0.5, 0.02, label, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", color="white",
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))

            fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=100)
            plt.close(fig)

        return snapshot


if __name__ == "__main__":
    """Run placer directly with --visualize support.

    Usage:
        uv run python submissions/bowrango/placer.py -b ibm01 --visualize 10

    Then create animation:
        ffmpeg -framerate 10 -pattern_type glob -i 'vis/ibm01/step_*.png' \
            -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p vis/ibm01.mp4
    """
    import argparse
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    parser = argparse.ArgumentParser(description="Run spiral-search macro placer")
    parser.add_argument("--benchmark", "-b", required=True)
    parser.add_argument("--visualize", "-v", type=int, default=0,
                        metavar="K", help="Save snapshot every K steps to vis/<benchmark>/")
    args = parser.parse_args()

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / args.benchmark
    benchmark, plc = load_benchmark_from_dir(str(root))

    placer = SAPlacer(visualize_every=args.visualize)
    placement = placer.place(benchmark)

    result = compute_proxy_cost(placement, benchmark, plc)
    print(f"proxy={result['proxy_cost']:.4f}  "
          f"(wl={result['wirelength_cost']:.3f} "
          f"den={result['density_cost']:.3f} "
          f"cong={result['congestion_cost']:.3f})  "
          f"overlaps={result['overlap_count']}")
    if args.visualize > 0:
        print(f"Frames saved to vis/{args.benchmark}/")
