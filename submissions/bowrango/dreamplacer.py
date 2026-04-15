"""
Minimal DREAMPlace-style analytical macro placer.

Core algorithm:
1. Initialize from benchmark positions
2. Gradient descent on: wirelength (log-sum-exp) + lambda * density (smooth)
3. Nesterov momentum, lambda ramped over iterations
4. Legalize with spiral search to remove remaining overlaps

Usage:
    uv run evaluate submissions/bowrango/dreamplacer.py -b ibm01
    uv run python submissions/bowrango/dreamplacer.py -b ibm01 --steps 500
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark

sys.path.insert(0, str(Path(__file__).resolve().parent))
from placer import _load_plc, _extract_hard_edges, spiralsearch, _progress


# ---------------------------------------------------------------------------
# Differentiable wirelength: log-sum-exp HPWL over nets
# ---------------------------------------------------------------------------

def _build_net_data(plc, n_hard):
    """Extract nets as padded tensors for batched LSE wirelength.

    Returns:
        nets_padded: (M, max_deg) int64 — macro indices per net
        nets_mask: (M, max_deg) bool — valid entries
    """
    name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx

    nets = []
    for driver, sinks in plc.nets.items():
        macros = set()
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])
        if len(macros) >= 2:
            nets.append(sorted(macros))

    if not nets:
        return None, None

    max_deg = max(len(net) for net in nets)
    M = len(nets)
    nets_padded = torch.zeros(M, max_deg, dtype=torch.long)
    nets_mask = torch.zeros(M, max_deg, dtype=torch.bool)
    for i, net in enumerate(nets):
        nets_padded[i, :len(net)] = torch.tensor(net, dtype=torch.long)
        nets_mask[i, :len(net)] = True

    return nets_padded, nets_mask


def lse_wirelength(pos, nets_padded, nets_mask, gamma):
    """Log-sum-exp approximation of HPWL over all nets.

    LSE-max ≈ max(x_i), LSE-min ≈ min(x_i)
    HPWL ≈ sum over nets of (LSE-max - LSE-min) in x and y
    """
    coords = pos[nets_padded]  # (M, max_deg, 2)

    big = 1e18
    mask_unsq = nets_mask.unsqueeze(-1)
    coords_max = coords.masked_fill(~mask_unsq, -big)
    coords_min = coords.masked_fill(~mask_unsq, big)

    lse_max = gamma * torch.logsumexp(coords_max / gamma, dim=1)  # (M, 2)
    lse_min = -gamma * torch.logsumexp(-coords_min / gamma, dim=1)  # (M, 2)

    return (lse_max - lse_min).sum()


# ---------------------------------------------------------------------------
# Smooth density: bell-shaped kernel on grid bins
# ---------------------------------------------------------------------------

def smooth_density(pos, sizes, canvas_w, canvas_h, num_bins_x, num_bins_y):
    """Compute smooth density on a grid using overlap-area kernel.

    Returns the density grid (num_bins_y, num_bins_x) and the overflow penalty.
    The penalty is sum of squared overflow above target density 1.0,
    matching the proxy cost focus on worst cells.
    """
    bin_w = canvas_w / num_bins_x
    bin_h = canvas_h / num_bins_y

    # Bin centers
    bx = torch.linspace(bin_w / 2, canvas_w - bin_w / 2, num_bins_x,
                         device=pos.device)
    by = torch.linspace(bin_h / 2, canvas_h - bin_h / 2, num_bins_y,
                         device=pos.device)

    hw = sizes[:, 0] / 2  # (N,)
    hh = sizes[:, 1] / 2

    # Smooth overlap in x: (N, num_bins_x)
    dx = pos[:, 0:1] - bx.unsqueeze(0)  # (N, Bx)
    overlap_x = torch.clamp(hw.unsqueeze(1) + bin_w / 2 - dx.abs(),
                             min=0, max=bin_w)

    # Smooth overlap in y: (N, num_bins_y)
    dy = pos[:, 1:2] - by.unsqueeze(0)  # (N, By)
    overlap_y = torch.clamp(hh.unsqueeze(1) + bin_h / 2 - dy.abs(),
                             min=0, max=bin_h)

    # Density grid: (num_bins_y, num_bins_x)
    # Each macro contributes overlap_y * overlap_x / bin_area to each bin
    density = torch.einsum('nj,ni->ji', overlap_x, overlap_y) / (bin_w * bin_h)

    # Penalty: squared overflow above target density
    target = 1.0
    overflow = F.relu(density - target)
    return (overflow ** 2).sum()


# ---------------------------------------------------------------------------
# Global placement: Nesterov gradient descent
# ---------------------------------------------------------------------------

def global_place(pos_init, sizes, movable, canvas_w, canvas_h,
                 nets_padded, nets_mask,
                 num_steps=300, base_lr=0.01,
                 gamma=None, num_bins=None):
    """Nesterov-accelerated gradient descent on LSE wirelength + density.

    Lambda (density weight) is ramped from small to large so that
    wirelength dominates early (spreading), then density forces
    push macros apart to reduce overlaps.
    """
    n = pos_init.shape[0]
    device = pos_init.device

    if nets_padded is None:
        return pos_init.clone()

    sizes_f = sizes.float().to(device)
    half_w = sizes_f[:, 0] / 2
    half_h = sizes_f[:, 1] / 2

    # Auto-tune gamma based on canvas size
    if gamma is None:
        gamma = max(1.0, math.sqrt(canvas_w * canvas_h) / 50.0)

    # Auto-tune grid resolution
    if num_bins is None:
        num_bins_x = max(8, min(64, int(canvas_w / (sizes_f[:, 0].mean() * 2))))
        num_bins_y = max(8, min(64, int(canvas_h / (sizes_f[:, 1].mean() * 2))))
    else:
        num_bins_x = num_bins_y = num_bins

    movable_mask = movable.bool().to(device)
    fixed_pos = pos_init.clone().float().to(device)

    # Nesterov state
    pos = pos_init.clone().float().to(device).requires_grad_(True)
    velocity = torch.zeros_like(pos)
    momentum = 0.9

    # Lambda schedule: ramp from lambda_0 to lambda_max
    lambda_0 = 0.001
    lambda_max = 5.0

    # Compute initial wirelength for normalization
    with torch.no_grad():
        wl_0 = lse_wirelength(pos, nets_padded, nets_mask, gamma).item()
    wl_norm = max(wl_0, 1.0)

    for step in range(num_steps):
        # Enforce fixed macros
        with torch.no_grad():
            pos.data[~movable_mask] = fixed_pos[~movable_mask]

        # Lambda ramp (log-linear)
        t = step / max(num_steps - 1, 1)
        lam = lambda_0 * (lambda_max / lambda_0) ** t

        # Forward
        wl = lse_wirelength(pos, nets_padded, nets_mask, gamma) / wl_norm
        den = smooth_density(pos, sizes_f, canvas_w, canvas_h,
                              num_bins_x, num_bins_y)
        loss = wl + lam * den

        # Backward
        loss.backward()

        # Nesterov update
        with torch.no_grad():
            grad = pos.grad.clone()
            grad[~movable_mask] = 0  # freeze fixed macros

            # Adaptive learning rate based on gradient magnitude
            grad_norm = grad.norm()
            lr = base_lr * canvas_w / max(grad_norm.item(), 1e-6)
            lr = min(lr, canvas_w * 0.1)  # cap at 10% of canvas

            velocity = momentum * velocity - lr * grad
            pos.data += velocity

            # Clamp to canvas
            pos.data[:, 0].clamp_(half_w, canvas_w - half_w)
            pos.data[:, 1].clamp_(half_h, canvas_h - half_h)
            pos.data[~movable_mask] = fixed_pos[~movable_mask]

        pos.grad.zero_()

        if (step + 1) % 50 == 0 or step == 0:
            _progress(step + 1, num_steps, "global")

    _progress(num_steps, num_steps, "global")
    return pos.detach()


# ---------------------------------------------------------------------------
# Main placer class
# ---------------------------------------------------------------------------

class SAPlacer:
    """DREAMPlace-style analytical placer + spiral search legalization."""

    def __init__(self, seed: int = 42, num_steps: int = 300):
        self.seed = seed
        self.num_steps = num_steps

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        n_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n_hard]
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        movable = benchmark.get_movable_mask()[:n_hard]

        plc = _load_plc(benchmark.name)

        # Build net data for LSE wirelength
        nets_padded, nets_mask = None, None
        if plc is not None:
            nets_padded, nets_mask = _build_net_data(plc, n_hard)

        # Extract edges for spiral search legalization
        if plc is not None:
            edges, edge_weights = _extract_hard_edges(plc)
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
            edge_weights = np.zeros(0, dtype=np.float64)

        # Phase 1: Global analytical placement
        pos_init = benchmark.macro_positions[:n_hard].clone()
        pos_global = global_place(
            pos_init, sizes, movable, cw, ch,
            nets_padded, nets_mask,
            num_steps=self.num_steps,
        )

        # Phase 2: Legalize with spiral search (resolve remaining overlaps)
        sizes_np = sizes.numpy().astype(np.float64)
        half_w = sizes_np[:, 0] / 2
        half_h = sizes_np[:, 1] / 2
        movable_np = movable.numpy()
        pos_np = pos_global.numpy().copy().astype(np.float64)

        pos_legal = spiralsearch(
            pos_np, movable_np, sizes_np, half_w, half_h, cw, ch, n_hard,
            edges=edges, edge_weights=edge_weights,
        )

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = pos_legal
        return full_pos


if __name__ == "__main__":
    import argparse
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    parser = argparse.ArgumentParser(description="DREAMPlace-style analytical placer")
    parser.add_argument("--benchmark", "-b", required=True)
    parser.add_argument("--steps", "-n", type=int, default=1000)
    args = parser.parse_args()

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / args.benchmark
    benchmark, plc = load_benchmark_from_dir(str(root))

    placer = SAPlacer(num_steps=args.steps)
    placement = placer.place(benchmark)

    result = compute_proxy_cost(placement, benchmark, plc)
    print(f"proxy={result['proxy_cost']:.4f}  "
          f"(wl={result['wirelength_cost']:.3f} "
          f"den={result['density_cost']:.3f} "
          f"cong={result['congestion_cost']:.3f})  "
          f"overlaps={result['overlap_count']}")
