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

def _build_net_data(plc, n_hard, n_soft=0):
    """Extract nets as padded tensors for batched LSE wirelength.

    Node indices: hard macros [0, n_hard), soft macros [n_hard, n_hard+n_soft).
    If n_soft=0, only hard-macro nets are extracted.

    Returns:
        nets_padded: (M, max_deg) int64 — node indices per net
        nets_mask: (M, max_deg) bool — valid entries
    """
    name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx
    if n_soft > 0:
        for bidx, idx in enumerate(plc.soft_macro_indices):
            name_to_bidx[plc.modules_w_pins[idx].get_name()] = n_hard + bidx

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
                 num_steps=300, num_bins=None):
    """Gradient descent on LSE wirelength + density with annealing.

    Key techniques from DREAMPlace:
    1. Gamma annealing: LSE smoothing decreases over time (smooth → sharp)
    2. Adaptive lambda: density weight set from WL/density gradient ratio
    3. Degree-based preconditioning: highly-connected macros get smaller steps
    4. Lipschitz-aware step size
    """
    n = pos_init.shape[0]
    device = pos_init.device

    if nets_padded is None:
        return pos_init.clone()

    sizes_f = sizes.float().to(device)
    nets_padded = nets_padded.to(device)
    nets_mask = nets_mask.to(device)
    half_w = sizes_f[:, 0] / 2
    half_h = sizes_f[:, 1] / 2

    # Auto-tune grid resolution
    if num_bins is None:
        num_bins_x = max(8, min(64, int(canvas_w / (sizes_f[:, 0].mean() * 2))))
        num_bins_y = max(8, min(64, int(canvas_h / (sizes_f[:, 1].mean() * 2))))
    else:
        num_bins_x = num_bins_y = num_bins

    movable_mask = movable.bool().to(device)
    fixed_pos = pos_init.clone().float().to(device)

    # Gamma annealing: start smooth (large gamma), end sharp (small gamma)
    canvas_diag = math.sqrt(canvas_w ** 2 + canvas_h ** 2)
    gamma_start = canvas_diag / 10.0   # smooth
    gamma_end = canvas_diag / 200.0    # sharp, closer to true HPWL

    # Degree-based preconditioning: scale gradient inversely by connectivity
    degree = torch.zeros(n, device=device)
    if nets_padded is not None:
        for i in range(nets_padded.shape[0]):
            net = nets_padded[i][nets_mask[i]]
            degree[net] += 1
    degree = degree.clamp(min=1)
    precond = (1.0 / degree).unsqueeze(-1)  # (N, 1) — inverse degree per macro

    # Nesterov state
    pos = pos_init.clone().float().to(device).requires_grad_(True)
    velocity = torch.zeros_like(pos)

    # Adaptive lambda state
    lam = 0.01

    for step in range(num_steps):
        t = step / max(num_steps - 1, 1)

        # Enforce fixed macros
        with torch.no_grad():
            pos.data[~movable_mask] = fixed_pos[~movable_mask]

        # Anneal gamma: log-linear from gamma_start to gamma_end
        gamma = gamma_start * (gamma_end / gamma_start) ** t

        # Momentum schedule: 0.5 → 0.9
        momentum = 0.5 + 0.4 * t

        # Forward: wirelength
        wl = lse_wirelength(pos, nets_padded, nets_mask, gamma)

        # Forward: density
        den = smooth_density(pos, sizes_f, canvas_w, canvas_h,
                              num_bins_x, num_bins_y)

        # Compute gradients separately to balance lambda
        wl.backward(retain_graph=True)
        wl_grad = pos.grad.clone()
        pos.grad.zero_()

        den.backward()
        den_grad = pos.grad.clone()
        pos.grad.zero_()

        with torch.no_grad():
            # Zero gradients for fixed macros
            wl_grad[~movable_mask] = 0
            den_grad[~movable_mask] = 0

            # Adaptive lambda: balance WL and density gradient magnitudes
            wl_grad_norm = wl_grad[movable_mask].norm()
            den_grad_norm = den_grad[movable_mask].norm()
            if den_grad_norm > 1e-8:
                # Target: lambda * den_grad ~ wl_grad, scaled by progress
                target_ratio = 0.01 + 0.99 * t  # 0.01 early → 1.0 late
                lam = target_ratio * wl_grad_norm / den_grad_norm
                lam = min(lam, 100.0)  # cap

            # Combined gradient with preconditioning
            grad = (wl_grad + lam * den_grad) * precond

            # Lipschitz-aware step size: lr ~ canvas_size / grad_magnitude
            grad_norm = grad[movable_mask].norm()
            lr = 0.01 * canvas_diag / max(grad_norm.item(), 1e-8)
            lr = min(lr, canvas_diag * 0.05)

            # Nesterov update
            velocity = momentum * velocity - lr * grad
            pos.data += velocity

            # Clamp to canvas
            pos.data[:, 0].clamp_(half_w, canvas_w - half_w)
            pos.data[:, 1].clamp_(half_h, canvas_h - half_h)
            pos.data[~movable_mask] = fixed_pos[~movable_mask]

        if (step + 1) % 50 == 0 or step == 0:
            _progress(step + 1, num_steps, "global")

    _progress(num_steps, num_steps, "global")
    return pos.detach()


# ---------------------------------------------------------------------------
# Soft macro optimization: gradient descent on wirelength + light density
# ---------------------------------------------------------------------------

def optimize_soft(hard_pos, soft_pos_init, hard_sizes, soft_sizes,
                  canvas_w, canvas_h, nets_padded, nets_mask,
                  num_steps=200, base_lr=0.01, gamma=None):
    """Optimize soft macro positions with hard macros fixed.

    Minimizes wirelength (LSE) + light density penalty.
    No overlap constraints — soft macros may overlap each other.
    Uses the same LSE wirelength over the combined hard+soft position tensor.
    """
    n_hard = hard_pos.shape[0]
    n_soft = soft_pos_init.shape[0]

    if nets_padded is None or n_soft == 0:
        return soft_pos_init.clone()

    device = soft_pos_init.device
    hard_pos = hard_pos.to(device).float()
    soft_sizes = soft_sizes.to(device)
    nets_padded = nets_padded.to(device)
    nets_mask = nets_mask.to(device)

    # Only soft macros are optimized
    soft_pos = soft_pos_init.clone().float().requires_grad_(True)

    half_w = soft_sizes[:, 0].float() / 2
    half_h = soft_sizes[:, 1].float() / 2

    if gamma is None:
        gamma = max(1.0, math.sqrt(canvas_w * canvas_h) / 50.0)

    # Light density: fewer bins, weaker penalty
    num_bins = max(8, min(32, int(math.sqrt(n_hard + n_soft))))

    optimizer = torch.optim.Adam([soft_pos], lr=base_lr)

    # Compute initial WL for normalization
    with torch.no_grad():
        combined = torch.cat([hard_pos.float(), soft_pos], dim=0)
        wl_0 = lse_wirelength(combined, nets_padded, nets_mask, gamma).item()
    wl_norm = max(wl_0, 1.0)

    for step in range(num_steps):
        optimizer.zero_grad()

        combined = torch.cat([hard_pos.float(), soft_pos], dim=0)

        # Wirelength over all nets (hard + soft)
        wl = lse_wirelength(combined, nets_padded, nets_mask, gamma) / wl_norm

        # Light density on soft macros only (they can overlap, so weak penalty)
        den = smooth_density(soft_pos, soft_sizes.float(), canvas_w, canvas_h,
                              num_bins, num_bins) * 0.1

        loss = wl + den
        loss.backward()
        optimizer.step()

        # Clamp to canvas
        with torch.no_grad():
            soft_pos.data[:, 0].clamp_(half_w, canvas_w - half_w)
            soft_pos.data[:, 1].clamp_(half_h, canvas_h - half_h)

        if (step + 1) % 50 == 0 or step == 0:
            _progress(step + 1, num_steps, "softs")

    _progress(num_steps, num_steps, "softs")
    return soft_pos.detach()


# ---------------------------------------------------------------------------
# Main placer class
# ---------------------------------------------------------------------------

class SAPlacer:
    """DREAMPlace-style 2-stage global placement with legalization loop.

    Following the DREAMPlace flow:
    1. Global placement (gradient descent on all macros)
    2. Converge? → Legalize hard macros (spiral search)
    3. Fix hard macros, re-optimize soft macros
    4. Macros fixed? → if not, repeat from 1 with legalized positions
    """

    def __init__(self, seed: int = 42, num_steps: int = 300, num_rounds: int = 2,
                 device: str = "auto"):
        self.seed = seed
        self.num_steps = num_steps
        self.num_rounds = num_rounds
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        n_hard = benchmark.num_hard_macros
        n_soft = benchmark.num_soft_macros
        hard_sizes = benchmark.macro_sizes[:n_hard]
        soft_sizes = benchmark.macro_sizes[n_hard:n_hard+n_soft] if n_soft > 0 else None
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        movable = benchmark.get_movable_mask()[:n_hard]

        plc = _load_plc(benchmark.name)

        # Build net data
        hard_nets, hard_nets_mask = None, None
        all_nets, all_nets_mask = None, None
        if plc is not None:
            hard_nets, hard_nets_mask = _build_net_data(plc, n_hard)
            if n_soft > 0:
                all_nets, all_nets_mask = _build_net_data(plc, n_hard, n_soft)

        # Edges for spiral search
        if plc is not None:
            edges, edge_weights = _extract_hard_edges(plc)
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
            edge_weights = np.zeros(0, dtype=np.float64)

        sizes_np = hard_sizes.numpy().astype(np.float64)
        half_w = sizes_np[:, 0] / 2
        half_h = sizes_np[:, 1] / 2
        movable_np = movable.numpy()

        # Start from benchmark positions
        hard_pos = benchmark.macro_positions[:n_hard].clone()
        soft_pos = benchmark.macro_positions[n_hard:n_hard+n_soft].clone() if n_soft > 0 else None

        sys.stderr.write(f"  device: {self.device}\n")

        for round_i in range(self.num_rounds):
            steps = self.num_steps // self.num_rounds
            label = f"round {round_i+1}/{self.num_rounds}"
            sys.stderr.write(f"  {label}\n")

            # --- Stage 1: Global placement (hard macros, gradient descent) ---
            hard_pos = global_place(
                hard_pos.to(self.device), hard_sizes, movable, cw, ch,
                hard_nets, hard_nets_mask,
                num_steps=steps,
            ).cpu()

            # --- Legalize: fix hard macro overlaps (numpy, CPU only) ---
            pos_np = hard_pos.numpy().copy().astype(np.float64)
            hard_pos = spiralsearch(
                pos_np, movable_np, sizes_np, half_w, half_h, cw, ch, n_hard,
                edges=edges, edge_weights=edge_weights,
            )

            # --- Stage 2: Re-optimize soft macros with hard macros fixed ---
            if n_soft > 0 and all_nets is not None:
                soft_pos = optimize_soft(
                    hard_pos, soft_pos.to(self.device),
                    hard_sizes, soft_sizes,
                    cw, ch,
                    all_nets, all_nets_mask,
                    num_steps=steps,
                ).cpu()

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = hard_pos
        if n_soft > 0 and soft_pos is not None:
            full_pos[n_hard:n_hard+n_soft] = soft_pos

        return full_pos


if __name__ == "__main__":
    import argparse
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    parser = argparse.ArgumentParser(description="DREAMPlace-style analytical placer")
    parser.add_argument("--benchmark", "-b", required=True)
    parser.add_argument("--steps", "-n", type=int, default=1000)
    parser.add_argument("--rounds", "-r", type=int, default=2)
    parser.add_argument("--device", "-d", default="auto",
                        help="'auto', 'cpu', 'cuda', or 'mps' (Apple GPU)")
    args = parser.parse_args()

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / args.benchmark
    benchmark, plc = load_benchmark_from_dir(str(root))

    placer = SAPlacer(num_steps=args.steps, num_rounds=args.rounds, device=args.device)
    placement = placer.place(benchmark)

    result = compute_proxy_cost(placement, benchmark, plc)
    print(f"proxy={result['proxy_cost']:.4f}  "
          f"(wl={result['wirelength_cost']:.3f} "
          f"den={result['density_cost']:.3f} "
          f"cong={result['congestion_cost']:.3f})  "
          f"overlaps={result['overlap_count']}")
