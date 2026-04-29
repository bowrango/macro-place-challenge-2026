"""
Minimal DREAMPlace-style analytical macro placer.

Core algorithm:
1. Initialize from benchmark positions
2. Gradient descent on: wirelength (log-sum-exp) + lambda * density (smooth)
3. Nesterov momentum, lambda ramped over iterations
4. Legalize via vectorized pairwise repulsion (GPU)

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
    """Smooth density via overlap-area kernel. Returns sum of squared overflow
    above target density 1.0, matching the proxy cost focus on worst cells."""
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
# Electrostatic density: FFT Poisson solver (DREAMPlace / ePlace)
# ---------------------------------------------------------------------------

def electrostatic_density(pos, sizes, canvas_w, canvas_h, num_bins_x, num_bins_y):
    """FFT-based electrostatic density loss, following DREAMPlace (Lin et al., DAC'19).

    Treat macros as charges rasterized onto a bin grid. Subtract the mean to
    neutralize total charge (required for a periodic-BC Poisson solver), then
    solve ∇²φ = -ρ in Fourier space where the Laplacian is multiplication by
    -(kx² + ky²). The electrostatic energy E = Σ ρ·φ is the density loss;
    autograd w.r.t. positions yields the electric field that repels macros
    from high-density regions — same mechanism DREAMPlace uses, just in
    PyTorch so torch.fft.rfft2 calls cuFFT on CUDA and MKL/FFTW on CPU.
    """
    device = pos.device
    bin_w = canvas_w / num_bins_x
    bin_h = canvas_h / num_bins_y

    # --- Rasterize macros onto the bin grid via 1D interval overlap ---
    bx = torch.linspace(bin_w / 2, canvas_w - bin_w / 2, num_bins_x, device=device)
    by = torch.linspace(bin_h / 2, canvas_h - bin_h / 2, num_bins_y, device=device)
    hw = sizes[:, 0:1] / 2  # (N, 1)
    hh = sizes[:, 1:2] / 2

    # Overlap of macro interval [pos - h, pos + h] with bin [center - size/2, center + size/2].
    left_x = torch.maximum(pos[:, 0:1] - hw, bx.unsqueeze(0) - bin_w / 2)
    right_x = torch.minimum(pos[:, 0:1] + hw, bx.unsqueeze(0) + bin_w / 2)
    overlap_x = (right_x - left_x).clamp(min=0)  # (N, Bx)

    left_y = torch.maximum(pos[:, 1:2] - hh, by.unsqueeze(0) - bin_h / 2)
    right_y = torch.minimum(pos[:, 1:2] + hh, by.unsqueeze(0) + bin_h / 2)
    overlap_y = (right_y - left_y).clamp(min=0)  # (N, By)

    # Shape (By, Bx) so rfft2's reduced axis (last) aligns with kx below.
    density = torch.einsum('ni,nj->ij', overlap_y, overlap_x) / (bin_w * bin_h)

    # --- Zero-mean charge (target density subtracted) ---
    density = density - density.mean()

    # --- Poisson solver: φ̂ = ρ̂ / (kx² + ky²), DC = 0 ---
    rho_hat = torch.fft.rfft2(density)  # (By, Bx // 2 + 1)
    kx = torch.fft.rfftfreq(num_bins_x, d=bin_w, device=device) * (2 * math.pi)
    ky = torch.fft.fftfreq(num_bins_y, d=bin_h, device=device) * (2 * math.pi)
    k2 = kx.unsqueeze(0) ** 2 + ky.unsqueeze(1) ** 2

    inv_k2 = torch.where(k2 > 0, 1.0 / k2.clamp(min=1e-12), torch.zeros_like(k2))
    phi = torch.fft.irfft2(rho_hat * inv_k2, s=(num_bins_y, num_bins_x))

    # Electrostatic energy; gradient w.r.t. pos is the density force.
    return (density * phi).sum() * (bin_w * bin_h)


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

        # Forward: density. Swap to `electrostatic_density` (FFT/Poisson, DREAMPlace
        # electrostatic) once the adaptive-lambda schedule is tuned for its larger
        # gradient magnitudes.
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
# Nearest-escape legalization
# ---------------------------------------------------------------------------

def legalize_snap(pos, sizes, movable_mask, canvas_w, canvas_h,
                  max_iter=50, eps=0.01):
    """Snap each movable macro to its nearest non-overlapping position.

    Processes macros largest-first. For each, if overlapping any already-
    placed macro, iteratively pushes along the smallest-overlap axis of the
    tightest-binding neighbor until free (or max_iter reached). No rings,
    no displacement-cost balancing — pure nearest-escape.
    """
    device = pos.device
    pos_np = pos.detach().cpu().numpy().astype(np.float64)
    sizes_np = sizes.cpu().numpy().astype(np.float64)
    movable_np = movable_mask.cpu().numpy()
    n = len(pos_np)

    order = np.argsort(-sizes_np[:, 0] * sizes_np[:, 1])
    placed = np.zeros(n, dtype=bool)

    for step_i, i in enumerate(order):
        _progress(step_i + 1, n, "legalize")
        if not movable_np[i]:
            placed[i] = True
            continue

        placed_idx = np.where(placed)[0]
        if placed_idx.size == 0:
            placed[i] = True
            continue

        x, y = pos_np[i]
        w, h = sizes_np[i]
        hw, hh = w / 2, h / 2
        px = pos_np[placed_idx, 0]
        py = pos_np[placed_idx, 1]
        sep_x = (w + sizes_np[placed_idx, 0]) / 2 + eps
        sep_y = (h + sizes_np[placed_idx, 1]) / 2 + eps

        for _ in range(max_iter):
            dx = x - px
            dy = y - py
            ox = np.maximum(0, sep_x - np.abs(dx))
            oy = np.maximum(0, sep_y - np.abs(dy))
            overlapping = (ox > 0) & (oy > 0)
            if not overlapping.any():
                break
            # Pick the overlap with smallest escape distance and push along that axis
            escape = np.minimum(ox, oy)
            escape[~overlapping] = np.inf
            j = int(escape.argmin())
            if ox[j] <= oy[j]:
                x += (1.0 if dx[j] >= 0 else -1.0) * ox[j]
                x = min(max(x, hw), canvas_w - hw)
            else:
                y += (1.0 if dy[j] >= 0 else -1.0) * oy[j]
                y = min(max(y, hh), canvas_h - hh)

        pos_np[i] = [x, y]
        placed[i] = True

    return torch.tensor(pos_np, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Main placer class
# ---------------------------------------------------------------------------

class SAPlacer:
    """DREAMPlace-style analytical placer.

    Flow:
    1. Gradient descent on hard macros (LSE wirelength + smooth density), on device.
    2. Snap each overlapping hard macro to its nearest non-overlapping position.
    3. Soft macros stay at their benchmark initial positions (validated to be good).
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
        hard_sizes = benchmark.macro_sizes[:n_hard]
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        movable = benchmark.get_movable_mask()[:n_hard]

        plc = _load_plc(benchmark.name)
        hard_nets, hard_nets_mask = (
            _build_net_data(plc, n_hard) if plc is not None else (None, None)
        )

        hard_pos = benchmark.macro_positions[:n_hard].clone().to(self.device)
        sys.stderr.write(f"  device: {self.device}\n")

        for round_i in range(self.num_rounds):
            steps = self.num_steps // self.num_rounds
            sys.stderr.write(f"  round {round_i+1}/{self.num_rounds}\n")

            hard_pos = global_place(
                hard_pos, hard_sizes, movable, cw, ch,
                hard_nets, hard_nets_mask,
                num_steps=steps,
            )

            hard_pos = legalize_snap(hard_pos, hard_sizes, movable, cw, ch)
            hard_pos = hard_pos.to(self.device)

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = hard_pos.cpu()
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
