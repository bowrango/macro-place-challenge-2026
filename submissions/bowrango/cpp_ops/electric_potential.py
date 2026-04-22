"""Python wrapper for DREAMPlace's FFT-based electric potential density op.

Loads the compiled extension built by setup.py and exposes a clean
torch.autograd.Function. Falls back to raising ImportError if the
extension isn't built, so callers can detect and degrade to the pure-
PyTorch smooth_density fallback.

The DREAMPlace op computes a grid density map, solves Poisson's
equation via FFT to get the electric potential, and returns the energy
(sum of density * potential) as the density loss. Autograd gives the
electric field as the gradient w.r.t. positions — exactly what the
analytical placer needs.
"""

from __future__ import annotations

import torch

try:
    from cpp_ops import _electric_potential as _ep
except ImportError as e:
    raise ImportError(
        "DREAMPlace electric_potential extension is not built. Run:\n"
        "    cd submissions/bowrango/cpp_ops && uv run python setup.py build_ext --inplace"
    ) from e


class ElectricPotential(torch.autograd.Function):
    """FFT-based density loss. Inputs on CUDA; gradient w.r.t. pos is the field."""

    @staticmethod
    def forward(ctx, pos, node_size_x, node_size_y,
                bin_center_x, bin_center_y,
                xl, yl, xh, yh,
                bin_size_x, bin_size_y,
                num_movable_nodes, num_filler_nodes):
        # Exact binding name may be density_map_forward / electric_potential_forward
        # depending on DREAMPlace version — adjust to match what's exported in
        # _electric_potential.PYBIND11_MODULE (dir(_ep) after import).
        output = _ep.forward(
            pos, node_size_x, node_size_y,
            bin_center_x, bin_center_y,
            xl, yl, xh, yh,
            bin_size_x, bin_size_y,
            num_movable_nodes, num_filler_nodes,
        )
        ctx.save_for_backward(pos, node_size_x, node_size_y)
        ctx.bin_info = (bin_center_x, bin_center_y, xl, yl, xh, yh,
                        bin_size_x, bin_size_y,
                        num_movable_nodes, num_filler_nodes)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pos, node_size_x, node_size_y = ctx.saved_tensors
        (bin_center_x, bin_center_y, xl, yl, xh, yh,
         bin_size_x, bin_size_y,
         num_movable_nodes, num_filler_nodes) = ctx.bin_info
        grad = _ep.backward(
            grad_output, pos, node_size_x, node_size_y,
            bin_center_x, bin_center_y,
            xl, yl, xh, yh,
            bin_size_x, bin_size_y,
            num_movable_nodes, num_filler_nodes,
        )
        return (grad,) + (None,) * 12


def electric_potential(pos, node_size_x, node_size_y, num_bins_x, num_bins_y,
                        xl, yl, xh, yh, num_movable_nodes=None, num_filler_nodes=0):
    """Convenience wrapper — auto-computes bin centers from grid dimensions.

    Args:
        pos: (2, N) float32 tensor of [x_all; y_all]. CUDA preferred.
        node_size_x, node_size_y: (N,) tensors of macro widths/heights.
        num_bins_x, num_bins_y: density grid resolution.
        xl, yl, xh, yh: canvas bounds.
        num_movable_nodes: count of movable macros (first N_m entries of pos).
        num_filler_nodes: filler count (usually 0 for macro-only placement).

    Returns:
        Scalar density loss (torch.Tensor with grad).
    """
    device = pos.device
    bin_size_x = (xh - xl) / num_bins_x
    bin_size_y = (yh - yl) / num_bins_y
    bin_center_x = torch.arange(num_bins_x, device=device, dtype=pos.dtype) * bin_size_x + xl + bin_size_x / 2
    bin_center_y = torch.arange(num_bins_y, device=device, dtype=pos.dtype) * bin_size_y + yl + bin_size_y / 2
    if num_movable_nodes is None:
        num_movable_nodes = node_size_x.shape[0]
    return ElectricPotential.apply(
        pos, node_size_x, node_size_y,
        bin_center_x, bin_center_y,
        xl, yl, xh, yh,
        bin_size_x, bin_size_y,
        num_movable_nodes, num_filler_nodes,
    )
