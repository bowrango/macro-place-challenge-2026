"""
Bookshelf I/O for the DREAMPlace adapter.

DREAMPlace's `limbo018/dreamplace:cuda` image consumes the UCLA Bookshelf
format (.aux/.nodes/.nets/.pl/.scl/.wts), and emits a `.gp.pl` after global
placement. This module converts a `Benchmark` to that format and reads the
result back.

Coordinates: Bookshelf uses lower-left corners and integer units. We scale
microns by `SCALE` (default 10_000) so 1 micron == 10_000 units, then convert
centers ↔ corners.

Net pins: our `Benchmark.net_nodes` carries macro-level membership only (no
per-pin offsets), so we attach every pin at (0, 0) — the macro center — which
matches the proxy-cost HPWL formulation in `macro_place.objective`.

I/O ports are emitted as fixed terminals when they appear in `plc.nets`, so
DREAMPlace sees the boundary anchors that pull macros toward their natural
positions instead of collapsing everything to the centroid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark


SCALE = 10_000  # 1 micron == 10_000 Bookshelf units


@dataclass
class BookshelfIndex:
    """Mapping between Benchmark indices and the names written to disk."""
    macro_names: list[str]                # benchmark macro idx -> bookshelf name
    port_names: dict[str, str]            # plc port name -> bookshelf terminal name
    macro_index_by_plc_name: dict[str, int]  # plc macro name -> benchmark idx


def _scale(v: float) -> int:
    return int(round(v * SCALE))


def write_bookshelf(
    benchmark: Benchmark,
    out_dir: Path,
    plc=None,
) -> BookshelfIndex:
    """Write benchmark as Bookshelf files in `out_dir`. Returns index mapping
    to recover positions after DREAMPlace runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = benchmark.name

    cw = _scale(benchmark.canvas_width)
    ch = _scale(benchmark.canvas_height)

    macro_names = [f"m{i:06d}" for i in range(benchmark.num_macros)]

    # ── Ports (only those referenced by nets become terminals) ───────────
    port_names: dict[str, str] = {}
    macro_index_by_plc_name: dict[str, int] = {}
    if plc is not None:
        for bi, pi in enumerate(plc.hard_macro_indices):
            macro_index_by_plc_name[plc.modules_w_pins[pi].get_name()] = bi
        for bi, pi in enumerate(plc.soft_macro_indices):
            macro_index_by_plc_name[plc.modules_w_pins[pi].get_name()] = (
                benchmark.num_hard_macros + bi
            )

        port_set = {plc.modules_w_pins[i].get_name() for i in plc.port_indices}
        used_ports = set()
        for driver, sinks in plc.nets.items():
            for pin in [driver] + sinks:
                parent = pin.split("/")[0]
                if parent in port_set:
                    used_ports.add(parent)
        for k, p in enumerate(sorted(used_ports)):
            port_names[p] = f"p{k:06d}"

    movable_idx = (~benchmark.macro_fixed).nonzero(as_tuple=True)[0].tolist()
    fixed_macro_idx = benchmark.macro_fixed.nonzero(as_tuple=True)[0].tolist()

    n_terminals = len(port_names) + len(fixed_macro_idx)
    n_nodes = benchmark.num_macros + len(port_names)

    # ── .nodes ───────────────────────────────────────────────────────────
    with open(out_dir / f"{name}.nodes", "w") as f:
        f.write("UCLA nodes 1.0\n\n")
        f.write(f"NumNodes : {n_nodes}\n")
        f.write(f"NumTerminals : {n_terminals}\n\n")
        # Movable macros first (DREAMPlace doesn't actually require this order
        # but it's the ISPD convention).
        for i in movable_idx:
            w = max(_scale(float(benchmark.macro_sizes[i, 0])), 1)
            h = max(_scale(float(benchmark.macro_sizes[i, 1])), 1)
            f.write(f"\t{macro_names[i]}\t{w}\t{h}\n")
        for i in fixed_macro_idx:
            w = max(_scale(float(benchmark.macro_sizes[i, 0])), 1)
            h = max(_scale(float(benchmark.macro_sizes[i, 1])), 1)
            f.write(f"\t{macro_names[i]}\t{w}\t{h}\tterminal\n")
        for pname in port_names.values():
            f.write(f"\t{pname}\t1\t1\tterminal\n")

    # ── .pl ──────────────────────────────────────────────────────────────
    with open(out_dir / f"{name}.pl", "w") as f:
        f.write("UCLA pl 1.0\n\n")
        for i in range(benchmark.num_macros):
            cx = float(benchmark.macro_positions[i, 0])
            cy = float(benchmark.macro_positions[i, 1])
            w = float(benchmark.macro_sizes[i, 0])
            h = float(benchmark.macro_sizes[i, 1])
            llx = _scale(cx - w / 2)
            lly = _scale(cy - h / 2)
            tag = " /FIXED" if bool(benchmark.macro_fixed[i]) else ""
            f.write(f"{macro_names[i]} {llx} {lly} : N{tag}\n")
        if plc is not None:
            for plc_name, bs_name in port_names.items():
                node = plc.modules_w_pins[
                    next(i for i in plc.port_indices
                         if plc.modules_w_pins[i].get_name() == plc_name)
                ]
                px, py = node.get_pos()
                f.write(f"{bs_name} {_scale(px)} {_scale(py)} : N /FIXED\n")

    # ── .nets ────────────────────────────────────────────────────────────
    nets_records: list[tuple[list[str], str]] = []
    if plc is not None:
        for driver, sinks in plc.nets.items():
            seen = []
            for pin in [driver] + sinks:
                parent = pin.split("/")[0]
                if parent in macro_index_by_plc_name:
                    bs = macro_names[macro_index_by_plc_name[parent]]
                elif parent in port_names:
                    bs = port_names[parent]
                else:
                    continue
                if bs not in seen:
                    seen.append(bs)
            if len(seen) >= 2:
                nets_records.append((seen, f"n{len(nets_records)}"))
    else:
        # Fall back to benchmark.net_nodes (no port info available).
        for k, nodes in enumerate(benchmark.net_nodes):
            ids = [int(x) for x in nodes.tolist()]
            seen = [macro_names[i] for i in ids]
            if len(seen) >= 2:
                nets_records.append((seen, f"n{k}"))

    total_pins = sum(len(s) for s, _ in nets_records)
    with open(out_dir / f"{name}.nets", "w") as f:
        f.write("UCLA nets 1.0\n\n")
        f.write(f"NumNets : {len(nets_records)}\n")
        f.write(f"NumPins : {total_pins}\n\n")
        for nodes, net_name in nets_records:
            f.write(f"NetDegree : {len(nodes)}   {net_name}\n")
            for nd in nodes:
                # All pins at macro center → offset (0, 0). Matches the
                # proxy-cost HPWL definition in macro_place.objective.
                f.write(f"\t{nd}\tB : 0.0\t0.0\n")

    # ── .scl (rows that span the full canvas) ────────────────────────────
    # DREAMPlace needs at least one row to define the placement region.
    # We synthesize a uniform grid of horizontal rows.
    row_h = max(ch // 200, 1)
    n_rows = max(ch // row_h, 1)
    with open(out_dir / f"{name}.scl", "w") as f:
        f.write("UCLA scl 1.0\n\n")
        f.write(f"NumRows : {n_rows}\n\n")
        for i in range(n_rows):
            y = i * row_h
            f.write("CoreRow Horizontal\n")
            f.write(f"  Coordinate    :   {y}\n")
            f.write(f"  Height        :   {row_h}\n")
            f.write("  Sitewidth     :   1\n")
            f.write("  Sitespacing   :   1\n")
            f.write("  Siteorient    :   1\n")
            f.write("  Sitesymmetry  :   1\n")
            f.write(f"  SubrowOrigin  :   0\tNumSites  :  {cw}\n")
            f.write("End\n")

    # ── .wts (empty: all nets weight 1.0) ────────────────────────────────
    with open(out_dir / f"{name}.wts", "w") as f:
        f.write("UCLA wts 1.0\n")

    # ── .aux ─────────────────────────────────────────────────────────────
    with open(out_dir / f"{name}.aux", "w") as f:
        f.write(
            f"RowBasedPlacement : {name}.nodes {name}.nets "
            f"{name}.wts {name}.pl {name}.scl\n"
        )

    return BookshelfIndex(
        macro_names=macro_names,
        port_names=port_names,
        macro_index_by_plc_name=macro_index_by_plc_name,
    )


def read_pl(
    pl_file: Path,
    benchmark: Benchmark,
    index: BookshelfIndex,
) -> torch.Tensor:
    """Read DREAMPlace's `.gp.pl` output back into a (num_macros, 2) tensor of
    centers (microns). Fixed macros are restored from the benchmark."""
    name_to_idx = {n: i for i, n in enumerate(index.macro_names)}
    out = benchmark.macro_positions.clone()

    with open(pl_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("UCLA"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            node_name = parts[0]
            try:
                llx = float(parts[1])
                lly = float(parts[2])
            except ValueError:
                continue
            i = name_to_idx.get(node_name)
            if i is None:
                continue
            w = float(benchmark.macro_sizes[i, 0])
            h = float(benchmark.macro_sizes[i, 1])
            cx = llx / SCALE + w / 2
            cy = lly / SCALE + h / 2
            out[i, 0] = cx
            out[i, 1] = cy

    return out


def write_dreamplace_config(
    work_dir_in_container: str,
    name: str,
    config_path: Path,
    iterations: int = 1000,
    target_density: float = 1.0,
    gpu: int = 0,
    num_threads: int = 8,
    legalize: bool = True,
    detailed: bool = False,
) -> None:
    """Write a DREAMPlace JSON config pointing at the bookshelf files. Paths
    are written using the container-side directory."""
    import json

    aux = f"{work_dir_in_container}/{name}.aux"
    result_dir = work_dir_in_container

    # 256x256 bins is plenty for the small ibm canvases without exploding
    # memory on cpu-only Docker.
    nb = 256

    cfg = {
        "aux_input": aux,
        "result_dir": result_dir,
        "gpu": gpu,
        "num_bins_x": nb,
        "num_bins_y": nb,
        "global_place_stages": [
            {
                "num_bins_x": nb,
                "num_bins_y": nb,
                "iteration": iterations,
                "learning_rate": 0.01,
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
                "Llambda_density_weight_iteration": 1,
                "Lsub_iteration": 1,
            }
        ],
        "target_density": target_density,
        "density_weight": 8e-5,
        "gamma": 4.0,
        "random_seed": 1000,
        "scale_factor": 1.0,
        "ignore_net_degree": 100,
        "enable_fillers": 1,
        "gp_noise_ratio": 0.025,
        "global_place_flag": 1,
        "legalize_flag": 1 if legalize else 0,
        "detailed_place_flag": 1 if detailed else 0,
        "detailed_place_engine": "",
        "detailed_place_command": "",
        "stop_overflow": 0.07,
        "dtype": "float32",
        "plot_flag": 0,
        "random_center_init_flag": 0,
        "sort_nets_by_degree": 0,
        "num_threads": num_threads,
        "deterministic_flag": 1,
    }

    config_path.write_text(json.dumps(cfg, indent=2))
