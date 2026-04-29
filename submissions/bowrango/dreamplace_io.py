"""
Bookshelf I/O for the DREAMPlace adapter.

Converts `Benchmark` ↔ UCLA Bookshelf (.aux/.nodes/.nets/.pl/.scl/.wts) and
writes the JSON config that drives `dreamplace/Placer.py`.

Conventions:
- Coordinates: Bookshelf uses integer lower-left corners. We scale µm by
  `SCALE = 10_000` and convert centers ↔ corners.
- Pin offsets are sourced from `plc.modules_w_pins[i].x_offset/y_offset`
  when a `PlacementCost` object is provided, else default to (0, 0).
- I/O ports become fixed terminals; without them DREAMPlace has no
  boundary anchor and macros collapse to the centroid.
- Hard and soft macros are both movable; DREAMPlace co-optimizes them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark


logger = logging.getLogger("dreamplace_adapter")


SCALE = 10_000  # 1 µm == 10_000 Bookshelf units


@dataclass
class BookshelfIndex:
    """Mapping between Benchmark indices and the names written to disk."""
    macro_names: list[str]                    # benchmark macro idx -> bookshelf name
    port_names: dict[str, str]                # plc port name -> bookshelf terminal name
    macro_index_by_plc_name: dict[str, int]   # plc macro name -> benchmark idx


def _scale(v: float) -> int:
    return int(round(v * SCALE))


# ── write ────────────────────────────────────────────────────────────────────

def write_bookshelf(
    benchmark: Benchmark,
    out_dir: Path,
    plc=None,
) -> BookshelfIndex:
    """Write benchmark as Bookshelf files in `out_dir`. Returns the index
    needed to recover positions after DREAMPlace runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = benchmark.name
    cw = _scale(benchmark.canvas_width)
    ch = _scale(benchmark.canvas_height)

    macro_names = [f"m{i:06d}" for i in range(benchmark.num_macros)]

    bookshelf_fixed = benchmark.macro_fixed
    movable_idx = (~bookshelf_fixed).nonzero(as_tuple=True)[0].tolist()
    fixed_macro_idx = bookshelf_fixed.nonzero(as_tuple=True)[0].tolist()

    port_names, macro_index_by_plc_name, port_pos, pin_offsets = _collect_plc_index(
        plc, benchmark.num_hard_macros
    )

    n_terminals = len(port_names) + len(fixed_macro_idx)
    n_nodes = benchmark.num_macros + len(port_names)

    _write_nodes(out_dir / f"{name}.nodes", benchmark, macro_names,
                 movable_idx, fixed_macro_idx, port_names,
                 n_nodes, n_terminals)
    _write_pl(out_dir / f"{name}.pl", benchmark, macro_names,
              bookshelf_fixed, port_pos)
    _write_nets(out_dir / f"{name}.nets", benchmark, plc, macro_names,
                macro_index_by_plc_name, port_names, pin_offsets)
    _write_scl(out_dir / f"{name}.scl", cw, ch)
    (out_dir / f"{name}.wts").write_text("UCLA wts 1.0\n")
    (out_dir / f"{name}.aux").write_text(
        f"RowBasedPlacement : {name}.nodes {name}.nets "
        f"{name}.wts {name}.pl {name}.scl\n"
    )

    return BookshelfIndex(
        macro_names=macro_names,
        port_names=port_names,
        macro_index_by_plc_name=macro_index_by_plc_name,
    )


def _collect_plc_index(plc, n_hard: int):
    """Build name → index lookups from a PlacementCost object. Returns
    (port_names, macro_index_by_plc_name, port_pos, pin_offsets) where
    port_pos maps the bookshelf port name to its (x, y) in µm and
    pin_offsets maps full pin name ("macro/pin") to (dx, dy) in µm
    relative to the parent macro's center."""
    port_names: dict[str, str] = {}
    macro_index_by_plc_name: dict[str, int] = {}
    port_pos: dict[str, tuple[float, float]] = {}
    pin_offsets: dict[str, tuple[float, float]] = {}
    if plc is None:
        return port_names, macro_index_by_plc_name, port_pos, pin_offsets

    for bi, pi in enumerate(plc.hard_macro_indices):
        macro_index_by_plc_name[plc.modules_w_pins[pi].get_name()] = bi
    for bi, pi in enumerate(plc.soft_macro_indices):
        macro_index_by_plc_name[plc.modules_w_pins[pi].get_name()] = n_hard + bi

    port_set = {plc.modules_w_pins[i].get_name() for i in plc.port_indices}
    used_ports = set()
    for driver, sinks in plc.nets.items():
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in port_set:
                used_ports.add(parent)

    name_to_plc_idx = {plc.modules_w_pins[i].get_name(): i for i in plc.port_indices}
    for k, p in enumerate(sorted(used_ports)):
        port_names[p] = f"p{k:06d}"
        node = plc.modules_w_pins[name_to_plc_idx[p]]
        port_pos[port_names[p]] = node.get_pos()

    # Pin offsets are µm relative to the parent macro's center. Pin name
    # comes back as "macro_name/pin_name". Pins without explicit offsets
    # fall back to the macro center.
    pin_index_lists = [getattr(plc, "hard_macro_pin_indices", [])]
    if hasattr(plc, "soft_macro_pin_indices"):
        pin_index_lists.append(plc.soft_macro_pin_indices)
    for indices in pin_index_lists:
        for idx in indices:
            pin = plc.modules_w_pins[idx]
            full_name = pin.get_name()
            if "/" not in full_name:
                continue
            try:
                dx = float(pin.x_offset)
                dy = float(pin.y_offset)
            except AttributeError:
                continue
            pin_offsets[full_name] = (dx, dy)

    return port_names, macro_index_by_plc_name, port_pos, pin_offsets


def _bookshelf_size(benchmark, i: int) -> tuple[int, int]:
    """Width and height in Bookshelf integer units."""
    w = max(_scale(float(benchmark.macro_sizes[i, 0])), 1)
    h = max(_scale(float(benchmark.macro_sizes[i, 1])), 1)
    return w, h


def _write_nodes(path, benchmark, macro_names, movable_idx, fixed_macro_idx,
                 port_names, n_nodes, n_terminals):
    with open(path, "w") as f:
        f.write("UCLA nodes 1.0\n\n")
        f.write(f"NumNodes : {n_nodes}\n")
        f.write(f"NumTerminals : {n_terminals}\n\n")
        for i in movable_idx:
            w, h = _bookshelf_size(benchmark, i)
            f.write(f"\t{macro_names[i]}\t{w}\t{h}\n")
        for i in fixed_macro_idx:
            w, h = _bookshelf_size(benchmark, i)
            f.write(f"\t{macro_names[i]}\t{w}\t{h}\tterminal\n")
        for pname in port_names.values():
            f.write(f"\t{pname}\t1\t1\tterminal\n")


def _write_pl(path, benchmark, macro_names, bookshelf_fixed, port_pos):
    with open(path, "w") as f:
        f.write("UCLA pl 1.0\n\n")
        for i in range(benchmark.num_macros):
            cx = float(benchmark.macro_positions[i, 0])
            cy = float(benchmark.macro_positions[i, 1])
            w, h = _bookshelf_size(benchmark, i)
            llx = _scale(cx) - w // 2
            lly = _scale(cy) - h // 2
            tag = " /FIXED" if bool(bookshelf_fixed[i]) else ""
            f.write(f"{macro_names[i]} {llx} {lly} : N{tag}\n")
        for bs_name, (px, py) in port_pos.items():
            f.write(f"{bs_name} {_scale(px)} {_scale(py)} : N /FIXED\n")


def _write_nets(path, benchmark, plc, macro_names, macro_index_by_plc_name,
                port_names, pin_offsets):
    """Write one entry per pin (multiple pins on the same node are allowed)
    so DREAMPlace's HPWL and macro legalizer get real per-pin spatial signal."""
    # Each entry: (entries, net_name) where entries is list of (bs, dx, dy).
    nets_records: list[tuple[list[tuple[str, float, float]], str]] = []
    if plc is not None:
        for driver, sinks in plc.nets.items():
            entries: list[tuple[str, float, float]] = []
            for pin in [driver] + sinks:
                parent = pin.split("/")[0]
                if parent in macro_index_by_plc_name:
                    bs = macro_names[macro_index_by_plc_name[parent]]
                    dx, dy = pin_offsets.get(pin, (0.0, 0.0))
                elif parent in port_names:
                    bs = port_names[parent]
                    dx, dy = 0.0, 0.0
                else:
                    continue
                entries.append((bs, dx, dy))
            if len(entries) >= 2:
                nets_records.append((entries, f"n{len(nets_records)}"))
    else:
        # Fallback: no plc → no pin info → all pins at macro center.
        for k, nodes in enumerate(benchmark.net_nodes):
            ids = [int(x) for x in nodes.tolist()]
            entries = [(macro_names[i], 0.0, 0.0) for i in ids]
            if len(entries) >= 2:
                nets_records.append((entries, f"n{k}"))

    total_pins = sum(len(e) for e, _ in nets_records)
    with open(path, "w") as f:
        f.write("UCLA nets 1.0\n\n")
        f.write(f"NumNets : {len(nets_records)}\n")
        f.write(f"NumPins : {total_pins}\n\n")
        for entries, net_name in nets_records:
            f.write(f"NetDegree : {len(entries)}   {net_name}\n")
            for bs, dx, dy in entries:
                f.write(f"\t{bs}\tB : {_scale(dx)}\t{_scale(dy)}\n")


def _write_scl(path, cw: int, ch: int):
    """Write a uniform horizontal-row grid. Sitewidth=1 lets macros align
    to any integer position after legalization."""
    row_h = max(ch // 200, 1)
    n_rows = max(ch // row_h, 1)
    with open(path, "w") as f:
        f.write("UCLA scl 1.0\n\n")
        f.write(f"NumRows : {n_rows}\n\n")
        for i in range(n_rows):
            f.write("CoreRow Horizontal\n")
            f.write(f"  Coordinate    :   {i * row_h}\n")
            f.write(f"  Height        :   {row_h}\n")
            f.write("  Sitewidth     :   1\n")
            f.write("  Sitespacing   :   1\n")
            f.write("  Siteorient    :   1\n")
            f.write("  Sitesymmetry  :   1\n")
            f.write(f"  SubrowOrigin  :   0\tNumSites  :  {cw}\n")
            f.write("End\n")


# ── read ─────────────────────────────────────────────────────────────────────

def read_pl(
    pl_file: Path,
    benchmark: Benchmark,
    index: BookshelfIndex,
) -> torch.Tensor:
    """Parse DREAMPlace's `.pl` into a (num_macros, 2) tensor of centers (µm).
    Nodes absent from the file keep their initial position. Warns if any
    coords are fractional — that signals macros the legalizer's LP step
    couldn't snap to an integer Hannan grid position."""
    name_to_idx = {n: i for i, n in enumerate(index.macro_names)}
    out = benchmark.macro_positions.clone()

    fractional: list[tuple[str, float, float]] = []

    with open(pl_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("UCLA"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            i = name_to_idx.get(parts[0])
            if i is None:
                continue
            try:
                llx, lly = float(parts[1]), float(parts[2])
            except ValueError:
                continue
            if llx != int(llx) or lly != int(lly):
                fractional.append((parts[0], llx, lly))
            w_int, h_int = _bookshelf_size(benchmark, i)
            out[i, 0] = (llx + w_int / 2) / SCALE
            out[i, 1] = (lly + h_int / 2) / SCALE

    if fractional:
        sample = ", ".join(f"{n}@({x},{y})" for n, x, y in fractional[:3])
        logger.warning(
            "%d node(s) in %s have non-integer Bookshelf coords "
            "(legalizer LP-stage leftovers, e.g. %s)",
            len(fractional), pl_file.name, sample,
        )

    return out


# ── config ───────────────────────────────────────────────────────────────────

def write_dreamplace_config(
    work_dir_in_container: str,
    name: str,
    config_path: Path,
    iterations: int = 3000,
    target_density: float = 0.80,
    gpu: int = 0,
    num_threads: int = 8,
    legalize: bool = True,
    detailed: bool = False,
    enable_fillers: bool = False,
    stop_overflow: float = 0.005,
    plot: bool = False,
    num_bins: int = 256,
) -> None:
    """Write a DREAMPlace JSON config. Paths are container-side."""
    nb = num_bins
    cfg = {
        "aux_input": f"{work_dir_in_container}/{name}.aux",
        "result_dir": work_dir_in_container,
        "gpu": gpu,
        "num_bins_x": nb,
        "num_bins_y": nb,
        "global_place_stages": [{
            "num_bins_x": nb,
            "num_bins_y": nb,
            "iteration": iterations,
            "learning_rate": 0.01,
            "wirelength": "weighted_average",
            "optimizer": "nesterov",
            "Llambda_density_weight_iteration": 1,
            "Lsub_iteration": 1,
        }],
        "target_density": target_density,
        "density_weight": 8e-5,
        "gamma": 4.0,
        "random_seed": 1000,
        "scale_factor": 1.0,
        "ignore_net_degree": 100,
        "enable_fillers": 1 if enable_fillers else 0,
        "gp_noise_ratio": 0.025,
        "global_place_flag": 1,
        "legalize_flag": 1 if legalize else 0,
        "detailed_place_flag": 1 if detailed else 0,
        "detailed_place_engine": "",
        "detailed_place_command": "",
        "stop_overflow": stop_overflow,
        "dtype": "float32",
        "plot_flag": 1 if plot else 0,
        "random_center_init_flag": 0,
        "sort_nets_by_degree": 0,
        "num_threads": num_threads,
        "deterministic_flag": 1,
    }
    config_path.write_text(json.dumps(cfg, indent=2))
