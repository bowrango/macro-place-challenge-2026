"""Optuna tuning for DREAMPlace hyperparameters.

Example:
    uv run python submissions/bowrango/gridsearch.py -b ibm01 --n-trials 20
    uv run python submissions/bowrango/gridsearch.py --all
    uv run python submissions/bowrango/gridsearch.py --all --feature-model
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import optuna
from optuna.trial import TrialState


HERE = Path(__file__).resolve().parent
REPO_ROOT = next(
    (p for p in (HERE, *HERE.parents) if (p / "macro_place").exists()),
    Path.cwd(),
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from macro_place.benchmark import Benchmark  # noqa: E402
from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402

from dreamplace_adapter import (  # noqa: E402
    DreamPlaceAdapter,
    DreamPlaceConfig,
)


DEFAULT_STORAGE = f"sqlite:///{HERE / 'gridsearch.db'}"
ICCAD04_ROOT = REPO_ROOT / "external/MacroPlacement/Testcases/ICCAD04"
FEATURE_KEYS = (
    "utilization",
    "log_num_macros",
    "log_num_nets",
    "hard_macro_ratio",
    "aspect_skew",
)
LOG_ITERATION_COEF_BOUND = 0.25
LOG_DENSITY_WEIGHT_COEF_BOUND = 0.45
TARGET_DENSITY_COEF_BOUND = 0.06
COARSE_TRIAL_FRACTION = 0.7


def _detect_gpu() -> int:
    """Return 1 when an NVIDIA Docker runtime is likely available."""
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    return 1 if proc.returncode == 0 else 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
    return parsed


def _discover_benchmarks(root: Path = ICCAD04_ROOT) -> list[str]:
    if not root.exists():
        return []
    return sorted(p.parent.name for p in root.glob("*/netlist.pb.txt"))


def _load_cases(names: list[str]) -> list[dict[str, Any]]:
    cases = []
    for name in names:
        benchmark_dir = ICCAD04_ROOT / name
        benchmark, plc = load_benchmark_from_dir(str(benchmark_dir))
        cases.append(
            {
                "name": name,
                "benchmark": benchmark,
                "plc": plc,
                "features": _benchmark_features(benchmark),
            }
        )
    feature_stats = _standardize_features(cases)
    for case in cases:
        case["feature_stats"] = feature_stats
    return cases


def _benchmark_features(benchmark: Benchmark) -> dict[str, float]:
    canvas_area = float(benchmark.canvas_width * benchmark.canvas_height)
    macro_area = float((benchmark.macro_sizes[:, 0] * benchmark.macro_sizes[:, 1]).sum())
    num_macros = max(float(benchmark.num_macros), 1.0)
    num_nets = max(float(benchmark.num_nets), 1.0)
    aspect_ratio = float(benchmark.canvas_width / benchmark.canvas_height)
    movable = benchmark.get_movable_mask()
    return {
        "canvas_area": canvas_area,
        "aspect_ratio": aspect_ratio,
        "utilization": macro_area / canvas_area if canvas_area else 0.0,
        "num_macros": num_macros,
        "num_hard_macros": float(benchmark.num_hard_macros),
        "num_soft_macros": float(benchmark.num_soft_macros),
        "num_nets": num_nets,
        "movable_ratio": float(movable.float().mean().item()) if benchmark.num_macros else 0.0,
        "log_canvas_area": math.log(max(canvas_area, 1e-12)),
        "log_num_macros": math.log(num_macros),
        "log_num_nets": math.log(num_nets),
        "hard_macro_ratio": float(benchmark.num_hard_macros) / num_macros,
        "aspect_skew": abs(math.log(max(aspect_ratio, 1e-12))),
    }


def _standardize_features(cases: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    # Normalize each selected feature across the benchmarks in this study:
    # z = (feature - study_mean) / study_std. Constant features become 0.
    stats = {}
    for key in FEATURE_KEYS:
        values = [float(case["features"][key]) for case in cases]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        stats[key] = {"mean": mean, "std": std}
        for case, value in zip(cases, values):
            feature_z = case.setdefault("feature_z", {})
            feature_z[key] = 0.0 if std < 1e-12 else (value - mean) / std
    return stats


def _active_feature_keys(cases: list[dict[str, Any]]) -> tuple[str, ...]:
    # Skip features that are constant for the selected benchmarks; their
    # coefficients would have no effect and only waste Optuna trials.
    return tuple(
        key
        for key in FEATURE_KEYS
        if any(abs(float(case["feature_z"][key])) > 1e-12 for case in cases)
    )


def _geomean(values: list[float]) -> float:
    eps = 1e-12
    return math.exp(sum(math.log(max(v, eps)) for v in values) / len(values))


def _aggregate(values: list[float], method: str) -> float:
    if method == "geomean":
        return _geomean(values)
    return sum(values) / len(values)


def _make_config(params: dict[str, Any], gpu: int) -> DreamPlaceConfig:
    return DreamPlaceConfig(gpu=gpu, **params)


def _sample_direct_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int(
            "iterations",
            args.min_iterations,
            args.max_iterations,
            step=args.iteration_step,
        ),
        "density_weight": trial.suggest_float(
            "density_weight",
            args.min_density_weight,
            args.max_density_weight,
            log=True,
        ),
        "target_density": trial.suggest_float(
            "target_density",
            args.min_target_density,
            args.max_target_density,
        ),
    }


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _round_iterations(value: float, args: argparse.Namespace) -> int:
    clipped = _clip(value, args.min_iterations, args.max_iterations)
    steps = round((clipped - args.min_iterations) / args.iteration_step)
    rounded = args.min_iterations + steps * args.iteration_step
    return int(_clip(rounded, args.min_iterations, args.max_iterations))


def _sample_param_model(
    trial: optuna.Trial,
    args: argparse.Namespace,
    feature_keys: tuple[str, ...],
) -> dict[str, Any]:
    # Optuna samples a linear feature model, not per-benchmark params:
    #   log(iterations)      = log(base) + sum(coef_i * z_i)
    #   log(density_weight)  = log(base) + sum(coef_i * z_i)
    #   target_density       = base      + sum(coef_i * z_i)
    model = {
        "iterations_base": trial.suggest_int(
            "iterations_base",
            args.min_iterations,
            args.max_iterations,
            step=args.iteration_step,
        ),
        "density_weight_base": trial.suggest_float(
            "density_weight_base",
            args.min_density_weight,
            args.max_density_weight,
            log=True,
        ),
        "target_density_base": trial.suggest_float(
            "target_density_base",
            args.min_target_density,
            args.max_target_density,
        ),
        "feature_keys": feature_keys,
        "iterations_logcoef": {},
        "density_weight_logcoef": {},
        "target_density_coef": {},
    }
    for key in feature_keys:
        model["iterations_logcoef"][key] = trial.suggest_float(
            f"iterations_logcoef__{key}",
            -LOG_ITERATION_COEF_BOUND,
            LOG_ITERATION_COEF_BOUND,
        )
        model["density_weight_logcoef"][key] = trial.suggest_float(
            f"density_weight_logcoef__{key}",
            -LOG_DENSITY_WEIGHT_COEF_BOUND,
            LOG_DENSITY_WEIGHT_COEF_BOUND,
        )
        model["target_density_coef"][key] = trial.suggest_float(
            f"target_density_coef__{key}",
            -TARGET_DENSITY_COEF_BOUND,
            TARGET_DENSITY_COEF_BOUND,
        )
    return model


def _params_for_case(
    model: dict[str, Any],
    feature_z: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    # Convert the sampled model into concrete DREAMPlace params for one
    # benchmark, then clip back to the user-provided search bounds.
    log_iterations = math.log(float(model["iterations_base"]))
    log_density_weight = math.log(float(model["density_weight_base"]))
    target_density = float(model["target_density_base"])

    for key in model["feature_keys"]:
        z = float(feature_z[key])
        log_iterations += model["iterations_logcoef"][key] * z
        log_density_weight += model["density_weight_logcoef"][key] * z
        target_density += model["target_density_coef"][key] * z

    return {
        "iterations": _round_iterations(math.exp(log_iterations), args),
        "density_weight": _clip(
            math.exp(log_density_weight),
            args.min_density_weight,
            args.max_density_weight,
        ),
        "target_density": _clip(
            target_density,
            args.min_target_density,
            args.max_target_density,
        ),
    }


def _record_case_attrs(
    trial: optuna.Trial,
    name: str,
    case: dict[str, Any],
    params: dict[str, Any],
    costs: dict[str, Any],
    elapsed: float,
) -> None:
    for feature, value in case["features"].items():
        trial.set_user_attr(f"{name}/feature/{feature}", value)
    for param, value in params.items():
        trial.set_user_attr(f"{name}/param/{param}", value)
    trial.set_user_attr(f"{name}/proxy_cost", float(costs["proxy_cost"]))
    trial.set_user_attr(f"{name}/wirelength_cost", float(costs["wirelength_cost"]))
    trial.set_user_attr(f"{name}/density_cost", float(costs["density_cost"]))
    trial.set_user_attr(f"{name}/congestion_cost", float(costs["congestion_cost"]))
    trial.set_user_attr(f"{name}/overlap_count", int(costs["overlap_count"]))
    trial.set_user_attr(f"{name}/seconds", elapsed)


def make_direct_objective(case: dict[str, Any], args: argparse.Namespace, gpu: int):
    def objective(trial: optuna.Trial) -> float:
        name = case["name"]
        params = _sample_direct_params(trial, args)
        placer = DreamPlaceAdapter(_make_config(params, gpu))
        start = time.time()
        placement = placer.place(case["benchmark"])
        costs = compute_proxy_cost(placement, case["benchmark"], case["plc"])
        elapsed = time.time() - start
        _record_case_attrs(trial, name, case, params, costs, elapsed)
        return float(costs["proxy_cost"])

    return objective


def make_feature_objective(
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    gpu: int,
    feature_keys: tuple[str, ...],
):
    def objective(trial: optuna.Trial) -> float:
        model = _sample_param_model(trial, args, feature_keys)
        scores = []

        for idx, case in enumerate(cases):
            name = case["name"]
            benchmark = case["benchmark"]
            plc = case["plc"]
            params = _params_for_case(model, case["feature_z"], args)
            placer = DreamPlaceAdapter(_make_config(params, gpu))
            start = time.time()

            for feature, value in case["feature_z"].items():
                trial.set_user_attr(f"{name}/feature_z/{feature}", value)

            placement = placer.place(benchmark)
            costs = compute_proxy_cost(placement, benchmark, plc)
            elapsed = time.time() - start

            score = float(costs["proxy_cost"])
            scores.append(score)

            _record_case_attrs(trial, name, case, params, costs, elapsed)

            trial.report(_aggregate(scores, args.aggregate), step=idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return _aggregate(scores, args.aggregate)

    return objective


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune DREAMPlace iterations, density_weight, and target_density."
    )
    parser.add_argument("--benchmark", "-b", action="append", dest="benchmarks")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--feature-model", action="store_true")
    parser.add_argument("--n-trials", type=_positive_int, default=20)
    parser.add_argument("--study-timeout", type=int, default=None)
    parser.add_argument("--study-name", default="dreamplace_direct_proxy_search")
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--aggregate", choices=("mean", "geomean"), default="mean")

    parser.add_argument("--min-iterations", type=_positive_int, default=200)
    parser.add_argument("--max-iterations", type=_positive_int, default=3000)
    parser.add_argument("--iteration-step", type=_positive_int, default=100)
    parser.add_argument("--min-density-weight", type=float, default=1e-5)
    parser.add_argument("--max-density-weight", type=float, default=2e-4)
    parser.add_argument("--min-target-density", type=float, default=0.60)
    parser.add_argument("--max-target-density", type=float, default=1)

    args = parser.parse_args()

    if args.min_iterations > args.max_iterations:
        parser.error("--min-iterations cannot exceed --max-iterations")
    if args.min_density_weight <= 0 or args.max_density_weight <= 0:
        parser.error("density weight bounds must be positive")
    if args.min_density_weight > args.max_density_weight:
        parser.error("--min-density-weight cannot exceed --max-density-weight")
    if not 0 < args.min_target_density <= args.max_target_density <= 1:
        parser.error("target density bounds must satisfy 0 < min <= max <= 1")

    if args.all:
        args.benchmarks = _discover_benchmarks()
    elif not args.benchmarks:
        args.benchmarks = ["ibm01"]
    if not args.benchmarks:
        parser.error("no benchmarks selected")

    missing = [
        name
        for name in args.benchmarks
        if not (ICCAD04_ROOT / name / "netlist.pb.txt").exists()
    ]
    if missing:
        parser.error(f"unknown benchmark(s): {', '.join(missing)}")

    return args


def _print_best(study: optuna.Study) -> None:
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"finished_trials={len(study.trials)} complete_trials={len(complete)}")
    if not complete:
        print("No completed trials.")
        return

    best = study.best_trial
    print(f"best_trial={best.number} best_value={best.value:.6f}")
    print("best_params=" + json.dumps(best.params, indent=2, sort_keys=True))
    benchmark_params = {}
    for key, value in best.user_attrs.items():
        parts = key.split("/")
        if len(parts) == 3 and parts[1] == "param":
            benchmark_params.setdefault(parts[0], {})[parts[2]] = value
    if benchmark_params:
        print("best_benchmark_params=" + json.dumps(benchmark_params, indent=2, sort_keys=True))

    top = sorted(complete, key=lambda t: t.value if t.value is not None else math.inf)[:5]
    print("top_trials=")
    for trial in top:
        print(
            f"  #{trial.number} value={trial.value:.6f} "
            f"params={json.dumps(trial.params, sort_keys=True)}"
        )


def _make_sampler(seed: int):
    try:
        return optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    except TypeError:
        return optuna.samplers.TPESampler(seed=seed)


def _completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def _best_direct_params(study: optuna.Study) -> Optional[dict[str, Any]]:
    if not _completed_trials(study):
        return None
    params = study.best_trial.params
    if not {"iterations", "density_weight", "target_density"}.issubset(params):
        return None
    return dict(params)


def _local_candidates(
    best_params: dict[str, Any],
    args: argparse.Namespace,
    limit: int,
) -> list[dict[str, Any]]:
    best_iterations = int(best_params["iterations"])
    best_density_weight = float(best_params["density_weight"])
    best_target_density = float(best_params["target_density"])
    candidates = []

    def add(iterations, density_weight, target_density) -> None:
        candidate = {
            "iterations": _round_iterations(iterations, args),
            "density_weight": _clip(
                density_weight,
                args.min_density_weight,
                args.max_density_weight,
            ),
            "target_density": _clip(
                target_density,
                args.min_target_density,
                args.max_target_density,
            ),
        }
        if candidate != best_params and candidate not in candidates:
            candidates.append(candidate)

    for offset in (-3, -2, -1, 1, 2, 3):
        add(best_iterations + offset * args.iteration_step, best_density_weight, best_target_density)
    for factor in (0.5, 0.7, 0.85, 1.15, 1.4, 2.0):
        add(best_iterations, best_density_weight * factor, best_target_density)
    for offset in (-0.08, -0.04, -0.02, 0.02, 0.04, 0.08):
        add(best_iterations, best_density_weight, best_target_density + offset)
    for factor in (0.75, 1.25):
        for offset in (-0.04, 0.04):
            add(best_iterations, best_density_weight * factor, best_target_density + offset)

    return candidates[:limit]


def _run_direct_study(
    case: dict[str, Any],
    args: argparse.Namespace,
    gpu: int,
) -> optuna.Study:
    study_name = f"{args.study_name}_{case['name']}"
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="minimize",
        sampler=_make_sampler(args.seed),
        load_if_exists=True,
    )
    study.set_user_attr("benchmark", case["name"])
    study.set_user_attr("mode", "direct")
    study.set_user_attr("gpu", gpu)

    coarse_trials = args.n_trials
    refine_trials = 0
    if args.n_trials >= 4:
        coarse_trials = max(1, int(math.ceil(args.n_trials * COARSE_TRIAL_FRACTION)))
        refine_trials = args.n_trials - coarse_trials

    print(f"\n[{case['name']}] broad search: {coarse_trials} trial(s)")
    study.optimize(
        make_direct_objective(case, args, gpu),
        n_trials=coarse_trials,
        timeout=args.study_timeout,
        gc_after_trial=True,
    )

    best_params = _best_direct_params(study)
    if refine_trials > 0 and best_params is not None:
        candidates = _local_candidates(best_params, args, refine_trials)
        for candidate in candidates:
            study.enqueue_trial(candidate)
        print(f"[{case['name']}] local refinement: {refine_trials} trial(s)")
        study.optimize(
            make_direct_objective(case, args, gpu),
            n_trials=refine_trials,
            timeout=args.study_timeout,
            gc_after_trial=True,
        )

    print(f"[{case['name']}] best")
    _print_best(study)
    return study


def _run_feature_study(
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    gpu: int,
) -> optuna.Study:
    feature_keys = _active_feature_keys(cases)
    feature_stats = cases[0]["feature_stats"]
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, len(cases)))
    study = optuna.create_study(
        study_name=f"{args.study_name}_feature_model",
        storage=args.storage,
        direction="minimize",
        sampler=_make_sampler(args.seed),
        pruner=pruner,
        load_if_exists=True,
    )
    study.set_user_attr("benchmarks", args.benchmarks)
    study.set_user_attr("aggregate", args.aggregate)
    study.set_user_attr("feature_keys", list(feature_keys))
    study.set_user_attr("feature_stats", feature_stats)
    study.set_user_attr("mode", "feature_model")
    study.set_user_attr("gpu", gpu)

    print(
        "Running feature-model Optuna tuning on "
        f"{', '.join(args.benchmarks)} with gpu={gpu}; "
        f"feature_keys={list(feature_keys)}"
    )
    study.optimize(
        make_feature_objective(cases, args, gpu, feature_keys),
        n_trials=args.n_trials,
        timeout=args.study_timeout,
        gc_after_trial=True,
    )
    _print_best(study)
    return study


def main() -> None:
    args = _parse_args()

    cases = _load_cases(args.benchmarks)
    gpu = _detect_gpu()

    probe = DreamPlaceAdapter(
        _make_config(
            {
                "iterations": args.min_iterations,
                "density_weight": args.min_density_weight,
                "target_density": args.min_target_density,
            },
            gpu,
        )
    )
    probe.validate_environment()

    if args.feature_model:
        _run_feature_study(cases, args, gpu)
        return

    best_by_benchmark = {}
    for case in cases:
        study = _run_direct_study(case, args, gpu)
        if _completed_trials(study):
            best_by_benchmark[case["name"]] = {
                "proxy_cost": study.best_value,
                "params": study.best_trial.params,
            }

    print("\nbest_by_benchmark=" + json.dumps(best_by_benchmark, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
