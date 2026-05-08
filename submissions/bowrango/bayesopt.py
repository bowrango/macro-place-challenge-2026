"""Optuna tuning for baseline DREAMPlace hyperparameters.

Examples:
    uv run python submissions/bowrango/bayesopt.py -b ibm01 --n-trials 20
    uv run python submissions/bowrango/bayesopt.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402

from dreamplace_adapter import DreamPlaceAdapter, DreamPlaceConfig  # noqa: E402


DEFAULT_STORAGE = f"sqlite:///{HERE / 'bayesopt.db'}"
DEFAULT_RESULTS = HERE / "bayesopt_results.json"
ICCAD04_ROOT = REPO_ROOT / "external/MacroPlacement/Testcases/ICCAD04"

GPU = 1
N_TRIALS = 20
SEED = 1000
STUDY_TIMEOUT = None

MIN_ITERATIONS = 200
MAX_ITERATIONS = 3000
ITERATION_STEP = 100
MIN_DENSITY_WEIGHT = 1e-5
MAX_DENSITY_WEIGHT = 2e-4
MIN_TARGET_DENSITY = 0.50
MAX_TARGET_DENSITY = 1.0


def _discover_benchmarks(root: Path = ICCAD04_ROOT) -> list[str]:
    return sorted(p.parent.name for p in root.glob("*/netlist.pb.txt"))


def _load_cases(names: list[str]) -> list[dict[str, Any]]:
    cases = []
    for name in names:
        benchmark, plc = load_benchmark_from_dir(str(ICCAD04_ROOT / name))
        cases.append({"name": name, "benchmark": benchmark, "plc": plc})
    return cases


def _sample_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int(
            "iterations",
            MIN_ITERATIONS,
            MAX_ITERATIONS,
            step=ITERATION_STEP,
        ),
        "density_weight": trial.suggest_float(
            "density_weight",
            MIN_DENSITY_WEIGHT,
            MAX_DENSITY_WEIGHT,
            log=True,
        ),
        "target_density": trial.suggest_float(
            "target_density",
            MIN_TARGET_DENSITY,
            MAX_TARGET_DENSITY,
        ),
    }


def _make_config(params: dict[str, Any], run_id: str) -> DreamPlaceConfig:
    return DreamPlaceConfig(gpu=GPU, keep_work=False, run_id=run_id, **params)


def _record_trial(
    trial: optuna.Trial,
    params: dict[str, Any],
    costs: dict[str, Any],
    elapsed: float,
) -> None:
    trial.set_user_attr("params", params)
    trial.set_user_attr("proxy_cost", float(costs["proxy_cost"]))
    trial.set_user_attr("wirelength_cost", float(costs["wirelength_cost"]))
    trial.set_user_attr("density_cost", float(costs["density_cost"]))
    trial.set_user_attr("congestion_cost", float(costs["congestion_cost"]))
    trial.set_user_attr("overlap_count", int(costs["overlap_count"]))
    trial.set_user_attr("seconds", elapsed)


def make_objective(case: dict[str, Any]):
    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial)
        placer = DreamPlaceAdapter(_make_config(params, f"trial_{trial.number}"))

        start = time.time()
        placement = placer.place(case["benchmark"])
        costs = compute_proxy_cost(placement, case["benchmark"], case["plc"])
        elapsed = time.time() - start

        _record_trial(trial, params, costs, elapsed)
        return float(costs["proxy_cost"])

    return objective


def _make_sampler() -> optuna.samplers.BaseSampler:
    try:
        return optuna.samplers.TPESampler(
            seed=SEED,
            multivariate=True,
            group=True,
        )
    except TypeError:
        return optuna.samplers.TPESampler(seed=SEED)


def _completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def _study_summary(study: optuna.Study) -> dict[str, Any]:
    complete = _completed_trials(study)
    summary: dict[str, Any] = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "complete_trials": len(complete),
    }
    if complete:
        best = study.best_trial
        summary.update(
            {
                "best_trial": best.number,
                "best_proxy_cost": best.value,
                "best_params": best.params,
                "best_costs": {
                    key: best.user_attrs[key]
                    for key in (
                        "wirelength_cost",
                        "density_cost",
                        "congestion_cost",
                        "overlap_count",
                    )
                    if key in best.user_attrs
                },
                "best_seconds": best.user_attrs.get("seconds"),
            }
        )
    return summary


def _write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _run_direct_study(case: dict[str, Any], args: argparse.Namespace) -> optuna.Study:
    study = optuna.create_study(
        study_name=f"{args.study_name}_{case['name']}",
        storage=args.storage,
        direction="minimize",
        sampler=_make_sampler(),
        load_if_exists=True,
    )
    study.set_user_attr("benchmark", case["name"])
    study.set_user_attr("mode", "direct")
    study.set_user_attr("gpu", GPU)

    study.optimize(
        make_objective(case),
        n_trials=args.n_trials,
        timeout=STUDY_TIMEOUT,
        gc_after_trial=True,
    )
    return study


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune DREAMPlace iterations, density_weight, and target_density."
    )
    parser.add_argument("--benchmark", "-b", action="append", dest="benchmarks")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--study-name", default="dreamplace_bayesopt")
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    if args.all:
        args.benchmarks = _discover_benchmarks()
    elif not args.benchmarks:
        args.benchmarks = ["ibm01"]
    return args


def main() -> None:
    args = _parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    studies = {}
    for case in _load_cases(args.benchmarks):
        study = _run_direct_study(case, args)
        studies[case["name"]] = _study_summary(study)
        _write_results(
            args.results,
            {
                "mode": "direct",
                "benchmarks": args.benchmarks,
                "storage": args.storage,
                "gpu": GPU,
                "search_space": {
                    "iterations": [MIN_ITERATIONS, MAX_ITERATIONS, ITERATION_STEP],
                    "density_weight": [MIN_DENSITY_WEIGHT, MAX_DENSITY_WEIGHT],
                    "target_density": [MIN_TARGET_DENSITY, MAX_TARGET_DENSITY],
                },
                "studies": studies,
            },
        )

    print(f"stored bayesopt results: {args.results}")


if __name__ == "__main__":
    main()
