"""Run a process-parallel, cumulative Monte Carlo exact-K sweep."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import pandas as pd

from solver.baselines import solve_aco, solve_ga, solve_nn, solve_pso
from solver.model import load_table_i_instance, prepare_fast_cost_oracle
from solver.proposed import ProposedConfig, solve_proposed
from solver.runner import _result_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("proposed", "nn", "aco", "pso", "ga")


def _solve(method: str, instance, solver_seed: int,
           args: argparse.Namespace):
    if method == "proposed":
        config = ProposedConfig(
            max_rounds=args.max_rounds,
            assignment_rounds=args.assignment_rounds,
            damping=args.damping,
            tolerance=args.tolerance,
            improvement_tolerance=args.improvement_tolerance,
            exact_global_limit=args.exact_global_limit,
            maximum_exact_trellis_bins=args.maximum_exact_trellis_bins,
            hypercube_radii=tuple(args.hypercube_radii),
            stagnation_patience=args.stagnation_patience,
            canonicalize_vehicle_symmetry=args.canonicalize_vehicle_symmetry,
            symmetry_dual_path=args.symmetry_dual_path,
            vehicle_gauss_seidel=args.vehicle_gauss_seidel,
            certified_route_messages=args.certified_route_messages,
            global_assignment_trajectory=args.global_assignment_trajectory,
        )
        return solve_proposed(instance, config), config
    if method == "nn":
        return solve_nn(instance), None
    if method == "aco":
        return solve_aco(
            instance, solver_seed, ants=args.aco_ants,
            iterations=args.aco_iterations), None
    if method == "pso":
        return solve_pso(
            instance, solver_seed, particles=args.pso_particles,
            iterations=args.pso_iterations), None
    return solve_ga(
        instance, solver_seed, population=args.ga_population,
        generations=args.ga_generations), None


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_id(method: str, n_bins: int, vehicles: int, start_hour: int,
            demand_seed: int) -> str:
    return (
        f"{method}_n{n_bins}_k{vehicles}_h{start_hour:02d}_s{demand_seed}"
    )


def planned_instances(
    base_seeds: list[int],
    focus_seeds: list[int],
    hours: list[int],
    vehicles: list[int],
    focus_vehicles: list[int],
) -> list[tuple[int, int, int]]:
    """Return unique (seed, hour, K) tasks in reproducible priority order."""

    if len(set(base_seeds)) != len(base_seeds):
        raise ValueError("base_seeds must be unique")
    if len(set(focus_seeds)) != len(focus_seeds):
        raise ValueError("focus_seeds must be unique")
    overlap = set(base_seeds) & set(focus_seeds)
    if overlap:
        raise ValueError(f"base and focus seeds overlap: {sorted(overlap)}")
    missing_focus = set(focus_vehicles) - set(vehicles)
    if missing_focus:
        raise ValueError(
            f"focus vehicles are outside the K sweep: {sorted(missing_focus)}")
    tasks = [
        (int(seed), int(hour), int(vehicle))
        for seed in base_seeds
        for hour in hours
        for vehicle in vehicles
    ]
    tasks.extend(
        (int(seed), int(hour), int(vehicle))
        for seed in focus_seeds
        for hour in hours
        for vehicle in focus_vehicles
    )
    return tasks


def _load_shard(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def _instance_worker(task: tuple) -> list[dict]:
    demand_seed, start_hour, vehicles, shard_path, route_dir, args = task
    demand_seed = int(demand_seed)
    start_hour = int(start_hour)
    vehicles = int(vehicles)
    shard_path = Path(shard_path)
    route_dir = Path(route_dir)
    run_ids = {
        method: _run_id(
            method, args.n_bins, vehicles, start_hour, demand_seed)
        for method in args.methods
    }
    rows = _load_shard(shard_path) if args.resume else []
    completed = {str(row["run_id"]) for row in rows}
    pending = [
        method for method in args.methods if run_ids[method] not in completed
    ]
    if not pending:
        return rows

    instance = load_table_i_instance(
        args.n_bins, vehicles, start_hour, demand_seed,
        target_total_demand_kg=args.target_total_demand_kg)
    oracle_prep_s = prepare_fast_cost_oracle(
        instance, adjacent_slots=args.oracle_slots)
    solver_seed = demand_seed
    for method in pending:
        wall_started = time.perf_counter()
        result, proposed_config = _solve(
            method, instance, solver_seed, args)
        if result.evaluation.makespan_s >= 3600.0 * args.oracle_slots:
            raise RuntimeError(
                f"{run_ids[method]} crosses beyond the prepared traffic "
                f"slots; increase --oracle-slots")
        row = _result_record(
            result, n_bins=args.n_bins, vehicles=vehicles,
            start_hour=start_hour, seed=demand_seed,
            oracle_prep_s=oracle_prep_s)
        row.update({
            "run_id": run_ids[method],
            "demand_seed": demand_seed,
            "solver_seed": solver_seed,
            "total_demand_kg": float(instance.demand_kg.sum()),
            "target_total_demand_kg": args.target_total_demand_kg,
            "total_runtime_s": oracle_prep_s + result.runtime_s,
            "wall_runtime_s": time.perf_counter() - wall_started,
        })
        rows.append(row)
        route_payload = {
            "run": row,
            "routes": result.routes,
            "vehicle_energy_kwh": result.evaluation.vehicle_energy_kwh,
            "vehicle_battery_remaining_kwh": [
                instance.vehicle_states[vehicle].battery_kwh - energy
                for vehicle, energy in enumerate(
                    result.evaluation.vehicle_energy_kwh)
            ],
            "vehicle_time_s": result.evaluation.vehicle_time_s,
            "metadata": getattr(result, "metadata", {}),
        }
        if proposed_config is not None:
            route_payload.update({
                "config": asdict(proposed_config),
                "scopes": result.scopes,
                "full_assignment_graph": result.full_assignment_graph,
                "route_message_mode": result.route_message_mode,
                "diagnostics": result.diagnostics,
            })
        _atomic_json(route_payload, route_dir / f"{row['run_id']}.json")
        _atomic_csv(pd.DataFrame(rows), shard_path)
        print(
            f"DONE {row['run_id']}: E={row['energy_kwh']:.3f} kWh, "
            f"solver={row['runtime_s']:.2f} s, "
            f"oracle={oracle_prep_s:.2f} s, feasible={row['feasible']}",
            flush=True)
    return rows


def _compatible_bootstrap_rows(
    path: Path | None,
    args: argparse.Namespace,
    planned: set[tuple[int, int, int]],
) -> list[dict]:
    if path is None or not path.exists():
        return []
    frame = pd.read_csv(path)
    required = {
        "run_id", "method", "n_bins", "vehicles", "start_hour", "seed",
        "energy_kwh", "makespan_s", "feasible", "active_vehicles",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"bootstrap CSV is missing required columns {sorted(missing)}")
    frame = frame[
        frame["method"].isin(args.methods)
        & (frame["n_bins"].astype(int) == args.n_bins)
    ].copy()
    frame = frame[frame.apply(
        lambda row: (
            int(row["seed"]), int(row["start_hour"]), int(row["vehicles"])
        ) in planned,
        axis=1,
    )]
    if "target_total_demand_kg" in frame.columns:
        target = pd.to_numeric(
            frame["target_total_demand_kg"], errors="coerce")
        frame = frame[
            target.sub(args.target_total_demand_kg).abs().le(1e-6)]
    frame["demand_seed"] = frame["seed"].astype(int)
    frame["solver_seed"] = frame["seed"].astype(int)
    if "total_runtime_s" not in frame.columns:
        frame["total_runtime_s"] = (
            frame["runtime_s"].astype(float)
            + frame["oracle_prep_s"].astype(float))
    if "wall_runtime_s" not in frame.columns:
        frame["wall_runtime_s"] = frame["runtime_s"].astype(float)
    return frame.to_dict("records")


def _all_rows(output: Path, shard_dir: Path,
              bootstrap_rows: list[dict]) -> list[dict]:
    rows = list(bootstrap_rows)
    if output.exists():
        rows.extend(pd.read_csv(output).to_dict("records"))
    for shard in sorted(shard_dir.glob("*.csv")):
        rows.extend(_load_shard(shard))
    return list({str(row["run_id"]): row for row in rows}.values())


def _write_outputs(rows: list[dict], output: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = (frame.drop_duplicates("run_id", keep="last")
             .sort_values(
                 ["vehicles", "demand_seed", "start_hour", "method"],
                 kind="stable"))
    _atomic_csv(frame, output)
    summary = (frame.groupby(["method", "vehicles"], as_index=False)
               .agg(
                   observations=("run_id", "count"),
                   demand_seeds=("demand_seed", "nunique"),
                   hours=("start_hour", "nunique"),
                   energy_mean_kwh=("energy_kwh", "mean"),
                   energy_std_kwh=("energy_kwh", "std"),
                   makespan_mean_s=("makespan_s", "mean"),
                   makespan_std_s=("makespan_s", "std"),
                   runtime_mean_s=("runtime_s", "mean"),
                   feasible_rate=("feasible", "mean")))
    _atomic_csv(summary, output.with_name(f"{output.stem}_summary.csv"))
    return frame


def _completed_instance_count(
    instances: list[tuple[int, int, int]],
    methods: list[str],
    n_bins: int,
    completed: set[str],
) -> int:
    return sum(
        all(
            _run_id(method, n_bins, vehicles, hour, seed) in completed
            for method in methods)
        for seed, hour, vehicles in instances
    )


def _update_figures(
    output: Path,
    figure_dir: Path,
    vehicles: list[int],
    confidence_level: float,
    bootstrap_draws: int,
    completed_instances: int,
    expected_instances: int,
) -> None:
    from .plotting.k_sweep_multiseed import create_figures

    paths = create_figures(
        output, figure_dir, vehicles=tuple(vehicles),
        confidence_level=confidence_level,
        bootstrap_draws=bootstrap_draws)
    print(
        f"FIGURES UPDATED ({completed_instances}/{expected_instances} "
        f"instances): " + ", ".join(path.name for path in paths),
        flush=True)


def run(args: argparse.Namespace) -> pd.DataFrame:
    if args.parallel_jobs < 1:
        raise ValueError("parallel-jobs must be at least one")
    output = args.output.resolve()
    figure_dir = args.figure_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output.parent / "shards"
    route_dir = output.parent / "routes"
    shard_dir.mkdir(parents=True, exist_ok=True)
    route_dir.mkdir(parents=True, exist_ok=True)

    instances = planned_instances(
        args.base_seeds, args.focus_seeds, args.hours,
        args.vehicles, args.focus_vehicles)
    planned = set(instances)
    expected = {
        _run_id(method, args.n_bins, vehicles, hour, seed)
        for seed, hour, vehicles in instances
        for method in args.methods
    }
    bootstrap_rows = _compatible_bootstrap_rows(
        args.bootstrap_csv.resolve() if args.bootstrap_csv else None,
        args, planned)
    existing_rows = _all_rows(
        output, shard_dir, bootstrap_rows) if args.resume else bootstrap_rows
    completed = {str(row["run_id"]) for row in existing_rows}
    if existing_rows:
        _write_outputs(existing_rows, output)
        completed_instances = _completed_instance_count(
            instances, args.methods, args.n_bins, completed)
        _update_figures(
            output, figure_dir, args.vehicles,
            args.confidence_level, args.bootstrap_draws,
            completed_instances, len(instances))
        _atomic_json({
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed_instances": completed_instances,
            "expected_instances": len(instances),
            "completed_runs": len(expected & completed),
            "expected_runs": len(expected),
        }, output.parent / "progress.json")

    tasks = []
    for demand_seed, start_hour, vehicles in instances:
        instance_ids = {
            _run_id(
                method, args.n_bins, vehicles, start_hour, demand_seed)
            for method in args.methods
        }
        if instance_ids <= completed:
            print(
                f"SKIP seed={demand_seed}, hour={start_hour:02d}, "
                f"K={vehicles} (all methods complete)", flush=True)
            continue
        shard = shard_dir / (
            f"s{demand_seed}_h{start_hour:02d}_k{vehicles:02d}.csv")
        tasks.append((
            demand_seed, start_hour, vehicles,
            str(shard), str(route_dir), args))

    def incorporate(result_rows: list[dict]) -> None:
        nonlocal existing_rows, completed
        by_id = {
            str(row["run_id"]): row
            for row in _all_rows(output, shard_dir, bootstrap_rows)
        }
        by_id.update({str(row["run_id"]): row for row in result_rows})
        existing_rows = list(by_id.values())
        completed = set(by_id)
        _write_outputs(existing_rows, output)
        completed_instances = _completed_instance_count(
            instances, args.methods, args.n_bins, completed)
        _update_figures(
            output, figure_dir, args.vehicles,
            args.confidence_level, args.bootstrap_draws,
            completed_instances, len(instances))
        _atomic_json({
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed_instances": completed_instances,
            "expected_instances": len(instances),
            "completed_runs": len(expected & completed),
            "expected_runs": len(expected),
        }, output.parent / "progress.json")

    if args.parallel_jobs == 1:
        for task in tasks:
            incorporate(_instance_worker(task))
    else:
        print(
            f"PARALLEL execution with {args.parallel_jobs} worker processes; "
            "each completed (seed, hour, K) process refreshes all figures",
            flush=True)
        with ProcessPoolExecutor(max_workers=args.parallel_jobs) as executor:
            futures = [executor.submit(_instance_worker, task) for task in tasks]
            for future in as_completed(futures):
                incorporate(future.result())

    frame = _write_outputs(
        _all_rows(output, shard_dir, bootstrap_rows), output)
    completed = set(frame["run_id"].astype(str)) if not frame.empty else set()
    manifest = {
        "n_bins": args.n_bins,
        "target_total_demand_kg": args.target_total_demand_kg,
        "hours": args.hours,
        "vehicles": args.vehicles,
        "base_seeds": args.base_seeds,
        "focus_seeds": args.focus_seeds,
        "focus_vehicles": args.focus_vehicles,
        "methods": args.methods,
        "parallel_jobs": args.parallel_jobs,
        "confidence_level": args.confidence_level,
        "bootstrap_draws": args.bootstrap_draws,
        "expected_instances": len(instances),
        "expected_runs": len(expected),
        "completed_runs": len(expected & completed),
        "complete": expected <= completed,
        "bootstrap_csv": (
            str(args.bootstrap_csv.resolve()) if args.bootstrap_csv else None),
    }
    _atomic_json(manifest, output.parent / "experiment_manifest.json")
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bins", type=int, default=84)
    parser.add_argument("--target-total-demand-kg", type=float, required=True)
    parser.add_argument("--hours", type=int, nargs="+", required=True)
    parser.add_argument("--vehicles", type=int, nargs="+", required=True)
    parser.add_argument("--base-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--focus-seeds", type=int, nargs="+", default=[])
    parser.add_argument("--focus-vehicles", type=int, nargs="+", default=[])
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--ga-population", type=int, default=100)
    parser.add_argument("--ga-generations", type=int, default=500)
    parser.add_argument("--pso-particles", type=int, default=60)
    parser.add_argument("--pso-iterations", type=int, default=200)
    parser.add_argument("--aco-ants", type=int, default=60)
    parser.add_argument("--aco-iterations", type=int, default=150)
    parser.add_argument("--max-rounds", type=int, default=32)
    parser.add_argument("--assignment-rounds", type=int, default=24)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--improvement-tolerance", type=float, default=1e-9)
    parser.add_argument("--exact-global-limit", type=int, default=14)
    parser.add_argument("--maximum-exact-trellis-bins", type=int, default=84)
    parser.add_argument(
        "--hypercube-radii", type=int, nargs="+",
        default=[1, 2, 4, 8, 12, 16, 24, 32])
    parser.add_argument("--stagnation-patience", type=int, default=8)
    parser.add_argument(
        "--canonicalize-vehicle-symmetry",
        dest="canonicalize_vehicle_symmetry", action="store_true",
        default=True)
    parser.add_argument(
        "--no-canonicalize-vehicle-symmetry",
        dest="canonicalize_vehicle_symmetry", action="store_false")
    parser.add_argument(
        "--symmetry-dual-path", dest="symmetry_dual_path",
        action="store_true", default=True)
    parser.add_argument(
        "--no-symmetry-dual-path", dest="symmetry_dual_path",
        action="store_false")
    parser.add_argument("--vehicle-gauss-seidel", action="store_true")
    parser.add_argument("--certified-route-messages", action="store_true")
    parser.add_argument("--global-assignment-trajectory", action="store_true")
    parser.add_argument("--oracle-slots", type=int, default=3)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-csv", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results/k_sweep/monte_carlo/k_sweep_mc_runs.csv")
    parser.add_argument(
        "--figure-dir", type=Path,
        default=PROJECT_ROOT / "figures/k_sweep")
    return parser


def main(arguments: list[str] | None = None) -> None:
    run(build_parser().parse_args(arguments))


if __name__ == "__main__":
    main()
