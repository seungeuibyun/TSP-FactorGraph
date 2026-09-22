"""Run a multi-layout N/K scalability sweep with process-level parallelism."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from solver.baselines import solve_aco, solve_ga, solve_nn, solve_pso
from solver.model import load_table_i_instance, prepare_fast_cost_oracle
from solver.proposed import ProposedConfig, solve_proposed
from solver.runner import _result_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("proposed", "nn", "aco", "pso", "ga")


def vehicles_for_bins(n_bins: int, bins_per_vehicle: float) -> int:
    """Nearest integer K, with half values rounded upward."""

    if n_bins < 1 or bins_per_vehicle <= 0.0:
        raise ValueError("n_bins and bins_per_vehicle must be positive")
    return max(1, int(math.floor(n_bins / bins_per_vehicle + 0.5)))


def _solve(method: str, instance, seed: int, args: argparse.Namespace):
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
            instance, seed, ants=args.aco_ants,
            iterations=args.aco_iterations), None
    if method == "pso":
        return solve_pso(
            instance, seed, particles=args.pso_particles,
            iterations=args.pso_iterations), None
    return solve_ga(
        instance, seed, population=args.ga_population,
        generations=args.ga_generations), None


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_outputs(rows: list[dict], output: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    sort_columns = [
        name for name in
        ("n_bins", "spatial_seed", "demand_seed", "method")
        if name in frame.columns
    ]
    frame = (frame.drop_duplicates("run_id", keep="last")
             .sort_values(sort_columns, kind="stable"))
    _atomic_csv(frame, output)
    summary = (frame.groupby(
        ["method", "n_bins", "vehicles", "assignment_edges"],
        as_index=False)
        .agg(
            runs=("run_id", "count"),
            solver_runtime_mean_s=("runtime_s", "mean"),
            solver_runtime_std_s=("runtime_s", "std"),
            oracle_runtime_mean_s=("oracle_prep_s", "mean"),
            total_runtime_mean_s=("total_runtime_s", "mean"),
            total_runtime_std_s=("total_runtime_s", "std"),
            energy_mean_kwh=("energy_kwh", "mean"),
            energy_std_kwh=("energy_kwh", "std"),
            energy_per_bin_mean_kwh=("energy_per_bin_kwh", "mean"),
            energy_per_bin_std_kwh=("energy_per_bin_kwh", "std"),
            makespan_mean_s=("makespan_s", "mean"),
            makespan_std_s=("makespan_s", "std"),
            feasible_rate=("feasible", "mean"),
            convergence_rate=("converged", "mean"),
            rounds_mean=("rounds", "mean"),
            maximum_route_bins=("maximum_route_bins", "max"),
        ))
    _atomic_csv(summary, output.with_name(f"{output.stem}_summary.csv"))
    return frame


def _run_id(method: str, n_bins: int, vehicles: int, start_hour: int,
            spatial_seed: int, demand_seed: int) -> str:
    return (
        f"{method}_n{n_bins}_k{vehicles}_h{start_hour:02d}_"
        f"sp{spatial_seed}_ds{demand_seed}"
    )


def _solver_seed(spatial_seed: int, demand_seed: int) -> int:
    """Give stochastic solvers a distinct reproducible seed per layout."""

    return int((spatial_seed * 1_000_003 + demand_seed) % (2**31 - 1))


def _prepare_layouts(source: Path, layout_dir: Path,
                     spatial_seeds: list[int], maximum_size: int) -> dict[int, Path]:
    source_frame = pd.read_csv(
        source, dtype={"SUMO_EDGE_ID": str}, encoding="utf-8-sig")
    feasible = source_frame[
        source_frame["ROUND_TRIP_FEASIBLE"].astype(str).str.lower() == "true"
    ].reset_index(drop=True)
    if len(feasible) < maximum_size:
        raise ValueError(
            f"requested up to {maximum_size} bins but only {len(feasible)} "
            "round-trip-feasible candidates are available")
    layout_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[int, Path] = {}
    for spatial_seed in spatial_seeds:
        path = layout_dir / f"layout_spatial_seed_{spatial_seed}.csv"
        order = np.random.default_rng(spatial_seed).permutation(len(feasible))
        layout = feasible.iloc[order].copy().reset_index(drop=True)
        layout.insert(0, "LAYOUT_RANK", np.arange(len(layout), dtype=int))
        layout["SPATIAL_SEED"] = int(spatial_seed)
        _atomic_csv(layout, path)
        outputs[int(spatial_seed)] = path
    return outputs


def _load_shard(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def _instance_worker(task: tuple) -> list[dict]:
    (n_bins, spatial_seed, demand_seed, layout_csv, shard_path,
     route_dir, args) = task
    n_bins = int(n_bins)
    spatial_seed = int(spatial_seed)
    demand_seed = int(demand_seed)
    layout_csv = Path(layout_csv)
    shard_path = Path(shard_path)
    route_dir = Path(route_dir)
    vehicles = vehicles_for_bins(n_bins, args.bins_per_vehicle)
    run_ids = {
        method: _run_id(
            method, n_bins, vehicles, args.start_hour,
            spatial_seed, demand_seed)
        for method in args.methods
    }
    rows = _load_shard(shard_path) if args.resume else []
    completed = {str(row["run_id"]) for row in rows}
    pending = [method for method in args.methods
               if run_ids[method] not in completed]
    if not pending:
        return rows

    instance = load_table_i_instance(
        n_bins, vehicles, args.start_hour, demand_seed,
        bins_csv=layout_csv)
    oracle_prep_s = prepare_fast_cost_oracle(
        instance, adjacent_slots=args.oracle_slots)
    solver_seed = _solver_seed(spatial_seed, demand_seed)
    for method in pending:
        started = time.perf_counter()
        result, proposed_config = _solve(
            method, instance, solver_seed, args)
        row = _result_record(
            result, n_bins=n_bins, vehicles=vehicles,
            start_hour=args.start_hour, seed=demand_seed,
            oracle_prep_s=oracle_prep_s)
        row.update({
            "run_id": run_ids[method],
            "spatial_seed": spatial_seed,
            "demand_seed": demand_seed,
            "solver_seed": solver_seed,
            "layout_id": f"spatial_seed_{spatial_seed}",
            "bins_per_vehicle_target": args.bins_per_vehicle,
            "bins_per_vehicle": n_bins / vehicles,
            "assignment_edges": n_bins * vehicles,
            "total_runtime_s": oracle_prep_s + result.runtime_s,
            "wall_runtime_s": time.perf_counter() - started,
            "energy_per_bin_kwh": result.evaluation.energy_kwh / n_bins,
            "total_demand_kg": float(instance.demand_kg.sum()),
            "minimum_route_bins": min(map(len, result.routes)),
            "maximum_route_bins": max(map(len, result.routes)),
        })
        rows.append(row)
        route_payload = {
            "run": row,
            "layout_csv": str(layout_csv),
            "routes": result.routes,
            "vehicle_energy_kwh": result.evaluation.vehicle_energy_kwh,
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
        (route_dir / f"{row['run_id']}.json").write_text(
            json.dumps(route_payload, indent=2, ensure_ascii=False),
            encoding="utf-8")
        _atomic_csv(pd.DataFrame(rows), shard_path)
        print(
            f"DONE {row['run_id']}: E={row['energy_kwh']:.3f} kWh, "
            f"solver={row['runtime_s']:.2f} s, "
            f"oracle={oracle_prep_s:.2f} s, "
            f"feasible={row['feasible']}", flush=True)
    return rows


def _all_rows(output: Path, shard_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if output.exists():
        rows.extend(pd.read_csv(output).to_dict("records"))
    for shard in sorted(shard_dir.glob("*.csv")):
        rows.extend(_load_shard(shard))
    return list({str(row["run_id"]): row for row in rows}.values())


def _update_figures(output: Path, figure_dir: Path, n_bins: int,
                    completed_instances: int, total_instances: int) -> None:
    """Refresh cumulative plots while allowing numerical work to continue."""

    try:
        from .plotting.scalability import create_figures

        paths = create_figures(output, figure_dir)
        names = ", ".join(path.name for path in paths)
        print(
            f"FIGURES UPDATED after N={n_bins} "
            f"({completed_instances}/{total_instances} instances complete): "
            f"{names}", flush=True)
    except Exception as exc:
        print(
            f"WARNING: figure update failed after N={n_bins}: "
            f"{type(exc).__name__}: {exc}", flush=True)


def run(args: argparse.Namespace) -> pd.DataFrame:
    output = args.output.resolve()
    figure_dir = args.figure_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    route_dir = output.parent / "routes"
    shard_dir = output.parent / "shards"
    layout_dir = output.parent / "layouts"
    route_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    bins_csv = args.bins_csv.resolve()
    layouts = _prepare_layouts(
        bins_csv, layout_dir, args.spatial_seeds, max(args.sizes))
    mapping = pd.DataFrame([{
        "n_bins": n_bins,
        "vehicles": vehicles_for_bins(n_bins, args.bins_per_vehicle),
        "bins_per_vehicle": n_bins / vehicles_for_bins(
            n_bins, args.bins_per_vehicle),
        "assignment_edges": n_bins * vehicles_for_bins(
            n_bins, args.bins_per_vehicle),
    } for n_bins in args.sizes])
    _atomic_csv(mapping, output.parent / "n_to_k_mapping.csv")

    expected = {
        _run_id(
            method, n_bins,
            vehicles_for_bins(n_bins, args.bins_per_vehicle),
            args.start_hour, spatial_seed, demand_seed)
        for n_bins in args.sizes
        for spatial_seed in args.spatial_seeds
        for demand_seed in args.demand_seeds
        for method in args.methods
    }
    existing_rows = _all_rows(output, shard_dir) if args.resume else []
    completed = {str(row["run_id"]) for row in existing_rows}
    tasks = []
    for n_bins in args.sizes:
        vehicles = vehicles_for_bins(n_bins, args.bins_per_vehicle)
        for spatial_seed in args.spatial_seeds:
            for demand_seed in args.demand_seeds:
                instance_ids = {
                    _run_id(method, n_bins, vehicles, args.start_hour,
                            spatial_seed, demand_seed)
                    for method in args.methods
                }
                if instance_ids <= completed:
                    print(
                        f"SKIP N={n_bins}, K={vehicles}, "
                        f"spatial={spatial_seed}, demand={demand_seed} "
                        "(all methods complete)", flush=True)
                    continue
                shard = shard_dir / (
                    f"n{n_bins}_sp{spatial_seed}_ds{demand_seed}.csv")
                tasks.append((
                    n_bins, spatial_seed, demand_seed,
                    str(layouts[spatial_seed]), str(shard),
                    str(route_dir), args))

    total_instances = len(args.spatial_seeds) * len(args.demand_seeds)

    def incorporate(result_rows: list[dict]) -> None:
        nonlocal existing_rows, completed
        by_id = {
            str(row["run_id"]): row
            for row in _all_rows(output, shard_dir)
        }
        by_id.update({str(row["run_id"]): row for row in result_rows})
        existing_rows = list(by_id.values())
        completed = set(by_id)
        _write_outputs(existing_rows, output)
        n_bins = int(result_rows[0]["n_bins"])
        vehicles = vehicles_for_bins(n_bins, args.bins_per_vehicle)
        complete_instances = sum(
            all(
                _run_id(method, n_bins, vehicles, args.start_hour,
                        spatial_seed, demand_seed) in completed
                for method in args.methods)
            for spatial_seed in args.spatial_seeds
            for demand_seed in args.demand_seeds
        )
        _update_figures(
            output, figure_dir, n_bins,
            complete_instances, total_instances)

    if args.parallel_jobs == 1:
        for task in tasks:
            incorporate(_instance_worker(task))
    else:
        print(
            f"PARALLEL execution with {args.parallel_jobs} worker processes; "
            "one oracle is shared by all methods inside each instance",
            flush=True)
        with ProcessPoolExecutor(max_workers=args.parallel_jobs) as executor:
            futures = [executor.submit(_instance_worker, task) for task in tasks]
            for future in as_completed(futures):
                incorporate(future.result())

    frame = _write_outputs(_all_rows(output, shard_dir), output)
    completed = set(frame["run_id"].astype(str)) if not frame.empty else set()
    manifest = {
        "sizes": args.sizes,
        "spatial_seeds": args.spatial_seeds,
        "demand_seeds": args.demand_seeds,
        "layouts_per_size": len(args.spatial_seeds),
        "instances_per_size": total_instances,
        "methods": args.methods,
        "start_hour": args.start_hour,
        "bins_per_vehicle_target": args.bins_per_vehicle,
        "vehicle_rounding": "nearest integer, half upward",
        "candidate_bins_csv": bins_csv.relative_to(PROJECT_ROOT).as_posix(),
        "layout_sampling": (
            "one deterministic permutation of the candidate pool per spatial "
            "seed; each N is the prefix of that layout"),
        "parallel_jobs": args.parallel_jobs,
        "expected_runs": len(expected),
        "completed_runs": len(expected & completed),
        "complete": expected <= completed,
    }
    (output.parent / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--bins-per-vehicle", type=float, default=8.4)
    parser.add_argument("--start-hour", type=int, default=11)
    parser.add_argument("--spatial-seeds", type=int, nargs="+", default=[107])
    parser.add_argument("--demand-seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--bins-csv", type=Path,
        default=PROJECT_ROOT / "simul/scalability_bins_500.csv")
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=(PROJECT_ROOT / "results/scalability/spatial_mc/"
                 "scalability_runs.csv"))
    parser.add_argument(
        "--figure-dir", type=Path,
        default=PROJECT_ROOT / "figures/scalability")
    return parser


def main(arguments: list[str] | None = None) -> None:
    args = build_parser().parse_args(arguments)
    if args.parallel_jobs < 1:
        raise ValueError("parallel-jobs must be at least one")
    run(args)


if __name__ == "__main__":
    main()
