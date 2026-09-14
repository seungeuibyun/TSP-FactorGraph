"""Run the N/K scalability sweep on a deterministic nested bin set."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

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
            canonicalize_vehicle_symmetry=(
                args.canonicalize_vehicle_symmetry),
            symmetry_dual_path=args.symmetry_dual_path,
            vehicle_gauss_seidel=args.vehicle_gauss_seidel,
            certified_route_messages=args.certified_route_messages,
            global_assignment_trajectory=(
                args.global_assignment_trajectory),
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


def _write_outputs(rows: list[dict], output: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values(
        ["n_bins", "seed", "method"], kind="stable")
    frame.to_csv(output, index=False)
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
    summary.to_csv(
        output.with_name(f"{output.stem}_summary.csv"), index=False)
    return frame


def _update_figures(output: Path, figure_dir: Path, n_bins: int, seed: int,
                    completed_seeds: int, total_seeds: int) -> None:
    """Refresh cumulative plots while allowing the numerical sweep to continue."""

    try:
        from .plotting.scalability import create_figures

        paths = create_figures(output, figure_dir)
        names = ", ".join(path.name for path in paths)
        print(
            f"FIGURES UPDATED after N={n_bins}, seed={seed} "
            f"({completed_seeds}/{total_seeds} seeds at this N): {names}",
            flush=True,
        )
    except Exception as exc:  # a plotting problem must not lose later N points
        print(
            f"WARNING: figure update failed after N={n_bins}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def run(args: argparse.Namespace) -> pd.DataFrame:
    output = args.output.resolve()
    figure_dir = args.figure_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    route_dir = output.parent / "routes"
    route_dir.mkdir(parents=True, exist_ok=True)
    bins_csv = args.bins_csv.resolve()
    if max(args.sizes) > len(pd.read_csv(
            bins_csv, usecols=["SUMO_EDGE_ID"])):
        raise ValueError("the scalability bin file is shorter than max(sizes)")

    mapping = pd.DataFrame([{
        "n_bins": n_bins,
        "vehicles": vehicles_for_bins(n_bins, args.bins_per_vehicle),
        "bins_per_vehicle": n_bins / vehicles_for_bins(
            n_bins, args.bins_per_vehicle),
        "assignment_edges": n_bins * vehicles_for_bins(
            n_bins, args.bins_per_vehicle),
    } for n_bins in args.sizes])
    mapping.to_csv(output.parent / "n_to_k_mapping.csv", index=False)

    rows: list[dict] = []
    if args.resume and output.exists():
        rows = pd.read_csv(output).to_dict("records")
        if args.global_assignment_trajectory:
            expected_mode = (
                "dual_labeled_canonical_global_assignment_boundary_sova"
                if args.symmetry_dual_path
                else "global_assignment_boundary_sova")
            stale = [row for row in rows
                     if (str(row.get("method")) == "proposed"
                         and str(row.get("route_message_mode"))
                         != expected_mode)]
            if stale:
                rows = [
                    row for row in rows
                    if not (str(row.get("method")) == "proposed"
                            and str(row.get("route_message_mode"))
                            != expected_mode)
                ]
                print(
                    f"INVALIDATE {len(stale)} proposed rows produced by a "
                    "different route-message mode; baseline rows are kept",
                    flush=True)
    completed = {str(row["run_id"]) for row in rows}

    for n_bins in args.sizes:
        vehicles = vehicles_for_bins(n_bins, args.bins_per_vehicle)
        for seed in args.seeds:
            run_ids = {
                method: (
                    f"{method}_n{n_bins}_k{vehicles}_h"
                    f"{args.start_hour:02d}_s{seed}")
                for method in args.methods
            }
            seed_run_ids = set(run_ids.values())
            seed_was_complete = seed_run_ids <= completed
            pending = [method for method in args.methods
                       if run_ids[method] not in completed]
            if not pending:
                print(
                    f"SKIP N={n_bins}, K={vehicles}, seed={seed} "
                    "(all methods complete)", flush=True)
                continue

            instance = load_table_i_instance(
                n_bins, vehicles, args.start_hour, seed,
                bins_csv=bins_csv)
            oracle_prep_s = prepare_fast_cost_oracle(
                instance, adjacent_slots=args.oracle_slots)
            for method in pending:
                started = time.perf_counter()
                result, proposed_config = _solve(
                    method, instance, seed, args)
                total_runtime_s = oracle_prep_s + result.runtime_s
                row = _result_record(
                    result, n_bins=n_bins, vehicles=vehicles,
                    start_hour=args.start_hour, seed=seed,
                    oracle_prep_s=oracle_prep_s)
                row.update({
                    "bins_per_vehicle_target": args.bins_per_vehicle,
                    "bins_per_vehicle": n_bins / vehicles,
                    "assignment_edges": n_bins * vehicles,
                    "total_runtime_s": total_runtime_s,
                    "wall_runtime_s": time.perf_counter() - started,
                    "energy_per_bin_kwh": (
                        result.evaluation.energy_kwh / n_bins),
                    "total_demand_kg": float(instance.demand_kg.sum()),
                    "minimum_route_bins": min(map(len, result.routes)),
                    "maximum_route_bins": max(map(len, result.routes)),
                })
                rows.append(row)
                completed.add(row["run_id"])
                route_payload = {
                    "run": row,
                    "routes": result.routes,
                    "vehicle_energy_kwh": (
                        result.evaluation.vehicle_energy_kwh),
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
                _write_outputs(rows, output)
                print(
                    f"DONE {row['run_id']}: E={row['energy_kwh']:.3f} kWh, "
                    f"solver={row['runtime_s']:.2f} s, "
                    f"oracle={oracle_prep_s:.2f} s, "
                    f"feasible={row['feasible']}", flush=True)
            if not seed_was_complete and seed_run_ids <= completed:
                completed_seed_count = sum(
                    all(
                        f"{method}_n{n_bins}_k{vehicles}_h"
                        f"{args.start_hour:02d}_s{candidate_seed}" in completed
                        for method in args.methods
                    )
                    for candidate_seed in args.seeds
                )
                _update_figures(
                    output, figure_dir, n_bins, seed,
                    completed_seed_count, len(args.seeds))

    frame = _write_outputs(rows, output)
    expected = {
        f"{method}_n{n_bins}_k"
        f"{vehicles_for_bins(n_bins, args.bins_per_vehicle)}_h"
        f"{args.start_hour:02d}_s{seed}"
        for n_bins in args.sizes for seed in args.seeds
        for method in args.methods
    }
    manifest = {
        "sizes": args.sizes,
        "seeds": args.seeds,
        "methods": args.methods,
        "start_hour": args.start_hour,
        "bins_per_vehicle_target": args.bins_per_vehicle,
        "vehicle_rounding": "nearest integer, half upward",
        "bins_csv": bins_csv.relative_to(PROJECT_ROOT).as_posix(),
        "expected_runs": len(expected),
        "completed_runs": len(expected & set(frame["run_id"].astype(str))),
        "complete": expected <= set(frame["run_id"].astype(str)),
    }
    (output.parent / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--bins-per-vehicle", type=float, default=8.4)
    parser.add_argument("--start-hour", type=int, default=11)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
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
        default=PROJECT_ROOT / "results/scalability/scalability_runs.csv")
    parser.add_argument(
        "--figure-dir", type=Path,
        default=PROJECT_ROOT / "figures/scalability",
        help="Overwrite cumulative EPS/PDF/PNG figures after each completed N.")
    return parser


def main(arguments: list[str] | None = None) -> None:
    run(build_parser().parse_args(arguments))


if __name__ == "__main__":
    main()
