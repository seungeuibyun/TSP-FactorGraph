"""Run the online controller from every hour on continuous daily traffic."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from . import online
from .plotting.hourly_online import create_figures


def _run_hour(hour: int, args: argparse.Namespace) -> pd.DataFrame:
    hour_dir = args.output_dir / f"h{hour:02d}"
    summary_path = hour_dir / "online_summary.csv"
    if args.resume and summary_path.exists():
        summary = pd.read_csv(summary_path)
        print(f"REUSE start hour {hour:02d}", flush=True)
    else:
        forwarded = online.build_parser().parse_args([
                "--n-bins", str(args.n_bins),
                "--vehicles", str(args.vehicles),
                "--start-hour", str(hour),
                "--max-simulation-minutes", str(args.max_simulation_minutes),
                "--traffic-step-s", str(args.traffic_step_s),
                "--replanning-interval-s", str(args.replanning_interval_s),
                "--traffic-model", "hourly_linear",
                "--demand-seed", str(args.demand_seed),
                "--solver-seed", str(args.solver_seed),
                "--methods", *args.methods,
                "--policies", "adaptive",
                "--oracle-slots", str(args.oracle_slots),
                "--max-rounds", str(args.max_rounds),
                "--assignment-rounds", str(args.assignment_rounds),
                "--damping", str(args.damping),
                "--tolerance", str(args.tolerance),
                "--output-dir", str(hour_dir),
                "--skip-figures",
        ])
        _, summary = online.run(forwarded)
    summary = summary.copy()
    summary.insert(0, "start_hour", int(hour))
    return summary


def _refresh(rows: list[pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    combined = pd.concat(rows, ignore_index=True).sort_values(
        ["start_hour", "method", "policy"], kind="stable")
    target = args.output_dir / "hourly_online_summary.csv"
    temporary = target.with_suffix(".tmp.csv")
    combined.to_csv(temporary, index=False)
    temporary.replace(target)
    create_figures(combined, args.figure_dir)
    return combined


def run(args: argparse.Namespace) -> pd.DataFrame:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    pending: list[int] = []
    for hour in args.hours:
        summary_path = args.output_dir / f"h{hour:02d}" / "online_summary.csv"
        if args.resume and summary_path.exists():
            rows.append(_run_hour(int(hour), args))
        else:
            pending.append(int(hour))
    if rows:
        _refresh(rows, args)

    jobs = max(1, min(int(args.parallel_jobs), len(pending) or 1))
    if jobs == 1:
        for hour in pending:
            rows.append(_run_hour(hour, args))
            _refresh(rows, args)
    else:
        print(f"PARALLEL start-hour execution with {jobs} workers", flush=True)
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(_run_hour, hour, args): hour
                for hour in pending
            }
            for future in as_completed(futures):
                hour = futures[future]
                rows.append(future.result())
                _refresh(rows, args)
                print(f"FIGURES UPDATED after start hour {hour:02d}", flush=True)
    return pd.concat(rows, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", nargs="+", type=int, default=list(range(24)))
    parser.add_argument("--n-bins", type=int, default=84)
    parser.add_argument("--vehicles", type=int, default=10)
    parser.add_argument("--demand-seed", type=int, default=7)
    parser.add_argument("--solver-seed", type=int, default=7)
    parser.add_argument("--methods", nargs="+", choices=online.METHODS,
                        default=["proposed"])
    parser.add_argument("--max-simulation-minutes", type=float, default=180.0)
    parser.add_argument("--traffic-step-s", type=float, default=60.0)
    parser.add_argument("--replanning-interval-s", type=float, default=300.0)
    parser.add_argument("--oracle-slots", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=32)
    parser.add_argument("--assignment-rounds", type=int, default=24)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    return parser


def main(arguments: list[str] | None = None) -> None:
    run(build_parser().parse_args(arguments))


if __name__ == "__main__":
    main()
