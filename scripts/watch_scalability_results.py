"""Continuously merge scalability shards and refresh cumulative figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from .plotting.scalability import create_figures
from .scalability import _load_shard, _write_outputs


def _process_exists(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        import os
        if os.name == "nt":
            import ctypes

            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information, False, int(pid))
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _merge_shards(shard_dir: Path, output: Path) -> pd.DataFrame:
    rows = []
    for shard in sorted(shard_dir.glob("*.csv")):
        rows.extend(_load_shard(shard))
    if not rows:
        return pd.DataFrame()
    return _write_outputs(rows, output)


def run(args: argparse.Namespace) -> None:
    shard_dir = args.shard_dir.resolve()
    output = args.output.resolve()
    figure_dir = args.figure_dir.resolve()
    last_signature: tuple[tuple[str, int, int], ...] | None = None
    while True:
        shards = sorted(shard_dir.glob("*.csv"))
        signature = tuple(
            (path.name, path.stat().st_size, path.stat().st_mtime_ns)
            for path in shards
        )
        if signature and signature != last_signature:
            try:
                frame = _merge_shards(shard_dir, output)
                paths = create_figures(output, figure_dir)
                print(
                    f"WATCH UPDATE rows={len(frame)} shards={len(shards)}: "
                    + ", ".join(path.name for path in paths),
                    flush=True)
                last_signature = signature
            except (OSError, PermissionError) as exc:
                print(
                    f"WATCH RETRY after transient file lock: {exc}",
                    flush=True)
        if not _process_exists(args.job_pid) and signature == last_signature:
            print("WATCH FINAL", flush=True)
            return
        time.sleep(args.interval_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-pid", type=int, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=60.0)
    return parser


def main(arguments: list[str] | None = None) -> None:
    run(build_parser().parse_args(arguments))


if __name__ == "__main__":
    main()
