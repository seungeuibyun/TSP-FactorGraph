"""Build a weekend operational traffic layer with the archived v3 pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "simul" / "archive" / "preprocessing"
HOURS = [f"~{hour:02d}시" for hour in range(1, 25)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _run(script: Path, cwd: Path) -> None:
    print(f"RUN {script.name}", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=cwd, check=True)


def build(source: Path, output_dir: Path, build_dir: Path) -> Path:
    source = source.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(source, dtype={"링크아이디": str})
    required = {"일자", "요일", "링크아이디", "거리", *HOURS}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"source workbook is missing columns: {sorted(missing)}")
    dates = sorted(raw["일자"].astype(str).unique().tolist())
    weekdays = sorted(raw["요일"].astype(str).unique().tolist())
    invalid_weekdays = sorted(set(weekdays) - {"토", "일"})
    if invalid_weekdays:
        raise ValueError(
            f"source contains non-weekend weekdays: {invalid_weekdays}")
    raw["링크아이디"] = raw["링크아이디"].str.strip()
    if raw.duplicated(["일자", "링크아이디"]).any():
        raise ValueError("duplicate TOPIS link IDs within a source date")
    numeric = raw[HOURS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or (numeric <= 0).any().any():
        raise ValueError("hourly speeds must be finite and positive")

    coverage = pd.read_csv(
        ARCHIVE / "topis_sumo_edge_coverage.csv",
        dtype={"TOPIS_LINK_ID": str, "SUMO_EDGE_ID": str},
    )
    needed = set(coverage["TOPIS_LINK_ID"].str.strip())
    available = set(raw["링크아이디"])
    absent = sorted(needed - available)
    if absent:
        raise ValueError(f"{len(absent)} mapped TOPIS links are absent: {absent[:5]}")

    aggregate = (
        raw.assign(**{hour: numeric[hour] for hour in HOURS})
        .groupby("링크아이디", as_index=False)[HOURS]
        .mean()
    )
    aggregate.to_csv(
        build_dir / "topis_speed.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(
        ARCHIVE / "topis_sumo_edge_coverage.csv",
        build_dir / "topis_sumo_edge_coverage.csv",
    )
    piecewise_name = "20_build_piecewise_observed_cost.py"
    piecewise_text = (ARCHIVE / piecewise_name).read_text(encoding="utf-8")
    piecewise_text = piecewise_text.replace(
        'BASE / "topis_speed.xlsx"', 'BASE / "topis_speed.csv"')
    piecewise_text = piecewise_text.replace(
        "pd.read_excel(\n    SPEED_FILE,", "pd.read_csv(\n    SPEED_FILE,")
    (build_dir / piecewise_name).write_text(
        piecewise_text, encoding="utf-8")
    shutil.copy2(
        ARCHIVE / "21_build_full_network_cost_v3.py",
        build_dir / "21_build_full_network_cost_v3.py")
    _link_or_copy(
        ROOT / "simul" / "seongbuk_buffer.net.xml",
        build_dir / "seongbuk_buffer.net.xml",
    )

    _run(build_dir / "20_build_piecewise_observed_cost.py", build_dir)
    _run(build_dir / "21_build_full_network_cost_v3.py", build_dir)

    full = pd.read_csv(
        build_dir / "sumo_hourly_cost_full_network_v3.csv",
        dtype={"SUMO_EDGE_ID": str}, encoding="utf-8-sig",
    )
    operational_edges = pd.read_csv(
        ROOT / "simul" / "operational_edge_list_v3.csv",
        dtype={"SUMO_EDGE_ID": str}, encoding="utf-8-sig",
    )
    operational = full[
        full["SUMO_EDGE_ID"].isin(set(operational_edges["SUMO_EDGE_ID"]))
    ].copy()
    operational = operational.merge(
        operational_edges[["SUMO_EDGE_ID", "REGION_TYPE"]],
        on="SUMO_EDGE_ID", how="left", validate="one_to_one",
    ).sort_values("SUMO_EDGE_ID", kind="stable").reset_index(drop=True)

    baseline = pd.read_csv(
        ROOT / "simul" / "sumo_hourly_cost_operational_v3.csv",
        dtype={"SUMO_EDGE_ID": str}, encoding="utf-8-sig",
    )
    if set(operational["SUMO_EDGE_ID"]) != set(baseline["SUMO_EDGE_ID"]):
        raise RuntimeError("weekend and baseline operational edge sets differ")

    date_tag = "-".join(dates)
    output = output_dir / f"sumo_hourly_cost_operational_weekend_{date_tag}.csv"
    operational.to_csv(output, index=False, encoding="utf-8-sig")

    qc = pd.DataFrame([{
        "DATES": ",".join(dates),
        "WEEKDAYS": ",".join(weekdays),
        "SOURCE_ROWS": len(raw),
        "SOURCE_LINKS": raw["링크아이디"].nunique(),
        "MAPPED_TOPIS_LINKS": len(needed),
        "MISSING_MAPPED_LINKS": len(absent),
        "OPERATIONAL_EDGES": len(operational),
        "MISSING_HOURLY_SPEEDS": int(numeric.isna().sum().sum()),
        "NONPOSITIVE_HOURLY_SPEEDS": int((numeric <= 0).sum().sum()),
        "MEAN_SOURCE_SPEED_KMH": float(numeric.stack().mean()),
    }])
    qc_path = output_dir / f"weekend_traffic_qc_{date_tag}.csv"
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    manifest = {
        "dates": dates,
        "weekdays": weekdays,
        "source": str(source),
        "source_sha256": _sha256(source),
        "pipeline": [
            "20_build_piecewise_observed_cost.py",
            "21_build_full_network_cost_v3.py",
            "operational_edge_list_v3.csv subset",
        ],
        "output": str(output.relative_to(ROOT)),
        "qc": str(qc_path.relative_to(ROOT)),
        "operational_edges": len(operational),
        "aggregation": "arithmetic mean speed by TOPIS link and hour",
        "note": "Observed Saturday-Sunday weekend profile.",
    }
    (output_dir / f"weekend_traffic_manifest_{date_tag}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CREATED {output}", flush=True)
    return output


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "simul" / "traffic" / "weekend_20260905")
    parser.add_argument(
        "--build-dir", type=Path,
        default=ROOT / "results" / "traffic_build" / "weekend_20260905")
    args = parser.parse_args(arguments)
    build(args.source, args.output_dir.resolve(), args.build_dir.resolve())


if __name__ == "__main__":
    main()
