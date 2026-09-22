"""Build the final operational edge grade/EV-energy layer from an elevated net."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iot_energy import (EVParameters, correct_road_grades, edge_energy_kwh,
                        read_sumo_grades)


def main():
    base = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", type=Path, default=base / "seongbuk_buffer_elevation.net.xml")
    ap.add_argument("--traffic", type=Path, default=base / "sumo_hourly_cost_operational_v3.csv")
    ap.add_argument("--output", type=Path, default=base / "operational_edge_energy.csv")
    ap.add_argument("--curb-mass-kg", type=float, default=7000)
    ap.add_argument("--payload-capacity-kg", type=float, default=2000)
    ap.add_argument("--max-abs-grade", type=float, default=0.20,
                    help="physical road-grade cap as a fraction (default: 0.20)")
    args = ap.parse_args()

    params = EVParameters(curb_mass_kg=args.curb_mass_kg,
                          payload_capacity_kg=args.payload_capacity_kg,
                          max_abs_road_grade=args.max_abs_grade)
    traffic = pd.read_csv(args.traffic, dtype={"SUMO_EDGE_ID": str}, encoding="utf-8-sig")
    required = set(traffic["SUMO_EDGE_ID"].astype(str))
    grades = read_sumo_grades(args.net, require_elevation=True,
                              required_edge_ids=required)
    grades = correct_road_grades(grades, args.max_abs_grade)
    out = grades.merge(traffic[["SUMO_EDGE_ID", "SUMO_LENGTH",
                                "FREEFLOW_SPEED_KMH"] +
                               [f"TT_~{h:02d}시" for h in range(1, 25)]],
                       on="SUMO_EDGE_ID", how="inner", validate="one_to_one")
    absent = required - set(out["SUMO_EDGE_ID"])
    if absent:
        raise ValueError(f"{len(absent):,} operational edges lack elevation geometry")
    for h in range(1, 25):
        tt = f"TT_~{h:02d}시"
        out[f"ENERGY_EMPTY_~{h:02d}시_KWH"] = [
            edge_energy_kwh(l, t, g, 0.0, params, vf / 3.6)
            for l, t, g, vf in zip(
                out.SUMO_LENGTH, out[tt], out.GRADE,
                out.FREEFLOW_SPEED_KMH)]
        out[f"ENERGY_FULL_~{h:02d}시_KWH"] = [
            edge_energy_kwh(
                l, t, g, params.payload_capacity_kg, params, vf / 3.6)
            for l, t, g, vf in zip(
                out.SUMO_LENGTH, out[tt], out.GRADE,
                out.FREEFLOW_SPEED_KMH)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"wrote {len(out):,} directed edges to {args.output}")
    corrected = int(out.GRADE_CORRECTED.sum())
    print(f"raw grade range: {out.GRADE_RAW.min():.5f} .. {out.GRADE_RAW.max():.5f}")
    print(f"physical grade range: {out.GRADE.min():.5f} .. {out.GRADE.max():.5f}")
    print(f"corrected edges: {corrected:,}/{len(out):,} ({100*corrected/len(out):.3f}%)")


if __name__ == "__main__":
    main()
