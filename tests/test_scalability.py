"""Regression checks for the scalability experiment definition."""

from pathlib import Path
import json
import unittest

import pandas as pd

from scripts.scalability import vehicles_for_bins


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScalabilityTests(unittest.TestCase):
    def test_monte_carlo_configuration_has_ten_unique_seeds(self) -> None:
        config = json.loads((
            PROJECT_ROOT / "experiments/scalability/config.json"
        ).read_text(encoding="utf-8"))
        seeds = config["instance"]["seeds"]
        self.assertEqual(len(seeds), 10)
        self.assertEqual(len(set(seeds)), 10)

    def test_requested_n_to_k_mapping(self) -> None:
        expected = {
            84: 10, 100: 12, 120: 14, 140: 17,
            160: 19, 180: 21, 200: 24, 220: 26,
            240: 29, 260: 31, 280: 33, 300: 36,
            350: 42, 400: 48, 450: 54, 500: 60,
        }
        self.assertEqual(
            {n: vehicles_for_bins(n, 8.4) for n in expected}, expected)

    def test_nested_bin_file_preserves_measured_prefix(self) -> None:
        measured = pd.read_csv(
            PROJECT_ROOT / "simul/seongbuk_bins_84.csv",
            dtype={"SUMO_EDGE_ID": str}, encoding="utf-8-sig")
        scalability = pd.read_csv(
            PROJECT_ROOT / "simul/scalability_bins_500.csv",
            dtype={"SUMO_EDGE_ID": str}, encoding="utf-8-sig")
        self.assertEqual(len(scalability), 500)
        self.assertEqual(
            measured["SUMO_EDGE_ID"].tolist(),
            scalability.iloc[:84]["SUMO_EDGE_ID"].tolist())
        self.assertEqual(scalability["SUMO_EDGE_ID"].nunique(), 500)
        self.assertEqual(
            int(scalability["SYNTHETIC"].astype(str).str.lower()
                .eq("true").sum()), 416)


if __name__ == "__main__":
    unittest.main()
