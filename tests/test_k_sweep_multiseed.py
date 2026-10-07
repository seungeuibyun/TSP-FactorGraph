"""Regression checks for the cumulative Monte Carlo K sweep."""

import unittest

import pandas as pd

from scripts.k_sweep_multiseed import planned_instances
from scripts.plotting.k_sweep_multiseed import (
    _absolute_summary,
    _paired_gap_summary,
)


class KSweepMonteCarloTests(unittest.TestCase):
    def test_two_tier_plan_has_twenty_focus_and_ten_other_seeds(self) -> None:
        base = list(range(7, 107, 10))
        focus = list(range(107, 207, 10))
        tasks = planned_instances(
            base, focus, [4, 11, 19], list(range(6, 15)), [6, 7, 8])
        self.assertEqual(len(tasks), 360)
        for vehicles in range(6, 15):
            seeds = {
                seed for seed, _, task_vehicles in tasks
                if task_vehicles == vehicles
            }
            self.assertEqual(len(seeds), 20 if vehicles <= 8 else 10)

    def test_hours_are_averaged_before_seed_bootstrap(self) -> None:
        rows = []
        for seed, proposed, aco in ((7, 10.0, 12.0), (17, 20.0, 22.0)):
            for hour in (4, 11):
                rows.extend([
                    {
                        "method": "proposed", "vehicles": 6,
                        "demand_seed": seed, "start_hour": hour,
                        "energy_kwh": proposed + hour,
                    },
                    {
                        "method": "aco", "vehicles": 6,
                        "demand_seed": seed, "start_hour": hour,
                        "energy_kwh": aco + hour,
                    },
                ])
        frame = pd.DataFrame(rows)
        absolute = _absolute_summary(
            frame, "energy_kwh", confidence_level=0.95, draws=200)
        proposed = absolute[absolute["method"] == "proposed"].iloc[0]
        self.assertEqual(int(proposed["seeds"]), 2)
        self.assertEqual(int(proposed["minimum_hours_per_seed"]), 2)
        self.assertAlmostEqual(float(proposed["mean"]), 22.5)

        gap = _paired_gap_summary(
            frame, "energy_kwh", confidence_level=0.95, draws=200)
        proposed_gap = gap[gap["method"] == "proposed"].iloc[0]
        self.assertAlmostEqual(float(proposed_gap["mean"]), 0.0)
        aco_gap = gap[gap["method"] == "aco"].iloc[0]
        self.assertGreater(float(aco_gap["mean"]), 0.0)

    def test_seed_tiers_must_not_overlap(self) -> None:
        with self.assertRaises(ValueError):
            planned_instances([7, 17], [17], [11], [6, 7], [6])


if __name__ == "__main__":
    unittest.main()
