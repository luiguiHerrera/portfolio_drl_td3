import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.costs.spread_costs import (
    aggregate_weekly_close_spreads,
    aggregate_weekly_spreads,
    build_proxy_weekly_spreads,
    compute_half_spread,
    compute_spread_cost,
    estimate_dynamic_spread_from_volatility,
)


class SpreadCostsTest(unittest.TestCase):
    def test_half_spread_calculation_is_correct(self):
        quotes = pd.DataFrame(
            {
                "timestamp": ["2024-01-05 15:59:00"],
                "asset": ["SPY"],
                "bid": [99.0],
                "ask": [101.0],
            }
        )
        result = compute_half_spread(quotes)
        self.assertAlmostEqual(result.loc[0, "mid"], 100.0)
        self.assertAlmostEqual(result.loc[0, "half_spread"], 0.01)

    def test_weekly_aggregation_works_and_cash_zero(self):
        spreads = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-05"]),
                "asset": ["SPY", "SPY", "CASH"],
                "half_spread": [0.001, 0.003, 0.5],
            }
        )
        weekly = aggregate_weekly_spreads(spreads)
        self.assertAlmostEqual(weekly.loc[pd.Timestamp("2024-01-05"), "SPY"], 0.002)
        self.assertAlmostEqual(weekly.loc[pd.Timestamp("2024-01-05"), "CASH"], 0.0)

    def test_close_spread_aggregation_uses_last_observation(self):
        spreads = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-05 09:30", "2024-01-05 16:00"]),
                "asset": ["SPY", "SPY"],
                "half_spread": [0.001, 0.004],
            }
        )
        weekly = aggregate_weekly_close_spreads(spreads)
        self.assertAlmostEqual(weekly.loc[pd.Timestamp("2024-01-05"), "SPY"], 0.004)

    def test_spread_cost_increases_with_turnover(self):
        target_low = pd.Series({"SPY": 0.55, "CASH": 0.45})
        target_high = pd.Series({"SPY": 0.90, "CASH": 0.10})
        drifted = pd.Series({"SPY": 0.50, "CASH": 0.50})
        spreads = pd.Series({"SPY": 0.001, "CASH": 0.0})
        self.assertGreater(
            compute_spread_cost(target_high, drifted, spreads),
            compute_spread_cost(target_low, drifted, spreads),
        )

    def test_cash_has_zero_spread_even_if_input_is_nonzero(self):
        target = pd.Series({"SPY": 0.0, "CASH": 1.0})
        drifted = pd.Series({"SPY": 1.0, "CASH": 0.0})
        spreads = pd.Series({"SPY": 0.001, "CASH": 1.0})
        self.assertAlmostEqual(compute_spread_cost(target, drifted, spreads), 0.001)

    def test_missing_spreads_are_handled_safely_with_warnings(self):
        with self.assertWarns(RuntimeWarning):
            frame, warnings = build_proxy_weekly_spreads(
                dates=["2024-01-05"],
                assets=["SPY", "CASH"],
                base_half_spreads={"SPY": 0.001},
            )
        self.assertIn("proxy spread scenario", warnings[0])
        self.assertAlmostEqual(frame.loc[pd.Timestamp("2024-01-05"), "CASH"], 0.0)

    def test_dynamic_spread_from_volatility_is_nonnegative(self):
        spreads = estimate_dynamic_spread_from_volatility(
            0.001,
            pd.Series([0.01, 0.02, 0.03]),
            beta=0.5,
        )
        self.assertTrue((spreads >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
