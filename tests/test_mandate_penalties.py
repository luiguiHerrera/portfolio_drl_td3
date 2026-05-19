"""Tests for pure mandate-aware penalty components."""

import unittest

from src.risk.mandate_penalties import (
    compute_cash_breach,
    compute_mandate_breaches,
    compute_mandate_penalty,
    compute_weighted_mandate_penalty,
    drawdown_breach,
    effective_assets_breach,
    max_weight_breach,
    positive_part,
    turnover_breach,
    volatility_breach,
)
from src.risk.mandate_profiles import get_mandate_limits


class MandatePenaltyTests(unittest.TestCase):
    def test_positive_part_returns_zero_for_negative_values(self):
        self.assertEqual(positive_part(-0.5), 0.0)

    def test_drawdown_breach_returns_zero_when_within_limit(self):
        self.assertEqual(drawdown_breach(-0.10, -0.20), 0.0)

    def test_drawdown_breach_returns_positive_excess_when_breached(self):
        self.assertAlmostEqual(drawdown_breach(-0.25, -0.20), 0.05)

    def test_volatility_breach_works(self):
        self.assertEqual(volatility_breach(0.20, 0.25), 0.0)
        self.assertAlmostEqual(volatility_breach(0.30, 0.25), 0.05)

    def test_max_weight_breach_works(self):
        self.assertEqual(max_weight_breach(0.70, 0.80), 0.0)
        self.assertAlmostEqual(max_weight_breach(0.95, 0.80), 0.15)

    def test_effective_assets_breach_works(self):
        self.assertEqual(effective_assets_breach(1.50, 1.25), 0.0)
        self.assertAlmostEqual(effective_assets_breach(1.10, 1.25), 0.15)

    def test_turnover_breach_works(self):
        self.assertEqual(turnover_breach(0.50, 0.75), 0.0)
        self.assertAlmostEqual(turnover_breach(0.90, 0.75), 0.15)

    def test_compute_cash_breach_returns_zero_within_normal_band(self):
        self.assertEqual(compute_cash_breach(0.05, normal_cash_max=0.10), 0.0)

    def test_compute_cash_breach_returns_excess_outside_risk_off(self):
        self.assertAlmostEqual(
            compute_cash_breach(
                cash_weight=0.35,
                normal_cash_max=0.10,
                risk_off_state=False,
            ),
            0.25,
        )

    def test_compute_cash_breach_returns_zero_when_risk_off_state_true(self):
        self.assertEqual(
            compute_cash_breach(
                cash_weight=0.35,
                normal_cash_max=0.10,
                risk_off_state=True,
            ),
            0.0,
        )

    def test_compute_cash_breach_rejects_invalid_cash_weight(self):
        for invalid_cash_weight in (-0.01, 1.01, True):
            with self.subTest(invalid_cash_weight=invalid_cash_weight):
                with self.assertRaises(ValueError):
                    compute_cash_breach(invalid_cash_weight)

    def test_compute_mandate_breaches_returns_expected_keys(self):
        result = compute_mandate_breaches(
            current_drawdown=-0.25,
            current_volatility=0.30,
            max_weight=0.95,
            effective_assets=1.10,
            turnover=0.90,
            mandate_limits=get_mandate_limits("moderate"),
        )

        self.assertEqual(
            set(result),
            {
                "drawdown_breach",
                "volatility_breach",
                "max_weight_breach",
                "effective_assets_breach",
                "turnover_breach",
            },
        )

    def test_compute_weighted_mandate_penalty_uses_default_weights(self):
        breaches = {
            "drawdown_breach": 0.05,
            "volatility_breach": 0.05,
            "max_weight_breach": 0.15,
            "effective_assets_breach": 0.15,
            "turnover_breach": 0.15,
        }

        result = compute_weighted_mandate_penalty(breaches)

        self.assertAlmostEqual(result, 0.55)

    def test_compute_weighted_mandate_penalty_applies_custom_weights(self):
        breaches = {
            "drawdown_breach": 0.05,
            "volatility_breach": 0.05,
            "max_weight_breach": 0.15,
            "effective_assets_breach": 0.15,
            "turnover_breach": 0.15,
        }

        result = compute_weighted_mandate_penalty(
            breaches,
            penalty_weights={"max_weight_breach": 2.0, "turnover_breach": 0.0},
        )

        self.assertAlmostEqual(result, 0.55)

    def test_compute_weighted_mandate_penalty_rejects_unknown_breach_keys(self):
        with self.assertRaisesRegex(ValueError, "Unknown mandate breach keys"):
            compute_weighted_mandate_penalty({"unknown": 0.1})

    def test_compute_weighted_mandate_penalty_rejects_unknown_penalty_weight_keys(self):
        with self.assertRaisesRegex(ValueError, "Unknown mandate penalty weight keys"):
            compute_weighted_mandate_penalty(
                {"drawdown_breach": 0.1},
                penalty_weights={"unknown": 1.0},
            )

    def test_compute_mandate_penalty_returns_penalty_and_breaches(self):
        result = compute_mandate_penalty(
            current_drawdown=-0.25,
            current_volatility=0.30,
            max_weight=0.95,
            effective_assets=1.10,
            turnover=0.90,
            mandate_limits=get_mandate_limits("moderate"),
        )

        self.assertIn("penalty", result)
        self.assertIn("breaches", result)
        self.assertGreater(result["penalty"], 0.0)

    def test_aggressive_profile_allows_max_weight_one_without_concentration_breach(self):
        result = compute_mandate_breaches(
            current_drawdown=-0.10,
            current_volatility=0.20,
            max_weight=1.0,
            effective_assets=1.0,
            turnover=0.50,
            mandate_limits=get_mandate_limits("aggressive"),
        )

        self.assertEqual(result["max_weight_breach"], 0.0)

    def test_moderate_profile_flags_high_max_weight_as_breach(self):
        result = compute_mandate_breaches(
            current_drawdown=-0.10,
            current_volatility=0.20,
            max_weight=0.95,
            effective_assets=1.25,
            turnover=0.50,
            mandate_limits=get_mandate_limits("moderate"),
        )

        self.assertAlmostEqual(result["max_weight_breach"], 0.15)

    def test_bool_inputs_are_rejected(self):
        calls = [
            lambda: positive_part(True),
            lambda: drawdown_breach(True, -0.20),
            lambda: volatility_breach(True, 0.25),
            lambda: max_weight_breach(True, 0.80),
            lambda: effective_assets_breach(True, 1.25),
            lambda: turnover_breach(True, 0.75),
            lambda: compute_cash_breach(True),
            lambda: compute_weighted_mandate_penalty(
                {"drawdown_breach": 0.1},
                penalty_weights={"drawdown_breach": True},
            ),
        ]

        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "numeric and not bool"):
                    call()

    def test_invalid_drawdown_inputs_are_rejected(self):
        for invalid_drawdown in (0.01, -1.0):
            with self.subTest(invalid_drawdown=invalid_drawdown):
                with self.assertRaisesRegex(ValueError, "current_drawdown"):
                    drawdown_breach(invalid_drawdown, -0.20)
        with self.assertRaisesRegex(ValueError, "max_drawdown_limit"):
            drawdown_breach(-0.10, 0.01)

    def test_invalid_volatility_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "current_volatility"):
            volatility_breach(-0.01, 0.25)
        with self.assertRaisesRegex(ValueError, "max_volatility_limit"):
            volatility_breach(0.20, 0.0)

    def test_invalid_max_weight_inputs_are_rejected(self):
        for invalid_max_weight in (-0.01, 1.01):
            with self.subTest(invalid_max_weight=invalid_max_weight):
                with self.assertRaisesRegex(ValueError, "max_weight"):
                    max_weight_breach(invalid_max_weight, 0.80)
        with self.assertRaisesRegex(ValueError, "max_weight_limit"):
            max_weight_breach(0.80, 0.0)

    def test_invalid_effective_assets_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "effective_assets"):
            effective_assets_breach(0.99, 1.25)
        with self.assertRaisesRegex(ValueError, "min_effective_assets"):
            effective_assets_breach(1.25, 0.99)

    def test_invalid_turnover_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "turnover"):
            turnover_breach(-0.01, 0.75)
        with self.assertRaisesRegex(ValueError, "max_turnover_limit"):
            turnover_breach(0.20, -0.01)


if __name__ == "__main__":
    unittest.main()
