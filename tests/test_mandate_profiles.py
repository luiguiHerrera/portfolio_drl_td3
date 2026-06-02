"""Tests for mandate profile limit infrastructure."""

import unittest

from src.risk.mandate_profiles import (
    MandateLimits,
    get_default_mandate_profiles,
    get_mandate_limits,
    load_mandate_limits_from_config,
)


class MandateProfileTests(unittest.TestCase):
    def test_default_profile_is_moderate(self):
        result = get_mandate_limits()

        self.assertEqual(result, get_default_mandate_profiles()["moderate"])

    def test_conservative_profile_returns_expected_limits(self):
        result = get_mandate_limits("conservative")

        self.assertEqual(result.max_drawdown_limit, -0.10)
        self.assertEqual(result.max_annualized_volatility, 0.10)
        self.assertEqual(result.min_effective_assets, 3.00)
        self.assertEqual(result.max_average_turnover, 0.05)
        self.assertEqual(result.max_weight_limit, 1.00)

    def test_moderate_profile_returns_expected_limits(self):
        result = get_mandate_limits("moderate")

        self.assertEqual(result.max_drawdown_limit, -0.15)
        self.assertEqual(result.max_annualized_volatility, 0.15)
        self.assertEqual(result.min_effective_assets, 2.30)
        self.assertEqual(result.max_average_turnover, 0.10)
        self.assertEqual(result.max_weight_limit, 1.00)

    def test_aggressive_profile_returns_expected_limits(self):
        result = get_mandate_limits("aggressive")

        self.assertEqual(result.max_drawdown_limit, -0.25)
        self.assertEqual(result.max_annualized_volatility, 0.25)
        self.assertEqual(result.min_effective_assets, 1.50)
        self.assertEqual(result.max_average_turnover, 0.20)
        self.assertEqual(result.max_weight_limit, 1.00)

    def test_profiles_are_strictly_ordered(self):
        profiles = get_default_mandate_profiles()

        self.assertGreater(profiles["conservative"].max_drawdown, profiles["moderate"].max_drawdown)
        self.assertGreater(profiles["moderate"].max_drawdown, profiles["aggressive"].max_drawdown)
        self.assertLess(
            profiles["conservative"].max_annualized_volatility,
            profiles["moderate"].max_annualized_volatility,
        )
        self.assertLess(
            profiles["moderate"].max_annualized_volatility,
            profiles["aggressive"].max_annualized_volatility,
        )
        self.assertGreater(
            profiles["conservative"].min_effective_assets,
            profiles["moderate"].min_effective_assets,
        )
        self.assertGreater(
            profiles["moderate"].min_effective_assets,
            profiles["aggressive"].min_effective_assets,
        )
        self.assertLess(
            profiles["conservative"].max_average_turnover,
            profiles["moderate"].max_average_turnover,
        )
        self.assertLess(
            profiles["moderate"].max_average_turnover,
            profiles["aggressive"].max_average_turnover,
        )

    def test_unknown_profile_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported mandate profile"):
            get_mandate_limits("unknown")

    def test_overrides_are_applied_on_selected_profile(self):
        result = get_mandate_limits(
            "conservative",
            overrides={"min_effective_assets": 3.50, "max_turnover_limit": 0.04},
        )

        self.assertEqual(result.max_drawdown_limit, -0.10)
        self.assertEqual(result.min_effective_assets, 3.50)
        self.assertEqual(result.max_average_turnover, 0.04)

    def test_invalid_drawdown_limit_raises_value_error(self):
        for invalid_value in (0.01, -1.0):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError, "max_drawdown"):
                    get_mandate_limits(overrides={"max_drawdown_limit": invalid_value})

    def test_invalid_volatility_limit_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "max_annualized_volatility"):
            get_mandate_limits(overrides={"max_volatility_limit": 0.0})

    def test_max_weight_is_not_a_canonical_override(self):
        with self.assertRaisesRegex(ValueError, "Unknown mandate override keys"):
            get_mandate_limits(overrides={"max_weight_limit": 0.50})

    def test_invalid_effective_assets_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "min_effective_assets"):
            get_mandate_limits(overrides={"min_effective_assets": 0.99})

    def test_invalid_turnover_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "max_average_turnover"):
            get_mandate_limits(overrides={"max_turnover_limit": -0.01})

    def test_bool_values_are_rejected(self):
        invalid_overrides = [
            {"max_drawdown_limit": True},
            {"max_volatility_limit": True},
            {"min_effective_assets": True},
            {"max_turnover_limit": True},
        ]

        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "must be numeric and not bool"):
                    get_mandate_limits(overrides=overrides)

    def test_missing_mandate_section_in_config_returns_moderate(self):
        result = load_mandate_limits_from_config({})

        self.assertEqual(result, get_default_mandate_profiles()["moderate"])

    def test_config_with_profile_returns_that_profile(self):
        result = load_mandate_limits_from_config(
            {"mandate": {"profile": "aggressive"}}
        )

        self.assertEqual(result, get_default_mandate_profiles()["aggressive"])

    def test_config_with_profile_plus_overrides_applies_overrides(self):
        result = load_mandate_limits_from_config(
            {
                "mandate": {
                    "profile": "aggressive",
                    "min_effective_assets": 1.10,
                }
            }
        )

        self.assertEqual(result.max_drawdown_limit, -0.25)
        self.assertEqual(result.min_effective_assets, 1.10)

    def test_config_with_overrides_and_no_profile_uses_moderate_base(self):
        result = load_mandate_limits_from_config(
            {"mandate": {"max_drawdown_limit": -0.25}}
        )

        self.assertEqual(result.max_drawdown_limit, -0.25)
        self.assertEqual(result.max_weight_limit, 1.00)

    def test_unknown_mandate_config_key_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown mandate config keys"):
            load_mandate_limits_from_config({"mandate": {"unknown": 1.0}})

    def test_to_dict_returns_expected_keys(self):
        result = MandateLimits(
            profile_name="moderate",
            max_drawdown=-0.15,
            max_annualized_volatility=0.15,
            min_effective_assets=2.30,
            max_average_turnover=0.10,
            description="test profile",
        ).to_dict()

        self.assertEqual(
            set(result),
            {
                "profile_name",
                "max_drawdown",
                "max_annualized_volatility",
                "min_effective_assets",
                "max_average_turnover",
                "description",
            },
        )
        self.assertNotIn("max_weight", result)
        self.assertNotIn("max_weight_limit", result)

    def test_default_profiles_return_fresh_objects(self):
        first = get_default_mandate_profiles()
        first["moderate"].min_effective_assets = 99.0

        second = get_default_mandate_profiles()

        self.assertEqual(second["moderate"].min_effective_assets, 2.30)


if __name__ == "__main__":
    unittest.main()
