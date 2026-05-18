"""Tests for regime attribution helpers."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.analysis.regime_attribution import (
    build_raw_walk_forward_features,
    build_regime_attribution_report,
    build_walk_forward_regime_attribution_report,
    get_candidate_regime_columns,
    merge_policy_history_with_features,
)


class RegimeAttributionTests(unittest.TestCase):
    def test_get_candidate_regime_columns_selects_regime_columns(self):
        features = self._features()

        result = get_candidate_regime_columns(features)

        self.assertIn("market_high_vol_regime", result)
        self.assertIn("market_risk_off_regime", result)

    def test_get_candidate_regime_columns_selects_macro_columns(self):
        result = get_candidate_regime_columns(self._features())

        self.assertIn("macro_VIX", result)
        self.assertIn("macro_high_vix_regime", result)

    def test_get_candidate_regime_columns_selects_econometric_columns(self):
        result = get_candidate_regime_columns(self._features())

        for column in [
            "macro_yield_curve_10y_2y",
            "macro_dollar_momentum_12p",
            "SPY_vol_12p",
            "SPY_rolling_drawdown_12p",
            "SPY_beta_vs_SPY_12p",
            "SPY_corr_vs_SPY_12p",
        ]:
            self.assertIn(column, result)

    def test_get_candidate_regime_columns_excludes_non_numeric_columns(self):
        features = self._features()
        features["macro_label"] = "risk_off"

        result = get_candidate_regime_columns(features)

        self.assertNotIn("macro_label", result)

    def test_get_candidate_regime_columns_excludes_all_nan_columns(self):
        features = self._features()
        features["macro_empty"] = float("nan")

        result = get_candidate_regime_columns(features)

        self.assertNotIn("macro_empty", result)

    def test_merge_policy_history_with_features_works_with_date_column(self):
        result = merge_policy_history_with_features(
            self._policy_history(),
            self._features(),
        )

        self.assertIn("market_high_vol_regime", result.columns)
        self.assertEqual(len(result), len(self._policy_history()))
        self.assertEqual(result.loc[0, "market_high_vol_regime"], 0.0)

    def test_merge_policy_history_with_features_works_with_datetime_index(self):
        policy_history = self._policy_history().drop(columns=["date"])
        policy_history.index = pd.date_range("2024-01-05", periods=4, freq="W-FRI")

        result = merge_policy_history_with_features(policy_history, self._features())

        self.assertIn("date", result.columns)
        self.assertEqual(result.loc[0, "date"], pd.Timestamp("2024-01-05"))

    def test_merge_policy_history_with_features_preserves_all_policy_rows(self):
        policy_history = self._policy_history()
        policy_history.loc[len(policy_history)] = {
            **policy_history.iloc[-1].to_dict(),
            "date": pd.Timestamp("2024-02-02"),
        }

        result = merge_policy_history_with_features(policy_history, self._features())

        self.assertEqual(len(result), len(policy_history))
        self.assertTrue(pd.isna(result.iloc[-1]["market_high_vol_regime"]))

    def test_merge_policy_history_with_features_raises_value_error_without_shared_dates(self):
        policy_history = self._policy_history()
        policy_history["date"] = pd.date_range("2030-01-04", periods=4, freq="W-FRI")

        with self.assertRaises(ValueError):
            merge_policy_history_with_features(policy_history, self._features())

    def test_merge_policy_history_with_features_raises_type_error_for_non_datetime_features(self):
        features = self._features()
        features.index = range(len(features))

        with self.assertRaises(TypeError):
            merge_policy_history_with_features(self._policy_history(), features)

    def test_build_raw_walk_forward_features_returns_expected_keys(self):
        with self._patched_raw_feature_dependencies():
            result = build_raw_walk_forward_features("config.yaml", self._fold())

        self.assertEqual(
            set(result.keys()),
            {
                "train_features_raw",
                "validation_features_raw",
                "test_features_raw",
                "train_returns",
                "validation_returns",
                "test_returns",
            },
        )
        self.assertFalse(result["test_features_raw"].empty)

    def test_build_raw_walk_forward_features_keeps_binary_regime_values_raw(self):
        with self._patched_raw_feature_dependencies():
            result = build_raw_walk_forward_features("config.yaml", self._fold())

        values = set(result["test_features_raw"]["market_high_vol_regime"].unique())
        self.assertTrue(values.issubset({0.0, 1.0}))

    def test_build_regime_attribution_report_saves_expected_csvs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = self._write_policy_history(temp_dir)
            result = build_regime_attribution_report(
                policy_history_path=policy_path,
                features=self._features(),
                output_dir=temp_dir,
                report_name="regime",
            )

            for key in [
                "merged_observations_path",
                "regime_attribution_path",
                "dominant_asset_distribution_path",
                "concentration_summary_path",
            ]:
                self.assertTrue(Path(result[key]).exists())

    def test_build_regime_attribution_report_returns_attribution_by_dominant_asset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = self._write_policy_history(temp_dir)
            result = build_regime_attribution_report(
                policy_history_path=policy_path,
                features=self._features(),
                regime_columns=["market_high_vol_regime"],
            )

        attribution = result["regime_attribution"]
        self.assertIn("dominant_asset", attribution.columns)
        self.assertIn("mean_market_high_vol_regime", attribution.columns)
        self.assertIn("SPY", attribution["dominant_asset"].tolist())

    def test_build_walk_forward_regime_attribution_report_works_for_test_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = self._write_policy_history(temp_dir)
            with patch(
                "src.analysis.regime_attribution.build_raw_walk_forward_features",
                return_value={
                    "validation_features_raw": self._features(),
                    "test_features_raw": self._features(),
                },
            ):
                result = build_walk_forward_regime_attribution_report(
                    config_path="config.yaml",
                    fold=self._fold(),
                    policy_history_path=policy_path,
                    split="test",
                )

        self.assertFalse(result["regime_attribution"].empty)

    def test_build_walk_forward_regime_attribution_report_works_for_validation_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = self._write_policy_history(temp_dir)
            with patch(
                "src.analysis.regime_attribution.build_raw_walk_forward_features",
                return_value={
                    "validation_features_raw": self._features(),
                    "test_features_raw": self._features(),
                },
            ):
                result = build_walk_forward_regime_attribution_report(
                    config_path="config.yaml",
                    fold=self._fold(),
                    policy_history_path=policy_path,
                    split="validation",
                )

        self.assertFalse(result["regime_attribution"].empty)

    def test_build_walk_forward_regime_attribution_report_rejects_invalid_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = self._write_policy_history(temp_dir)

            with self.assertRaises(ValueError):
                build_walk_forward_regime_attribution_report(
                    config_path="config.yaml",
                    fold=self._fold(),
                    policy_history_path=policy_path,
                    split="train",
                )

    def _features(self):
        index = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
        return pd.DataFrame(
            {
                "market_high_vol_regime": [0.0, 1.0, 1.0, 0.0],
                "market_risk_off_regime": [0.0, 1.0, 0.0, 1.0],
                "macro_VIX": [14.0, 20.0, 25.0, 16.0],
                "macro_high_vix_regime": [0.0, 1.0, 1.0, 0.0],
                "macro_yield_curve_10y_2y": [0.5, 0.4, -0.1, 0.2],
                "macro_dollar_momentum_12p": [0.01, 0.02, -0.01, 0.00],
                "SPY_vol_12p": [0.10, 0.12, 0.20, 0.15],
                "SPY_rolling_drawdown_12p": [0.0, -0.02, -0.05, -0.01],
                "SPY_beta_vs_SPY_12p": [1.0, 1.0, 1.0, 1.0],
                "SPY_corr_vs_SPY_12p": [1.0, 1.0, 1.0, 1.0],
                "SPY_ret_1p": [0.01, -0.01, 0.02, 0.00],
            },
            index=index,
        )

    def _policy_history(self):
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-05", periods=4, freq="W-FRI"),
                "portfolio_return": [0.01, -0.02, 0.03, 0.00],
                "portfolio_value": [101000.0, 98980.0, 101949.4, 101949.4],
                "drawdown": [0.0, -0.02, 0.0, 0.0],
                "turnover": [0.2, 0.3, 0.4, 0.1],
                "transaction_cost": [0.0002, 0.0003, 0.0004, 0.0001],
                "max_weight": [0.8, 0.9, 0.85, 0.75],
                "cash_weight": [0.1, 0.05, 0.05, 0.75],
                "weight_SPY": [0.8, 0.05, 0.05, 0.10],
                "weight_GLD": [0.05, 0.9, 0.05, 0.05],
                "weight_BTC-USD": [0.05, 0.00, 0.85, 0.10],
                "weight_CASH": [0.10, 0.05, 0.05, 0.75],
            }
        )

    def _write_policy_history(self, directory: str) -> str:
        path = Path(directory) / "policy_history.csv"
        self._policy_history().to_csv(path, index=False)
        return str(path)

    def _patched_raw_feature_dependencies(self):
        index = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
        returns = pd.DataFrame(
            {
                "SPY": [0.01] * 8,
                "TLT": [0.00] * 8,
                "GLD": [0.00] * 8,
                "BTC-USD": [0.00] * 8,
                "CASH": [0.00] * 8,
            },
            index=index,
        )
        raw_features = pd.DataFrame(
            {
                "market_high_vol_regime": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "macro_VIX": [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
            },
            index=index,
        )

        class PatchedDependencies:
            def __enter__(self_inner):
                self_inner.load_config_patcher = patch(
                    "src.analysis.regime_attribution.load_config",
                    return_value={"features": {"version": "v3"}},
                )
                self_inner.build_returns_patcher = patch(
                    "src.analysis.regime_attribution.build_returns_dataset",
                    return_value=returns,
                )
                self_inner.build_features_patcher = patch(
                    "src.analysis.regime_attribution.build_configured_features",
                    return_value=raw_features,
                )
                self_inner.load_config_patcher.start()
                self_inner.build_returns_patcher.start()
                self_inner.build_features_patcher.start()

            def __exit__(self_inner, exc_type, exc_value, traceback):
                self_inner.build_features_patcher.stop()
                self_inner.build_returns_patcher.stop()
                self_inner.load_config_patcher.stop()

        return PatchedDependencies()

    @staticmethod
    def _fold():
        return {
            "fold_id": "F1",
            "description": "test_fold",
            "train_start": "2024-01-12",
            "train_end": "2024-01-26",
            "validation_start": "2024-02-02",
            "validation_end": "2024-02-09",
            "test_start": "2024-02-16",
            "test_end": "2024-02-23",
        }


if __name__ == "__main__":
    unittest.main()
