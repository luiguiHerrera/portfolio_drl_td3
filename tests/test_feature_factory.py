"""Tests for configured feature builder selection."""

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data.feature_factory import build_configured_features


class FeatureFactoryTests(unittest.TestCase):
    def setUp(self):
        self.returns = pd.DataFrame(
            {"SPY": [0.01, 0.02], "CASH": [0.0, 0.0]},
            index=pd.date_range("2024-01-05", periods=2, freq="W-FRI"),
        )
        self.v1_features = pd.DataFrame({"v1_feature": [1.0, 2.0]}, index=self.returns.index)
        self.v2_features = pd.DataFrame({"v2_feature": [3.0, 4.0]}, index=self.returns.index)
        self.v3_features = pd.DataFrame({"v3_feature": [5.0, 6.0]}, index=self.returns.index)
        self.v4_features = pd.DataFrame({"v4_feature": [7.0, 8.0]}, index=self.returns.index)
        self.v5_features = pd.DataFrame({"v5_feature": [9.0, 10.0]}, index=self.returns.index)
        self.macro_data = pd.DataFrame({"VIX": [20.0]}, index=[self.returns.index[0]])

    def test_default_config_without_features_section_uses_v1(self):
        with self._patched_builders() as (
            build_features_mock,
            build_features_v2_mock,
            build_features_v3_mock,
            build_features_v4_mock,
            build_features_v5_mock,
            load_macro_data_mock,
        ):
            result = build_configured_features(self.returns, config={})

        build_features_mock.assert_called_once_with(self.returns)
        build_features_v2_mock.assert_not_called()
        build_features_v3_mock.assert_not_called()
        build_features_v4_mock.assert_not_called()
        build_features_v5_mock.assert_not_called()
        load_macro_data_mock.assert_not_called()
        pd.testing.assert_frame_equal(result, self.v1_features)

    def test_v1_config_uses_v1(self):
        config = {"features": {"version": "v1"}}

        with self._patched_builders() as (
            build_features_mock,
            build_features_v2_mock,
            build_features_v3_mock,
            build_features_v4_mock,
            build_features_v5_mock,
            load_macro_data_mock,
        ):
            result = build_configured_features(self.returns, config=config)

        build_features_mock.assert_called_once_with(self.returns)
        build_features_v2_mock.assert_not_called()
        build_features_v3_mock.assert_not_called()
        build_features_v4_mock.assert_not_called()
        build_features_v5_mock.assert_not_called()
        load_macro_data_mock.assert_not_called()
        pd.testing.assert_frame_equal(result, self.v1_features)

    def test_v2_config_uses_v2_with_configured_parameters(self):
        config = {
            "features": {
                "version": "v2",
                "market_asset": "SPY",
                "short_window": 5,
                "long_window": 13,
                "ewma_span": 8,
            }
        }

        with self._patched_builders() as (
            build_features_mock,
            build_features_v2_mock,
            build_features_v3_mock,
            build_features_v4_mock,
            build_features_v5_mock,
            load_macro_data_mock,
        ):
            result = build_configured_features(self.returns, config=config)

        build_features_mock.assert_not_called()
        build_features_v2_mock.assert_called_once_with(
            self.returns,
            market_asset="SPY",
            short_window=5,
            long_window=13,
            ewma_span=8,
        )
        build_features_v3_mock.assert_not_called()
        build_features_v4_mock.assert_not_called()
        build_features_v5_mock.assert_not_called()
        load_macro_data_mock.assert_not_called()
        pd.testing.assert_frame_equal(result, self.v2_features)

    def test_v3_config_uses_v3_with_configured_parameters(self):
        config = {
            "features": {
                "version": "v3",
                "market_asset": "SPY",
                "short_window": 5,
                "long_window": 13,
                "ewma_span": 8,
            }
        }

        with self._patched_builders() as (
            build_features_mock,
            build_features_v2_mock,
            build_features_v3_mock,
            build_features_v4_mock,
            build_features_v5_mock,
            load_macro_data_mock,
        ):
            result = build_configured_features(self.returns, config=config)

        build_features_mock.assert_not_called()
        build_features_v2_mock.assert_not_called()
        build_features_v3_mock.assert_called_once_with(
            self.returns,
            macro_data=None,
            market_asset="SPY",
            short_window=5,
            long_window=13,
            ewma_span=8,
        )
        build_features_v4_mock.assert_not_called()
        build_features_v5_mock.assert_not_called()
        load_macro_data_mock.assert_not_called()
        pd.testing.assert_frame_equal(result, self.v3_features)

    def test_v4_config_uses_v4_with_configured_parameters(self):
        config = {
            "features": {
                "version": "v4",
                "market_asset": "SPY",
                "short_window": 5,
                "long_window": 13,
                "ewma_span": 8,
                "include_garch_features": True,
                "garch_include_relative": False,
                "garch_omega": 2e-6,
                "garch_alpha": 0.04,
                "garch_beta": 0.91,
                "garch_periods_per_year": 52,
            }
        }

        with self._patched_builders() as (
            build_features_mock,
            build_features_v2_mock,
            build_features_v3_mock,
            build_features_v4_mock,
            build_features_v5_mock,
            load_macro_data_mock,
        ):
            result = build_configured_features(self.returns, config=config)

        build_features_mock.assert_not_called()
        build_features_v2_mock.assert_not_called()
        build_features_v3_mock.assert_not_called()
        build_features_v4_mock.assert_called_once_with(
            self.returns,
            market_asset="SPY",
            short_window=5,
            long_window=13,
            ewma_span=8,
            include_garch_features=True,
            garch_include_relative=False,
            garch_omega=2e-6,
            garch_alpha=0.04,
            garch_beta=0.91,
            garch_periods_per_year=52,
        )
        build_features_v5_mock.assert_not_called()
        load_macro_data_mock.assert_not_called()
        pd.testing.assert_frame_equal(result, self.v4_features)

    def test_v5_config_uses_v5_with_configured_parameters(self):
        config = {
            "features": {
                "version": "v5",
                "market_asset": "SPY",
                "short_window": 5,
                "long_window": 13,
                "ewma_span": 8,
                "correlation_window": 9,
                "drawdown_window": 10,
                "risk_off_threshold": 3.0,
            }
        }

        with self._patched_builders() as (
            build_features_mock,
            build_features_v2_mock,
            build_features_v3_mock,
            build_features_v4_mock,
            build_features_v5_mock,
            load_macro_data_mock,
        ):
            result = build_configured_features(self.returns, config=config)

        build_features_mock.assert_not_called()
        build_features_v2_mock.assert_not_called()
        build_features_v3_mock.assert_not_called()
        build_features_v4_mock.assert_not_called()
        build_features_v5_mock.assert_called_once_with(
            self.returns,
            market_asset="SPY",
            short_window=5,
            long_window=13,
            ewma_span=8,
            correlation_window=9,
            drawdown_window=10,
            risk_off_threshold=3.0,
        )
        load_macro_data_mock.assert_not_called()
        pd.testing.assert_frame_equal(result, self.v5_features)

    def test_v3_without_macro_path_calls_v3_with_macro_data_none(self):
        config = {"features": {"version": "v3"}}

        with self._patched_builders() as (
            build_features_mock,
            build_features_v2_mock,
            build_features_v3_mock,
            build_features_v4_mock,
            build_features_v5_mock,
            load_macro_data_mock,
        ):
            build_configured_features(self.returns, config=config)

        build_features_mock.assert_not_called()
        build_features_v2_mock.assert_not_called()
        build_features_v4_mock.assert_not_called()
        build_features_v5_mock.assert_not_called()
        load_macro_data_mock.assert_not_called()
        self.assertIsNone(build_features_v3_mock.call_args.kwargs["macro_data"])

    def test_v3_with_macro_path_loads_and_passes_macro_data(self):
        config = {
            "features": {
                "version": "v3",
                "macro_path": "local_macro.csv",
                "macro_date_column": "observation_date",
            }
        }

        with self._patched_builders() as (
            build_features_mock,
            build_features_v2_mock,
            build_features_v3_mock,
            build_features_v4_mock,
            build_features_v5_mock,
            load_macro_data_mock,
        ):
            build_configured_features(self.returns, config=config)

        build_features_mock.assert_not_called()
        build_features_v2_mock.assert_not_called()
        build_features_v4_mock.assert_not_called()
        build_features_v5_mock.assert_not_called()
        load_macro_data_mock.assert_called_once_with(
            "local_macro.csv",
            date_column="observation_date",
        )
        pd.testing.assert_frame_equal(
            build_features_v3_mock.call_args.kwargs["macro_data"],
            self.macro_data,
        )

    def test_v3_with_macro_path_loads_csv_and_passes_non_empty_macro_data(self):
        config = {
            "features": {
                "version": "v3",
                "macro_path": str(
                    Path(__file__).parent / "fixtures" / "macro_weekly_test.csv"
                ),
            }
        }

        with patch(
            "src.data.feature_factory.build_features",
            return_value=self.v1_features,
        ) as build_features_mock, patch(
            "src.data.feature_factory.build_features_v2",
            return_value=self.v2_features,
        ) as build_features_v2_mock, patch(
            "src.data.feature_factory.build_features_v3",
            return_value=self.v3_features,
        ) as build_features_v3_mock:
            build_configured_features(self.returns, config=config)

        build_features_mock.assert_not_called()
        build_features_v2_mock.assert_not_called()
        macro_data = build_features_v3_mock.call_args.kwargs["macro_data"]
        self.assertIsInstance(macro_data, pd.DataFrame)
        self.assertFalse(macro_data.empty)

    def test_unsupported_version_raises_value_error(self):
        config = {"features": {"version": "v6"}}

        with self.assertRaisesRegex(ValueError, "Unsupported feature version: v6."):
            build_configured_features(self.returns, config=config)

    def _patched_builders(self):
        build_features_patcher = patch(
            "src.data.feature_factory.build_features",
            return_value=self.v1_features,
        )
        build_features_v2_patcher = patch(
            "src.data.feature_factory.build_features_v2",
            return_value=self.v2_features,
        )
        build_features_v3_patcher = patch(
            "src.data.feature_factory.build_features_v3",
            return_value=self.v3_features,
        )
        build_features_v4_patcher = patch(
            "src.data.feature_factory.build_features_v4",
            return_value=self.v4_features,
        )
        build_features_v5_patcher = patch(
            "src.data.feature_factory.build_features_v5",
            return_value=self.v5_features,
        )
        load_macro_data_patcher = patch(
            "src.data.feature_factory.load_macro_data_from_csv",
            return_value=self.macro_data,
        )

        class PatchedBuilders:
            def __enter__(self_inner):
                build_features_mock = build_features_patcher.__enter__()
                build_features_v2_mock = build_features_v2_patcher.__enter__()
                build_features_v3_mock = build_features_v3_patcher.__enter__()
                build_features_v4_mock = build_features_v4_patcher.__enter__()
                build_features_v5_mock = build_features_v5_patcher.__enter__()
                load_macro_data_mock = load_macro_data_patcher.__enter__()
                return (
                    build_features_mock,
                    build_features_v2_mock,
                    build_features_v3_mock,
                    build_features_v4_mock,
                    build_features_v5_mock,
                    load_macro_data_mock,
                )

            def __exit__(self_inner, exc_type, exc_value, traceback):
                load_macro_data_patcher.__exit__(exc_type, exc_value, traceback)
                build_features_v5_patcher.__exit__(exc_type, exc_value, traceback)
                build_features_v4_patcher.__exit__(exc_type, exc_value, traceback)
                build_features_v3_patcher.__exit__(exc_type, exc_value, traceback)
                build_features_v2_patcher.__exit__(exc_type, exc_value, traceback)
                build_features_patcher.__exit__(exc_type, exc_value, traceback)
                return False

        return PatchedBuilders()


if __name__ == "__main__":
    unittest.main()
