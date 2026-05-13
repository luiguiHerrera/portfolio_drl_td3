"""Tests for configured feature builder selection."""

import unittest
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

    def test_default_config_without_features_section_uses_v1(self):
        with self._patched_builders() as (build_features_mock, build_features_v2_mock):
            result = build_configured_features(self.returns, config={})

        build_features_mock.assert_called_once_with(self.returns)
        build_features_v2_mock.assert_not_called()
        pd.testing.assert_frame_equal(result, self.v1_features)

    def test_v1_config_uses_v1(self):
        config = {"features": {"version": "v1"}}

        with self._patched_builders() as (build_features_mock, build_features_v2_mock):
            result = build_configured_features(self.returns, config=config)

        build_features_mock.assert_called_once_with(self.returns)
        build_features_v2_mock.assert_not_called()
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

        with self._patched_builders() as (build_features_mock, build_features_v2_mock):
            result = build_configured_features(self.returns, config=config)

        build_features_mock.assert_not_called()
        build_features_v2_mock.assert_called_once_with(
            self.returns,
            market_asset="SPY",
            short_window=5,
            long_window=13,
            ewma_span=8,
        )
        pd.testing.assert_frame_equal(result, self.v2_features)

    def test_unsupported_version_raises_value_error(self):
        config = {"features": {"version": "v3"}}

        with self.assertRaisesRegex(ValueError, "Unsupported feature version: v3."):
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

        class PatchedBuilders:
            def __enter__(self_inner):
                build_features_mock = build_features_patcher.__enter__()
                build_features_v2_mock = build_features_v2_patcher.__enter__()
                return build_features_mock, build_features_v2_mock

            def __exit__(self_inner, exc_type, exc_value, traceback):
                build_features_v2_patcher.__exit__(exc_type, exc_value, traceback)
                build_features_patcher.__exit__(exc_type, exc_value, traceback)
                return False

        return PatchedBuilders()


if __name__ == "__main__":
    unittest.main()
