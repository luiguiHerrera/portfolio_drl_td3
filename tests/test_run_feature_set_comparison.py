"""Tests for feature set comparison runner."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src.experiments.run_feature_set_comparison import (
    DEFAULT_FEATURE_SETS,
    run_feature_set_comparison,
)


class RunFeatureSetComparisonTests(unittest.TestCase):
    def test_returns_expected_top_level_keys(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_feature_set_comparison(
                    config_path,
                    output_dir=temp_dir,
                    folds=self._folds(),
                    seeds=[7],
                )

        self.assertEqual(
            set(result.keys()),
            {
                "output_dir",
                "configs_dir",
                "results_path",
                "ranking_path",
                "results",
                "ranking",
                "feature_set_outputs",
            },
        )

    def test_default_feature_sets_are_v1_v2_v3_macro(self):
        self.assertEqual(
            [feature_set["feature_set_id"] for feature_set in DEFAULT_FEATURE_SETS],
            ["V1", "V2", "V3_macro"],
        )

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_feature_set_comparison(
                    config_path,
                    output_dir=temp_dir,
                    folds=self._folds(),
                    seeds=[7],
                )

        self.assertEqual(list(result["results"]["feature_set_id"]), ["V1", "V2", "V3_macro"])

    def test_custom_feature_sets_are_accepted(self):
        feature_sets = [
            {
                "feature_set_id": "custom_v1",
                "description": "custom_default",
                "features": None,
            },
            {
                "feature_set_id": "custom_v2",
                "description": "custom_v2",
                "features": {"version": "v2", "market_asset": "SPY"},
            },
        ]

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner() as runner_mock:
                result = run_feature_set_comparison(
                    config_path,
                    output_dir=temp_dir,
                    feature_sets=feature_sets,
                    folds=self._folds(),
                    seeds=[7],
                )

        self.assertEqual(runner_mock.call_count, len(feature_sets))
        self.assertEqual(list(result["results"]["feature_set_id"]), ["custom_v1", "custom_v2"])

    def test_runner_is_called_once_per_feature_set(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner() as runner_mock:
                run_feature_set_comparison(
                    config_path,
                    output_dir=temp_dir,
                    folds=self._folds(),
                    seeds=[7, 21],
                )

        self.assertEqual(runner_mock.call_count, len(DEFAULT_FEATURE_SETS))

    def test_v1_generated_config_removes_existing_features_section(self):
        with self._temporary_config(include_features=True) as config_path:
            with tempfile.TemporaryDirectory() as temp_dir, self._patched_runner():
                result = run_feature_set_comparison(
                    config_path,
                    output_dir=temp_dir,
                    folds=self._folds(),
                    seeds=[7],
                )
                generated_config = self._read_generated_config(result, "V1")

        self.assertNotIn("features", generated_config)

    def test_v2_generated_config_contains_v2_features(self):
        with self._temporary_config() as config_path:
            with tempfile.TemporaryDirectory() as temp_dir, self._patched_runner():
                result = run_feature_set_comparison(
                    config_path,
                    output_dir=temp_dir,
                    folds=self._folds(),
                    seeds=[7],
                )
                generated_config = self._read_generated_config(result, "V2")

        self.assertEqual(generated_config["features"]["version"], "v2")

    def test_v3_generated_config_contains_v3_features_and_macro_path(self):
        with self._temporary_config() as config_path:
            with tempfile.TemporaryDirectory() as temp_dir, self._patched_runner():
                result = run_feature_set_comparison(
                    config_path,
                    output_dir=temp_dir,
                    folds=self._folds(),
                    seeds=[7],
                )
                generated_config = self._read_generated_config(result, "V3_macro")

        self.assertEqual(generated_config["features"]["version"], "v3")
        self.assertIn("macro_path", generated_config["features"])

    def test_generated_configs_override_training_and_td3_hyperparameters(self):
        with self._temporary_config() as config_path:
            with tempfile.TemporaryDirectory() as temp_dir, self._patched_runner():
                result = run_feature_set_comparison(
                    config_path,
                    output_dir=temp_dir,
                    folds=self._folds(),
                    seeds=[7],
                    episodes=11,
                    batch_size=22,
                    actor_learning_rate=0.001,
                    critic_learning_rate=0.002,
                )
                generated_config = self._read_generated_config(result, "V2")

        self.assertEqual(generated_config["training"]["episodes"], 11)
        self.assertEqual(generated_config["td3"]["batch_size"], 22)
        self.assertEqual(generated_config["td3"]["actor_learning_rate"], 0.001)
        self.assertEqual(generated_config["td3"]["critic_learning_rate"], 0.002)

    def test_aggregate_results_has_one_row_per_feature_set(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_feature_set_comparison(
                    config_path,
                    output_dir=temp_dir,
                    folds=self._folds(),
                    seeds=[7],
                )

        self.assertEqual(len(result["results"]), len(DEFAULT_FEATURE_SETS))

    def test_aggregate_results_contains_robust_metric_columns(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_feature_set_comparison(
                    config_path,
                    output_dir=temp_dir,
                    folds=self._folds(),
                    seeds=[7],
                )

        expected_columns = {
            "robust_test_agent_sharpe_score_05",
            "robust_test_agent_sharpe_score_10",
            "robust_test_agent_information_ratio_score_05",
            "robust_test_agent_capm_alpha_score_05",
        }
        self.assertTrue(expected_columns.issubset(set(result["results"].columns)))

    def test_ranking_is_sorted_by_robust_sharpe_descending(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_feature_set_comparison(
                    config_path,
                    output_dir=temp_dir,
                    folds=self._folds(),
                    seeds=[7],
                )

        robust_scores = result["ranking"]["robust_test_agent_sharpe_score_05"].tolist()
        self.assertEqual(robust_scores, sorted(robust_scores, reverse=True))

    def test_csv_result_and_ranking_files_are_saved(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_feature_set_comparison(
                    config_path,
                    output_dir=temp_dir,
                    folds=self._folds(),
                    seeds=[7],
                )

            self.assertTrue(Path(result["results_path"]).is_file())
            self.assertTrue(Path(result["ranking_path"]).is_file())

    def test_feature_set_outputs_are_keyed_by_feature_set_id(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_feature_set_comparison(
                    config_path,
                    output_dir=temp_dir,
                    folds=self._folds(),
                    seeds=[7],
                )

        self.assertEqual(set(result["feature_set_outputs"].keys()), {"V1", "V2", "V3_macro"})

    def _patched_runner(self):
        return patch(
            "src.experiments.run_feature_set_comparison.run_walk_forward_seed_validation",
            side_effect=self._mock_run_walk_forward_seed_validation,
        )

    @staticmethod
    def _mock_run_walk_forward_seed_validation(
        base_config_path: str,
        output_dir: str,
        experiment_name: str,
        folds: list[dict],
        seeds: list[int],
        episodes: int,
        batch_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
    ) -> dict:
        score_by_feature_set = {
            "V1": 0.10,
            "V2": 0.30,
            "V3_macro": 0.20,
            "custom_v1": 0.15,
            "custom_v2": 0.25,
        }
        robust_sharpe = score_by_feature_set.get(experiment_name, 0.05)
        output_path = Path(output_dir) / experiment_name

        return {
            "output_dir": str(output_path),
            "summary_path": str(output_path / "walk_forward_seed_summary.csv"),
            "by_fold_summary_path": str(output_path / "walk_forward_seed_by_fold_summary.csv"),
            "by_seed_summary_path": str(output_path / "walk_forward_seed_by_seed_summary.csv"),
            "summary": pd.DataFrame(
                [
                    {
                        "n_observations": len(folds) * len(seeds),
                        "n_folds": len(folds),
                        "n_seeds": len(seeds),
                        "mean_test_agent_sharpe": robust_sharpe + 0.1,
                        "std_test_agent_sharpe": 0.2,
                        "robust_test_agent_sharpe_score_05": robust_sharpe,
                        "robust_test_agent_sharpe_score_10": robust_sharpe - 0.1,
                        "mean_test_agent_sortino": robust_sharpe + 0.2,
                        "robust_test_agent_sortino_score_05": robust_sharpe + 0.05,
                        "mean_test_agent_information_ratio_vs_equal_weight_rebalanced_net": (
                            robust_sharpe - 0.1
                        ),
                        "robust_test_agent_information_ratio_score_05": robust_sharpe - 0.2,
                        "mean_test_agent_capm_alpha_vs_SPY": robust_sharpe - 0.05,
                        "robust_test_agent_capm_alpha_score_05": robust_sharpe - 0.15,
                        "mean_test_agent_cumulative_return": robust_sharpe + 0.3,
                        "mean_test_agent_max_drawdown": -0.1,
                        "worst_test_agent_sharpe": robust_sharpe - 0.4,
                        "worst_test_agent_cumulative_return": -0.2,
                        "worst_test_agent_max_drawdown": -0.3,
                        "positive_sharpe_rate": 0.8,
                        "positive_sortino_rate": 0.7,
                        "positive_capm_alpha_rate": 0.6,
                        "positive_information_ratio_rate": 0.5,
                        "win_rate_best_policy_agent": 0.4,
                        "win_rate_vs_best_individual_buyhold_by_sharpe": 0.3,
                        "mean_test_average_turnover": 0.2,
                        "mean_test_average_effective_number_of_assets": 1.5,
                    }
                ]
            ),
        }

    @staticmethod
    def _folds() -> list[dict]:
        return [
            {
                "fold_id": "F1",
                "description": "fold_one",
                "train_start": "2021-01-01",
                "train_end": "2021-12-31",
                "validation_start": "2022-01-01",
                "validation_end": "2022-06-30",
                "test_start": "2022-07-01",
                "test_end": "2022-12-31",
            }
        ]

    @staticmethod
    def _read_generated_config(result: dict, feature_set_id: str) -> dict:
        config_path = Path(result["configs_dir"]) / f"{feature_set_id}.yaml"
        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _temporary_config(self, include_features: bool = False):
        features_section = ""
        if include_features:
            features_section = """
features:
  version: v2
  market_asset: SPY
"""
        config_text = f"""
project:
  name: portfolio_drl_td3_feature_set_comparison_test

data:
  assets:
    - SPY
    - TLT
    - GLD
    - BTC-USD
    - CASH
  frequency: weekly
  start_date: 2020-01-01
  end_date: 2024-12-31

environment:
  initial_cash: 100000
  transaction_cost: 0.001

reward:
  lambda_return: 1.0

td3:
  actor_learning_rate: 0.0003
  critic_learning_rate: 0.0003
  gamma: 0.99
  tau: 0.005
  policy_noise: 0.2
  noise_clip: 0.5
  policy_delay: 2
  batch_size: 32
  replay_buffer_size: 1000

training:
  seed: 42
  episodes: 3
  train_ratio: 0.7
  validation_ratio: 0.15
  test_ratio: 0.15
{features_section}
"""
        temp_dir = tempfile.TemporaryDirectory()
        config_path = Path(temp_dir.name) / "config.yaml"
        config_path.write_text(config_text, encoding="utf-8")

        class TemporaryConfig:
            def __enter__(self_inner):
                return str(config_path)

            def __exit__(self_inner, exc_type, exc_value, traceback):
                temp_dir.cleanup()
                return False

        return TemporaryConfig()


if __name__ == "__main__":
    unittest.main()
