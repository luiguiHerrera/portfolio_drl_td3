"""Tests for the fresh-market-data basic experiment runner."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from scripts.run_basic_experiment_with_fresh_market_data import (
    build_config_with_returns_path,
    run_basic_experiment_with_fresh_market_data,
    summarize_returns_snapshot,
    write_generated_config,
)


class FreshMarketDataRunnerTests(unittest.TestCase):
    def test_build_config_with_returns_path_injects_returns_path(self):
        config = self._base_config()

        result = build_config_with_returns_path(config, "data/returns.csv")

        self.assertEqual(result["data"]["returns_path"], "data/returns.csv")

    def test_build_config_with_returns_path_preserves_existing_fields(self):
        config = self._base_config()

        result = build_config_with_returns_path(config, "data/returns.csv")

        self.assertEqual(result["project"], config["project"])
        self.assertEqual(result["training"], config["training"])

    def test_build_config_with_returns_path_sets_date_column(self):
        config = self._base_config()

        result = build_config_with_returns_path(
            config,
            "data/returns.csv",
            date_column="observation_date",
        )

        self.assertEqual(result["data"]["returns_date_column"], "observation_date")

    def test_build_config_with_returns_path_sets_snapshot_dates_when_provided(self):
        config = self._base_config()

        result = build_config_with_returns_path(
            config,
            "data/returns.csv",
            start_date="2024-01-05",
            end_date="2024-02-02",
        )

        self.assertEqual(result["data"]["start_date"], "2024-01-05")
        self.assertEqual(result["data"]["end_date"], "2024-02-02")

    def test_build_config_with_returns_path_does_not_modify_original_config(self):
        config = self._base_config()
        original = copy.deepcopy(config)

        build_config_with_returns_path(config, "data/returns.csv")

        self.assertEqual(config, original)

    def test_summarize_returns_snapshot_returns_expected_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns_csv(temp_dir)

            result = summarize_returns_snapshot(returns_path)

        self.assertEqual(result["market_data_start"], "2024-01-05")
        self.assertEqual(result["market_data_end"], "2024-01-12")
        self.assertEqual(result["market_data_shape"], [2, 3])
        self.assertEqual(result["assets"], ["SPY", "TLT", "CASH"])
        self.assertEqual(result["missing_values"], 0)

    def test_summarize_returns_snapshot_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            summarize_returns_snapshot("missing_returns.csv")

    def test_write_generated_config_writes_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_generated_config(
                self._base_config(),
                output_dir=temp_dir,
                experiment_name="fresh_test",
            )
            loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        self.assertEqual(loaded["project"]["name"], "test_project")

    def test_runner_calls_refresh_before_basic_experiment(self):
        call_order = []

        def fake_refresh(**kwargs):
            call_order.append("refresh")
            returns_path = self._write_returns_csv(kwargs["output_path"])
            return {"processed_path": returns_path, "returns": pd.DataFrame()}

        def fake_run(config_path):
            call_order.append("run")
            return {"training_summary": {}}

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            returns_path = str(Path(temp_dir) / "returns.csv")
            with patch(
                "scripts.run_basic_experiment_with_fresh_market_data.write_market_data_outputs",
                side_effect=fake_refresh,
            ), patch(
                "scripts.run_basic_experiment_with_fresh_market_data.run_basic_experiment",
                side_effect=fake_run,
            ):
                run_basic_experiment_with_fresh_market_data(
                    base_config_path=config_path,
                    output_dir=temp_dir,
                    experiment_name="fresh_test",
                    returns_output_path=returns_path,
                )

        self.assertEqual(call_order, ["refresh", "run"])

    def test_runner_defaults_to_latest_end_date_not_config_end_date(self):
        seen_kwargs = {}

        def fake_refresh(**kwargs):
            seen_kwargs.update(kwargs)
            returns_path = self._write_returns_csv(kwargs["output_path"])
            return {"processed_path": returns_path, "returns": pd.DataFrame()}

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            with patch(
                "scripts.run_basic_experiment_with_fresh_market_data.DEFAULT_END_DATE",
                "2026-05-19",
            ), patch(
                "scripts.run_basic_experiment_with_fresh_market_data.write_market_data_outputs",
                side_effect=fake_refresh,
            ), patch(
                "scripts.run_basic_experiment_with_fresh_market_data.run_basic_experiment",
                return_value={"training_summary": {}},
            ):
                result = run_basic_experiment_with_fresh_market_data(
                    base_config_path=config_path,
                    output_dir=temp_dir,
                    experiment_name="fresh_test",
                    returns_output_path=str(Path(temp_dir) / "returns.csv"),
                )

        self.assertEqual(seen_kwargs["end_date"], "2026-05-19")
        self.assertNotEqual(seen_kwargs["end_date"], self._base_config()["data"]["end_date"])
        self.assertFalse(result["metadata"]["respected_config_end_date"])

    def test_runner_uses_config_end_date_when_respect_config_end_date_true(self):
        seen_kwargs = {}

        def fake_refresh(**kwargs):
            seen_kwargs.update(kwargs)
            returns_path = self._write_returns_csv(kwargs["output_path"])
            return {"processed_path": returns_path, "returns": pd.DataFrame()}

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            with patch(
                "scripts.run_basic_experiment_with_fresh_market_data.write_market_data_outputs",
                side_effect=fake_refresh,
            ), patch(
                "scripts.run_basic_experiment_with_fresh_market_data.run_basic_experiment",
                return_value={"training_summary": {}},
            ):
                result = run_basic_experiment_with_fresh_market_data(
                    base_config_path=config_path,
                    output_dir=temp_dir,
                    experiment_name="fresh_test",
                    returns_output_path=str(Path(temp_dir) / "returns.csv"),
                    respect_config_end_date=True,
                )

        self.assertEqual(seen_kwargs["end_date"], self._base_config()["data"]["end_date"])
        self.assertTrue(result["metadata"]["respected_config_end_date"])

    def test_runner_uses_explicit_end_date_when_provided(self):
        seen_kwargs = {}

        def fake_refresh(**kwargs):
            seen_kwargs.update(kwargs)
            returns_path = self._write_returns_csv(kwargs["output_path"])
            return {"processed_path": returns_path, "returns": pd.DataFrame()}

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            with patch(
                "scripts.run_basic_experiment_with_fresh_market_data.write_market_data_outputs",
                side_effect=fake_refresh,
            ), patch(
                "scripts.run_basic_experiment_with_fresh_market_data.run_basic_experiment",
                return_value={"training_summary": {}},
            ):
                result = run_basic_experiment_with_fresh_market_data(
                    base_config_path=config_path,
                    output_dir=temp_dir,
                    experiment_name="fresh_test",
                    returns_output_path=str(Path(temp_dir) / "returns.csv"),
                    end_date="2025-12-31",
                    respect_config_end_date=True,
                )

        self.assertEqual(seen_kwargs["end_date"], "2025-12-31")
        self.assertFalse(result["metadata"]["respected_config_end_date"])

    def test_runner_passes_generated_config_path_to_basic_experiment(self):
        seen_paths = []

        def fake_run(config_path):
            seen_paths.append(config_path)
            return {"training_summary": {}}

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            returns_path = self._write_returns_csv(Path(temp_dir) / "returns.csv")
            with patch(
                "scripts.run_basic_experiment_with_fresh_market_data.write_market_data_outputs",
                return_value={"processed_path": returns_path, "returns": pd.DataFrame()},
            ), patch(
                "scripts.run_basic_experiment_with_fresh_market_data.run_basic_experiment",
                side_effect=fake_run,
            ):
                result = run_basic_experiment_with_fresh_market_data(
                    base_config_path=config_path,
                    output_dir=temp_dir,
                    experiment_name="fresh_test",
                    returns_output_path=returns_path,
                )

        self.assertEqual(seen_paths, [result["generated_config_path"]])

    def test_runner_generated_config_uses_snapshot_end_and_returns_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            returns_path = self._write_returns_csv(Path(temp_dir) / "returns.csv")
            with patch(
                "scripts.run_basic_experiment_with_fresh_market_data.write_market_data_outputs",
                return_value={"processed_path": returns_path, "returns": pd.DataFrame()},
            ), patch(
                "scripts.run_basic_experiment_with_fresh_market_data.run_basic_experiment",
                return_value={"training_summary": {}},
            ):
                result = run_basic_experiment_with_fresh_market_data(
                    base_config_path=config_path,
                    output_dir=temp_dir,
                    experiment_name="fresh_test",
                    returns_output_path=returns_path,
                )
            generated = yaml.safe_load(
                Path(result["generated_config_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(generated["data"]["returns_path"], returns_path)
        self.assertEqual(generated["data"]["returns_date_column"], "date")
        self.assertEqual(generated["data"]["end_date"], "2024-01-12")

    def test_runner_raises_clear_error_if_refresh_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            with patch(
                "scripts.run_basic_experiment_with_fresh_market_data.write_market_data_outputs",
                side_effect=ValueError("download failed"),
            ), patch(
                "scripts.run_basic_experiment_with_fresh_market_data.run_basic_experiment",
            ) as run_mock:
                with self.assertRaisesRegex(RuntimeError, "Market data refresh failed"):
                    run_basic_experiment_with_fresh_market_data(
                        base_config_path=config_path,
                        output_dir=temp_dir,
                        experiment_name="fresh_test",
                    )

        run_mock.assert_not_called()

    def test_runner_writes_metadata_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            returns_path = self._write_returns_csv(Path(temp_dir) / "returns.csv")
            with patch(
                "scripts.run_basic_experiment_with_fresh_market_data.write_market_data_outputs",
                return_value={"processed_path": returns_path, "returns": pd.DataFrame()},
            ), patch(
                "scripts.run_basic_experiment_with_fresh_market_data.run_basic_experiment",
                return_value={"training_summary": {}},
            ):
                result = run_basic_experiment_with_fresh_market_data(
                    base_config_path=config_path,
                    output_dir=temp_dir,
                    experiment_name="fresh_test",
                    returns_output_path=returns_path,
                )
            metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))

        self.assertEqual(metadata["returns_path"], returns_path)
        self.assertEqual(metadata["requested_start_date"], "2024-01-01")
        self.assertIn("requested_end_date", metadata)
        self.assertIn("respected_config_end_date", metadata)
        self.assertEqual(metadata["snapshot_end_used_in_generated_config"], "2024-01-12")
        self.assertIn("run_timestamp_utc", metadata)

    def test_runner_does_not_modify_original_base_config_object(self):
        config = self._base_config()
        original = copy.deepcopy(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            returns_path = self._write_returns_csv(Path(temp_dir) / "returns.csv")
            with patch(
                "scripts.run_basic_experiment_with_fresh_market_data.write_market_data_outputs",
                return_value={"processed_path": returns_path, "returns": pd.DataFrame()},
            ), patch(
                "scripts.run_basic_experiment_with_fresh_market_data.run_basic_experiment",
                return_value={"training_summary": {}},
            ):
                run_basic_experiment_with_fresh_market_data(
                    base_config_path=str(config_path),
                    output_dir=temp_dir,
                    experiment_name="fresh_test",
                    returns_output_path=returns_path,
                )

        self.assertEqual(config, original)

    def _write_returns_csv(self, target) -> str:
        path = Path(target)
        if path.suffix != ".csv":
            path = path / "returns.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        returns = pd.DataFrame(
            {
                "date": ["2024-01-05", "2024-01-12"],
                "SPY": [0.01, 0.02],
                "TLT": [0.00, 0.01],
                "CASH": [0.0, 0.0],
            }
        )
        returns.to_csv(path, index=False)

        return str(path)

    def _write_config(self, directory: str) -> str:
        path = Path(directory) / "base_config.yaml"
        path.write_text(
            yaml.safe_dump(self._base_config(), sort_keys=False),
            encoding="utf-8",
        )

        return str(path)

    def _base_config(self) -> dict:
        return {
            "project": {"name": "test_project"},
            "data": {
                "assets": ["SPY", "TLT", "CASH"],
                "frequency": "weekly",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
            },
            "environment": {"initial_cash": 100000, "transaction_cost": 0.001},
            "reward": {
                "lambda_return": 1.0,
                "lambda_transaction_cost": 1.0,
            },
            "td3": {
                "actor_learning_rate": 0.0005,
                "critic_learning_rate": 0.0005,
                "gamma": 0.99,
                "tau": 0.005,
                "policy_noise": 0.2,
                "noise_clip": 0.5,
                "policy_delay": 2,
                "batch_size": 32,
                "replay_buffer_size": 10000,
            },
            "training": {
                "seed": 42,
                "episodes": 1,
                "train_ratio": 0.7,
                "validation_ratio": 0.15,
                "test_ratio": 0.15,
            },
        }


if __name__ == "__main__":
    unittest.main()
