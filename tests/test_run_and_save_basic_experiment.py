"""Tests for the run-and-save basic experiment workflow."""

import unittest
from unittest.mock import patch

from src.experiments.run_and_save_basic_experiment import run_and_save_basic_experiment


class RunAndSaveBasicExperimentTests(unittest.TestCase):
    def setUp(self):
        self.experiment_result = {
            "training_summary": {"total_episodes": 2},
            "raw_result": {"agent": object(), "replay_buffer": object()},
        }
        self.saved_paths = {
            "output_dir": "outputs/tables/basic_td3_experiment",
            "training_summary": "outputs/tables/basic_td3_experiment/training_summary.csv",
        }

    def test_returns_expected_top_level_keys(self):
        result = self._run_workflow()

        self.assertEqual(set(result.keys()), {"experiment_result", "saved_paths"})

    def test_calls_run_basic_experiment_once_with_config_path(self):
        config_path = "configs/config.yaml"

        with self._patched_dependencies() as (run_mock, _):
            run_and_save_basic_experiment(config_path)

        run_mock.assert_called_once_with(config_path)

    def test_calls_save_basic_experiment_outputs_once_with_expected_arguments(self):
        output_dir = "custom/output"
        experiment_name = "custom_experiment"

        with self._patched_dependencies() as (_, save_mock):
            run_and_save_basic_experiment(
                "configs/config.yaml",
                output_dir=output_dir,
                experiment_name=experiment_name,
            )

        save_mock.assert_called_once_with(
            self.experiment_result,
            output_dir=output_dir,
            experiment_name=experiment_name,
        )

    def test_passes_custom_output_dir_and_experiment_name_correctly(self):
        output_dir = "tmp/tables"
        experiment_name = "smoke_test"

        with self._patched_dependencies() as (_, save_mock):
            run_and_save_basic_experiment(
                "configs/config.yaml",
                output_dir=output_dir,
                experiment_name=experiment_name,
            )

        self.assertEqual(save_mock.call_args.kwargs["output_dir"], output_dir)
        self.assertEqual(save_mock.call_args.kwargs["experiment_name"], experiment_name)

    def test_returns_original_experiment_result_object(self):
        result = self._run_workflow()

        self.assertIs(result["experiment_result"], self.experiment_result)

    def test_returns_original_saved_paths_object(self):
        result = self._run_workflow()

        self.assertIs(result["saved_paths"], self.saved_paths)

    def _run_workflow(self):
        with self._patched_dependencies():
            return run_and_save_basic_experiment("configs/config.yaml")

    def _patched_dependencies(self):
        run_patch = patch(
            "src.experiments.run_and_save_basic_experiment.run_basic_experiment",
            return_value=self.experiment_result,
        )
        save_patch = patch(
            "src.experiments.run_and_save_basic_experiment.save_basic_experiment_outputs",
            return_value=self.saved_paths,
        )

        class PatchedDependencies:
            def __enter__(self_inner):
                return run_patch.start(), save_patch.start()

            def __exit__(self_inner, exc_type, exc_value, traceback):
                save_patch.stop()
                run_patch.stop()
                return False

        return PatchedDependencies()


if __name__ == "__main__":
    unittest.main()
