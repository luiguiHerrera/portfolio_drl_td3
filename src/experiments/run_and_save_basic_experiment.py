"""Run and save workflow for the basic TD3 experiment.

This module connects the in-memory basic experiment runner with the CSV saving
utility. It is intentionally importable only and does not expose a CLI, print
reports, create plots, or save model objects.
"""

from src.experiments.run_basic_experiment import run_basic_experiment
from src.experiments.save_experiment_outputs import save_basic_experiment_outputs


def run_and_save_basic_experiment(
    config_path: str,
    output_dir: str = "outputs/tables",
    experiment_name: str = "basic_td3_experiment",
) -> dict:
    """Run the basic experiment and save selected CSV outputs."""
    experiment_result = run_basic_experiment(config_path)
    saved_paths = save_basic_experiment_outputs(
        experiment_result,
        output_dir=output_dir,
        experiment_name=experiment_name,
    )

    return {
        "experiment_result": experiment_result,
        "saved_paths": saved_paths,
    }
