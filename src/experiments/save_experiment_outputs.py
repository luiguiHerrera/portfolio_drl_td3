"""CSV output utilities for basic experiment results.

This module saves selected tabular outputs from an in-memory experiment result.
It does not save models, replay buffers, raw results, plots, or reports.
"""

from pathlib import Path

import pandas as pd


REQUIRED_EXPERIMENT_RESULT_KEYS = (
    "training_summary",
    "validation_metrics_table",
    "test_metrics_table",
    "validation_comparison_summary",
    "test_comparison_summary",
    "validation_diagnostics",
    "test_diagnostics",
)


def save_basic_experiment_outputs(
    experiment_result: dict,
    output_dir: str = "outputs/tables",
    experiment_name: str = "basic_td3_experiment",
) -> dict:
    """Save selected basic experiment outputs as CSV files."""
    _validate_experiment_result(experiment_result)

    experiment_output_dir = Path(output_dir) / experiment_name
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    paths = {"output_dir": str(experiment_output_dir)}
    paths["validation_metrics_table"] = _save_dataframe(
        experiment_result["validation_metrics_table"],
        experiment_output_dir / "validation_metrics_table.csv",
    )
    paths["test_metrics_table"] = _save_dataframe(
        experiment_result["test_metrics_table"],
        experiment_output_dir / "test_metrics_table.csv",
    )
    paths["training_summary"] = _save_dict(
        experiment_result["training_summary"],
        experiment_output_dir / "training_summary.csv",
    )
    paths["validation_comparison_summary"] = _save_dict(
        experiment_result["validation_comparison_summary"],
        experiment_output_dir / "validation_comparison_summary.csv",
    )
    paths["test_comparison_summary"] = _save_dict(
        experiment_result["test_comparison_summary"],
        experiment_output_dir / "test_comparison_summary.csv",
    )
    paths["validation_diagnostics"] = _save_dict(
        _flatten_diagnostics(experiment_result["validation_diagnostics"]),
        experiment_output_dir / "validation_diagnostics.csv",
    )
    paths["test_diagnostics"] = _save_dict(
        _flatten_diagnostics(experiment_result["test_diagnostics"]),
        experiment_output_dir / "test_diagnostics.csv",
    )

    return paths


def _validate_experiment_result(experiment_result: dict) -> None:
    if not isinstance(experiment_result, dict):
        raise TypeError("experiment_result must be a dictionary.")

    missing_keys = [
        key for key in REQUIRED_EXPERIMENT_RESULT_KEYS if key not in experiment_result
    ]
    if missing_keys:
        raise KeyError(f"experiment_result is missing required keys: {missing_keys}")

    for key in ("validation_metrics_table", "test_metrics_table"):
        if not isinstance(experiment_result[key], pd.DataFrame):
            raise TypeError(f"experiment_result['{key}'] must be a pandas DataFrame.")

    for key in (
        "training_summary",
        "validation_comparison_summary",
        "test_comparison_summary",
        "validation_diagnostics",
        "test_diagnostics",
    ):
        if not isinstance(experiment_result[key], dict):
            raise TypeError(f"experiment_result['{key}'] must be a dictionary.")

    for key in ("validation_diagnostics", "test_diagnostics"):
        diagnostics = experiment_result[key]
        if "final_weights" not in diagnostics:
            raise KeyError(f"experiment_result['{key}'] must contain final_weights.")
        if not isinstance(diagnostics["final_weights"], dict):
            raise TypeError(
                f"experiment_result['{key}']['final_weights'] must be a dictionary."
            )


def _save_dataframe(dataframe: pd.DataFrame, path: Path) -> str:
    dataframe.to_csv(path)
    return str(path)


def _save_dict(values: dict, path: Path) -> str:
    pd.DataFrame([values]).to_csv(path, index=False)
    return str(path)


def _flatten_diagnostics(diagnostics: dict) -> dict:
    flattened = {}
    for key, value in diagnostics.items():
        if key == "final_weights":
            for asset_name, weight in value.items():
                flattened[f"final_weight_{asset_name}"] = weight
        else:
            flattened[key] = value

    return flattened
