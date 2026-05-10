"""Plot utilities for hyperparameter grid experiment results."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "experiment_id",
    "description",
    "test_agent_cumulative_return",
    "test_agent_sharpe_ratio",
    "test_agent_max_drawdown",
    "test_average_turnover",
    "test_average_effective_number_of_assets",
    "test_final_max_weight",
}


def plot_hyperparameter_grid_results(
    ranking_csv_path: str,
    output_dir: str = "outputs/figures/hyperparameter_grid",
) -> dict:
    """Create PDF plots from a hyperparameter grid ranking CSV."""
    ranking = pd.read_csv(ranking_csv_path)
    _validate_required_columns(ranking)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "sharpe_by_experiment": str(output_path / "fig_sharpe_by_experiment.pdf"),
        "effective_assets_vs_sharpe": str(
            output_path / "fig_effective_assets_vs_sharpe.pdf"
        ),
        "turnover_vs_sharpe": str(output_path / "fig_turnover_vs_sharpe.pdf"),
        "drawdown_vs_sharpe": str(output_path / "fig_drawdown_vs_sharpe.pdf"),
        "return_vs_sharpe": str(output_path / "fig_return_vs_sharpe.pdf"),
    }

    _plot_sharpe_by_experiment(ranking, paths["sharpe_by_experiment"])
    _plot_scatter(
        ranking,
        x_column="test_average_effective_number_of_assets",
        y_column="test_agent_sharpe_ratio",
        x_label="Average effective number of assets",
        y_label="Test Sharpe ratio",
        title="Portfolio concentration and risk-adjusted performance",
        path=paths["effective_assets_vs_sharpe"],
    )
    _plot_scatter(
        ranking,
        x_column="test_average_turnover",
        y_column="test_agent_sharpe_ratio",
        x_label="Average turnover",
        y_label="Test Sharpe ratio",
        title="Turnover and risk-adjusted performance",
        path=paths["turnover_vs_sharpe"],
    )
    _plot_scatter(
        ranking,
        x_column="test_agent_max_drawdown",
        y_column="test_agent_sharpe_ratio",
        x_label="Test maximum drawdown",
        y_label="Test Sharpe ratio",
        title="Drawdown and risk-adjusted performance",
        path=paths["drawdown_vs_sharpe"],
    )
    _plot_scatter(
        ranking,
        x_column="test_agent_cumulative_return",
        y_column="test_agent_sharpe_ratio",
        x_label="Test cumulative return",
        y_label="Test Sharpe ratio",
        title="Return and risk-adjusted performance",
        path=paths["return_vs_sharpe"],
    )

    return paths


def _validate_required_columns(ranking: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(ranking.columns)
    if missing_columns:
        raise KeyError(f"ranking CSV is missing required columns: {sorted(missing_columns)}")


def _plot_sharpe_by_experiment(ranking: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ranking["experiment_id"], ranking["test_agent_sharpe_ratio"])
    ax.set_xlabel("Test Sharpe ratio")
    ax.set_ylabel("Experiment")
    ax.set_title("Test Sharpe by hyperparameter experiment")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _plot_scatter(
    ranking: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ranking[x_column], ranking[y_column])
    for _, row in ranking.iterrows():
        ax.annotate(
            str(row["experiment_id"]),
            (row[x_column], row[y_column]),
            textcoords="offset points",
            xytext=(5, 5),
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)
