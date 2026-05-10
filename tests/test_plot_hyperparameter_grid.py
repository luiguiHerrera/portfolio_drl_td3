"""Tests for hyperparameter grid plotting utilities."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.visualization.plot_hyperparameter_grid import (
    plot_hyperparameter_grid_results,
)


class PlotHyperparameterGridTests(unittest.TestCase):
    def test_plot_hyperparameter_grid_results_returns_expected_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ranking_csv_path = self._write_ranking_csv(temp_dir)
            output_dir = Path(temp_dir) / "figures"

            paths = plot_hyperparameter_grid_results(
                str(ranking_csv_path),
                output_dir=str(output_dir),
            )

        self.assertEqual(
            set(paths.keys()),
            {
                "sharpe_by_experiment",
                "effective_assets_vs_sharpe",
                "turnover_vs_sharpe",
                "drawdown_vs_sharpe",
                "return_vs_sharpe",
            },
        )

    def test_plot_hyperparameter_grid_results_creates_all_pdf_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ranking_csv_path = self._write_ranking_csv(temp_dir)
            output_dir = Path(temp_dir) / "figures"

            paths = plot_hyperparameter_grid_results(
                str(ranking_csv_path),
                output_dir=str(output_dir),
            )

            for path in paths.values():
                self.assertTrue(Path(path).is_file())
                self.assertEqual(Path(path).suffix, ".pdf")

    def test_missing_required_column_raises_key_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ranking = self._ranking_frame().drop(columns=["test_agent_sharpe_ratio"])
            ranking_csv_path = Path(temp_dir) / "ranking.csv"
            ranking.to_csv(ranking_csv_path, index=False)

            with self.assertRaises(KeyError):
                plot_hyperparameter_grid_results(
                    str(ranking_csv_path),
                    output_dir=str(Path(temp_dir) / "figures"),
                )

    @staticmethod
    def _write_ranking_csv(temp_dir: str) -> Path:
        ranking_csv_path = Path(temp_dir) / "ranking.csv"
        PlotHyperparameterGridTests._ranking_frame().to_csv(ranking_csv_path, index=False)
        return ranking_csv_path

    @staticmethod
    def _ranking_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "experiment_id": ["B", "A", "C"],
                "description": ["high_sharpe", "baseline", "low_turnover"],
                "test_agent_cumulative_return": [0.04, 0.02, 0.03],
                "test_agent_sharpe_ratio": [1.4, 0.9, 1.1],
                "test_agent_max_drawdown": [-0.02, -0.05, -0.03],
                "test_average_turnover": [0.25, 0.15, 0.10],
                "test_average_effective_number_of_assets": [3.5, 4.0, 3.8],
                "test_final_max_weight": [0.45, 0.35, 0.40],
            }
        )


if __name__ == "__main__":
    unittest.main()
