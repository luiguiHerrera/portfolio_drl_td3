"""Hyperparameter by seed robustness runner for TD3 portfolio experiments.

This module evaluates multiple hyperparameter configurations across multiple
random seeds by delegating each configuration to the seed sensitivity runner.
It saves one aggregate summary table and one robust ranking table.
"""

from pathlib import Path

import pandas as pd

from src.experiments.run_hyperparameter_grid import DEFAULT_HYPERPARAMETER_EXPERIMENTS
from src.experiments.run_seed_sensitivity import DEFAULT_SEEDS, run_seed_sensitivity


SUMMARY_COLUMNS = (
    "n_seeds",
    "mean_test_agent_sharpe",
    "std_test_agent_sharpe",
    "robust_test_agent_sharpe_score_05",
    "robust_test_agent_sharpe_score_10",
    "mean_test_agent_sortino",
    "std_test_agent_sortino",
    "robust_test_agent_sortino_score_05",
    "mean_test_agent_information_ratio_vs_equal_weight_rebalanced_net",
    "std_test_agent_information_ratio_vs_equal_weight_rebalanced_net",
    "robust_test_agent_information_ratio_score_05",
    "mean_test_agent_capm_beta_vs_SPY",
    "mean_test_agent_capm_alpha_vs_SPY",
    "std_test_agent_capm_alpha_vs_SPY",
    "robust_test_agent_capm_alpha_score_05",
    "mean_test_agent_cumulative_return",
    "mean_test_agent_max_drawdown",
    "worst_test_agent_sharpe",
    "worst_test_agent_cumulative_return",
    "worst_test_agent_max_drawdown",
    "positive_sharpe_rate",
    "positive_sortino_rate",
    "positive_capm_alpha_rate",
    "positive_information_ratio_rate",
    "win_rate_best_policy_agent",
    "win_rate_vs_best_individual_buyhold_by_sharpe",
    "mean_test_average_turnover",
    "mean_test_average_effective_number_of_assets",
    "mean_minus_worst_sharpe_gap",
)


RANKING_COLUMNS = (
    "robust_test_agent_sharpe_score_05",
    "robust_test_agent_information_ratio_score_05",
    "robust_test_agent_capm_alpha_score_05",
    "positive_sharpe_rate",
    "mean_test_agent_max_drawdown",
)


def run_hyperparameter_seed_grid(
    base_config_path: str,
    output_dir: str = "outputs/tables",
    grid_name: str = "td3_hyperparameter_seed_grid",
    experiments: list[dict] | None = None,
    seeds: list[int] | None = None,
) -> dict:
    """Run hyperparameter configurations across random seeds and rank robustness."""
    selected_experiments = experiments or DEFAULT_HYPERPARAMETER_EXPERIMENTS
    selected_seeds = DEFAULT_SEEDS if seeds is None else seeds
    grid_output_dir = Path(output_dir) / grid_name
    grid_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    experiment_outputs = {}
    for experiment in selected_experiments:
        experiment_name = f"{_experiment_name(experiment)}_seed_sensitivity"
        experiment_output = run_seed_sensitivity(
            base_config_path=base_config_path,
            output_dir=str(grid_output_dir),
            experiment_name=experiment_name,
            seeds=selected_seeds,
            episodes=experiment["episodes"],
            batch_size=experiment["batch_size"],
            actor_learning_rate=experiment["actor_learning_rate"],
            critic_learning_rate=experiment["critic_learning_rate"],
        )
        experiment_outputs[experiment["experiment_id"]] = experiment_output
        rows.append(_build_aggregate_row(experiment, experiment_output))

    aggregate_results = pd.DataFrame(rows)
    aggregate_results_path = grid_output_dir / "hyperparameter_seed_grid_results.csv"
    aggregate_results.to_csv(aggregate_results_path, index=False)

    ranking_results = aggregate_results.sort_values(
        by=list(RANKING_COLUMNS),
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    ranking_results_path = grid_output_dir / "hyperparameter_seed_grid_ranking.csv"
    ranking_results.to_csv(ranking_results_path, index=False)

    return {
        "grid_output_dir": str(grid_output_dir),
        "aggregate_results_path": str(aggregate_results_path),
        "ranking_results_path": str(ranking_results_path),
        "aggregate_results": aggregate_results,
        "ranking_results": ranking_results,
        "experiment_outputs": experiment_outputs,
    }


def _experiment_name(experiment: dict) -> str:
    description = str(experiment["description"]).replace(" ", "_")
    return f"experiment_{experiment['experiment_id']}_{description}"


def _build_aggregate_row(experiment: dict, experiment_output: dict) -> dict:
    summary = experiment_output["summary"].iloc[0]
    row = {
        "experiment_id": experiment["experiment_id"],
        "description": experiment["description"],
        "episodes": experiment["episodes"],
        "batch_size": experiment["batch_size"],
        "actor_learning_rate": experiment["actor_learning_rate"],
        "critic_learning_rate": experiment["critic_learning_rate"],
        "seed_sensitivity_output_dir": experiment_output["output_dir"],
        "seed_sensitivity_results_path": experiment_output["results_path"],
        "seed_sensitivity_summary_path": experiment_output["summary_path"],
    }
    row.update({column: summary[column] for column in SUMMARY_COLUMNS})

    return row
