"""Hyperparameter grid runner for basic TD3 portfolio experiments.

This module runs a small set of importable experiment configurations, saves
selected CSV outputs for each experiment, and creates aggregate CSV summaries.
It does not save models, replay buffers, raw results, plots, or reports.
"""

from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from src.experiments.run_and_save_basic_experiment import run_and_save_basic_experiment


DEFAULT_HYPERPARAMETER_EXPERIMENTS = [
    {
        "experiment_id": "A",
        "description": "baseline_short_smoke",
        "episodes": 3,
        "batch_size": 32,
        "actor_learning_rate": 0.0003,
        "critic_learning_rate": 0.0003,
    },
    {
        "experiment_id": "B",
        "description": "baseline_50_episodes",
        "episodes": 50,
        "batch_size": 32,
        "actor_learning_rate": 0.0003,
        "critic_learning_rate": 0.0003,
    },
    {
        "experiment_id": "C",
        "description": "baseline_100_episodes",
        "episodes": 100,
        "batch_size": 32,
        "actor_learning_rate": 0.0003,
        "critic_learning_rate": 0.0003,
    },
    {
        "experiment_id": "D",
        "description": "lower_learning_rate",
        "episodes": 100,
        "batch_size": 32,
        "actor_learning_rate": 0.0001,
        "critic_learning_rate": 0.0001,
    },
    {
        "experiment_id": "E",
        "description": "larger_batch_size",
        "episodes": 100,
        "batch_size": 64,
        "actor_learning_rate": 0.0003,
        "critic_learning_rate": 0.0003,
    },
    {
        "experiment_id": "F",
        "description": "conservative_actor_learning_rate",
        "episodes": 100,
        "batch_size": 32,
        "actor_learning_rate": 0.0001,
        "critic_learning_rate": 0.0003,
    },
    {
        "experiment_id": "G",
        "description": "longer_training_200_episodes",
        "episodes": 200,
        "batch_size": 32,
        "actor_learning_rate": 0.0003,
        "critic_learning_rate": 0.0003,
    },
    {
        "experiment_id": "H",
        "description": "higher_learning_rate",
        "episodes": 100,
        "batch_size": 32,
        "actor_learning_rate": 0.0005,
        "critic_learning_rate": 0.0005,
    },
]


def run_hyperparameter_grid(
    base_config_path: str,
    output_dir: str = "outputs/tables",
    grid_name: str = "td3_hyperparameter_grid",
    experiments: list[dict] | None = None,
) -> dict:
    """Run a hyperparameter grid and save aggregate CSV summaries."""
    base_config = _load_yaml_config(base_config_path)
    selected_experiments = experiments or DEFAULT_HYPERPARAMETER_EXPERIMENTS
    grid_output_dir = Path(output_dir) / grid_name
    configs_dir = grid_output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    experiment_outputs = {}
    for experiment in selected_experiments:
        experiment_name = _experiment_name(experiment)
        experiment_config = _build_experiment_config(base_config, experiment)
        experiment_config_path = configs_dir / f"{experiment_name}.yaml"
        _write_yaml_config(experiment_config, experiment_config_path)

        experiment_output = run_and_save_basic_experiment(
            config_path=str(experiment_config_path),
            output_dir=str(grid_output_dir),
            experiment_name=experiment_name,
        )
        experiment_outputs[experiment["experiment_id"]] = experiment_output
        rows.append(
            _build_aggregate_row(
                experiment,
                experiment_name,
                experiment_config_path,
                experiment_output,
            )
        )

    aggregate_results = pd.DataFrame(rows)
    aggregate_results_path = grid_output_dir / "hyperparameter_grid_results.csv"
    aggregate_results.to_csv(aggregate_results_path, index=False)

    ranking_results = aggregate_results.sort_values(
        by=[
            "test_agent_sharpe_ratio",
            "test_agent_max_drawdown",
            "test_agent_cumulative_return",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranking_results_path = grid_output_dir / "hyperparameter_grid_ranking.csv"
    ranking_results.to_csv(ranking_results_path, index=False)

    return {
        "grid_output_dir": str(grid_output_dir),
        "aggregate_results_path": str(aggregate_results_path),
        "ranking_results_path": str(ranking_results_path),
        "aggregate_results": aggregate_results,
        "ranking_results": ranking_results,
        "experiment_outputs": experiment_outputs,
    }


def _load_yaml_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise TypeError("base config must be a YAML mapping.")

    return config


def _write_yaml_config(config: dict, config_path: Path) -> None:
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def _build_experiment_config(base_config: dict, experiment: dict) -> dict:
    config = deepcopy(base_config)
    config["training"]["episodes"] = experiment["episodes"]
    config["td3"]["batch_size"] = experiment["batch_size"]
    config["td3"]["actor_learning_rate"] = experiment["actor_learning_rate"]
    config["td3"]["critic_learning_rate"] = experiment["critic_learning_rate"]

    return config


def _experiment_name(experiment: dict) -> str:
    description = str(experiment["description"]).replace(" ", "_")
    return f"experiment_{experiment['experiment_id']}_{description}"


def _build_aggregate_row(
    experiment: dict,
    experiment_name: str,
    experiment_config_path: Path,
    experiment_output: dict,
) -> dict:
    experiment_result = experiment_output["experiment_result"]
    saved_paths = experiment_output["saved_paths"]
    validation_metrics = experiment_result["validation_metrics_table"]
    test_metrics = experiment_result["test_metrics_table"]
    validation_summary = experiment_result["validation_comparison_summary"]
    test_summary = experiment_result["test_comparison_summary"]
    validation_diagnostics = experiment_result["validation_diagnostics"]
    test_diagnostics = experiment_result["test_diagnostics"]
    training_summary = experiment_result["training_summary"]

    return {
        "experiment_id": experiment["experiment_id"],
        "description": experiment["description"],
        "episodes": experiment["episodes"],
        "batch_size": experiment["batch_size"],
        "actor_learning_rate": experiment["actor_learning_rate"],
        "critic_learning_rate": experiment["critic_learning_rate"],
        "experiment_name": experiment_name,
        "config_path": str(experiment_config_path),
        "output_dir": saved_paths["output_dir"],
        "training_final_portfolio_value": training_summary["final_portfolio_value"],
        "training_final_total_reward": training_summary["final_total_reward"],
        "training_final_average_turnover": training_summary["final_average_turnover"],
        "training_final_average_transaction_cost": training_summary[
            "final_average_transaction_cost"
        ],
        "training_final_max_weight": training_summary["final_max_weight"],
        "training_final_cash_weight": training_summary["final_cash_weight"],
        **_policy_metrics("validation_agent", validation_metrics, "agent"),
        **_policy_metrics("test_agent", test_metrics, "agent"),
        **_comparison_summary("validation", validation_summary),
        **_comparison_summary("test", test_summary),
        **_allocation_diagnostics("validation", validation_diagnostics),
        **_allocation_diagnostics("test", test_diagnostics),
        **_benchmark_test_metrics(test_metrics),
    }


def _policy_metrics(prefix: str, metrics_table: pd.DataFrame, policy_name: str) -> dict:
    return {
        f"{prefix}_cumulative_return": metrics_table.loc[policy_name, "cumulative_return"],
        f"{prefix}_annualized_return": metrics_table.loc[policy_name, "annualized_return"],
        f"{prefix}_annualized_volatility": metrics_table.loc[
            policy_name,
            "annualized_volatility",
        ],
        f"{prefix}_sharpe_ratio": metrics_table.loc[policy_name, "sharpe_ratio"],
        f"{prefix}_max_drawdown": metrics_table.loc[policy_name, "max_drawdown"],
    }


def _comparison_summary(prefix: str, summary: dict) -> dict:
    return {
        f"{prefix}_best_policy_by_sharpe": summary["best_policy_by_sharpe"],
        f"{prefix}_best_sharpe_ratio": summary["best_sharpe_ratio"],
        f"{prefix}_agent_rank_by_sharpe": summary["agent_rank_by_sharpe"],
        f"{prefix}_agent_vs_equal_weight_rebalanced_net_sharpe_diff": summary[
            "agent_vs_equal_weight_rebalanced_net_sharpe_diff"
        ],
        f"{prefix}_agent_vs_buy_and_hold_sharpe_diff": summary[
            "agent_vs_buy_and_hold_sharpe_diff"
        ],
        f"{prefix}_agent_vs_equal_weight_rebalanced_net_cumulative_return_diff": summary[
            "agent_vs_equal_weight_rebalanced_net_cumulative_return_diff"
        ],
        f"{prefix}_agent_vs_buy_and_hold_cumulative_return_diff": summary[
            "agent_vs_buy_and_hold_cumulative_return_diff"
        ],
    }


def _allocation_diagnostics(prefix: str, diagnostics: dict) -> dict:
    keys = (
        "average_max_weight",
        "final_max_weight",
        "average_cash_weight",
        "final_cash_weight",
        "average_herfindahl_index",
        "final_herfindahl_index",
        "average_effective_number_of_assets",
        "final_effective_number_of_assets",
        "average_entropy",
        "final_entropy",
        "average_turnover",
        "final_turnover",
        "average_transaction_cost",
        "final_transaction_cost",
    )

    return {f"{prefix}_{key}": diagnostics[key] for key in keys}


def _benchmark_test_metrics(test_metrics: pd.DataFrame) -> dict:
    return {
        "test_equal_weight_gross_cumulative_return": test_metrics.loc[
            "equal_weight_gross",
            "cumulative_return",
        ],
        "test_equal_weight_gross_sharpe_ratio": test_metrics.loc[
            "equal_weight_gross",
            "sharpe_ratio",
        ],
        "test_equal_weight_rebalanced_net_cumulative_return": test_metrics.loc[
            "equal_weight_rebalanced_net",
            "cumulative_return",
        ],
        "test_equal_weight_rebalanced_net_sharpe_ratio": test_metrics.loc[
            "equal_weight_rebalanced_net",
            "sharpe_ratio",
        ],
        "test_buy_and_hold_cumulative_return": test_metrics.loc[
            "buy_and_hold",
            "cumulative_return",
        ],
        "test_buy_and_hold_sharpe_ratio": test_metrics.loc[
            "buy_and_hold",
            "sharpe_ratio",
        ],
    }
