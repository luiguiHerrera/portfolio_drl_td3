"""True chronological walk-forward validation runner for TD3 experiments."""

from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.data.walk_forward_split import build_walk_forward_datasets
from src.experiments.run_basic_experiment import summarize_metrics_table
from src.experiments.save_experiment_outputs import save_basic_experiment_outputs
from src.train.train_td3 import train_td3_on_datasets
from src.utils.config import load_config


DEFAULT_WALK_FORWARD_FOLDS = [
    {
        "fold_id": "F1",
        "description": "train_2021_2022_validate_2023H1_test_2023H2",
        "train_start": "2021-01-01",
        "train_end": "2022-12-31",
        "validation_start": "2023-01-01",
        "validation_end": "2023-06-30",
        "test_start": "2023-07-01",
        "test_end": "2023-12-31",
    },
    {
        "fold_id": "F2",
        "description": "train_2021H2_2023H1_validate_2023H2_test_2024H1",
        "train_start": "2021-07-01",
        "train_end": "2023-06-30",
        "validation_start": "2023-07-01",
        "validation_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-06-30",
    },
    {
        "fold_id": "F3",
        "description": "train_2022_2023_validate_2024H1_test_2024H2",
        "train_start": "2022-01-01",
        "train_end": "2023-12-31",
        "validation_start": "2024-01-01",
        "validation_end": "2024-06-30",
        "test_start": "2024-07-01",
        "test_end": "2024-12-31",
    },
]


def run_walk_forward_validation(
    base_config_path: str,
    output_dir: str = "outputs/tables",
    experiment_name: str = "td3_walk_forward_validation",
    folds: list[dict] | None = None,
    seed: int = 42,
    episodes: int = 100,
    batch_size: int = 32,
    actor_learning_rate: float = 0.0005,
    critic_learning_rate: float = 0.0005,
) -> dict:
    """Run explicit train/validation/test date folds and save aggregate tables."""
    base_config = load_config(base_config_path)
    selected_folds = DEFAULT_WALK_FORWARD_FOLDS if folds is None else folds
    walk_forward_output_dir = Path(output_dir) / experiment_name
    walk_forward_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    fold_outputs = {}
    for fold in selected_folds:
        fold_config = _build_fold_config(
            base_config=base_config,
            seed=seed,
            episodes=episodes,
            batch_size=batch_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
        )
        datasets = build_walk_forward_datasets(base_config_path, fold)
        raw_result = train_td3_on_datasets(datasets, fold_config)
        experiment_result = _build_experiment_result(raw_result)
        saved_paths = save_basic_experiment_outputs(
            experiment_result,
            output_dir=str(walk_forward_output_dir),
            experiment_name=f"fold_{fold['fold_id']}_{fold['description']}",
        )
        fold_output = {
            "experiment_result": experiment_result,
            "saved_paths": saved_paths,
            "raw_result": raw_result,
        }
        fold_outputs[fold["fold_id"]] = fold_output
        rows.append(
            _build_fold_row(
                fold=fold,
                seed=seed,
                episodes=episodes,
                batch_size=batch_size,
                actor_learning_rate=actor_learning_rate,
                critic_learning_rate=critic_learning_rate,
                experiment_result=experiment_result,
            )
        )

    results = pd.DataFrame(rows)
    results_path = walk_forward_output_dir / "walk_forward_results.csv"
    results.to_csv(results_path, index=False)

    summary = _build_summary(results)
    summary_path = walk_forward_output_dir / "walk_forward_summary.csv"
    summary.to_csv(summary_path, index=False)

    return {
        "output_dir": str(walk_forward_output_dir),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "results": results,
        "summary": summary,
        "fold_outputs": fold_outputs,
    }


def _build_fold_config(
    base_config: dict,
    seed: int,
    episodes: int,
    batch_size: int,
    actor_learning_rate: float,
    critic_learning_rate: float,
) -> dict:
    config = deepcopy(base_config)
    config["training"]["seed"] = seed
    config["training"]["episodes"] = episodes
    config["td3"]["batch_size"] = batch_size
    config["td3"]["actor_learning_rate"] = actor_learning_rate
    config["td3"]["critic_learning_rate"] = critic_learning_rate

    return config


def _build_experiment_result(raw_result: dict) -> dict:
    episode_logs = raw_result["episode_logs"]
    final_episode = episode_logs[-1]
    training_summary = {
        "total_episodes": len(episode_logs),
        "final_episode": final_episode["episode"],
        "final_portfolio_value": final_episode["final_portfolio_value"],
        "final_total_reward": final_episode["total_reward"],
        "final_average_turnover": final_episode["average_turnover"],
        "final_average_transaction_cost": final_episode["average_transaction_cost"],
        "final_max_weight": final_episode["max_weight"],
        "final_cash_weight": final_episode["cash_weight"],
    }
    validation_metrics_table = raw_result["validation_comparison"]["metrics_table"]
    test_metrics_table = raw_result["test_comparison"]["metrics_table"]

    return {
        "training_summary": training_summary,
        "validation_metrics_table": validation_metrics_table,
        "test_metrics_table": test_metrics_table,
        "validation_comparison_summary": summarize_metrics_table(validation_metrics_table),
        "test_comparison_summary": summarize_metrics_table(test_metrics_table),
        "validation_diagnostics": raw_result["validation_evaluation"]["diagnostics"],
        "test_diagnostics": raw_result["test_evaluation"]["diagnostics"],
        "validation_policy_history": raw_result["validation_evaluation"].get(
            "policy_history"
        ),
        "test_policy_history": raw_result["test_evaluation"].get("policy_history"),
        "raw_result": raw_result,
    }


def _build_fold_row(
    fold: dict,
    seed: int,
    episodes: int,
    batch_size: int,
    actor_learning_rate: float,
    critic_learning_rate: float,
    experiment_result: dict,
) -> dict:
    validation_metrics = experiment_result["validation_metrics_table"]
    test_metrics = experiment_result["test_metrics_table"]
    validation_summary = experiment_result["validation_comparison_summary"]
    test_summary = experiment_result["test_comparison_summary"]
    test_diagnostics = experiment_result["test_diagnostics"]

    return {
        **fold,
        "seed": seed,
        "episodes": episodes,
        "batch_size": batch_size,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        **_agent_metrics("validation", validation_metrics),
        **_agent_metrics("test", test_metrics),
        **_comparison_summary("validation", validation_summary),
        **_comparison_summary("test", test_summary),
        "test_average_turnover": test_diagnostics["average_turnover"],
        "test_average_effective_number_of_assets": test_diagnostics[
            "average_effective_number_of_assets"
        ],
        "test_final_max_weight": test_diagnostics["final_max_weight"],
        "test_average_transaction_cost": test_diagnostics["average_transaction_cost"],
    }


def _agent_metrics(prefix: str, metrics_table: pd.DataFrame) -> dict:
    columns = (
        "cumulative_return",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "information_ratio_vs_equal_weight_rebalanced_net",
        "capm_beta_vs_SPY",
        "capm_alpha_vs_SPY",
        "max_drawdown",
    )
    return {
        f"{prefix}_agent_{column}": metrics_table.loc["agent", column]
        for column in columns
    }


def _comparison_summary(prefix: str, summary: dict) -> dict:
    keys = (
        "best_policy_by_sharpe",
        "agent_rank_by_sharpe",
        "best_individual_buyhold_by_sharpe",
        "agent_vs_best_individual_buyhold_sharpe_diff",
    )
    return {f"{prefix}_{key}": summary[key] for key in keys}


def _build_summary(results: pd.DataFrame) -> pd.DataFrame:
    mean_test_sharpe = results["test_agent_sharpe_ratio"].mean()
    std_test_sharpe = results["test_agent_sharpe_ratio"].std()
    worst_test_sharpe = results["test_agent_sharpe_ratio"].min()

    summary = {
        "n_folds": len(results),
        "mean_test_agent_sharpe": mean_test_sharpe,
        "std_test_agent_sharpe": std_test_sharpe,
        "min_test_agent_sharpe": worst_test_sharpe,
        "max_test_agent_sharpe": results["test_agent_sharpe_ratio"].max(),
        "robust_test_agent_sharpe_score_05": mean_test_sharpe - 0.5 * std_test_sharpe,
        "robust_test_agent_sharpe_score_10": mean_test_sharpe - 1.0 * std_test_sharpe,
        "mean_test_agent_sortino": results["test_agent_sortino_ratio"].mean(),
        "mean_test_agent_information_ratio_vs_equal_weight_rebalanced_net": results[
            "test_agent_information_ratio_vs_equal_weight_rebalanced_net"
        ].mean(),
        "mean_test_agent_capm_alpha_vs_SPY": results["test_agent_capm_alpha_vs_SPY"].mean(),
        "mean_test_agent_cumulative_return": results["test_agent_cumulative_return"].mean(),
        "mean_test_agent_max_drawdown": results["test_agent_max_drawdown"].mean(),
        "worst_test_agent_sharpe": worst_test_sharpe,
        "worst_test_agent_cumulative_return": results["test_agent_cumulative_return"].min(),
        "worst_test_agent_max_drawdown": results["test_agent_max_drawdown"].min(),
        "positive_sharpe_rate": (results["test_agent_sharpe_ratio"] > 0).mean(),
        "positive_sortino_rate": (results["test_agent_sortino_ratio"] > 0).mean(),
        "positive_capm_alpha_rate": (results["test_agent_capm_alpha_vs_SPY"] > 0).mean(),
        "positive_information_ratio_rate": (
            results["test_agent_information_ratio_vs_equal_weight_rebalanced_net"] > 0
        ).mean(),
        "win_rate_best_policy_agent": (
            results["test_best_policy_by_sharpe"] == "agent"
        ).mean(),
        "win_rate_vs_best_individual_buyhold_by_sharpe": (
            results["test_agent_vs_best_individual_buyhold_sharpe_diff"] > 0
        ).mean(),
        "mean_test_average_turnover": results["test_average_turnover"].mean(),
        "mean_test_average_effective_number_of_assets": results[
            "test_average_effective_number_of_assets"
        ].mean(),
    }

    return pd.DataFrame([summary])
