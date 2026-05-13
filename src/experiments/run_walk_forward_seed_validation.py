"""Walk-forward validation runner repeated across multiple TD3 random seeds.

This module combines explicit chronological folds with seed sensitivity. It
saves fold-seed experiment CSV outputs plus aggregate tables, but it does not
save models, replay buffers, plots, or reports.
"""

from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.data.walk_forward_split import build_walk_forward_datasets
from src.experiments.run_basic_experiment import summarize_metrics_table
from src.experiments.run_seed_sensitivity import DEFAULT_SEEDS
from src.experiments.run_walk_forward_validation import DEFAULT_WALK_FORWARD_FOLDS
from src.experiments.save_experiment_outputs import save_basic_experiment_outputs
from src.train.train_td3 import train_td3_on_datasets
from src.utils.config import load_config


def run_walk_forward_seed_validation(
    base_config_path: str,
    output_dir: str = "outputs/tables",
    experiment_name: str = "td3_walk_forward_seed_validation",
    folds: list[dict] | None = None,
    seeds: list[int] | None = None,
    episodes: int = 100,
    batch_size: int = 32,
    actor_learning_rate: float = 0.0005,
    critic_learning_rate: float = 0.0005,
) -> dict:
    """Run each walk-forward fold across multiple random seeds."""
    base_config = load_config(base_config_path)
    selected_folds = DEFAULT_WALK_FORWARD_FOLDS if folds is None else folds
    selected_seeds = DEFAULT_SEEDS if seeds is None else seeds
    run_output_dir = Path(output_dir) / experiment_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    fold_seed_outputs = {}
    for fold in selected_folds:
        datasets = build_walk_forward_datasets(base_config_path, fold)
        for seed in selected_seeds:
            fold_seed_config = _build_fold_seed_config(
                base_config=base_config,
                seed=seed,
                episodes=episodes,
                batch_size=batch_size,
                actor_learning_rate=actor_learning_rate,
                critic_learning_rate=critic_learning_rate,
            )
            raw_result = train_td3_on_datasets(datasets, fold_seed_config)
            experiment_result = _build_experiment_result(raw_result)
            fold_seed_key = f"{fold['fold_id']}_seed_{seed}"
            saved_paths = save_basic_experiment_outputs(
                experiment_result,
                output_dir=str(run_output_dir),
                experiment_name=(
                    f"fold_{fold['fold_id']}_seed_{seed}_{fold['description']}"
                ),
            )
            fold_seed_outputs[fold_seed_key] = {
                "experiment_result": experiment_result,
                "saved_paths": saved_paths,
                "raw_result": raw_result,
            }
            rows.append(
                _build_fold_seed_row(
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
    results_path = run_output_dir / "walk_forward_seed_results.csv"
    results.to_csv(results_path, index=False)

    summary = _build_summary(results)
    summary_path = run_output_dir / "walk_forward_seed_summary.csv"
    summary.to_csv(summary_path, index=False)

    by_fold_summary = _build_grouped_summary(
        results,
        group_columns=[
            "fold_id",
            "description",
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "test_start",
            "test_end",
        ],
    )
    by_fold_summary_path = run_output_dir / "walk_forward_seed_by_fold_summary.csv"
    by_fold_summary.to_csv(by_fold_summary_path, index=False)

    by_seed_summary = _build_grouped_summary(results, group_columns=["seed"])
    by_seed_summary_path = run_output_dir / "walk_forward_seed_by_seed_summary.csv"
    by_seed_summary.to_csv(by_seed_summary_path, index=False)

    return {
        "output_dir": str(run_output_dir),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "by_fold_summary_path": str(by_fold_summary_path),
        "by_seed_summary_path": str(by_seed_summary_path),
        "results": results,
        "summary": summary,
        "by_fold_summary": by_fold_summary,
        "by_seed_summary": by_seed_summary,
        "fold_seed_outputs": fold_seed_outputs,
    }


def _build_fold_seed_config(
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
    validation_metrics_table = raw_result["validation_comparison"]["metrics_table"]
    test_metrics_table = raw_result["test_comparison"]["metrics_table"]

    return {
        "training_summary": {
            "total_episodes": len(episode_logs),
            "final_episode": final_episode["episode"],
            "final_portfolio_value": final_episode["final_portfolio_value"],
            "final_total_reward": final_episode["total_reward"],
            "final_average_turnover": final_episode["average_turnover"],
            "final_average_transaction_cost": final_episode[
                "average_transaction_cost"
            ],
            "final_max_weight": final_episode["max_weight"],
            "final_cash_weight": final_episode["cash_weight"],
        },
        "validation_metrics_table": validation_metrics_table,
        "test_metrics_table": test_metrics_table,
        "validation_comparison_summary": summarize_metrics_table(
            validation_metrics_table
        ),
        "test_comparison_summary": summarize_metrics_table(test_metrics_table),
        "validation_diagnostics": raw_result["validation_evaluation"]["diagnostics"],
        "test_diagnostics": raw_result["test_evaluation"]["diagnostics"],
        "validation_policy_history": raw_result["validation_evaluation"].get(
            "policy_history"
        ),
        "test_policy_history": raw_result["test_evaluation"].get("policy_history"),
        "raw_result": raw_result,
    }


def _build_fold_seed_row(
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
        "test_average_transaction_cost": test_diagnostics[
            "average_transaction_cost"
        ],
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


def _build_grouped_summary(
    results: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for group_values, group in results.groupby(group_columns, sort=False, dropna=False):
        group_values = _normalize_group_values(group_values, group_columns)
        group_metadata = dict(zip(group_columns, group_values))
        summary_row = _build_summary(group).iloc[0].to_dict()
        rows.append({**group_metadata, **summary_row})

    return pd.DataFrame(rows)


def _normalize_group_values(group_values, group_columns: list[str]) -> tuple:
    if len(group_columns) == 1:
        if isinstance(group_values, tuple):
            return (group_values[0],)

        return (group_values,)

    if isinstance(group_values, tuple):
        return group_values

    return (group_values,)


def _build_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Build one robust performance summary row for result observations."""
    test_sharpe = results["test_agent_sharpe_ratio"]
    test_sortino = results["test_agent_sortino_ratio"]
    test_information_ratio = results[
        "test_agent_information_ratio_vs_equal_weight_rebalanced_net"
    ]
    test_capm_alpha = results["test_agent_capm_alpha_vs_SPY"]

    mean_test_sharpe = test_sharpe.mean()
    std_test_sharpe = _std_or_zero(test_sharpe)
    mean_test_sortino = test_sortino.mean()
    std_test_sortino = _std_or_zero(test_sortino)
    mean_test_information_ratio = test_information_ratio.mean()
    std_test_information_ratio = _std_or_zero(test_information_ratio)
    mean_test_capm_alpha = test_capm_alpha.mean()
    std_test_capm_alpha = _std_or_zero(test_capm_alpha)
    worst_test_sharpe = test_sharpe.min()

    summary = {
        "n_observations": len(results),
        "n_folds": results["fold_id"].nunique(),
        "n_seeds": results["seed"].nunique(),
        "mean_test_agent_sharpe": mean_test_sharpe,
        "std_test_agent_sharpe": std_test_sharpe,
        "min_test_agent_sharpe": test_sharpe.min(),
        "max_test_agent_sharpe": test_sharpe.max(),
        "robust_test_agent_sharpe_score_05": (
            mean_test_sharpe - 0.5 * std_test_sharpe
        ),
        "robust_test_agent_sharpe_score_10": (
            mean_test_sharpe - 1.0 * std_test_sharpe
        ),
        "mean_test_agent_sortino": mean_test_sortino,
        "std_test_agent_sortino": std_test_sortino,
        "robust_test_agent_sortino_score_05": (
            mean_test_sortino - 0.5 * std_test_sortino
        ),
        "mean_test_agent_information_ratio_vs_equal_weight_rebalanced_net": (
            mean_test_information_ratio
        ),
        "std_test_agent_information_ratio_vs_equal_weight_rebalanced_net": (
            std_test_information_ratio
        ),
        "robust_test_agent_information_ratio_score_05": (
            mean_test_information_ratio - 0.5 * std_test_information_ratio
        ),
        "mean_test_agent_capm_alpha_vs_SPY": mean_test_capm_alpha,
        "std_test_agent_capm_alpha_vs_SPY": std_test_capm_alpha,
        "robust_test_agent_capm_alpha_score_05": (
            mean_test_capm_alpha - 0.5 * std_test_capm_alpha
        ),
        "mean_test_agent_cumulative_return": results[
            "test_agent_cumulative_return"
        ].mean(),
        "mean_test_agent_max_drawdown": results["test_agent_max_drawdown"].mean(),
        "worst_test_agent_sharpe": worst_test_sharpe,
        "worst_test_agent_cumulative_return": results[
            "test_agent_cumulative_return"
        ].min(),
        "worst_test_agent_max_drawdown": results["test_agent_max_drawdown"].min(),
        "positive_sharpe_rate": (test_sharpe > 0).mean(),
        "positive_sortino_rate": (test_sortino > 0).mean(),
        "positive_capm_alpha_rate": (test_capm_alpha > 0).mean(),
        "positive_information_ratio_rate": (test_information_ratio > 0).mean(),
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


def _std_or_zero(series: pd.Series) -> float:
    std = series.std()
    if pd.isna(std):
        return 0.0

    return std
