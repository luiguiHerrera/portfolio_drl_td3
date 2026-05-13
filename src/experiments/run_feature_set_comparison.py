"""Compare configured feature sets through walk-forward seed validation.

This runner creates temporary feature-set-specific configs and delegates the
actual training/evaluation work to the existing walk-forward seed validation
runner. It does not alter reward, environment, agent, benchmark, or feature
formula logic.
"""

from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from src.experiments.run_seed_sensitivity import DEFAULT_SEEDS
from src.experiments.run_walk_forward_seed_validation import (
    run_walk_forward_seed_validation,
)
from src.experiments.run_walk_forward_validation import DEFAULT_WALK_FORWARD_FOLDS
from src.utils.config import load_config


DEFAULT_FEATURE_SETS = [
    {
        "feature_set_id": "V1",
        "description": "default_return_features",
        "features": None,
    },
    {
        "feature_set_id": "V2",
        "description": "return_risk_regime_features",
        "features": {
            "version": "v2",
            "market_asset": "SPY",
            "short_window": 4,
            "long_window": 12,
            "ewma_span": 12,
        },
    },
    {
        "feature_set_id": "V3_macro",
        "description": "v2_plus_local_macro_features",
        "features": {
            "version": "v3",
            "market_asset": "SPY",
            "short_window": 4,
            "long_window": 12,
            "ewma_span": 12,
            "macro_path": "tests/fixtures/macro_weekly_2020_2024_test.csv",
            "macro_date_column": "date",
        },
    },
]

SUMMARY_COLUMNS = [
    "n_observations",
    "n_folds",
    "n_seeds",
    "mean_test_agent_sharpe",
    "std_test_agent_sharpe",
    "robust_test_agent_sharpe_score_05",
    "robust_test_agent_sharpe_score_10",
    "mean_test_agent_sortino",
    "robust_test_agent_sortino_score_05",
    "mean_test_agent_information_ratio_vs_equal_weight_rebalanced_net",
    "robust_test_agent_information_ratio_score_05",
    "mean_test_agent_capm_alpha_vs_SPY",
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
]

RANKING_COLUMNS = [
    "robust_test_agent_sharpe_score_05",
    "robust_test_agent_information_ratio_score_05",
    "robust_test_agent_capm_alpha_score_05",
    "positive_sharpe_rate",
]


def run_feature_set_comparison(
    base_config_path: str,
    output_dir: str = "outputs/tables",
    experiment_name: str = "td3_feature_set_comparison",
    feature_sets: list[dict] | None = None,
    folds: list[dict] | None = None,
    seeds: list[int] | None = None,
    episodes: int = 100,
    batch_size: int = 32,
    actor_learning_rate: float = 0.0005,
    critic_learning_rate: float = 0.0005,
) -> dict:
    """Compare feature sets under identical folds, seeds, and TD3 settings."""
    base_config = load_config(base_config_path)
    selected_feature_sets = DEFAULT_FEATURE_SETS if feature_sets is None else feature_sets
    selected_folds = DEFAULT_WALK_FORWARD_FOLDS if folds is None else folds
    selected_seeds = DEFAULT_SEEDS if seeds is None else seeds
    comparison_output_dir = Path(output_dir) / experiment_name
    configs_dir = comparison_output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    feature_set_outputs = {}
    for feature_set in selected_feature_sets:
        feature_set_id = feature_set["feature_set_id"]
        feature_config = _build_feature_set_config(
            base_config=base_config,
            feature_set=feature_set,
            episodes=episodes,
            batch_size=batch_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
        )
        config_path = configs_dir / f"{feature_set_id}.yaml"
        _write_yaml_config(feature_config, config_path)

        feature_set_output = run_walk_forward_seed_validation(
            base_config_path=str(config_path),
            output_dir=str(comparison_output_dir),
            experiment_name=feature_set_id,
            folds=selected_folds,
            seeds=selected_seeds,
            episodes=episodes,
            batch_size=batch_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
        )
        feature_set_outputs[feature_set_id] = feature_set_output
        rows.append(
            _build_feature_set_row(
                feature_set=feature_set,
                config_path=config_path,
                feature_set_output=feature_set_output,
            )
        )

    results = pd.DataFrame(rows)
    results_path = comparison_output_dir / "feature_set_comparison_results.csv"
    results.to_csv(results_path, index=False)

    ranking = _build_ranking(results)
    ranking_path = comparison_output_dir / "feature_set_comparison_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    return {
        "output_dir": str(comparison_output_dir),
        "configs_dir": str(configs_dir),
        "results_path": str(results_path),
        "ranking_path": str(ranking_path),
        "results": results,
        "ranking": ranking,
        "feature_set_outputs": feature_set_outputs,
    }


def _build_feature_set_config(
    base_config: dict,
    feature_set: dict,
    episodes: int,
    batch_size: int,
    actor_learning_rate: float,
    critic_learning_rate: float,
) -> dict:
    config = deepcopy(base_config)
    features = feature_set.get("features")
    if features is None:
        config.pop("features", None)
    else:
        config["features"] = deepcopy(features)

    config["training"]["episodes"] = episodes
    config["td3"]["batch_size"] = batch_size
    config["td3"]["actor_learning_rate"] = actor_learning_rate
    config["td3"]["critic_learning_rate"] = critic_learning_rate

    return config


def _build_feature_set_row(
    feature_set: dict,
    config_path: Path,
    feature_set_output: dict,
) -> dict:
    summary = feature_set_output["summary"].iloc[0].to_dict()
    row = {
        "feature_set_id": feature_set["feature_set_id"],
        "description": feature_set["description"],
        "config_path": str(config_path),
        "output_dir": feature_set_output["output_dir"],
        "summary_path": feature_set_output["summary_path"],
        "by_fold_summary_path": feature_set_output["by_fold_summary_path"],
        "by_seed_summary_path": feature_set_output["by_seed_summary_path"],
    }
    for column in SUMMARY_COLUMNS:
        row[column] = summary[column]

    return row


def _build_ranking(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values(
        by=RANKING_COLUMNS,
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _write_yaml_config(config: dict, config_path: Path) -> None:
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
