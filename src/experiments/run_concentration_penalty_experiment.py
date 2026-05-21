"""Controlled soft concentration penalty experiment runner.

This module runs an experiment-only lambda_concentration grid. It does not
change default configs, reward implementation, TD3 architecture, environment
dynamics, or training logic.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.audit_reward_incentives import compute_reward_incentive_flags
from src.analysis.mandate_aware_score import (
    assign_drawdown_bucket,
    calculate_recovery_required,
    get_drawdown_multiplier,
)
from src.analysis.robust_score import build_robust_score_report
from src.experiments.run_feature_block_ablation import (
    ACTOR_LR,
    BASE_CONFIG_PATH,
    BATCH_SIZE,
    CRITIC_LR,
    EXPANDING_FOLDS,
    RETURNS_PATH,
    _actual_fold_row,
    _build_base_config,
    _build_experiment_result,
    _metric_row,
    aggregate_metric_rows,
    build_ablation_fold_datasets,
    build_returns_dataset_from_config,
    train_td3_ablation_on_datasets,
)
from src.experiments.run_protocol_pure_td3_revalidation import (
    DSR_METHOD,
    PROTOCOL_CANDIDATES,
    TIMING_CONVENTION,
    _build_candidate_run_config,
    _build_feature_context,
    _candidate_auxiliary_features,
    _candidate_raw_features,
    _select_candidates,
    _validate_protocol_reward_semantics,
    _write_yaml,
)
from src.experiments.save_experiment_outputs import save_basic_experiment_outputs


DEFAULT_OUTPUT_DIR = "outputs/tables/concentration_penalty_experiment_v5_60ep_5seeds"
DEFAULT_CANDIDATE = "V5_no_volatility_block"
DEFAULT_LAMBDA_GRID = [0.0, 0.01, 0.03, 0.05]
DEFAULT_SEEDS = [7, 21, 42, 84, 101]
DEFAULT_EPISODES = 60
SMOKE_EPISODES = 5
SMOKE_SEEDS = [7]
MATERIAL_SCORE_DETERIORATION = 0.05
MEANINGFUL_DIVERSIFICATION_IMPROVEMENT = 0.20


def run_concentration_penalty_experiment(
    returns_path: str = RETURNS_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    candidate: str = DEFAULT_CANDIDATE,
    lambda_concentration_grid: list[float] | None = None,
    episodes: int = DEFAULT_EPISODES,
    seeds: list[int] | None = None,
    max_folds: int | None = None,
    transaction_cost: float = 0.001,
    base_config_path: str = BASE_CONFIG_PATH,
    batch_size: int = BATCH_SIZE,
    actor_learning_rate: float = ACTOR_LR,
    critic_learning_rate: float = CRITIC_LR,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run an isolated lambda_concentration grid for one protocol candidate."""
    lambda_grid = (
        list(DEFAULT_LAMBDA_GRID)
        if lambda_concentration_grid is None
        else list(lambda_concentration_grid)
    )
    selected_seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    selected_episodes = episodes
    selected_folds = list(EXPANDING_FOLDS)
    if max_folds is not None:
        selected_folds = selected_folds[:max_folds]
    if smoke:
        selected_episodes = min(selected_episodes, SMOKE_EPISODES)
        selected_seeds = selected_seeds[:1] if selected_seeds else SMOKE_SEEDS
        selected_folds = selected_folds[:1]

    destination = Path(output_dir)
    configs_dir = destination / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    base_config = _build_base_config(base_config_path, returns_path, selected_episodes)
    base_config["environment"]["transaction_cost"] = transaction_cost
    _validate_protocol_reward_semantics(base_config)

    base_candidate = deepcopy(_select_candidates([candidate])[0])
    experiment_candidates = build_concentration_grid_candidates(
        base_candidate,
        lambda_grid,
    )

    returns = build_returns_dataset_from_config(
        base_config,
        configs_dir / "_returns_config.yaml",
    )
    feature_context = _build_feature_context(returns, base_config)

    fold_rows = []
    metric_rows = []
    run_configs = {}
    candidate_outputs = {}

    for experiment_candidate in experiment_candidates:
        candidate_name = experiment_candidate["name"]
        candidate_outputs[candidate_name] = {}
        raw_features = _candidate_raw_features(experiment_candidate, feature_context)
        raw_auxiliary = _candidate_auxiliary_features(experiment_candidate, feature_context)
        include_auxiliary = raw_auxiliary is not None

        for fold in selected_folds:
            datasets = build_ablation_fold_datasets(
                returns=returns,
                raw_features=raw_features,
                raw_auxiliary_features=(
                    raw_auxiliary
                    if raw_auxiliary is not None
                    else pd.DataFrame(index=returns.index)
                ),
                fold=fold,
                include_auxiliary=include_auxiliary,
            )
            fold_rows.append(_actual_fold_row(fold, datasets))
            for seed in selected_seeds:
                config = build_candidate_run_config_with_concentration(
                    base_config=base_config,
                    candidate=experiment_candidate,
                    seed=seed,
                    episodes=selected_episodes,
                    batch_size=batch_size,
                    actor_learning_rate=actor_learning_rate,
                    critic_learning_rate=critic_learning_rate,
                )
                run_key = f"{fold['fold_id']}_{candidate_name}_seed_{seed}"
                config_path = configs_dir / f"{run_key}.yaml"
                _write_yaml(config, config_path)
                run_configs[run_key] = {
                    "path": str(config_path),
                    "reward": config["reward"],
                    "features": config["features"],
                    "lambda_concentration": experiment_candidate[
                        "lambda_concentration"
                    ],
                }

                raw_result = train_td3_ablation_on_datasets(datasets, config)
                experiment_result = _build_experiment_result(raw_result)
                saved_paths = save_basic_experiment_outputs(
                    experiment_result,
                    output_dir=str(destination),
                    experiment_name=run_key,
                )
                candidate_outputs[candidate_name][f"{fold['fold_id']}_seed_{seed}"] = {
                    "saved_paths": saved_paths,
                    "raw_result": raw_result,
                }
                for split_name in ("validation", "test"):
                    metric_rows.append(
                        _metric_row(
                            variant=candidate_name,
                            fold=fold,
                            seed=seed,
                            split=split_name,
                            metrics=experiment_result[f"{split_name}_metrics_table"].loc[
                                "agent"
                            ],
                            diagnostics=experiment_result[
                                f"{split_name}_diagnostics"
                            ],
                            policy_history=experiment_result[
                                f"{split_name}_policy_history"
                            ],
                        )
                    )

    actual_folds = pd.DataFrame(fold_rows).drop_duplicates()
    actual_folds.to_csv(destination / "actual_fold_dates.csv", index=False)
    seed_results = pd.DataFrame(metric_rows)
    seed_results.to_csv(destination / "seed_fold_strategy_results.csv", index=False)

    overall = aggregate_metric_rows(seed_results, group_columns=["strategy", "split"])
    overall.to_csv(destination / "overall_aggregate_by_strategy_split.csv", index=False)
    seed_level = aggregate_metric_rows(
        seed_results,
        group_columns=["strategy", "split", "seed"],
    )
    seed_level.to_csv(destination / "seed_level_aggregate_by_strategy_split.csv", index=False)
    fold_level = aggregate_metric_rows(
        seed_results,
        group_columns=["strategy", "split", "fold"],
    )
    fold_level.to_csv(destination / "fold_level_aggregate_by_strategy_split.csv", index=False)

    robust_report = build_robust_score_report(str(destination), output_dir=str(destination))
    summary = build_concentration_penalty_summary(
        overall,
        robust_report["ranking"],
        experiment_candidates,
        episodes=selected_episodes,
    )
    rankings = build_concentration_penalty_rankings(summary)
    metrics = build_concentration_penalty_metrics(seed_results, experiment_candidates)

    metrics_path = destination / "concentration_penalty_metrics.csv"
    summary_path = destination / "concentration_penalty_summary.csv"
    rankings_path = destination / "concentration_penalty_rankings.csv"
    metadata_path = destination / "concentration_penalty_metadata.json"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    rankings.to_csv(rankings_path, index=False)
    metadata = build_concentration_penalty_metadata(
        returns_path=returns_path,
        output_dir=str(destination),
        candidate=candidate,
        lambda_grid=lambda_grid,
        episodes=selected_episodes,
        seeds=selected_seeds,
        folds=selected_folds,
        actual_folds=actual_folds,
        transaction_cost=transaction_cost,
        base_config_path=base_config_path,
        run_configs=run_configs,
        smoke=smoke,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "output_dir": str(destination),
        "metrics": metrics,
        "summary": summary,
        "rankings": rankings,
        "metadata": metadata,
        "paths": {
            "concentration_penalty_metrics": str(metrics_path),
            "concentration_penalty_summary": str(summary_path),
            "concentration_penalty_rankings": str(rankings_path),
            "concentration_penalty_metadata": str(metadata_path),
        },
        "robust_score": robust_report,
        "candidate_outputs": candidate_outputs,
    }


def parse_lambda_grid(value: str | None) -> list[float]:
    """Parse a comma-separated lambda_concentration grid."""
    if value is None or not value.strip():
        return list(DEFAULT_LAMBDA_GRID)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_seed_list(value: str | None) -> list[int] | None:
    """Parse comma-separated integer seeds."""
    if value is None:
        return None
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def concentration_candidate_name(base_name: str, lambda_concentration: float) -> str:
    """Build a stable isolated candidate name for a concentration value."""
    return f"{base_name}_lconc_{_format_lambda(lambda_concentration)}"


def build_concentration_grid_candidates(
    base_candidate: dict,
    lambda_grid: list[float],
) -> list[dict]:
    """Create isolated candidate definitions for the lambda_concentration grid."""
    candidates = []
    for lambda_concentration in lambda_grid:
        candidate = deepcopy(base_candidate)
        candidate["base_candidate"] = base_candidate["name"]
        candidate["lambda_concentration"] = float(lambda_concentration)
        candidate["name"] = concentration_candidate_name(
            base_candidate["name"],
            lambda_concentration,
        )
        candidates.append(candidate)
    return candidates


def build_candidate_run_config_with_concentration(
    base_config: dict,
    candidate: dict,
    seed: int,
    episodes: int,
    batch_size: int,
    actor_learning_rate: float,
    critic_learning_rate: float,
) -> dict:
    """Build a run config with only lambda_concentration overridden."""
    config = _build_candidate_run_config(
        base_config=base_config,
        candidate=candidate,
        seed=seed,
        episodes=episodes,
        batch_size=batch_size,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
    )
    config["reward"]["lambda_concentration"] = float(candidate["lambda_concentration"])
    _validate_protocol_reward_semantics(config)
    return config


def build_concentration_penalty_metrics(
    seed_results: pd.DataFrame,
    candidates: list[dict],
) -> pd.DataFrame:
    """Attach lambda grid metadata to seed/fold metrics."""
    metadata = _candidate_metadata_frame(candidates)
    metrics = seed_results.rename(columns={"strategy": "candidate_name"}).copy()
    return metrics.merge(metadata, on="candidate_name", how="left")


def build_concentration_penalty_summary(
    overall: pd.DataFrame,
    robust_ranking: pd.DataFrame,
    candidates: list[dict],
    episodes: int,
) -> pd.DataFrame:
    """Build aggregate summary with robust, mandate, and concentration diagnostics."""
    metadata = _candidate_metadata_frame(candidates)
    summary = overall.rename(columns={"strategy": "candidate_name"}).copy()
    summary = summary.merge(metadata, on="candidate_name", how="left")
    summary["episodes"] = episodes
    summary = summary.rename(
        columns={
            "mean_sharpe": "sharpe",
            "mean_sortino": "sortino",
            "mean_calmar": "calmar",
            "mean_cumulative_return": "cumulative_return",
            "mean_annualized_return": "annualized_return",
            "mean_annualized_volatility": "annualized_volatility",
            "mean_max_drawdown": "max_drawdown",
            "mean_average_turnover": "average_turnover",
            "mean_average_effective_number_of_assets": (
                "average_effective_number_of_assets"
            ),
            "mean_average_max_weight": "average_max_weight",
        }
    )
    robust = robust_ranking.rename(columns={"strategy": "candidate_name"}).copy()
    robust_keep = [
        column
        for column in [
            "candidate_name",
            "robust_score",
            "median_run_dsr_n25",
            "date_averaged_dsr_n25",
            "dsr_method",
        ]
        if column in robust.columns
    ]
    summary = summary.merge(robust.loc[:, robust_keep], on="candidate_name", how="left")
    summary["mandate_bucket"] = summary["max_drawdown"].apply(assign_drawdown_bucket)
    summary["recovery_required"] = summary["max_drawdown"].apply(
        calculate_recovery_required
    )
    summary["drawdown_multiplier"] = summary["max_drawdown"].apply(
        get_drawdown_multiplier
    )
    summary["mandate_aware_score"] = (
        summary["robust_score"] * summary["drawdown_multiplier"]
    )
    summary.loc[summary["mandate_bucket"] == "not_eligible", "mandate_aware_score"] = 0.0
    summary_for_flags = summary.rename(columns={"candidate_name": "strategy"})
    flagged = compute_reward_incentive_flags(summary_for_flags)
    flagged = flagged.rename(columns={"strategy": "candidate_name"})
    return _select_summary_columns(flagged)


def build_concentration_penalty_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline deltas and conservative diagnostic decision labels."""
    test = summary.loc[summary["split"].astype(str) == "test"].copy()
    if test.empty:
        return pd.DataFrame()
    baseline = test.loc[test["lambda_concentration"] == 0.0]
    if baseline.empty:
        raise ValueError("Baseline lambda_concentration = 0.0 is required.")
    baseline_row = baseline.iloc[0]
    for metric in [
        "average_effective_number_of_assets",
        "average_max_weight",
        "average_turnover",
        "sharpe",
        "robust_score",
        "mandate_aware_score",
        "max_drawdown",
        "annualized_return",
    ]:
        test[f"delta_{metric}_vs_baseline"] = (
            pd.to_numeric(test[metric], errors="coerce")
            - pd.to_numeric(pd.Series([baseline_row[metric]]), errors="coerce").iloc[0]
        )
    test["decision_label"] = test.apply(label_concentration_decision, axis=1)
    return test.sort_values(
        ["mandate_aware_score", "robust_score"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def label_concentration_decision(row: pd.Series) -> str:
    """Assign a conservative experiment decision label versus baseline."""
    if float(row.get("lambda_concentration", 0.0)) == 0.0:
        return "baseline"
    diversification_delta = row.get(
        "delta_average_effective_number_of_assets_vs_baseline",
        0.0,
    )
    robust_delta = row.get("delta_robust_score_vs_baseline", 0.0)
    mandate_delta = row.get("delta_mandate_aware_score_vs_baseline", 0.0)
    if pd.isna(diversification_delta):
        return "unstable_or_inconclusive"
    if diversification_delta <= MEANINGFUL_DIVERSIFICATION_IMPROVEMENT:
        return "no_behavioral_improvement"
    if (
        pd.notna(robust_delta)
        and robust_delta < -MATERIAL_SCORE_DETERIORATION
    ) or (
        pd.notna(mandate_delta)
        and mandate_delta < -MATERIAL_SCORE_DETERIORATION
    ):
        return "diversifies_but_hurts_performance"
    if (
        pd.notna(robust_delta)
        and robust_delta >= -MATERIAL_SCORE_DETERIORATION
        and (
            pd.isna(mandate_delta)
            or mandate_delta >= -MATERIAL_SCORE_DETERIORATION
        )
    ):
        return "dominates_baseline"
    return "unstable_or_inconclusive"


def build_concentration_penalty_metadata(
    returns_path: str,
    output_dir: str,
    candidate: str,
    lambda_grid: list[float],
    episodes: int,
    seeds: list[int],
    folds: list[dict],
    actual_folds: pd.DataFrame,
    transaction_cost: float,
    base_config_path: str,
    run_configs: dict,
    smoke: bool,
) -> dict[str, Any]:
    """Build reproducibility metadata for the experiment."""
    return {
        "runner": "src.experiments.run_concentration_penalty_experiment",
        "git_commit_hash": _git_commit_hash(),
        "returns_path": returns_path,
        "output_dir": output_dir,
        "candidate": candidate,
        "lambda_concentration_grid": lambda_grid,
        "episodes": episodes,
        "seeds": seeds,
        "folds": folds,
        "actual_folds": actual_folds.to_dict(orient="records"),
        "transaction_cost": transaction_cost,
        "base_config_path": base_config_path,
        "timing_convention": TIMING_CONVENTION,
        "DSR_method_policy": DSR_METHOD,
        "experiment_only_warning": (
            "lambda_concentration values are experiment-only and are not default "
            "configuration changes."
        ),
        "evaluation_only_note": (
            "robust_score and mandate_aware_score are evaluation-only and are "
            "not used as training reward."
        ),
        "run_configs": run_configs,
        "smoke_mode": bool(smoke),
    }


def _select_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_name",
        "base_candidate",
        "lambda_concentration",
        "split",
        "n_folds",
        "n_seeds",
        "episodes",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "robust_score",
        "mandate_aware_score",
        "max_drawdown",
        "worst_max_drawdown",
        "average_turnover",
        "mean_transaction_cost",
        "average_effective_number_of_assets",
        "average_max_weight",
        "mean_cash_weight",
        "cash_above_10_rate",
        "concentration_classification",
        "suspicious_or_lazy_concentration_candidate",
        "justified_concentration_candidate",
        "concentration_reason",
        "lazy_reason",
        "justification_reason",
    ]
    result = summary.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, columns]


def _candidate_metadata_frame(candidates: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_name": candidate["name"],
                "base_candidate": candidate["base_candidate"],
                "lambda_concentration": candidate["lambda_concentration"],
            }
            for candidate in candidates
        ]
    )


def _format_lambda(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _git_commit_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a controlled lambda_concentration experiment.",
    )
    parser.add_argument("--returns-path", default=RETURNS_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--lambda-concentration-grid",
        default=",".join(str(value) for value in DEFAULT_LAMBDA_GRID),
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--base-config-path", default=BASE_CONFIG_PATH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    report = run_concentration_penalty_experiment(
        returns_path=args.returns_path,
        output_dir=args.output_dir,
        candidate=args.candidate,
        lambda_concentration_grid=parse_lambda_grid(args.lambda_concentration_grid),
        episodes=args.episodes,
        seeds=parse_seed_list(args.seeds),
        max_folds=args.max_folds,
        transaction_cost=args.transaction_cost,
        base_config_path=args.base_config_path,
        smoke=args.smoke,
    )
    print(f"Output folder: {report['output_dir']}")
    print("\nConcentration penalty rankings:")
    print(report["rankings"].to_string(index=False))


if __name__ == "__main__":
    main()
