"""Experiment-only max-weight cap runner for TD3 allocation.

This module tests post-action allocation caps without changing default
environment behavior, TD3 architecture, reward logic, or training logic.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import src.experiments.run_feature_block_ablation as ablation_module
from src.analysis.audit_reward_incentives import compute_reward_incentive_flags
from src.analysis.mandate_aware_score import (
    assign_drawdown_bucket,
    calculate_recovery_required,
    get_drawdown_multiplier,
)
from src.analysis.robust_score import build_robust_score_report
from src.env.portfolio_env import PortfolioEnv
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


DEFAULT_OUTPUT_DIR = "outputs/tables/max_weight_cap_experiment_v5_60ep_5seeds"
DEFAULT_CANDIDATE = "V5_no_volatility_block"
DEFAULT_MAX_WEIGHT_GRID = [None, 0.80, 0.70, 0.60]
DEFAULT_SEEDS = [7, 21, 42, 84, 101]
DEFAULT_EPISODES = 60
SMOKE_EPISODES = 5
MEANINGFUL_DIVERSIFICATION_IMPROVEMENT = 0.20
MATERIAL_SCORE_DETERIORATION = 0.05
DRAWDOWN_WORSENING_TOLERANCE = -0.02
TURNOVER_IMPROVEMENT = -0.05


def run_max_weight_cap_experiment(
    returns_path: str = RETURNS_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    candidate: str = DEFAULT_CANDIDATE,
    max_weight_grid: list[float | None] | None = None,
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
    """Run an isolated max-weight cap grid for one protocol TD3 candidate."""
    cap_grid = list(DEFAULT_MAX_WEIGHT_GRID if max_weight_grid is None else max_weight_grid)
    selected_seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    selected_episodes = episodes
    selected_folds = list(EXPANDING_FOLDS)
    if max_folds is not None:
        selected_folds = selected_folds[:max_folds]
    if smoke:
        selected_episodes = min(selected_episodes, SMOKE_EPISODES)
        selected_seeds = selected_seeds[:1] if selected_seeds else [7]
        selected_folds = selected_folds[:1]

    destination = Path(output_dir)
    configs_dir = destination / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    base_config = _build_base_config(base_config_path, returns_path, selected_episodes)
    base_config["environment"]["transaction_cost"] = transaction_cost
    _validate_protocol_reward_semantics(base_config)
    base_candidate = deepcopy(_select_candidates([candidate])[0])
    experiment_candidates = build_max_weight_grid_candidates(base_candidate, cap_grid)

    returns = build_returns_dataset_from_config(
        base_config,
        configs_dir / "_returns_config.yaml",
    )
    feature_context = _build_feature_context(returns, base_config, [base_candidate])

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
                config = build_candidate_run_config_for_cap(
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
                    "max_weight_cap": experiment_candidate["max_weight_cap"],
                }

                with patched_portfolio_env(experiment_candidate["max_weight_cap"]):
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
    summary = build_max_weight_cap_summary(
        overall,
        robust_report["ranking"],
        experiment_candidates,
        episodes=selected_episodes,
    )
    rankings = build_max_weight_cap_rankings(summary)
    metrics = build_max_weight_cap_metrics(seed_results, experiment_candidates)

    metrics_path = destination / "max_weight_cap_metrics.csv"
    summary_path = destination / "max_weight_cap_summary.csv"
    rankings_path = destination / "max_weight_cap_rankings.csv"
    metadata_path = destination / "max_weight_cap_metadata.json"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    rankings.to_csv(rankings_path, index=False)
    metadata = build_max_weight_cap_metadata(
        returns_path=returns_path,
        output_dir=str(destination),
        candidate=candidate,
        max_weight_grid=cap_grid,
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
            "max_weight_cap_metrics": str(metrics_path),
            "max_weight_cap_summary": str(summary_path),
            "max_weight_cap_rankings": str(rankings_path),
            "max_weight_cap_metadata": str(metadata_path),
        },
        "robust_score": robust_report,
        "candidate_outputs": candidate_outputs,
    }


def project_weights_to_max_cap(
    weights,
    max_weight: float | None,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Project long-only weights onto the capped simplex."""
    weights_array = np.asarray(weights, dtype=float)
    if max_weight is None:
        return weights_array.copy()
    if weights_array.ndim != 1:
        raise ValueError("weights must be one-dimensional.")
    if not np.isfinite(weights_array).all():
        raise ValueError("weights must be finite.")
    if np.any(weights_array < -tolerance):
        raise ValueError("weights must be non-negative.")
    n_assets = len(weights_array)
    if n_assets == 0:
        raise ValueError("weights must not be empty.")
    cap = float(max_weight)
    if cap <= 0.0 or cap > 1.0:
        raise ValueError("max_weight must be in (0, 1].")
    if cap * n_assets < 1.0 - tolerance:
        raise ValueError("max_weight cap is infeasible for the number of assets.")

    clipped = np.clip(weights_array, 0.0, None)
    total = float(clipped.sum())
    if total <= tolerance:
        clipped = np.full(n_assets, 1.0 / n_assets)
    else:
        clipped = clipped / total

    capped = np.minimum(clipped, cap)
    for _ in range(n_assets * 2):
        deficit = 1.0 - float(capped.sum())
        if abs(deficit) <= tolerance:
            break
        room = np.maximum(cap - capped, 0.0)
        room_total = float(room.sum())
        if room_total <= tolerance:
            break
        capped += room / room_total * deficit
        capped = np.minimum(capped, cap)

    capped = np.clip(capped, 0.0, cap)
    total = float(capped.sum())
    if abs(total - 1.0) > 1e-9:
        room = np.maximum(cap - capped, 0.0)
        room_total = float(room.sum())
        if room_total <= tolerance:
            capped = capped / total
        else:
            capped += room / room_total * (1.0 - total)
    return capped / capped.sum()


def apply_max_weight_cap_to_action(weights, max_weight: float | None) -> np.ndarray:
    """Apply the experiment-only max-weight cap to normalized action weights."""
    return project_weights_to_max_cap(weights, max_weight)


class CappedPortfolioEnv(PortfolioEnv):
    """PortfolioEnv variant used only inside the max-weight cap experiment."""

    max_weight_cap: float | None = None

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        weights = super()._normalize_action(action)
        return apply_max_weight_cap_to_action(weights, self.max_weight_cap)


@contextmanager
def patched_portfolio_env(max_weight_cap: float | None):
    """Temporarily patch the ablation module to use the capped env."""
    if max_weight_cap is None:
        yield
        return
    original = ablation_module.PortfolioEnv

    class ExperimentCappedPortfolioEnv(CappedPortfolioEnv):
        pass

    ExperimentCappedPortfolioEnv.max_weight_cap = max_weight_cap
    ablation_module.PortfolioEnv = ExperimentCappedPortfolioEnv
    try:
        yield
    finally:
        ablation_module.PortfolioEnv = original


def parse_max_weight_grid(value: str | None) -> list[float | None]:
    """Parse max-weight grid values with `uncapped` as None."""
    if value is None or not value.strip():
        return list(DEFAULT_MAX_WEIGHT_GRID)
    result: list[float | None] = []
    for item in value.split(","):
        text = item.strip().lower()
        if not text:
            continue
        if text in {"uncapped", "none", "baseline"}:
            result.append(None)
        else:
            result.append(float(text))
    return result


def parse_seed_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def max_weight_candidate_name(base_name: str, max_weight_cap: float | None) -> str:
    suffix = "uncapped" if max_weight_cap is None else _format_cap(max_weight_cap)
    return f"{base_name}_cap_{suffix}"


def build_max_weight_grid_candidates(
    base_candidate: dict,
    max_weight_grid: list[float | None],
) -> list[dict]:
    candidates = []
    for max_weight_cap in max_weight_grid:
        candidate = deepcopy(base_candidate)
        candidate["base_candidate"] = base_candidate["name"]
        candidate["max_weight_cap"] = max_weight_cap
        candidate["name"] = max_weight_candidate_name(
            base_candidate["name"],
            max_weight_cap,
        )
        candidates.append(candidate)
    return candidates


def build_candidate_run_config_for_cap(
    base_config: dict,
    candidate: dict,
    seed: int,
    episodes: int,
    batch_size: int,
    actor_learning_rate: float,
    critic_learning_rate: float,
) -> dict:
    config = _build_candidate_run_config(
        base_config=base_config,
        candidate=candidate,
        seed=seed,
        episodes=episodes,
        batch_size=batch_size,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
    )
    _validate_protocol_reward_semantics(config)
    return config


def build_max_weight_cap_metrics(
    seed_results: pd.DataFrame,
    candidates: list[dict],
) -> pd.DataFrame:
    metadata = _candidate_metadata_frame(candidates)
    metrics = seed_results.rename(columns={"strategy": "candidate_name"}).copy()
    return metrics.merge(metadata, on="candidate_name", how="left")


def build_max_weight_cap_summary(
    overall: pd.DataFrame,
    robust_ranking: pd.DataFrame,
    candidates: list[dict],
    episodes: int,
) -> pd.DataFrame:
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
    flagged = compute_reward_incentive_flags(
        summary.rename(columns={"candidate_name": "strategy"})
    ).rename(columns={"strategy": "candidate_name"})
    return _select_summary_columns(flagged)


def build_max_weight_cap_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    test = summary.loc[summary["split"].astype(str) == "test"].copy()
    if test.empty:
        return pd.DataFrame()
    baseline = test.loc[test["max_weight_cap"].isna()]
    if baseline.empty:
        raise ValueError("Uncapped baseline is required.")
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
        baseline_value = pd.to_numeric(
            pd.Series([baseline_row[metric]]),
            errors="coerce",
        ).iloc[0]
        test[f"delta_{metric}_vs_baseline"] = (
            pd.to_numeric(test[metric], errors="coerce") - baseline_value
        )
    test["decision_label"] = test.apply(label_max_weight_cap_decision, axis=1)
    return test.sort_values(
        ["mandate_aware_score", "robust_score"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def label_max_weight_cap_decision(row: pd.Series) -> str:
    if pd.isna(row.get("max_weight_cap")):
        return "baseline"
    effective_delta = row.get("delta_average_effective_number_of_assets_vs_baseline")
    max_weight_delta = row.get("delta_average_max_weight_vs_baseline")
    robust_delta = row.get("delta_robust_score_vs_baseline")
    mandate_delta = row.get("delta_mandate_aware_score_vs_baseline")
    turnover_delta = row.get("delta_average_turnover_vs_baseline")
    drawdown_delta = row.get("delta_max_drawdown_vs_baseline")
    if pd.isna(effective_delta):
        return "unstable_or_inconclusive"
    if effective_delta <= MEANINGFUL_DIVERSIFICATION_IMPROVEMENT:
        return "no_behavioral_improvement"
    hurts_score = (
        pd.notna(robust_delta)
        and robust_delta < -MATERIAL_SCORE_DETERIORATION
    ) or (
        pd.notna(mandate_delta)
        and mandate_delta < -MATERIAL_SCORE_DETERIORATION
    )
    if hurts_score:
        return "diversifies_but_hurts_performance"
    if (
        pd.notna(turnover_delta)
        and turnover_delta < TURNOVER_IMPROVEMENT
        and pd.notna(robust_delta)
        and robust_delta >= -MATERIAL_SCORE_DETERIORATION
        and (
            pd.isna(mandate_delta)
            or mandate_delta >= -MATERIAL_SCORE_DETERIORATION
        )
    ):
        return "reduces_concentration_and_turnover"
    if (
        pd.notna(max_weight_delta)
        and max_weight_delta < 0.0
        and pd.notna(robust_delta)
        and robust_delta >= -MATERIAL_SCORE_DETERIORATION
        and (
            pd.isna(mandate_delta)
            or mandate_delta >= -MATERIAL_SCORE_DETERIORATION
        )
        and (
            pd.isna(drawdown_delta)
            or drawdown_delta >= DRAWDOWN_WORSENING_TOLERANCE
        )
    ):
        return "dominates_baseline"
    return "unstable_or_inconclusive"


def build_max_weight_cap_metadata(
    returns_path: str,
    output_dir: str,
    candidate: str,
    max_weight_grid: list[float | None],
    episodes: int,
    seeds: list[int],
    folds: list[dict],
    actual_folds: pd.DataFrame,
    transaction_cost: float,
    base_config_path: str,
    run_configs: dict,
    smoke: bool,
) -> dict[str, Any]:
    return {
        "runner": "src.experiments.run_max_weight_cap_experiment",
        "git_commit_hash": _git_commit_hash(),
        "returns_path": returns_path,
        "output_dir": output_dir,
        "candidate": candidate,
        "max_weight_grid": max_weight_grid,
        "episodes": episodes,
        "seeds": seeds,
        "folds": folds,
        "actual_folds": actual_folds.to_dict(orient="records"),
        "transaction_cost": transaction_cost,
        "base_config_path": base_config_path,
        "timing_convention": TIMING_CONVENTION,
        "DSR_method_policy": DSR_METHOD,
        "experiment_only_warning": (
            "max-weight caps are experiment-only allocation projections and "
            "are not default behavior."
        ),
        "evaluation_only_note": (
            "robust_score and mandate_aware_score are evaluation-only and are "
            "not used as training reward."
        ),
        "cap_implementation": (
            "The runner temporarily patches the ablation PortfolioEnv with a "
            "local subclass that projects normalized action weights onto the "
            "capped long-only simplex."
        ),
        "run_configs": run_configs,
        "smoke_mode": bool(smoke),
    }


def _select_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_name",
        "base_candidate",
        "max_weight_cap",
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
                "max_weight_cap": candidate["max_weight_cap"],
            }
            for candidate in candidates
        ]
    )


def _format_cap(value: float) -> str:
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
        description="Run an experiment-only TD3 max-weight cap grid.",
    )
    parser.add_argument("--returns-path", default=RETURNS_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--max-weight-grid", default="uncapped,0.80,0.70,0.60")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--base-config-path", default=BASE_CONFIG_PATH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = run_max_weight_cap_experiment(
        returns_path=args.returns_path,
        output_dir=args.output_dir,
        candidate=args.candidate,
        max_weight_grid=parse_max_weight_grid(args.max_weight_grid),
        episodes=args.episodes,
        seeds=parse_seed_list(args.seeds),
        max_folds=args.max_folds,
        transaction_cost=args.transaction_cost,
        base_config_path=args.base_config_path,
        smoke=args.smoke,
    )
    print(f"Output folder: {report['output_dir']}")
    print("\nMax-weight cap rankings:")
    print(report["rankings"].to_string(index=False))


if __name__ == "__main__":
    main()
