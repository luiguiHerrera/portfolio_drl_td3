"""Protocol-pure TD3 revalidation runner.

This runner trains selected TD3 candidates under the current common protocol
state and writes outputs compatible with the protocol TD3 comparison runner.
It reuses the existing dataset, training, output, aggregation, and robust-score
utilities rather than changing TD3 architecture, reward functions, environment
dynamics, or the training loop.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.analysis.robust_score import build_robust_score_report
from src.analysis.validate_v3_macro_current import (
    build_macro_coverage_table,
    validate_macro_coverage,
)
from src.data.feature_factory import build_configured_features
from src.data.features_v2 import build_features_v2
from src.data.features_v3 import build_features_v3
from src.data.features_v7 import build_features_v7
from src.data.features_v8 import build_features_v8
from src.data.features_v5 import (
    build_features_v5,
    build_v5_regime_auxiliary_features,
)
from src.data.garch_features import (
    GARCH_FALLBACK_ROLLING_REALIZED,
    GARCH_MODE_ROLLING_FITTED,
    build_garch_feature_set_by_mode,
)
from src.data.macro_loader import load_macro_data_from_csv
from src.experiments.run_feature_block_ablation import (
    ACTOR_LR,
    BASE_CONFIG_PATH,
    BATCH_SIZE,
    CRITIC_LR,
    EPISODES,
    EXPANDING_FOLDS,
    RETURNS_PATH,
    SEEDS,
    _actual_fold_row,
    _build_base_config,
    _build_experiment_result,
    _metric_row,
    aggregate_metric_rows,
    build_ablation_fold_datasets,
    build_feature_block_map,
    build_returns_dataset_from_config,
    select_feature_columns,
    train_td3_ablation_on_datasets,
)
from src.experiments.save_experiment_outputs import save_basic_experiment_outputs


OUTPUT_DIR = "outputs/tables/protocol_pure_td3_revalidation_30ep_5seeds"
TIMING_CONVENTION = "information through t-1, weights for t, realized return at t"
DSR_METHOD = "median_run -> date_averaged -> pooled -> fallback_from_sharpe"
DEFAULT_V3_MACRO_PATH = "data/processed/macro_weekly_latest.csv"
DEFAULT_V3_REALTIME_MACRO_PATH = "data/processed/macro_weekly_realtime_latest.csv"
DEFAULT_V3_REALTIME_CLEAN_NO_DXY_MACRO_PATH = (
    "data/processed/macro_weekly_realtime_clean_latest.csv"
)

PROTOCOL_CANDIDATES = [
    {
        "name": "V2_reference_full",
        "feature_version": "v2",
        "description": "Full V2 reference features with current cleaned reference reward.",
        "default_enabled": True,
        "exclude_blocks": [],
        "use_dynamic_cash": False,
        "cash_risk_off_column": None,
    },
    {
        "name": "V3_real_macro_current",
        "feature_version": "v3",
        "description": (
            "V2 plus current-window local real macro features. Requires a valid "
            "macro CSV with coverage through the returns end date."
        ),
        "default_enabled": False,
        "macro_path": DEFAULT_V3_MACRO_PATH,
        "macro_date_column": "date",
        "exclude_blocks": [],
        "use_dynamic_cash": False,
        "cash_risk_off_column": None,
    },
    {
        "name": "V3_real_macro_vintage",
        "feature_version": "v3",
        "description": (
            "V2 plus real-time/as-of macro features. Uses FRED as-of vintage "
            "data where available, with an explicit DXY fallback documented "
            "in the macro validation report."
        ),
        "default_enabled": False,
        "macro_path": DEFAULT_V3_REALTIME_MACRO_PATH,
        "macro_date_column": "date",
        "macro_source": "realtime_asof_with_dxy_fallback",
        "exclude_blocks": [],
        "use_dynamic_cash": False,
        "cash_risk_off_column": None,
    },
    {
        "name": "V3_real_macro_vintage_clean_no_dxy",
        "feature_version": "v3",
        "description": (
            "V2 plus clean real-time/as-of macro features with DXY excluded. "
            "Uses only full-window fresh true-vintage macro series with no fallback."
        ),
        "default_enabled": False,
        "macro_path": DEFAULT_V3_REALTIME_CLEAN_NO_DXY_MACRO_PATH,
        "macro_date_column": "date",
        "macro_source": "realtime_asof_no_dxy_no_fallback",
        "dollar_proxy": "excluded",
        "reason": (
            "no full-window fresh true-vintage dollar proxy exists for 2015-2026 "
            "without fallback/discontinuation"
        ),
        "exclude_blocks": [],
        "use_dynamic_cash": False,
        "cash_risk_off_column": None,
    },
    {
        "name": "V4_real_garch_current",
        "feature_version": "v4",
        "description": (
            "V2 plus rolling fitted real GARCH(1,1) volatility features using "
            "the arch package, with CASH excluded from fitted GARCH."
        ),
        "default_enabled": False,
        "exclude_blocks": [],
        "use_dynamic_cash": False,
        "cash_risk_off_column": None,
        "garch_mode": GARCH_MODE_ROLLING_FITTED,
        "garch_exclude_cash": True,
        "garch_fallback": GARCH_FALLBACK_ROLLING_REALIZED,
        "garch_min_history": 104,
        "garch_window": 156,
        "garch_allow_scipy_fallback": False,
    },
    {
        "name": "V7_real_macro_garch_current",
        "feature_version": "v7",
        "description": (
            "V3 current-window real macro features plus V4 rolling fitted "
            "real GARCH(1,1) features using arch_model."
        ),
        "default_enabled": False,
        "macro_path": DEFAULT_V3_MACRO_PATH,
        "macro_date_column": "date",
        "exclude_blocks": [],
        "use_dynamic_cash": False,
        "cash_risk_off_column": None,
        "garch_mode": GARCH_MODE_ROLLING_FITTED,
        "garch_exclude_cash": True,
        "garch_fallback": GARCH_FALLBACK_ROLLING_REALIZED,
        "garch_min_history": 104,
        "garch_window": 156,
        "garch_allow_scipy_fallback": False,
    },
    {
        "name": "V8_ewma_garch_vol_current",
        "feature_version": "v8",
        "description": (
            "V2 plus rolling fitted real GARCH volatility, lagged EWMA volatility, "
            "and GARCH/EWMA comparison features."
        ),
        "default_enabled": False,
        "exclude_blocks": [],
        "use_dynamic_cash": False,
        "cash_risk_off_column": None,
        "ewma_lambda": 0.94,
        "garch_mode": GARCH_MODE_ROLLING_FITTED,
        "garch_exclude_cash": True,
        "garch_fallback": GARCH_FALLBACK_ROLLING_REALIZED,
        "garch_min_history": 104,
        "garch_window": 156,
        "garch_allow_scipy_fallback": False,
    },
    {
        "name": "V5_no_volatility_block",
        "feature_version": "v5",
        "description": "V5 with volatility block removed, dynamic CASH penalty active.",
        "default_enabled": True,
        "exclude_blocks": ["volatility"],
        "use_dynamic_cash": True,
        "cash_risk_off_column": "risk_off_state",
    },
    {
        "name": "V6_financial_state",
        "feature_version": "v6",
        "description": "Parsimonious V6 financial state with cash-permission auxiliary signal.",
        "default_enabled": True,
        "exclude_blocks": [],
        "use_dynamic_cash": True,
        "cash_risk_off_column": "cash_permission_score",
    },
]


def run_protocol_pure_td3_revalidation(
    base_config_path: str = BASE_CONFIG_PATH,
    returns_path: str = RETURNS_PATH,
    output_dir: str = OUTPUT_DIR,
    candidates: list[str] | None = None,
    folds: list[dict] | None = None,
    seeds: list[int] | None = None,
    episodes: int = EPISODES,
    batch_size: int = BATCH_SIZE,
    actor_learning_rate: float = ACTOR_LR,
    critic_learning_rate: float = CRITIC_LR,
    max_folds: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run selected TD3 candidates under the common protocol."""
    selected_candidates = _select_candidates(candidates)
    selected_folds = list(EXPANDING_FOLDS if folds is None else folds)
    selected_seeds = list(SEEDS if seeds is None else seeds)
    selected_episodes = episodes
    if max_folds is not None:
        selected_folds = selected_folds[:max_folds]
    if smoke:
        selected_folds = selected_folds[:1]
        selected_seeds = selected_seeds[:1] if selected_seeds else [7]
        selected_episodes = min(selected_episodes, 1)

    destination = Path(output_dir)
    configs_dir = destination / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    base_config = _build_base_config(base_config_path, returns_path, selected_episodes)
    _validate_protocol_reward_semantics(base_config)
    returns = build_returns_dataset_from_config(base_config, configs_dir / "_returns_config.yaml")
    feature_context = _build_feature_context(returns, base_config, selected_candidates)

    fold_rows = []
    metric_rows = []
    candidate_outputs = {}
    run_configs = {}

    for candidate in selected_candidates:
        candidate_name = candidate["name"]
        candidate_outputs[candidate_name] = {}
        raw_features = _candidate_raw_features(candidate, feature_context)
        raw_auxiliary = _candidate_auxiliary_features(candidate, feature_context)
        include_auxiliary = raw_auxiliary is not None

        for fold in selected_folds:
            datasets = build_ablation_fold_datasets(
                returns=returns,
                raw_features=raw_features,
                raw_auxiliary_features=(
                    raw_auxiliary if raw_auxiliary is not None else pd.DataFrame(index=returns.index)
                ),
                fold=fold,
                include_auxiliary=include_auxiliary,
            )
            fold_rows.append(_actual_fold_row(fold, datasets))
            for seed in selected_seeds:
                config = _build_candidate_run_config(
                    base_config=base_config,
                    candidate=candidate,
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
                            diagnostics=experiment_result[f"{split_name}_diagnostics"],
                            policy_history=experiment_result[f"{split_name}_policy_history"],
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

    metadata = _build_metadata(
        base_config_path=base_config_path,
        returns_path=returns_path,
        output_dir=str(destination),
        candidates=selected_candidates,
        folds=selected_folds,
        actual_folds=actual_folds,
        seeds=selected_seeds,
        episodes=selected_episodes,
        batch_size=batch_size,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        base_config=base_config,
        run_configs=run_configs,
        smoke=smoke,
    )
    metadata_path = destination / "protocol_pure_td3_revalidation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    robust = build_robust_score_report(str(destination), output_dir=str(destination))

    return {
        "output_dir": str(destination),
        "seed_results": seed_results,
        "overall_aggregate": overall,
        "seed_level_aggregate": seed_level,
        "fold_level_aggregate": fold_level,
        "actual_folds": actual_folds,
        "metadata": metadata,
        "metadata_path": str(metadata_path),
        "robust_score": robust,
        "candidate_outputs": candidate_outputs,
    }


def _build_feature_context(
    returns: pd.DataFrame,
    base_config: dict,
    candidates: list[dict] | None = None,
) -> dict[str, Any]:
    requested_versions = _requested_feature_versions(candidates)
    context: dict[str, Any] = {
        "block_map": build_feature_block_map(list(returns.columns)),
    }

    if "v2" in requested_versions:
        context["v2_features"] = build_features_v2(returns)
    if "v3" in requested_versions:
        v3_candidate = _candidate_for_feature_version(candidates, "v3")
        context["v3_features"] = _build_v3_features(
            returns=returns,
            base_config=base_config,
            candidate=v3_candidate,
        )
    if "v4" in requested_versions:
        v4_candidate = _candidate_for_feature_version(candidates, "v4")
        context["v4_features"] = _build_v4_features(
            returns=returns,
            base_config=base_config,
            candidate=v4_candidate,
        )
    if "v7" in requested_versions:
        v7_candidate = _candidate_for_feature_version(candidates, "v7")
        context["v7_features"] = _build_v7_features(
            returns=returns,
            base_config=base_config,
            candidate=v7_candidate,
        )
    if "v8" in requested_versions:
        v8_candidate = _candidate_for_feature_version(candidates, "v8")
        context["v8_features"] = _build_v8_features(
            returns=returns,
            base_config=base_config,
            candidate=v8_candidate,
        )
    if "v5" in requested_versions:
        context["v5_features"] = build_features_v5(returns)
        context["v5_auxiliary"] = (
            build_v5_regime_auxiliary_features(returns).shift(1).dropna()
        )
    if "v6" in requested_versions:
        v6_config = deepcopy(base_config)
        v6_config["features"] = _feature_config("v6")
        context["v6_features"] = build_configured_features(returns, v6_config)
        context["v6_auxiliary"] = (
            context["v6_features"][["cash_permission_score"]].shift(1).dropna()
        )

    return context


def _requested_feature_versions(candidates: list[dict] | None) -> set[str]:
    if candidates is None:
        candidates = _select_candidates(None)
    return {candidate["feature_version"] for candidate in candidates}


def _candidate_for_feature_version(
    candidates: list[dict] | None,
    feature_version: str,
) -> dict:
    if candidates is None:
        candidates = _select_candidates(None)
    for candidate in candidates:
        if candidate["feature_version"] == feature_version:
            return candidate
    raise ValueError(f"No candidate found for feature version: {feature_version}")


def _build_v3_features(
    returns: pd.DataFrame,
    base_config: dict,
    candidate: dict,
) -> pd.DataFrame:
    """Build guarded current-window V3 macro features."""
    features_config = _feature_config("v3", candidate)
    macro_path = features_config.get("macro_path")
    if not macro_path:
        raise ValueError(f"{candidate.get('name', 'V3 candidate')} requires features.macro_path.")
    if not Path(macro_path).exists():
        raise FileNotFoundError(f"V3 macro path does not exist: {macro_path}")

    macro_data = load_macro_data_from_csv(
        macro_path,
        date_column=features_config.get("macro_date_column", "date"),
    )
    coverage = build_macro_coverage_table(
        returns,
        macro_data,
        returns_path=base_config.get("data", {}).get("returns_path", "<in-memory>"),
        macro_path=macro_path,
    )
    validate_macro_coverage(coverage)

    features = build_features_v3(
        returns=returns,
        macro_data=macro_data,
        market_asset=features_config.get("market_asset", "SPY"),
        short_window=features_config.get("short_window", 4),
        long_window=features_config.get("long_window", 12),
        ewma_span=features_config.get("ewma_span", 12),
    )
    _validate_v3_feature_alignment(returns, features)
    return features


def _validate_v3_feature_alignment(
    returns: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    macro_columns = [
        column
        for column in features.columns
        if isinstance(column, str) and column.startswith("macro_")
    ]
    if not macro_columns:
        raise ValueError("V3 candidate produced no macro feature columns.")
    if features.index.max() > returns.index.max():
        raise ValueError("V3 feature dates overrun returns dates.")

    shifted_features = features.shift(1).dropna()
    aligned_index = returns.index[returns.index.isin(shifted_features.index)]
    aligned_features = shifted_features.loc[aligned_index]
    if aligned_features.empty:
        raise ValueError("V3 aligned features are empty after one-period shift.")
    if aligned_features.index.max() > returns.index.max():
        raise ValueError("V3 aligned feature dates overrun returns dates.")
    if aligned_features.loc[:, macro_columns].isna().any().any():
        raise ValueError("V3 aligned features contain missing macro values.")


def _build_v4_features(
    returns: pd.DataFrame,
    base_config: dict,
    candidate: dict,
) -> pd.DataFrame:
    """Build guarded current-window V4 real GARCH features."""
    features_config = _feature_config("v4", candidate)
    garch_features, diagnostics = build_garch_feature_set_by_mode(
        returns=returns,
        assets=list(returns.columns),
        market_asset=features_config.get("market_asset", "SPY"),
        include_relative=features_config.get("garch_include_relative", True),
        mode=features_config.get("garch_mode", GARCH_MODE_ROLLING_FITTED),
        periods_per_year=features_config.get("garch_periods_per_year", 52),
        min_history=features_config.get("garch_min_history", 104),
        window=features_config.get("garch_window", 156),
        annualize=features_config.get("garch_annualize", False),
        exclude_cash=features_config.get("garch_exclude_cash", True),
        fallback=features_config.get("garch_fallback", GARCH_FALLBACK_ROLLING_REALIZED),
        return_diagnostics=True,
    )
    _validate_v4_garch_diagnostics(
        diagnostics=diagnostics,
        candidate=candidate,
    )
    v2_features = build_features_v2(
        returns,
        market_asset=features_config.get("market_asset", "SPY"),
        short_window=features_config.get("short_window", 4),
        long_window=features_config.get("long_window", 12),
        ewma_span=features_config.get("ewma_span", 12),
    )
    features = pd.concat([v2_features, garch_features], axis=1, sort=False).dropna()
    if features.empty:
        raise ValueError(
            "V4_real_garch_current output is empty after aligning V2 and GARCH features."
        )
    if not features_config.get("include_garch_features", True):
        raise ValueError("V4_real_garch_current requires GARCH features to be enabled.")
    _validate_v4_feature_alignment(
        returns=returns,
        features=features,
        garch_features=garch_features,
        exclude_cash=features_config.get("garch_exclude_cash", True),
    )
    return features


def _build_v7_features(
    returns: pd.DataFrame,
    base_config: dict,
    candidate: dict,
) -> pd.DataFrame:
    """Build guarded current-window V7 real macro plus real GARCH features."""
    features_config = _feature_config("v7", candidate)
    macro_path = features_config.get("macro_path")
    if not macro_path:
        raise ValueError("V7_real_macro_garch_current requires features.macro_path.")
    if not Path(macro_path).exists():
        raise FileNotFoundError(f"V7 macro path does not exist: {macro_path}")
    macro_data = load_macro_data_from_csv(
        macro_path,
        date_column=features_config.get("macro_date_column", "date"),
    )
    coverage = build_macro_coverage_table(
        returns,
        macro_data,
        returns_path=base_config.get("data", {}).get("returns_path", "<in-memory>"),
        macro_path=macro_path,
    )
    validate_macro_coverage(coverage)
    features, diagnostics = build_features_v7(
        returns=returns,
        macro_data=macro_data,
        market_asset=features_config.get("market_asset", "SPY"),
        short_window=features_config.get("short_window", 4),
        long_window=features_config.get("long_window", 12),
        ewma_span=features_config.get("ewma_span", 12),
        garch_include_relative=features_config.get("garch_include_relative", True),
        garch_periods_per_year=features_config.get("garch_periods_per_year", 52),
        garch_mode=features_config.get("garch_mode", GARCH_MODE_ROLLING_FITTED),
        garch_min_history=features_config.get("garch_min_history", 104),
        garch_window=features_config.get("garch_window", 156),
        garch_annualize=features_config.get("garch_annualize", False),
        garch_exclude_cash=features_config.get("garch_exclude_cash", True),
        garch_fallback=features_config.get("garch_fallback", GARCH_FALLBACK_ROLLING_REALIZED),
        return_garch_diagnostics=True,
    )
    _validate_v3_feature_alignment(returns, features)
    _validate_v4_garch_diagnostics(diagnostics=diagnostics, candidate=candidate)
    garch_columns = [
        column
        for column in features.columns
        if isinstance(column, str) and column.startswith("garch_")
    ]
    _validate_v4_feature_alignment(
        returns=returns,
        features=features,
        garch_features=features.loc[:, garch_columns],
        exclude_cash=features_config.get("garch_exclude_cash", True),
    )
    return features


def _build_v8_features(
    returns: pd.DataFrame,
    base_config: dict,
    candidate: dict,
) -> pd.DataFrame:
    """Build guarded current-window V8 EWMA plus real GARCH features."""
    features_config = _feature_config("v8", candidate)
    features, diagnostics = build_features_v8(
        returns=returns,
        market_asset=features_config.get("market_asset", "SPY"),
        short_window=features_config.get("short_window", 4),
        long_window=features_config.get("long_window", 12),
        ewma_span=features_config.get("ewma_span", 12),
        ewma_lambda=features_config.get("ewma_lambda", 0.94),
        garch_include_relative=features_config.get("garch_include_relative", True),
        garch_periods_per_year=features_config.get("garch_periods_per_year", 52),
        garch_mode=features_config.get("garch_mode", GARCH_MODE_ROLLING_FITTED),
        garch_min_history=features_config.get("garch_min_history", 104),
        garch_window=features_config.get("garch_window", 156),
        garch_annualize=features_config.get("garch_annualize", False),
        garch_exclude_cash=features_config.get("garch_exclude_cash", True),
        garch_fallback=features_config.get("garch_fallback", GARCH_FALLBACK_ROLLING_REALIZED),
        return_diagnostics=True,
    )
    _validate_v4_garch_diagnostics(
        diagnostics=diagnostics["garch"],
        candidate=candidate,
    )
    _validate_v8_feature_alignment(
        returns=returns,
        features=features,
        exclude_cash=features_config.get("garch_exclude_cash", True),
    )
    return features


def _validate_v4_garch_diagnostics(
    diagnostics: pd.DataFrame,
    candidate: dict,
) -> None:
    if diagnostics is None or diagnostics.empty:
        raise ValueError("V4_real_garch_current requires fitted GARCH diagnostics.")
    allow_scipy_fallback = bool(candidate.get("garch_allow_scipy_fallback", False))
    fitted = diagnostics.loc[diagnostics["status"] == "fitted"]
    if fitted.empty:
        raise ValueError("V4_real_garch_current produced no fitted GARCH diagnostics.")
    if not bool(diagnostics["arch_available"].any()) and not allow_scipy_fallback:
        raise ValueError(
            "V4_real_garch_current requires arch_model unless scipy fallback is allowed."
        )
    if bool(diagnostics["arch_available"].any()):
        non_arch_backends = set(fitted["backend"].dropna()) - {"arch_model"}
        if non_arch_backends and not allow_scipy_fallback:
            raise ValueError(
                "V4_real_garch_current expected arch_model backend for fitted rows; "
                f"found {sorted(non_arch_backends)}."
            )
    fit_failures = diagnostics[
        diagnostics["fallback_reason"].fillna("").astype(str).str.contains("fit_failed")
    ]
    if not fit_failures.empty and not allow_scipy_fallback:
        raise ValueError("V4_real_garch_current encountered GARCH fit failures.")


def _validate_v4_feature_alignment(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    garch_features: pd.DataFrame,
    exclude_cash: bool,
) -> None:
    if features.index.max() > returns.index.max():
        raise ValueError("V4 feature dates overrun returns dates.")
    garch_columns = [
        column
        for column in features.columns
        if isinstance(column, str) and column.startswith("garch_")
    ]
    if not garch_columns:
        raise ValueError("V4_real_garch_current produced no fitted GARCH feature columns.")
    if exclude_cash:
        cash_columns = [
            column
            for column in garch_columns
            if column.endswith("_CASH") or "_CASH_" in column or column.endswith("CASH")
        ]
        if cash_columns:
            raise ValueError(
                "V4_real_garch_current produced CASH fitted GARCH columns while "
                f"CASH exclusion is enabled: {cash_columns}"
            )
        if any(column == "garch_vol_CASH" for column in garch_features.columns):
            raise ValueError("V4 fitted GARCH features include garch_vol_CASH.")

    shifted_features = features.shift(1).dropna()
    aligned_index = returns.index[returns.index.isin(shifted_features.index)]
    aligned_features = shifted_features.loc[aligned_index]
    if aligned_features.empty:
        raise ValueError("V4 aligned features are empty after one-period shift.")
    if aligned_features.index.max() > returns.index.max():
        raise ValueError("V4 aligned feature dates overrun returns dates.")
    if aligned_features.isna().any().any():
        raise ValueError("V4 aligned features contain missing values.")


def _validate_v8_feature_alignment(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    exclude_cash: bool,
) -> None:
    if features.index.max() > returns.index.max():
        raise ValueError("V8 feature dates overrun returns dates.")
    ewma_columns = [
        column
        for column in features.columns
        if isinstance(column, str) and column.startswith("ewma_vol_")
    ]
    garch_columns = [
        column
        for column in features.columns
        if isinstance(column, str) and column.startswith("garch_vol_")
    ]
    comparison_columns = [
        column
        for column in features.columns
        if isinstance(column, str)
        and (
            column.startswith("garch_minus_ewma_vol_")
            or column.startswith("garch_to_ewma_vol_ratio_")
        )
    ]
    if not ewma_columns:
        raise ValueError("V8_ewma_garch_vol_current produced no EWMA volatility columns.")
    if not garch_columns:
        raise ValueError("V8_ewma_garch_vol_current produced no GARCH volatility columns.")
    if not comparison_columns:
        raise ValueError("V8_ewma_garch_vol_current produced no comparison columns.")
    if exclude_cash:
        cash_columns = [
            column
            for column in ewma_columns + garch_columns + comparison_columns
            if column.endswith("_CASH") or "_CASH_" in column or column.endswith("CASH")
        ]
        if cash_columns:
            raise ValueError(
                "V8_ewma_garch_vol_current produced CASH volatility columns while "
                f"CASH exclusion is enabled: {cash_columns}"
            )

    shifted_features = features.shift(1).dropna()
    aligned_index = returns.index[returns.index.isin(shifted_features.index)]
    aligned_features = shifted_features.loc[aligned_index]
    if aligned_features.empty:
        raise ValueError("V8 aligned features are empty after one-period shift.")
    if aligned_features.index.max() > returns.index.max():
        raise ValueError("V8 aligned feature dates overrun returns dates.")
    if aligned_features.isna().any().any():
        raise ValueError("V8 aligned features contain missing values.")


def _candidate_raw_features(candidate: dict, context: dict[str, Any]) -> pd.DataFrame:
    feature_version = candidate["feature_version"]
    if feature_version == "v2":
        return context["v2_features"]
    if feature_version == "v3":
        return context["v3_features"]
    if feature_version == "v4":
        return context["v4_features"]
    if feature_version == "v7":
        return context["v7_features"]
    if feature_version == "v8":
        return context["v8_features"]
    if feature_version == "v6":
        return context["v6_features"]
    if feature_version == "v5":
        selected_columns = select_feature_columns(
            context["v5_features"].columns,
            {
                "variant": candidate["name"],
                "exclude_blocks": candidate.get("exclude_blocks", []),
                "include_only_blocks": candidate.get("include_only_blocks"),
            },
            context["block_map"],
        )
        return context["v5_features"].loc[:, selected_columns]
    raise ValueError(f"Unsupported candidate feature version: {feature_version}")


def _candidate_auxiliary_features(
    candidate: dict,
    context: dict[str, Any],
) -> pd.DataFrame | None:
    if not candidate.get("use_dynamic_cash", False):
        return None
    if candidate["feature_version"] == "v5":
        return context["v5_auxiliary"]
    if candidate["feature_version"] == "v6":
        return context["v6_auxiliary"]
    return None


def _build_candidate_run_config(
    base_config: dict,
    candidate: dict,
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
    config["environment"]["transaction_cost"] = 0.001
    config["features"] = _feature_config(candidate["feature_version"], candidate)
    config["reward"]["use_mandate_penalty"] = False

    if candidate.get("use_dynamic_cash", False):
        config["reward"].update(
            {
                "turnover_penalty_mode": "excess_linear",
                "turnover_free_band": 0.20,
                "turnover_quadratic_weight": 0.0,
                "use_cash_risk_off_penalty": True,
                "normal_cash_max": 0.10,
                "cash_penalty_weight": 0.025,
                "cash_risk_off_column": candidate["cash_risk_off_column"],
            }
        )
    else:
        config["reward"]["use_cash_risk_off_penalty"] = False
        config["reward"]["turnover_penalty_mode"] = "linear"
        config["reward"].pop("cash_risk_off_column", None)
    _validate_protocol_reward_semantics(config)
    return config


def _feature_config(version: str, candidate: dict | None = None) -> dict:
    if version == "v2":
        return {
            "version": "v2",
            "market_asset": "SPY",
            "short_window": 4,
            "long_window": 12,
            "ewma_span": 12,
        }
    if version == "v3":
        candidate = {} if candidate is None else candidate
        return {
            "version": "v3",
            "market_asset": "SPY",
            "short_window": 4,
            "long_window": 12,
            "ewma_span": 12,
            "macro_path": candidate.get("macro_path", DEFAULT_V3_MACRO_PATH),
            "macro_date_column": candidate.get("macro_date_column", "date"),
        }
    if version == "v4":
        candidate = {} if candidate is None else candidate
        return {
            "version": "v4",
            "market_asset": "SPY",
            "short_window": 4,
            "long_window": 12,
            "ewma_span": 12,
            "include_garch_features": True,
            "garch_include_relative": True,
            "garch_mode": candidate.get("garch_mode", GARCH_MODE_ROLLING_FITTED),
            "garch_min_history": candidate.get("garch_min_history", 104),
            "garch_window": candidate.get("garch_window", 156),
            "garch_periods_per_year": candidate.get("garch_periods_per_year", 52),
            "garch_annualize": candidate.get("garch_annualize", False),
            "garch_exclude_cash": candidate.get("garch_exclude_cash", True),
            "garch_fallback": candidate.get(
                "garch_fallback",
                GARCH_FALLBACK_ROLLING_REALIZED,
            ),
        }
    if version == "v7":
        candidate = {} if candidate is None else candidate
        return {
            "version": "v7",
            "market_asset": "SPY",
            "short_window": 4,
            "long_window": 12,
            "ewma_span": 12,
            "macro_path": candidate.get("macro_path", DEFAULT_V3_MACRO_PATH),
            "macro_date_column": candidate.get("macro_date_column", "date"),
            "garch_include_relative": True,
            "garch_mode": candidate.get("garch_mode", GARCH_MODE_ROLLING_FITTED),
            "garch_min_history": candidate.get("garch_min_history", 104),
            "garch_window": candidate.get("garch_window", 156),
            "garch_periods_per_year": candidate.get("garch_periods_per_year", 52),
            "garch_annualize": candidate.get("garch_annualize", False),
            "garch_exclude_cash": candidate.get("garch_exclude_cash", True),
            "garch_fallback": candidate.get(
                "garch_fallback",
                GARCH_FALLBACK_ROLLING_REALIZED,
            ),
        }
    if version == "v8":
        candidate = {} if candidate is None else candidate
        return {
            "version": "v8",
            "market_asset": "SPY",
            "short_window": 4,
            "long_window": 12,
            "ewma_span": 12,
            "ewma_lambda": candidate.get("ewma_lambda", 0.94),
            "garch_include_relative": True,
            "garch_mode": candidate.get("garch_mode", GARCH_MODE_ROLLING_FITTED),
            "garch_min_history": candidate.get("garch_min_history", 104),
            "garch_window": candidate.get("garch_window", 156),
            "garch_periods_per_year": candidate.get("garch_periods_per_year", 52),
            "garch_annualize": candidate.get("garch_annualize", False),
            "garch_exclude_cash": candidate.get("garch_exclude_cash", True),
            "garch_fallback": candidate.get(
                "garch_fallback",
                GARCH_FALLBACK_ROLLING_REALIZED,
            ),
        }
    if version == "v5":
        return {
            "version": "v5",
            "market_asset": "SPY",
            "short_window": 4,
            "long_window": 12,
            "ewma_span": 12,
            "correlation_window": 12,
            "drawdown_window": 12,
            "risk_off_threshold": 2.0,
        }
    if version == "v6":
        return {
            "version": "v6",
            "market_asset": "SPY",
            "short_window": 4,
            "medium_window": 12,
            "long_window": 26,
            "ewma_short_span": 4,
            "ewma_long_span": 12,
            "correlation_window": 12,
            "zscore_window": 52,
        }
    raise ValueError(f"Unsupported feature version: {version}")


def _select_candidates(candidate_names: list[str] | None) -> list[dict]:
    if candidate_names is None:
        return [
            deepcopy(candidate)
            for candidate in PROTOCOL_CANDIDATES
            if candidate.get("default_enabled", True)
        ]
    known = {candidate["name"]: candidate for candidate in PROTOCOL_CANDIDATES}
    missing = [name for name in candidate_names if name not in known]
    if missing:
        raise ValueError(f"Unknown candidates: {missing}")
    return [deepcopy(known[name]) for name in candidate_names]


def _validate_protocol_reward_semantics(config: dict) -> None:
    if "lambda_sharpe" in config.get("reward", {}):
        raise ValueError("reward.lambda_sharpe is inactive and must not be present.")


def _build_metadata(
    base_config_path: str,
    returns_path: str,
    output_dir: str,
    candidates: list[dict],
    folds: list[dict],
    actual_folds: pd.DataFrame,
    seeds: list[int],
    episodes: int,
    batch_size: int,
    actor_learning_rate: float,
    critic_learning_rate: float,
    base_config: dict,
    run_configs: dict[str, dict],
    smoke: bool,
) -> dict[str, Any]:
    return {
        "runner": "src.experiments.run_protocol_pure_td3_revalidation",
        "base_config_path": base_config_path,
        "returns_path": returns_path,
        "output_dir": output_dir,
        "candidates": [candidate["name"] for candidate in candidates],
        "feature_versions": {
            candidate["name"]: candidate["feature_version"] for candidate in candidates
        },
        "candidate_descriptions": {
            candidate["name"]: candidate["description"] for candidate in candidates
        },
        "candidate_metadata": {
            candidate["name"]: {
                key: candidate[key]
                for key in (
                    "macro_path",
                    "macro_date_column",
                    "macro_source",
                    "dollar_proxy",
                    "reason",
                )
                if key in candidate
            }
            for candidate in candidates
        },
        "seeds": seeds,
        "episodes": episodes,
        "batch_size": batch_size,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "folds": folds,
        "actual_folds": actual_folds.to_dict(orient="records"),
        "git_commit_hash": _git_commit_hash(),
        "transaction_cost_rate": base_config["environment"]["transaction_cost"],
        "timing_convention": TIMING_CONVENTION,
        "turnover_convention": "sum(abs(w_t - w_{t-1}))",
        "DSR_method_policy": DSR_METHOD,
        "lambda_sharpe_present_or_active": False,
        "robust_score_training_usage": "evaluation_only",
        "reward_config_base": base_config["reward"],
        "run_configs": run_configs,
        "smoke_mode": bool(smoke),
    }


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


def _write_yaml(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run protocol-pure TD3 revalidation for selected candidates.",
    )
    parser.add_argument("--base-config-path", default=BASE_CONFIG_PATH)
    parser.add_argument("--returns-path", default=RETURNS_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    result = run_protocol_pure_td3_revalidation(
        base_config_path=args.base_config_path,
        returns_path=args.returns_path,
        output_dir=args.output_dir,
        candidates=_parse_csv_strings(args.candidates),
        seeds=_parse_csv_ints(args.seeds),
        episodes=args.episodes,
        max_folds=args.max_folds,
        smoke=args.smoke,
    )
    print(f"Output folder: {result['output_dir']}")
    print("\nOverall aggregate:")
    print(result["overall_aggregate"].to_string(index=False))
    print("\nRobust score ranking:")
    print(result["robust_score"]["ranking"].to_string(index=False))


if __name__ == "__main__":
    main()
