"""Feature-block ablation runner for dominant-asset timing diagnostics.

This module builds controlled V2/V5 feature variants, runs walk-forward TD3
experiments, and attaches ex-post timing diagnostics. It is intentionally kept
outside the default training entry points: the reward, TD3 architecture,
environment dynamics, and shared training runner are not changed.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis.concentration_quality_diagnostics import (
    build_concentration_quality_report,
)
from src.analysis.decision_attribution import build_decision_attribution_report
from src.analysis.robust_score import build_robust_score_report
from src.backtest.allocation_diagnostics import allocation_diagnostics
from src.backtest.benchmarks import (
    buy_and_hold_returns,
    equal_weight_rebalanced_benchmark,
    equal_weight_returns,
    individual_buy_and_hold_returns,
)
from src.backtest.evaluate_agent import build_policy_history, flatten_transaction_cost_info
from src.backtest.evaluate_policy import summary_metrics
from src.backtest.performance_metrics import extended_summary_metrics
from src.data.build_dataset import build_returns_dataset
from src.data.features_v2 import build_features_v2
from src.data.features_v5 import (
    build_features_v5,
    build_v5_regime_auxiliary_features,
)
from src.data.normalize import normalize_train_validation_test
from src.data.walk_forward_split import slice_dataset_by_date
from src.env.portfolio_env import PortfolioEnv
from src.experiments.run_basic_experiment import summarize_metrics_table
from src.experiments.save_experiment_outputs import save_basic_experiment_outputs
from src.memory.replay_buffer import ReplayBuffer
from src.models.td3_agent import TD3Agent
from src.train.exploration import apply_behavior_exploration_noise
from src.utils.config import load_config
from src.utils.seed import set_seed


OUTPUT_DIR = "outputs/tables/v5_feature_block_ablation_timing_30ep_5seeds"
RETURNS_PATH = "data/processed/returns_weekly_latest.csv"
BASE_CONFIG_PATH = "configs/empirical_long_history.yaml"
SEEDS = [7, 21, 42, 84, 101]
EPISODES = 30
BATCH_SIZE = 32
ACTOR_LR = 0.0005
CRITIC_LR = 0.0005

EXPANDING_FOLDS = [
    {
        "fold_id": "F1",
        "description": "test_2022",
        "train_start": "2015-04-03",
        "train_end": "2020-12-25",
        "validation_start": "2021-01-01",
        "validation_end": "2021-12-31",
        "test_start": "2022-01-07",
        "test_end": "2022-12-30",
    },
    {
        "fold_id": "F2",
        "description": "test_2023",
        "train_start": "2015-04-03",
        "train_end": "2021-12-31",
        "validation_start": "2022-01-07",
        "validation_end": "2022-12-30",
        "test_start": "2023-01-06",
        "test_end": "2023-12-29",
    },
    {
        "fold_id": "F3",
        "description": "test_2024",
        "train_start": "2015-04-03",
        "train_end": "2022-12-30",
        "validation_start": "2023-01-06",
        "validation_end": "2023-12-29",
        "test_start": "2024-01-05",
        "test_end": "2024-12-27",
    },
    {
        "fold_id": "F4",
        "description": "test_2025_2026",
        "train_start": "2015-04-03",
        "train_end": "2023-12-29",
        "validation_start": "2024-01-05",
        "validation_end": "2024-12-27",
        "test_start": "2025-01-03",
        "test_end": "2026-05-15",
    },
]


FEATURE_VARIANTS = [
    {
        "variant": "V2_reference_full",
        "feature_version": "v2",
        "description": "Full V2 reference feature set with the V2 reference reward.",
        "exclude_blocks": [],
        "include_only_blocks": None,
        "use_v5_dynamic_cash": False,
    },
    {
        "variant": "V5_full_dynamic_cash_025",
        "feature_version": "v5",
        "description": "Full V5 feature set with dynamic CASH penalty weight 0.025.",
        "exclude_blocks": [],
        "include_only_blocks": None,
        "use_v5_dynamic_cash": True,
    },
    {
        "variant": "V5_no_momentum_block",
        "feature_version": "v5",
        "description": "V5 with asset and market momentum/trend signals removed.",
        "exclude_blocks": ["momentum"],
        "include_only_blocks": None,
        "use_v5_dynamic_cash": True,
    },
    {
        "variant": "V5_no_volatility_block",
        "feature_version": "v5",
        "description": "V5 with rolling and EWMA volatility signals removed.",
        "exclude_blocks": ["volatility"],
        "include_only_blocks": None,
        "use_v5_dynamic_cash": True,
    },
    {
        "variant": "V5_no_drawdown_block",
        "feature_version": "v5",
        "description": "V5 with rolling drawdown and drawdown-stress signals removed.",
        "exclude_blocks": ["drawdown"],
        "include_only_blocks": None,
        "use_v5_dynamic_cash": True,
    },
    {
        "variant": "V5_no_correlation_block",
        "feature_version": "v5",
        "description": "V5 with beta, correlation, hedge, and diversification signals removed.",
        "exclude_blocks": ["correlation"],
        "include_only_blocks": None,
        "use_v5_dynamic_cash": True,
    },
    {
        "variant": "V5_no_regime_block",
        "feature_version": "v5",
        "description": "V5 with explicit regime and risk-off state signals removed from observations.",
        "exclude_blocks": ["regime"],
        "include_only_blocks": None,
        "use_v5_dynamic_cash": True,
    },
    {
        "variant": "V5_momentum_only_or_minimal_momentum_regime",
        "feature_version": "v5",
        "description": "Minimal V5 using one-period returns plus momentum/trend signals.",
        "exclude_blocks": [],
        "include_only_blocks": ["base", "momentum"],
        "use_v5_dynamic_cash": True,
    },
]


BLOCK_DESCRIPTIONS = {
    "base": "One-period asset return features.",
    "momentum": "Asset momentum plus market momentum/trend features.",
    "volatility": "Rolling volatility, EWMA volatility, and high-volatility features.",
    "drawdown": "Asset and market rolling drawdown features.",
    "correlation": "Beta, asset-market correlation, pairwise correlation, and hedge signals.",
    "regime": "Explicit market regime, risk-off, and composite state flags.",
}


def run_feature_block_ablation(
    base_config_path: str = BASE_CONFIG_PATH,
    returns_path: str = RETURNS_PATH,
    output_dir: str = OUTPUT_DIR,
    folds: list[dict] | None = None,
    seeds: list[int] | None = None,
    episodes: int = EPISODES,
    batch_size: int = BATCH_SIZE,
    actor_learning_rate: float = ACTOR_LR,
    critic_learning_rate: float = CRITIC_LR,
    variants: list[dict] | None = None,
) -> dict:
    """Run the feature-block ablation experiment and write diagnostic outputs."""
    selected_folds = EXPANDING_FOLDS if folds is None else folds
    selected_seeds = SEEDS if seeds is None else seeds
    selected_variants = FEATURE_VARIANTS if variants is None else variants
    destination = Path(output_dir)
    configs_dir = destination / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    base_config = _build_base_config(base_config_path, returns_path, episodes)
    returns = build_returns_dataset_from_config(base_config, configs_dir / "_returns_config.yaml")
    raw_v2_features = build_features_v2(returns)
    raw_v5_features = build_features_v5(returns)
    raw_v5_auxiliary = build_v5_regime_auxiliary_features(returns).shift(1).dropna()

    block_map = build_feature_block_map(list(returns.columns))
    _write_feature_metadata(
        destination,
        selected_variants,
        block_map,
        raw_v2_features,
        raw_v5_features,
    )

    fold_rows = []
    metric_rows = []
    policy_history_paths = []
    policy_strategy_names = []
    variant_outputs = {}

    for variant in selected_variants:
        variant_name = variant["variant"]
        variant_outputs[variant_name] = {}
        raw_features = raw_v2_features if variant["feature_version"] == "v2" else raw_v5_features
        selected_columns = select_feature_columns(raw_features.columns, variant, block_map)
        selected_raw_features = raw_features.loc[:, selected_columns]
        for fold in selected_folds:
            datasets = build_ablation_fold_datasets(
                returns=returns,
                raw_features=selected_raw_features,
                raw_auxiliary_features=raw_v5_auxiliary,
                fold=fold,
                include_auxiliary=variant["use_v5_dynamic_cash"],
            )
            fold_rows.append(_actual_fold_row(fold, datasets))
            for seed in selected_seeds:
                config = _build_run_config(
                    base_config,
                    variant,
                    seed=seed,
                    episodes=episodes,
                    batch_size=batch_size,
                    actor_learning_rate=actor_learning_rate,
                    critic_learning_rate=critic_learning_rate,
                )
                config_path = configs_dir / f"{fold['fold_id']}_{variant_name}_seed_{seed}.yaml"
                _write_yaml(config, config_path)
                raw_result = train_td3_ablation_on_datasets(datasets, config)
                experiment_result = _build_experiment_result(raw_result)
                experiment_name = f"{fold['fold_id']}_{variant_name}_seed_{seed}"
                saved_paths = save_basic_experiment_outputs(
                    experiment_result,
                    output_dir=str(destination),
                    experiment_name=experiment_name,
                )
                variant_outputs[variant_name][f"{fold['fold_id']}_seed_{seed}"] = {
                    "saved_paths": saved_paths,
                    "raw_result": raw_result,
                }
                for split_name in ("validation", "test"):
                    metric_rows.append(
                        _metric_row(
                            variant=variant_name,
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
                policy_history_paths.append(saved_paths["test_policy_history"])
                policy_strategy_names.append(variant_name)

    pd.DataFrame(fold_rows).drop_duplicates().to_csv(
        destination / "actual_fold_dates.csv",
        index=False,
    )
    seed_results = pd.DataFrame(metric_rows)
    seed_results.to_csv(destination / "seed_fold_strategy_results.csv", index=False)
    overall = aggregate_metric_rows(seed_results, group_columns=["strategy", "split"])
    overall.to_csv(destination / "overall_aggregate_by_strategy_split.csv", index=False)
    fold_level = aggregate_metric_rows(
        seed_results,
        group_columns=["strategy", "split", "fold"],
    )
    fold_level.to_csv(destination / "fold_level_aggregate_by_strategy_split.csv", index=False)

    cash_attribution = build_cash_attribution_summary(
        policy_history_paths,
        policy_strategy_names,
        raw_v5_auxiliary,
    )
    cash_attribution["by_file"].to_csv(
        destination / "test_cash_attribution_by_drl_strategy.csv",
        index=False,
    )
    cash_attribution["aggregate"].to_csv(
        destination / "test_cash_attribution_aggregate_by_drl_strategy.csv",
        index=False,
    )

    concentration = build_concentration_quality_report(
        policy_history_paths,
        asset_returns_path=returns_path,
        strategy_names=policy_strategy_names,
        horizons=[12],
        output_dir=str(destination / "concentration_quality"),
    )
    concentration_aggregate = _aggregate_concentration_quality(concentration["summary"])
    concentration_aggregate.to_csv(
        destination / "test_concentration_quality_h12_aggregate_by_drl_strategy.csv",
        index=False,
    )

    decision = build_decision_attribution_report(
        comparison_dir=str(destination),
        returns_path=returns_path,
        strategies=[variant["variant"] for variant in selected_variants],
        horizons=[1, 4, 12],
        output_dir=str(destination / "decision_attribution"),
    )
    robust = build_robust_score_report(str(destination), output_dir=str(destination))

    return {
        "output_dir": str(destination),
        "seed_results": seed_results,
        "overall_aggregate": overall,
        "cash_attribution": cash_attribution,
        "concentration_quality_h12": concentration_aggregate,
        "decision_attribution": decision,
        "robust_score": robust,
        "variant_outputs": variant_outputs,
    }


def build_returns_dataset_from_config(config: dict, config_path: Path) -> pd.DataFrame:
    """Persist a temporary config and use the shared returns loader."""
    _write_yaml(config, config_path)
    return build_returns_dataset(str(config_path))


def build_feature_block_map(asset_names: list[str]) -> dict[str, list[str]]:
    """Return explicit feature-block column definitions for V2/V5."""
    risky_assets = [asset for asset in asset_names if asset != "CASH"]
    blocks = {block: [] for block in BLOCK_DESCRIPTIONS}
    for asset in asset_names:
        blocks["base"].append(f"{asset}_ret_1p")
        blocks["momentum"].extend([f"{asset}_mom_4p", f"{asset}_mom_12p"])
        blocks["volatility"].extend(
            [f"{asset}_vol_4p", f"{asset}_vol_12p", f"{asset}_ewma_vol_12p"]
        )
        if asset != "SPY":
            blocks["correlation"].extend(
                [f"{asset}_beta_vs_SPY_12p", f"{asset}_corr_vs_SPY_12p"]
            )
        blocks["drawdown"].append(f"{asset}_rolling_drawdown_12p")

    blocks["momentum"].extend(
        [
            "market_trend_regime",
            "market_defensive_regime",
        ]
    )
    blocks["volatility"].extend(
        [
            "market_high_vol_regime",
            "regime_market_high_vol",
        ]
    )
    blocks["drawdown"].extend(
        [
            "regime_market_drawdown_stress",
        ]
    )
    blocks["correlation"].extend(
        [
            "avg_pairwise_corr_12p",
            "correlation_stress",
            "diversification_benefit_score",
            "tlt_equity_hedge_signal",
            "gld_equity_hedge_signal",
        ]
    )
    blocks["regime"].extend(
        [
            "market_high_vol_regime",
            "market_risk_off_regime",
            "market_trend_regime",
            "market_defensive_regime",
            "regime_market_drawdown_stress",
            "regime_market_high_vol",
            "correlation_stress",
            "risk_off_score",
            "risk_off_state",
            "tlt_equity_hedge_signal",
            "gld_equity_hedge_signal",
        ]
    )
    return {block: sorted(set(columns)) for block, columns in blocks.items()}


def select_feature_columns(
    available_columns,
    variant: dict,
    block_map: dict[str, list[str]],
) -> list[str]:
    """Select final feature columns for a variant from explicit block definitions."""
    available = list(available_columns)
    available_set = set(available)
    if variant.get("include_only_blocks") is not None:
        selected = set()
        for block in variant["include_only_blocks"]:
            selected.update(column for column in block_map[block] if column in available_set)
        return [column for column in available if column in selected]

    excluded = set()
    for block in variant.get("exclude_blocks", []):
        excluded.update(column for column in block_map[block] if column in available_set)
    selected_columns = [column for column in available if column not in excluded]
    if not selected_columns:
        raise ValueError(f"Variant {variant['variant']} selected no feature columns.")
    return selected_columns


def build_ablation_fold_datasets(
    returns: pd.DataFrame,
    raw_features: pd.DataFrame,
    raw_auxiliary_features: pd.DataFrame,
    fold: dict,
    include_auxiliary: bool,
) -> dict:
    """Build shifted, sliced, and normalized datasets for one ablation fold."""
    features_available_before_return = raw_features.shift(1).dropna()
    train_returns, train_features = slice_dataset_by_date(
        returns,
        features_available_before_return,
        fold["train_start"],
        fold["train_end"],
    )
    validation_returns, validation_features = slice_dataset_by_date(
        returns,
        features_available_before_return,
        fold["validation_start"],
        fold["validation_end"],
    )
    test_returns, test_features = slice_dataset_by_date(
        returns,
        features_available_before_return,
        fold["test_start"],
        fold["test_end"],
    )
    train_features_norm, validation_features_norm, test_features_norm, scaler = (
        normalize_train_validation_test(train_features, validation_features, test_features)
    )
    datasets = {
        "train_returns": train_returns,
        "validation_returns": validation_returns,
        "test_returns": test_returns,
        "train_features": train_features_norm,
        "validation_features": validation_features_norm,
        "test_features": test_features_norm,
        "feature_scaler": scaler,
    }
    if include_auxiliary:
        datasets["train_auxiliary_features"] = raw_auxiliary_features.loc[train_returns.index]
        datasets["validation_auxiliary_features"] = raw_auxiliary_features.loc[
            validation_returns.index
        ]
        datasets["test_auxiliary_features"] = raw_auxiliary_features.loc[test_returns.index]
    return datasets


def train_td3_ablation_on_datasets(datasets: dict, config: dict) -> dict:
    """Run TD3 for ablation datasets, with optional auxiliary reward features."""
    training_config = config["training"]
    environment_config = config["environment"]
    td3_config = config["td3"]
    reward_config = config["reward"]
    set_seed(training_config["seed"])
    exploration_rng = np.random.default_rng(training_config["seed"])
    exploration_noise = float(training_config.get("exploration_noise", 0.0))
    exploration_noise_clip = training_config.get("exploration_noise_clip")
    active_max_weight = _active_max_weight_cap(environment_config)

    env = PortfolioEnv(
        returns=datasets["train_returns"],
        features=datasets["train_features"],
        auxiliary_features=datasets.get("train_auxiliary_features"),
        initial_cash=environment_config["initial_cash"],
        transaction_cost=environment_config["transaction_cost"],
        transaction_cost_mode=environment_config.get("transaction_cost_mode", "scalar"),
        asset_transaction_cost_bps=environment_config.get("asset_transaction_cost_bps"),
        reward_config=reward_config,
    )
    agent = TD3Agent(
        state_dim=env.observation_dim,
        action_dim=env.n_assets,
        actor_learning_rate=td3_config["actor_learning_rate"],
        critic_learning_rate=td3_config["critic_learning_rate"],
        gamma=td3_config["gamma"],
        tau=td3_config["tau"],
        policy_noise=td3_config["policy_noise"],
        noise_clip=td3_config["noise_clip"],
        policy_delay=td3_config["policy_delay"],
        max_weight_cap=getattr(env, "max_weight_cap", active_max_weight),
    )
    replay_buffer = ReplayBuffer(
        state_dim=env.observation_dim,
        action_dim=env.n_assets,
        max_size=td3_config["replay_buffer_size"],
        seed=training_config["seed"],
    )

    episode_logs = []
    for episode in range(1, training_config["episodes"] + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        latest_losses = {"critic_1_loss": None, "critic_2_loss": None, "actor_loss": None}
        info = {"portfolio_value": env.portfolio_value}
        episode_turnover = []
        episode_transaction_costs = []
        episode_weights = []
        while not done:
            deterministic_action = agent.select_action(state)
            action = apply_behavior_exploration_noise(
                deterministic_action,
                noise_std=exploration_noise,
                rng=exploration_rng,
                noise_clip=exploration_noise_clip,
                max_weight=getattr(env, "max_weight_cap", active_max_weight),
            )
            next_state, reward, done, info = env.step(action)
            executed_action = info.get("executed_action", info["weights"])
            replay_buffer.add(state, executed_action, reward, next_state, done)
            episode_turnover.append(info["turnover"])
            episode_transaction_costs.append(info["transaction_cost"])
            episode_weights.append(info["weights"])
            if len(replay_buffer) >= td3_config["batch_size"]:
                latest_losses = agent.train_step(replay_buffer.sample(td3_config["batch_size"]))
            state = next_state
            total_reward += reward
            steps += 1

        final_weights = {
            asset_name: float(weight)
            for asset_name, weight in zip(env.asset_names, episode_weights[-1])
        }
        episode_logs.append(
            {
                "episode": episode,
                "final_portfolio_value": info["portfolio_value"],
                "total_reward": total_reward,
                "steps": steps,
                "average_turnover": sum(episode_turnover) / steps,
                "average_transaction_cost": sum(episode_transaction_costs) / steps,
                "final_weights": final_weights,
                "max_weight": max(final_weights.values()),
                "cash_weight": final_weights.get("CASH", 0.0),
                "critic_1_loss": latest_losses["critic_1_loss"],
                "critic_2_loss": latest_losses["critic_2_loss"],
                "actor_loss": latest_losses["actor_loss"],
            }
        )

    validation_evaluation = evaluate_agent_ablation(
        agent,
        datasets["validation_returns"],
        datasets["validation_features"],
        datasets.get("validation_auxiliary_features"),
        config,
    )
    test_evaluation = evaluate_agent_ablation(
        agent,
        datasets["test_returns"],
        datasets["test_features"],
        datasets.get("test_auxiliary_features"),
        config,
    )
    validation_comparison = compare_agent_to_basic_benchmarks_ablation(
        agent,
        datasets["validation_returns"],
        datasets["validation_features"],
        datasets.get("validation_auxiliary_features"),
        config,
    )
    test_comparison = compare_agent_to_basic_benchmarks_ablation(
        agent,
        datasets["test_returns"],
        datasets["test_features"],
        datasets.get("test_auxiliary_features"),
        config,
    )
    return {
        "agent": agent,
        "replay_buffer": replay_buffer,
        "episode_logs": episode_logs,
        "train_returns": datasets["train_returns"],
        "train_features": datasets["train_features"],
        "validation_returns": datasets["validation_returns"],
        "validation_features": datasets["validation_features"],
        "test_returns": datasets["test_returns"],
        "test_features": datasets["test_features"],
        "validation_evaluation": validation_evaluation,
        "test_evaluation": test_evaluation,
        "validation_comparison": validation_comparison,
        "test_comparison": test_comparison,
    }


def _active_max_weight_cap(environment_config: dict) -> float | None:
    max_weight = environment_config.get("max_weight_per_asset")
    if max_weight is None:
        return None
    max_weight = float(max_weight)
    if max_weight >= 1.0:
        return None
    return max_weight


def evaluate_agent_ablation(
    agent,
    returns: pd.DataFrame,
    features: pd.DataFrame,
    auxiliary_features: pd.DataFrame | None,
    config: dict,
) -> dict:
    """Evaluate an agent with optional auxiliary reward features."""
    env_config = config["environment"]
    episode = run_policy_episode_ablation(
        agent,
        returns,
        features,
        auxiliary_features,
        initial_cash=env_config["initial_cash"],
        transaction_cost=env_config["transaction_cost"],
        transaction_cost_mode=env_config.get("transaction_cost_mode", "scalar"),
        asset_transaction_cost_bps=env_config.get("asset_transaction_cost_bps"),
        reward_config=config["reward"],
    )
    return {
        "episode": episode,
        "metrics": summary_metrics(episode["financial_net_returns"]),
        "diagnostics": summarize_episode_diagnostics_ablation(episode),
        "policy_history": build_policy_history(episode),
    }


def run_policy_episode_ablation(
    agent,
    returns: pd.DataFrame,
    features: pd.DataFrame,
    auxiliary_features: pd.DataFrame | None,
    initial_cash: float,
    transaction_cost: float,
    reward_config: dict,
    transaction_cost_mode: str = "scalar",
    asset_transaction_cost_bps: dict | None = None,
) -> dict:
    """Run one policy episode with optional auxiliary reward features."""
    env = PortfolioEnv(
        returns=returns,
        features=features,
        auxiliary_features=auxiliary_features,
        initial_cash=initial_cash,
        transaction_cost=transaction_cost,
        transaction_cost_mode=transaction_cost_mode,
        asset_transaction_cost_bps=asset_transaction_cost_bps,
        reward_config=reward_config,
    )
    state = env.reset()
    done = False
    rows = {
        "rewards": [],
        "policy_returns": [],
        "financial_net_returns": [],
        "portfolio_values": [],
        "turnover": [],
        "transaction_costs": [],
        "drawdown": [],
        "concentration": [],
        "weights": [],
    }
    info_frames = []
    transaction_cost_info_frames = []
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        rows["rewards"].append(reward)
        rows["policy_returns"].append(info["portfolio_return"])
        rows["financial_net_returns"].append(info["financial_net_return"])
        rows["portfolio_values"].append(info["portfolio_value"])
        rows["turnover"].append(info["turnover"])
        rows["transaction_costs"].append(info["transaction_cost"])
        rows["drawdown"].append(info["drawdown"])
        rows["concentration"].append(info["concentration"])
        rows["weights"].append(info["weights"])
        info_frames.append(
            {
                key: info.get(key, pd.NA)
                for key in (
                    "turnover_penalty",
                    "turnover_penalty_mode",
                    "turnover_free_band",
                    "turnover_excess",
                    "cash_penalty",
                    "cash_breach",
                    "normal_cash_max",
                    "cash_risk_off_state",
                    "cash_risk_off_column",
                )
            }
        )
        transaction_cost_info_frames.append(flatten_transaction_cost_info(info))
        state = next_state

    index = env.returns.index
    episode = {
        key: pd.Series(values, index=index, name=key)
        for key, values in rows.items()
        if key != "weights"
    }
    episode["weights"] = pd.DataFrame(rows["weights"], index=index, columns=env.asset_names)
    episode["turnover_reward_info"] = pd.DataFrame(info_frames, index=index)
    episode["transaction_cost_info"] = pd.DataFrame(
        transaction_cost_info_frames,
        index=index,
    )
    episode["final_portfolio_value"] = rows["portfolio_values"][-1]
    return episode


def summarize_episode_diagnostics_ablation(episode: dict) -> dict:
    """Summarize allocation diagnostics for an ablation episode."""
    final_weights = {
        asset_name: float(weight) for asset_name, weight in episode["weights"].iloc[-1].items()
    }
    allocation_summary = allocation_diagnostics(
        episode["weights"],
        turnover=episode["turnover"],
        transaction_costs=episode["transaction_costs"],
    )
    return {
        "final_portfolio_value": float(episode["final_portfolio_value"]),
        **allocation_summary,
        "final_weights": final_weights,
        "max_weight": allocation_summary["final_max_weight"],
        "cash_weight": allocation_summary["final_cash_weight"],
    }


def compare_agent_to_basic_benchmarks_ablation(
    agent,
    returns: pd.DataFrame,
    features: pd.DataFrame,
    auxiliary_features: pd.DataFrame | None,
    config: dict,
) -> dict:
    """Compare ablation agent with basic benchmarks in memory."""
    agent_evaluation = evaluate_agent_ablation(agent, returns, features, auxiliary_features, config)
    agent_returns = agent_evaluation["episode"]["financial_net_returns"]
    aligned_returns = returns.loc[agent_returns.index]
    equal_weight_gross_series = equal_weight_returns(aligned_returns)
    equal_weight_rebalanced = equal_weight_rebalanced_benchmark(
        aligned_returns,
        transaction_cost=config["environment"]["transaction_cost"],
    )
    equal_weight_rebalanced_net_series = equal_weight_rebalanced["net_returns"]
    individual_buy_hold_series = individual_buy_and_hold_returns(aligned_returns)
    policy_return_series = {
        "agent": agent_returns,
        "equal_weight_gross": equal_weight_gross_series,
        "equal_weight_rebalanced_net": equal_weight_rebalanced_net_series,
        "buy_and_hold": buy_and_hold_returns(aligned_returns),
        **individual_buy_hold_series,
    }
    market_returns = aligned_returns["SPY"] if "SPY" in aligned_returns.columns else None
    policy_metrics = {
        name: _policy_summary_metrics(
            policy_returns=series,
            benchmark_returns=equal_weight_rebalanced_net_series,
            market_returns=market_returns,
        )
        for name, series in policy_return_series.items()
    }
    return {
        "agent": agent_evaluation,
        "benchmarks": {},
        "metrics_table": pd.DataFrame(policy_metrics).T,
    }


def _policy_summary_metrics(
    policy_returns: pd.Series,
    benchmark_returns: pd.Series,
    market_returns: pd.Series | None,
) -> dict:
    base_metrics = summary_metrics(policy_returns)
    extended = extended_summary_metrics(
        policy_returns,
        benchmark_returns=benchmark_returns,
        market_returns=market_returns,
    )
    result = {
        **base_metrics,
        "sortino_ratio": extended["sortino_ratio"],
        "calmar_ratio": extended["calmar_ratio"],
        "tracking_error_vs_equal_weight_rebalanced_net": extended["tracking_error"],
        "information_ratio_vs_equal_weight_rebalanced_net": extended["information_ratio"],
    }
    if "capm_beta" in extended:
        result["capm_beta_vs_SPY"] = extended["capm_beta"]
        result["capm_alpha_vs_SPY"] = extended["capm_alpha"]
    return result


def aggregate_metric_rows(metrics: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Aggregate saved fold/seed metric rows."""
    rows = []
    for keys, group in metrics.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["strategy_type"] = "drl"
        row["n_folds"] = group["fold"].nunique()
        row["n_seeds"] = group["seed"].nunique()
        row["n_observations"] = len(group)
        row["mean_sharpe"] = group["sharpe_ratio"].mean()
        row["std_sharpe"] = group["sharpe_ratio"].std()
        row["robust_sharpe_score_05"] = row["mean_sharpe"] - 0.5 * (
            0.0 if pd.isna(row["std_sharpe"]) else row["std_sharpe"]
        )
        for source, target in (
            ("sortino_ratio", "mean_sortino"),
            ("calmar_ratio", "mean_calmar"),
            ("cumulative_return", "mean_cumulative_return"),
            ("annualized_return", "mean_annualized_return"),
            ("annualized_volatility", "mean_annualized_volatility"),
            ("max_drawdown", "mean_max_drawdown"),
            ("average_turnover", "mean_average_turnover"),
            ("average_effective_number_of_assets", "mean_average_effective_number_of_assets"),
            ("average_max_weight", "mean_average_max_weight"),
            ("cash_weight", "mean_cash_weight"),
            ("cash_above_10_rate", "cash_above_10_rate"),
            ("unjustified_cash_excess", "unjustified_cash_excess"),
            ("cash_penalty", "mean_cash_penalty"),
            ("cash_breach", "mean_cash_breach"),
            ("turnover_penalty", "mean_turnover_penalty"),
            ("transaction_cost", "mean_transaction_cost"),
        ):
            row[target] = group[source].mean()
        row["worst_max_drawdown"] = group["max_drawdown"].min()
        rows.append(row)
    return pd.DataFrame(rows)


def build_cash_attribution_summary(
    policy_history_paths: list[str],
    strategy_names: list[str],
    raw_auxiliary_features: pd.DataFrame,
) -> dict:
    """Build simple CASH attribution summaries for saved test policies."""
    rows = []
    for path, strategy in zip(policy_history_paths, strategy_names):
        history = pd.read_csv(path)
        history["date"] = pd.to_datetime(history["date"], errors="coerce")
        signals = raw_auxiliary_features[["risk_off_state"]].copy()
        signals["date"] = signals.index
        merged = history.merge(signals, how="left", on="date")
        cash_weight = pd.to_numeric(merged.get("cash_weight", 0.0), errors="coerce").fillna(0.0)
        risk_off = pd.to_numeric(merged["risk_off_state"], errors="coerce").fillna(0.0) >= 0.5
        cash_excess = (cash_weight - 0.10).clip(lower=0.0)
        row = {
            "strategy": strategy,
            "policy_history_path": path,
            "n_observations": len(merged),
            "mean_cash_weight": cash_weight.mean(),
            "cash_above_10_rate": (cash_weight > 0.10).mean(),
            "risk_off_rate": risk_off.mean(),
            "high_cash_in_risk_off_rate": ((cash_weight > 0.10) & risk_off).mean(),
            "mean_unjustified_cash_excess": cash_excess.where(~risk_off, 0.0).mean(),
            "mean_cash_penalty": pd.to_numeric(
                merged.get("cash_penalty", pd.Series(0.0, index=merged.index)),
                errors="coerce",
            ).fillna(0.0).mean(),
            "mean_cash_breach": pd.to_numeric(
                merged.get("cash_breach", pd.Series(0.0, index=merged.index)),
                errors="coerce",
            ).fillna(0.0).mean(),
        }
        rows.append(row)
    by_file = pd.DataFrame(rows)
    aggregate = (
        by_file.groupby("strategy", as_index=False)
        .agg(
            n_observations=("n_observations", "sum"),
            mean_cash_weight=("mean_cash_weight", "mean"),
            cash_above_10_rate=("cash_above_10_rate", "mean"),
            risk_off_rate=("risk_off_rate", "mean"),
            high_cash_in_risk_off_rate=("high_cash_in_risk_off_rate", "mean"),
            mean_unjustified_cash_excess=("mean_unjustified_cash_excess", "mean"),
            mean_cash_penalty=("mean_cash_penalty", "mean"),
            mean_cash_breach=("mean_cash_breach", "mean"),
        )
        .reset_index(drop=True)
    )
    return {"by_file": by_file, "aggregate": aggregate}


def _aggregate_concentration_quality(summary: pd.DataFrame) -> pd.DataFrame:
    h12 = summary.loc[summary["horizon"] == 12].copy()
    return (
        h12.groupby(["strategy_name", "horizon"], as_index=False)
        .mean(numeric_only=True)
        .rename(columns={"strategy_name": "strategy"})
    )


def _build_base_config(base_config_path: str, returns_path: str, episodes: int) -> dict:
    config = load_config(base_config_path)
    returns = pd.read_csv(returns_path)
    max_date = pd.to_datetime(returns["date"]).max().date().isoformat()
    config["data"]["returns_path"] = returns_path
    config["data"]["returns_date_column"] = "date"
    config["data"]["end_date"] = max_date
    config["training"]["episodes"] = episodes
    return config


def _build_run_config(
    base_config: dict,
    variant: dict,
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
    config["features"] = _feature_config(variant["feature_version"])
    config["reward"]["use_mandate_penalty"] = False
    if variant["use_v5_dynamic_cash"]:
        config["reward"].update(
            {
                "turnover_penalty_mode": "excess_linear",
                "turnover_free_band": 0.20,
                "turnover_quadratic_weight": 0.0,
                "use_cash_risk_off_penalty": True,
                "normal_cash_max": 0.10,
                "cash_penalty_weight": 0.025,
                "cash_risk_off_column": "risk_off_state",
            }
        )
    else:
        config["reward"]["use_cash_risk_off_penalty"] = False
        config["reward"]["turnover_penalty_mode"] = "linear"
    return config


def _feature_config(version: str) -> dict:
    config = {
        "version": version,
        "market_asset": "SPY",
        "short_window": 4,
        "long_window": 12,
        "ewma_span": 12,
    }
    if version == "v5":
        config.update(
            {
                "correlation_window": 12,
                "drawdown_window": 12,
                "risk_off_threshold": 2.0,
            }
        )
    return config


def _build_experiment_result(raw_result: dict) -> dict:
    episode_logs = raw_result["episode_logs"]
    return {
        "training_summary": {
            "total_episodes": len(episode_logs),
            "final_episode": episode_logs[-1]["episode"],
            "final_portfolio_value": episode_logs[-1]["final_portfolio_value"],
            "final_total_reward": episode_logs[-1]["total_reward"],
            "final_average_turnover": episode_logs[-1]["average_turnover"],
            "final_average_transaction_cost": episode_logs[-1][
                "average_transaction_cost"
            ],
            "final_max_weight": episode_logs[-1]["max_weight"],
            "final_cash_weight": episode_logs[-1]["cash_weight"],
        },
        "validation_metrics_table": raw_result["validation_comparison"]["metrics_table"],
        "test_metrics_table": raw_result["test_comparison"]["metrics_table"],
        "validation_comparison_summary": summarize_metrics_table(
            raw_result["validation_comparison"]["metrics_table"]
        ),
        "test_comparison_summary": summarize_metrics_table(
            raw_result["test_comparison"]["metrics_table"]
        ),
        "validation_diagnostics": raw_result["validation_evaluation"]["diagnostics"],
        "test_diagnostics": raw_result["test_evaluation"]["diagnostics"],
        "validation_policy_history": raw_result["validation_evaluation"]["policy_history"],
        "test_policy_history": raw_result["test_evaluation"]["policy_history"],
    }


def _metric_row(
    variant: str,
    fold: dict,
    seed: int,
    split: str,
    metrics: pd.Series,
    diagnostics: dict,
    policy_history: pd.DataFrame,
) -> dict:
    cash_weight = pd.to_numeric(policy_history.get("cash_weight", 0.0), errors="coerce")
    cash_excess = (cash_weight - 0.10).clip(lower=0.0)
    risk_off = pd.to_numeric(
        policy_history.get("cash_risk_off_state", pd.Series(0.0, index=policy_history.index)),
        errors="coerce",
    ).fillna(0.0) >= 0.5
    return {
        "strategy": variant,
        "fold": fold["fold_id"],
        "seed": seed,
        "split": split,
        "sharpe_ratio": metrics["sharpe_ratio"],
        "sortino_ratio": metrics["sortino_ratio"],
        "calmar_ratio": metrics["calmar_ratio"],
        "cumulative_return": metrics["cumulative_return"],
        "annualized_return": metrics["annualized_return"],
        "annualized_volatility": metrics["annualized_volatility"],
        "max_drawdown": metrics["max_drawdown"],
        "average_turnover": diagnostics["average_turnover"],
        "average_effective_number_of_assets": diagnostics[
            "average_effective_number_of_assets"
        ],
        "average_max_weight": diagnostics["average_max_weight"],
        "cash_weight": cash_weight.mean(),
        "cash_above_10_rate": (cash_weight > 0.10).mean(),
        "unjustified_cash_excess": cash_excess.where(~risk_off, 0.0).mean(),
        "cash_penalty": pd.to_numeric(
            policy_history.get("cash_penalty", pd.Series(0.0, index=policy_history.index)),
            errors="coerce",
        ).fillna(0.0).mean(),
        "cash_breach": pd.to_numeric(
            policy_history.get("cash_breach", pd.Series(0.0, index=policy_history.index)),
            errors="coerce",
        ).fillna(0.0).mean(),
        "turnover_penalty": pd.to_numeric(
            policy_history.get("turnover_penalty", pd.Series(0.0, index=policy_history.index)),
            errors="coerce",
        ).fillna(0.0).mean(),
        "transaction_cost": pd.to_numeric(
            policy_history.get("transaction_cost", pd.Series(0.0, index=policy_history.index)),
            errors="coerce",
        ).fillna(0.0).mean(),
    }


def _actual_fold_row(fold: dict, datasets: dict) -> dict:
    return {
        "fold": fold["fold_id"],
        "train_start": datasets["train_returns"].index.min().date().isoformat(),
        "train_end": datasets["train_returns"].index.max().date().isoformat(),
        "n_train": len(datasets["train_returns"]),
        "validation_start": datasets["validation_returns"].index.min().date().isoformat(),
        "validation_end": datasets["validation_returns"].index.max().date().isoformat(),
        "n_validation": len(datasets["validation_returns"]),
        "test_start": datasets["test_returns"].index.min().date().isoformat(),
        "test_end": datasets["test_returns"].index.max().date().isoformat(),
        "n_test": len(datasets["test_returns"]),
    }


def _write_feature_metadata(
    destination: Path,
    variants: list[dict],
    block_map: dict[str, list[str]],
    v2_features: pd.DataFrame,
    v5_features: pd.DataFrame,
) -> None:
    block_rows = []
    for block, columns in block_map.items():
        for column in columns:
            block_rows.append(
                {
                    "block": block,
                    "description": BLOCK_DESCRIPTIONS[block],
                    "column": column,
                }
            )
    pd.DataFrame(block_rows).to_csv(destination / "feature_block_columns.csv", index=False)

    definition_rows = []
    column_rows = []
    for variant in variants:
        features = v2_features if variant["feature_version"] == "v2" else v5_features
        selected = select_feature_columns(features.columns, variant, block_map)
        definition_rows.append(
            {
                "variant": variant["variant"],
                "feature_version": variant["feature_version"],
                "description": variant["description"],
                "excluded_blocks": ",".join(variant.get("exclude_blocks") or []),
                "included_only_blocks": ",".join(variant.get("include_only_blocks") or []),
                "n_features": len(selected),
            }
        )
        for order, column in enumerate(selected):
            column_rows.append(
                {
                    "variant": variant["variant"],
                    "feature_order": order,
                    "feature_column": column,
                }
            )
    pd.DataFrame(definition_rows).to_csv(
        destination / "feature_block_definitions.csv",
        index=False,
    )
    pd.DataFrame(column_rows).to_csv(
        destination / "feature_columns_by_variant.csv",
        index=False,
    )


def _write_yaml(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


if __name__ == "__main__":
    report = run_feature_block_ablation()
    print(f"Output folder: {report['output_dir']}")
    print("\nOverall aggregate:")
    print(report["overall_aggregate"].to_string(index=False))
    print("\nRobust score ranking:")
    print(report["robust_score"]["ranking"].to_string(index=False))
