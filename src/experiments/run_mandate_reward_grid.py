"""Controlled mini-grid runner for mandate-aware reward calibration.

This module varies only mandate penalty configuration. It does not add reward
logic or change TD3, environment dynamics, features, or the training loop.
"""

from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from src.analysis.cash_allocation_diagnostics import build_cash_allocation_report
from src.analysis.concentration_quality_diagnostics import (
    build_concentration_quality_report,
)
from src.analysis.shadow_mandate_penalty_report import (
    build_shadow_mandate_penalty_report,
)
from src.experiments.run_and_save_basic_experiment import run_and_save_basic_experiment


MANDATE_DEBUG_COLUMNS = (
    "mandate_penalty",
    "mandate_drawdown_breach",
    "mandate_volatility_breach",
    "mandate_max_weight_breach",
    "mandate_effective_assets_breach",
    "mandate_turnover_breach",
)


def get_default_mandate_reward_grid() -> list[dict]:
    """Return the narrow default mandate reward calibration grid."""
    balanced_weights = {
        "turnover_breach": 1.00,
        "effective_assets_breach": 1.00,
        "max_weight_breach": 0.75,
        "volatility_breach": 0.50,
        "drawdown_breach": 0.50,
    }

    return [
        {
            "run_name": "baseline_no_mandate",
            "use_mandate_penalty": False,
            "lambda_mandate": 0.0,
            "penalty_set": "none",
            "mandate_penalty_weights": {},
        },
        {
            "run_name": "moderate_balanced_lambda_001",
            "use_mandate_penalty": True,
            "lambda_mandate": 0.001,
            "penalty_set": "balanced",
            "mandate_penalty_weights": balanced_weights,
        },
        {
            "run_name": "moderate_balanced_lambda_003",
            "use_mandate_penalty": True,
            "lambda_mandate": 0.003,
            "penalty_set": "balanced",
            "mandate_penalty_weights": balanced_weights,
        },
        {
            "run_name": "moderate_balanced_lambda_005",
            "use_mandate_penalty": True,
            "lambda_mandate": 0.005,
            "penalty_set": "balanced",
            "mandate_penalty_weights": balanced_weights,
        },
        {
            "run_name": "moderate_balanced_lambda_0075",
            "use_mandate_penalty": True,
            "lambda_mandate": 0.0075,
            "penalty_set": "balanced",
            "mandate_penalty_weights": balanced_weights,
        },
    ]


def build_mandate_reward_config(
    base_config: dict,
    grid_item: dict,
    returns_path: str,
    episodes: int,
    seed: int | None = None,
    returns_date_column: str = "date",
    returns_start_date: str | None = None,
    returns_end_date: str | None = None,
) -> dict:
    """Build one generated config for a mandate reward grid item."""
    config = deepcopy(base_config)
    config.setdefault("data", {})
    config.setdefault("training", {})
    config.setdefault("reward", {})

    config["training"]["episodes"] = episodes
    if seed is not None:
        config["training"]["seed"] = seed
    config["data"]["returns_path"] = returns_path
    config["data"]["returns_date_column"] = returns_date_column
    if returns_start_date is not None:
        config["data"]["start_date"] = returns_start_date
    if returns_end_date is not None:
        config["data"]["end_date"] = returns_end_date

    reward = config["reward"]
    if grid_item.get("use_mandate_penalty", False):
        reward["use_mandate_penalty"] = True
        reward["lambda_mandate"] = float(grid_item["lambda_mandate"])
        reward["mandate_profile"] = "moderate"
        reward["mandate_penalty_weights"] = deepcopy(
            grid_item["mandate_penalty_weights"]
        )
    else:
        reward["use_mandate_penalty"] = False
        reward.pop("lambda_mandate", None)
        reward.pop("mandate_profile", None)
        reward.pop("mandate_penalty_weights", None)

    return config


def run_mandate_reward_grid(
    base_config_path: str,
    output_dir: str,
    returns_path: str,
    episodes: int = 20,
    grid: list[dict] | None = None,
    seeds: list[int] | None = None,
    run_diagnostics: bool = True,
) -> dict:
    """Run the mandate reward mini-grid and save summary outputs."""
    selected_returns_path = _resolve_returns_path(base_config_path, returns_path)
    if not Path(selected_returns_path).exists():
        raise FileNotFoundError(f"Returns snapshot not found: {selected_returns_path}")

    base_config = _load_yaml_config(base_config_path)
    returns_summary = _summarize_returns_snapshot(selected_returns_path)
    selected_grid = get_default_mandate_reward_grid() if grid is None else grid
    selected_seeds = [42] if seeds is None else seeds
    grid_output_dir = Path(output_dir)
    configs_dir = grid_output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    experiment_outputs = {}
    test_policy_history_paths = []
    strategy_names = []
    for grid_item in selected_grid:
        run_name = grid_item["run_name"]
        for seed in selected_seeds:
            run_id = f"{run_name}_seed_{seed}"
            run_config = build_mandate_reward_config(
                base_config=base_config,
                grid_item=grid_item,
                returns_path=selected_returns_path,
                episodes=episodes,
                seed=seed,
                returns_start_date=returns_summary["start_date"],
                returns_end_date=returns_summary["end_date"],
            )
            config_path = configs_dir / f"{run_id}.yaml"
            _write_yaml_config(run_config, config_path)

            experiment_output = run_and_save_basic_experiment(
                config_path=str(config_path),
                output_dir=str(grid_output_dir),
                experiment_name=run_id,
            )
            experiment_outputs[run_id] = experiment_output
            saved_paths = experiment_output["saved_paths"]
            test_policy_history_path = saved_paths.get("test_policy_history")
            if test_policy_history_path is not None:
                test_policy_history_paths.append(test_policy_history_path)
                strategy_names.append(run_id)
            rows.append(
                _build_summary_row(
                    grid_item=grid_item,
                    seed=seed,
                    run_id=run_id,
                    config_path=config_path,
                    experiment_output=experiment_output,
                )
            )

    results = pd.DataFrame(rows)
    results_path = grid_output_dir / "mandate_reward_grid_results.csv"
    results.to_csv(results_path, index=False)
    summary = _build_aggregate_summary(results)
    summary_path = grid_output_dir / "mandate_reward_grid_summary.csv"
    summary.to_csv(summary_path, index=False)

    diagnostics = None
    if run_diagnostics and test_policy_history_paths:
        diagnostics = _run_diagnostics(
            history_paths=test_policy_history_paths,
            strategy_names=strategy_names,
            asset_returns_path=selected_returns_path,
            diagnostics_dir=grid_output_dir / "diagnostics",
        )

    return {
        "output_dir": str(grid_output_dir),
        "configs_dir": str(configs_dir),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "results": results,
        "summary": summary,
        "experiment_outputs": experiment_outputs,
        "diagnostics": diagnostics,
    }


def _build_summary_row(
    grid_item: dict,
    seed: int,
    run_id: str,
    config_path: Path,
    experiment_output: dict,
) -> dict:
    experiment_result = experiment_output["experiment_result"]
    validation_summary = experiment_result["validation_comparison_summary"]
    test_summary = experiment_result["test_comparison_summary"]
    test_diagnostics = experiment_result["test_diagnostics"]
    validation_diagnostics = experiment_result["validation_diagnostics"]
    test_policy_history = experiment_result.get("test_policy_history")
    penalty_means = _mandate_debug_means(test_policy_history)

    return {
        "run_name": grid_item["run_name"],
        "run_id": run_id,
        "seed": seed,
        "use_mandate_penalty": bool(grid_item.get("use_mandate_penalty", False)),
        "lambda_mandate": float(grid_item.get("lambda_mandate", 0.0)),
        "penalty_set": grid_item.get("penalty_set", "none"),
        "config_path": str(config_path),
        "validation_sharpe": validation_summary["agent_sharpe_ratio"],
        "test_sharpe": test_summary["agent_sharpe_ratio"],
        "test_cumulative_return": test_summary["agent_cumulative_return"],
        "test_max_drawdown": test_summary["agent_max_drawdown"],
        "average_turnover": test_diagnostics["average_turnover"],
        "average_max_weight": test_diagnostics["average_max_weight"],
        "average_effective_number_of_assets": test_diagnostics[
            "average_effective_number_of_assets"
        ],
        "final_cash_weight": test_diagnostics["final_cash_weight"],
        "validation_average_turnover": validation_diagnostics["average_turnover"],
        **penalty_means,
    }


def _build_aggregate_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["run_name", "lambda_mandate"]
    for (run_name, lambda_mandate), group in results.groupby(group_columns, dropna=False):
        std_test_sharpe = group["test_sharpe"].std()
        if pd.isna(std_test_sharpe):
            std_test_sharpe = 0.0
        rows.append(
            {
                "run_name": run_name,
                "lambda_mandate": lambda_mandate,
                "n_seeds": group["seed"].nunique(),
                "mean_test_sharpe": group["test_sharpe"].mean(),
                "std_test_sharpe": std_test_sharpe,
                "robust_test_sharpe_score_05": (
                    group["test_sharpe"].mean() - 0.5 * std_test_sharpe
                ),
                "mean_test_cumulative_return": group[
                    "test_cumulative_return"
                ].mean(),
                "mean_test_max_drawdown": group["test_max_drawdown"].mean(),
                "worst_test_max_drawdown": group["test_max_drawdown"].min(),
                "mean_average_turnover": group["average_turnover"].mean(),
                "mean_average_max_weight": group["average_max_weight"].mean(),
                "mean_average_effective_number_of_assets": group[
                    "average_effective_number_of_assets"
                ].mean(),
                "mean_final_cash_weight": group["final_cash_weight"].mean(),
                "mean_mandate_penalty": group["mean_mandate_penalty"].mean(),
                "mean_mandate_drawdown_breach": group[
                    "mean_mandate_drawdown_breach"
                ].mean(),
                "mean_mandate_volatility_breach": group[
                    "mean_mandate_volatility_breach"
                ].mean(),
                "mean_mandate_max_weight_breach": group[
                    "mean_mandate_max_weight_breach"
                ].mean(),
                "mean_mandate_effective_assets_breach": group[
                    "mean_mandate_effective_assets_breach"
                ].mean(),
                "mean_mandate_turnover_breach": group[
                    "mean_mandate_turnover_breach"
                ].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["robust_test_sharpe_score_05", "mean_test_sharpe"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _mandate_debug_means(policy_history: pd.DataFrame | None) -> dict:
    means = {}
    for column in MANDATE_DEBUG_COLUMNS:
        output_column = f"mean_{column}"
        if policy_history is None or column not in policy_history.columns:
            means[output_column] = pd.NA
        else:
            means[output_column] = policy_history[column].mean()

    return means


def _run_diagnostics(
    history_paths: list[str],
    strategy_names: list[str],
    asset_returns_path: str,
    diagnostics_dir: Path,
) -> dict:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    shadow = build_shadow_mandate_penalty_report(
        history_paths=history_paths,
        strategy_names=strategy_names,
        mandate_profiles=["moderate"],
        output_dir=str(diagnostics_dir / "shadow_mandate_penalties"),
        return_column="financial_net_return",
    )
    concentration_quality = build_concentration_quality_report(
        policy_history_paths=history_paths,
        asset_returns_path=asset_returns_path,
        strategy_names=strategy_names,
        horizons=[1, 4, 12],
        output_dir=str(diagnostics_dir / "concentration_quality"),
    )
    cash_allocation = build_cash_allocation_report(
        policy_history_paths=history_paths,
        asset_returns_path=asset_returns_path,
        strategy_names=strategy_names,
        horizons=[1, 4, 12],
        normal_cash_max=0.10,
        output_dir=str(diagnostics_dir / "cash_allocation"),
    )

    return {
        "shadow_mandate_penalties": shadow,
        "concentration_quality": concentration_quality,
        "cash_allocation": cash_allocation,
    }


def _resolve_returns_path(base_config_path: str, returns_path: str | None) -> str:
    if returns_path:
        return returns_path

    config = _load_yaml_config(base_config_path)
    configured_returns_path = config.get("data", {}).get("returns_path")
    if configured_returns_path:
        return str(configured_returns_path)

    raise ValueError("returns_path must be provided or configured under data.returns_path.")


def _summarize_returns_snapshot(returns_path: str) -> dict:
    data = pd.read_csv(returns_path)
    if "date" not in data.columns:
        raise KeyError("Returns snapshot must include a date column.")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    if data.empty:
        raise ValueError("Returns snapshot has no usable rows.")

    return {
        "start_date": str(data["date"].min().date()),
        "end_date": str(data["date"].max().date()),
    }


def _load_yaml_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise TypeError("base config must be a YAML mapping.")

    return config


def _write_yaml_config(config: dict, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
