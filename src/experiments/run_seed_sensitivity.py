"""Seed sensitivity runner for one selected TD3 experiment configuration.

This module repeats one TD3 configuration across several random seeds, saves
the usual per-seed experiment CSV outputs, and writes aggregate sensitivity
tables. It does not save models, replay buffers, raw results, plots, or reports.
"""

from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from src.experiments.run_and_save_basic_experiment import run_and_save_basic_experiment


DEFAULT_SEEDS = [7, 21, 42, 73, 101]


def run_seed_sensitivity(
    base_config_path: str,
    output_dir: str = "outputs/tables",
    experiment_name: str = "td3_seed_sensitivity_E",
    seeds: list[int] | None = None,
    episodes: int = 100,
    batch_size: int = 64,
    actor_learning_rate: float = 0.0003,
    critic_learning_rate: float = 0.0003,
) -> dict:
    """Run one TD3 configuration across random seeds and save summary tables."""
    base_config = _load_yaml_config(base_config_path)
    selected_seeds = DEFAULT_SEEDS if seeds is None else seeds
    seed_output_dir = Path(output_dir) / experiment_name
    configs_dir = seed_output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    experiment_outputs = {}
    for seed in selected_seeds:
        seed_config = _build_seed_config(
            base_config=base_config,
            seed=seed,
            episodes=episodes,
            batch_size=batch_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
        )
        seed_config_path = configs_dir / f"seed_{seed}.yaml"
        _write_yaml_config(seed_config, seed_config_path)

        seed_experiment_name = f"seed_{seed}"
        experiment_output = run_and_save_basic_experiment(
            config_path=str(seed_config_path),
            output_dir=str(seed_output_dir),
            experiment_name=seed_experiment_name,
        )
        experiment_outputs[seed] = experiment_output
        rows.append(
            _build_seed_row(
                seed=seed,
                episodes=episodes,
                batch_size=batch_size,
                actor_learning_rate=actor_learning_rate,
                critic_learning_rate=critic_learning_rate,
                experiment_output=experiment_output,
            )
        )

    results = pd.DataFrame(rows)
    results_path = seed_output_dir / "seed_sensitivity_results.csv"
    results.to_csv(results_path, index=False)

    summary = _build_summary(results)
    summary_path = seed_output_dir / "seed_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    return {
        "output_dir": str(seed_output_dir),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "results": results,
        "summary": summary,
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


def _build_seed_config(
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


def _build_seed_row(
    seed: int,
    episodes: int,
    batch_size: int,
    actor_learning_rate: float,
    critic_learning_rate: float,
    experiment_output: dict,
) -> dict:
    experiment_result = experiment_output["experiment_result"]
    validation_metrics = experiment_result["validation_metrics_table"]
    test_metrics = experiment_result["test_metrics_table"]
    validation_summary = experiment_result["validation_comparison_summary"]
    test_summary = experiment_result["test_comparison_summary"]
    test_diagnostics = experiment_result["test_diagnostics"]

    return {
        "seed": seed,
        "episodes": episodes,
        "batch_size": batch_size,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "test_agent_cumulative_return": test_metrics.loc["agent", "cumulative_return"],
        "test_agent_sharpe_ratio": test_metrics.loc["agent", "sharpe_ratio"],
        "test_agent_max_drawdown": test_metrics.loc["agent", "max_drawdown"],
        "test_average_turnover": test_diagnostics["average_turnover"],
        "test_average_effective_number_of_assets": test_diagnostics[
            "average_effective_number_of_assets"
        ],
        "test_final_max_weight": test_diagnostics["final_max_weight"],
        "test_best_policy_by_sharpe": test_summary["best_policy_by_sharpe"],
        "test_agent_rank_by_sharpe": test_summary["agent_rank_by_sharpe"],
        "test_best_individual_buyhold_by_sharpe": test_summary[
            "best_individual_buyhold_by_sharpe"
        ],
        "test_best_individual_buyhold_sharpe_ratio": test_summary[
            "best_individual_buyhold_sharpe_ratio"
        ],
        "test_best_individual_buyhold_cumulative_return": test_summary[
            "best_individual_buyhold_cumulative_return"
        ],
        "test_agent_vs_best_individual_buyhold_sharpe_diff": test_summary[
            "agent_vs_best_individual_buyhold_sharpe_diff"
        ],
        "test_agent_vs_best_individual_buyhold_cumulative_return_diff": test_summary[
            "agent_vs_best_individual_buyhold_cumulative_return_diff"
        ],
        "validation_agent_sharpe_ratio": validation_metrics.loc["agent", "sharpe_ratio"],
        "validation_agent_cumulative_return": validation_metrics.loc[
            "agent",
            "cumulative_return",
        ],
        "validation_agent_rank_by_sharpe": validation_summary["agent_rank_by_sharpe"],
        "validation_best_policy_by_sharpe": validation_summary["best_policy_by_sharpe"],
        "validation_best_individual_buyhold_by_sharpe": validation_summary[
            "best_individual_buyhold_by_sharpe"
        ],
        "validation_agent_vs_best_individual_buyhold_sharpe_diff": validation_summary[
            "agent_vs_best_individual_buyhold_sharpe_diff"
        ],
    }


def _build_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "n_seeds": len(results),
        "mean_test_agent_sharpe": results["test_agent_sharpe_ratio"].mean(),
        "std_test_agent_sharpe": results["test_agent_sharpe_ratio"].std(),
        "min_test_agent_sharpe": results["test_agent_sharpe_ratio"].min(),
        "max_test_agent_sharpe": results["test_agent_sharpe_ratio"].max(),
        "mean_test_agent_cumulative_return": results[
            "test_agent_cumulative_return"
        ].mean(),
        "mean_test_agent_max_drawdown": results["test_agent_max_drawdown"].mean(),
        "mean_test_average_turnover": results["test_average_turnover"].mean(),
        "mean_test_average_effective_number_of_assets": results[
            "test_average_effective_number_of_assets"
        ].mean(),
        "win_rate_vs_best_individual_buyhold_by_sharpe": (
            results["test_agent_vs_best_individual_buyhold_sharpe_diff"] > 0
        ).mean(),
        "win_rate_best_policy_agent": (
            results["test_best_policy_by_sharpe"] == "agent"
        ).mean(),
    }

    return pd.DataFrame([summary])
