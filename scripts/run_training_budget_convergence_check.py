"""Run the final-corrected TD3 training-budget convergence robustness check.

This script trains only the explicitly selected candidate/cap pairs and writes
outputs under a separate robustness directory. It does not overwrite final
corrected reports and does not create new final winners.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.robust_score import build_robust_score_report
from src.analysis.training_budget_convergence_report import (
    EPISODE_BUDGETS,
    build_training_budget_convergence_report,
)
from src.experiments.run_feature_block_ablation import (
    ACTOR_LR,
    BATCH_SIZE,
    CRITIC_LR,
    EXPANDING_FOLDS,
    _actual_fold_row,
    _build_base_config,
    _build_experiment_result,
    _metric_row,
    aggregate_metric_rows,
    build_ablation_fold_datasets,
    build_returns_dataset_from_config,
    train_td3_ablation_on_datasets,
)
from src.experiments.run_max_weight_cap_experiment import (
    build_candidate_run_config_for_cap,
    build_max_weight_cap_summary,
    build_max_weight_grid_candidates,
    patched_portfolio_env,
)
from src.experiments.run_protocol_pure_td3_revalidation import (
    _build_feature_context,
    _candidate_auxiliary_features,
    _candidate_raw_features,
    _select_candidates,
    _write_yaml,
)
from src.experiments.save_experiment_outputs import save_basic_experiment_outputs


DEFAULT_OUTPUT_DIR = "~/Projects/portfolio_drl_outputs/final_corrected_training_budget_convergence"
ZERO_RETURNS_PATH = "data/processed/returns_weekly_latest.csv"
ZERO_CONFIG_PATH = "outputs/tables/final_corrected_limited_td3_60ep_10seeds/final_corrected_config.yaml"
BIL_RETURNS_PATH = "data/processed/returns_weekly_latest_cash_bil_proxy.csv"
BIL_CONFIG_PATH = (
    "~/Projects/portfolio_drl_outputs/final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds/"
    "final_corrected_bil_cash_config.yaml"
)
SEEDS = [7, 21, 42, 84, 101]
EXPECTED_HISTORIES_PER_CASE = len(SEEDS) * len(EXPANDING_FOLDS)


@dataclass(frozen=True)
class ConvergenceSpec:
    cash_assumption: str
    candidate: str
    cap: float
    returns_path: str
    base_config_path: str
    expected_cash_bps: float

    @property
    def cap_label(self) -> str:
        return f"{self.cap:.2f}".replace(".", "p")

    @property
    def candidate_name(self) -> str:
        return f"{self.candidate}_cap_{self.cap_label}"


SPECS = [
    ConvergenceSpec(
        cash_assumption="zero_cash",
        candidate="V3_real_macro_vintage_clean_no_dxy",
        cap=0.70,
        returns_path=ZERO_RETURNS_PATH,
        base_config_path=ZERO_CONFIG_PATH,
        expected_cash_bps=0.0,
    ),
    ConvergenceSpec(
        cash_assumption="bil_cash",
        candidate="V7_real_macro_vintage_clean_no_dxy_garch",
        cap=0.80,
        returns_path=BIL_RETURNS_PATH,
        base_config_path=BIL_CONFIG_PATH,
        expected_cash_bps=2.0,
    ),
    ConvergenceSpec(
        cash_assumption="zero_cash",
        candidate="V5_no_volatility_block",
        cap=0.50,
        returns_path=ZERO_RETURNS_PATH,
        base_config_path=ZERO_CONFIG_PATH,
        expected_cash_bps=0.0,
    ),
    ConvergenceSpec(
        cash_assumption="bil_cash",
        candidate="V8_ewma_garch_vol_current",
        cap=0.70,
        returns_path=BIL_RETURNS_PATH,
        base_config_path=BIL_CONFIG_PATH,
        expected_cash_bps=2.0,
    ),
]


def run_training_budget_convergence_check(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    episode_budgets: list[int] | None = None,
    dry_run: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    selected_budgets = list(EPISODE_BUDGETS if episode_budgets is None else episode_budgets)
    completed = []
    skipped = []
    for spec in SPECS:
        validate_spec_paths(spec)
        for episodes in selected_budgets:
            case_dir = case_output_dir(output_path, spec, episodes)
            if resume and case_complete(case_dir):
                print(f"[skip] {spec.cash_assumption} {spec.candidate_name} episodes={episodes}: already complete", flush=True)
                skipped.append({"case_dir": str(case_dir), "reason": "already_complete"})
                continue
            if dry_run:
                print(f"[dry-run] {spec.cash_assumption} {spec.candidate_name} episodes={episodes}", flush=True)
                skipped.append({"case_dir": str(case_dir), "reason": "dry_run"})
                continue
            print(f"[run] {spec.cash_assumption} {spec.candidate_name} episodes={episodes}", flush=True)
            completed.append(run_case(spec, episodes, case_dir))
            print(f"[done] {spec.cash_assumption} {spec.candidate_name} episodes={episodes}", flush=True)
    report = None
    if not dry_run:
        report = build_training_budget_convergence_report(output_dir=str(output_path))
    metadata = {
        "runner": "scripts/run_training_budget_convergence_check.py",
        "output_dir": str(output_path),
        "episode_budgets": selected_budgets,
        "seeds": SEEDS,
        "folds": [fold["fold_id"] for fold in EXPANDING_FOLDS],
        "expected_histories_per_case": EXPECTED_HISTORIES_PER_CASE,
        "specs": [asdict(spec) for spec in SPECS],
        "completed_cases": completed,
        "skipped_cases": skipped,
        "dry_run": dry_run,
        "resume": resume,
        "creates_new_final_winners": False,
        "convergence_report_paths": report["paths"] if report is not None else {},
    }
    (output_path / "training_budget_convergence_run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def run_case(spec: ConvergenceSpec, episodes: int, case_dir: Path) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = case_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    base_config = _build_base_config(str(Path(spec.base_config_path).expanduser()), spec.returns_path, episodes)
    validate_cost_model(base_config, spec.expected_cash_bps)
    base_candidate = deepcopy(_select_candidates([spec.candidate])[0])
    experiment_candidate = build_max_weight_grid_candidates(base_candidate, [spec.cap])[0]
    experiment_candidate["name"] = spec.candidate_name

    returns = build_returns_dataset_from_config(base_config, configs_dir / "_returns_config.yaml")
    feature_context = _build_feature_context(returns, base_config, [base_candidate])
    raw_features = _candidate_raw_features(experiment_candidate, feature_context)
    raw_auxiliary = _candidate_auxiliary_features(experiment_candidate, feature_context)
    include_auxiliary = raw_auxiliary is not None

    fold_rows = []
    metric_rows = []
    run_configs = {}
    for fold in EXPANDING_FOLDS:
        datasets = build_ablation_fold_datasets(
            returns=returns,
            raw_features=raw_features,
            raw_auxiliary_features=raw_auxiliary if raw_auxiliary is not None else pd.DataFrame(index=returns.index),
            fold=fold,
            include_auxiliary=include_auxiliary,
        )
        fold_rows.append(_actual_fold_row(fold, datasets))
        for seed in SEEDS:
            print(
                f"  training {spec.cash_assumption} {spec.candidate_name} episodes={episodes} "
                f"fold={fold['fold_id']} seed={seed}",
                flush=True,
            )
            config = build_candidate_run_config_for_cap(
                base_config=base_config,
                candidate=experiment_candidate,
                seed=seed,
                episodes=episodes,
                batch_size=BATCH_SIZE,
                actor_learning_rate=ACTOR_LR,
                critic_learning_rate=CRITIC_LR,
            )
            run_key = f"{fold['fold_id']}_{spec.candidate_name}_seed_{seed}"
            config_path = configs_dir / f"{run_key}.yaml"
            _write_yaml(config, config_path)
            run_configs[run_key] = {"path": str(config_path), "max_weight_cap": spec.cap}

            with patched_portfolio_env(spec.cap):
                raw_result = train_td3_ablation_on_datasets(datasets, config)
            experiment_result = _build_experiment_result(raw_result)
            save_basic_experiment_outputs(
                experiment_result,
                output_dir=str(case_dir),
                experiment_name=run_key,
            )
            for split_name in ("validation", "test"):
                metric_rows.append(
                    _metric_row(
                        variant=spec.candidate_name,
                        fold=fold,
                        seed=seed,
                        split=split_name,
                        metrics=experiment_result[f"{split_name}_metrics_table"].loc["agent"],
                        diagnostics=experiment_result[f"{split_name}_diagnostics"],
                        policy_history=experiment_result[f"{split_name}_policy_history"],
                    )
                )

    actual_folds = pd.DataFrame(fold_rows).drop_duplicates()
    seed_results = pd.DataFrame(metric_rows)
    overall = aggregate_metric_rows(seed_results, group_columns=["strategy", "split"])
    seed_level = aggregate_metric_rows(seed_results, group_columns=["strategy", "split", "seed"])
    fold_level = aggregate_metric_rows(seed_results, group_columns=["strategy", "split", "fold"])

    actual_folds.to_csv(case_dir / "actual_fold_dates.csv", index=False)
    seed_results.to_csv(case_dir / "seed_fold_strategy_results.csv", index=False)
    overall.to_csv(case_dir / "overall_aggregate_by_strategy_split.csv", index=False)
    seed_level.to_csv(case_dir / "seed_level_aggregate_by_strategy_split.csv", index=False)
    fold_level.to_csv(case_dir / "fold_level_aggregate_by_strategy_split.csv", index=False)

    robust_report = build_robust_score_report(str(case_dir), output_dir=str(case_dir))
    summary = build_max_weight_cap_summary(
        overall,
        robust_report["ranking"],
        [experiment_candidate],
        episodes=episodes,
    )
    summary["cash_assumption"] = spec.cash_assumption
    summary["cap_label"] = spec.cap_label
    summary["returns_path"] = spec.returns_path
    summary["base_config_path"] = str(Path(spec.base_config_path).expanduser())
    test_summary = summary[summary["split"].astype(str) == "test"].copy()
    test_summary["completed_test_histories"] = count_test_histories(case_dir)
    test_summary.to_csv(case_dir / "training_budget_case_summary.csv", index=False)

    case_metadata = {
        "spec": asdict(spec),
        "episodes": episodes,
        "seeds": SEEDS,
        "folds": [fold["fold_id"] for fold in EXPANDING_FOLDS],
        "case_dir": str(case_dir),
        "completed_test_histories": count_test_histories(case_dir),
        "asset_specific_costs_preserved": True,
        "creates_new_final_winners": False,
        "run_configs": run_configs,
    }
    (case_dir / "training_budget_case_metadata.json").write_text(
        json.dumps(case_metadata, indent=2),
        encoding="utf-8",
    )
    return {
        "case_dir": str(case_dir),
        "candidate": spec.candidate,
        "cap": spec.cap,
        "episodes": episodes,
        "completed_test_histories": case_metadata["completed_test_histories"],
    }


def validate_spec_paths(spec: ConvergenceSpec) -> None:
    if not Path(spec.returns_path).exists():
        raise FileNotFoundError(f"Missing returns file for {spec.candidate}: {spec.returns_path}")
    if not Path(spec.base_config_path).expanduser().exists():
        raise FileNotFoundError(f"Missing config for {spec.candidate}: {spec.base_config_path}")


def validate_cost_model(base_config: dict[str, Any], expected_cash_bps: float) -> None:
    environment = base_config.get("environment", {})
    mode = environment.get("transaction_cost_mode")
    costs = environment.get("asset_transaction_cost_bps", {})
    if mode != "asset_specific":
        raise ValueError(f"Expected asset_specific transaction_cost_mode, got {mode!r}")
    if float(costs.get("CASH")) != expected_cash_bps:
        raise ValueError(f"Expected CASH cost {expected_cash_bps} bps, got {costs.get('CASH')}")
    if float(costs.get("BTC-USD")) != 10.0:
        raise ValueError(f"Expected BTC-USD cost 10 bps, got {costs.get('BTC-USD')}")


def case_output_dir(output_path: Path, spec: ConvergenceSpec, episodes: int) -> Path:
    return output_path / "cases" / f"{spec.cash_assumption}_{spec.candidate}_cap_{spec.cap_label}" / f"episodes_{episodes}"


def case_complete(case_dir: Path) -> bool:
    return (
        (case_dir / "training_budget_case_summary.csv").exists()
        and count_test_histories(case_dir) == EXPECTED_HISTORIES_PER_CASE
    )


def count_test_histories(case_dir: Path) -> int:
    return sum(1 for _ in case_dir.glob("F*_*/test_policy_history.csv"))


def parse_episode_budgets(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final-corrected training-budget convergence check.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode-budgets", default=",".join(str(value) for value in EPISODE_BUDGETS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_training_budget_convergence_check(
        output_dir=args.output_dir,
        episode_budgets=parse_episode_budgets(args.episode_budgets),
        dry_run=args.dry_run,
        resume=not args.no_resume,
    )
    print("Training-budget convergence check metadata:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
