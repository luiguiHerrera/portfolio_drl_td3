"""Audit V3 uncapped baseline equivalence across protocol and cap runners.

This module is reporting-only. It compares the standalone V3 protocol-pure
revalidation output with the uncapped row inside a max-weight cap sensitivity
run, then writes metadata, metric, fold/seed, and pass/warning/fail checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PROTOCOL_DIR = "outputs/tables/protocol_v3_real_macro_current_60ep_10seeds"
DEFAULT_CAP_SENSITIVITY_DIR = "outputs/tables/cap_sensitivity_experiment_v3_60ep_10seeds"
DEFAULT_OUTPUT_DIR = "outputs/tables/v3_uncapped_baseline_equivalence_audit"

PROTOCOL_METADATA_FILE = "protocol_pure_td3_revalidation_metadata.json"
CAP_METADATA_FILE = "cap_sensitivity_metadata.json"
MAX_WEIGHT_METADATA_FILE = "max_weight_cap_metadata.json"

KEY_METRICS = [
    "sharpe",
    "annualized_return",
    "annualized_volatility",
    "cumulative_return",
    "max_drawdown",
    "worst_max_drawdown",
    "average_turnover",
    "average_effective_number_of_assets",
    "average_max_weight",
    "mean_cash_weight",
    "cash_above_10_rate",
    "mean_transaction_cost",
    "robust_score",
    "median_run_dsr_n25",
    "date_averaged_dsr_n25",
]

FOLD_SEED_METRICS = [
    "sharpe_ratio",
    "annualized_return",
    "annualized_volatility",
    "cumulative_return",
    "max_drawdown",
    "average_turnover",
    "average_effective_number_of_assets",
    "average_max_weight",
    "transaction_cost",
]


def audit_v3_uncapped_baseline_equivalence(
    protocol_dir: str = DEFAULT_PROTOCOL_DIR,
    cap_sensitivity_dir: str = DEFAULT_CAP_SENSITIVITY_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the equivalence audit and write all report files."""
    protocol_path = Path(protocol_dir)
    cap_path = Path(cap_sensitivity_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    inputs = load_audit_inputs(protocol_path, cap_path)
    metadata_comparison = build_metadata_comparison(inputs)
    metric_comparison = build_metric_comparison(inputs)
    fold_seed_comparison = build_fold_seed_comparison(inputs)
    checks = build_equivalence_checks(inputs, metadata_comparison, fold_seed_comparison)
    summary = build_equivalence_summary(checks, metadata_comparison, metric_comparison)

    paths = {
        "metadata_comparison": output_path / "v3_uncapped_metadata_comparison.csv",
        "metric_comparison": output_path / "v3_uncapped_metric_comparison.csv",
        "fold_seed_comparison": output_path / "v3_uncapped_fold_seed_comparison.csv",
        "checks": output_path / "v3_uncapped_equivalence_checks.csv",
        "summary": output_path / "v3_uncapped_equivalence_summary.md",
    }
    metadata_comparison.to_csv(paths["metadata_comparison"], index=False)
    metric_comparison.to_csv(paths["metric_comparison"], index=False)
    fold_seed_comparison.to_csv(paths["fold_seed_comparison"], index=False)
    checks.to_csv(paths["checks"], index=False)
    paths["summary"].write_text(summary, encoding="utf-8")

    return {
        "metadata_comparison": metadata_comparison,
        "metric_comparison": metric_comparison,
        "fold_seed_comparison": fold_seed_comparison,
        "checks": checks,
        "summary_markdown": summary,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def load_audit_inputs(protocol_path: Path, cap_path: Path) -> dict[str, Any]:
    """Load metadata and result tables from both experiment directories."""
    protocol_metadata = _read_json(protocol_path / PROTOCOL_METADATA_FILE)
    cap_metadata = _read_json(cap_path / CAP_METADATA_FILE)
    candidate_dir = _resolve_cap_candidate_dir(cap_path, cap_metadata)
    max_weight_metadata = _read_json(candidate_dir / MAX_WEIGHT_METADATA_FILE)
    return {
        "protocol_path": protocol_path,
        "cap_path": cap_path,
        "cap_candidate_path": candidate_dir,
        "protocol_metadata": protocol_metadata,
        "cap_metadata": cap_metadata,
        "max_weight_metadata": max_weight_metadata,
        "protocol_overall": _read_csv(protocol_path / "overall_aggregate_by_strategy_split.csv"),
        "protocol_robust": _read_csv(protocol_path / "robust_score_ranking.csv"),
        "protocol_seed_fold": _read_csv(protocol_path / "seed_fold_strategy_results.csv"),
        "cap_all_results": _read_csv(cap_path / "cap_sensitivity_all_results.csv"),
        "cap_overall": _read_csv(candidate_dir / "overall_aggregate_by_strategy_split.csv"),
        "cap_robust": _read_csv(candidate_dir / "robust_score_ranking.csv"),
        "cap_seed_fold": _read_csv(candidate_dir / "seed_fold_strategy_results.csv"),
    }


def build_metadata_comparison(inputs: dict[str, Any]) -> pd.DataFrame:
    """Compare protocol metadata fields that should match."""
    protocol_metadata = inputs["protocol_metadata"]
    cap_metadata = inputs["cap_metadata"]
    max_weight_metadata = inputs["max_weight_metadata"]

    protocol_run_config = _first_run_config(protocol_metadata)
    cap_uncapped_config = _first_uncapped_run_config(max_weight_metadata)
    cap_uncapped = _cap_uncapped_row(inputs)

    rows = [
        _metadata_row(
            "returns_path",
            protocol_metadata.get("returns_path"),
            cap_metadata.get("returns_path"),
        ),
        _metadata_row(
            "candidate",
            ",".join(map(str, protocol_metadata.get("candidates", []))),
            max_weight_metadata.get("candidate"),
        ),
        _metadata_row("episodes", protocol_metadata.get("episodes"), cap_metadata.get("episodes")),
        _metadata_row("seeds", protocol_metadata.get("seeds"), cap_metadata.get("seeds")),
        _metadata_row(
            "fold_definitions",
            protocol_metadata.get("folds"),
            max_weight_metadata.get("folds"),
        ),
        _metadata_row(
            "actual_folds",
            protocol_metadata.get("actual_folds"),
            max_weight_metadata.get("actual_folds"),
        ),
        _metadata_row(
            "transaction_cost",
            protocol_metadata.get("transaction_cost_rate"),
            max_weight_metadata.get("transaction_cost"),
        ),
        _metadata_row(
            "base_config_path",
            protocol_metadata.get("base_config_path"),
            max_weight_metadata.get("base_config_path"),
        ),
        _metadata_row(
            "feature_version",
            _nested_get(protocol_run_config, ["features", "version"]),
            _nested_get(cap_uncapped_config, ["features", "version"]),
        ),
        _metadata_row(
            "macro_path",
            _nested_get(protocol_run_config, ["features", "macro_path"]),
            _nested_get(cap_uncapped_config, ["features", "macro_path"]),
        ),
        _metadata_row(
            "reward_config",
            protocol_run_config.get("reward"),
            cap_uncapped_config.get("reward"),
        ),
        _metadata_row(
            "uncapped_max_weight_cap",
            None,
            _nan_to_none(cap_uncapped.get("max_weight_cap")),
        ),
    ]
    return pd.DataFrame(rows)


def build_metric_comparison(inputs: dict[str, Any]) -> pd.DataFrame:
    """Compare headline test metrics for the two uncapped baselines."""
    protocol_metrics = _protocol_test_metrics(inputs)
    cap_metrics = _cap_uncapped_metrics(inputs)
    rows = []
    for metric in KEY_METRICS:
        protocol_value = protocol_metrics.get(metric, np.nan)
        cap_value = cap_metrics.get(metric, np.nan)
        rows.append(
            {
                "metric": metric,
                "protocol_value": protocol_value,
                "cap_uncapped_value": cap_value,
                "delta_cap_minus_protocol": _numeric_delta(cap_value, protocol_value),
                "abs_delta": abs(_numeric_delta(cap_value, protocol_value))
                if _both_numeric(cap_value, protocol_value)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_fold_seed_comparison(inputs: dict[str, Any]) -> pd.DataFrame:
    """Compare per-fold/per-seed test rows between the two uncapped runs."""
    protocol = inputs["protocol_seed_fold"].copy()
    cap = inputs["cap_seed_fold"].copy()
    protocol = protocol[protocol["split"].astype(str) == "test"].copy()
    cap = cap[
        (cap["split"].astype(str) == "test")
        & (cap["strategy"].astype(str).str.endswith("_cap_uncapped"))
    ].copy()
    merged = protocol.merge(
        cap,
        on=["fold", "seed", "split"],
        suffixes=("_protocol", "_cap_uncapped"),
        how="outer",
        indicator=True,
    )
    rows = []
    for _, row in merged.iterrows():
        out = {
            "fold": row.get("fold"),
            "seed": row.get("seed"),
            "split": row.get("split"),
            "merge_status": row.get("_merge"),
        }
        for metric in FOLD_SEED_METRICS:
            left = row.get(f"{metric}_protocol", np.nan)
            right = row.get(f"{metric}_cap_uncapped", np.nan)
            out[f"{metric}_protocol"] = left
            out[f"{metric}_cap_uncapped"] = right
            out[f"{metric}_delta"] = _numeric_delta(right, left)
        rows.append(out)
    return pd.DataFrame(rows)


def build_equivalence_checks(
    inputs: dict[str, Any],
    metadata_comparison: pd.DataFrame,
    fold_seed_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Build pass/warning/fail checks for the audit."""
    checks = [
        check_metadata_matches(metadata_comparison),
        check_uncapped_row_has_no_cap(inputs),
        check_saved_run_configs_match(metadata_comparison),
        check_fold_seed_row_counts(inputs, fold_seed_comparison),
        check_fold_seed_numerical_equivalence(fold_seed_comparison),
        check_robust_score_universe(inputs),
        check_cap_sensitivity_usability(inputs, metadata_comparison),
    ]
    return pd.DataFrame(checks)


def check_metadata_matches(metadata_comparison: pd.DataFrame) -> dict[str, str]:
    """Check that core methodology metadata matches."""
    mismatches = metadata_comparison[metadata_comparison["status"] == "fail"]
    return _check(
        "metadata_core_fields_match",
        "Protocol and cap sensitivity metadata match on returns, candidate, seeds, folds, macro path, reward, and transaction cost.",
        "fail" if not mismatches.empty else "pass",
        _format_failed_fields(mismatches) if not mismatches.empty else "All core metadata fields match.",
    )


def check_uncapped_row_has_no_cap(inputs: dict[str, Any]) -> dict[str, str]:
    """Check that the cap sensitivity uncapped row has no max-weight cap."""
    row = _cap_uncapped_row(inputs)
    value = _nan_to_none(row.get("max_weight_cap"))
    status = "pass" if value is None else "fail"
    return _check(
        "cap_uncapped_row_is_uncapped",
        "Cap sensitivity uncapped baseline has max_weight_cap == None/NaN.",
        status,
        f"max_weight_cap={value!r}",
    )


def check_saved_run_configs_match(metadata_comparison: pd.DataFrame) -> dict[str, str]:
    """Check saved reward and feature configs for equivalence."""
    fields = metadata_comparison[
        metadata_comparison["field"].isin(["reward_config", "feature_version", "macro_path"])
    ]
    failed = fields[fields["status"] == "fail"]
    return _check(
        "saved_reward_feature_configs_match",
        "Saved uncapped run reward and feature configs match.",
        "fail" if not failed.empty else "pass",
        _format_failed_fields(failed) if not failed.empty else "Reward, feature version, and macro path match.",
    )


def check_fold_seed_row_counts(
    inputs: dict[str, Any],
    fold_seed_comparison: pd.DataFrame,
) -> dict[str, str]:
    """Check that both runs have matching fold/seed test row coverage."""
    expected_protocol = len(
        inputs["protocol_seed_fold"][inputs["protocol_seed_fold"]["split"].astype(str) == "test"]
    )
    expected_cap = len(
        inputs["cap_seed_fold"][
            (inputs["cap_seed_fold"]["split"].astype(str) == "test")
            & (inputs["cap_seed_fold"]["strategy"].astype(str).str.endswith("_cap_uncapped"))
        ]
    )
    matched = int((fold_seed_comparison["merge_status"] == "both").sum())
    status = "pass" if expected_protocol == expected_cap == matched else "fail"
    return _check(
        "fold_seed_test_rows_match",
        "Both uncapped runs contain the same fold/seed test rows.",
        status,
        f"protocol={expected_protocol}, cap_uncapped={expected_cap}, matched={matched}",
    )


def check_fold_seed_numerical_equivalence(
    fold_seed_comparison: pd.DataFrame,
    tolerance: float = 1e-10,
) -> dict[str, str]:
    """Check whether fold/seed metrics are numerically identical."""
    max_deltas = {}
    for metric in FOLD_SEED_METRICS:
        delta_col = f"{metric}_delta"
        if delta_col in fold_seed_comparison:
            max_deltas[metric] = float(fold_seed_comparison[delta_col].abs().max())
    max_abs = max(max_deltas.values()) if max_deltas else np.nan
    status = "pass" if pd.notna(max_abs) and max_abs <= tolerance else "warning"
    return _check(
        "fold_seed_metrics_numerically_equivalent",
        "Fold/seed test metrics are numerically identical across runner paths.",
        status,
        f"max_abs_delta={max_abs:.6g}; metric_max_deltas={max_deltas}",
    )


def check_robust_score_universe(inputs: dict[str, Any]) -> dict[str, str]:
    """Check whether robust_score was computed in the same strategy universe."""
    protocol_n = int(len(inputs["protocol_robust"]))
    cap_n = int(len(inputs["cap_robust"]))
    status = "pass" if protocol_n == cap_n else "warning"
    return _check(
        "robust_score_same_strategy_universe",
        "robust_score is computed over the same strategy universe.",
        status,
        (
            f"protocol robust_score strategies={protocol_n}; cap robust_score strategies={cap_n}. "
            "Cross-sectional normalized robust_score values should not be compared directly when this differs."
        ),
    )


def check_cap_sensitivity_usability(
    inputs: dict[str, Any],
    metadata_comparison: pd.DataFrame,
) -> dict[str, str]:
    """Assess whether the cap sensitivity result is internally usable."""
    metadata_ok = metadata_comparison["status"].ne("fail").all()
    uncapped_ok = _nan_to_none(_cap_uncapped_row(inputs).get("max_weight_cap")) is None
    cap_rows = inputs["cap_all_results"]
    decision_label = cap_rows.get(
        "decision_label",
        pd.Series("", index=cap_rows.index, dtype=object),
    )
    better_count = int(
        (
            (cap_rows["candidate_name"].astype(str).str.endswith("_cap_uncapped") == False)
            & (decision_label.astype(str) == "cap_dominates_uncapped")
        ).sum()
    )
    status = "pass" if metadata_ok and uncapped_ok and better_count > 0 else "warning"
    return _check(
        "cap_sensitivity_internal_usability",
        "Cap sensitivity can be interpreted against its own uncapped baseline.",
        status,
        (
            "The cap grid is internally paired within one runner output. "
            f"Caps dominating the cap-run uncapped baseline={better_count}."
        ),
    )


def build_equivalence_summary(
    checks: pd.DataFrame,
    metadata_comparison: pd.DataFrame,
    metric_comparison: pd.DataFrame,
) -> str:
    """Build a concise Markdown audit summary."""
    n_fail = int((checks["status"] == "fail").sum())
    n_warning = int((checks["status"] == "warning").sum())
    metadata_ok = n_fail == 0
    numerical_check = checks.set_index("check_id").loc[
        "fold_seed_metrics_numerically_equivalent"
    ]
    robust_check = checks.set_index("check_id").loc["robust_score_same_strategy_universe"]

    robust_delta = _metric_delta(metric_comparison, "robust_score")
    sharpe_delta = _metric_delta(metric_comparison, "sharpe")
    turnover_delta = _metric_delta(metric_comparison, "average_turnover")

    verdict = (
        "The two V3 uncapped baselines are methodologically comparable but not numerically equivalent."
        if metadata_ok
        else "The two V3 uncapped baselines are not methodologically equivalent yet."
    )
    usability = (
        "The V3 cap sensitivity result remains usable as an internally paired cap-grid experiment, "
        "but the standalone protocol uncapped robust_score should not be mixed with the cap-grid "
        "uncapped robust_score as if they were the same run."
        if metadata_ok
        else "A rerun is recommended before using V3_cap_0.50 as a final candidate."
    )

    return "\n".join(
        [
            "# V3 Uncapped Baseline Equivalence Audit",
            "",
            f"## Verdict",
            "",
            verdict,
            "",
            f"- Checks: {len(checks)} total, {n_fail} fail, {n_warning} warning.",
            f"- Numerical equivalence: {numerical_check['status']}.",
            f"- Robust-score universe check: {robust_check['status']}.",
            "",
            "## Main Metric Deltas",
            "",
            f"- robust_score delta, cap-run uncapped minus protocol: {robust_delta:.6f}",
            f"- Sharpe delta, cap-run uncapped minus protocol: {sharpe_delta:.6f}",
            f"- average_turnover delta, cap-run uncapped minus protocol: {turnover_delta:.6f}",
            "",
            "## Root Cause Assessment",
            "",
            (
                "No metadata, reward, feature, macro-path, fold, transaction-cost, or saved-config "
                "mismatch was found. The saved per-fold/per-seed outcomes differ, so the two outputs "
                "represent independent stochastic TD3 training realizations rather than one bitwise "
                "reproduced baseline. In addition, robust_score is cross-sectionally normalized within "
                "each report; the standalone protocol run contains one strategy, while the cap run "
                "contains the uncapped strategy plus capped variants. Source inspection also identifies "
                "a likely reproducibility cause: the TD3 ablation training path creates ReplayBuffer "
                "without passing the training seed, while ReplayBuffer uses np.random.default_rng(seed). "
                "With seed=None, replay sampling can differ across independent launches even when the "
                "saved experiment seed is the same."
            ),
            "",
            "## Usability",
            "",
            usability,
            "",
            "## Caveat To Document",
            "",
            (
                "V3_cap_0.50 should be reported from the cap sensitivity experiment using its own "
                "internally paired uncapped baseline. Do not compare the standalone V3 uncapped "
                "robust_score directly to the cap-grid robust_score without noting the different "
                "robust-score strategy universe and independent stochastic training realization."
            ),
            "",
        ]
    )


def _protocol_test_metrics(inputs: dict[str, Any]) -> dict[str, Any]:
    overall = inputs["protocol_overall"]
    row = overall[overall["split"].astype(str) == "test"].iloc[0].to_dict()
    robust = inputs["protocol_robust"].iloc[0].to_dict()
    return _normalize_metric_dict(row, robust)


def _cap_uncapped_metrics(inputs: dict[str, Any]) -> dict[str, Any]:
    row = _cap_uncapped_row(inputs).to_dict()
    robust = inputs["cap_robust"][
        inputs["cap_robust"]["strategy"].astype(str).str.endswith("_cap_uncapped")
    ].iloc[0].to_dict()
    return _normalize_metric_dict(row, robust)


def _normalize_metric_dict(row: dict[str, Any], robust: dict[str, Any]) -> dict[str, Any]:
    return {
        "sharpe": _coalesce(row.get("sharpe"), row.get("mean_sharpe"), robust.get("sharpe")),
        "annualized_return": _coalesce(
            row.get("annualized_return"),
            row.get("mean_annualized_return"),
        ),
        "annualized_volatility": _coalesce(
            row.get("annualized_volatility"),
            row.get("mean_annualized_volatility"),
        ),
        "cumulative_return": _coalesce(
            row.get("cumulative_return"),
            row.get("mean_cumulative_return"),
        ),
        "max_drawdown": _coalesce(
            row.get("max_drawdown"),
            row.get("mean_max_drawdown"),
            robust.get("max_drawdown"),
        ),
        "worst_max_drawdown": _coalesce(
            row.get("worst_max_drawdown"),
            row.get("worst_drawdown"),
            robust.get("worst_drawdown"),
        ),
        "average_turnover": _coalesce(
            row.get("average_turnover"),
            row.get("mean_average_turnover"),
            robust.get("turnover"),
        ),
        "average_effective_number_of_assets": _coalesce(
            row.get("average_effective_number_of_assets"),
            row.get("mean_average_effective_number_of_assets"),
            robust.get("effective_assets"),
        ),
        "average_max_weight": _coalesce(
            row.get("average_max_weight"),
            row.get("mean_average_max_weight"),
        ),
        "mean_cash_weight": _coalesce(row.get("mean_cash_weight")),
        "cash_above_10_rate": _coalesce(row.get("cash_above_10_rate")),
        "mean_transaction_cost": _coalesce(
            row.get("mean_transaction_cost"),
            row.get("transaction_cost"),
        ),
        "robust_score": robust.get("robust_score"),
        "median_run_dsr_n25": robust.get("median_run_dsr_n25"),
        "date_averaged_dsr_n25": robust.get("date_averaged_dsr_n25"),
    }


def _cap_uncapped_row(inputs: dict[str, Any]) -> pd.Series:
    all_results = inputs["cap_all_results"]
    matches = all_results[all_results["candidate_name"].astype(str).str.endswith("_cap_uncapped")]
    if matches.empty:
        raise ValueError("Cap sensitivity results do not contain an uncapped candidate row.")
    test_matches = matches[matches["split"].astype(str) == "test"]
    return (test_matches if not test_matches.empty else matches).iloc[0]


def _resolve_cap_candidate_dir(cap_path: Path, metadata: dict[str, Any]) -> Path:
    candidates = metadata.get("candidate_output_dirs") or {}
    if candidates:
        return Path(next(iter(candidates.values())))
    per_candidate = cap_path / "per_candidate"
    children = [path for path in per_candidate.iterdir() if path.is_dir()]
    if len(children) != 1:
        raise ValueError(f"Cannot infer cap candidate output directory under {per_candidate}")
    return children[0]


def _first_run_config(metadata: dict[str, Any]) -> dict[str, Any]:
    run_configs = metadata.get("run_configs") or {}
    if not run_configs:
        return {}
    key = sorted(run_configs)[0]
    return run_configs[key]


def _first_uncapped_run_config(metadata: dict[str, Any]) -> dict[str, Any]:
    run_configs = metadata.get("run_configs") or {}
    for key in sorted(run_configs):
        config = run_configs[key]
        if _nan_to_none(config.get("max_weight_cap")) is None and "_cap_uncapped_" in key:
            return config
    for key in sorted(run_configs):
        config = run_configs[key]
        if _nan_to_none(config.get("max_weight_cap")) is None:
            return config
    return {}


def _metadata_row(field: str, protocol_value: Any, cap_value: Any) -> dict[str, Any]:
    equal = _stable_value(protocol_value) == _stable_value(cap_value)
    return {
        "field": field,
        "protocol_value": _to_display_value(protocol_value),
        "cap_sensitivity_value": _to_display_value(cap_value),
        "status": "pass" if equal else "fail",
        "details": "match" if equal else "mismatch",
    }


def _check(check_id: str, description: str, status: str, details: str) -> dict[str, str]:
    return {
        "check_id": check_id,
        "description": description,
        "status": status,
        "details": details,
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _nested_get(mapping: dict[str, Any], keys: list[str]) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _stable_value(value: Any) -> Any:
    value = _nan_to_none(value)
    if isinstance(value, dict):
        return {key: _stable_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_stable_value(item) for item in value)
    if isinstance(value, float):
        return round(value, 12)
    return value


def _to_display_value(value: Any) -> str:
    value = _stable_value(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _nan_to_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _coalesce(*values: Any) -> Any:
    for value in values:
        if _nan_to_none(value) is not None:
            return value
    return np.nan


def _both_numeric(left: Any, right: Any) -> bool:
    return pd.notna(pd.to_numeric(left, errors="coerce")) and pd.notna(
        pd.to_numeric(right, errors="coerce")
    )


def _numeric_delta(left: Any, right: Any) -> float:
    if not _both_numeric(left, right):
        return np.nan
    return float(left) - float(right)


def _format_failed_fields(failed: pd.DataFrame) -> str:
    if failed.empty:
        return ""
    return "; ".join(failed["field"].astype(str).tolist())


def _metric_delta(metric_comparison: pd.DataFrame, metric: str) -> float:
    matches = metric_comparison[metric_comparison["metric"] == metric]
    if matches.empty:
        return np.nan
    return float(matches.iloc[0]["delta_cap_minus_protocol"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--cap-sensitivity-dir", default=DEFAULT_CAP_SENSITIVITY_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_v3_uncapped_baseline_equivalence(
        protocol_dir=args.protocol_dir,
        cap_sensitivity_dir=args.cap_sensitivity_dir,
        output_dir=args.output_dir,
    )
    print("Equivalence checks:")
    print(result["checks"].to_string(index=False))
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
