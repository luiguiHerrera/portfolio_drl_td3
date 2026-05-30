"""Build a combined report for asset-specific transaction cost retraining.

This reporting layer combines already-generated limited cap-sensitivity outputs.
It does not retrain models or alter scoring logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_V3_DIR = "outputs/tables/asset_specific_cost_v3_clean_no_dxy_60ep_10seeds"
DEFAULT_V7_DIR = "outputs/tables/asset_specific_cost_v7_clean_no_dxy_garch_60ep_10seeds"
DEFAULT_V4_DIR = "outputs/tables/asset_specific_cost_v4_garch_60ep_10seeds"
DEFAULT_OUTPUT_DIR = "outputs/tables/asset_specific_cost_limited_final_report"

CAP_RESULTS_FILE = "cap_sensitivity_all_results.csv"
CAP_METADATA_FILE = "cap_sensitivity_metadata.json"

REPORT_COLUMNS = [
    "rank_mandate_aware",
    "rank_robust",
    "strategy_name",
    "base_candidate",
    "cap_label",
    "max_weight_cap",
    "transaction_cost_mode",
    "asset_transaction_cost_bps",
    "mean_transaction_cost",
    "average_turnover",
    "average_btc_cost_contribution",
    "average_btc_allocation",
    "average_max_weight",
    "average_effective_number_of_assets",
    "max_drawdown",
    "worst_max_drawdown",
    "sharpe",
    "annualized_return",
    "annualized_volatility",
    "robust_score",
    "mandate_aware_score",
    "decision_label",
    "candidate_output_dir",
    "score_comparability_note",
]

SCORE_COMPARABILITY_NOTE = (
    "Scores are imported from limited cap-sensitivity reports. Because robust "
    "and mandate-aware components may be normalized within each report universe, "
    "cross-candidate score comparisons should be treated as preliminary."
)


def build_asset_specific_cost_final_report(
    v3_dir: str = DEFAULT_V3_DIR,
    v7_dir: str = DEFAULT_V7_DIR,
    v4_dir: str = DEFAULT_V4_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Combine limited asset-specific retraining outputs into one report."""
    input_dirs = {
        "v3_clean_no_dxy": Path(v3_dir),
        "v7_clean_no_dxy_garch": Path(v7_dir),
        "v4_garch": Path(v4_dir),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    metadata_inputs = {}
    warnings = []
    for label, directory in input_dirs.items():
        if not directory.exists():
            warnings.append(f"Missing input directory: {directory}")
            continue
        frame = load_cap_results(directory)
        frame["input_label"] = label
        metadata_inputs[label] = load_metadata(directory)
        rows.append(frame)

    if not rows:
        raise ValueError("No asset-specific cap-sensitivity inputs were found.")

    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined = enrich_with_history_diagnostics(combined)
    combined["score_comparability_note"] = SCORE_COMPARABILITY_NOTE
    main_ranking = build_main_ranking(combined)
    selected = build_selected_candidates(main_ranking)
    markdown = build_summary_markdown(main_ranking, selected, warnings)
    metadata = build_metadata(
        input_dirs=input_dirs,
        metadata_inputs=metadata_inputs,
        output_dir=output_path,
        warnings=warnings,
    )

    paths = {
        "selected_candidates": output_path / "asset_specific_cost_selected_candidates.csv",
        "main_ranking": output_path / "asset_specific_cost_main_ranking.csv",
        "markdown_summary": output_path / "asset_specific_cost_summary.md",
        "metadata": output_path / "asset_specific_cost_metadata.json",
    }
    selected.to_csv(paths["selected_candidates"], index=False)
    main_ranking.to_csv(paths["main_ranking"], index=False)
    paths["markdown_summary"].write_text(markdown, encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "selected_candidates": selected,
        "main_ranking": main_ranking,
        "markdown_summary": markdown,
        "metadata": metadata,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def load_cap_results(directory: Path) -> pd.DataFrame:
    """Load one cap-sensitivity result table."""
    path = directory / CAP_RESULTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing cap sensitivity results: {path}")
    frame = pd.read_csv(path)
    frame = frame[frame.get("split", "test") == "test"].copy()
    if "cap_label" not in frame.columns:
        frame["cap_label"] = frame["max_weight_cap"].map(format_cap_label)
    return frame


def enrich_with_history_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach transaction-cost and BTC diagnostics from saved policy histories."""
    records = []
    for _, row in frame.iterrows():
        record = row.to_dict()
        diagnostics = history_diagnostics_for_candidate(row)
        record.update(diagnostics)
        records.append(record)
    return pd.DataFrame(records)


def history_diagnostics_for_candidate(row: pd.Series) -> dict[str, Any]:
    """Compute diagnostic means from matching per-fold/per-seed policy histories."""
    candidate_name = str(row["candidate_name"])
    output_dir = Path(str(row["candidate_output_dir"]))
    histories = sorted(output_dir.glob(f"F*_{candidate_name}_seed_*/test_policy_history.csv"))
    config_paths = sorted(output_dir.glob(f"configs/F*_{candidate_name}_seed_*.yaml"))
    config_info = read_transaction_cost_config(config_paths[0]) if config_paths else {}

    if not histories:
        return {
            **config_info,
            "average_btc_cost_contribution": pd.NA,
            "average_btc_allocation": pd.NA,
            "history_files_found": 0,
        }

    frames = [pd.read_csv(path) for path in histories]
    history = pd.concat(frames, ignore_index=True, sort=False)
    return {
        **config_info,
        "average_btc_cost_contribution": mean_or_na(
            history,
            "asset_transaction_cost_contribution_BTC-USD",
        ),
        "average_btc_allocation": mean_or_na(history, "weight_BTC-USD"),
        "history_files_found": len(histories),
    }


def read_transaction_cost_config(path: Path) -> dict[str, Any]:
    """Read transaction-cost settings from one generated run config."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    environment = config.get("environment", {})
    return {
        "transaction_cost_mode": environment.get("transaction_cost_mode", "scalar"),
        "asset_transaction_cost_bps": environment.get("asset_transaction_cost_bps"),
    }


def build_main_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    """Build all-row ranking table."""
    result = frame.copy()
    result["strategy_name"] = result["candidate_name"]
    result["rank_mandate_aware"] = (
        pd.to_numeric(result["mandate_aware_score"], errors="coerce")
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    result["rank_robust"] = (
        pd.to_numeric(result["robust_score"], errors="coerce")
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    for column in REPORT_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result = result[REPORT_COLUMNS + ["history_files_found", "input_label"]]
    return result.sort_values(
        ["rank_mandate_aware", "rank_robust", "strategy_name"],
        na_position="last",
    ).reset_index(drop=True)


def build_selected_candidates(main_ranking: pd.DataFrame) -> pd.DataFrame:
    """Identify best rows by the requested diagnostic lenses."""
    selections = [
        ("best_by_mandate_aware_score", "mandate_aware_score", False),
        ("best_by_robust_score", "robust_score", False),
        ("best_by_drawdown", "max_drawdown", False),
        ("best_by_turnover", "average_turnover", True),
        ("best_by_effective_assets", "average_effective_number_of_assets", False),
    ]
    rows = []
    for selection, metric, ascending in selections:
        valid = main_ranking.copy()
        valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
        valid = valid.dropna(subset=[metric])
        if valid.empty:
            continue
        best = valid.sort_values(metric, ascending=ascending).iloc[0].to_dict()
        best["selection"] = selection
        best["selection_metric"] = metric
        best["selection_metric_value"] = best[metric]
        rows.append(best)
    if not rows:
        return pd.DataFrame()
    columns = [
        "selection",
        "selection_metric",
        "selection_metric_value",
        "strategy_name",
        "base_candidate",
        "cap_label",
        "mandate_aware_score",
        "robust_score",
        "max_drawdown",
        "average_turnover",
        "average_effective_number_of_assets",
        "average_max_weight",
        "average_btc_cost_contribution",
        "average_btc_allocation",
        "transaction_cost_mode",
        "asset_transaction_cost_bps",
    ]
    return pd.DataFrame(rows).loc[:, columns]


def build_summary_markdown(
    main_ranking: pd.DataFrame,
    selected: pd.DataFrame,
    warnings: list[str],
) -> str:
    """Build cautious markdown summary."""
    top_mandate = main_ranking.sort_values("rank_mandate_aware").iloc[0]
    top_robust = main_ranking.sort_values("rank_robust").iloc[0]
    lines = [
        "# Asset-Specific Transaction Cost Limited Final Report",
        "",
        "This report combines limited TD3 retraining runs under asset-specific "
        "transaction costs. It is reporting-only and does not retrain models.",
        "",
        "The comparison is limited to V3 clean no-DXY macro, V7 clean no-DXY "
        "+ GARCH, and V4 real GARCH candidates. It is not the full original "
        "candidate universe.",
        "",
        "Scalar-cost and asset-specific-cost results are not directly "
        "interchangeable. The imported robust and mandate-aware scores may be "
        "normalized within each cap-sensitivity report universe, so cross-candidate "
        "score comparisons should be treated as preliminary.",
        "",
        "## Top Rows",
        "",
        f"- Best by mandate-aware score: `{top_mandate['strategy_name']}` "
        f"({float(top_mandate['mandate_aware_score']):.6f}).",
        f"- Best by robust score: `{top_robust['strategy_name']}` "
        f"({float(top_robust['robust_score']):.6f}).",
        "",
        "## Interpretation",
        "",
        "Preliminary evidence suggests the preferred TD3 candidate may change "
        "under asset-specific-cost-aware training. In this limited set, the "
        "combined report should be used as a prioritization diagnostic, not as "
        "a final superiority claim.",
        "",
        "Do not claim final superiority until benchmark comparisons and "
        "statistical validation are regenerated under the same asset-specific "
        "cost model.",
    ]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(["", "## Selection Table", ""])
    if selected.empty:
        lines.append("No selected candidates could be computed.")
    else:
        for _, row in selected.iterrows():
            lines.append(
                f"- {row['selection']}: `{row['strategy_name']}` "
                f"({row['selection_metric']} = {float(row['selection_metric_value']):.6f})."
            )
    return "\n".join(lines) + "\n"


def build_metadata(
    input_dirs: dict[str, Path],
    metadata_inputs: dict[str, dict[str, Any]],
    output_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Build metadata for the combined report."""
    cost_assumptions = extract_cost_assumptions(metadata_inputs)
    return {
        "runner": "src.analysis.asset_specific_cost_final_report",
        "output_dir": str(output_dir),
        "input_dirs": {key: str(path) for key, path in input_dirs.items()},
        "cost_assumptions": cost_assumptions,
        "warnings": warnings,
        "caveats": [
            "Limited retraining subset only; this is not the full original candidate universe.",
            "Scalar-cost and asset-specific-cost results are not directly interchangeable.",
            "Imported robust and mandate-aware scores may be normalized within each report universe.",
            "Benchmark comparisons and statistical validation must be regenerated under the same cost model before final claims.",
        ],
    }


def load_metadata(directory: Path) -> dict[str, Any]:
    """Load optional cap sensitivity metadata."""
    path = directory / CAP_METADATA_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_cost_assumptions(metadata_inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Collect cost assumptions from input metadata when available."""
    assumptions = {
        "transaction_cost_mode": "asset_specific",
        "asset_transaction_cost_bps": {
            "SPY": 2.0,
            "TLT": 2.0,
            "GLD": 2.0,
            "BTC-USD": 10.0,
            "CASH": 0.0,
        },
    }
    for metadata in metadata_inputs.values():
        if "transaction_cost" in metadata:
            assumptions["legacy_scalar_transaction_cost_field"] = metadata[
                "transaction_cost"
            ]
    return assumptions


def mean_or_na(frame: pd.DataFrame, column: str) -> float | Any:
    """Return column mean or missing value if unavailable."""
    if column not in frame.columns:
        return pd.NA
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return pd.NA
    return float(values.mean())


def format_cap_label(value: Any) -> str:
    """Format cap labels consistently."""
    if pd.isna(value):
        return "uncapped"
    return f"{float(value):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build combined report for asset-specific transaction cost retraining.",
    )
    parser.add_argument("--v3-dir", default=DEFAULT_V3_DIR)
    parser.add_argument("--v7-dir", default=DEFAULT_V7_DIR)
    parser.add_argument("--v4-dir", default=DEFAULT_V4_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report = build_asset_specific_cost_final_report(
        v3_dir=args.v3_dir,
        v7_dir=args.v7_dir,
        v4_dir=args.v4_dir,
        output_dir=args.output_dir,
    )
    print("Asset-specific cost selected candidates:")
    print(report["selected_candidates"].to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
