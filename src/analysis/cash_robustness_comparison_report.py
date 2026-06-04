"""Compare zero-CASH and BIL-CASH TD3 robustness reports.

This module is reporting-only. It consumes completed cap-sensitivity reports
and does not call TD3 training code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ZERO_DIR = "outputs/tables/final_corrected_limited_td3_60ep_10seeds"
BIL_DIR = (
    "/Users/thiagoherrera/Projects/portfolio_drl_outputs/"
    "final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds"
)
OUTPUT_DIR = "outputs/tables/final_corrected_cash_robustness_comparison"

ALL_RESULTS_FILE = "cap_sensitivity_all_results.csv"
SUMMARY_FILE = "cap_sensitivity_summary.csv"
METADATA_FILE = "cap_sensitivity_metadata.json"

CORE_METRICS = [
    "mandate_aware_score",
    "robust_score",
    "max_drawdown",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "average_turnover",
    "average_effective_number_of_assets",
    "mean_cash_weight",
    "mean_btc_weight",
    "mean_cash_transaction_cost_contribution",
    "mean_btc_transaction_cost_contribution",
]


def build_cash_robustness_comparison_report(
    zero_dir: str = ZERO_DIR,
    bil_dir: str = BIL_DIR,
    output_dir: str = OUTPUT_DIR,
) -> dict[str, Any]:
    """Build zero-CASH vs BIL-CASH robustness comparison outputs."""
    zero_path = Path(zero_dir)
    bil_path = Path(bil_dir).expanduser()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    zero_all, zero_summary = _load_experiment(zero_path, "zero-CASH")
    bil_all, bil_summary = _load_experiment(bil_path, "BIL-CASH")
    _validate_candidate_sets(zero_all, bil_all)

    candidate_comparison = build_candidate_comparison(
        zero_all=zero_all,
        zero_summary=zero_summary,
        bil_all=bil_all,
        bil_summary=bil_summary,
    )
    all_candidate_caps = build_all_candidate_caps_comparison(zero_all, bil_all)
    winner_summary = build_winner_summary(candidate_comparison, zero_all, bil_all)
    metadata = build_metadata(
        zero_path=zero_path,
        bil_path=bil_path,
        zero_all=zero_all,
        bil_all=bil_all,
    )
    markdown = build_markdown_summary(
        candidate_comparison=candidate_comparison,
        winner_summary=winner_summary,
    )

    paths = {
        "candidate_comparison": destination / "cash_robustness_candidate_comparison.csv",
        "all_candidate_caps": destination / "cash_robustness_all_candidate_caps.csv",
        "winner_summary": destination / "cash_robustness_winner_summary.csv",
        "summary_md": destination / "cash_robustness_summary.md",
        "metadata": destination / "cash_robustness_metadata.json",
    }
    candidate_comparison.to_csv(paths["candidate_comparison"], index=False)
    all_candidate_caps.to_csv(paths["all_candidate_caps"], index=False)
    winner_summary.to_csv(paths["winner_summary"], index=False)
    paths["summary_md"].write_text(markdown, encoding="utf-8")
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    return {
        "candidate_comparison": candidate_comparison,
        "all_candidate_caps": all_candidate_caps,
        "winner_summary": winner_summary,
        "metadata": metadata,
        "markdown": markdown,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def build_candidate_comparison(
    zero_all: pd.DataFrame,
    zero_summary: pd.DataFrame,
    bil_all: pd.DataFrame,
    bil_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare best-by-mandate rows per base candidate."""
    rows = []
    zero_winner = _winner_by_mandate(zero_all)
    bil_winner = _winner_by_mandate(bil_all)
    for candidate in sorted(zero_all["base_candidate"].dropna().unique()):
        zero_row = _best_row_for_candidate(zero_all, zero_summary, candidate)
        bil_row = _best_row_for_candidate(bil_all, bil_summary, candidate)
        zero_cap = str(zero_row["cap_label"])
        bil_cap = str(bil_row["cap_label"])
        rows.append(
            {
                "candidate": candidate,
                "zero_best_cap_by_mandate": zero_cap,
                "zero_mandate_aware_score": _num(zero_row["mandate_aware_score"]),
                "zero_robust_score": _num(zero_row["robust_score"]),
                "bil_best_cap_by_mandate": bil_cap,
                "bil_mandate_aware_score": _num(bil_row["mandate_aware_score"]),
                "bil_robust_score": _num(bil_row["robust_score"]),
                "delta_mandate_aware": _num(bil_row["mandate_aware_score"])
                - _num(zero_row["mandate_aware_score"]),
                "delta_robust": _num(bil_row["robust_score"])
                - _num(zero_row["robust_score"]),
                "cap_changed": zero_cap != bil_cap,
                "winner_changed": (
                    str(zero_winner["base_candidate"]) != str(bil_winner["base_candidate"])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["bil_mandate_aware_score", "delta_mandate_aware"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_all_candidate_caps_comparison(
    zero_all: pd.DataFrame,
    bil_all: pd.DataFrame,
) -> pd.DataFrame:
    """Compare every candidate/cap row where both experiments have the row."""
    zero = _normalize_all_results(zero_all, "zero")
    bil = _normalize_all_results(bil_all, "bil")
    merged = zero.merge(
        bil,
        on=["base_candidate", "cap_label"],
        how="inner",
        validate="one_to_one",
    )
    for metric in CORE_METRICS:
        zero_column = f"zero_{metric}"
        bil_column = f"bil_{metric}"
        if zero_column in merged and bil_column in merged:
            merged[f"delta_{metric}"] = (
                pd.to_numeric(merged[bil_column], errors="coerce")
                - pd.to_numeric(merged[zero_column], errors="coerce")
            )
    return merged.sort_values(
        ["base_candidate", "cap_label"],
        na_position="first",
    ).reset_index(drop=True)


def build_winner_summary(
    candidate_comparison: pd.DataFrame,
    zero_all: pd.DataFrame,
    bil_all: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize winner and cap changes across protocols."""
    zero_winner = _winner_by_mandate(zero_all)
    bil_winner = _winner_by_mandate(bil_all)
    zero_robust = _winner_by_robust(zero_all)
    bil_robust = _winner_by_robust(bil_all)
    improved = candidate_comparison.loc[
        pd.to_numeric(candidate_comparison["delta_mandate_aware"], errors="coerce") > 0,
        "candidate",
    ].tolist()
    deteriorated = candidate_comparison.loc[
        pd.to_numeric(candidate_comparison["delta_mandate_aware"], errors="coerce") < 0,
        "candidate",
    ].tolist()
    return pd.DataFrame(
        [
            {
                "zero_cash_winner_by_mandate": zero_winner["candidate_name"],
                "zero_cash_winner_cap": zero_winner["cap_label"],
                "zero_cash_winner_mandate_aware_score": zero_winner[
                    "mandate_aware_score"
                ],
                "zero_cash_winner_robust_score": zero_winner["robust_score"],
                "bil_cash_winner_by_mandate": bil_winner["candidate_name"],
                "bil_cash_winner_cap": bil_winner["cap_label"],
                "bil_cash_winner_mandate_aware_score": bil_winner[
                    "mandate_aware_score"
                ],
                "bil_cash_winner_robust_score": bil_winner["robust_score"],
                "winner_changed": str(zero_winner["candidate_name"])
                != str(bil_winner["candidate_name"]),
                "zero_cash_winner_by_robust": zero_robust["candidate_name"],
                "bil_cash_winner_by_robust": bil_robust["candidate_name"],
                "improved_under_bil_cash": ", ".join(improved),
                "deteriorated_under_bil_cash": ", ".join(deteriorated),
                "n_candidates_improved": len(improved),
                "n_candidates_deteriorated": len(deteriorated),
            }
        ]
    )


def build_markdown_summary(
    candidate_comparison: pd.DataFrame,
    winner_summary: pd.DataFrame,
) -> str:
    row = winner_summary.iloc[0]
    improved = row["improved_under_bil_cash"] or "None"
    deteriorated = row["deteriorated_under_bil_cash"] or "None"
    cap_changes = candidate_comparison.loc[
        candidate_comparison["cap_changed"].astype(bool),
        "candidate",
    ].tolist()
    lines = [
        "# Zero-CASH vs BIL-CASH Robustness Comparison",
        "",
        "This report compares completed corrected limited TD3 experiments. It does not retrain TD3.",
        "",
        "## Winner Comparison",
        "",
        (
            f"- Main zero-CASH protocol winner: `{row['zero_cash_winner_by_mandate']}` "
            f"at cap `{row['zero_cash_winner_cap']}` "
            f"(mandate-aware {float(row['zero_cash_winner_mandate_aware_score']):.6f}, "
            f"robust {float(row['zero_cash_winner_robust_score']):.6f})."
        ),
        (
            f"- BIL-CASH robustness winner: `{row['bil_cash_winner_by_mandate']}` "
            f"at cap `{row['bil_cash_winner_cap']}` "
            f"(mandate-aware {float(row['bil_cash_winner_mandate_aware_score']):.6f}, "
            f"robust {float(row['bil_cash_winner_robust_score']):.6f})."
        ),
        f"- Winner changed: `{bool(row['winner_changed'])}`.",
        f"- Preferred cap changed for: {', '.join(cap_changes) if cap_changes else 'none'}.",
        "",
        "## Candidate Movement",
        "",
        f"- Improved under BIL-CASH: {improved}.",
        f"- Deteriorated under BIL-CASH: {deteriorated}.",
        "",
        "## Interpretation",
        "",
        "The cash-return assumption materially affects the selected TD3 specification.",
        "",
        "Caution: This is robustness evidence, not statistical superiority evidence.",
        "",
    ]
    return "\n".join(lines)


def build_metadata(
    zero_path: Path,
    bil_path: Path,
    zero_all: pd.DataFrame,
    bil_all: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "report": "cash_robustness_comparison",
        "reporting_only": True,
        "td3_training_called": False,
        "zero_cash_dir": str(zero_path),
        "bil_cash_dir": str(bil_path),
        "zero_cash_rows": int(len(zero_all)),
        "bil_cash_rows": int(len(bil_all)),
        "zero_cash_candidates": sorted(zero_all["base_candidate"].dropna().unique().tolist()),
        "bil_cash_candidates": sorted(bil_all["base_candidate"].dropna().unique().tolist()),
        "zero_cash_protocol": "synthetic zero-return CASH",
        "bil_cash_protocol": "CASH returns mapped to BIL short-term Treasury ETF proxy",
        "caveat": "Robustness comparison only; not statistical superiority evidence.",
    }


def _load_experiment(path: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_path = path / ALL_RESULTS_FILE
    summary_path = path / SUMMARY_FILE
    if not all_path.exists():
        raise FileNotFoundError(f"Missing {label} all-results file: {all_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {label} summary file: {summary_path}")
    all_results = pd.read_csv(all_path)
    summary = pd.read_csv(summary_path)
    required = {"base_candidate", "candidate_name", "cap_label"}
    missing = required - set(all_results.columns)
    if missing:
        raise ValueError(f"{label} all-results missing columns: {sorted(missing)}")
    return all_results, summary


def _validate_candidate_sets(zero_all: pd.DataFrame, bil_all: pd.DataFrame) -> None:
    zero_candidates = set(zero_all["base_candidate"].dropna().astype(str))
    bil_candidates = set(bil_all["base_candidate"].dropna().astype(str))
    if zero_candidates != bil_candidates:
        raise ValueError(
            "Candidate sets do not match: "
            f"zero_only={sorted(zero_candidates - bil_candidates)}, "
            f"bil_only={sorted(bil_candidates - zero_candidates)}"
        )


def _best_row_for_candidate(
    all_results: pd.DataFrame,
    summary: pd.DataFrame,
    candidate: str,
) -> pd.Series:
    summary_row = summary.loc[summary["base_candidate"].astype(str) == candidate]
    if not summary_row.empty and "best_cap_by_mandate_aware_score" in summary_row:
        cap_label = str(summary_row.iloc[0]["best_cap_by_mandate_aware_score"])
        matches = all_results.loc[
            (all_results["base_candidate"].astype(str) == candidate)
            & (all_results["cap_label"].astype(str) == cap_label)
        ]
        if not matches.empty:
            return matches.iloc[0]
    group = all_results.loc[all_results["base_candidate"].astype(str) == candidate]
    return group.sort_values(
        ["mandate_aware_score", "robust_score"],
        ascending=[False, False],
    ).iloc[0]


def _normalize_all_results(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = ["base_candidate", "candidate_name", "cap_label", *CORE_METRICS]
    result = frame.copy()
    for column in keep:
        if column not in result:
            result[column] = pd.NA
    result = result.loc[:, keep]
    rename = {
        column: f"{prefix}_{column}"
        for column in result.columns
        if column not in {"base_candidate", "cap_label"}
    }
    return result.rename(columns=rename)


def _winner_by_mandate(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(
        ["mandate_aware_score", "robust_score"],
        ascending=[False, False],
    ).iloc[0]


def _winner_by_robust(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(
        ["robust_score", "mandate_aware_score"],
        ascending=[False, False],
    ).iloc[0]


def _num(value) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build zero-CASH vs BIL-CASH robustness comparison report."
    )
    parser.add_argument("--zero-dir", default=ZERO_DIR)
    parser.add_argument("--bil-dir", default=BIL_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    report = build_cash_robustness_comparison_report(
        zero_dir=args.zero_dir,
        bil_dir=args.bil_dir,
        output_dir=args.output_dir,
    )
    winner = report["winner_summary"].iloc[0]
    print("Cash robustness comparison complete.")
    print("zero_winner:", winner["zero_cash_winner_by_mandate"])
    print("bil_winner:", winner["bil_cash_winner_by_mandate"])
    print("winner_changed:", winner["winner_changed"])
    print("outputs:")
    for path in report["paths"].values():
        print(path)


if __name__ == "__main__":
    main()
