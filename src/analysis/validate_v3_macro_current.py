"""Validate current-window V3 macro data and feature alignment.

This module is a reporting gate for V3 real macro features. It does not train
TD3 and does not change feature, reward, environment, or scoring logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.features_v3 import build_features_v3
from src.data.macro_loader import load_macro_data_from_csv


DEFAULT_RETURNS_PATH = "data/processed/returns_weekly_latest.csv"
DEFAULT_MACRO_PATH = "data/processed/macro_weekly_latest.csv"
DEFAULT_OUTPUT_DIR = "outputs/tables/v3_macro_current_validation"
CPI_LAG_NOTE = (
    "CPI uses the existing conservative four-week availability lag. This is "
    "not a real-time vintage macro database or a full release-calendar model."
)

PROTOCOL_FOLDS = [
    {
        "fold": "F1",
        "train_start": "2015-04-03",
        "train_end": "2020-12-25",
        "validation_start": "2021-01-01",
        "validation_end": "2021-12-31",
        "test_start": "2022-01-07",
        "test_end": "2022-12-30",
    },
    {
        "fold": "F2",
        "train_start": "2015-04-03",
        "train_end": "2021-12-31",
        "validation_start": "2022-01-07",
        "validation_end": "2022-12-30",
        "test_start": "2023-01-06",
        "test_end": "2023-12-29",
    },
    {
        "fold": "F3",
        "train_start": "2015-04-03",
        "train_end": "2022-12-30",
        "validation_start": "2023-01-06",
        "validation_end": "2023-12-29",
        "test_start": "2024-01-05",
        "test_end": "2024-12-27",
    },
    {
        "fold": "F4",
        "train_start": "2015-04-03",
        "train_end": "2023-12-29",
        "validation_start": "2024-01-05",
        "validation_end": "2024-12-27",
        "test_start": "2025-01-03",
        "test_end": "2026-05-15",
    },
]


def validate_v3_macro_current(
    returns_path: str = DEFAULT_RETURNS_PATH,
    macro_path: str = DEFAULT_MACRO_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Validate V3 macro coverage, feature availability, and fold alignment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    returns = load_returns_csv(returns_path)
    macro = load_macro_data_from_csv(macro_path, date_column="date")
    coverage = build_macro_coverage_table(returns, macro, returns_path, macro_path)
    validate_macro_coverage(coverage)

    raw_features = build_features_v3(returns, macro_data=macro)
    shifted_features = raw_features.shift(1).dropna()
    aligned_index = returns.index[returns.index.isin(shifted_features.index)]
    aligned_returns = returns.loc[aligned_index]
    aligned_features = shifted_features.loc[aligned_index]

    feature_summary = build_feature_summary(
        returns=returns,
        macro=macro,
        raw_features=raw_features,
        aligned_returns=aligned_returns,
        aligned_features=aligned_features,
    )
    alignment_checks = build_alignment_checks(
        returns=returns,
        macro=macro,
        aligned_features=aligned_features,
    )
    validate_alignment_checks(alignment_checks)

    coverage_path = output_path / "v3_macro_current_coverage.csv"
    feature_summary_path = output_path / "v3_macro_current_feature_summary.csv"
    alignment_checks_path = output_path / "v3_macro_current_alignment_checks.csv"
    summary_path = output_path / "v3_macro_current_summary.md"

    coverage.to_csv(coverage_path, index=False)
    feature_summary.to_csv(feature_summary_path, index=False)
    alignment_checks.to_csv(alignment_checks_path, index=False)
    summary = build_summary_markdown(coverage, feature_summary, alignment_checks)
    summary_path.write_text(summary, encoding="utf-8")

    return {
        "coverage": coverage,
        "feature_summary": feature_summary,
        "alignment_checks": alignment_checks,
        "summary": summary,
        "paths": {
            "coverage": str(coverage_path),
            "feature_summary": str(feature_summary_path),
            "alignment_checks": str(alignment_checks_path),
            "summary": str(summary_path),
        },
    }


def load_returns_csv(path: str) -> pd.DataFrame:
    """Load weekly returns with a date index."""
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise KeyError("Returns CSV must contain a date column.")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.empty:
        raise ValueError("Returns CSV produced an empty DataFrame.")
    if frame.isna().any().any():
        raise ValueError("Returns CSV contains missing or non-numeric values.")
    return frame


def build_macro_coverage_table(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    returns_path: str,
    macro_path: str,
) -> pd.DataFrame:
    """Build one-row macro/returns coverage table."""
    returns_start = returns.index.min()
    returns_end = returns.index.max()
    macro_start = macro.index.min()
    macro_end = macro.index.max()
    return pd.DataFrame(
        [
            {
                "returns_path": returns_path,
                "macro_path": macro_path,
                "returns_start": returns_start,
                "returns_end": returns_end,
                "macro_start": macro_start,
                "macro_end": macro_end,
                "macro_covers_returns_start": bool(macro_start <= returns_start),
                "macro_covers_returns_end": bool(macro_end >= returns_end),
                "macro_missing_values": int(macro.isna().sum().sum()),
                "returns_missing_values": int(returns.isna().sum().sum()),
                "macro_columns": ",".join(macro.columns.astype(str)),
                "cpi_lag_note": CPI_LAG_NOTE,
                "is_current_window_covered": bool(
                    macro_start <= returns_start
                    and macro_end >= returns_end
                    and int(macro.isna().sum().sum()) == 0
                ),
            }
        ]
    )


def validate_macro_coverage(coverage: pd.DataFrame) -> None:
    """Fail fast when macro coverage is stale or unusable."""
    row = coverage.iloc[0]
    if not bool(row["macro_covers_returns_end"]):
        raise ValueError(
            "V3 macro coverage is stale: macro_end "
            f"{row['macro_end']} is before returns_end {row['returns_end']}."
        )
    if not bool(row["macro_covers_returns_start"]):
        raise ValueError(
            "V3 macro coverage starts after returns_start: macro_start "
            f"{row['macro_start']} > returns_start {row['returns_start']}."
        )
    if int(row["macro_missing_values"]) != 0:
        raise ValueError("V3 macro data contains missing values.")


def build_feature_summary(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    raw_features: pd.DataFrame,
    aligned_returns: pd.DataFrame,
    aligned_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build compact V3 feature smoke summary."""
    macro_columns = [column for column in raw_features.columns if column.startswith("macro_")]
    return pd.DataFrame(
        [
            {
                "returns_rows": len(returns),
                "macro_rows": len(macro),
                "raw_feature_rows": len(raw_features),
                "aligned_rows": len(aligned_features),
                "aligned_returns_rows": len(aligned_returns),
                "n_features": raw_features.shape[1],
                "n_macro_features": len(macro_columns),
                "missing_raw_features": int(raw_features.isna().sum().sum()),
                "missing_aligned_features": int(aligned_features.isna().sum().sum()),
                "missing_aligned_macro_features": int(
                    aligned_features.loc[:, macro_columns].isna().sum().sum()
                ),
                "first_aligned_date": aligned_features.index.min(),
                "last_aligned_date": aligned_features.index.max(),
                "macro_feature_columns": ",".join(macro_columns),
            }
        ]
    )


def build_alignment_checks(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    aligned_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build global and fold-level timing/alignment checks."""
    rows = [
        {
            "check_name": "macro_no_backfill_required",
            "status": "pass" if macro.index.min() <= returns.index.min() else "fail",
            "detail": "macro_start <= returns_start",
            "actual_start": macro.index.min(),
            "actual_end": macro.index.max(),
            "requested_start": returns.index.min(),
            "requested_end": returns.index.max(),
            "n_rows": len(macro),
        },
        {
            "check_name": "features_do_not_overrun_returns",
            "status": "pass"
            if aligned_features.index.max() <= returns.index.max()
            else "fail",
            "detail": "last feature date <= last return date",
            "actual_start": aligned_features.index.min(),
            "actual_end": aligned_features.index.max(),
            "requested_start": returns.index.min(),
            "requested_end": returns.index.max(),
            "n_rows": len(aligned_features),
        },
    ]

    for fold in PROTOCOL_FOLDS:
        for split in ("train", "validation", "test"):
            requested_start = pd.Timestamp(fold[f"{split}_start"])
            requested_end = pd.Timestamp(fold[f"{split}_end"])
            actual = aligned_features.loc[
                (aligned_features.index >= requested_start)
                & (aligned_features.index <= requested_end)
            ]
            status = "pass" if not actual.empty else "fail"
            if split in {"validation", "test"} and not actual.empty:
                status = (
                    "pass"
                    if actual.index.min() == requested_start
                    and actual.index.max() == requested_end
                    else "warning"
                )
            rows.append(
                {
                    "check_name": f"{fold['fold']}_{split}_window_alignment",
                    "status": status,
                    "detail": (
                        "validation/test should match requested windows; train can "
                        "start later only if feature warmup requires it"
                    ),
                    "actual_start": actual.index.min() if not actual.empty else pd.NaT,
                    "actual_end": actual.index.max() if not actual.empty else pd.NaT,
                    "requested_start": requested_start,
                    "requested_end": requested_end,
                    "n_rows": len(actual),
                }
            )
    return pd.DataFrame(rows)


def validate_alignment_checks(alignment_checks: pd.DataFrame) -> None:
    """Raise if any required alignment check fails."""
    failing = alignment_checks[alignment_checks["status"] == "fail"]
    if not failing.empty:
        names = ", ".join(failing["check_name"].astype(str).tolist())
        raise ValueError(f"V3 macro alignment checks failed: {names}")


def build_summary_markdown(
    coverage: pd.DataFrame,
    feature_summary: pd.DataFrame,
    alignment_checks: pd.DataFrame,
) -> str:
    """Build a short Markdown smoke report."""
    coverage_row = coverage.iloc[0]
    feature_row = feature_summary.iloc[0]
    status_counts = alignment_checks["status"].value_counts().to_dict()
    eligible = (
        bool(coverage_row["is_current_window_covered"])
        and int(feature_row["missing_aligned_macro_features"]) == 0
        and "fail" not in status_counts
    )
    return "\n".join(
        [
            "# V3 Macro Current Validation",
            "",
            f"Returns coverage: {coverage_row['returns_start']} to {coverage_row['returns_end']}.",
            f"Macro coverage: {coverage_row['macro_start']} to {coverage_row['macro_end']}.",
            f"Macro missing values: {coverage_row['macro_missing_values']}.",
            f"V3 feature count: {feature_row['n_features']}.",
            f"Macro feature count: {feature_row['n_macro_features']}.",
            f"Aligned rows: {feature_row['aligned_rows']}.",
            f"Alignment status counts: {status_counts}.",
            "",
            f"CPI limitation: {CPI_LAG_NOTE}",
            "",
            (
                "Eligibility for protocol smoke: yes."
                if eligible
                else "Eligibility for protocol smoke: no."
            ),
            "",
            "This validation is data/feature smoke only. It does not run TD3 training.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate current-window V3 macro data and feature alignment.",
    )
    parser.add_argument("--returns-path", default=DEFAULT_RETURNS_PATH)
    parser.add_argument("--macro-path", default=DEFAULT_MACRO_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = validate_v3_macro_current(
        returns_path=args.returns_path,
        macro_path=args.macro_path,
        output_dir=args.output_dir,
    )
    print("Coverage:")
    print(result["coverage"].to_string(index=False))
    print("\nFeature summary:")
    print(result["feature_summary"].to_string(index=False))
    print("\nAlignment checks:")
    print(result["alignment_checks"].to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
