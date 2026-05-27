"""Validate V3 real-time vintage macro data and alignment.

This is a reporting gate only. It checks that weekly macro values and their
provenance use information available as of each weekly date, compares the
result with the current-vintage macro dataset, and writes audit tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.validate_v3_macro_current import (
    build_alignment_checks,
    build_feature_summary,
    build_macro_coverage_table,
    load_returns_csv,
    validate_alignment_checks,
    validate_macro_coverage,
)
from src.data.features_v3 import build_features_v3
from src.data.macro_loader import load_macro_data_from_csv


DEFAULT_RETURNS_PATH = "data/processed/returns_weekly_latest.csv"
DEFAULT_MACRO_REALTIME_PATH = "data/processed/macro_weekly_realtime_latest.csv"
DEFAULT_MACRO_CURRENT_PATH = "data/processed/macro_weekly_latest.csv"
DEFAULT_OUTPUT_DIR = "outputs/tables/v3_macro_realtime_validation"
REALTIME_MACRO_NOTE = (
    "Real-time macro values use FRED as-of observations where available. "
    "Series unavailable in ALFRED are included only with explicit fallback flags."
)


def validate_v3_macro_realtime(
    returns_path: str = DEFAULT_RETURNS_PATH,
    macro_realtime_path: str = DEFAULT_MACRO_REALTIME_PATH,
    macro_current_path: str = DEFAULT_MACRO_CURRENT_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    metadata_path: str | None = None,
) -> dict[str, Any]:
    """Validate real-time macro coverage, leakage, and feature alignment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_path or str(
        output_path / "v3_macro_realtime_series_metadata.csv"
    )

    returns = load_returns_csv(returns_path)
    realtime_macro = load_macro_data_from_csv(macro_realtime_path, date_column="date")
    current_macro = load_macro_data_from_csv(macro_current_path, date_column="date")
    metadata = load_realtime_metadata(metadata_path)

    coverage = build_macro_coverage_table(
        returns,
        realtime_macro,
        returns_path,
        macro_realtime_path,
    )
    coverage["cpi_lag_note"] = REALTIME_MACRO_NOTE
    validate_macro_coverage(coverage)

    raw_features = build_features_v3(returns, macro_data=realtime_macro)
    shifted_features = raw_features.shift(1).dropna()
    aligned_index = returns.index[returns.index.isin(shifted_features.index)]
    aligned_returns = returns.loc[aligned_index]
    aligned_features = shifted_features.loc[aligned_index]

    feature_summary = build_feature_summary(
        returns=returns,
        macro=realtime_macro,
        raw_features=raw_features,
        aligned_returns=aligned_returns,
        aligned_features=aligned_features,
    )
    alignment_checks = build_alignment_checks(
        returns=returns,
        macro=realtime_macro,
        aligned_features=aligned_features,
    )
    leakage_checks = build_leakage_checks(metadata)
    comparison = compare_current_vs_realtime(current_macro, realtime_macro)
    series_summary = build_series_metadata_summary(metadata)

    validate_alignment_checks(alignment_checks)
    validate_leakage_checks(leakage_checks)

    paths = {
        "coverage": output_path / "v3_macro_realtime_coverage.csv",
        "series_metadata": output_path / "v3_macro_realtime_series_metadata.csv",
        "series_summary": output_path / "v3_macro_realtime_series_summary.csv",
        "alignment_checks": output_path / "v3_macro_realtime_alignment_checks.csv",
        "leakage_checks": output_path / "v3_macro_realtime_leakage_checks.csv",
        "current_vs_realtime": output_path / "v3_macro_realtime_current_vs_realtime.csv",
        "summary": output_path / "v3_macro_realtime_summary.md",
    }
    coverage.to_csv(paths["coverage"], index=False)
    metadata.to_csv(paths["series_metadata"], index=False)
    series_summary.to_csv(paths["series_summary"], index=False)
    alignment_checks.to_csv(paths["alignment_checks"], index=False)
    leakage_checks.to_csv(paths["leakage_checks"], index=False)
    comparison.to_csv(paths["current_vs_realtime"], index=False)
    summary = build_summary_markdown(
        coverage=coverage,
        feature_summary=feature_summary,
        alignment_checks=alignment_checks,
        leakage_checks=leakage_checks,
        series_summary=series_summary,
        comparison=comparison,
    )
    paths["summary"].write_text(summary, encoding="utf-8")

    return {
        "coverage": coverage,
        "feature_summary": feature_summary,
        "alignment_checks": alignment_checks,
        "leakage_checks": leakage_checks,
        "series_summary": series_summary,
        "current_vs_realtime": comparison,
        "summary": summary,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def load_realtime_metadata(path: str) -> pd.DataFrame:
    """Load builder provenance metadata."""
    frame = pd.read_csv(path)
    required = {
        "date",
        "series_id",
        "output_name",
        "observation_date_used",
        "as_of_date",
        "realtime_start_used",
        "realtime_end_used",
        "vintage_method",
        "true_vintage_data_available",
        "fallback_used",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            "Real-time macro metadata is missing columns: "
            f"{', '.join(sorted(missing))}"
        )
    for column in ("date", "observation_date_used", "as_of_date", "realtime_start_used"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["realtime_end_parsed"] = frame["realtime_end_used"].map(_parse_realtime_end)
    return frame


def build_leakage_checks(metadata: pd.DataFrame) -> pd.DataFrame:
    """Build row-level as-of leakage checks."""
    rows = []
    checks = {
        "as_of_date_not_after_weekly_date": metadata["as_of_date"] <= metadata["date"],
        "observation_date_not_after_weekly_date": metadata["observation_date_used"]
        <= metadata["date"],
        "realtime_start_not_after_weekly_date": metadata["realtime_start_used"]
        <= metadata["date"],
        "weekly_date_inside_vintage_interval": metadata["realtime_end_parsed"]
        >= metadata["date"],
    }
    for name, mask in checks.items():
        failures = int((~mask.fillna(False)).sum())
        rows.append(
            {
                "check_name": name,
                "status": "pass" if failures == 0 else "fail",
                "failure_count": failures,
                "n_rows": len(metadata),
            }
        )

    cpi = metadata[metadata["output_name"] == "CPI"]
    cpi_failures = int(
        (
            (cpi["realtime_start_used"] > cpi["date"])
            | (cpi["observation_date_used"] > cpi["date"])
        ).sum()
    )
    rows.append(
        {
            "check_name": "cpi_release_timing_respected",
            "status": "pass" if cpi_failures == 0 else "fail",
            "failure_count": cpi_failures,
            "n_rows": len(cpi),
        }
    )
    return pd.DataFrame(rows)


def validate_leakage_checks(leakage_checks: pd.DataFrame) -> None:
    failing = leakage_checks[leakage_checks["status"] == "fail"]
    if not failing.empty:
        names = ", ".join(failing["check_name"].astype(str))
        raise ValueError(f"V3 real-time macro leakage checks failed: {names}")


def compare_current_vs_realtime(
    current_macro: pd.DataFrame,
    realtime_macro: pd.DataFrame,
) -> pd.DataFrame:
    """Compare current-vintage and real-time-vintage weekly values."""
    common_index = current_macro.index.intersection(realtime_macro.index)
    common_columns = [column for column in realtime_macro.columns if column in current_macro.columns]
    rows = []
    for column in common_columns:
        diff = (
            current_macro.loc[common_index, column].astype(float)
            - realtime_macro.loc[common_index, column].astype(float)
        ).abs()
        max_diff = float(diff.max()) if not diff.empty else float("nan")
        rows.append(
            {
                "series": column,
                "n_common_dates": len(diff),
                "mean_absolute_difference": float(diff.mean()) if not diff.empty else float("nan"),
                "max_absolute_difference": max_diff,
                "date_max_difference": diff.idxmax() if not diff.empty else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


def build_series_metadata_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    """Summarize vintage/fallback status by output series."""
    grouped = []
    for output_name, frame in metadata.groupby("output_name", sort=True):
        fallback_values = []
        if "fallback_method" in frame.columns:
            fallback_values = [
                str(value)
                for value in frame["fallback_method"].dropna().unique()
                if str(value)
            ]
        grouped.append(
            {
                "output_name": output_name,
                "series_id": frame["series_id"].iloc[0],
                "vintage_method": ",".join(sorted(frame["vintage_method"].astype(str).unique())),
                "true_vintage_data_available": bool(
                    frame["true_vintage_data_available"].astype(bool).all()
                ),
                "fallback_used_count": int(frame["fallback_used"].astype(bool).sum()),
                "fallback_method": ",".join(sorted(fallback_values)),
                "first_weekly_date": frame["date"].min(),
                "last_weekly_date": frame["date"].max(),
                "latest_observation_date_used": frame["observation_date_used"].max(),
                "latest_as_of_date_used": frame["as_of_date"].max(),
                "n_rows": len(frame),
            }
        )
    return pd.DataFrame(grouped)


def build_summary_markdown(
    coverage: pd.DataFrame,
    feature_summary: pd.DataFrame,
    alignment_checks: pd.DataFrame,
    leakage_checks: pd.DataFrame,
    series_summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    coverage_row = coverage.iloc[0]
    feature_row = feature_summary.iloc[0]
    alignment_status = alignment_checks["status"].value_counts().to_dict()
    leakage_status = leakage_checks["status"].value_counts().to_dict()
    fallback_count = int(series_summary["fallback_used_count"].sum())
    eligible = (
        bool(coverage_row["is_current_window_covered"])
        and int(feature_row["missing_aligned_macro_features"]) == 0
        and "fail" not in alignment_status
        and "fail" not in leakage_status
    )
    revision_lines = [
        (
            f"- {row['series']}: mean abs diff "
            f"{row['mean_absolute_difference']:.6g}, max abs diff "
            f"{row['max_absolute_difference']:.6g} on {row['date_max_difference']}"
        )
        for _, row in comparison.iterrows()
    ]
    series_lines = [
        (
            f"- {row['output_name']} ({row['series_id']}): "
            f"method={row['vintage_method']}, "
            f"true_vintage={row['true_vintage_data_available']}, "
            f"fallback_used_count={row['fallback_used_count']}"
        )
        for _, row in series_summary.iterrows()
    ]
    return "\n".join(
        [
            "# V3 Macro Real-Time Vintage Validation",
            "",
            f"Returns coverage: {coverage_row['returns_start']} to {coverage_row['returns_end']}.",
            f"Real-time macro coverage: {coverage_row['macro_start']} to {coverage_row['macro_end']}.",
            f"Macro missing values: {coverage_row['macro_missing_values']}.",
            f"Macro feature count: {feature_row['n_macro_features']}.",
            f"Aligned rows: {feature_row['aligned_rows']}.",
            f"Alignment status counts: {alignment_status}.",
            f"Leakage status counts: {leakage_status}.",
            f"Fallback rows: {fallback_count}.",
            "",
            "Series vintage status:",
            *series_lines,
            "",
            "Current-vintage versus real-time-vintage differences:",
            *revision_lines,
            "",
            (
                (
                    "Eligibility for V3_real_macro_vintage protocol smoke: yes, "
                    "with documented fallback caveats."
                )
                if eligible
                else "Eligibility for V3_real_macro_vintage protocol smoke: no."
            ),
            "",
            "This validation is data/feature smoke only. It does not run TD3 training.",
            "",
        ]
    )


def _parse_realtime_end(value: object) -> pd.Timestamp:
    text = str(value)
    if text == "9999-12-31":
        return pd.Timestamp.max.normalize()
    return pd.Timestamp(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate V3 real-time vintage macro data.",
    )
    parser.add_argument("--returns-path", default=DEFAULT_RETURNS_PATH)
    parser.add_argument("--macro-realtime-path", default=DEFAULT_MACRO_REALTIME_PATH)
    parser.add_argument("--macro-current-path", default=DEFAULT_MACRO_CURRENT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-path", default=None)
    args = parser.parse_args()

    try:
        result = validate_v3_macro_realtime(
            returns_path=args.returns_path,
            macro_realtime_path=args.macro_realtime_path,
            macro_current_path=args.macro_current_path,
            output_dir=args.output_dir,
            metadata_path=args.metadata_path,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print("Coverage:")
    print(result["coverage"].to_string(index=False))
    print("\nSeries metadata:")
    print(result["series_summary"].to_string(index=False))
    print("\nLeakage checks:")
    print(result["leakage_checks"].to_string(index=False))
    print("\nCurrent-vintage comparison:")
    print(result["current_vs_realtime"].to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
