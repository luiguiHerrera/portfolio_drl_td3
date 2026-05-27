"""Validate current-window V7 macro plus real GARCH features.

This reporting gate checks that V7 combines current-window local macro data
with rolling fitted real-GARCH features without introducing missing aligned
features, CASH GARCH columns, stale macro data, or feature dates beyond returns.
It does not train TD3.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.validate_v3_macro_current import (
    DEFAULT_MACRO_PATH,
    DEFAULT_RETURNS_PATH,
    PROTOCOL_FOLDS,
    build_macro_coverage_table,
    load_returns_csv,
    validate_macro_coverage,
)
from src.data.features_v7 import build_features_v7
from src.data.macro_loader import load_macro_data_from_csv


DEFAULT_OUTPUT_DIR = "outputs/tables/v7_macro_garch_current_validation"
MODEL_SPEC = "zero-mean normal GARCH(1,1), one-step-ahead forecast"


def validate_v7_macro_garch_current(
    returns_path: str = DEFAULT_RETURNS_PATH,
    macro_path: str = DEFAULT_MACRO_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Validate V7 feature coverage, GARCH diagnostics, and alignment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    returns = load_returns_csv(returns_path)
    macro = load_macro_data_from_csv(macro_path, date_column="date")
    macro_coverage = build_macro_coverage_table(returns, macro, returns_path, macro_path)
    validate_macro_coverage(macro_coverage)

    raw_features, diagnostics = build_features_v7(
        returns=returns,
        macro_data=macro,
        garch_mode="rolling_fitted",
        garch_min_history=104,
        garch_window=156,
        garch_exclude_cash=True,
        garch_fallback="rolling_realized_vol",
        return_garch_diagnostics=True,
    )
    shifted = raw_features.shift(1).dropna()
    aligned_index = returns.index[returns.index.isin(shifted.index)]
    aligned_features = shifted.loc[aligned_index]

    coverage = build_coverage_table(
        returns=returns,
        macro_coverage=macro_coverage,
        features=raw_features,
        diagnostics=diagnostics,
        returns_path=returns_path,
        macro_path=macro_path,
    )
    feature_summary = build_feature_summary(
        raw_features=raw_features,
        aligned_features=aligned_features,
    )
    alignment_checks = build_alignment_checks(
        returns=returns,
        raw_features=raw_features,
        aligned_features=aligned_features,
        diagnostics=diagnostics,
    )
    validate_alignment_checks(alignment_checks)

    paths = {
        "coverage": output_path / "v7_macro_garch_current_coverage.csv",
        "feature_summary": output_path / "v7_macro_garch_current_feature_summary.csv",
        "alignment_checks": output_path / "v7_macro_garch_current_alignment_checks.csv",
        "summary": output_path / "v7_macro_garch_current_summary.md",
    }
    coverage.to_csv(paths["coverage"], index=False)
    feature_summary.to_csv(paths["feature_summary"], index=False)
    alignment_checks.to_csv(paths["alignment_checks"], index=False)
    summary = build_summary_markdown(coverage, feature_summary, alignment_checks)
    paths["summary"].write_text(summary, encoding="utf-8")

    return {
        "coverage": coverage,
        "feature_summary": feature_summary,
        "alignment_checks": alignment_checks,
        "summary": summary,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def build_coverage_table(
    returns: pd.DataFrame,
    macro_coverage: pd.DataFrame,
    features: pd.DataFrame,
    diagnostics: pd.DataFrame,
    returns_path: str,
    macro_path: str,
) -> pd.DataFrame:
    """Build compact V7 coverage table."""
    fitted_assets = sorted(diagnostics["asset"].dropna().unique()) if not diagnostics.empty else []
    macro_row = macro_coverage.iloc[0]
    return pd.DataFrame(
        [
            {
                "returns_path": returns_path,
                "macro_path": macro_path,
                "returns_start": returns.index.min(),
                "returns_end": returns.index.max(),
                "macro_start": macro_row["macro_start"],
                "macro_end": macro_row["macro_end"],
                "feature_start": features.index.min(),
                "feature_end": features.index.max(),
                "returns_rows": len(returns),
                "feature_rows": len(features),
                "macro_covers_returns_end": bool(macro_row["macro_covers_returns_end"]),
                "garch_mode": "rolling_fitted",
                "garch_backend": ",".join(sorted(diagnostics["backend"].dropna().astype(str).unique())),
                "model_spec": MODEL_SPEC,
                "volatility_unit": "weekly",
                "cash_handling": "excluded_from_fitted_garch",
                "n_assets_fitted": len(fitted_assets),
                "assets_fitted": ",".join(fitted_assets),
                "fit_success_count": int((diagnostics["status"] == "fitted").sum()),
                "fallback_count": int((diagnostics["status"] == "fallback").sum()),
                "fit_failure_count": int(
                    diagnostics["fallback_reason"].fillna("").astype(str).str.contains("fit_failed").sum()
                ),
            }
        ]
    )


def build_feature_summary(
    raw_features: pd.DataFrame,
    aligned_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature-count summary."""
    macro_columns = [column for column in raw_features.columns if column.startswith("macro_")]
    garch_columns = [column for column in raw_features.columns if column.startswith("garch_")]
    cash_garch_columns = [column for column in garch_columns if "CASH" in column]
    return pd.DataFrame(
        [
            {
                "raw_feature_rows": len(raw_features),
                "aligned_feature_rows": len(aligned_features),
                "n_features": raw_features.shape[1],
                "n_macro_features": len(macro_columns),
                "n_garch_features": len(garch_columns),
                "cash_garch_column_count": len(cash_garch_columns),
                "cash_garch_columns": ",".join(cash_garch_columns),
                "missing_raw_features": int(raw_features.isna().sum().sum()),
                "missing_aligned_features": int(aligned_features.isna().sum().sum()),
                "first_aligned_date": aligned_features.index.min(),
                "last_aligned_date": aligned_features.index.max(),
            }
        ]
    )


def build_alignment_checks(
    returns: pd.DataFrame,
    raw_features: pd.DataFrame,
    aligned_features: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Build pass/fail alignment and leakage guard checks."""
    macro_columns = [column for column in raw_features.columns if column.startswith("macro_")]
    garch_columns = [column for column in raw_features.columns if column.startswith("garch_")]
    fitted = diagnostics[diagnostics["status"] == "fitted"]
    rows = [
        _check(
            "feature_dates_do_not_overrun_returns",
            bool(raw_features.index.max() <= returns.index.max()),
            f"feature_end={raw_features.index.max()}, returns_end={returns.index.max()}",
        ),
        _check(
            "aligned_features_have_no_missing_values",
            int(aligned_features.isna().sum().sum()) == 0,
            f"missing_aligned_features={int(aligned_features.isna().sum().sum())}",
        ),
        _check(
            "macro_columns_present",
            bool(macro_columns),
            f"n_macro_features={len(macro_columns)}",
        ),
        _check(
            "garch_columns_present",
            bool(garch_columns),
            f"n_garch_features={len(garch_columns)}",
        ),
        _check(
            "cash_garch_excluded",
            not any("CASH" in column for column in garch_columns),
            "CASH must not receive fitted GARCH columns.",
        ),
        _check(
            "garch_diagnostics_present",
            not diagnostics.empty,
            f"n_diagnostic_rows={len(diagnostics)}",
        ),
        _check(
            "garch_uses_arch_model",
            (not fitted.empty) and set(fitted["backend"].dropna()) == {"arch_model"},
            f"fitted_backends={sorted(fitted['backend'].dropna().astype(str).unique())}",
        ),
        _check(
            "garch_fit_failures_absent",
            not diagnostics["fallback_reason"].fillna("").astype(str).str.contains("fit_failed").any(),
            "fallback reasons should not include fit_failed.",
        ),
        _check(
            "validation_test_windows_align",
            protocol_windows_align(aligned_features),
            "Protocol validation/test windows have V7 feature coverage where possible.",
        ),
    ]
    return pd.DataFrame(rows)


def validate_alignment_checks(alignment_checks: pd.DataFrame) -> None:
    """Raise when any alignment check fails."""
    failed = alignment_checks[alignment_checks["status"] != "pass"]
    if not failed.empty:
        raise ValueError(
            "V7 macro/GARCH validation failed: "
            + "; ".join(failed["check_name"].astype(str))
        )


def protocol_windows_align(aligned_features: pd.DataFrame) -> bool:
    """Check protocol validation/test starts when those dates are in range."""
    if aligned_features.empty:
        return False
    for fold in PROTOCOL_FOLDS:
        for split in ("validation", "test"):
            start = pd.Timestamp(fold[f"{split}_start"])
            if aligned_features.index.min() <= start <= aligned_features.index.max():
                if start not in aligned_features.index:
                    return False
    return True


def build_summary_markdown(
    coverage: pd.DataFrame,
    feature_summary: pd.DataFrame,
    alignment_checks: pd.DataFrame,
) -> str:
    """Build V7 validation summary."""
    row = coverage.iloc[0]
    summary = feature_summary.iloc[0]
    status = (
        "V7_real_macro_garch_current is eligible for protocol smoke."
        if (alignment_checks["status"] == "pass").all()
        else "V7_real_macro_garch_current is not eligible for protocol smoke."
    )
    return "\n".join(
        [
            "# V7 Real Macro + GARCH Current Validation",
            "",
            f"- Returns coverage: {row['returns_start']} to {row['returns_end']}",
            f"- Macro coverage: {row['macro_start']} to {row['macro_end']}",
            f"- Feature coverage: {row['feature_start']} to {row['feature_end']}",
            f"- Feature count: {int(summary['n_features'])}",
            f"- Macro feature count: {int(summary['n_macro_features'])}",
            f"- GARCH feature count: {int(summary['n_garch_features'])}",
            f"- GARCH backend: {row['garch_backend']}",
            f"- Assets fitted: {row['assets_fitted']}",
            f"- CASH handling: {row['cash_handling']}",
            f"- Fit successes: {int(row['fit_success_count'])}",
            f"- Fallback usage: {int(row['fallback_count'])}",
            f"- Fit failures: {int(row['fit_failure_count'])}",
            f"- Missing aligned features: {int(summary['missing_aligned_features'])}",
            f"- Status: {status}",
            "",
        ]
    )


def _check(check_name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "status": "pass" if passed else "fail",
        "detail": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate V7 current-window macro plus real GARCH features.",
    )
    parser.add_argument("--returns-path", default=DEFAULT_RETURNS_PATH)
    parser.add_argument("--macro-path", default=DEFAULT_MACRO_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = validate_v7_macro_garch_current(
        returns_path=args.returns_path,
        macro_path=args.macro_path,
        output_dir=args.output_dir,
    )
    print("V7 coverage:")
    print(result["coverage"].to_string(index=False))
    print("\nV7 feature summary:")
    print(result["feature_summary"].to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
