"""Validate current-window V8 EWMA plus real GARCH volatility features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.validate_v3_macro_current import (
    DEFAULT_RETURNS_PATH,
    PROTOCOL_FOLDS,
    load_returns_csv,
)
from src.data.features_v8 import build_features_v8


DEFAULT_OUTPUT_DIR = "outputs/tables/v8_ewma_garch_current_validation"
MODEL_SPEC = "zero-mean normal GARCH(1,1), one-step-ahead forecast"


def validate_v8_ewma_garch_current(
    returns_path: str = DEFAULT_RETURNS_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    ewma_lambda: float = 0.94,
    garch_mode: str = "rolling_fitted",
    garch_min_history: int = 104,
    garch_window: int | None = 156,
    garch_fallback: str = "rolling_realized_vol",
    exclude_cash: bool = True,
) -> dict[str, Any]:
    """Validate V8 feature coverage, diagnostics, and timing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    returns = load_returns_csv(returns_path)
    raw_features, diagnostics = build_features_v8(
        returns=returns,
        ewma_lambda=ewma_lambda,
        garch_mode=garch_mode,
        garch_min_history=garch_min_history,
        garch_window=garch_window,
        garch_exclude_cash=exclude_cash,
        garch_fallback=garch_fallback,
        return_diagnostics=True,
    )
    shifted = raw_features.shift(1).dropna()
    aligned_index = returns.index[returns.index.isin(shifted.index)]
    aligned_features = shifted.loc[aligned_index]

    coverage = build_coverage_table(
        returns=returns,
        features=raw_features,
        diagnostics=diagnostics,
        returns_path=returns_path,
        ewma_lambda=ewma_lambda,
    )
    feature_summary = build_feature_summary(raw_features, aligned_features)
    alignment_checks = build_alignment_checks(
        returns=returns,
        raw_features=raw_features,
        aligned_features=aligned_features,
        diagnostics=diagnostics,
        exclude_cash=exclude_cash,
    )
    validate_alignment_checks(alignment_checks)

    paths = {
        "coverage": output_path / "v8_ewma_garch_current_coverage.csv",
        "feature_summary": output_path / "v8_ewma_garch_current_feature_summary.csv",
        "alignment_checks": output_path / "v8_ewma_garch_current_alignment_checks.csv",
        "diagnostics": output_path / "v8_ewma_garch_current_diagnostics.csv",
        "summary": output_path / "v8_ewma_garch_current_summary.md",
    }
    coverage.to_csv(paths["coverage"], index=False)
    feature_summary.to_csv(paths["feature_summary"], index=False)
    alignment_checks.to_csv(paths["alignment_checks"], index=False)
    build_diagnostics_table(diagnostics).to_csv(paths["diagnostics"], index=False)
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
    features: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
    returns_path: str,
    ewma_lambda: float,
) -> pd.DataFrame:
    """Build compact V8 coverage table."""
    garch = diagnostics["garch"]
    ewma = diagnostics["ewma"]
    fitted_assets = sorted(garch["asset"].dropna().unique()) if not garch.empty else []
    return pd.DataFrame(
        [
            {
                "returns_path": returns_path,
                "returns_start": returns.index.min(),
                "returns_end": returns.index.max(),
                "feature_start": features.index.min(),
                "feature_end": features.index.max(),
                "returns_rows": len(returns),
                "feature_rows": len(features),
                "garch_mode": "rolling_fitted",
                "garch_backend": ",".join(sorted(garch["backend"].dropna().astype(str).unique())),
                "model_spec": MODEL_SPEC,
                "volatility_unit": "weekly",
                "ewma_lambda": ewma_lambda,
                "cash_handling": "excluded_from_fitted_volatility",
                "n_assets_fitted": len(fitted_assets),
                "assets_fitted": ",".join(fitted_assets),
                "fit_success_count": int((garch["status"] == "fitted").sum()),
                "fallback_count": int((garch["status"] == "fallback").sum()),
                "fit_failure_count": int(
                    garch["fallback_reason"].fillna("").astype(str).str.contains("fit_failed").sum()
                ),
                "ewma_assets": ",".join(sorted(ewma["asset"].dropna().astype(str).unique())),
            }
        ]
    )


def build_feature_summary(
    raw_features: pd.DataFrame,
    aligned_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature-count summary."""
    garch_columns = [column for column in raw_features.columns if str(column).startswith("garch_")]
    ewma_columns = [column for column in raw_features.columns if str(column).startswith("ewma_vol_")]
    comparison_columns = [
        column
        for column in raw_features.columns
        if str(column).startswith("garch_minus_ewma_vol_")
        or str(column).startswith("garch_to_ewma_vol_ratio_")
    ]
    cash_vol_columns = [
        column
        for column in garch_columns + ewma_columns + comparison_columns
        if "CASH" in str(column)
    ]
    ratio_columns = [
        column
        for column in raw_features.columns
        if str(column).startswith("garch_to_ewma_vol_ratio_")
    ]
    ratio_values = raw_features[ratio_columns].to_numpy(dtype=float) if ratio_columns else np.array([])
    return pd.DataFrame(
        [
            {
                "raw_feature_rows": len(raw_features),
                "aligned_feature_rows": len(aligned_features),
                "n_features": raw_features.shape[1],
                "n_garch_features": len(garch_columns),
                "n_ewma_features": len(ewma_columns),
                "n_comparison_features": len(comparison_columns),
                "cash_volatility_column_count": len(cash_vol_columns),
                "cash_volatility_columns": ",".join(cash_vol_columns),
                "missing_raw_features": int(raw_features.isna().sum().sum()),
                "missing_aligned_features": int(aligned_features.isna().sum().sum()),
                "nonfinite_ratio_count": int((~np.isfinite(ratio_values)).sum()) if ratio_columns else 0,
                "first_aligned_date": aligned_features.index.min(),
                "last_aligned_date": aligned_features.index.max(),
            }
        ]
    )


def build_diagnostics_table(diagnostics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine GARCH and EWMA diagnostics with a source marker."""
    garch = diagnostics["garch"].copy()
    ewma = diagnostics["ewma"].copy()
    garch["diagnostic_source"] = "garch"
    ewma["diagnostic_source"] = "ewma"
    return pd.concat([garch, ewma], ignore_index=True, sort=False)


def build_alignment_checks(
    returns: pd.DataFrame,
    raw_features: pd.DataFrame,
    aligned_features: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
    exclude_cash: bool,
) -> pd.DataFrame:
    """Build pass/fail alignment and leakage guard checks."""
    garch = diagnostics["garch"]
    fitted = garch[garch["status"] == "fitted"]
    garch_columns = [column for column in raw_features.columns if str(column).startswith("garch_")]
    ewma_columns = [column for column in raw_features.columns if str(column).startswith("ewma_vol_")]
    comparison_columns = [
        column
        for column in raw_features.columns
        if str(column).startswith("garch_minus_ewma_vol_")
        or str(column).startswith("garch_to_ewma_vol_ratio_")
    ]
    ratio_columns = [
        column
        for column in raw_features.columns
        if str(column).startswith("garch_to_ewma_vol_ratio_")
    ]
    shock_check = synthetic_shock_timing_check()
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
        _check("garch_columns_present", bool(garch_columns), f"n={len(garch_columns)}"),
        _check("ewma_columns_present", bool(ewma_columns), f"n={len(ewma_columns)}"),
        _check(
            "comparison_columns_present",
            bool(comparison_columns),
            f"n={len(comparison_columns)}",
        ),
        _check(
            "cash_volatility_excluded",
            (not exclude_cash)
            or not any("CASH" in str(column) for column in garch_columns + ewma_columns + comparison_columns),
            "CASH must not receive normal fitted GARCH/EWMA columns.",
        ),
        _check(
            "garch_uses_arch_model",
            (not fitted.empty) and set(fitted["backend"].dropna()) == {"arch_model"},
            f"fitted_backends={sorted(fitted['backend'].dropna().astype(str).unique())}",
        ),
        _check(
            "garch_fit_failures_absent",
            not garch["fallback_reason"].fillna("").astype(str).str.contains("fit_failed").any(),
            "fallback reasons should not include fit_failed.",
        ),
        _check(
            "ratios_are_finite",
            bool(ratio_columns)
            and np.isfinite(raw_features[ratio_columns].to_numpy(dtype=float)).all(),
            f"ratio_columns={len(ratio_columns)}",
        ),
        _check(
            "synthetic_shock_no_same_period_leakage",
            shock_check["same_period_unchanged"],
            shock_check["details"],
        ),
        _check(
            "synthetic_shock_affects_future_forecasts",
            shock_check["future_changed"],
            shock_check["details"],
        ),
        _check(
            "validation_test_windows_align",
            protocol_windows_align(aligned_features),
            "Protocol validation/test windows have V8 feature coverage where possible.",
        ),
        _check(
            "v8_differs_from_v4_feature_set",
            bool(ewma_columns and comparison_columns),
            "V8 includes EWMA and GARCH/EWMA comparison columns absent from V4.",
        ),
    ]
    return pd.DataFrame(rows)


def synthetic_shock_timing_check() -> dict[str, Any]:
    """Check that a return shock at t does not alter V8 forecasts assigned to t."""
    returns = _synthetic_returns()
    shocked = returns.copy()
    shock_date = returns.index[20]
    future_date = returns.index[21]
    shocked.loc[shock_date, "SPY"] = 0.50
    base_features = build_features_v8(
        returns,
        garch_min_history=8,
        garch_window=12,
    )
    shocked_features = build_features_v8(
        shocked,
        garch_min_history=8,
        garch_window=12,
    )
    check_columns = [
        column
        for column in base_features.columns
        if column in shocked_features.columns
        and (
            column == "garch_vol_SPY"
            or column == "ewma_vol_SPY"
            or column == "garch_minus_ewma_vol_SPY"
            or column == "garch_to_ewma_vol_ratio_SPY"
        )
    ]
    same_period_unchanged = np.allclose(
        base_features.loc[shock_date, check_columns],
        shocked_features.loc[shock_date, check_columns],
        rtol=1e-10,
        atol=1e-10,
    )
    future_changed = not np.allclose(
        base_features.loc[future_date, check_columns],
        shocked_features.loc[future_date, check_columns],
        rtol=1e-10,
        atol=1e-10,
    )
    return {
        "same_period_unchanged": bool(same_period_unchanged),
        "future_changed": bool(future_changed),
        "details": f"shock_date={shock_date}, future_date={future_date}, columns={check_columns}",
    }


def validate_alignment_checks(alignment_checks: pd.DataFrame) -> None:
    """Raise when any alignment check fails."""
    failed = alignment_checks[alignment_checks["status"] != "pass"]
    if not failed.empty:
        raise ValueError(
            "V8 EWMA/GARCH validation failed: "
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
    """Build V8 validation summary."""
    coverage_row = coverage.iloc[0]
    feature_row = feature_summary.iloc[0]
    status = (
        "V8_ewma_garch_vol_current eligible for protocol smoke"
        if (alignment_checks["status"] == "pass").all()
        else "V8_ewma_garch_vol_current not eligible"
    )
    return "\n".join(
        [
            "# V8 EWMA/GARCH Current Validation",
            "",
            f"- Status: {status}",
            f"- Returns coverage: {coverage_row['returns_start']} to {coverage_row['returns_end']}",
            f"- Feature coverage: {coverage_row['feature_start']} to {coverage_row['feature_end']}",
            f"- GARCH backend: {coverage_row['garch_backend']}",
            f"- GARCH model: {coverage_row['model_spec']}",
            f"- EWMA lambda: {coverage_row['ewma_lambda']}",
            f"- Volatility unit: {coverage_row['volatility_unit']}",
            f"- CASH handling: {coverage_row['cash_handling']}",
            f"- Feature count: {feature_row['n_features']}",
            f"- GARCH feature count: {feature_row['n_garch_features']}",
            f"- EWMA feature count: {feature_row['n_ewma_features']}",
            f"- Comparison feature count: {feature_row['n_comparison_features']}",
            f"- Fit success count: {coverage_row['fit_success_count']}",
            f"- Fallback count: {coverage_row['fallback_count']}",
            f"- Fit failure count: {coverage_row['fit_failure_count']}",
            "",
            "## Alignment Checks",
            *[
                f"- {row['check_name']}: {row['status']} ({row['details']})"
                for _, row in alignment_checks.iterrows()
            ],
            "",
        ]
    )


def _synthetic_returns() -> pd.DataFrame:
    index = pd.date_range("2022-01-07", periods=36, freq="W-FRI")
    return pd.DataFrame(
        {
            "SPY": ([0.010, 0.012, -0.008, 0.006, -0.004, 0.009] * 6)[: len(index)],
            "TLT": ([0.002, -0.001, 0.003, 0.001, -0.002, 0.004] * 6)[: len(index)],
            "GLD": ([0.004, 0.001, -0.003, 0.002, 0.006, -0.001] * 6)[: len(index)],
            "BTC-USD": ([0.030, -0.025, 0.020, -0.015, 0.035, -0.020] * 6)[: len(index)],
            "CASH": [0.0] * len(index),
        },
        index=index,
    )


def _check(check_name: str, passed: bool, details: str) -> dict[str, str]:
    return {
        "check_name": check_name,
        "status": "pass" if passed else "fail",
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate V8 EWMA/GARCH features.")
    parser.add_argument("--returns-path", default=DEFAULT_RETURNS_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ewma-lambda", type=float, default=0.94)
    parser.add_argument("--garch-mode", default="rolling_fitted")
    parser.add_argument("--garch-min-history", type=int, default=104)
    parser.add_argument("--garch-window", type=int, default=156)
    parser.add_argument("--garch-fallback", default="rolling_realized_vol")
    parser.add_argument("--exclude-cash", action="store_true")
    args = parser.parse_args()
    result = validate_v8_ewma_garch_current(
        returns_path=args.returns_path,
        output_dir=args.output_dir,
        ewma_lambda=args.ewma_lambda,
        garch_mode=args.garch_mode,
        garch_min_history=args.garch_min_history,
        garch_window=args.garch_window,
        garch_fallback=args.garch_fallback,
        exclude_cash=args.exclude_cash,
    )
    print(result["summary"])


if __name__ == "__main__":
    main()
