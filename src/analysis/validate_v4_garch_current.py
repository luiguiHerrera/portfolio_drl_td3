"""Validate current-window V4 rolling-fitted GARCH features.

This module is a reporting gate for V4 real GARCH features. It does not train
TD3 and does not change reward, environment, or scoring logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.validate_v3_macro_current import PROTOCOL_FOLDS, load_returns_csv
from src.data.features_v4 import build_features_v4
from src.data.garch_features import (
    GARCH_FALLBACK_ROLLING_REALIZED,
    GARCH_MODE_DETERMINISTIC,
    GARCH_MODE_ROLLING_FITTED,
    build_garch_feature_set_by_mode,
)


DEFAULT_RETURNS_PATH = "data/processed/returns_weekly_latest.csv"
DEFAULT_OUTPUT_DIR = "outputs/tables/v4_garch_current_validation"
MODEL_SPEC = "zero-mean normal GARCH(1,1), one-step-ahead forecast"


def validate_v4_garch_current(
    returns_path: str = DEFAULT_RETURNS_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    garch_mode: str = GARCH_MODE_ROLLING_FITTED,
    min_history: int = 104,
    window: int | None = 156,
    fallback: str = GARCH_FALLBACK_ROLLING_REALIZED,
    exclude_cash: bool = True,
) -> dict[str, Any]:
    """Validate V4 rolling-fitted GARCH feature coverage and timing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    returns = load_returns_csv(returns_path)

    garch_result = build_garch_feature_set_by_mode(
        returns=returns,
        assets=list(returns.columns),
        market_asset="SPY",
        include_relative=True,
        mode=garch_mode,
        min_history=min_history,
        window=window,
        annualize=False,
        exclude_cash=exclude_cash,
        fallback=fallback,
        return_diagnostics=True,
    )
    garch_features, diagnostics = garch_result
    raw_features = build_features_v4(
        returns,
        garch_mode=garch_mode,
        garch_min_history=min_history,
        garch_window=window,
        garch_annualize=False,
        garch_exclude_cash=exclude_cash,
        garch_fallback=fallback,
    )
    shifted_features = raw_features.shift(1).dropna()
    aligned_index = returns.index[returns.index.isin(shifted_features.index)]
    aligned_features = shifted_features.loc[aligned_index]
    deterministic = build_features_v4(
        returns,
        garch_mode=GARCH_MODE_DETERMINISTIC,
    )

    coverage = build_coverage_table(
        returns=returns,
        features=raw_features,
        diagnostics=diagnostics,
        returns_path=returns_path,
        garch_mode=garch_mode,
        min_history=min_history,
        window=window,
        exclude_cash=exclude_cash,
    )
    feature_summary = build_feature_summary(
        returns=returns,
        garch_features=garch_features,
        raw_features=raw_features,
        aligned_features=aligned_features,
        deterministic_features=deterministic,
    )
    alignment_checks = build_alignment_checks(
        returns=returns,
        raw_features=raw_features,
        aligned_features=aligned_features,
        deterministic_features=deterministic,
        min_history=min_history,
        window=window,
        exclude_cash=exclude_cash,
    )
    fit_diagnostics = build_fit_diagnostics(diagnostics)
    validate_alignment_checks(alignment_checks)

    paths = {
        "coverage": output_path / "v4_garch_current_coverage.csv",
        "feature_summary": output_path / "v4_garch_current_feature_summary.csv",
        "alignment_checks": output_path / "v4_garch_current_alignment_checks.csv",
        "fit_diagnostics": output_path / "v4_garch_current_fit_diagnostics.csv",
        "summary": output_path / "v4_garch_current_summary.md",
    }
    coverage.to_csv(paths["coverage"], index=False)
    feature_summary.to_csv(paths["feature_summary"], index=False)
    alignment_checks.to_csv(paths["alignment_checks"], index=False)
    fit_diagnostics.to_csv(paths["fit_diagnostics"], index=False)
    summary = build_summary_markdown(
        coverage,
        feature_summary,
        alignment_checks,
        fit_diagnostics,
    )
    paths["summary"].write_text(summary, encoding="utf-8")

    return {
        "coverage": coverage,
        "feature_summary": feature_summary,
        "alignment_checks": alignment_checks,
        "fit_diagnostics": fit_diagnostics,
        "summary": summary,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def build_coverage_table(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    diagnostics: pd.DataFrame,
    returns_path: str,
    garch_mode: str,
    min_history: int,
    window: int | None,
    exclude_cash: bool,
) -> pd.DataFrame:
    """Build compact coverage table."""
    fitted_assets = sorted(diagnostics["asset"].dropna().unique()) if not diagnostics.empty else []
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
                "garch_mode": garch_mode,
                "model_spec": MODEL_SPEC,
                "volatility_unit": "weekly",
                "min_history": min_history,
                "window": window,
                "exclude_cash": exclude_cash,
                "cash_handling": "excluded" if exclude_cash else "included",
                "n_assets_fitted": len(fitted_assets),
                "assets_fitted": ",".join(fitted_assets),
                "feature_dates_within_returns": bool(
                    features.index.min() >= returns.index.min()
                    and features.index.max() <= returns.index.max()
                ),
            }
        ]
    )


def build_feature_summary(
    returns: pd.DataFrame,
    garch_features: pd.DataFrame,
    raw_features: pd.DataFrame,
    aligned_features: pd.DataFrame,
    deterministic_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature summary table."""
    garch_columns = [column for column in raw_features.columns if column.startswith("garch_")]
    cash_garch_columns = [column for column in garch_columns if column.endswith("_CASH")]
    common = sorted(set(raw_features.columns).intersection(deterministic_features.columns))
    max_difference = 0.0
    if common:
        joined = raw_features[common].align(deterministic_features[common], join="inner")[0]
        other = deterministic_features.loc[joined.index, common]
        max_difference = float((joined - other).abs().max().max())
    return pd.DataFrame(
        [
            {
                "returns_rows": len(returns),
                "raw_feature_rows": len(raw_features),
                "aligned_feature_rows": len(aligned_features),
                "n_features": raw_features.shape[1],
                "n_garch_features": len(garch_columns),
                "garch_feature_rows": len(garch_features),
                "missing_raw_features": int(raw_features.isna().sum().sum()),
                "missing_aligned_features": int(aligned_features.isna().sum().sum()),
                "cash_garch_columns": ",".join(cash_garch_columns),
                "cash_garch_column_count": len(cash_garch_columns),
                "real_garch_differs_from_deterministic": bool(max_difference > 1e-12),
                "max_abs_difference_vs_deterministic": max_difference,
                "first_aligned_date": aligned_features.index.min(),
                "last_aligned_date": aligned_features.index.max(),
            }
        ]
    )


def build_alignment_checks(
    returns: pd.DataFrame,
    raw_features: pd.DataFrame,
    aligned_features: pd.DataFrame,
    deterministic_features: pd.DataFrame,
    min_history: int,
    window: int | None,
    exclude_cash: bool,
) -> pd.DataFrame:
    """Build timing/leakage checks."""
    shock_checks = synthetic_shock_timing_checks(
        min_history=min(8, min_history),
        window=min(12, window) if window is not None else None,
        exclude_cash=exclude_cash,
    )
    common = sorted(set(raw_features.columns).intersection(deterministic_features.columns))
    max_difference = 0.0
    if common:
        common_index = raw_features.index.intersection(deterministic_features.index)
        max_difference = float(
            (
                raw_features.loc[common_index, common]
                - deterministic_features.loc[common_index, common]
            )
            .abs()
            .max()
            .max()
        )
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
            "validation_test_windows_align",
            protocol_windows_align(aligned_features),
            "Protocol validation/test windows have feature coverage where possible.",
        ),
        _check(
            "synthetic_shock_no_same_period_leakage",
            shock_checks["same_period_unchanged"],
            shock_checks["same_period_details"],
        ),
        _check(
            "synthetic_shock_changes_future_forecast",
            shock_checks["future_changed"],
            shock_checks["future_details"],
        ),
        _check(
            "cash_excluded_or_explicit",
            (not exclude_cash) or not any(column.endswith("_CASH") for column in raw_features.columns),
            f"exclude_cash={exclude_cash}",
        ),
        _check(
            "real_fitted_differs_from_deterministic_filter",
            bool(max_difference > 1e-12),
            f"max_abs_difference={max_difference}",
        ),
    ]
    return pd.DataFrame(rows)


def build_fit_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-date/per-asset fit diagnostics."""
    if diagnostics.empty:
        return pd.DataFrame(
            [
                {
                    "n_rows": 0,
                    "fit_success_count": 0,
                    "fallback_count": 0,
                    "fit_failure_count": 0,
                    "fallback_reasons": "",
                    "arch_available": False,
                    "scipy_available": False,
                }
            ]
        )
    fallback = diagnostics[diagnostics["status"] == "fallback"]
    fit_failure_count = int(
        fallback["fallback_reason"].astype(str).str.startswith("fit_failed").sum()
    )
    return pd.DataFrame(
        [
            {
                "n_rows": len(diagnostics),
                "fit_success_count": int((diagnostics["status"] == "fitted").sum()),
                "fallback_count": len(fallback),
                "fit_failure_count": fit_failure_count,
                "fallback_reasons": ",".join(
                    sorted(reason for reason in fallback["fallback_reason"].dropna().astype(str).unique() if reason)
                ),
                "arch_available": bool(diagnostics["arch_available"].any()),
                "scipy_available": bool(diagnostics["scipy_available"].any()),
                "backend": ",".join(sorted(diagnostics["backend"].dropna().astype(str).unique())),
                "volatility_unit": ",".join(
                    sorted(diagnostics["volatility_unit"].dropna().astype(str).unique())
                ),
            }
        ]
    )


def synthetic_shock_timing_checks(
    min_history: int = 8,
    window: int | None = 12,
    exclude_cash: bool = True,
) -> dict[str, Any]:
    """Check that a shock at t changes t+1 forecasts, not t forecasts."""
    index = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    base = pd.DataFrame(
        {
            "SPY": [0.001] * len(index),
            "TLT": [0.0005] * len(index),
            "GLD": [0.0008] * len(index),
            "CASH": [0.0] * len(index),
        },
        index=index,
    )
    base["SPY"] = [0.002 if i % 2 == 0 else -0.001 for i in range(len(index))]
    shocked = base.copy()
    shock_position = min_history + 5
    shocked.iloc[shock_position, shocked.columns.get_loc("SPY")] = 0.20
    base_features = build_garch_feature_set_by_mode(
        base,
        mode=GARCH_MODE_ROLLING_FITTED,
        include_relative=False,
        min_history=min_history,
        window=window,
        exclude_cash=exclude_cash,
        return_diagnostics=False,
    )
    shocked_features = build_garch_feature_set_by_mode(
        shocked,
        mode=GARCH_MODE_ROLLING_FITTED,
        include_relative=False,
        min_history=min_history,
        window=window,
        exclude_cash=exclude_cash,
        return_diagnostics=False,
    )
    shock_date = index[shock_position]
    next_date = index[shock_position + 1]
    column = "garch_vol_SPY"
    same_delta = abs(float(shocked_features.loc[shock_date, column] - base_features.loc[shock_date, column]))
    future_delta = abs(float(shocked_features.loc[next_date, column] - base_features.loc[next_date, column]))
    return {
        "same_period_unchanged": bool(same_delta <= 1e-12),
        "future_changed": bool(future_delta > 1e-10),
        "same_period_details": f"shock_date={shock_date}, same_period_delta={same_delta}",
        "future_details": f"next_date={next_date}, future_delta={future_delta}",
    }


def protocol_windows_align(aligned_features: pd.DataFrame) -> bool:
    """Return True when protocol validation/test starts are covered where possible."""
    if aligned_features.empty:
        return False
    starts = []
    for fold in PROTOCOL_FOLDS:
        starts.extend([pd.Timestamp(fold["validation_start"]), pd.Timestamp(fold["test_start"])])
    relevant = [
        start
        for start in starts
        if aligned_features.index.min() <= start <= aligned_features.index.max()
    ]
    if not relevant:
        return True
    covered = [start for start in relevant if start in aligned_features.index]
    return len(covered) == len(relevant)


def validate_alignment_checks(alignment_checks: pd.DataFrame) -> None:
    """Raise if any validation check fails."""
    failed = alignment_checks[~alignment_checks["passed"]]
    if not failed.empty:
        details = "; ".join(failed["check"].astype(str))
        raise ValueError(f"V4 GARCH validation checks failed: {details}")


def build_summary_markdown(
    coverage: pd.DataFrame,
    feature_summary: pd.DataFrame,
    alignment_checks: pd.DataFrame,
    fit_diagnostics: pd.DataFrame,
) -> str:
    """Build Markdown summary."""
    c = coverage.iloc[0]
    f = feature_summary.iloc[0]
    d = fit_diagnostics.iloc[0]
    failed = alignment_checks[~alignment_checks["passed"]]
    eligibility = "eligible for protocol smoke" if failed.empty else "not eligible yet"
    return "\n".join(
        [
            "# V4 Real GARCH Current Validation",
            "",
            f"- Mode: `{c['garch_mode']}`",
            f"- Model: {MODEL_SPEC}",
            f"- Volatility unit: `{c['volatility_unit']}`",
            f"- Returns coverage: {c['returns_start']} to {c['returns_end']}",
            f"- Feature coverage: {c['feature_start']} to {c['feature_end']}",
            f"- Assets fitted: {c['assets_fitted']}",
            f"- CASH handling: {c['cash_handling']}",
            f"- Fit successes: {d['fit_success_count']}",
            f"- Fallback usage: {d['fallback_count']}",
            f"- Fit failures: {d['fit_failure_count']}",
            f"- Fallback reasons: {d['fallback_reasons']}",
            f"- Missing aligned features: {f['missing_aligned_features']}",
            f"- Real fitted differs from deterministic filter: {f['real_garch_differs_from_deterministic']}",
            f"- Status: V4_real_garch_current is {eligibility}.",
            "",
        ]
    )


def _check(check: str, passed: bool, details: str) -> dict[str, Any]:
    return {"check": check, "passed": bool(passed), "details": details}


def _parse_window(value: str) -> int | None:
    if str(value).lower() in {"none", "expanding"}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returns-path", default=DEFAULT_RETURNS_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--garch-mode", default=GARCH_MODE_ROLLING_FITTED)
    parser.add_argument("--min-history", type=int, default=104)
    parser.add_argument("--window", default="156")
    parser.add_argument("--fallback", default=GARCH_FALLBACK_ROLLING_REALIZED)
    parser.add_argument("--exclude-cash", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_v4_garch_current(
        returns_path=args.returns_path,
        output_dir=args.output_dir,
        garch_mode=args.garch_mode,
        min_history=args.min_history,
        window=_parse_window(args.window),
        fallback=args.fallback,
        exclude_cash=args.exclude_cash,
    )
    print("V4 GARCH coverage:")
    print(result["coverage"].to_string(index=False))
    print("\nV4 GARCH fit diagnostics:")
    print(result["fit_diagnostics"].to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
