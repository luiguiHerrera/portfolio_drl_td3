"""Compare final strategies under reusable mandate profile limits.

This is a reporting-only layer. It reads existing final constrained TD3
metrics, applies the diagnostic mandate profiles from ``src.risk``, and writes
profile-specific rankings. It does not retrain models and does not replace the
production robust_score or mandate_aware_score calculations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.risk.mandate_profiles import MandateLimits, get_default_mandate_profiles


DEFAULT_FINAL_REPORT_DIR = (
    "outputs/tables/"
    "final_constrained_td3_report_with_v3_clean_no_dxy_v7_clean_garch_v4_v7_v8_60ep_10seeds"
)
DEFAULT_OUTPUT_DIR = "outputs/tables/mandate_profile_comparison_final"
DEFAULT_ASSET_SPECIFIC_COMBINED_REPORT_DIR = (
    "outputs/tables/asset_specific_cost_benchmark_comparison"
)
ASSET_SPECIFIC_COMBINED_RANKING_FILE = "asset_specific_cost_combined_ranking.csv"
ASSET_SPECIFIC_COMBINED_METADATA_FILE = (
    "asset_specific_cost_benchmark_comparison_metadata.json"
)

PROFILE_ORDER = ["conservative", "moderate", "aggressive"]
KEY_CANDIDATE_V3_CLEAN = "V3_real_macro_vintage_clean_no_dxy_cap_0.50"
KEY_CANDIDATE_V7_CLEAN_GARCH = "V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50"
KEY_CANDIDATE_V5_ASSET_SPECIFIC = "V5_no_volatility_block_cap_0p50"
KEY_CANDIDATE_V3_CLEAN_ASSET_SPECIFIC = (
    "V3_real_macro_vintage_clean_no_dxy_cap_0p70"
)


def build_mandate_profile_comparison_report(
    final_report_dir: str = DEFAULT_FINAL_REPORT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    combined_report_dir: str | None = None,
    benchmark_dir: str | None = None,
    asset_specific_only: bool = False,
) -> dict[str, Any]:
    """Build mandate profile comparison outputs from an existing final report."""
    final_dir = Path(final_report_dir)
    combined_dir = Path(combined_report_dir) if combined_report_dir else None
    benchmark_path = Path(benchmark_dir) if benchmark_dir else None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    strategies, source_metadata = load_strategy_metrics(
        final_dir,
        combined_report_dir=combined_dir,
        benchmark_dir=benchmark_path,
        asset_specific_only=asset_specific_only,
    )
    profiles = {
        name: limits
        for name, limits in get_default_mandate_profiles().items()
        if name in PROFILE_ORDER
    }
    scores = score_strategies_for_profiles(strategies, profiles)
    rankings = build_profile_rankings(scores)
    winners = build_profile_winners(rankings)

    scores_path = output_path / "mandate_profile_strategy_scores.csv"
    winners_path = output_path / "mandate_profile_winners.csv"
    rankings_path = output_path / "mandate_profile_rankings.csv"
    summary_path = output_path / "mandate_profile_summary.md"
    metadata_path = output_path / "mandate_profile_metadata.json"

    scores.to_csv(scores_path, index=False)
    winners.to_csv(winners_path, index=False)
    rankings.to_csv(rankings_path, index=False)

    summary = build_summary_markdown(winners, rankings)
    summary_path.write_text(summary, encoding="utf-8")

    metadata = {
        "final_report_dir": str(final_dir),
        "combined_report_dir": str(combined_dir) if combined_dir else None,
        "benchmark_dir": str(benchmark_path) if benchmark_path else None,
        "output_dir": str(output_path),
        "reporting_only": True,
        "does_not_retrain": True,
        "does_not_replace_main_result": True,
        "asset_specific_only": asset_specific_only,
        "source_metadata": source_metadata,
        "profile_thresholds": {
            name: limits.to_dict()
            for name, limits in profiles.items()
        },
        "scoring_note": (
            "Profile scores multiply robust_score by reporting-only penalties for "
            "violating available profile limits. Hard eligibility is reported "
            "separately and is not used to alter production scores."
        ),
    }
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")

    return {
        "strategy_scores": scores,
        "rankings": rankings,
        "winners": winners,
        "summary": summary,
        "metadata": metadata,
        "paths": {
            "scores": str(scores_path),
            "winners": str(winners_path),
            "rankings": str(rankings_path),
            "summary": str(summary_path),
            "metadata": str(metadata_path),
        },
    }


def load_final_strategy_metrics(final_report_dir: Path) -> pd.DataFrame:
    """Load the final all-strategy ranking table."""
    main_path = final_report_dir / "final_constrained_td3_main_ranking.csv"
    if not main_path.exists():
        raise FileNotFoundError(f"Missing final ranking table: {main_path}")

    data = pd.read_csv(main_path)
    if "strategy_name" not in data.columns:
        raise ValueError("final_constrained_td3_main_ranking.csv must include strategy_name.")
    if "robust_score" not in data.columns:
        raise ValueError("final_constrained_td3_main_ranking.csv must include robust_score.")
    return data


def load_strategy_metrics(
    final_report_dir: Path,
    combined_report_dir: Path | None = None,
    benchmark_dir: Path | None = None,
    asset_specific_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load legacy final ranking or asset-specific combined TD3+benchmark ranking."""
    if combined_report_dir is not None:
        ranking_path = combined_report_dir / ASSET_SPECIFIC_COMBINED_RANKING_FILE
        metadata_path = combined_report_dir / ASSET_SPECIFIC_COMBINED_METADATA_FILE
        if not ranking_path.exists():
            raise FileNotFoundError(f"Missing asset-specific combined ranking: {ranking_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing asset-specific combined metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if asset_specific_only:
            validate_asset_specific_sources(final_report_dir, metadata, benchmark_dir)
        data = pd.read_csv(ranking_path)
        if data.empty:
            raise ValueError("asset-specific combined ranking must not be empty.")
        if "transaction_cost_mode" not in data.columns:
            raise ValueError("asset-specific combined ranking must include transaction_cost_mode.")
        modes = set(data["transaction_cost_mode"].dropna().astype(str).unique().tolist())
        if asset_specific_only and modes != {"asset_specific"}:
            raise ValueError(f"Non asset-specific strategies found in combined ranking: {sorted(modes)}")
        if "robust_score" not in data.columns:
            raise ValueError("asset-specific combined ranking must include robust_score.")
        return data, {
            "input_mode": "asset_specific_combined",
            "combined_ranking": str(ranking_path),
            "combined_metadata": str(metadata_path),
            "combined_score_scope": metadata.get("combined_score_scope"),
            "n_strategies": int(len(data)),
            "transaction_cost_modes": sorted(modes),
        }

    data = load_final_strategy_metrics(final_report_dir)
    return data, {
        "input_mode": "legacy_final_report",
        "n_strategies": int(len(data)),
    }


def validate_asset_specific_sources(
    final_report_dir: Path,
    combined_metadata: dict[str, Any],
    benchmark_dir: Path | None,
) -> None:
    """Fail loudly if cost metadata or benchmark histories are not asset-specific."""
    td3_metadata_path = final_report_dir / "asset_specific_cost_metadata.json"
    if not td3_metadata_path.exists():
        raise FileNotFoundError(f"Missing TD3 asset-specific metadata: {td3_metadata_path}")
    td3_metadata = json.loads(td3_metadata_path.read_text(encoding="utf-8"))
    cost_model = td3_metadata.get("cost_model", {})
    if cost_model.get("transaction_cost_mode") != "asset_specific":
        raise ValueError("TD3 report is not asset-specific cost aware.")
    if combined_metadata.get("cost_model", {}).get("transaction_cost_mode") != "asset_specific":
        raise ValueError("Combined report is not asset-specific cost aware.")
    if benchmark_dir is not None:
        histories_dir = benchmark_dir / "histories"
    else:
        benchmark_dir_value = combined_metadata.get("benchmark_dir")
        histories_dir = (
            Path(benchmark_dir_value) / "histories"
            if benchmark_dir_value
            else None
        )
    if histories_dir is None or not histories_dir.exists():
        raise FileNotFoundError(f"Missing benchmark histories directory: {histories_dir}")
    history_paths = sorted(histories_dir.glob("*_history.csv"))
    if not history_paths:
        raise FileNotFoundError(f"No benchmark history files found in {histories_dir}")
    for path in history_paths:
        history = pd.read_csv(path, nrows=5)
        if "transaction_cost_mode" not in history.columns:
            raise ValueError(f"Benchmark history missing transaction_cost_mode: {path}")
        modes = set(history["transaction_cost_mode"].dropna().astype(str).unique().tolist())
        if modes != {"asset_specific"}:
            raise ValueError(f"Scalar/non asset-specific benchmark history detected in {path}: {sorted(modes)}")


def score_strategies_for_profiles(
    strategies: pd.DataFrame,
    profiles: dict[str, MandateLimits] | None = None,
) -> pd.DataFrame:
    """Return one scored row per strategy and profile."""
    selected_profiles = get_default_mandate_profiles() if profiles is None else profiles
    rows: list[dict[str, Any]] = []

    for profile_name in PROFILE_ORDER:
        limits = selected_profiles[profile_name]
        for _, strategy in strategies.iterrows():
            rows.append(score_strategy_for_profile(strategy, profile_name, limits))

    return pd.DataFrame(rows)


def score_strategy_for_profile(
    strategy: pd.Series,
    profile_name: str,
    limits: MandateLimits,
) -> dict[str, Any]:
    """Score one strategy under one mandate profile."""
    robust_score = _numeric(strategy.get("robust_score"), default=0.0)
    drawdown_result = _score_lower_bound(
        strategy.get("max_drawdown"),
        limits.max_drawdown_limit,
        higher_is_better=True,
        negative_drawdown=True,
    )
    volatility_result = _score_upper_bound(
        strategy.get("annualized_volatility"),
        limits.max_volatility_limit,
    )
    max_weight_result = _score_upper_bound(
        strategy.get("average_max_weight"),
        limits.max_weight_limit,
    )
    effective_assets_result = _score_lower_bound(
        strategy.get("average_effective_number_of_assets"),
        limits.min_effective_assets,
        higher_is_better=True,
    )
    turnover_result = _score_upper_bound(
        strategy.get("average_turnover"),
        limits.max_turnover_limit,
    )

    checks = {
        "drawdown": drawdown_result,
        "volatility": volatility_result,
        "max_weight": max_weight_result,
        "effective_assets": effective_assets_result,
        "turnover": turnover_result,
    }
    available_checks = [check for check in checks.values() if check["available"]]
    eligible = bool(available_checks) and all(check["passes"] for check in available_checks)
    multiplier = float(np.prod([check["multiplier"] for check in available_checks])) if available_checks else 1.0
    profile_score = robust_score * multiplier

    failed_constraints = [
        name
        for name, check in checks.items()
        if check["available"] and not check["passes"]
    ]

    return {
        "profile": profile_name,
        "strategy_name": strategy.get("strategy_name"),
        "strategy_type": strategy.get("strategy_type"),
        "strategy_group": strategy.get("strategy_group"),
        "base_candidate": strategy.get("base_candidate"),
        "feature_family": strategy.get("feature_family"),
        "constraint_status": strategy.get("constraint_status"),
        "robust_score": robust_score,
        "original_mandate_aware_score": _numeric(strategy.get("mandate_aware_score"), default=np.nan),
        "profile_score": profile_score,
        "profile_multiplier": multiplier,
        "profile_eligible": eligible,
        "failed_constraints": ";".join(failed_constraints),
        "annualized_return": _numeric(strategy.get("annualized_return"), default=np.nan),
        "annualized_volatility": _numeric(strategy.get("annualized_volatility"), default=np.nan),
        "sharpe": _numeric(strategy.get("sharpe"), default=np.nan),
        "max_drawdown": _numeric(strategy.get("max_drawdown"), default=np.nan),
        "average_turnover": _numeric(strategy.get("average_turnover"), default=np.nan),
        "average_effective_number_of_assets": _numeric(
            strategy.get("average_effective_number_of_assets"),
            default=np.nan,
        ),
        "average_max_weight": _numeric(strategy.get("average_max_weight"), default=np.nan),
        "max_drawdown_limit": limits.max_drawdown_limit,
        "max_volatility_limit": limits.max_volatility_limit,
        "max_weight_limit": limits.max_weight_limit,
        "min_effective_assets": limits.min_effective_assets,
        "max_turnover_limit": limits.max_turnover_limit,
        "drawdown_pass": drawdown_result["passes"],
        "volatility_pass": volatility_result["passes"],
        "max_weight_pass": max_weight_result["passes"],
        "effective_assets_pass": effective_assets_result["passes"],
        "turnover_pass": turnover_result["passes"],
        "drawdown_multiplier": drawdown_result["multiplier"],
        "volatility_multiplier": volatility_result["multiplier"],
        "max_weight_multiplier": max_weight_result["multiplier"],
        "effective_assets_multiplier": effective_assets_result["multiplier"],
        "turnover_multiplier": turnover_result["multiplier"],
    }


def build_profile_rankings(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank strategies within each mandate profile."""
    ranked = scores.copy()
    ranked["profile_rank"] = (
        ranked.groupby("profile")["profile_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    ranked["profile_eligible_rank"] = np.nan
    eligible = ranked["profile_eligible"] == True
    ranked.loc[eligible, "profile_eligible_rank"] = (
        ranked.loc[eligible]
        .groupby("profile")["profile_score"]
        .rank(method="min", ascending=False)
    )
    return ranked.sort_values(["profile", "profile_rank", "strategy_name"]).reset_index(drop=True)


def build_profile_winners(rankings: pd.DataFrame) -> pd.DataFrame:
    """Summarize winners by profile."""
    rows: list[dict[str, Any]] = []
    for profile_name in PROFILE_ORDER:
        profile_rows = rankings[rankings["profile"] == profile_name].sort_values("profile_rank")
        if profile_rows.empty:
            continue
        best_overall = profile_rows.iloc[0]
        td3_rows = profile_rows[profile_rows["strategy_type"] == "td3"]
        benchmark_rows = profile_rows[profile_rows["strategy_type"] == "benchmark"]
        best_td3 = td3_rows.iloc[0] if not td3_rows.empty else None
        best_benchmark = benchmark_rows.iloc[0] if not benchmark_rows.empty else None

        rows.append(
            {
                "profile": profile_name,
                "overall_winner": best_overall["strategy_name"],
                "overall_winner_type": best_overall["strategy_type"],
                "overall_winner_score": best_overall["profile_score"],
                "best_td3_candidate": _winner_name(best_td3),
                "best_td3_score": _winner_score(best_td3),
                "best_benchmark": _winner_name(best_benchmark),
                "best_benchmark_score": _winner_score(best_benchmark),
                "td3_beats_best_benchmark": (
                    bool(best_td3 is not None and best_benchmark is not None and best_td3["profile_score"] > best_benchmark["profile_score"])
                ),
                "v3_clean_no_dxy_rank": _rank_for(profile_rows, KEY_CANDIDATE_V3_CLEAN),
                "v3_clean_no_dxy_score": _score_for(profile_rows, KEY_CANDIDATE_V3_CLEAN),
                "v3_clean_no_dxy_is_best_td3": (
                    best_td3 is not None and best_td3["strategy_name"] == KEY_CANDIDATE_V3_CLEAN
                ),
                "v7_clean_no_dxy_garch_rank": _rank_for(profile_rows, KEY_CANDIDATE_V7_CLEAN_GARCH),
                "v7_clean_no_dxy_garch_score": _score_for(profile_rows, KEY_CANDIDATE_V7_CLEAN_GARCH),
                "v7_clean_no_dxy_garch_is_best_td3": (
                    best_td3 is not None and best_td3["strategy_name"] == KEY_CANDIDATE_V7_CLEAN_GARCH
                ),
                "v5_asset_specific_rank": _rank_for(profile_rows, KEY_CANDIDATE_V5_ASSET_SPECIFIC),
                "v5_asset_specific_score": _score_for(profile_rows, KEY_CANDIDATE_V5_ASSET_SPECIFIC),
                "v5_asset_specific_is_best_td3": (
                    best_td3 is not None and best_td3["strategy_name"] == KEY_CANDIDATE_V5_ASSET_SPECIFIC
                ),
                "v3_clean_asset_specific_rank": _rank_for(
                    profile_rows,
                    KEY_CANDIDATE_V3_CLEAN_ASSET_SPECIFIC,
                ),
                "v3_clean_asset_specific_score": _score_for(
                    profile_rows,
                    KEY_CANDIDATE_V3_CLEAN_ASSET_SPECIFIC,
                ),
                "v3_clean_asset_specific_is_best_td3": (
                    best_td3 is not None
                    and best_td3["strategy_name"] == KEY_CANDIDATE_V3_CLEAN_ASSET_SPECIFIC
                ),
                "aggressive_admits_higher_return_less_conservative": _aggressive_admission_note(profile_name, best_overall),
            }
        )
    return pd.DataFrame(rows)


def build_summary_markdown(winners: pd.DataFrame, rankings: pd.DataFrame) -> str:
    """Create a concise markdown summary."""
    lines = [
        "# Mandate Profile Comparison",
        "",
        "This is a reporting-only evaluation layer. It does not retrain models, does not modify scoring logic, and does not replace the current main result.",
        "",
        "The report applies the mandate profiles from `src/risk/mandate_profiles.py` to already-generated final strategy metrics.",
        "",
        "## Winners by Profile",
        "",
    ]
    for _, row in winners.iterrows():
        lines.append(
            f"- {row['profile']}: overall winner `{row['overall_winner']}`; "
            f"best TD3 `{row['best_td3_candidate']}`; best benchmark `{row['best_benchmark']}`."
        )

    best_td3_by_profile = winners.set_index("profile")["best_td3_candidate"].to_dict()
    unique_best_td3 = {candidate for candidate in best_td3_by_profile.values() if pd.notna(candidate)}
    overall_winners = winners.set_index("profile")["overall_winner"].to_dict()
    benchmark_profile_wins = winners[winners["overall_winner_type"] == "benchmark"]
    lines.extend(["", "## Interpretation", ""])
    if len(unique_best_td3) == 1:
        candidate = next(iter(unique_best_td3))
        lines.append(f"The preferred TD3 model does not change across profiles: `{candidate}` remains the best TD3 candidate.")
    else:
        lines.append("The preferred TD3 model changes by mandate profile, so model preference is mandate-dependent.")

    if KEY_CANDIDATE_V5_ASSET_SPECIFIC in unique_best_td3:
        if len(unique_best_td3) == 1:
            lines.append("Under the asset-specific-cost universe, `V5_no_volatility_block_cap_0p50` remains the preferred TD3 candidate across mandate profiles.")
        else:
            lines.append("Under the asset-specific-cost universe, `V5_no_volatility_block_cap_0p50` is preferred in at least one mandate profile.")
    elif KEY_CANDIDATE_V5_ASSET_SPECIFIC in set(rankings["strategy_name"].dropna()):
        lines.append("`V5_no_volatility_block_cap_0p50` is included, but another TD3 candidate is preferred under these mandate-profile rankings.")

    if KEY_CANDIDATE_V3_CLEAN_ASSET_SPECIFIC in unique_best_td3:
        lines.append("`V3_real_macro_vintage_clean_no_dxy_cap_0p70` remains the best TD3 candidate in at least one asset-specific-cost mandate profile.")
    elif KEY_CANDIDATE_V3_CLEAN_ASSET_SPECIFIC in set(rankings["strategy_name"].dropna()):
        lines.append("`V3_real_macro_vintage_clean_no_dxy_cap_0p70` is included as the clean macro comparison candidate, but it does not lead every profile.")

    if KEY_CANDIDATE_V3_CLEAN in unique_best_td3:
        lines.append("`V3_real_macro_vintage_clean_no_dxy_cap_0.50` remains preferred in at least one profile.")
    if KEY_CANDIDATE_V7_CLEAN_GARCH in unique_best_td3:
        lines.append("`V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50` becomes preferred in at least one profile.")
    elif KEY_CANDIDATE_V7_CLEAN_GARCH in set(rankings["strategy_name"].dropna()):
        lines.append("`V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50` is evaluated but does not displace the preferred TD3 candidate in these profile rankings.")

    if benchmark_profile_wins.empty:
        lines.append("No benchmark wins outright in these mandate-profile rankings.")
    else:
        profiles = ", ".join(benchmark_profile_wins["profile"].astype(str).tolist())
        lines.append(f"Benchmarks win outright in these profiles: {profiles}.")

    aggressive_winner = winners[winners["profile"] == "aggressive"]
    if not aggressive_winner.empty:
        lines.append(
            "The aggressive profile can admit higher-return or less conservative strategies because its drawdown, volatility, concentration, and turnover limits are looser."
        )
    if overall_winners:
        lines.append(
            "Profile winners are: "
            + "; ".join(f"{profile} = `{winner}`" for profile, winner in overall_winners.items())
            + "."
        )

    lines.extend(
        [
            "",
            "These rankings should be read as mandate diagnostics, not statistical proof of superiority.",
            "",
            "## Caveats",
            "",
            "- The profile score is reporting-only and is separate from production `robust_score` and `mandate_aware_score`.",
            "- Missing metric columns are not penalized; available constraints are reported explicitly.",
            "- Bootstrap validation and regime analysis remain necessary context.",
        ]
    )
    return "\n".join(lines) + "\n"


def _score_upper_bound(value: Any, limit: float) -> dict[str, Any]:
    numeric = _numeric(value, default=np.nan)
    if pd.isna(numeric):
        return {"available": False, "passes": True, "multiplier": 1.0}
    passes = numeric <= limit
    if passes:
        multiplier = 1.0
    elif numeric <= 0.0:
        multiplier = 1.0
    else:
        multiplier = max(0.0, min(1.0, limit / numeric))
    return {"available": True, "passes": bool(passes), "multiplier": float(multiplier)}


def _score_lower_bound(
    value: Any,
    limit: float,
    *,
    higher_is_better: bool,
    negative_drawdown: bool = False,
) -> dict[str, Any]:
    numeric = _numeric(value, default=np.nan)
    if pd.isna(numeric):
        return {"available": False, "passes": True, "multiplier": 1.0}
    if not higher_is_better:
        raise ValueError("Only higher-is-better lower-bound scoring is supported.")
    passes = numeric >= limit
    if passes:
        multiplier = 1.0
    elif negative_drawdown:
        multiplier = max(0.0, min(1.0, abs(limit) / max(abs(numeric), 1e-12)))
    elif limit <= 0.0:
        multiplier = 0.0
    else:
        multiplier = max(0.0, min(1.0, numeric / limit))
    return {"available": True, "passes": bool(passes), "multiplier": float(multiplier)}


def _numeric(value: Any, default: float) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return default
    return float(numeric)


def _winner_name(row: pd.Series | None) -> str | float:
    if row is None:
        return np.nan
    return row["strategy_name"]


def _winner_score(row: pd.Series | None) -> float:
    if row is None:
        return np.nan
    return float(row["profile_score"])


def _rank_for(rows: pd.DataFrame, strategy_name: str) -> float:
    matched = rows[rows["strategy_name"] == strategy_name]
    if matched.empty:
        return np.nan
    return float(matched.iloc[0]["profile_rank"])


def _score_for(rows: pd.DataFrame, strategy_name: str) -> float:
    matched = rows[rows["strategy_name"] == strategy_name]
    if matched.empty:
        return np.nan
    return float(matched.iloc[0]["profile_score"])


def _aggressive_admission_note(profile_name: str, winner: pd.Series) -> bool:
    if profile_name != "aggressive":
        return False
    return not bool(winner.get("profile_eligible", False)) or bool(
        _numeric(winner.get("max_drawdown"), default=0.0) < -0.20
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (dict, list, str, bool)) else False:
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build mandate profile comparison report from final constrained TD3 outputs.",
    )
    parser.add_argument("--final-report-dir", default=DEFAULT_FINAL_REPORT_DIR)
    parser.add_argument(
        "--combined-report-dir",
        default=None,
        help="Optional asset-specific TD3 + benchmark combined report directory.",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        help="Optional benchmark history directory used to validate asset-specific histories.",
    )
    parser.add_argument(
        "--asset-specific-only",
        action="store_true",
        help="Fail if inputs are not asset-specific transaction-cost outputs.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = build_mandate_profile_comparison_report(
        final_report_dir=args.final_report_dir,
        combined_report_dir=args.combined_report_dir,
        benchmark_dir=args.benchmark_dir,
        asset_specific_only=args.asset_specific_only,
        output_dir=args.output_dir,
    )

    print("Winners by mandate profile:")
    print(result["winners"].to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
