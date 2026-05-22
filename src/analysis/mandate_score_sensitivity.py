"""Mandate-aware score sensitivity across drawdown threshold scenarios.

This module is reporting-only. It recalculates mandate-aware rankings under
alternative drawdown bucket thresholds without changing production scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = "outputs/tables/capped_td3_protocol_comparison_60ep_10seeds_cap060"
DEFAULT_OUTPUT_DIR = "outputs/tables/mandate_score_sensitivity_60ep_10seeds_cap060"
INPUT_FILE = "capped_td3_vs_benchmarks_summary.csv"

FOCUS_STRATEGIES = [
    "V2_cap_0.60",
    "V5_cap_0.60",
    "V6_cap_0.60",
    "V2_uncapped",
    "V5_uncapped",
    "V6_uncapped",
    "BuyHold_GLD",
    "trend_spy_cash_12p",
    "rolling_markowitz_min_variance_52p",
    "defensive_risk_off_12p",
    "rolling_risk_parity_inverse_vol_12p",
    "60_40_SPY_TLT",
]


@dataclass(frozen=True)
class MandateScenario:
    """Drawdown thresholds for mandate bucket sensitivity."""

    name: str
    clean_threshold: float
    yellow_threshold: float
    red_threshold: float


SCENARIOS = [
    MandateScenario("strict", clean_threshold=-0.15, yellow_threshold=-0.25, red_threshold=-0.30),
    MandateScenario("base", clean_threshold=-0.20, yellow_threshold=-0.25, red_threshold=-0.30),
    MandateScenario("flexible", clean_threshold=-0.25, yellow_threshold=-0.30, red_threshold=-0.35),
]

ALL_SCENARIOS_COLUMNS = [
    "scenario",
    "strategy_name",
    "strategy_type",
    "constraint_status",
    "robust_score",
    "scenario_mandate_aware_score",
    "mandate_bucket",
    "max_drawdown",
    "recovery_required",
    "drawdown_multiplier",
    "annualized_return",
    "sharpe",
    "average_turnover",
    "average_effective_number_of_assets",
    "average_max_weight",
    "scenario_rank",
]


def build_mandate_score_sensitivity(
    input_dir: str = DEFAULT_INPUT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build sensitivity tables and Markdown summary."""
    input_path = Path(input_dir) / INPUT_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input comparison summary: {input_path}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(input_path)
    all_scenarios = build_all_scenarios_table(source)
    top10 = build_top10_by_scenario(all_scenarios)
    td3_focus = build_td3_focus_table(all_scenarios)
    summary = build_sensitivity_summary(all_scenarios)
    markdown = build_sensitivity_markdown(summary)

    paths = {
        "all_scenarios": output_path / "mandate_score_sensitivity_all_scenarios.csv",
        "top10_by_scenario": output_path / "mandate_score_sensitivity_top10_by_scenario.csv",
        "td3_focus": output_path / "mandate_score_sensitivity_td3_focus.csv",
        "summary": output_path / "mandate_score_sensitivity_summary.csv",
        "markdown_summary": output_path / "mandate_score_sensitivity_summary.md",
    }
    all_scenarios.to_csv(paths["all_scenarios"], index=False)
    top10.to_csv(paths["top10_by_scenario"], index=False)
    td3_focus.to_csv(paths["td3_focus"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["markdown_summary"].write_text(markdown, encoding="utf-8")
    return {
        "all_scenarios": all_scenarios,
        "top10_by_scenario": top10,
        "td3_focus": td3_focus,
        "summary": summary,
        "markdown_summary": markdown,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def build_all_scenarios_table(source: pd.DataFrame) -> pd.DataFrame:
    """Compute rankings for every mandate threshold scenario."""
    frames = [score_scenario(source, scenario) for scenario in SCENARIOS]
    return pd.concat(frames, ignore_index=True)


def score_scenario(source: pd.DataFrame, scenario: MandateScenario) -> pd.DataFrame:
    """Recompute mandate score for one drawdown threshold scenario."""
    result = source.copy()
    result["scenario"] = scenario.name
    result["max_drawdown"] = pd.to_numeric(result["max_drawdown"], errors="coerce")
    result["robust_score"] = pd.to_numeric(result["robust_score"], errors="coerce")
    result["mandate_bucket"] = result["max_drawdown"].apply(
        lambda value: assign_scenario_bucket(value, scenario)
    )
    result["recovery_required"] = result["max_drawdown"].apply(calculate_recovery_required)
    result["drawdown_multiplier"] = result["recovery_required"].apply(
        lambda value: max(0.0, 1.0 - float(value))
    )
    result["scenario_mandate_aware_score"] = (
        result["robust_score"] * result["drawdown_multiplier"]
    )
    result.loc[result["mandate_bucket"] == "not_eligible", "scenario_mandate_aware_score"] = 0.0
    result["scenario_rank"] = (
        result["scenario_mandate_aware_score"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    result = result.sort_values(
        ["scenario_rank", "robust_score"],
        ascending=[True, False],
    ).reset_index(drop=True)
    return _select_columns(result, ALL_SCENARIOS_COLUMNS)


def assign_scenario_bucket(max_drawdown: float, scenario: MandateScenario) -> str:
    """Assign scenario-specific mandate bucket."""
    value = float(max_drawdown)
    if value >= scenario.clean_threshold:
        return "clean_mandate"
    if value >= scenario.yellow_threshold:
        return "eligible_yellow"
    if value >= scenario.red_threshold:
        return "eligible_red"
    return "not_eligible"


def calculate_recovery_required(max_drawdown: float) -> float:
    """Return recovery required from a negative max drawdown."""
    abs_dd = abs(float(max_drawdown))
    if abs_dd >= 1.0:
        return float("inf")
    return abs_dd / (1.0 - abs_dd)


def build_top10_by_scenario(all_scenarios: pd.DataFrame) -> pd.DataFrame:
    """Return top 10 strategies per scenario."""
    return (
        all_scenarios.sort_values(["scenario", "scenario_rank", "robust_score"])
        .groupby("scenario", group_keys=False)
        .head(10)
        .reset_index(drop=True)
    )


def build_td3_focus_table(all_scenarios: pd.DataFrame) -> pd.DataFrame:
    """Return sensitivity table for TD3 and key benchmark focus strategies."""
    focus = all_scenarios[all_scenarios["strategy_name"].isin(FOCUS_STRATEGIES)].copy()
    order = {name: index for index, name in enumerate(FOCUS_STRATEGIES)}
    focus["_strategy_order"] = focus["strategy_name"].map(order).fillna(999)
    return (
        focus.sort_values(["scenario", "_strategy_order"])
        .drop(columns=["_strategy_order"])
        .reset_index(drop=True)
    )


def build_sensitivity_summary(all_scenarios: pd.DataFrame) -> pd.DataFrame:
    """Summarize winners and key ranks for each scenario."""
    rows = []
    for scenario, group in all_scenarios.groupby("scenario", sort=False):
        ranked = group.sort_values(
            ["scenario_mandate_aware_score", "robust_score"],
            ascending=[False, False],
        )
        td3 = ranked[ranked["strategy_type"].astype(str).str.startswith("td3")]
        benchmarks = ranked[ranked["strategy_type"].astype(str) == "benchmark"]
        top = ranked.iloc[0]
        best_td3 = td3.iloc[0] if not td3.empty else pd.Series(dtype=object)
        best_benchmark = (
            benchmarks.iloc[0] if not benchmarks.empty else pd.Series(dtype=object)
        )
        rows.append(
            {
                "scenario": scenario,
                "top_strategy": top["strategy_name"],
                "top_strategy_type": top["strategy_type"],
                "top_score": top["scenario_mandate_aware_score"],
                "best_td3_strategy": best_td3.get("strategy_name", pd.NA),
                "best_td3_score": best_td3.get("scenario_mandate_aware_score", pd.NA),
                "best_benchmark_strategy": best_benchmark.get("strategy_name", pd.NA),
                "best_benchmark_score": best_benchmark.get(
                    "scenario_mandate_aware_score",
                    pd.NA,
                ),
                "td3_beats_best_benchmark": _beats(
                    best_td3.get("scenario_mandate_aware_score", pd.NA),
                    best_benchmark.get("scenario_mandate_aware_score", pd.NA),
                ),
                "V5_cap_rank": _rank_for(group, "V5_cap_0.60"),
                "V2_cap_rank": _rank_for(group, "V2_cap_0.60"),
                "V6_cap_rank": _rank_for(group, "V6_cap_0.60"),
                "BuyHold_GLD_rank": _rank_for(group, "BuyHold_GLD"),
                "trend_spy_cash_rank": _rank_for(group, "trend_spy_cash_12p"),
            }
        )
    return pd.DataFrame(rows)


def build_sensitivity_markdown(summary: pd.DataFrame) -> str:
    """Create concise Markdown interpretation."""
    lines = [
        "# Mandate Score Sensitivity Summary",
        "",
        "This is a reporting-only sensitivity analysis. It does not retrain models "
        "or change production `mandate_aware_score` logic.",
        "",
        "## Scenario Results",
        "",
    ]
    robust_all = True
    v5_v2_above_gld_all = True
    for _, row in summary.iterrows():
        scenario = row["scenario"]
        td3_beats = bool(row["td3_beats_best_benchmark"])
        robust_all = robust_all and td3_beats
        v5_v2_above = _rank_less_than(row["V5_cap_rank"], row["BuyHold_GLD_rank"]) and (
            _rank_less_than(row["V2_cap_rank"], row["BuyHold_GLD_rank"])
        )
        v5_v2_above_gld_all = v5_v2_above_gld_all and v5_v2_above
        lines.append(
            f"- `{scenario}`: top strategy = `{row['top_strategy']}`; "
            f"best TD3 = `{row['best_td3_strategy']}`; "
            f"best benchmark = `{row['best_benchmark_strategy']}`; "
            f"TD3 beats best benchmark = {td3_beats}; "
            f"V5/V2 above BuyHold_GLD = {v5_v2_above}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Capped TD3 remains the top mandate-aware family across the tested "
                "threshold scenarios."
                if robust_all
                else "The capped TD3 conclusion is threshold-sensitive under at least one scenario."
            ),
            (
                "`V5_cap_0.60` and `V2_cap_0.60` remain above `BuyHold_GLD` across "
                "all tested scenarios."
                if v5_v2_above_gld_all
                else "`V5_cap_0.60` and/or `V2_cap_0.60` do not remain above "
                "`BuyHold_GLD` under every tested scenario."
            ),
            "",
            "## Caveat",
            "",
            "This analysis changes only mandate thresholds in the reporting layer. "
            "It does not validate a new training policy or prove general superiority.",
            "",
        ]
    )
    return "\n".join(lines)


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, columns]


def _rank_for(group: pd.DataFrame, strategy_name: str) -> int | pd.NA:
    row = group[group["strategy_name"] == strategy_name]
    if row.empty:
        return pd.NA
    return int(row.iloc[0]["scenario_rank"])


def _beats(left: Any, right: Any) -> bool:
    left_value = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    right_value = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    if pd.isna(left_value) or pd.isna(right_value):
        return False
    return float(left_value) > float(right_value)


def _rank_less_than(left: Any, right: Any) -> bool:
    left_value = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    right_value = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    if pd.isna(left_value) or pd.isna(right_value):
        return False
    return int(left_value) < int(right_value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mandate-aware score threshold sensitivity analysis.",
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = build_mandate_score_sensitivity(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    print(result["summary"].to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
