"""Build compact final figures for asset-specific-cost results.

This module is reporting-only. It reads already-generated asset-specific-cost
TD3, benchmark, statistical validation, WRC, regime, and mandate-profile
reports, then writes compact PNG figures. It does not retrain models or modify
ranking/scoring logic.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TD3_REPORT_DIR = "outputs/tables/asset_specific_cost_full_final_report"
DEFAULT_BENCHMARK_COMPARISON_DIR = "outputs/tables/asset_specific_cost_benchmark_comparison"
DEFAULT_STATISTICAL_VALIDATION_DIR = "outputs/tables/asset_specific_cost_statistical_validation"
DEFAULT_WRC_DIR = "outputs/tables/asset_specific_cost_white_reality_check"
DEFAULT_REGIME_ANALYSIS_DIR = "outputs/tables/asset_specific_cost_regime_analysis"
DEFAULT_MANDATE_PROFILE_DIR = "outputs/tables/asset_specific_cost_mandate_profile_comparison"
DEFAULT_OUTPUT_DIR = "outputs/figures/asset_specific_cost_final"

COMBINED_RANKING_FILE = "asset_specific_cost_combined_ranking.csv"
SELECTED_CANDIDATES_FILE = "asset_specific_cost_selected_candidates.csv"
WRC_SUMMARY_FILE = "white_reality_check_summary.csv"
PAIRWISE_BOOTSTRAP_FILE = "statistical_validation_pairwise_bootstrap.csv"
REGIME_METRICS_FILE = "regime_strategy_metrics.csv"
MANDATE_WINNERS_FILE = "mandate_profile_winners.csv"

FIGURE_FILENAMES = {
    "combined_mandate_vs_drawdown": "combined_mandate_score_vs_max_drawdown.png",
    "td3_selected_mandate_scores": "td3_selected_mandate_scores.png",
    "mandate_profile_winners": "mandate_profile_winners.png",
    "wrc_p_values": "statistical_caution_wrc_p_values.png",
    "regime_relative_performance": "regime_relative_performance.png",
}

KEY_STRATEGIES = [
    "V5_no_volatility_block_cap_0p50",
    "V3_real_macro_vintage_clean_no_dxy_cap_0p70",
    "V4_real_garch_current_cap_0p50",
    "trend_spy_cash_12p",
    "BuyHold_GLD",
    "Equal_Weight",
]

REGIME_STRATEGIES = [
    "V5_no_volatility_block_cap_0p50",
    "V4_real_garch_current_cap_0p50",
    "V3_real_macro_vintage_clean_no_dxy_cap_0p70",
]

SHORT_LABELS = {
    "V5_no_volatility_block_cap_0p50": "V5 cap 0.50",
    "V3_real_macro_vintage_clean_no_dxy_cap_0p70": "V3 clean cap 0.70",
    "V4_real_garch_current_cap_0p50": "V4 GARCH cap 0.50",
    "V7_real_macro_vintage_clean_no_dxy_garch_cap_0p50": "V7 clean+GARCH",
    "V8_ewma_garch_vol_current_cap_0p80": "V8 EWMA/GARCH",
    "V2_reference_full_cap_0p80": "V2 cap 0.80",
    "V6_financial_state_cap_0p50": "V6 cap 0.50",
    "trend_spy_cash_12p": "trend cash",
    "BuyHold_GLD": "GLD",
    "Equal_Weight": "equal weight",
    "rolling_markowitz_min_variance_52p": "min variance",
}


def build_asset_specific_cost_figures(
    td3_report_dir: str = DEFAULT_TD3_REPORT_DIR,
    benchmark_comparison_dir: str = DEFAULT_BENCHMARK_COMPARISON_DIR,
    statistical_validation_dir: str = DEFAULT_STATISTICAL_VALIDATION_DIR,
    white_reality_check_dir: str = DEFAULT_WRC_DIR,
    regime_analysis_dir: str = DEFAULT_REGIME_ANALYSIS_DIR,
    mandate_profile_dir: str = DEFAULT_MANDATE_PROFILE_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build all asset-specific-cost final figures."""
    td3_path = Path(td3_report_dir)
    benchmark_path = Path(benchmark_comparison_dir)
    stat_path = Path(statistical_validation_dir)
    wrc_path = Path(white_reality_check_dir)
    regime_path = Path(regime_analysis_dir)
    mandate_path = Path(mandate_profile_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    combined = _read_csv(benchmark_path / COMBINED_RANKING_FILE, warnings)
    selected = _read_csv(td3_path / SELECTED_CANDIDATES_FILE, warnings)
    wrc = _read_csv(wrc_path / WRC_SUMMARY_FILE, warnings)
    pairwise = _read_csv(stat_path / PAIRWISE_BOOTSTRAP_FILE, warnings)
    regime = _read_csv(regime_path / REGIME_METRICS_FILE, warnings)
    mandate_winners = _read_csv(mandate_path / MANDATE_WINNERS_FILE, warnings)

    figure_paths = {
        name: str(output_path / filename)
        for name, filename in FIGURE_FILENAMES.items()
    }
    plot_combined_mandate_vs_drawdown(
        combined,
        Path(figure_paths["combined_mandate_vs_drawdown"]),
        warnings,
    )
    plot_td3_selected_mandate_scores(
        selected,
        Path(figure_paths["td3_selected_mandate_scores"]),
        warnings,
    )
    plot_mandate_profile_winners(
        mandate_winners,
        Path(figure_paths["mandate_profile_winners"]),
        warnings,
    )
    plot_wrc_p_values(
        wrc,
        pairwise,
        Path(figure_paths["wrc_p_values"]),
        warnings,
    )
    plot_regime_relative_performance(
        regime,
        Path(figure_paths["regime_relative_performance"]),
        warnings,
    )

    metadata = {
        "runner": "src.analysis.build_asset_specific_cost_figures",
        "td3_report_dir": td3_report_dir,
        "benchmark_comparison_dir": benchmark_comparison_dir,
        "statistical_validation_dir": statistical_validation_dir,
        "white_reality_check_dir": white_reality_check_dir,
        "regime_analysis_dir": regime_analysis_dir,
        "mandate_profile_dir": mandate_profile_dir,
        "output_dir": output_dir,
        "figure_paths": figure_paths,
        "warnings": warnings,
        "reporting_only_note": "Figures are built from existing asset-specific-cost reports; no training is run.",
    }
    summary = build_summary_markdown(metadata)
    metadata_path = output_path / "asset_specific_cost_final_figures_metadata.json"
    summary_path = output_path / "asset_specific_cost_final_figures_summary.md"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")
    return {
        "figure_paths": figure_paths,
        "warnings": warnings,
        "metadata": metadata,
        "summary": summary,
        "metadata_path": str(metadata_path),
        "summary_path": str(summary_path),
    }


def plot_combined_mandate_vs_drawdown(
    df: pd.DataFrame,
    output_path: Path,
    warnings: list[str],
) -> None:
    required = ["strategy_name", "strategy_type", "max_drawdown", "mandate_aware_score"]
    if not _has_columns(df, required, warnings, "combined_mandate_vs_drawdown"):
        _placeholder(output_path, "Missing combined ranking columns")
        return
    plot_df = _numeric_frame(df, ["max_drawdown", "mandate_aware_score"])
    if plot_df.empty:
        _placeholder(output_path, "No combined ranking data")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"td3": "#4c78a8", "benchmark": "#f58518"}
    for strategy_type, frame in plot_df.groupby("strategy_type", dropna=False):
        ax.scatter(
            frame["max_drawdown"],
            frame["mandate_aware_score"],
            label=str(strategy_type),
            color=colors.get(str(strategy_type), "#8c8c8c"),
            s=58,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.6,
        )
    _annotate(ax, plot_df, "max_drawdown", "mandate_aware_score", KEY_STRATEGIES)
    ax.axvline(-0.30, color="0.45", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Max drawdown")
    ax.set_ylabel("Mandate-aware score")
    ax.set_title("Asset-Specific Costs: Mandate Score vs. Drawdown")
    ax.grid(True, alpha=0.22)
    _add_margins(ax, plot_df["max_drawdown"], plot_df["mandate_aware_score"])
    _legend_outside(ax)
    _save(fig, output_path)


def plot_td3_selected_mandate_scores(
    df: pd.DataFrame,
    output_path: Path,
    warnings: list[str],
) -> None:
    name_col = "candidate_name" if "candidate_name" in df.columns else "strategy_name"
    required = [name_col, "mandate_aware_score"]
    if not _has_columns(df, required, warnings, "td3_selected_mandate_scores"):
        _placeholder(output_path, "Missing selected TD3 columns")
        return
    plot_df = df.copy()
    plot_df["mandate_aware_score"] = pd.to_numeric(plot_df["mandate_aware_score"], errors="coerce")
    plot_df = plot_df.dropna(subset=["mandate_aware_score"]).sort_values("mandate_aware_score")
    if plot_df.empty:
        _placeholder(output_path, "No selected TD3 scores")
        return
    fig, ax = plt.subplots(figsize=(7.2, max(3.8, len(plot_df) * 0.36)))
    labels = [_short(name) for name in plot_df[name_col]]
    colors = [
        "#4c78a8" if name != "V5_no_volatility_block_cap_0p50" else "#2f6f4e"
        for name in plot_df[name_col].astype(str)
    ]
    ax.barh(labels, plot_df["mandate_aware_score"], color=colors)
    ax.set_xlabel("Mandate-aware score")
    ax.set_title("Selected TD3 Candidates by Mandate-Aware Score")
    ax.grid(axis="x", alpha=0.22)
    for y_pos, value in enumerate(plot_df["mandate_aware_score"]):
        ax.text(value + 0.008, y_pos, f"{value:.3f}", va="center", fontsize=8)
    _save(fig, output_path)


def plot_mandate_profile_winners(
    df: pd.DataFrame,
    output_path: Path,
    warnings: list[str],
) -> None:
    required = ["profile", "overall_winner", "overall_winner_score", "overall_winner_type"]
    if not _has_columns(df, required, warnings, "mandate_profile_winners"):
        _placeholder(output_path, "Missing mandate-profile winners")
        return
    plot_df = df.copy()
    plot_df["overall_winner_score"] = pd.to_numeric(plot_df["overall_winner_score"], errors="coerce")
    plot_df = plot_df.dropna(subset=["overall_winner_score"])
    if plot_df.empty:
        _placeholder(output_path, "No mandate-profile winners")
        return
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(plot_df))
    colors = ["#4c78a8" if value == "td3" else "#f58518" for value in plot_df["overall_winner_type"]]
    ax.bar(x, plot_df["overall_winner_score"], color=colors, width=0.58)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["profile"], fontsize=9)
    ax.set_ylabel("Profile score")
    ax.set_title("Mandate Profile Winners")
    ax.grid(axis="y", alpha=0.22)
    for x_pos, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            x_pos,
            row["overall_winner_score"] + 0.015,
            _short(row["overall_winner"]),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=12,
        )
    _save(fig, output_path)


def plot_wrc_p_values(
    wrc: pd.DataFrame,
    pairwise: pd.DataFrame,
    output_path: Path,
    warnings: list[str],
) -> None:
    required = ["benchmark", "p_value", "best_candidate_by_mean_diff"]
    if not _has_columns(wrc, required, warnings, "wrc_p_values"):
        _placeholder(output_path, "Missing WRC summary")
        return
    plot_df = wrc.copy()
    plot_df["p_value"] = pd.to_numeric(plot_df["p_value"], errors="coerce")
    plot_df = plot_df.dropna(subset=["p_value"]).sort_values("p_value")
    if plot_df.empty:
        _placeholder(output_path, "No WRC p-values")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = [_short(name) for name in plot_df["benchmark"]]
    ax.barh(labels, plot_df["p_value"], color="#9d755d")
    ax.axvline(0.05, color="#d62728", linestyle="--", linewidth=1.0, label="0.05")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("White Reality Check p-value")
    ax.set_title("Statistical Caution: WRC p-values")
    ax.grid(axis="x", alpha=0.22)
    for y_pos, value in enumerate(plot_df["p_value"]):
        ax.text(min(value + 0.025, 0.96), y_pos, f"{value:.3f}", va="center", fontsize=8)
    subtitle = _pairwise_caution_text(pairwise)
    if subtitle:
        ax.text(0.0, -0.22, subtitle, transform=ax.transAxes, fontsize=8, color="0.35")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, output_path)


def plot_regime_relative_performance(
    df: pd.DataFrame,
    output_path: Path,
    warnings: list[str],
) -> None:
    value_col = "mandate_style_score"
    required = ["regime_name", "strategy_name", value_col]
    if not _has_columns(df, required, warnings, "regime_relative_performance"):
        _placeholder(output_path, "Missing regime metrics")
        return
    available = [strategy for strategy in REGIME_STRATEGIES if strategy in set(df["strategy_name"])]
    if not available:
        _placeholder(output_path, "No requested regime strategies")
        warnings.append("regime_relative_performance: requested strategies unavailable.")
        return
    pivot = (
        df[df["strategy_name"].isin(available)]
        .pivot_table(index="regime_name", columns="strategy_name", values=value_col, aggfunc="mean")
        .reindex(columns=available)
    )
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        _placeholder(output_path, "No regime observations for selected TD3 candidates")
        return
    fig, ax = plt.subplots(figsize=(7.4, max(4.0, len(pivot) * 0.34)))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#eeeeee")
    matrix = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([_short(col) for col in pivot.columns], rotation=28, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Regime Summary: V5/V4/V3 Mandate-Style Score")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(value_col)
    ax.text(
        0.0,
        -0.18,
        "Gray cells indicate unavailable histories.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.35",
    )
    _save(fig, output_path)


def build_summary_markdown(metadata: dict[str, Any]) -> str:
    warnings = metadata["warnings"]
    return "\n".join(
        [
            "# Asset-Specific-Cost Final Figures",
            "",
            "Generated compact publication-ready figures:",
            *[f"- `{Path(path).name}`" for path in metadata["figure_paths"].values()],
            "",
            "Interpretation:",
            "- These figures use the asset-specific transaction-cost TD3 + benchmark universe.",
            "- They are reporting-only and do not retrain or alter rankings.",
            "- The WRC p-value figure is a statistical caution layer; it does not support a superiority claim when p-values remain high.",
            "- The regime figure emphasizes that the TD3 advantage is regime-specific rather than broad dominance.",
            "",
            "Warnings:",
            *([f"- {warning}" for warning in warnings] if warnings else ["- None."]),
            "",
        ]
    )


def _read_csv(path: Path, warnings: list[str]) -> pd.DataFrame:
    if not path.exists():
        warnings.append(f"Missing input file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _has_columns(
    df: pd.DataFrame,
    columns: list[str],
    warnings: list[str],
    figure_name: str,
) -> bool:
    missing = [column for column in columns if column not in df.columns]
    if df.empty:
        warnings.append(f"{figure_name}: input table is empty.")
        return False
    if missing:
        warnings.append(f"{figure_name}: missing columns {missing}.")
        return False
    return True


def _numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=columns)


def _annotate(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    labels: list[str],
) -> None:
    offsets = {
        "V5_no_volatility_block_cap_0p50": (8, 10),
        "V3_real_macro_vintage_clean_no_dxy_cap_0p70": (8, -18),
        "V4_real_garch_current_cap_0p50": (8, 18),
        "trend_spy_cash_12p": (-88, -16),
        "BuyHold_GLD": (-64, 10),
        "Equal_Weight": (8, 10),
    }
    label_df = df[df["strategy_name"].astype(str).isin(labels)].copy()
    for _, row in label_df.iterrows():
        name = str(row["strategy_name"])
        offset = offsets.get(name, (8, 8))
        ax.annotate(
            _short(name),
            (row[x_col], row[y_col]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha="left" if offset[0] >= 0 else "right",
            arrowprops={"arrowstyle": "-", "linewidth": 0.4, "color": "0.35"},
        )


def _pairwise_caution_text(pairwise: pd.DataFrame) -> str:
    if pairwise.empty or "probability_candidate_beats" not in pairwise.columns:
        return ""
    sharpe = pairwise[pairwise.get("metric") == "sharpe"].copy()
    if sharpe.empty:
        return ""
    prob = pd.to_numeric(sharpe["probability_candidate_beats"], errors="coerce").dropna()
    if prob.empty:
        return ""
    return f"Pairwise bootstrap beat probabilities span {prob.min():.2f}-{prob.max():.2f}; uncertainty remains."


def _short(strategy_name: Any) -> str:
    name = str(strategy_name)
    return SHORT_LABELS.get(name, name.replace("_", " "))


def _add_margins(ax: plt.Axes, x_values: pd.Series, y_values: pd.Series) -> None:
    x_values = pd.to_numeric(x_values, errors="coerce").dropna()
    y_values = pd.to_numeric(y_values, errors="coerce").dropna()
    if not x_values.empty:
        x_range = max(float(x_values.max() - x_values.min()), 0.01)
        ax.set_xlim(float(x_values.min() - 0.10 * x_range), float(x_values.max() + 0.22 * x_range))
    if not y_values.empty:
        y_range = max(float(y_values.max() - y_values.min()), 0.01)
        ax.set_ylim(float(y_values.min() - 0.12 * y_range), float(y_values.max() + 0.20 * y_range))


def _legend_outside(ax: plt.Axes) -> None:
    ax.legend(
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )


def _placeholder(output_path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
    ax.set_axis_off()
    _save(fig, output_path)


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build asset-specific-cost final figures.")
    parser.add_argument("--td3-report-dir", default=DEFAULT_TD3_REPORT_DIR)
    parser.add_argument("--benchmark-comparison-dir", default=DEFAULT_BENCHMARK_COMPARISON_DIR)
    parser.add_argument("--statistical-validation-dir", default=DEFAULT_STATISTICAL_VALIDATION_DIR)
    parser.add_argument("--white-reality-check-dir", default=DEFAULT_WRC_DIR)
    parser.add_argument("--regime-analysis-dir", default=DEFAULT_REGIME_ANALYSIS_DIR)
    parser.add_argument("--mandate-profile-dir", default=DEFAULT_MANDATE_PROFILE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = build_asset_specific_cost_figures(
        td3_report_dir=args.td3_report_dir,
        benchmark_comparison_dir=args.benchmark_comparison_dir,
        statistical_validation_dir=args.statistical_validation_dir,
        white_reality_check_dir=args.white_reality_check_dir,
        regime_analysis_dir=args.regime_analysis_dir,
        mandate_profile_dir=args.mandate_profile_dir,
        output_dir=args.output_dir,
    )
    print("Figures created:")
    for path in result["figure_paths"].values():
        print(path)
    if result["warnings"]:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
