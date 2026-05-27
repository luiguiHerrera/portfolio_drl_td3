"""Build final publication-ready figures for constrained TD3 results.

This module is reporting-only. It reads existing final comparison and regime
analysis tables, creates PNG figures with matplotlib, and writes metadata plus
a short summary. It does not retrain models or modify scoring logic.
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


DEFAULT_FINAL_REPORT_DIR = "outputs/tables/final_constrained_td3_report_with_v3_v4_v7_v8_60ep_10seeds"
DEFAULT_REGIME_ANALYSIS_DIR = "outputs/tables/regime_analysis_final_v3_v4"
DEFAULT_OUTPUT_DIR = "outputs/figures/final_v3_v4_v7_v8"

KEY_STRATEGIES = [
    "V3_cap_0.60",
    "V4_cap_0.50",
    "BuyHold_GLD",
    "trend_spy_cash_12p",
    "momentum_winner_12p",
]

EFFECTIVE_ASSETS_KEY_STRATEGIES = [
    "V3_cap_0.60",
    "V4_cap_0.50",
    "BuyHold_GLD",
    "trend_spy_cash_12p",
]

HEATMAP_STRATEGIES = [
    "V3_cap_0.60",
    "V4_cap_0.50",
    "V7_cap_0.50",
    "V8_cap_0.50",
    "BuyHold_GLD",
    "trend_spy_cash_12p",
    "rolling_risk_parity_inverse_vol_12p",
    "rolling_markowitz_min_variance_52p",
    "60_40_SPY_TLT",
]

SHORT_STRATEGY_LABELS = {
    "rolling_risk_parity_inverse_vol_12p": "risk_parity",
    "rolling_markowitz_min_variance_52p": "min_var",
    "trend_spy_cash_12p": "trend_cash",
    "60_40_SPY_TLT": "60/40",
}

FIGURE_FILENAMES = {
    "mandate_score_vs_max_drawdown": "mandate_score_vs_max_drawdown.png",
    "robust_score_vs_max_drawdown": "robust_score_vs_max_drawdown.png",
    "effective_assets_vs_mandate_score": "effective_assets_vs_mandate_score.png",
    "regime_mandate_heatmap": "regime_mandate_heatmap.png",
    "regime_winners_bar": "regime_winners_bar.png",
}


def build_final_figures(
    final_report_dir: str = DEFAULT_FINAL_REPORT_DIR,
    regime_analysis_dir: str = DEFAULT_REGIME_ANALYSIS_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Create final figures and write summary/metadata."""
    final_path = Path(final_report_dir)
    regime_path = Path(regime_analysis_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    final_ranking = pd.read_csv(final_path / "final_constrained_td3_main_ranking.csv")
    regime_metrics = pd.read_csv(regime_path / "regime_strategy_metrics.csv")
    regime_winners = pd.read_csv(regime_path / "regime_winners_summary.csv")

    warnings: list[str] = []
    figure_paths = {
        name: str(output_path / filename)
        for name, filename in FIGURE_FILENAMES.items()
    }

    plot_mandate_score_vs_max_drawdown(
        final_ranking,
        Path(figure_paths["mandate_score_vs_max_drawdown"]),
        warnings,
    )
    plot_robust_score_vs_max_drawdown(
        final_ranking,
        Path(figure_paths["robust_score_vs_max_drawdown"]),
        warnings,
    )
    plot_effective_assets_vs_mandate_score(
        final_ranking,
        Path(figure_paths["effective_assets_vs_mandate_score"]),
        warnings,
    )
    plot_regime_mandate_heatmap(
        regime_metrics,
        Path(figure_paths["regime_mandate_heatmap"]),
        warnings,
    )
    plot_regime_winners_bar(
        regime_winners,
        Path(figure_paths["regime_winners_bar"]),
        warnings,
    )

    metadata = {
        "runner": "src.analysis.build_final_figures",
        "final_report_dir": final_report_dir,
        "regime_analysis_dir": regime_analysis_dir,
        "output_dir": output_dir,
        "figure_paths": figure_paths,
        "warnings": warnings,
        "reporting_only_note": "Figures are built from existing report tables; no TD3 training is run.",
    }
    summary = build_summary_markdown(metadata)
    metadata_path = output_path / "final_figures_metadata.json"
    summary_path = output_path / "final_figures_summary.md"
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


def plot_mandate_score_vs_max_drawdown(
    df: pd.DataFrame,
    output_path: Path,
    warnings: list[str],
) -> None:
    required = ["max_drawdown", "mandate_aware_score", "strategy_name"]
    if not _has_columns(df, required, warnings, "mandate_score_vs_max_drawdown"):
        _write_placeholder(output_path, "Missing columns for mandate/drawdown scatter")
        return
    plot_df = _ensure_strategy_group(
        _numeric_frame(df, ["max_drawdown", "mandate_aware_score"]),
        warnings,
        "mandate_score_vs_max_drawdown",
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = _group_colors(plot_df["strategy_group"])
    for group, frame in plot_df.groupby("strategy_group", dropna=False):
        ax.scatter(
            frame["max_drawdown"],
            frame["mandate_aware_score"],
            label=str(group),
            color=colors.get(str(group)),
            s=_point_sizes(frame),
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
        )
    _annotate_key_strategies(
        ax,
        plot_df,
        "max_drawdown",
        "mandate_aware_score",
        labels=KEY_STRATEGIES,
    )
    ax.axvline(-0.30, color="0.35", linestyle="--", linewidth=1.0, label="-30% threshold")
    ax.set_xlabel("Max drawdown")
    ax.set_ylabel("Mandate-aware score")
    ax.set_title("Mandate-Aware Score vs. Max Drawdown")
    ax.grid(True, alpha=0.25)
    _add_plot_margins(ax, plot_df["max_drawdown"], plot_df["mandate_aware_score"])
    _legend_outside(ax)
    _save(fig, output_path)


def plot_robust_score_vs_max_drawdown(
    df: pd.DataFrame,
    output_path: Path,
    warnings: list[str],
) -> None:
    required = ["max_drawdown", "robust_score", "strategy_name"]
    if not _has_columns(df, required, warnings, "robust_score_vs_max_drawdown"):
        _write_placeholder(output_path, "Missing columns for robust/drawdown scatter")
        return
    plot_df = _ensure_strategy_group(
        _numeric_frame(df, ["max_drawdown", "robust_score"]),
        warnings,
        "robust_score_vs_max_drawdown",
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = _group_colors(plot_df["strategy_group"])
    for group, frame in plot_df.groupby("strategy_group", dropna=False):
        ax.scatter(
            frame["max_drawdown"],
            frame["robust_score"],
            label=str(group),
            color=colors.get(str(group)),
            s=_point_sizes(frame),
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
        )
    _annotate_key_strategies(
        ax,
        plot_df,
        "max_drawdown",
        "robust_score",
        labels=KEY_STRATEGIES,
    )
    ax.axvline(-0.30, color="0.35", linestyle="--", linewidth=1.0, label="-30% threshold")
    ax.set_xlabel("Max drawdown")
    ax.set_ylabel("Robust score")
    ax.set_title("Robust Score vs. Max Drawdown")
    ax.grid(True, alpha=0.25)
    _add_plot_margins(ax, plot_df["max_drawdown"], plot_df["robust_score"])
    _legend_outside(ax)
    _save(fig, output_path)


def plot_effective_assets_vs_mandate_score(
    df: pd.DataFrame,
    output_path: Path,
    warnings: list[str],
) -> None:
    required = [
        "average_effective_number_of_assets",
        "mandate_aware_score",
        "strategy_name",
    ]
    if not _has_columns(df, required, warnings, "effective_assets_vs_mandate_score"):
        _write_placeholder(output_path, "Missing columns for effective-assets scatter")
        return
    plot_df = _ensure_strategy_group(
        _numeric_frame(
            df,
            ["average_effective_number_of_assets", "mandate_aware_score"],
        ),
        warnings,
        "effective_assets_vs_mandate_score",
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = _group_colors(plot_df["strategy_group"])
    for group, frame in plot_df.groupby("strategy_group", dropna=False):
        ax.scatter(
            frame["average_effective_number_of_assets"],
            frame["mandate_aware_score"],
            label=str(group),
            color=colors.get(str(group)),
            s=_point_sizes(frame),
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
        )
    _annotate_key_strategies(
        ax,
        plot_df,
        "average_effective_number_of_assets",
        "mandate_aware_score",
        labels=EFFECTIVE_ASSETS_KEY_STRATEGIES,
    )
    ax.set_xlabel("Average effective number of assets")
    ax.set_ylabel("Mandate-aware score")
    ax.set_title("Diversification and Mandate-Aware Quality")
    ax.grid(True, alpha=0.25)
    _add_plot_margins(
        ax,
        plot_df["average_effective_number_of_assets"],
        plot_df["mandate_aware_score"],
    )
    _legend_outside(ax)
    _save(fig, output_path)


def plot_regime_mandate_heatmap(
    metrics: pd.DataFrame,
    output_path: Path,
    warnings: list[str],
) -> None:
    value_column = "mandate_style_score"
    if value_column not in metrics.columns:
        value_column = "sharpe"
        warnings.append("regime_mandate_heatmap: mandate_style_score missing; using Sharpe.")
    required = ["regime_name", "strategy_name", value_column]
    if not _has_columns(metrics, required, warnings, "regime_mandate_heatmap"):
        _write_placeholder(output_path, "Missing columns for regime heatmap")
        return
    available_strategies = [
        strategy for strategy in HEATMAP_STRATEGIES if strategy in set(metrics["strategy_name"])
    ]
    if not available_strategies:
        warnings.append("regime_mandate_heatmap: no requested strategies available.")
        _write_placeholder(output_path, "No requested strategies available")
        return
    pivot = (
        metrics[metrics["strategy_name"].isin(available_strategies)]
        .pivot_table(index="regime_name", columns="strategy_name", values=value_column, aggfunc="mean")
        .reindex(columns=available_strategies)
    )
    fig_width = max(10, len(available_strategies) * 1.25)
    fig_height = max(6, len(pivot.index) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#eeeeee")
    matrix = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [_short_strategy_label(column) for column in pivot.columns],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Regime Mandate-Style Score Heatmap")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(value_column)
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


def plot_regime_winners_bar(
    winners: pd.DataFrame,
    output_path: Path,
    warnings: list[str],
) -> None:
    column = "best_by_mandate_style_score"
    if column not in winners.columns:
        warnings.append("regime_winners_bar: best_by_mandate_style_score missing.")
        _write_placeholder(output_path, "Missing regime winners column")
        return
    counts = winners[column].dropna().astype(str).value_counts().sort_values(ascending=True)
    if counts.empty:
        warnings.append("regime_winners_bar: no winners available.")
        _write_placeholder(output_path, "No regime winners available")
        return
    fig, ax = plt.subplots(figsize=(9, max(4, len(counts) * 0.45)))
    ax.barh(counts.index, counts.values, color="#4c78a8")
    ax.set_xlabel("Number of regime/calendar wins")
    ax.set_title("Regime Wins: No Universal Dominance")
    ax.grid(axis="x", alpha=0.25)
    for y_pos, value in enumerate(counts.values):
        ax.text(value + 0.03, y_pos, str(int(value)), va="center", fontsize=9)
    _save(fig, output_path)


def build_summary_markdown(metadata: dict[str, Any]) -> str:
    """Build short figure summary."""
    warnings = metadata["warnings"]
    return "\n".join(
        [
            "# Final Figures Summary",
            "",
            "Generated figures:",
            *[
                f"- `{Path(path).name}`"
                for path in metadata["figure_paths"].values()
            ],
            "",
            "Interpretation:",
            (
                "- The figures are designed to show that constrained TD3 is "
                "competitive under mandate-aware evaluation, while aggressive "
                "benchmarks can remain strong on robust score but fail drawdown "
                "eligibility."
            ),
            (
                "- The regime heatmap and winners bar emphasize that performance is "
                "regime-dependent; no strategy dominates every regime."
            ),
            "- Gray cells in the heatmap indicate unavailable histories, not zero scores.",
            "",
            "Warnings:",
            *([f"- {warning}" for warning in warnings] if warnings else ["- None."]),
            "",
        ]
    )


def _has_columns(
    df: pd.DataFrame,
    columns: list[str],
    warnings: list[str],
    figure_name: str,
) -> bool:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        warnings.append(f"{figure_name}: missing columns {missing}.")
        return False
    return True


def _numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=columns)


def _ensure_strategy_group(
    df: pd.DataFrame,
    warnings: list[str],
    figure_name: str,
) -> pd.DataFrame:
    result = df.copy()
    if "strategy_group" not in result.columns:
        result["strategy_group"] = "unknown"
        warnings.append(f"{figure_name}: strategy_group missing; using 'unknown'.")
    return result


def _point_sizes(df: pd.DataFrame) -> pd.Series:
    names = df["strategy_name"].astype(str)
    return names.isin(KEY_STRATEGIES).map({True: 90, False: 45})


def _group_colors(groups: pd.Series) -> dict[str, str]:
    palette = [
        "#4c78a8",
        "#f58518",
        "#54a24b",
        "#e45756",
        "#72b7b2",
        "#b279a2",
        "#9d755d",
        "#bab0ac",
    ]
    unique = sorted(str(group) for group in groups.dropna().unique())
    return {group: palette[index % len(palette)] for index, group in enumerate(unique)}


def _annotate_key_strategies(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    labels: list[str],
) -> None:
    offsets = {
        "V3_cap_0.60": (9, 10),
        "V4_cap_0.50": (10, -16),
        "BuyHold_GLD": (-86, 12),
        "trend_spy_cash_12p": (8, -18),
        "momentum_winner_12p": (-108, 10),
    }
    label_df = df[df["strategy_name"].astype(str).isin(set(labels))].copy()
    for index, (_, row) in enumerate(label_df.iterrows()):
        name = str(row["strategy_name"])
        ax.annotate(
            name,
            (row[x_col], row[y_col]),
            xytext=offsets.get(name, (8, 10 + (index % 3) * 6)),
            textcoords="offset points",
            fontsize=8,
            ha="left" if offsets.get(name, (1, 0))[0] >= 0 else "right",
            arrowprops={"arrowstyle": "-", "linewidth": 0.4, "color": "0.35"},
        )


def _add_plot_margins(ax: plt.Axes, x_values: pd.Series, y_values: pd.Series) -> None:
    x_values = pd.to_numeric(x_values, errors="coerce").dropna()
    y_values = pd.to_numeric(y_values, errors="coerce").dropna()
    if not x_values.empty:
        x_range = max(float(x_values.max() - x_values.min()), 0.01)
        ax.set_xlim(float(x_values.min() - x_range * 0.10), float(x_values.max() + x_range * 0.18))
    if not y_values.empty:
        y_range = max(float(y_values.max() - y_values.min()), 0.01)
        ax.set_ylim(float(y_values.min() - y_range * 0.12), float(y_values.max() + y_range * 0.18))


def _legend_outside(ax: plt.Axes) -> None:
    ax.legend(
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )


def _short_strategy_label(strategy_name: str) -> str:
    return SHORT_STRATEGY_LABELS.get(str(strategy_name), str(strategy_name))


def _write_placeholder(output_path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.set_axis_off()
    _save(fig, output_path)


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final constrained TD3 figures.")
    parser.add_argument("--final-report-dir", default=DEFAULT_FINAL_REPORT_DIR)
    parser.add_argument("--regime-analysis-dir", default=DEFAULT_REGIME_ANALYSIS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = build_final_figures(
        final_report_dir=args.final_report_dir,
        regime_analysis_dir=args.regime_analysis_dir,
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
