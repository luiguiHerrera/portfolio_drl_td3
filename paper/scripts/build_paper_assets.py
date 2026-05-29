from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
FINAL_DIR = (
    ROOT
    / "outputs/tables/final_constrained_td3_report_with_v3_clean_no_dxy_v7_clean_garch_v4_v7_v8_60ep_10seeds"
)
V3_CAP_DIR = (
    ROOT
    / "outputs/tables/cap_sensitivity_experiment_v3_clean_no_dxy_60ep_10seeds_replication"
)
V7_CAP_DIR = (
    ROOT
    / "outputs/tables/cap_sensitivity_experiment_v7_clean_no_dxy_garch_60ep_10seeds"
)


def short_name(name: str) -> str:
    replacements = {
        "V3_real_macro_vintage_clean_no_dxy_cap_0.50": "V3 clean no-DXY\ncap 0.50",
        "V3_real_macro_vintage_clean_no_dxy_cap_0.60": "V3 clean no-DXY\ncap 0.60",
        "V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50": "V7 clean no-DXY\n+ GARCH cap 0.50",
        "V3_real_macro_vintage_cap_0.50": "V3 vintage\ncap 0.50",
        "V3_cap_0.60": "V3 current\ncap 0.60",
        "V4_cap_0.50": "V4 GARCH\ncap 0.50",
        "BuyHold_GLD": "BuyHold GLD",
        "trend_spy_cash_12p": "Trend SPY/CASH",
        "rolling_markowitz_min_variance_52p": "Rolling min-var",
    }
    return replacements.get(name, name.replace("_", " "))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def build_top_scores() -> None:
    ranking = pd.read_csv(FINAL_DIR / "final_constrained_td3_mandate_ranking.csv")
    top = ranking.head(8).copy()
    top["label"] = top["strategy_name"].map(short_name)
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = range(len(top))
    ax.barh([i - 0.18 for i in y], top["mandate_aware_score"], height=0.34, label="Mandate-aware score", color="#2f6f73")
    ax.barh([i + 0.18 for i in y], top["robust_score"], height=0.34, label="Robust score", color="#d08c45")
    ax.set_yticks(list(y), top["label"])
    ax.set_xlim(0, max(top["robust_score"].max(), top["mandate_aware_score"].max()) + 0.08)
    ax.set_xlabel("Score")
    ax.set_title("Top strategies under the mandate-aware ranking")
    ax.legend(loc="lower right")
    save(fig, "paper_top_scores.png")


def build_constraint_map() -> None:
    ranking = pd.read_csv(FINAL_DIR / "final_constrained_td3_mandate_ranking.csv")
    keep = ranking.head(22).copy()
    colors = keep["strategy_type"].map({"td3": "#2f6f73", "benchmark": "#d08c45"}).fillna("#5b6770")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(
        keep["average_max_weight"],
        keep["max_drawdown"],
        s=60 + 180 * keep["mandate_aware_score"].clip(lower=0),
        c=colors,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.8,
    )
    leader = keep.iloc[0]
    ax.scatter([leader["average_max_weight"]], [leader["max_drawdown"]], s=320, facecolors="none", edgecolors="#1b1b1b", linewidth=1.6)
    ax.annotate(
        "Leading TD3",
        xy=(leader["average_max_weight"], leader["max_drawdown"]),
        xytext=(leader["average_max_weight"] + 0.06, leader["max_drawdown"] + 0.012),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    ax.axvline(0.50, color="#555555", linestyle="--", linewidth=1.0, label="0.50 weight cap")
    ax.axhline(-0.20, color="#8a3ffc", linestyle=":", linewidth=1.1, label="Clean mandate drawdown line")
    ax.set_xlabel("Average maximum asset weight")
    ax.set_ylabel("Maximum drawdown")
    ax.set_title("Concentration and drawdown determine mandate quality")
    ax.legend(loc="lower left")
    save(fig, "paper_constraint_map.png")


def build_cap_sensitivity() -> None:
    v3 = pd.read_csv(V3_CAP_DIR / "cap_sensitivity_summary.csv").iloc[0]
    v7 = pd.read_csv(V7_CAP_DIR / "cap_sensitivity_summary.csv").iloc[0]
    rows = pd.DataFrame(
        [
            {
                "candidate": "V3 clean no-DXY",
                "uncapped": v3["uncapped_mandate_aware_score"],
                "best": v3["best_cap_mandate_aware_score"],
                "cap": v3["best_cap_by_mandate_aware_score"],
            },
            {
                "candidate": "V7 clean no-DXY + GARCH",
                "uncapped": v7["uncapped_mandate_aware_score"],
                "best": v7["best_cap_mandate_aware_score"],
                "cap": v7["best_cap_by_mandate_aware_score"],
            },
        ]
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = range(len(rows))
    ax.bar([i - 0.18 for i in x], rows["uncapped"], width=0.34, label="Uncapped", color="#9aa0a6")
    ax.bar([i + 0.18 for i in x], rows["best"], width=0.34, label="Best capped", color="#2f6f73")
    ax.set_xticks(list(x), rows["candidate"])
    ax.set_ylabel("Mandate-aware score")
    ax.set_title("Max-weight caps materially change policy quality")
    for i, row in rows.iterrows():
        ax.text(i + 0.18, row["best"] + 0.015, f"cap={row['cap']:.2f}", ha="center", va="bottom", fontsize=9)
    ax.legend()
    save(fig, "paper_cap_sensitivity.png")


def build_benchmark_gap() -> None:
    ranking = pd.read_csv(FINAL_DIR / "final_constrained_td3_mandate_ranking.csv")
    names = [
        "V3_real_macro_vintage_clean_no_dxy_cap_0.50",
        "BuyHold_GLD",
        "trend_spy_cash_12p",
        "rolling_markowitz_min_variance_52p",
        "defensive_risk_off_12p",
    ]
    data = ranking[ranking["strategy_name"].isin(names)].copy()
    data["label"] = data["strategy_name"].map(short_name)
    data = data.sort_values("mandate_aware_score", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(data["label"], data["mandate_aware_score"], color="#2f6f73")
    ax.set_xlabel("Mandate-aware score")
    ax.set_title("Leading TD3 versus clean benchmark references")
    for _, row in data.iterrows():
        ax.text(row["mandate_aware_score"] + 0.012, row["label"], f"{row['mandate_aware_score']:.3f}", va="center", fontsize=9)
    save(fig, "paper_benchmark_gap.png")


def main() -> None:
    setup_style()
    build_top_scores()
    build_constraint_map()
    build_cap_sensitivity()
    build_benchmark_gap()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
