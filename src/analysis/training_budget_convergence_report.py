"""Training-budget convergence robustness report.

This module aggregates an already-run convergence screening grid. It does not
train TD3 by itself. The companion script
``scripts/run_training_budget_convergence_check.py`` performs the limited,
explicit training runs and then calls this reporting layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "~/Projects/portfolio_drl_outputs/final_corrected_training_budget_convergence"
EPISODE_BUDGETS = [30, 60, 100, 150]
BASE_EPISODES = 60
MATERIAL_SHARPE_DELTA = 0.10
MATERIAL_DRAWDOWN_DELTA = 0.03
MATERIAL_TURNOVER_RELATIVE_DELTA = 0.25


def build_training_budget_convergence_report(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Aggregate convergence case summaries and write report files."""
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    case_summaries = load_case_summaries(output_path)
    if case_summaries.empty:
        raise FileNotFoundError(f"No training_budget_case_summary.csv files found under {output_path}")

    all_results = add_seed_dispersion(case_summaries, output_path)
    convergence_summary = build_convergence_summary(all_results)
    by_candidate = build_by_candidate_summary(convergence_summary)
    metadata = build_metadata(output_path, all_results, convergence_summary)
    markdown = build_summary_markdown(convergence_summary, by_candidate, metadata)

    paths = {
        "all_results": output_path / "training_budget_convergence_all_results.csv",
        "summary": output_path / "training_budget_convergence_summary.csv",
        "by_candidate": output_path / "training_budget_convergence_by_candidate.csv",
        "metadata": output_path / "training_budget_convergence_metadata.json",
        "markdown": output_path / "training_budget_convergence_summary.md",
    }
    all_results.to_csv(paths["all_results"], index=False)
    convergence_summary.to_csv(paths["summary"], index=False)
    by_candidate.to_csv(paths["by_candidate"], index=False)
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["markdown"].write_text(markdown, encoding="utf-8")

    return {
        "all_results": all_results,
        "summary": convergence_summary,
        "by_candidate": by_candidate,
        "metadata": metadata,
        "markdown": markdown,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def load_case_summaries(output_path: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(output_path.glob("cases/*/*/training_budget_case_summary.csv")):
        frame = pd.read_csv(path)
        frame["case_summary_path"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def add_seed_dispersion(case_summaries: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    for _, row in case_summaries.iterrows():
        result = row.to_dict()
        seed_path = Path(str(row["case_summary_path"])).parent / "seed_level_aggregate_by_strategy_split.csv"
        if seed_path.exists():
            seed_level = pd.read_csv(seed_path)
            seed_test = seed_level[seed_level["split"].astype(str) == "test"].copy()
            for source_col, output_col in [
                ("mean_sharpe", "seed_std_sharpe"),
                ("mean_max_drawdown", "seed_std_max_drawdown"),
                ("mean_average_turnover", "seed_std_average_turnover"),
            ]:
                if source_col in seed_test.columns:
                    result[output_col] = pd.to_numeric(seed_test[source_col], errors="coerce").std(ddof=0)
        rows.append(result)
    return pd.DataFrame(rows)


def build_convergence_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["cash_assumption", "base_candidate", "cap_label"]
    for keys, group in all_results.groupby(group_columns, dropna=False):
        baseline = group[group["episodes"] == BASE_EPISODES]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        for _, row in group.sort_values("episodes").iterrows():
            rows.append(compare_to_baseline(keys, row, base))
    return pd.DataFrame(rows)


def compare_to_baseline(keys: tuple[Any, Any, Any], row: pd.Series, base: pd.Series) -> dict[str, Any]:
    cash_assumption, base_candidate, cap_label = keys
    episodes = int(row.get("episodes"))
    is_longer_budget = episodes > BASE_EPISODES
    sharpe_delta = _num(row.get("sharpe")) - _num(base.get("sharpe"))
    drawdown_delta = _num(row.get("max_drawdown")) - _num(base.get("max_drawdown"))
    turnover_base = _num(base.get("average_turnover"))
    turnover_delta = _num(row.get("average_turnover")) - turnover_base
    turnover_relative_delta = turnover_delta / abs(turnover_base) if abs(turnover_base) > 1e-12 else 0.0
    longer_improves_sharpe = is_longer_budget and sharpe_delta > MATERIAL_SHARPE_DELTA
    longer_degrades_sharpe = is_longer_budget and sharpe_delta < -MATERIAL_SHARPE_DELTA
    longer_increases_turnover = is_longer_budget and turnover_relative_delta > MATERIAL_TURNOVER_RELATIVE_DELTA
    longer_worsens_drawdown = is_longer_budget and drawdown_delta < -MATERIAL_DRAWDOWN_DELTA
    flags = {
        "material_sharpe_change": abs(sharpe_delta) > MATERIAL_SHARPE_DELTA,
        "material_drawdown_change": abs(drawdown_delta) > MATERIAL_DRAWDOWN_DELTA,
        "material_turnover_change": abs(turnover_relative_delta) > MATERIAL_TURNOVER_RELATIVE_DELTA,
    }
    return {
        "cash_assumption": cash_assumption,
        "base_candidate": base_candidate,
        "cap_label": cap_label,
        "candidate_name": row.get("candidate_name"),
        "episodes": episodes,
        "annualized_return": row.get("annualized_return"),
        "annualized_volatility": row.get("annualized_volatility"),
        "sharpe": row.get("sharpe"),
        "max_drawdown": row.get("max_drawdown"),
        "average_turnover": row.get("average_turnover"),
        "average_effective_number_of_assets": row.get("average_effective_number_of_assets"),
        "average_max_weight": row.get("average_max_weight"),
        "mean_cash_weight": row.get("mean_cash_weight"),
        "mean_btc_weight": row.get("mean_btc_weight"),
        "robust_score": row.get("robust_score"),
        "mandate_aware_score": row.get("mandate_aware_score"),
        "seed_std_sharpe": row.get("seed_std_sharpe"),
        "seed_std_max_drawdown": row.get("seed_std_max_drawdown"),
        "delta_sharpe_vs_60": sharpe_delta,
        "delta_max_drawdown_vs_60": drawdown_delta,
        "delta_turnover_vs_60": turnover_delta,
        "relative_delta_turnover_vs_60": turnover_relative_delta,
        "longer_training_improves_sharpe_materially": longer_improves_sharpe,
        "longer_training_degrades_sharpe_materially": longer_degrades_sharpe,
        "longer_training_increases_turnover_materially": longer_increases_turnover,
        "longer_training_worsens_drawdown_materially": longer_worsens_drawdown,
        "sixty_episode_undertraining_evidence": longer_improves_sharpe,
        **flags,
        "material_change": any(flags.values()),
    }


def build_by_candidate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["cash_assumption", "base_candidate", "cap_label"]
    for keys, group in summary.groupby(group_columns, dropna=False):
        non_base = group[group["episodes"] != BASE_EPISODES]
        base = group[group["episodes"] == BASE_EPISODES].iloc[0]
        best_sharpe = group.sort_values("sharpe", ascending=False).iloc[0]
        material = bool(non_base["material_change"].any()) if not non_base.empty else False
        longer = group[group["episodes"] > BASE_EPISODES].copy()
        shorter = group[group["episodes"] < BASE_EPISODES].copy()
        longer_improves = _any_true(longer, "longer_training_improves_sharpe_materially")
        longer_degrades = _any_true(longer, "longer_training_degrades_sharpe_materially")
        longer_turnover_rises = _any_true(longer, "longer_training_increases_turnover_materially")
        longer_drawdown_worsens = _any_true(longer, "longer_training_worsens_drawdown_materially")
        shorter_improves = (
            bool((shorter["delta_sharpe_vs_60"] > MATERIAL_SHARPE_DELTA).any()) if not shorter.empty else False
        )
        undertraining_evidence = longer_improves
        if undertraining_evidence:
            conclusion = "60_episodes_potentially_undertrained"
        elif longer_degrades or longer_turnover_rises or longer_drawdown_worsens:
            conclusion = "longer_training_degrades_or_destabilizes"
        elif shorter_improves or material:
            conclusion = "budget_sensitive_but_not_undertrained"
        elif material:
            conclusion = "mixed_budget_sensitivity"
        else:
            conclusion = "60_episodes_appears_adequate"
        rows.append(
            {
                "cash_assumption": keys[0],
                "base_candidate": keys[1],
                "cap_label": keys[2],
                "candidate_name": base["candidate_name"],
                "baseline_60_sharpe": base["sharpe"],
                "best_episode_budget_by_sharpe": int(best_sharpe["episodes"]),
                "best_sharpe": best_sharpe["sharpe"],
                "n_material_changes_vs_60": int(non_base["material_change"].sum()) if not non_base.empty else 0,
                "longer_training_improves_sharpe_materially": longer_improves,
                "longer_training_degrades_sharpe_materially": longer_degrades,
                "longer_training_increases_turnover_materially": longer_turnover_rises,
                "longer_training_destabilizes_drawdown_materially": longer_drawdown_worsens,
                "sixty_episode_undertraining_evidence": undertraining_evidence,
                "conclusion": conclusion,
                "conclusion_refined": _refined_conclusion_text(conclusion),
            }
        )
    return pd.DataFrame(rows)


def build_metadata(output_path: Path, all_results: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    return {
        "report_type": "training_budget_convergence_robustness",
        "reporting_only": True,
        "creates_new_final_winners": False,
        "output_dir": str(output_path),
        "episode_budgets": EPISODE_BUDGETS,
        "baseline_episode_budget": BASE_EPISODES,
        "seeds": [7, 21, 42, 84, 101],
        "folds": 4,
        "selected_candidate_caps": all_results[
            ["cash_assumption", "base_candidate", "cap_label"]
        ].drop_duplicates().to_dict(orient="records"),
        "n_case_rows": int(len(all_results)),
        "completed_histories": int(all_results.get("completed_test_histories", pd.Series(dtype=float)).sum()),
        "thresholds": {
            "material_sharpe_delta": MATERIAL_SHARPE_DELTA,
            "material_drawdown_delta_abs": MATERIAL_DRAWDOWN_DELTA,
            "material_turnover_relative_delta": MATERIAL_TURNOVER_RELATIVE_DELTA,
        },
        "material_change_rows": int(summary["material_change"].sum()) if "material_change" in summary else 0,
        "sixty_episode_undertraining_evidence_rows": (
            int(summary["sixty_episode_undertraining_evidence"].sum())
            if "sixty_episode_undertraining_evidence" in summary
            else 0
        ),
        "validation_notes": [
            "This is a convergence robustness check, not a new model-selection layer.",
            "The main corrected protocol is not overwritten.",
            "Longer-budget degradation is tracked separately from evidence that 60 episodes is undertrained.",
        ],
    }


def build_summary_markdown(summary: pd.DataFrame, by_candidate: pd.DataFrame, metadata: dict[str, Any]) -> str:
    lines = [
        "# Training-Budget Convergence Robustness",
        "",
        "This report compares selected final-corrected candidate/cap pairs across 30, 60, 100, and 150 training episodes.",
        "",
        "It is a robustness check, not a new model-selection layer.",
        "",
        "## Candidate Conclusions",
        "",
    ]
    for _, row in by_candidate.iterrows():
        lines.append(
            f"- `{row['candidate_name']}` ({row['cash_assumption']}): "
            f"best Sharpe budget `{row['best_episode_budget_by_sharpe']}`, "
            f"baseline 60 Sharpe `{_fmt(row['baseline_60_sharpe'])}`, "
            f"conclusion `{row['conclusion']}`. "
            f"{row.get('conclusion_refined', '')}"
        )
    overall = decide_overall_conclusion(by_candidate)
    lines.extend(
        [
            "",
            "## Overall Conclusion",
            "",
            overall,
            "",
            "## Materiality Thresholds",
            "",
            f"- Sharpe change greater than `{MATERIAL_SHARPE_DELTA}` versus 60 episodes.",
            f"- Max drawdown change greater than `{MATERIAL_DRAWDOWN_DELTA}` absolute versus 60 episodes.",
            f"- Turnover change greater than `{MATERIAL_TURNOVER_RELATIVE_DELTA:.0%}` versus 60 episodes.",
        ]
    )
    return "\n".join(lines)


def decide_overall_conclusion(by_candidate: pd.DataFrame) -> str:
    conclusions = set(by_candidate.get("conclusion", pd.Series(dtype=str)).astype(str))
    if "sixty_episode_undertraining_evidence" in by_candidate and bool(
        by_candidate["sixty_episode_undertraining_evidence"].fillna(False).astype(bool).any()
    ):
        return "At least one selected candidate shows material Sharpe improvement at longer budgets; a longer 10-seed convergence check should be considered before changing the main protocol."
    if "60_episodes_potentially_undertrained" in conclusions:
        return "At least one selected candidate shows material Sharpe improvement at longer budgets; a longer 10-seed convergence check should be considered before changing the main protocol."
    if conclusions == {"60_episodes_appears_adequate"}:
        return "60 episodes appears adequate for the corrected limited protocol."
    return (
        "The 5-seed convergence check does not support rerunning the main protocol with longer training "
        "budgets. The 60-episode budget is not obviously undertrained. Longer budgets often reduce Sharpe "
        "or increase turnover. A 10-seed extension can be treated as optional publication-grade confirmation, "
        "not as a prerequisite for the TFM narrative."
    )


def _any_true(frame: pd.DataFrame, column: str) -> bool:
    return bool(column in frame and frame[column].fillna(False).astype(bool).any())


def _refined_conclusion_text(conclusion: str) -> str:
    if conclusion == "60_episodes_potentially_undertrained":
        return "A longer budget materially improves Sharpe versus 60 episodes."
    if conclusion == "longer_training_degrades_or_destabilizes":
        return "Longer budgets create degradation or instability, not evidence that 60 episodes is undertrained."
    if conclusion == "budget_sensitive_but_not_undertrained":
        return "Budget sensitivity is present, but it does not point to longer-training underfit."
    if conclusion == "60_episodes_appears_adequate":
        return "No material evidence suggests that 60 episodes is undertrained."
    return "Budget evidence is mixed but does not automatically imply undertraining."


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training-budget convergence report.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_training_budget_convergence_report(output_dir=args.output_dir)
    print("Training-budget convergence report written:")
    for path in report["paths"].values():
        print(path)


if __name__ == "__main__":
    main()
