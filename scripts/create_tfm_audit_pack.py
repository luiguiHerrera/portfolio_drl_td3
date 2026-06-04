from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "data/raw",
    "data/processed",
    "node_modules",
}

TEXT_EXTS = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".ini",
    ".cfg",
    ".csv",
}

MAX_TEXT_BYTES = 220_000

SECRET_PATTERNS = [
    r"(?i)(api[_-]?key\s*=\s*)['\"][^'\"]+['\"]",
    r"(?i)(secret\s*=\s*)['\"][^'\"]+['\"]",
    r"(?i)(token\s*=\s*)['\"][^'\"]+['\"]",
    r"(?i)(password\s*=\s*)['\"][^'\"]+['\"]",
    r"(?i)(fred[_-]?api[_-]?key\s*=\s*)['\"][^'\"]+['\"]",
]

FINAL_CORRECTED_DIRS = [
    "final_corrected_limited_td3_60ep_10seeds",
    "final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds",
    "final_corrected_zero_cash_benchmark_comparison",
    "final_corrected_bil_cash_benchmark_comparison",
    "final_corrected_zero_cash_statistical_validation",
    "final_corrected_bil_cash_statistical_validation",
    "final_corrected_zero_cash_white_reality_check",
    "final_corrected_bil_cash_white_reality_check",
    "final_corrected_zero_cash_regime_analysis",
    "final_corrected_bil_cash_regime_analysis",
    "final_corrected_zero_cash_constraint_pareto",
    "final_corrected_bil_cash_constraint_pareto",
    "final_corrected_zero_cash_mandate_profile_comparison",
    "final_corrected_bil_cash_mandate_profile_comparison",
    "final_corrected_cash_robustness_comparison",
]

SUMMARY_NAME_HINTS = (
    "summary",
    "metadata",
    "ranking",
    "rankings",
    "winner",
    "winners",
    "best_caps",
    "all_results",
    "combined",
    "pairwise",
    "frontier",
    "pass_fail",
    "checks",
    "inventory",
    "key_results",
)

SKIP_NAME_HINTS = (
    "test_policy_history.csv",
    "train_policy_history.csv",
    "validation_policy_history.csv",
    "bootstrap_distribution.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a lightweight TFM audit pack with repo and final-corrected output summaries."
    )
    parser.add_argument(
        "--external-outputs-dir",
        default="~/Projects/portfolio_drl_outputs",
        help="External final-corrected outputs directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="tfm_audit_pack",
        help="Directory where audit pack files are written.",
    )
    parser.add_argument(
        "--zip-name",
        default="tfm_audit_pack.zip",
        help="Zip file name/path to create at the end.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip compileall and unittest commands.",
    )
    parser.add_argument(
        "--max-csv-rows",
        type=int,
        default=25,
        help="Maximum CSV data rows to include in markdown summaries.",
    )
    return parser.parse_args()


def redacted(text: str) -> str:
    for pat in SECRET_PATTERNS:
        text = re.sub(pat, r"\1'***REDACTED***'", text)
    return text


def run_command(cmd: list[str], timeout: int = 900) -> str:
    try:
        p = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return f"$ {' '.join(cmd)}\nreturncode={p.returncode}\n\n{redacted(p.stdout.strip())}"
    except Exception as exc:
        return f"$ {' '.join(cmd)}\n[ERROR: {exc}]"


def is_excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    parts = rel.split("/")
    for ex in EXCLUDE_DIRS:
        if rel == ex or rel.startswith(ex + "/"):
            return True
    return any(part in EXCLUDE_DIRS for part in parts)


def safe_read(path: Path) -> str:
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_TEXT_BYTES:
            prefix = f"[FILE TOO LARGE: {len(raw)} bytes. Showing first {MAX_TEXT_BYTES} bytes]\n\n"
            return prefix + redacted(raw[:MAX_TEXT_BYTES].decode("utf-8", errors="replace"))
        return redacted(raw.decode("utf-8", errors="replace"))
    except Exception as exc:
        return f"[Could not read file: {exc}]"


def csv_summary(path: Path, max_csv_rows: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            rows = []
            for i, row in enumerate(reader):
                rows.append(row)
                if i >= max_csv_rows:
                    break

        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            total_rows = sum(1 for _ in f)

        if not rows:
            return "[EMPTY CSV]"

        header = rows[0]
        body = rows[1:]
        out = [
            f"CSV rows including header: {total_rows}",
            f"Columns ({len(header)}): {header}",
            f"First {len(body)} data rows:",
        ]
        out.extend(str(r) for r in body)
        return "\n".join(out)
    except Exception as exc:
        return f"[CSV summary failed: {exc}]"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def collect_repo_files() -> list[tuple[str, int, Path]]:
    files = []
    for p in ROOT.rglob("*"):
        if p.is_file() and not is_excluded(p):
            rel = p.relative_to(ROOT).as_posix()
            files.append((rel, p.stat().st_size, p))
    return sorted(files, key=lambda x: x[0])


def is_summary_file(path: Path) -> bool:
    if path.suffix.lower() not in {".csv", ".md", ".json", ".txt"}:
        return False
    path_text = path.as_posix()
    if "/per_candidate/" in path_text:
        return False
    lower_name = path.name.lower()
    if any(hint in lower_name for hint in SKIP_NAME_HINTS):
        return False
    return any(hint in lower_name for hint in SUMMARY_NAME_HINTS)


def find_final_corrected_dirs(external_outputs_dir: Path) -> dict[str, Path]:
    candidates = [ROOT / "outputs" / "tables", external_outputs_dir]
    found: dict[str, Path] = {}
    for dirname in FINAL_CORRECTED_DIRS:
        for base in candidates:
            path = base / dirname
            if path.exists():
                found[dirname] = path
                break
    return found


def count_td3_histories(exp_dir: Path | None) -> int:
    if exp_dir is None or not exp_dir.exists():
        return 0
    return sum(1 for _ in exp_dir.glob("per_candidate/*/*/test_policy_history.csv"))


def count_benchmark_histories(benchmark_dir: Path | None) -> int:
    if benchmark_dir is None or not benchmark_dir.exists():
        return 0
    history_patterns = [
        "benchmarks/histories/*_history.csv",
        "benchmarks/*_history.csv",
        "histories/*_history.csv",
        "*_history.csv",
    ]
    paths: set[Path] = set()
    for pattern in history_patterns:
        paths.update(benchmark_dir.glob(pattern))
    return len(paths)


def cap_summary_rows(exp_dir: Path | None) -> int:
    if exp_dir is None:
        return 0
    rows = read_csv_rows(exp_dir / "cap_sensitivity_summary.csv")
    return len(rows)


def best_from_cap_summary(exp_dir: Path | None) -> dict[str, str] | None:
    if exp_dir is None:
        return None
    rows = read_csv_rows(exp_dir / "cap_sensitivity_summary.csv")
    if not rows:
        rows = read_csv_rows(exp_dir / "cap_sensitivity_best_caps.csv")
    if not rows:
        return None

    def score(row: dict[str, str]) -> float:
        for col in (
            "mandate_aware_score",
            "best_cap_mandate_aware_score",
            "best_mandate_aware_score",
            "mandate_score_or_available_score",
            "robust_score",
            "best_cap_robust_score",
            "best_robust_score",
            "sharpe",
        ):
            try:
                value = float(row.get(col, "nan"))
                if math.isfinite(value):
                    return value
            except ValueError:
                continue
        return float("-inf")

    return max(rows, key=score)


def cap_label_to_name(cap_value: str | None) -> str:
    if not cap_value:
        return "unknown"
    label = str(cap_value).strip()
    if label.lower() in {"uncapped", "none", "nan"}:
        return "uncapped"
    return label.replace(".", "p")


def winner_name(row: dict[str, str] | None) -> str | None:
    if not row:
        return None
    for col in ("candidate_name", "candidate", "strategy_name", "strategy"):
        if row.get(col):
            return row[col]
    base = row.get("base_candidate")
    cap = row.get("best_cap_by_mandate_aware_score") or row.get("best_by_mandate_aware_score")
    if base and cap:
        return f"{base}_cap_{cap_label_to_name(cap)}"
    return base


def find_metadata_files(exp_dir: Path | None) -> list[Path]:
    if exp_dir is None or not exp_dir.exists():
        return []
    return sorted(p for p in exp_dir.rglob("*.json") if "metadata" in p.name.lower())


def contains_cost_value(metadata_files: list[Path], asset: str, expected: float) -> bool | None:
    found_cost_field = False
    for path in metadata_files:
        data = read_json(path)
        if data is None:
            continue
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, value in item.items():
                    if key in {"asset_transaction_cost_bps", "transaction_cost_bps"} and isinstance(value, dict):
                        found_cost_field = True
                        if asset in value:
                            try:
                                if abs(float(value[asset]) - expected) < 1e-9:
                                    return True
                            except (TypeError, ValueError):
                                pass
                    stack.append(value)
            elif isinstance(item, list):
                stack.extend(item)
    return None if not found_cost_field else False


def check(name: str, actual: Any, expected: Any, passed: bool, detail: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
        "detail": detail,
    }


def build_audit_checks(found: dict[str, Path]) -> list[dict[str, Any]]:
    zero_td3 = found.get("final_corrected_limited_td3_60ep_10seeds")
    bil_td3 = found.get("final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds")
    zero_bench = found.get("final_corrected_zero_cash_benchmark_comparison")
    bil_bench = found.get("final_corrected_bil_cash_benchmark_comparison")

    zero_histories = count_td3_histories(zero_td3)
    bil_histories = count_td3_histories(bil_td3)
    zero_benchmark_histories = count_benchmark_histories(zero_bench)
    bil_benchmark_histories = count_benchmark_histories(bil_bench)
    zero_summary_rows = cap_summary_rows(zero_td3)
    bil_summary_rows = cap_summary_rows(bil_td3)
    zero_winner = best_from_cap_summary(zero_td3)
    bil_winner = best_from_cap_summary(bil_td3)

    zero_meta = find_metadata_files(zero_td3) + find_metadata_files(zero_bench)
    bil_meta = find_metadata_files(bil_td3) + find_metadata_files(bil_bench)
    zero_cash_cost = contains_cost_value(zero_meta, "CASH", 0.0)
    bil_cash_cost = contains_cost_value(bil_meta, "CASH", 2.0)
    zero_btc_cost = contains_cost_value(zero_meta, "BTC-USD", 10.0)
    bil_btc_cost = contains_cost_value(bil_meta, "BTC-USD", 10.0)

    checks = [
        check("zero_cash_td3_histories", zero_histories, 800, zero_histories == 800),
        check("bil_cash_td3_histories", bil_histories, 800, bil_histories == 800),
        check("zero_cash_benchmark_histories", zero_benchmark_histories, 14, zero_benchmark_histories == 14),
        check("bil_cash_benchmark_histories", bil_benchmark_histories, 14, bil_benchmark_histories == 14),
        check("zero_cash_cap_sensitivity_summary_rows", zero_summary_rows, 5, zero_summary_rows == 5),
        check("bil_cash_cap_sensitivity_summary_rows", bil_summary_rows, 5, bil_summary_rows == 5),
        check(
            "zero_cash_winner_captured",
            winner_name(zero_winner),
            "non-empty",
            bool(winner_name(zero_winner)),
        ),
        check(
            "bil_cash_winner_captured",
            winner_name(bil_winner),
            "non-empty",
            bool(winner_name(bil_winner)),
        ),
        check(
            "zero_cash_statistical_dir_exists",
            str(found.get("final_corrected_zero_cash_statistical_validation")),
            "exists",
            "final_corrected_zero_cash_statistical_validation" in found,
        ),
        check(
            "bil_cash_statistical_dir_exists",
            str(found.get("final_corrected_bil_cash_statistical_validation")),
            "exists",
            "final_corrected_bil_cash_statistical_validation" in found,
        ),
        check(
            "zero_cash_wrc_dir_exists",
            str(found.get("final_corrected_zero_cash_white_reality_check")),
            "exists",
            "final_corrected_zero_cash_white_reality_check" in found,
        ),
        check(
            "bil_cash_wrc_dir_exists",
            str(found.get("final_corrected_bil_cash_white_reality_check")),
            "exists",
            "final_corrected_bil_cash_white_reality_check" in found,
        ),
        check(
            "zero_cash_regime_dir_exists",
            str(found.get("final_corrected_zero_cash_regime_analysis")),
            "exists",
            "final_corrected_zero_cash_regime_analysis" in found,
        ),
        check(
            "bil_cash_regime_dir_exists",
            str(found.get("final_corrected_bil_cash_regime_analysis")),
            "exists",
            "final_corrected_bil_cash_regime_analysis" in found,
        ),
        check(
            "zero_cash_constraint_pareto_dir_exists",
            str(found.get("final_corrected_zero_cash_constraint_pareto")),
            "exists",
            "final_corrected_zero_cash_constraint_pareto" in found,
        ),
        check(
            "bil_cash_constraint_pareto_dir_exists",
            str(found.get("final_corrected_bil_cash_constraint_pareto")),
            "exists",
            "final_corrected_bil_cash_constraint_pareto" in found,
        ),
        check(
            "zero_cash_mandate_profile_dir_exists",
            str(found.get("final_corrected_zero_cash_mandate_profile_comparison")),
            "exists",
            "final_corrected_zero_cash_mandate_profile_comparison" in found,
        ),
        check(
            "bil_cash_mandate_profile_dir_exists",
            str(found.get("final_corrected_bil_cash_mandate_profile_comparison")),
            "exists",
            "final_corrected_bil_cash_mandate_profile_comparison" in found,
        ),
        check("zero_cash_has_cash_cost_0_bps", zero_cash_cost, True, zero_cash_cost is True),
        check("bil_cash_has_cash_cost_2_bps", bil_cash_cost, True, bil_cash_cost is True),
        check("zero_cash_has_btc_cost_10_bps", zero_btc_cost, True, zero_btc_cost is True),
        check("bil_cash_has_btc_cost_10_bps", bil_btc_cost, True, bil_btc_cost is True),
    ]
    return checks


def write_repo_snapshot(out_dir: Path, files_sorted: list[tuple[str, int, Path]], max_csv_rows: int) -> None:
    snapshot = [
        "# TFM Audit Snapshot",
        "",
        f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"Root: {ROOT}",
        "",
        "## 1. Git / environment",
        "",
    ]

    commands = {
        "pwd": ["pwd"],
        "python_version": [sys.executable, "--version"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "git_branch": ["git", "branch", "--show-current"],
        "git_status": ["git", "status", "--short"],
        "git_log_last_20": ["git", "log", "--oneline", "-n", "20"],
    }
    for name, cmd in commands.items():
        snapshot.extend([f"### {name}", "```text", run_command(cmd, timeout=180), "```", ""])

    snapshot.extend(["## 2. Project tree", "", "```text"])
    for rel, size, _ in files_sorted:
        snapshot.append(f"{rel}    ({size} bytes)")
    snapshot.extend(["```", "", "## 3. Key files content", ""])

    priority_prefixes = ["README", "requirements", "pyproject", "configs/", "src/", "tests/", "docs/"]
    for rel, _, path in files_sorted:
        include = (
            any(rel.startswith(prefix) for prefix in priority_prefixes)
            or rel in {"README.md", "requirements.txt", "environment.yml", "pyproject.toml"}
        )
        if not include or path.suffix.lower() not in TEXT_EXTS:
            continue
        snapshot.extend([f"### `{rel}`", ""])
        if path.suffix.lower() == ".csv":
            snapshot.extend(["```text", csv_summary(path, max_csv_rows), "```"])
        else:
            lang = "python" if path.suffix == ".py" else "text"
            snapshot.extend([f"```{lang}", safe_read(path), "```"])
        snapshot.append("")

    snapshot.extend(["## 4. Outputs / tables summaries", ""])
    for odir in [ROOT / "outputs", ROOT / "reports", ROOT / "results"]:
        if not odir.exists():
            continue
        for path in sorted(odir.rglob("*")):
            if path.is_file() and not is_excluded(path) and is_summary_file(path):
                rel = path.relative_to(ROOT).as_posix()
                snapshot.extend([f"### `{rel}`", "", "```text"])
                if path.suffix.lower() == ".csv":
                    snapshot.append(csv_summary(path, max_csv_rows))
                else:
                    snapshot.append(safe_read(path))
                snapshot.extend(["```", ""])

    (out_dir / "repo_audit_snapshot.md").write_text("\n".join(snapshot), encoding="utf-8")


def write_final_corrected_inventory(
    out_dir: Path,
    found: dict[str, Path],
    max_csv_rows: int,
) -> list[dict[str, Any]]:
    inventory = [
        "# Final Corrected Experiment Inventory",
        "",
        "This inventory includes top-level summaries and metadata only. Per-run histories are counted but not copied.",
        "",
    ]
    included_files: list[dict[str, Any]] = []

    for dirname in FINAL_CORRECTED_DIRS:
        path = found.get(dirname)
        inventory.append(f"## {dirname}")
        if path is None:
            inventory.extend(["", "Status: missing", ""])
            continue

        history_count = count_td3_histories(path)
        benchmark_count = count_benchmark_histories(path)
        inventory.extend(
            [
                "",
                f"Path: `{path}`",
                f"TD3 history count: {history_count}",
                f"Benchmark history count: {benchmark_count}",
                "",
            ]
        )

        summary_files = [p for p in sorted(path.rglob("*")) if p.is_file() and is_summary_file(p)]
        if not summary_files:
            inventory.extend(["No top-level summary/metadata files found.", ""])
            continue

        inventory.append("Included summaries:")
        for summary_path in summary_files:
            rel = summary_path.relative_to(path).as_posix()
            inventory.append(f"- `{rel}`")
            included_files.append(
                {
                    "experiment": dirname,
                    "source_path": str(summary_path),
                    "relative_path": rel,
                    "size_bytes": summary_path.stat().st_size,
                }
            )
        inventory.append("")

        for summary_path in summary_files:
            rel = summary_path.relative_to(path).as_posix()
            inventory.extend([f"### `{dirname}/{rel}`", "", "```text"])
            if summary_path.suffix.lower() == ".csv":
                inventory.append(csv_summary(summary_path, max_csv_rows))
            else:
                inventory.append(safe_read(summary_path))
            inventory.extend(["```", ""])

    (out_dir / "final_corrected_experiment_inventory.md").write_text(
        "\n".join(inventory),
        encoding="utf-8",
    )
    return included_files


def top_strategy_from_csv(path: Path, strategy_cols: tuple[str, ...], score_cols: tuple[str, ...]) -> str:
    rows = read_csv_rows(path)
    if not rows:
        return "not available"

    def row_score(row: dict[str, str]) -> float:
        for col in score_cols:
            if col in row:
                try:
                    value = float(row[col])
                    if math.isfinite(value):
                        return value
                except ValueError:
                    continue
        return float("-inf")

    best = max(rows, key=row_score)
    for col in strategy_cols:
        if best.get(col):
            return best[col]
    return str(best)


def write_key_results(out_dir: Path, found: dict[str, Path]) -> None:
    zero_td3 = found.get("final_corrected_limited_td3_60ep_10seeds")
    bil_td3 = found.get("final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds")
    zero_bench = found.get("final_corrected_zero_cash_benchmark_comparison")
    bil_bench = found.get("final_corrected_bil_cash_benchmark_comparison")

    lines = [
        "# Final Corrected Key Results",
        "",
        "These are captured from the available final-corrected summary files. They are audit evidence, not regenerated experiments.",
        "",
        "## TD3-only cap sensitivity",
        "",
    ]

    for label, path in [("Zero-CASH", zero_td3), ("BIL-CASH", bil_td3)]:
        if path is None:
            lines.append(f"- {label}: not available")
            continue
        best = best_from_cap_summary(path)
        if best is None:
            lines.append(f"- {label}: summary present but winner not available")
            continue
        strategy = winner_name(best) or "unknown"
        mandate = (
            best.get("mandate_aware_score")
            or best.get("best_cap_mandate_aware_score")
            or best.get("best_mandate_aware_score")
            or "n/a"
        )
        robust = best.get("robust_score") or best.get("best_cap_robust_score") or best.get("best_robust_score") or "n/a"
        lines.append(f"- {label}: `{strategy}`; mandate-aware `{mandate}`; robust `{robust}`")

    lines.extend(["", "## TD3 vs benchmark combined rankings", ""])
    combined_specs = [
        ("Zero-CASH", zero_bench, "final_corrected_zero_cash_combined_ranking.csv"),
        ("BIL-CASH", bil_bench, "final_corrected_bil_cash_combined_ranking.csv"),
    ]
    for label, base, filename in combined_specs:
        if base is None:
            lines.append(f"- {label}: comparison directory missing")
            continue
        top = top_strategy_from_csv(
            base / filename,
            ("strategy_name", "candidate_name", "strategy"),
            ("mandate_aware_score", "mandate_score_or_available_score", "robust_score", "sharpe"),
        )
        lines.append(f"- {label}: top combined ranking `{top}`")

    lines.extend(["", "## Statistical and WRC outputs", ""])
    for label, dirname in [
        ("Zero-CASH statistical validation", "final_corrected_zero_cash_statistical_validation"),
        ("BIL-CASH statistical validation", "final_corrected_bil_cash_statistical_validation"),
        ("Zero-CASH White Reality Check", "final_corrected_zero_cash_white_reality_check"),
        ("BIL-CASH White Reality Check", "final_corrected_bil_cash_white_reality_check"),
    ]:
        path = found.get(dirname)
        status = "available" if path is not None else "missing"
        lines.append(f"- {label}: {status}")

    lines.extend(["", "## Reporting layers", ""])
    for label, dirname in [
        ("Zero-CASH regime analysis", "final_corrected_zero_cash_regime_analysis"),
        ("BIL-CASH regime analysis", "final_corrected_bil_cash_regime_analysis"),
        ("Zero-CASH constraint/Pareto", "final_corrected_zero_cash_constraint_pareto"),
        ("BIL-CASH constraint/Pareto", "final_corrected_bil_cash_constraint_pareto"),
        ("Zero-CASH mandate profile", "final_corrected_zero_cash_mandate_profile_comparison"),
        ("BIL-CASH mandate profile", "final_corrected_bil_cash_mandate_profile_comparison"),
        ("Cash robustness comparison", "final_corrected_cash_robustness_comparison"),
    ]:
        path = found.get(dirname)
        status = "available" if path is not None else "missing"
        lines.append(f"- {label}: {status}")

    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "The audit pack intentionally excludes bulk per-run histories. It records their counts and includes top-level summaries/metadata for traceability.",
        ]
    )
    (out_dir / "final_corrected_key_results.md").write_text("\n".join(lines), encoding="utf-8")


def write_file_index(
    out_dir: Path,
    repo_files: list[tuple[str, int, Path]],
    external_files: list[dict[str, Any]],
) -> None:
    file_index = [{"path": rel, "size_bytes": size, "source": "repo"} for rel, size, _ in repo_files]
    for item in external_files:
        file_index.append({**item, "source": "external_summary"})
    (out_dir / "file_index.json").write_text(json.dumps(file_index, indent=2), encoding="utf-8")


def write_test_outputs(out_dir: Path, skip_tests: bool) -> None:
    if skip_tests:
        skipped = "Skipped by --skip-tests."
        (out_dir / "compileall_output.txt").write_text(skipped, encoding="utf-8")
        (out_dir / "unittest_output.txt").write_text(skipped, encoding="utf-8")
        return

    compile_cmd = [sys.executable, "-m", "compileall", "-q", "src", "tests"]
    unittest_cmd = [sys.executable, "-m", "unittest", "discover", "tests"]
    (out_dir / "compileall_output.txt").write_text(run_command(compile_cmd), encoding="utf-8")
    (out_dir / "unittest_output.txt").write_text(run_command(unittest_cmd, timeout=3600), encoding="utf-8")


def make_zip(out_dir: Path, zip_name: str, files_sorted: list[tuple[str, int, Path]]) -> Path:
    zip_path = Path(zip_name).expanduser()
    if not zip_path.is_absolute():
        zip_path = ROOT / zip_path

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(ROOT))

        for rel, size, path in files_sorted:
            if size > MAX_TEXT_BYTES or path.suffix.lower() not in TEXT_EXTS:
                continue
            if (
                rel.startswith("src/")
                or rel.startswith("tests/")
                or rel.startswith("configs/")
                or rel.startswith("docs/")
                or rel == "scripts/create_tfm_audit_pack.py"
                or rel in {"README.md", "requirements.txt", "pyproject.toml", "environment.yml"}
            ):
                z.write(path, path.relative_to(ROOT))
    return zip_path


def main() -> None:
    args = parse_args()
    external_outputs_dir = Path(args.external_outputs_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files_sorted = collect_repo_files()
    found = find_final_corrected_dirs(external_outputs_dir)

    write_repo_snapshot(out_dir, files_sorted, args.max_csv_rows)
    external_files = write_final_corrected_inventory(out_dir, found, args.max_csv_rows)
    write_key_results(out_dir, found)

    checks = build_audit_checks(found)
    (out_dir / "audit_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    write_test_outputs(out_dir, args.skip_tests)
    write_file_index(out_dir, files_sorted, external_files)
    zip_path = make_zip(out_dir, args.zip_name, files_sorted)

    passed = sum(1 for item in checks if item["passed"])
    failed = len(checks) - passed
    print("Audit pack created:")
    print(zip_path)
    print("")
    print("Main files:")
    for name in [
        "repo_audit_snapshot.md",
        "final_corrected_experiment_inventory.md",
        "final_corrected_key_results.md",
        "audit_checks.json",
        "unittest_output.txt",
        "compileall_output.txt",
        "file_index.json",
    ]:
        print(out_dir / name)
    print("")
    print(f"Audit checks: {passed} passed, {failed} failed")


if __name__ == "__main__":
    main()
