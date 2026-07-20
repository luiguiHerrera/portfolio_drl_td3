"""CLI for the metrics-then-average paper comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.paper_seed_aggregated_comparison import (
    build_paper_seed_aggregated_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--aligned-dir",
        default="outputs/paper_aligned_comparison",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_seed_aggregated_comparison",
    )
    parser.add_argument("--expected-observations", type=int, default=228)
    args = parser.parse_args()
    result = build_paper_seed_aggregated_comparison(
        repo_root=Path(args.repo_root),
        external_root=Path(args.external_root),
        aligned_dir=Path(args.aligned_dir),
        output_dir=Path(args.output_dir),
        expected_observations=args.expected_observations,
    )
    validation = result["validation"]
    print(f"Output directory: {result['output_dir']}")
    for key, value in validation.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
