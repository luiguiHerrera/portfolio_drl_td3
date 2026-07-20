"""Validate the metrics-then-average paper comparison package."""

from __future__ import annotations

import argparse

from src.analysis.paper_seed_aggregated_comparison import (
    validate_seed_aggregated_output,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_seed_aggregated_comparison",
    )
    parser.add_argument("--expected-observations", type=int, default=228)
    args = parser.parse_args()
    validation = validate_seed_aggregated_output(
        args.output_dir,
        expected_observations=args.expected_observations,
    )
    for key, value in validation.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
