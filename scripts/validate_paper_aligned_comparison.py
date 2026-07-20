"""Validate all temporal and downstream dependencies in the paper package."""

from __future__ import annotations

import argparse

from src.analysis.paper_aligned_comparison import validate_output_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/paper_aligned_comparison")
    parser.add_argument("--expected-observations", type=int, default=228)
    args = parser.parse_args()
    validation = validate_output_directory(
        args.output_dir,
        expected_observations=args.expected_observations,
    )
    print(f"N strategies compared per protocol: {validation['n_strategies_compared_per_protocol']}")
    print(f"Common observations: {validation['common_observations']}")
    print(f"Common start date: {validation['common_start_date']}")
    print(f"Common end date: {validation['common_end_date']}")
    print(f"Frequency: {validation['frequency']}")
    print(f"All indices identical: {validation['all_indices_identical']}")
    print(f"Duplicate dates: {validation['duplicate_dates']}")
    print(f"Missing values: {validation['missing_values']}")
    print(
        "Metrics derived from aligned histories: "
        f"{validation['metrics_derived_from_aligned_histories']}"
    )
    print(
        "Ranking derived from aligned metrics: "
        f"{validation['ranking_derived_from_aligned_metrics']}"
    )
    print(
        "Mandates derived from aligned metrics: "
        f"{validation['mandates_derived_from_aligned_metrics']}"
    )
    print(f"Pareto derived from aligned metrics: {validation['pareto_derived_from_aligned_metrics']}")
    print(f"Metadata consistent with CSV: {validation['metadata_consistent_with_csv']}")


if __name__ == "__main__":
    main()
