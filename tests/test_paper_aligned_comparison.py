import unittest

import pandas as pd

from src.analysis.paper_aligned_comparison import (
    exact_common_index,
    normalize_history_dates,
)


class PaperAlignedComparisonTests(unittest.TestCase):
    def test_normalize_history_dates_sorts_weekly_dates(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-14", "2022-01-07"],
                "financial_net_return": [0.02, 0.01],
            }
        )
        normalized, timezone_kind = normalize_history_dates(frame, "fixture")
        self.assertEqual(timezone_kind, "naive")
        self.assertEqual(
            normalized["date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2022-01-07", "2022-01-14"],
        )

    def test_normalize_history_dates_rejects_duplicates(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-07", "2022-01-07"],
                "financial_net_return": [0.01, 0.02],
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate dates"):
            normalize_history_dates(frame, "fixture")

    def test_normalize_history_dates_rejects_non_friday(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-06"],
                "financial_net_return": [0.01],
            }
        )
        with self.assertRaisesRegex(ValueError, "Friday"):
            normalize_history_dates(frame, "fixture")

    def test_exact_common_index_requires_timestamp_equality(self) -> None:
        canonical = pd.DatetimeIndex(
            pd.to_datetime(["2022-01-07", "2022-01-14"]), name="date"
        )
        same = pd.DatetimeIndex(
            pd.to_datetime(["2022-01-07", "2022-01-14"]), name="date"
        )
        actual = exact_common_index(
            canonical,
            [same],
            expected_observations=2,
            label="fixture",
        )
        self.assertTrue(actual.equals(canonical))

    def test_exact_common_index_rejects_equal_length_different_dates(self) -> None:
        canonical = pd.DatetimeIndex(
            pd.to_datetime(["2022-01-07", "2022-01-14"]), name="date"
        )
        shifted = pd.DatetimeIndex(
            pd.to_datetime(["2022-01-14", "2022-01-21"]), name="date"
        )
        with self.assertRaisesRegex(ValueError, "expected 2 common dates"):
            exact_common_index(
                canonical,
                [shifted],
                expected_observations=2,
                label="fixture",
            )


if __name__ == "__main__":
    unittest.main()
