"""Tests for reproducibility utilities."""

import unittest

from src.utils.seed import set_seed


class SeedTests(unittest.TestCase):
    def test_set_seed_runs_without_error(self):
        set_seed(42)


if __name__ == "__main__":
    unittest.main()
