"""Tests for configuration loading and minimal validation."""

import unittest

from src.utils.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_returns_dict(self):
        config = load_config("configs/config.yaml")

        self.assertIsInstance(config, dict)

    def test_config_assets_include_cash(self):
        config = load_config("configs/config.yaml")

        self.assertIn("CASH", config["data"]["assets"])

    def test_config_assets_not_empty(self):
        config = load_config("configs/config.yaml")

        self.assertTrue(config["data"]["assets"])


if __name__ == "__main__":
    unittest.main()
