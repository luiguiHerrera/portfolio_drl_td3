"""Tests for portfolio reward functions."""

import unittest
from pathlib import Path

import numpy as np

from src.rewards.reward import (
    compute_net_return_reward,
    compute_risk_aware_reward,
    compute_turnover_penalty,
    concentration_penalty,
    drawdown_penalty,
)


class RewardTests(unittest.TestCase):
    def test_net_return_reward_still_works(self):
        self.assertAlmostEqual(compute_net_return_reward(0.05, 0.01), 0.04)

    def test_concentration_penalty_equal_weights(self):
        self.assertAlmostEqual(concentration_penalty(np.full(5, 0.2)), 0.2)

    def test_concentration_penalty_fully_concentrated(self):
        self.assertAlmostEqual(concentration_penalty(np.array([1.0, 0.0, 0.0])), 1.0)

    def test_concentration_penalty_rejects_negative_weights(self):
        with self.assertRaisesRegex(ValueError, "weights must be non-negative"):
            concentration_penalty(np.array([0.5, -0.1, 0.6]))

    def test_concentration_penalty_rejects_weights_not_summing_to_one(self):
        with self.assertRaisesRegex(ValueError, "weights must sum to 1"):
            concentration_penalty(np.array([0.2, 0.2, 0.2]))

    def test_drawdown_penalty_returns_zero_at_peak(self):
        self.assertAlmostEqual(drawdown_penalty(100000.0, 100000.0), 0.0)

    def test_drawdown_penalty_returns_positive_drawdown_below_peak(self):
        self.assertAlmostEqual(drawdown_penalty(90000.0, 100000.0), 0.1)

    def test_drawdown_penalty_returns_zero_above_peak(self):
        self.assertAlmostEqual(drawdown_penalty(110000.0, 100000.0), 0.0)

    def test_compute_risk_aware_reward_matches_expected_formula(self):
        weights = np.array([0.5, 0.5])
        reward = compute_risk_aware_reward(
            portfolio_return=0.04,
            transaction_cost=0.01,
            turnover=0.3,
            weights=weights,
            portfolio_value=90000.0,
            peak_portfolio_value=100000.0,
            reward_config={
                "lambda_return": 1.5,
                "lambda_transaction_cost": 0.2,
                "lambda_turnover": 0.1,
                "lambda_concentration": 0.4,
                "lambda_drawdown": 0.5,
            },
        )

        expected_reward = 1.5 * 0.04 - 0.2 * 0.01 - 0.1 * 0.3 - 0.4 * 0.5 - 0.5 * 0.1
        self.assertAlmostEqual(reward, expected_reward)

    def test_compute_risk_aware_reward_default_lambdas_equal_net_return_reward(self):
        reward = compute_risk_aware_reward(
            portfolio_return=0.04,
            transaction_cost=0.01,
            turnover=0.3,
            weights=np.array([0.5, 0.5]),
            portfolio_value=100000.0,
            peak_portfolio_value=100000.0,
            reward_config={},
        )

        self.assertAlmostEqual(reward, compute_net_return_reward(0.04, 0.01))

    def test_robust_score_and_dsr_are_not_training_reward_dependencies(self):
        forbidden_terms = (
            "compute_deflated_sharpe_ratio",
            "compute_composite_robust_score",
            "robust_score",
        )
        checked_roots = (
            Path("src/rewards"),
            Path("src/env"),
            Path("src/train"),
        )

        offenders = []
        for root in checked_roots:
            for source_path in sorted(root.rglob("*.py")):
                source_text = source_path.read_text(encoding="utf-8")
                for term in forbidden_terms:
                    if term in source_text:
                        offenders.append(f"{source_path}:{term}")

        self.assertEqual(offenders, [])

    def test_compute_risk_aware_reward_rejects_negative_lambda(self):
        with self.assertRaisesRegex(ValueError, "lambda_turnover"):
            compute_risk_aware_reward(
                portfolio_return=0.04,
                transaction_cost=0.01,
                turnover=0.3,
                weights=np.array([0.5, 0.5]),
                portfolio_value=100000.0,
                peak_portfolio_value=100000.0,
                reward_config={"lambda_turnover": -0.1},
            )

    def test_compute_turnover_penalty_modes(self):
        self.assertAlmostEqual(
            compute_turnover_penalty(0.3, 0.5, mode="linear"),
            0.15,
        )
        self.assertEqual(
            compute_turnover_penalty(0.3, 0.5, mode="none"),
            0.0,
        )
        self.assertAlmostEqual(
            compute_turnover_penalty(
                0.3,
                0.5,
                mode="excess_linear",
                free_band=0.1,
            ),
            0.1,
        )
        self.assertAlmostEqual(
            compute_turnover_penalty(
                0.3,
                0.5,
                mode="excess_quadratic",
                free_band=0.1,
                quadratic_weight=0.25,
            ),
            0.11,
        )

    def test_compute_turnover_penalty_rejects_invalid_inputs(self):
        invalid_calls = (
            {"turnover": True, "lambda_turnover": 0.1},
            {"turnover": -0.1, "lambda_turnover": 0.1},
            {"turnover": 0.1, "lambda_turnover": True},
            {"turnover": 0.1, "lambda_turnover": -0.1},
            {"turnover": 0.1, "lambda_turnover": 0.1, "mode": "bad"},
            {"turnover": 0.1, "lambda_turnover": 0.1, "free_band": -0.1},
            {
                "turnover": 0.1,
                "lambda_turnover": 0.1,
                "quadratic_weight": -0.1,
            },
        )

        for kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    compute_turnover_penalty(**kwargs)


if __name__ == "__main__":
    unittest.main()
