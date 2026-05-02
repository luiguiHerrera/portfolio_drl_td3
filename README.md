# Portfolio DRL TD3

## Overview

This repository is an academic implementation project for a portfolio allocation
agent based on Twin Delayed Deep Deterministic Policy Gradient (TD3). The goal is
to build the main reinforcement learning components from scratch in PyTorch,
with a modular structure that supports inspection, testing, and methodological
discussion.

The project is currently in an early scaffolding stage. The repository defines
the intended architecture and module boundaries, but it does not yet provide a
working TD3 training pipeline or validated investment results.

## Research Objective

The research objective is to study whether a continuous-control reinforcement
learning agent can learn dynamic portfolio allocation policies under realistic
portfolio constraints and transaction frictions.

The implementation is intended for a Master's thesis context, with emphasis on:

- academic explainability of each model component;
- robustness under chronological out-of-sample validation;
- explicit control of overfitting risk;
- comparison against classical portfolio construction benchmarks;
- reproducibility through configuration, deterministic seeds, and documented
  experiments.

Stable-Baselines3 is intentionally not used at the initial stage. The TD3 agent
will be implemented directly in PyTorch to support understanding, auditability,
and academic defense.

## Initial Asset Universe

The initial universe is configured as:

- `SPY`: U.S. equity market exposure;
- `TLT`: long-duration U.S. Treasury exposure;
- `GLD`: gold exposure;
- `BTC-USD`: Bitcoin exposure.
- `CASH`: synthetic cash allocation with zero return in the initial implementation.


This universe is intentionally compact. It combines traditional assets and a
high-volatility digital asset while keeping the first implementation tractable. 

## Portfolio Constraints

The initial portfolio setting is:

- long-only;
- fully invested across the selected asset universe, including the synthetic cash component;
- no short selling;
- no leverage;
- portfolio weights must be non-negative;
- portfolio weights must sum to one.

The actor action will represent portfolio weights. A planned design is for the
actor to output logits or continuous signals, then transform them into valid
weights through a softmax operation.

## TD3 Architecture

The planned TD3 implementation includes:

- an actor network that maps market states to portfolio allocation signals;
- twin critic networks estimating `Q(s, a)` to reduce overestimation bias;
- target networks for the actor and critics;
- soft target updates controlled by `tau`;
- delayed policy updates;
- target policy smoothing noise;
- an off-policy replay buffer;
- a training loop implemented directly in PyTorch.

The current files under `src/models/` and `src/memory/` are architectural
scaffolds only. They document responsibilities but do not yet implement TD3
logic.

## Reward Function

The planned reward design combines portfolio performance and risk-control terms:

```text
reward =
    net return
    + dynamic EWMA Sharpe component
    - drawdown penalty
    - transaction cost penalty
    - turnover penalty
```

The exact reward formula, scaling, and diagnostic decomposition will be defined
and tested before training the agent. Reward components should remain
interpretable so that the learned behavior can be analyzed beyond final return
statistics.

## Validation Strategy

Validation will follow time-series constraints. Random shuffling is not
appropriate for this setting.

The planned strategy includes:

- chronological train, validation, and test splits;
- walk-forward validation;
- out-of-sample evaluation;
- sensitivity analysis across seeds, reward weights, transaction costs, and
  selected hyperparameters;
- comparison against deterministic and classical finance benchmarks.

The objective is not only to train an agent, but to evaluate whether its
behavior is robust outside the training period.

## Benchmarks

The planned benchmark set includes:

- equal-weight portfolio;
- buy-and-hold portfolio;
- rolling Markowitz mean-variance allocation;
- risk parity allocation.

These baselines are required before interpreting TD3 results. They provide
reference points for allocation quality, turnover, risk exposure, and
out-of-sample behavior.

## Repository Structure

```text
portfolio_drl_td3/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── models/
│   └── tables/
├── reports/
├── src/
│   ├── backtest/
│   │   ├── benchmarks.py
│   │   ├── evaluate_policy.py
│   │   ├── markowitz.py
│   │   └── risk_parity.py
│   ├── data/
│   ├── env/
│   ├── memory/
│   │   └── replay_buffer.py
│   ├── models/
│   │   ├── actor.py
│   │   ├── critic.py
│   │   └── td3_agent.py
│   ├── rewards/
│   ├── train/
│   │   └── train_td3.py
│   ├── utils/
│   ├── validation/
│   │   ├── sensitivity_analysis.py
│   │   └── walk_forward.py
│   └── visualization/
│       ├── plot_equity_curves.py
│       ├── plot_reward_components.py
│       └── plot_weights.py
├── tests/
├── requirements.txt
└── README.md
```

Data, generated outputs, and reports are kept outside version control by
default. Source code, configuration, tests, and architectural scaffolds are
intended to be versioned.

## Development Roadmap

1. Define configuration loading, reproducibility utilities, and basic project
   documentation.
2. Implement the data pipeline for downloading prices, resampling to the target
   frequency, and preparing return/features matrices.
3. Implement and test the portfolio environment with long-only, fully invested
   constraints and transaction costs.
4. Implement benchmark policies and an evaluation interface before training TD3.
5. Implement the replay buffer, actor, twin critics, target networks, and TD3
   update logic in PyTorch.
6. Add chronological validation, walk-forward experiments, and sensitivity
   analysis.
7. Produce visualizations and tables for academic interpretation.

## Current Status

The repository currently contains:

- base project configuration in `configs/config.yaml`;
- package structure under `src/`;
- scaffold modules for TD3 models, replay memory, training, backtesting,
  validation, and visualization;
- dependency declarations in `requirements.txt`;
- empty notebook and test directories preserved for future work.

The following components are not implemented yet:

- data download and preprocessing;
- portfolio environment dynamics;
- reward calculation;
- actor and critic neural networks;
- TD3 update logic;
- replay buffer behavior;
- benchmark calculations;
- model training;
- backtest evaluation;
- empirical results.

## Academic Notes

This project is designed as a research implementation rather than a production
trading system. The code should support clear reasoning about assumptions,
constraints, reward design, validation methodology, and failure modes.

Any future empirical claim should be backed by reproducible experiments,
chronological out-of-sample testing, benchmark comparison, and sensitivity
analysis. Until those steps are implemented and executed, the repository should
be understood as an architectural and methodological foundation.
