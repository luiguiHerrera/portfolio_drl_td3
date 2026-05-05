# Portfolio DRL TD3

## Overview

This repository is an academic implementation project for a portfolio allocation
agent based on Twin Delayed Deep Deterministic Policy Gradient (TD3). The goal is
to build the main reinforcement learning components from scratch in PyTorch,
with a modular structure that supports inspection, testing, and methodological
discussion.

The project now includes implemented and tested building blocks for
configuration, data preparation, portfolio simulation, benchmark evaluation,
replay memory, actor and critic networks, and the core TD3 sampled-batch update.
It does not yet include a full training loop, empirical experiments, or
validated investment results.

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

Stable-Baselines3 is intentionally not used at this stage. The TD3 agent is
implemented directly in PyTorch to support understanding, auditability, and
academic defense.

## Initial Asset Universe

The initial universe is configured as:

- `SPY`: U.S. equity market exposure;
- `TLT`: long-duration U.S. Treasury exposure;
- `GLD`: gold exposure;
- `BTC-USD`: Bitcoin exposure;
- `CASH`: synthetic cash allocation with zero return in the initial
  implementation.

This universe is intentionally compact. It combines traditional assets and a
high-volatility digital asset while keeping the first implementation tractable.
`CASH` is not downloaded from Yahoo Finance; it is represented as a synthetic
zero-return series aligned with the market assets.

## Portfolio Constraints

The initial portfolio setting is:

- long-only;
- fully invested across the selected asset universe, including `CASH`;
- no short selling;
- no leverage;
- portfolio weights must be non-negative;
- portfolio weights must sum to one.

The actor maps state observations to portfolio weights. It produces logits
internally and applies a softmax output so that actions are valid portfolio
weights by construction.

## TD3 Architecture

The implemented TD3 core currently includes:

- `ActorNetwork`: PyTorch MLP mapping state observations to softmax portfolio
  weights;
- `CriticNetwork`: PyTorch MLP estimating `Q(s, a)` from concatenated state and
  action tensors;
- twin critics and target critics inside `TD3Agent`;
- actor and actor target networks;
- hard target initialization and soft target updates;
- delayed actor updates;
- target policy smoothing with projection back to long-only, fully invested
  weights;
- NumPy replay buffer for off-policy transitions;
- one sampled-batch `train_step` method.

The project does not yet include a full training loop, environment rollout
collection, checkpointing, experiment tracking, or empirical model evaluation.

## Reward Function

The current reward implementation is a minimal net-return baseline:

```text
reward = portfolio_return - transaction_cost
```

The planned research reward design may extend this baseline with interpretable
risk-control terms:

```text
reward =
    net return
    + dynamic EWMA Sharpe component
    - drawdown penalty
    - transaction cost penalty
    - turnover penalty
```

Those additional components are not implemented yet. They should be introduced
only after the environment, data preparation, and basic training path are stable
and testable.

## State Features

The repository includes minimal return-based feature engineering:

- current weekly return;
- 4-week rolling compounded momentum;
- 12-week rolling compounded momentum;
- 4-week rolling volatility;
- 12-week rolling volatility.

Feature normalization is fit only on the training split to avoid data leakage.
Realized asset returns are not normalized because they are used to compute
portfolio returns and rewards.

Future state features may include macroeconomic variables and GARCH-based
expected volatility estimates. These are not implemented yet.

## Validation Strategy

Validation follows time-series constraints. Random shuffling is not appropriate
for this setting.

The implemented data utilities include:

- chronological train, validation, and test splitting;
- train-only feature standardization;
- aligned return and feature splits for model training and evaluation.

The planned validation strategy also includes:

- walk-forward validation;
- out-of-sample evaluation;
- sensitivity analysis across seeds, reward weights, transaction costs, and
  selected hyperparameters;
- comparison against deterministic and classical finance benchmarks.

## Benchmarks

The repository currently implements:

- equal-weight return calculation;
- buy-and-hold return calculation;
- shared evaluation metrics such as cumulative return, annualized return,
  annualized volatility, Sharpe ratio, maximum drawdown, and summary metrics.

The planned benchmark set also includes:

- rolling Markowitz mean-variance allocation;
- risk parity allocation.

Markowitz and risk parity modules are present as architectural placeholders but
are not implemented yet.

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
│   │   ├── build_dataset.py
│   │   ├── download.py
│   │   ├── features.py
│   │   ├── normalize.py
│   │   ├── prepare_dataset.py
│   │   ├── preprocess.py
│   │   └── split.py
│   ├── env/
│   │   └── portfolio_env.py
│   ├── memory/
│   │   └── replay_buffer.py
│   ├── models/
│   │   ├── actor.py
│   │   ├── critic.py
│   │   └── td3_agent.py
│   ├── rewards/
│   │   └── reward.py
│   ├── train/
│   │   └── train_td3.py
│   ├── utils/
│   │   ├── config.py
│   │   └── seed.py
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

Data, generated outputs, saved models, and reports are kept outside version
control by default. Source code, configuration, tests, and architectural modules
are intended to be versioned.

## Development Roadmap

1. Completed: define project configuration, reproducibility utilities, and base
   documentation.
2. Completed: implement minimal data download, preprocessing, return dataset
   construction, feature engineering, chronological splitting, and train-only
   feature normalization.
3. Completed: implement a minimal portfolio environment that separates realized
   returns from feature observations.
4. Completed: implement basic benchmark return utilities and shared evaluation
   metrics.
5. Completed: implement NumPy replay memory, PyTorch actor and critic networks,
   target networks, and core TD3 sampled-batch update logic.
6. Next: implement a controlled training loop that connects prepared datasets,
   `PortfolioEnv`, `ReplayBuffer`, and `TD3Agent`.
7. Next: add model evaluation workflows, benchmark comparison tables, and
   reproducible experiment outputs.
8. Later: implement walk-forward validation, sensitivity analysis, Markowitz,
   risk parity, plotting modules, and richer state features.

## Current Status

Implemented and tested components include:

- YAML configuration loading and minimal schema validation;
- reproducibility seed utility;
- Yahoo Finance price download for market assets, excluding synthetic `CASH`;
- return preprocessing with weekly resampling and zero-return `CASH`;
- returns dataset builder;
- return-based feature engineering;
- chronological train, validation, and test splitting;
- train-only standard feature normalization;
- prepared dataset builder returning aligned returns and normalized features;
- `PortfolioEnv` with separate realized returns and optional feature
  observations;
- minimal net-return reward;
- equal-weight and buy-and-hold benchmark returns;
- portfolio evaluation metrics;
- NumPy replay buffer;
- PyTorch actor and critic networks;
- TD3 agent core utilities and one sampled-batch update step.

Not implemented yet:

- full TD3 training loop;
- environment rollout collection for replay buffer population;
- model checkpointing or experiment logging;
- walk-forward validation execution;
- sensitivity analysis execution;
- Markowitz and risk parity benchmark logic;
- plotting and report generation;
- macroeconomic features;
- GARCH expected volatility features;
- empirical results.

## Academic Notes

This project is designed as a research implementation rather than a production
trading system. The code should support clear reasoning about assumptions,
constraints, reward design, validation methodology, and failure modes.

Any future empirical claim should be backed by reproducible experiments,
chronological out-of-sample testing, benchmark comparison, and sensitivity
analysis. Until those steps are implemented and executed, the repository should
be understood as a tested methodological foundation, not as evidence of
profitable or validated trading performance.
