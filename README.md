# Portfolio DRL TD3

## Overview

This repository contains an academic PyTorch implementation of a Twin Delayed Deep Deterministic Policy Gradient (TD3) agent for dynamic portfolio allocation.

The project is built for Master's thesis research. It is not production trading software. The goal is to study whether a DRL allocation agent can add value against simple, transparent, and harder-to-fool portfolio rules.

No performance claim should be inferred from the current implementation. Any result must survive chronological out-of-sample testing, benchmark comparison, seed sensitivity, and clear risk diagnostics.

## Current Scope

The repository currently provides a research pipeline for:

- weekly multi-asset portfolio allocation;
- TD3 training and evaluation;
- feature set comparison;
- walk-forward validation;
- dynamic benchmark comparison;
- mandate-aware diagnostics;
- robust evaluation scoring;
- controlled reward experiments.

The current uncomfortable but useful lesson is simple: TD3 must beat real decision rules, not decorative benchmarks.

## Implemented Components

### Data Pipeline

- YAML configuration loading and validation.
- Yahoo Finance market-data acquisition as a separate step.
- Synthetic `CASH` asset with zero return.
- Weekly return construction.
- Local return snapshots through `data.returns_path`, so training can run from a reproducible CSV instead of downloading data inside the experiment.
- Feature engineering with V1, V2, and V3 feature sets.
- Optional local macro CSV features for V3.
- Chronological train, validation, and test splitting.
- Train-only feature normalization.

Fresh-market experiments can update market data first, write a local returns snapshot, and then train from that snapshot. The usual latest snapshot is:

```text
data/processed/returns_weekly_latest.csv
```

Frozen historical snapshots can still be created explicitly.

### Environment and Reward

- Long-only, fully invested `PortfolioEnv`.
- Portfolio weights are non-negative and sum to one.
- Financial portfolio value uses net realized return:

```text
financial_net_return = portfolio_return - transaction_cost
```

The base reward is configurable:

```text
reward =
    lambda_return * portfolio_return
    - lambda_transaction_cost * transaction_cost
    - lambda_turnover * turnover
    - lambda_concentration * concentration
    - lambda_drawdown * drawdown
```

Mandate-aware reward penalties are opt-in. Default reward behavior remains unchanged unless explicitly enabled.

### TD3 Model

- NumPy replay buffer.
- PyTorch actor network with softmax portfolio weights.
- Twin critic networks.
- Target networks.
- Delayed actor updates.
- Target policy smoothing.
- Minimal TD3 training loop.

### Evaluation and Diagnostics

The project reports:

- cumulative return;
- annualized return;
- annualized volatility;
- Sharpe ratio;
- Sortino ratio;
- Calmar ratio;
- maximum drawdown;
- turnover;
- transaction costs;
- max weight;
- cash weight;
- Herfindahl index;
- effective number of assets;
- Deflated Sharpe / `robust_score`.

`robust_score` is evaluation-only. It is not the reward function. It combines
Deflated Sharpe, Sortino, Calmar, drawdown control, stability, and mandate
discipline so I do not over-trust one clean-looking Sharpe number after many
experiments.

Saved policy histories allow ex-post behavior analysis:

- shadow mandate penalties;
- concentration quality;
- cash allocation quality;
- dominant-asset attribution;
- regime attribution.

Concentration is not automatically bad. The useful question is whether the model found edge, followed the mandate, or gamed the penalty.

## Methodology

The base universe is:

- `SPY`: U.S. equity market exposure;
- `TLT`: long-duration U.S. Treasury exposure;
- `GLD`: gold exposure;
- `BTC-USD`: Bitcoin exposure;
- `CASH`: synthetic zero-return cash.

At date `t`, the agent observes features available through `t-1`, chooses weights for period `t`, and then receives realized returns for `t`. Feature normalization is fitted only on the training split to reduce leakage risk.

## Feature Sets

The feature pipeline is versioned:

- `V1`: default return-based features.
- `V2`: return, risk, and simple regime features.
- `V3`: V2 plus optional local macro CSV features.
- `V4`: V2 plus deterministic GARCH-style volatility features. Tested, but not better than V2.
- `V5`: V2 plus regime and correlation features. Promising internally, but currently affected by CASH behavior and benchmark weakness.

V3 does not download macro data during training, evaluation, feature construction, feature comparison, or walk-forward validation. It only reads a local CSV when `macro_path` is configured.

Example:

```yaml
features:
  version: v3
  market_asset: SPY
  short_window: 4
  long_window: 12
  ewma_span: 12
  macro_path: data/processed/macro_weekly_2015_2024.csv
  macro_date_column: date
```

Local macro data can be prepared separately with:

```text
scripts/download_fred_macro_data.py
```

The macro step is deliberately outside the training loop. No hidden live-data dependency. No backfill. No magic.

## Benchmarks

The benchmark layer includes:

- equal-weight portfolio;
- transaction-cost-aware equal-weight rebalanced portfolio;
- buy-and-hold portfolio;
- individual buy-and-hold assets;
- simple dynamic allocation rules:
  - momentum winner;
  - risk-adjusted momentum;
  - SPY/CASH trend rule;
  - defensive risk-off rule.

In the current preliminary walk-forward tests, simple dynamic rules beat the current TD3 policies on several risk-adjusted metrics. That is not a failure. That is the point of doing the research properly.

TD3 needs to beat simple rules before it earns complexity.

The latest comparison is still uncomfortable in the useful way: the V5 dynamic
CASH penalty improves internal cash discipline, but it does not beat simple
traditional benchmarks. Under the composite `robust_score`, V2 is still ahead
of `V5_dynamic_cash_025`.

## Mandate-Aware Layer

The project now includes mandate-aware infrastructure:

- risk profiles;
- quantitative mandate limits;
- pure mandate penalty components;
- optional mandate-aware reward;
- ex-post mandate diagnostics.

Example mandate dimensions:

- maximum drawdown;
- maximum volatility;
- maximum weight;
- minimum effective assets;
- maximum turnover.

The mandate layer is not meant to blindly punish concentration. A concentrated allocation can be valid if it is justified by state, regime, risk limits, and realized quality.

The current working principle:

```text
Do not punish concentration.
Punish unjustified behavior.
```

## Experiment Workflow

`run_basic_experiment(config_path)` runs the minimal TD3 workflow and returns results in memory.

`run_and_save_basic_experiment(config_path, output_dir, experiment_name)` runs the same workflow and saves selected CSV outputs.

The fresh-market runner updates market data first, creates a local return snapshot, generates a config pointing to that snapshot, and then runs the experiment:

```text
scripts/run_basic_experiment_with_fresh_market_data.py
```

Training itself should consume a local snapshot. It should not download market data as a hidden side effect.

## Repository Structure

```text
portfolio_drl_td3/
├── configs/
├── data/
├── docs/
├── notebooks/
├── outputs/
├── reports/
├── scripts/
├── src/
│   ├── analysis/
│   ├── backtest/
│   ├── data/
│   ├── env/
│   ├── experiments/
│   ├── memory/
│   ├── models/
│   ├── rewards/
│   ├── risk/
│   ├── train/
│   ├── utils/
│   ├── validation/
│   └── visualization/
├── tests/
├── requirements.txt
└── README.md
```

Data, generated outputs, saved models, and reports are excluded from version control by default.

## Roadmap

- Keep V2 as the clean reference until a stronger model beats it.
- Treat V5 dynamic CASH as experimental, not default.
- Use `robust_score` and Deflated Sharpe only as evaluation tools, not reward.
- Compare every TD3 variant against simple benchmarks before claiming value.
- Improve the model only where diagnostics show a real weakness.
- Keep cutting anything that looks smart but fails out-of-sample.

## Academic Disclaimer

This repository is research code, not investment advice or production trading software.

Any empirical claim requires reproducible experiments, chronological out-of-sample testing, benchmark comparison, and sensitivity analysis.
