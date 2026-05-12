# Portfolio DRL TD3

## Overview

This repository contains an academic PyTorch implementation of a
Twin Delayed Deep Deterministic Policy Gradient (TD3) agent for dynamic
portfolio allocation. The project is intended for Master's thesis research and
focuses on transparent, modular implementation rather than production trading
use.

The codebase implements the main components needed to study a continuous-action
portfolio allocation agent under long-only, fully invested constraints. It is
designed to support methodological discussion, testing, and future empirical
experiments.

## Current Scope

The repository currently provides a methodological prototype. It includes a
minimal end-to-end training, evaluation, benchmark comparison, and CSV output
workflow, but it does not provide validated investment results or empirical
conclusions.

No performance claim should be inferred from the current implementation. Future
claims must be supported by reproducible out-of-sample experiments, benchmark
comparison, sensitivity analysis, and appropriate validation design.

## Implemented Components

### Data Pipeline

- YAML configuration loading and validation.
- Yahoo Finance download support for market assets.
- Synthetic `CASH` asset handling with zero return.
- Price preprocessing and weekly return construction.
- Return-based feature engineering.
- Chronological train, validation, and test splitting.
- Train-only feature normalization.
- Prepared dataset builder with aligned returns and features.

### Environment and Reward

- `PortfolioEnv` for long-only, fully invested portfolio simulation.
- Separation between observed state features and realized asset returns.
- Financial portfolio value is updated with net realized return:
  `financial_net_return = portfolio_return - transaction_cost`.
- The learning reward is a separate configurable risk-aware signal:

```text
reward =
    lambda_return * portfolio_return
    - lambda_transaction_cost * transaction_cost
    - lambda_turnover * turnover
    - lambda_concentration * concentration
    - lambda_drawdown * drawdown
```

Dynamic Sharpe reward terms are not implemented yet.

### TD3 Model

- NumPy replay buffer.
- PyTorch `ActorNetwork` with softmax portfolio weights.
- PyTorch `CriticNetwork`.
- `TD3Agent` with twin critics, target networks, delayed actor updates, target
  policy smoothing, and sampled-batch `train_step`.
- Minimal `train_td3` loop connecting datasets, environment, replay buffer, and
  TD3 agent.

### Evaluation and Benchmarks

- Agent policy episode evaluation.
- Portfolio metrics including cumulative return, annualized return, annualized
  volatility, Sharpe ratio, and maximum drawdown.
- Agent performance metrics use financial net returns after transaction costs;
  gross policy returns remain available as diagnostics.
- Allocation risk diagnostics including max weight, cash weight, Herfindahl
  index, effective number of assets, entropy, turnover, and transaction cost.
- Basic benchmark comparison workflow.
- Compact validation and test comparison summaries.

### Experiment Workflow

- Minimal in-memory experiment runner.
- Run-and-save workflow for selected CSV outputs.
- CSV saving utility with defensive validation before writing outputs.
- No model, replay buffer, plot, report, or raw result persistence.

### Testing

- Unit tests cover implemented data, environment, benchmark, model, training,
  experiment, and saving utilities.

## Methodology

The initial allocation problem is defined over:

- `SPY`: U.S. equity market exposure;
- `TLT`: long-duration U.S. Treasury exposure;
- `GLD`: gold exposure;
- `BTC-USD`: Bitcoin exposure;
- `CASH`: synthetic cash asset with zero return.

The portfolio is long-only and fully invested across all assets, including
`CASH`. Actor outputs are transformed into non-negative weights that sum to one.

State features are separated from realized returns. At date `t`, features are
shifted so the agent observes information available through `t-1`, then selects
portfolio weights for period `t`. Realized returns at `t` are observed after the
decision, and portfolio return and reward are computed from those current action
weights net of transaction cost. Feature normalization is fitted only on the
training split to reduce data leakage risk. Validation and test splits are
chronological.

## Benchmarks

The current benchmark workflow includes:

- gross equal-weight portfolio;
- transaction-cost-aware equal-weight rebalanced portfolio;
- gross buy-and-hold portfolio.

Benchmark comparison is performed in memory and produces metrics tables for the
agent and benchmark policies. Transaction costs are currently modeled only for
the equal-weight rebalanced benchmark. Markowitz and risk parity benchmarks are
planned but not implemented yet.

## Experiment Workflow

`run_basic_experiment(config_path)` runs the minimal TD3 workflow and organizes
results in memory. It returns:

- training summary;
- validation and test metrics tables;
- validation and test comparison summaries;
- validation and test diagnostics, including allocation concentration and
  transaction-cost diagnostics;
- raw in-memory result.

`run_and_save_basic_experiment(config_path, output_dir, experiment_name)` runs
the same experiment and saves selected CSV outputs. Saved files include:

- `training_summary.csv`;
- `validation_metrics_table.csv`;
- `test_metrics_table.csv`;
- `validation_comparison_summary.csv`;
- `test_comparison_summary.csv`;
- `validation_diagnostics.csv`;
- `test_diagnostics.csv`.

Diagnostic CSV outputs preserve allocation risk fields and flattened final
portfolio weights.

The workflow does not save `raw_result`, the agent, replay buffer, model
checkpoints, plots, or reports.

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
│   ├── data/
│   ├── env/
│   ├── experiments/
│   ├── memory/
│   ├── models/
│   ├── rewards/
│   ├── train/
│   ├── utils/
│   ├── validation/
│   └── visualization/
├── tests/
├── requirements.txt
└── README.md
```

Data, generated outputs, saved models, and reports are excluded from version
control by default. Source code, configuration, and tests are intended to be
versioned.

## Roadmap

- Extend the reward function with dynamic Sharpe, drawdown, transaction cost,
  and turnover terms.
- Implement Markowitz and risk parity benchmark logic.
- Add walk-forward validation.
- Run sensitivity analysis across seeds, costs, and hyperparameters.
- Add macroeconomic and GARCH-based state features.
- Add plotting and report generation.
- Conduct reproducible empirical experiment analysis.

## Academic Disclaimer

This repository is research code, not production trading software or investment
advice. Any empirical claim requires reproducible experiments, chronological
out-of-sample testing, benchmark comparison, and sensitivity analysis.
