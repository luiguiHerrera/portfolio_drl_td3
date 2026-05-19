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
- Local return snapshots through `data.returns_path` for reproducible runs
  without downloading inside training.
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

Mandate-aware reward penalties are opt-in. Default reward behavior remains
unchanged.

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
- Individual buy-and-hold asset benchmarks for checking whether the agent adds
  dynamic allocation value beyond concentrated exposure to a single winning
  asset.

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

### Feature Sets

The feature pipeline is versioned so new state variables can be tested without
changing older experiments.

- V1 is the default return-based feature set.
- V2 is opt-in and adds richer return, risk, and simple regime features.
- V3 is opt-in and extends V2 with optional local macro CSV features.

V3 does not download macro data during training, evaluation, feature
construction, feature comparison, or walk-forward validation. It only reads a
local CSV when `macro_path` is configured, which keeps model runs reproducible
and avoids hidden live-data dependencies. Macro observations are aligned by
date, forward-filled only, and then shifted externally by one period during
dataset preparation. Backfill is not allowed because it can introduce
information that was not available at the decision time.

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

Local real macro CSVs can be prepared as a separate manual step with
`scripts/download_fred_macro_data.py`. The script downloads FRED-style CSVs for
`DGS10`, `DGS2`, `VIXCLS`, `DTWEXBGS`, and `CPIAUCSL` into `data/raw/macro/`,
then builds `data/processed/macro_weekly_2015_2024.csv`. This acquisition step
is deliberately outside the training and evaluation pipeline. CPI uses a
simple lag approximation before weekly alignment, not a full release-calendar
model.

Synthetic macro fixtures remain only for testing the plumbing. They are not
evidence about the usefulness of macro variables. This project is not trying to
win a backtest by adding more columns; the goal is a controlled research
pipeline where each signal can be tested, challenged, and removed if it fails
validation.

The project also includes a feature set comparison runner for V1, V2, and V3
variants under the same walk-forward folds, seeds, reward configuration, TD3
hyperparameters, and benchmark pipeline. A local macro dataset builder can
construct weekly V3 macro input from raw CSV files for `DGS10`, `DGS2`,
`VIXCLS`, `DTWEXBGS`, and `CPIAUCSL`; it uses no live downloads, APIs, FRED
calls, or yfinance calls. Daily series are aligned to weekly Friday using the
last available observation and forward-fill only. CPI is shifted with a simple
availability lag before weekly alignment. No backfill is used.

In the latest long-history V1/V2/V3 comparison, V3_real_macro ranked marginally
first by robust Sharpe with 0.5 dispersion penalty: V3_real_macro = 0.5709,
V2 = 0.5610, and V1 = -0.0739. The difference between V3_real_macro and V2 is
small, benchmark win rates remain weak, and this is not enough to claim robust
TD3 superiority.

After return construction, date boundaries are clipped so final model returns
respect `data.start_date` and `data.end_date`.

## Benchmarks

The current benchmark workflow includes:

- gross equal-weight portfolio;
- transaction-cost-aware equal-weight rebalanced portfolio;
- gross buy-and-hold portfolio;
- individual buy-and-hold asset benchmarks for `SPY`, `TLT`, `GLD`,
  `BTC-USD`, and `CASH`.

Benchmark comparison is performed in memory and produces metrics tables for the
agent and benchmark policies. Individual asset buy-and-hold benchmarks are
included because the TD3 agent can become highly concentrated; comparing against
single-asset buy-and-hold returns helps assess whether it adds dynamic
allocation value or primarily replicates exposure to a winning asset.
Transaction costs are currently modeled only for the equal-weight rebalanced
benchmark. Markowitz and risk parity benchmarks are planned but not implemented
yet.

The project now also has a reproducible local market-data step:
`scripts/download_market_data.py` builds
`data/processed/returns_weekly_2015_2024.csv` for `SPY`, `TLT`, `GLD`,
`BTC-USD`, and `CASH`. On top of that file, simple dynamic allocation rules are
available as a tougher diagnostic hurdle: 12-period momentum winner,
risk-adjusted momentum, SPY/CASH trend following, and a defensive risk-off
rule.

Right now, the useful uncomfortable result is that these simple dynamic rules
beat the current TD3 policies on risk-adjusted metrics. That is not a failure;
it is the point of doing the research properly. TD3 needs to beat real
decision rules, not decorative benchmarks. Sortino and Calmar outputs also
carry safety flags so extreme or infinite ratios are visible instead of being
quietly over-interpreted.

## Experiment Workflow

`run_basic_experiment(config_path)` runs the minimal TD3 workflow and organizes
results in memory. It returns:

- training summary;
- validation and test metrics tables;
- validation and test comparison summaries;
- validation and test diagnostics, including allocation concentration and
  transaction-cost diagnostics;
- raw in-memory result.

Comparison summaries include the best individual buy-and-hold benchmark by
Sharpe ratio and the agent's metric differences versus that benchmark.

Fresh-market experiments can update market data first, write a local returns
snapshot, and then train from that snapshot. Training itself does not need to
download data.

`run_and_save_basic_experiment(config_path, output_dir, experiment_name)` runs
the same experiment and saves selected CSV outputs. Saved files include:

- `training_summary.csv`;
- `validation_metrics_table.csv`;
- `test_metrics_table.csv`;
- `validation_comparison_summary.csv`;
- `test_comparison_summary.csv`;
- `validation_diagnostics.csv`;
- `test_diagnostics.csv`.
- `validation_policy_history.csv`;
- `test_policy_history.csv`.

Saved outputs now support ex-post behavior checks:

- per-period policy history;
- shadow mandate penalties;
- concentration quality;
- cash allocation quality;
- dominant-asset and regime attribution.

Concentration is not automatically bad. The useful question is whether the
model found edge, followed a valid mandate, or gamed the penalty. Do not tune
reward blindly.

An explicit walk-forward validation workflow is also implemented for manually
defined chronological folds. Each fold trains on its own train window, validates
on the next period, and tests on the following period. The workflow saves
fold-level result tables and aggregate summary tables, reducing dependence on a
single fixed split and supporting stronger empirical validation. Current
walk-forward results remain preliminary and should not be interpreted as
evidence of robust empirical superiority.

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
- Extend walk-forward validation across additional folds, seeds, and stronger
  benchmark comparisons.
- Run sensitivity analysis across seeds, costs, and hyperparameters.
- Add macroeconomic and GARCH-based state features.
- Add plotting and report generation.
- Conduct reproducible empirical experiment analysis.

## Academic Disclaimer

This repository is research code, not production trading software or investment
advice. Any empirical claim requires reproducible experiments, chronological
out-of-sample testing, benchmark comparison, and sensitivity analysis.
