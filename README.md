# Portfolio DRL TD3

Research code for dynamic portfolio allocation with **Twin Delayed Deep Deterministic Policy Gradient (TD3)**.

This project is part of my Master's thesis work in quantitative finance. The goal is not to sell a trading bot or overfit a pretty backtest. The goal is to test, under a reproducible protocol, whether a continuous-action reinforcement learning agent can produce useful portfolio allocation decisions once realistic constraints, transaction costs, drawdown behavior, and benchmark comparisons are taken seriously.

## Core idea

The agent allocates weekly across:

- `SPY` — U.S. equities
- `TLT` — long-duration U.S. Treasuries
- `GLD` — gold
- `BTC-USD` — Bitcoin
- `CASH` — synthetic zero-return cash

The portfolio is long-only and fully invested. The actor outputs portfolio weights through a softmax layer.

At each decision date, the model observes state variables available up to the previous period and chooses weights for the next return period. Features are shifted to reduce look-ahead risk, and feature normalization is fitted only on the training window.

## What this repository contains

### TD3 portfolio engine

- PyTorch actor and twin-critic networks
- Replay buffer
- TD3 target networks, delayed policy updates, and target policy smoothing
- Portfolio environment with transaction costs, turnover, drawdown, concentration, and cash diagnostics
- Weekly train / validation / test workflow
- Walk-forward style protocol revalidation

### Feature sets

The project is versioned by feature candidates:

- `V2_reference_full`: return, momentum, volatility, and regime-style features
- `V3_real_macro_current`: V2 plus local macro features
- `V5_no_volatility_block`: ablation candidate without the volatility block
- `V6_financial_state`: financial-state candidate with cash/risk-off structure

V3 uses local macro data only. Macro data is prepared outside training and loaded from a processed CSV. No macro download happens during training or evaluation.

Current V3 macro variables:

- `DGS10`
- `DGS2`
- `VIX`
- `DXY`
- `CPI`

Important caveat: the macro dataset uses current-vintage data and a conservative CPI lag approximation. It is not yet a full real-time vintage or release-calendar macro database.

### Benchmarks

The agent is compared against both simple and dynamic baselines, including:

- Buy-and-hold assets
- Equal weight
- 60/40 SPY/TLT
- Momentum winner
- Risk-adjusted momentum winner
- SPY/CASH trend following
- Defensive risk-off rule
- Rolling inverse-volatility risk parity
- Rolling constrained Markowitz variants

This is intentional. TD3 should not be compared only against decorative benchmarks. If a simple rule beats the agent, the simple rule wins. No drama.

## Evaluation philosophy

The project separates two evaluation layers.

### 1. Performance-oriented evaluation

Metrics include:

- cumulative return
- annualized return
- annualized volatility
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- max drawdown
- turnover
- transaction costs
- effective number of assets
- average max weight
- cash exposure

The project also reports a robust score using risk-adjusted and stability-oriented components.

### 2. Mandate-aware evaluation

Pure performance is not enough. A portfolio can look good while being unusable for a real mandate.

The mandate-aware layer penalizes strategies that require excessive drawdown recovery or fail drawdown eligibility. This helps distinguish between:

- aggressive strategies that win raw performance rankings but suffer very large drawdowns;
- constrained strategies that may be more realistic for an investment mandate.

## Current research status

The most important result so far:

> Unconstrained TD3 does not dominate the benchmark suite.

That was uncomfortable, but useful.

The unconstrained TD3 policies often learned extreme concentration, behaving almost like single-asset selectors. Because of that, the project added controlled max-weight constraint experiments.

The stronger current finding is:

> TD3 with a max-weight allocation constraint becomes materially more competitive under mandate-aware evaluation.

Recent cap sensitivity experiments suggest that the benefit does not come from one magical cap value. The best cap is candidate-sensitive.

Examples from the current experimental path:

- `V2` improved most with a stricter cap around `0.50`
- `V5` improved most around `0.70`
- `V6` improved most around `0.50`
- `V3_real_macro_current` became much more interesting once tested with a cap, especially around `0.50`

The latest V3 capped result is promising, but still under audit because the uncapped V3 baseline differs between runner paths. Until that consistency check is closed, I treat the result as promising, not final.

That is the tone of the whole project: useful evidence, but no victory lap before the checks pass.

## Main lesson so far

The contribution is not:

> "TD3 magically beats everything."

The more honest claim is:

> Unconstrained TD3 is fragile and tends to concentrate. However, when realistic portfolio constraints are added, TD3 becomes a much more credible allocation framework and can become competitive against clean mandate-aware benchmarks.

This is closer to how an actual investment process should be tested.

## Repository structure

    portfolio_drl_td3/
    ├── configs/          # YAML experiment configuration
    ├── docs/             # research log and protocol notes
    ├── notebooks/        # exploratory notebooks
    ├── scripts/          # standalone data acquisition / preparation scripts
    ├── src/
    │   ├── analysis/     # reports, audits, sensitivity analysis
    │   ├── backtest/     # benchmark and allocation logic
    │   ├── data/         # data loading and feature engineering
    │   ├── env/          # portfolio environment
    │   ├── experiments/  # experiment runners
    │   ├── models/       # actor, critic, TD3 agent
    │   ├── rewards/      # reward functions
    │   ├── train/        # training loop
    │   └── utils/        # config and shared utilities
    ├── tests/            # unit tests
    ├── requirements.txt
    └── README.md

Generated data, outputs, logs, and model artifacts are excluded from version control by default.

## Running the project

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Run the test suite:

    python3 -m unittest discover tests

Run a protocol TD3 revalidation example:

    python3 -m src.experiments.run_protocol_pure_td3_revalidation \
      --returns-path data/processed/returns_weekly_latest.csv \
      --episodes 60 \
      --seeds 7,21,42,84,101,123,202,303,404,505 \
      --output-dir outputs/tables/protocol_revalidation_example

Run a selected candidate only:

    python3 -m src.experiments.run_protocol_pure_td3_revalidation \
      --returns-path data/processed/returns_weekly_latest.csv \
      --episodes 60 \
      --seeds 7,21,42,84,101,123,202,303,404,505 \
      --candidates V3_real_macro_current \
      --output-dir outputs/tables/protocol_v3_real_macro_current

Run cap sensitivity:

    python3 -m src.experiments.run_cap_sensitivity_experiment \
      --returns-path data/processed/returns_weekly_latest.csv \
      --output-dir outputs/tables/cap_sensitivity_experiment \
      --candidates V2_reference_full,V5_no_volatility_block,V6_financial_state \
      --episodes 60 \
      --seeds 7,21,42,84,101,123,202,303,404,505 \
      --max-weight-grid uncapped,0.50,0.60,0.70,0.80

## What I am testing

The research question behind the code is simple:

> Can a TD3 agent learn a dynamic allocation policy that remains competitive once transaction costs, drawdowns, concentration, benchmark rules, and realistic mandate constraints are included?

The current answer is nuanced:

- unconstrained TD3 is not enough;
- simple benchmarks are very hard to beat;
- max-weight constraints improve TD3 behavior substantially;
- macro features may help, but only after the allocation problem is constrained;
- final claims must be based on out-of-sample protocol comparisons, not isolated backtests.

## Academic disclaimer

This repository is research code. It is not production trading software, financial advice, or an investment recommendation.

Any empirical claim in this project must be supported by reproducible experiments, chronological out-of-sample testing, benchmark comparison, sensitivity analysis, and audit checks.
