# Portfolio DRL TD3

Dynamic portfolio allocation with **Twin Delayed Deep Deterministic Policy Gradient (TD3)**.

This repository is part of my Master's thesis work in quantitative finance. The goal is not to build a “magic trading bot”. The real goal is more serious:

> Test whether a continuous-action reinforcement learning agent can learn useful portfolio allocation policies once realistic constraints, transaction costs, drawdown behavior, benchmark comparison, statistical uncertainty, and regime sensitivity are taken seriously.

The main finding so far is not “TD3 beats everything”.

It is this:

> Unconstrained TD3 tends to collapse into concentrated, fragile allocation policies. When the action space is economically constrained, especially through max-weight caps, TD3 becomes much more stable and competitive under mandate-aware evaluation.

That is the research story.

## Why this project exists

A lot of DRL portfolio allocation work looks good until you ask uncomfortable questions:

- What happens after transaction costs?
- What happens under drawdown constraints?
- Is the agent just concentrating into one asset?
- Does it beat simple benchmarks?
- Does it survive different regimes?
- Is the result statistically clear, or just a lucky backtest?
- Would anyone with real money accept the risk profile?

This project is built around those questions.

The uncomfortable answers are part of the work.

## Asset universe

The agent allocates weekly across:

- `SPY` — U.S. equities
- `TLT` — long-duration U.S. Treasuries
- `GLD` — gold
- `BTC-USD` — Bitcoin
- `CASH` — synthetic zero-return cash

The portfolio is long-only and fully invested. The actor outputs portfolio weights through a softmax layer.

At each decision date, the model observes features available up to the previous period and chooses the allocation for the next return period. Features are shifted to reduce look-ahead risk, and normalization is fitted only on the training window.

## What is inside

### TD3 portfolio engine

- PyTorch actor and twin-critic networks
- Replay buffer with explicit seed control
- TD3 target networks
- delayed policy updates
- target policy smoothing
- portfolio environment with transaction costs, turnover, drawdown, cash and concentration diagnostics
- weekly train / validation / test workflow
- multi-fold, multi-seed protocol revalidation

### Feature candidates

The project is structured around candidate feature sets:

- `V2_reference_full`: base return, momentum, volatility, and regime-style features
- `V3_real_macro_current`: V2 plus local macro features
- `V4_real_garch_current`: V2 plus real rolling fitted GARCH volatility forecasts
- `V5_no_volatility_block`: ablation candidate without the volatility block
- `V6_financial_state`: financial-state candidate with cash/risk-off structure
- `V7_real_macro_garch_current`: macro + real GARCH combined
- `V8_ewma_garch_vol_current`: GARCH + EWMA volatility state

V3 uses local macro data only. Macro data is prepared outside training and loaded from a processed CSV. No macro download happens during training or evaluation.

Current V3 macro variables:

- `DGS10`
- `DGS2`
- `VIX`
- `DXY`
- `CPI`

Important caveat: the macro dataset uses current-vintage data and a conservative CPI lag approximation. It is not yet a full real-time vintage or release-calendar macro database.

V4 uses real rolling fitted GARCH forecasts:

- backend: `arch_model`
- model: zero-mean normal GARCH(1,1)
- forecast horizon: 1 week
- timing: forecast at date `t` uses returns through `t-1`
- CASH is excluded from fitted GARCH volatility features
- fallback is used only for insufficient-history warmup periods

## Benchmarks

The agent is compared against simple and dynamic baselines, including:

- buy-and-hold assets
- equal weight
- 60/40 SPY/TLT
- momentum winner
- risk-adjusted momentum winner
- SPY/CASH trend following
- defensive risk-off rule
- rolling inverse-volatility risk parity
- rolling constrained Markowitz variants

This is intentional. TD3 should not be compared against decorative benchmarks. If a simple rule beats the agent, the simple rule wins. No drama.

## Evaluation layers

### 1. Standard performance metrics

The project reports:

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

### 2. Robust score

The robust score combines risk-adjusted performance, drawdown behavior, stability and discipline-oriented metrics.

It is useful for ranking, but it is not treated as divine truth. It is a scoring layer, not a law of physics.

### 3. Mandate-aware score

Pure performance is not enough. A strategy can look great and still be unusable.

The mandate-aware layer penalizes strategies that require excessive drawdown recovery or fail drawdown eligibility. It helps separate:

- aggressive strategies with high raw performance but unacceptable drawdowns
- constrained strategies with more realistic risk behavior

The key idea:

> The max-weight cap acts as an economic regularizer on the action space.

That turned out to be one of the most important insights of the project.

### 4. Statistical validation

The project includes a reporting-only statistical validation layer based on available return histories.

This layer estimates confidence intervals and paired bootstrap comparisons against clean benchmarks such as:

- `BuyHold_GLD`
- `trend_spy_cash_12p`

Current statistical validation does not support a strong claim that V3/V4 are statistically superior in Sharpe to the strongest clean benchmarks.

That matters.

The correct conclusion is:

> Constrained TD3 is economically and behaviorally promising, but Sharpe superiority is statistically uncertain.

That is not a failure. That is honest research.

### 5. Regime analysis

The project also includes regime analysis.

The goal is to understand when constrained TD3 works, when it does not, and whether the behavior makes economic sense.

Current regime analysis shows:

- constrained TD3 is regime-sensitive
- V3 and V4 are competitive in several out-of-sample slices
- clean benchmarks still lead important regimes
- no strategy dominates everywhere

Again: good. Suspiciously perfect results should not be a flex.

## Current research status

The strongest current constrained TD3 candidates are:

- `V3_cap_0.60`: best TD3 candidate by mandate-aware score
- `V4_cap_0.50`: best TD3 candidate by robust score
- `V7_cap_0.50`: improves materially with a cap, but does not beat V3/V4
- `V8_cap_0.50`: improves materially with a cap, but does not beat V3/V4

Current top mandate-aware ranking from the final constrained report:

1. `V3_cap_0.60`
2. `V4_cap_0.50`
3. `V7_cap_0.50`
4. `V6_cap_0.50`
5. `V2_cap_0.50`
6. `V8_cap_0.50`
7. `V5_cap_0.70`
8. `BuyHold_GLD`

The important result is not that TD3 wins everything.

It does not.

The stronger result is:

> Unconstrained TD3 is fragile and tends to concentrate. Constrained TD3, especially with macro or GARCH-based state variables, becomes competitive under a mandate-aware evaluation framework.

## What the experiments taught me

### 1. Concentration was the structural problem

Unconstrained TD3 often learned near single-asset behavior.

That looked like intelligence at first glance, but it was mostly fragility: high concentration, unstable drawdowns, and questionable policy behavior.

### 2. Max-weight caps changed the game

The cap experiments were not cosmetic. They materially improved:

- drawdown behavior
- turnover
- effective number of assets
- mandate-aware score
- robustness of the allocation behavior

### 3. Econometric features help only after the action space is controlled

Macro and GARCH features became much more interesting once the allocation problem was constrained.

Without the cap, the agent often remained fragile.

### 4. More features do not automatically mean better policy

V7 and V8 are useful negative results.

- V7 combined macro + GARCH
- V8 combined GARCH + EWMA volatility

Both improved with caps, but neither beat the simpler V3/V4 constrained candidates.

That is valuable. It prevents the project from becoming feature soup.

### 5. Benchmarks are still hard to beat

Simple and dynamic benchmarks remain strong. That is finance.

The thesis is not:

> “TD3 destroys benchmarks.”

The thesis is closer to:

> “Realistic constraints can turn fragile DRL portfolio policies into economically competitive allocation rules, but superiority depends on regime, benchmark, and statistical uncertainty.”

## Repository structure

    portfolio_drl_td3/
    ├── configs/          # YAML experiment configuration
    ├── docs/             # research log and protocol notes
    ├── notebooks/        # exploratory notebooks
    ├── scripts/          # standalone data acquisition / preparation scripts
    ├── src/
    │   ├── analysis/     # reports, audits, validation, figures
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

Generated data, outputs, logs and model artifacts are excluded from version control by default.

## Running the project

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Run the test suite:

    .venv/bin/python -m unittest discover tests

Run a protocol TD3 revalidation example:

    .venv/bin/python -m src.experiments.run_protocol_pure_td3_revalidation \
      --returns-path data/processed/returns_weekly_latest.csv \
      --episodes 60 \
      --seeds 7,21,42,84,101,123,202,303,404,505 \
      --output-dir outputs/tables/protocol_revalidation_example

Run a selected candidate only:

    .venv/bin/python -m src.experiments.run_protocol_pure_td3_revalidation \
      --returns-path data/processed/returns_weekly_latest.csv \
      --episodes 60 \
      --seeds 7,21,42,84,101,123,202,303,404,505 \
      --candidates V4_real_garch_current \
      --output-dir outputs/tables/protocol_v4_real_garch_current

Run cap sensitivity:

    .venv/bin/python -m src.experiments.run_cap_sensitivity_experiment \
      --returns-path data/processed/returns_weekly_latest.csv \
      --output-dir outputs/tables/cap_sensitivity_experiment \
      --candidates V3_real_macro_current,V4_real_garch_current \
      --episodes 60 \
      --seeds 7,21,42,84,101,123,202,303,404,505 \
      --max-weight-grid uncapped,0.50,0.60,0.70,0.80

Build the final constrained report:

    .venv/bin/python -m src.analysis.build_final_constrained_td3_report \
      --cap-sensitivity-dir outputs/tables/cap_sensitivity_experiment_60ep_10seeds \
      --v3-cap-sensitivity-dir outputs/tables/cap_sensitivity_experiment_v3_60ep_10seeds_seeded \
      --v4-cap-sensitivity-dir outputs/tables/cap_sensitivity_experiment_v4_60ep_10seeds \
      --v7-cap-sensitivity-dir outputs/tables/cap_sensitivity_experiment_v7_60ep_10seeds \
      --v8-cap-sensitivity-dir outputs/tables/cap_sensitivity_experiment_v8_60ep_10seeds \
      --benchmark-comparison-dir outputs/tables/capped_td3_protocol_comparison_60ep_10seeds_cap060 \
      --output-dir outputs/tables/final_constrained_td3_report_with_v3_v4_v7_v8_60ep_10seeds

Build statistical validation:

    .venv/bin/python -m src.analysis.statistical_validation_report \
      --final-report-dir outputs/tables/final_constrained_td3_report_with_v3_v4_v7_v8_60ep_10seeds \
      --output-dir outputs/tables/statistical_validation_final_v3_v4

Build regime analysis:

    .venv/bin/python -m src.analysis.regime_analysis_report \
      --final-report-dir outputs/tables/final_constrained_td3_report_with_v3_v4_v7_v8_60ep_10seeds \
      --output-dir outputs/tables/regime_analysis_final_v3_v4

Build final figures:

    .venv/bin/python -m src.analysis.build_final_figures \
      --final-report-dir outputs/tables/final_constrained_td3_report_with_v3_v4_v7_v8_60ep_10seeds \
      --regime-analysis-dir outputs/tables/regime_analysis_final_v3_v4 \
      --output-dir outputs/figures/final_v3_v4_v7_v8

## What I am testing

The research question behind the code is:

> Can a TD3 agent learn a dynamic allocation policy that remains competitive once transaction costs, drawdowns, concentration, benchmark rules, statistical uncertainty, regime sensitivity and realistic mandate constraints are included?

The current answer is nuanced:

- unconstrained TD3 is not enough
- simple benchmarks are very hard to beat
- max-weight constraints improve TD3 behavior substantially
- macro and GARCH features become useful mainly when the allocation problem is constrained
- adding more features does not automatically improve policy quality
- final claims must be based on out-of-sample protocol comparisons, not isolated backtests
- statistical validation currently supports caution rather than strong superiority claims

## Academic disclaimer

This repository is research code. It is not production trading software, financial advice, or an investment recommendation.

Any empirical claim in this project must be supported by reproducible experiments, chronological out-of-sample testing, benchmark comparison, sensitivity analysis, statistical validation, regime analysis and audit checks.
