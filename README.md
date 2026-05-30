# Portfolio DRL TD3

Master's thesis research code for dynamic portfolio allocation with **Twin Delayed Deep Deterministic Policy Gradient (TD3)**.

This repository evaluates whether TD3-based dynamic portfolio allocation can become mandate-credible once concentration, drawdown, turnover, benchmark comparison, regime sensitivity, and statistical uncertainty are treated as first-class constraints.

The main result is not that TD3 beats the market.

It is this:

> Unconstrained TD3 tends to collapse into concentrated, fragile allocation policies, while max-weight constraints stabilize the action space and make the strongest TD3 specifications more credible under mandate-aware evaluation.

That is the research story.

## Research Questions

The main research question is:

> Can TD3-based dynamic portfolio allocation become mandate-credible under realistic concentration, drawdown, turnover, and benchmark constraints?

A diagnostic subquestion runs through the whole project:

> Does unconstrained TD3 concentration reflect useful selection, or is it a concentration failure mode?

The evidence points to the second answer often enough that concentration control becomes central to the project.

## Asset Universe

The agent allocates weekly across:

- `SPY` — U.S. equities
- `TLT` — long-duration U.S. Treasuries
- `GLD` — gold
- `BTC-USD` — Bitcoin
- `CASH` — synthetic zero-return cash in the standard protocol

The portfolio is long-only and fully invested across the asset set. The actor outputs portfolio weights through a softmax layer.

At each decision date, the model observes state variables available through `t-1`, chooses weights for period `t`, and realizes the portfolio return at `t`. Features are shifted to reduce look-ahead risk, and normalization is fitted only on the training window.

## What Is Inside

### TD3 Portfolio Engine

- PyTorch actor and twin-critic networks
- replay buffer with explicit seed control
- TD3 target networks
- delayed policy updates
- target policy smoothing
- portfolio environment with transaction costs, turnover, drawdown, cash, and concentration diagnostics
- weekly train / validation / test workflow
- multi-fold, multi-seed protocol revalidation

### Feature Candidates

The project is structured around separate state specifications. Each state specification is trained as a separate TD3 policy under the same protocol. It is not one model reused across incompatible state spaces.

Current candidate families:

- `V1` baseline financial state
- `V2_reference_full` rich financial/reference state
- `V3_real_macro_current` financial + current-vintage macro comparison
- `V3_real_macro_vintage_clean_no_dxy` financial + clean real-time/as-of macro state
- `V4_real_garch_current` financial + rolling fitted GARCH volatility state
- `V5_no_volatility_block` regime/no-volatility ablation
- `V6_financial_state` parsimonious financial state
- `V7_real_macro_vintage_clean_no_dxy_garch` clean no-DXY macro + rolling fitted GARCH
- `V8_ewma_garch_vol_current` EWMA/GARCH hybrid volatility state

Current-vintage and DXY-fallback V3 variants remain in the audit trail as historical diagnostics. The clean no-DXY specification is the final macro evidence used for the leading result.

## Macro Data

The leading macro specification is `V3_real_macro_vintage_clean_no_dxy`.

It uses real-time/as-of FRED vintage macro data for:

- `DGS10`
- `DGS2`
- `VIX`
- `CPI`

The dollar proxy is excluded. No full-window fresh true-vintage dollar proxy was available for the 2015-2026 protocol window without fallback, discontinuation, or current-vintage relabeling.

The project therefore avoids saying that the clean leading specification includes DXY. It does not.

Macro data is prepared outside TD3 training and loaded from local processed CSV files. No macro download happens inside training or evaluation.

## GARCH Data

The real GARCH candidates use rolling one-step-ahead forecasts:

- backend: `arch_model`
- model: zero-mean normal GARCH(1,1)
- forecast horizon: 1 week
- timing: forecast at date `t` uses returns through `t-1`
- volatility unit: weekly
- CASH is excluded from fitted GARCH estimation
- fallback rolling realized volatility is used only for insufficient-history warmup

The GARCH pipeline is designed to avoid full-sample fitting and same-period leakage.

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

This is intentional. TD3 should not be compared against decorative benchmarks. If a simple rule beats the agent, the simple rule wins.

## Evaluation Layers

### Standard Metrics

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

### Robust Score

The robust score combines risk-adjusted performance, drawdown behavior, stability, and discipline-oriented metrics. It is useful for ranking, but it is still a reporting layer, not a proof of superiority.

### Mandate-Aware Score

Pure performance is not enough. A strategy can look attractive while being unusable for a real mandate.

The mandate-aware layer penalizes strategies that require excessive drawdown recovery or fail drawdown eligibility. It helps separate:

- aggressive strategies with high raw performance but unacceptable drawdowns
- constrained strategies with more realistic risk behavior

The key empirical insight is:

> The max-weight cap acts as an economic regularizer on the action space.

### Statistical Validation

The project includes reporting-only statistical validation based on available out-of-sample return histories.

This includes:

- bootstrap confidence intervals
- paired bootstrap comparisons
- White Reality Check style multiple-model validation

The statistical validation does not establish TD3 statistical dominance over clean benchmarks.

### Regime Analysis

The regime analysis asks when constrained TD3 works, when benchmarks remain better, and whether the behavior is economically interpretable.

Current interpretation:

- constrained TD3 is regime-sensitive
- clean benchmarks still lead important regimes
- no strategy dominates everywhere

### Mandate Profile Sensitivity

The mandate-profile layer evaluates the same existing results under conservative, moderate, and aggressive investor profiles.

Current profile-specific winners:

- Conservative: `V3_real_macro_vintage_clean_no_dxy_cap_0.50`
- Moderate: `V3_real_macro_vintage_clean_no_dxy_cap_0.60`
- Aggressive: `Equal_Weight` overall; `V3_real_macro_vintage_clean_no_dxy_cap_0.60` remains the best TD3 candidate

The preferred cap is mandate-dependent.

### White Reality Check

The White Reality Check layer is reporting-only validation.

It uses aligned realized out-of-sample weekly net returns and corrects for selecting the best candidate among the evaluated TD3 strategies. It tests mean return differentials, not mandate-aware score.

Current White Reality Check results:

- Against `BuyHold_GLD`: best searched TD3 is `V6_cap_0.50`, weekly mean differential `-0.001891`, p-value `0.9065`
- Against `trend_spy_cash_12p`: best searched TD3 is `V6_cap_0.50`, weekly mean differential `0.000765`, p-value `0.4503`

Interpretation:

> There is no evidence of searched TD3 statistical superiority by mean return differential.

This does not invalidate the mandate-aware stabilization result. It means the statistical claim must remain cautious.

## Current Empirical Conclusions

The leading clean mandate-aware TD3 candidate is:

1. `V3_real_macro_vintage_clean_no_dxy_cap_0.50`

Current central constrained TD3 ranking from the paper's main table:

1. `V3_real_macro_vintage_clean_no_dxy_cap_0.50`
2. `V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50`
3. `V4_cap_0.50`
4. `V6_cap_0.50`
5. `V2_cap_0.50`
6. `V8_cap_0.50`
7. `V5_cap_0.70`

Important interpretation:

- `V3_real_macro_vintage_clean_no_dxy_cap_0.50` is the leading clean mandate-aware TD3 candidate.
- `V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50` is competitive but does not outperform the simpler V3 clean no-DXY specification.
- `V4_cap_0.50` remains a strong volatility-state comparison candidate.
- Adding more econometric complexity, such as GARCH features, remains competitive but does not automatically improve policy quality.
- Unconstrained TD3 tends to produce concentrated and fragile policies.
- Max-weight caps materially improve diversification, drawdown behavior, turnover, and mandate-aware performance.

The defensible claim is:

> TD3 can become mandate-credible only after realistic max-weight constraints materially reduce degenerate concentration. It is competitive under mandate-aware evaluation, but it does not statistically dominate clean benchmarks.

## What The Experiments Taught Me

### 1. Concentration Was The Structural Problem

Unconstrained TD3 often learned near single-asset behavior.

That can look like selection skill, but the diagnostics show fragile concentration, unstable drawdowns, and questionable policy behavior.

### 2. Max-Weight Caps Changed The Behavior

The cap experiments were not cosmetic. They materially improved:

- drawdown behavior
- turnover
- effective number of assets
- mandate-aware score
- robustness of allocation behavior

### 3. Econometric Features Help Only After The Action Space Is Controlled

Macro and GARCH features became more useful once the allocation problem was constrained.

Without max-weight constraints, the agent often remained fragile.

### 4. More Features Do Not Automatically Mean Better Policy

V7 and V8 are useful negative or mixed results.

V7 clean no-DXY + GARCH remains competitive, but it does not beat the simpler V3 clean no-DXY candidate. V8 improves with caps but does not become a leading candidate.

That prevents the project from becoming feature soup.

### 5. Benchmarks Are Still Hard To Beat

Simple and dynamic benchmarks remain strong. That is finance.

The thesis is not:

> TD3 destroys benchmarks.

The thesis is closer to:

> Realistic constraints can turn fragile DRL portfolio policies into more credible allocation rules, but superiority depends on regime, benchmark, mandate, and statistical uncertainty.

## Paper

The current paper artifacts live under:

- `paper/main.tex`
- `paper/main.pdf`
- `paper/references.bib`
- `paper/figures/`
- `paper/scripts/`

The paper follows the same framing as this README: mandate credibility, concentration diagnostics, constrained TD3 comparison, regime sensitivity, and statistical uncertainty.

## Repository Structure

    portfolio_drl_td3/
    ├── configs/          # YAML experiment configuration
    ├── docs/             # research log, protocol notes, freeze notes
    ├── notebooks/        # exploratory notebooks
    ├── paper/            # manuscript, figures, bibliography, paper scripts
    ├── scripts/          # standalone data acquisition / preparation scripts
    ├── src/
    │   ├── analysis/     # reports, audits, validation, figures
    │   │   ├── mandate_profile_comparison_report.py
    │   │   └── white_reality_check_report.py
    │   ├── backtest/     # benchmark and allocation logic
    │   ├── data/         # data loading and feature engineering
    │   ├── env/          # portfolio environment
    │   ├── experiments/  # experiment runners
    │   ├── models/       # actor, critic, TD3 agent
    │   ├── rewards/      # reward functions
    │   ├── risk/         # mandate profiles and risk penalty helpers
    │   ├── train/        # training loop
    │   └── utils/        # config and shared utilities
    ├── tests/            # unit tests
    ├── requirements.txt
    └── README.md

Generated data, outputs, logs, model artifacts, and paper build byproducts are excluded from version control by default.

## Running The Project

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
      --candidates V3_real_macro_vintage_clean_no_dxy \
      --output-dir outputs/tables/protocol_v3_clean_no_dxy

Run cap sensitivity:

    .venv/bin/python -m src.experiments.run_cap_sensitivity_experiment \
      --returns-path data/processed/returns_weekly_latest.csv \
      --output-dir outputs/tables/cap_sensitivity_experiment \
      --candidates V2_reference_full,V5_no_volatility_block,V6_financial_state \
      --episodes 60 \
      --seeds 7,21,42,84,101,123,202,303,404,505 \
      --max-weight-grid uncapped,0.50,0.60,0.70,0.80

Build the mandate profile comparison:

    .venv/bin/python -m src.analysis.mandate_profile_comparison_report \
      --final-report-dir outputs/tables/final_constrained_td3_report_with_v3_clean_no_dxy_v7_clean_garch_v4_v7_v8_60ep_10seeds \
      --output-dir outputs/tables/mandate_profile_comparison_final

Run the White Reality Check:

    .venv/bin/python -m src.analysis.white_reality_check_report \
      --final-report-dir outputs/tables/final_constrained_td3_report_with_v3_clean_no_dxy_v7_clean_garch_v4_v7_v8_60ep_10seeds \
      --v3-clean-no-dxy-cap-sensitivity-dir outputs/tables/cap_sensitivity_experiment_v3_clean_no_dxy_60ep_10seeds \
      --v3-vintage-cap-sensitivity-dir outputs/tables/cap_sensitivity_experiment_v3_vintage_60ep_10seeds \
      --v3-cap-sensitivity-dir outputs/tables/cap_sensitivity_experiment_v3_60ep_10seeds_seeded \
      --v4-cap-sensitivity-dir outputs/tables/cap_sensitivity_experiment_v4_60ep_10seeds \
      --v7-cap-sensitivity-dir outputs/tables/cap_sensitivity_experiment_v7_60ep_10seeds \
      --v7-clean-no-dxy-garch-cap-sensitivity-dir outputs/tables/cap_sensitivity_experiment_v7_clean_no_dxy_garch_60ep_10seeds \
      --v8-cap-sensitivity-dir outputs/tables/cap_sensitivity_experiment_v8_60ep_10seeds \
      --benchmarks BuyHold_GLD,trend_spy_cash_12p \
      --n-bootstrap 2000 \
      --block-length 8 \
      --seed 123 \
      --output-dir outputs/tables/white_reality_check_final

Build the paper:

    cd paper
    make

or:

    cd paper
    tectonic main.tex

## What I Am Testing

The current answer is nuanced:

- unconstrained TD3 is not enough
- simple benchmarks are very hard to beat
- max-weight constraints improve TD3 behavior substantially
- macro and GARCH features become useful mainly when the allocation problem is constrained
- adding more features does not automatically improve policy quality
- final claims must be based on out-of-sample protocol comparisons, not isolated backtests
- pairwise bootstrap and White Reality Check support caution rather than statistical dominance claims

## Academic Disclaimer

This repository is research code. It is not production trading software, financial advice, or an investment recommendation.

There is no deployable alpha claim here.

Any empirical claim in this project must be supported by reproducible experiments, chronological out-of-sample testing, benchmark comparison, sensitivity analysis, statistical validation, regime analysis, and audit checks.
