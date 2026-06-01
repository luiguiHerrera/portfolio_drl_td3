# Portfolio DRL TD3

Master's thesis research code for dynamic portfolio allocation with **Twin Delayed Deep Deterministic Policy Gradient (TD3)**.

This repository asks a practical question:

> Can TD3-based dynamic portfolio allocation become mandate-credible once concentration, drawdown, turnover, benchmark comparison, regime sensitivity, transaction costs, and statistical uncertainty are treated as first-class constraints?

The answer is not "TD3 beats the market."

The honest result is narrower and more useful: unconstrained TD3 often collapses into concentrated, fragile policies. Max-weight constraints materially stabilize the action space. Under more realistic asset-specific transaction costs, the preferred TD3 specification changes again. Constrained TD3 can become mandate-credible, but it does not statistically dominate clean benchmarks.

## Asset Universe

The agent allocates weekly across:

- `SPY` - U.S. equities
- `TLT` - long-duration U.S. Treasuries
- `GLD` - gold
- `BTC-USD` - Bitcoin
- `CASH` - synthetic zero-return cash

The portfolio is long-only and fully invested. At each decision date, the state uses information available through `t-1`, the policy chooses weights for period `t`, and the realized portfolio return is observed at `t`.

## Model Families

Each feature specification is trained as a separate TD3 policy under the same walk-forward protocol. It is not one model reused across incompatible state spaces.

Current candidate families:

- `V2_reference_full` - rich financial/reference state
- `V3_real_macro_current` - current-vintage macro comparison
- `V3_real_macro_vintage_clean_no_dxy` - clean real-time/as-of macro state
- `V4_real_garch_current` - rolling fitted GARCH volatility state
- `V5_no_volatility_block` - no-volatility ablation
- `V6_financial_state` - parsimonious financial state
- `V7_real_macro_vintage_clean_no_dxy_garch` - clean macro plus GARCH
- `V8_ewma_garch_vol_current` - EWMA/GARCH volatility hybrid

Current-vintage and DXY-fallback macro variants remain in the audit trail. The clean no-DXY macro specification is the clean macro evidence.

## Macro and GARCH Data

The clean macro specification uses real-time/as-of FRED vintage data for:

- `DGS10`
- `DGS2`
- `VIX`
- `CPI`

The dollar proxy is excluded because no full-window fresh true-vintage dollar proxy was available for 2015-2026 without fallback, discontinuation, or current-vintage relabeling. The clean specification does not include DXY.

GARCH candidates use rolling one-step-ahead forecasts via `arch_model`, with zero-mean normal GARCH(1,1), weekly volatility, and forecasts at `t` based only on returns through `t-1`. CASH is excluded from fitted GARCH estimation. Warmup fallback uses rolling realized volatility only when history is insufficient.

## Transaction Costs

The earlier protocol used scalar proportional turnover costs.

The current asset-specific-cost branch adds an explicit cost-aware layer:

- `transaction_cost_mode = asset_specific`
- `SPY`: 2 bps
- `TLT`: 2 bps
- `GLD`: 2 bps
- `BTC-USD`: 10 bps
- `CASH`: 0 bps

These are broker/exchange-style trading-cost proxies. They do not model fiat ramps, exchange transfers, withdrawal fees, custody frictions, taxes, market impact, or delays between broker and crypto-exchange accounts.

Scalar-cost and asset-specific-cost results should not be mixed casually. They are related experiments, not interchangeable rankings.

## Evaluation

The project reports standard portfolio metrics:

- cumulative and annualized return
- volatility, Sharpe, Sortino, Calmar
- max drawdown and worst drawdown
- turnover and transaction costs
- effective number of assets
- average max weight and cash exposure

It also includes reporting layers for:

- robust score
- mandate-aware score
- statistical bootstrap validation
- White Reality Check
- regime analysis
- mandate-profile sensitivity
- transaction-cost sensitivity
- final figures

The White Reality Check tests mean return differentials after accounting for model search. It is not an SPA test and it does not test mandate-aware score directly.

## Current Research Status

### Scalar-Cost Result

Under the scalar-cost protocol, the leading clean mandate-aware TD3 candidate was:

- `V3_real_macro_vintage_clean_no_dxy_cap_0.50`

That result supported the main stabilization claim: max-weight constraints reduce degenerate concentration and make TD3 more mandate-credible.

### Asset-Specific-Cost Result

Under official full asset-specific-cost-aware TD3 revalidation, the TD3-only selected leader changed to:

- `V3_real_macro_vintage_clean_no_dxy_cap_0.70`

After combining selected TD3 candidates with deterministic benchmarks under the same asset-specific cost model, the best overall and best TD3 strategy by recomputed combined mandate-aware ranking became:

- `V5_no_volatility_block_cap_0.50`

The best benchmark in that combined universe is:

- `trend_spy_cash_12p`

Mandate-profile winners under asset-specific costs:

- Conservative: `V4_real_garch_current_cap_0.50`
- Moderate: `V5_no_volatility_block_cap_0.50`
- Aggressive: `V5_no_volatility_block_cap_0.50`

Statistical validation and White Reality Check do not support a statistical superiority claim for `V5_no_volatility_block_cap_0.50` over clean benchmarks. Regime analysis also shows the V5 advantage is regime-specific, not broad dominance.

## What The Experiments Taught Me

1. Unconstrained TD3 concentration is a real failure mode.
2. Max-weight caps are not cosmetic. They change allocation behavior, drawdown, turnover, and diversification.
3. Econometric features help only when the action space is controlled.
4. More features do not automatically improve the policy. V7 and V8 are useful mixed results.
5. Asset-specific transaction costs matter. They changed the preferred TD3 specification.
6. Benchmarks are still hard to beat. That is the point of using serious benchmarks.
7. No statistical dominance claim is supported. The defensible claim is mandate credibility, not market conquest.

## Key Outputs

Generated outputs are excluded from version control by default.

Main scalar-cost final report:

- `outputs/tables/final_constrained_td3_report_with_v3_clean_no_dxy_v7_clean_garch_v4_v7_v8_60ep_10seeds`

Asset-specific-cost reporting layer:

- `outputs/tables/asset_specific_cost_full_final_report`
- `outputs/tables/asset_specific_cost_benchmark_comparison`
- `outputs/tables/asset_specific_cost_statistical_validation`
- `outputs/tables/asset_specific_cost_white_reality_check`
- `outputs/tables/asset_specific_cost_regime_analysis`
- `outputs/tables/asset_specific_cost_mandate_profile_comparison`
- `outputs/figures/asset_specific_cost_final`

Paper artifacts:

- `paper/main.tex`
- `paper/main.pdf`
- `paper/references.bib`
- `paper/figures/`
- `paper/scripts/`

## Repository Structure

    portfolio_drl_td3/
    ├── configs/          # YAML experiment configuration
    ├── docs/             # research log, design notes, freeze notes
    ├── notebooks/        # exploratory notebooks
    ├── paper/            # manuscript, figures, bibliography, scripts
    ├── scripts/          # standalone data acquisition / preparation scripts
    ├── src/
    │   ├── analysis/     # reports, audits, validation, figures
    │   ├── backtest/     # deterministic benchmark logic
    │   ├── data/         # data loading and feature engineering
    │   ├── env/          # portfolio environment
    │   ├── experiments/  # experiment runners
    │   ├── models/       # actor, critic, TD3 agent
    │   ├── rewards/      # reward functions
    │   ├── risk/         # mandate profiles and risk helpers
    │   ├── train/        # training loop
    │   └── utils/        # config and shared utilities
    ├── tests/
    ├── requirements.txt
    └── README.md

## Running The Project

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Run tests:

    .venv/bin/python -m unittest discover tests

Run a selected TD3 candidate:

    .venv/bin/python -m src.experiments.run_protocol_pure_td3_revalidation \
      --returns-path data/processed/returns_weekly_latest.csv \
      --episodes 60 \
      --seeds 7,21,42,84,101,123,202,303,404,505 \
      --candidates V3_real_macro_vintage_clean_no_dxy \
      --output-dir outputs/tables/protocol_v3_clean_no_dxy

Build the official asset-specific TD3 full report from completed histories:

    .venv/bin/python -m src.analysis.asset_specific_cost_final_report \
      --v2-v6-dir outputs/tables/asset_specific_cost_full_final_candidates_60ep_10seeds \
      --v7-dir outputs/tables/asset_specific_cost_v7_full_grid_60ep_10seeds \
      --v8-dir outputs/tables/asset_specific_cost_v8_full_grid_60ep_10seeds \
      --output-dir outputs/tables/asset_specific_cost_full_final_report

Build asset-specific deterministic benchmarks:

    .venv/bin/python -m src.experiments.run_protocol_benchmark_comparison \
      --returns-path data/processed/returns_weekly_latest.csv \
      --output-dir outputs/tables/asset_specific_cost_benchmark_comparison/benchmarks \
      --base-config-path outputs/tables/asset_specific_cost_limited_retraining/asset_specific_config.yaml

Build the combined asset-specific TD3 vs benchmark report:

    .venv/bin/python -m src.analysis.asset_specific_cost_benchmark_comparison_report \
      --td3-report-dir outputs/tables/asset_specific_cost_full_final_report \
      --benchmark-dir outputs/tables/asset_specific_cost_benchmark_comparison/benchmarks \
      --output-dir outputs/tables/asset_specific_cost_benchmark_comparison

Run asset-specific statistical validation:

    .venv/bin/python -m src.analysis.statistical_validation_report \
      --final-report-dir outputs/tables/asset_specific_cost_full_final_report \
      --benchmark-dir outputs/tables/asset_specific_cost_benchmark_comparison/benchmarks \
      --asset-specific-only \
      --output-dir outputs/tables/asset_specific_cost_statistical_validation

Run asset-specific White Reality Check:

    .venv/bin/python -m src.analysis.white_reality_check_report \
      --final-report-dir outputs/tables/asset_specific_cost_full_final_report \
      --benchmark-dir outputs/tables/asset_specific_cost_benchmark_comparison/benchmarks \
      --benchmarks trend_spy_cash_12p,BuyHold_GLD,Equal_Weight \
      --n-bootstrap 2000 \
      --block-length 8 \
      --seed 123 \
      --asset-specific-only \
      --output-dir outputs/tables/asset_specific_cost_white_reality_check

Run asset-specific regime analysis:

    .venv/bin/python -m src.analysis.regime_analysis_report \
      --final-report-dir outputs/tables/asset_specific_cost_full_final_report \
      --benchmark-dir outputs/tables/asset_specific_cost_benchmark_comparison/benchmarks \
      --asset-specific-only \
      --output-dir outputs/tables/asset_specific_cost_regime_analysis

Run asset-specific mandate-profile comparison:

    .venv/bin/python -m src.analysis.mandate_profile_comparison_report \
      --final-report-dir outputs/tables/asset_specific_cost_full_final_report \
      --combined-report-dir outputs/tables/asset_specific_cost_benchmark_comparison \
      --benchmark-dir outputs/tables/asset_specific_cost_benchmark_comparison/benchmarks \
      --asset-specific-only \
      --output-dir outputs/tables/asset_specific_cost_mandate_profile_comparison

Build asset-specific final figures:

    .venv/bin/python -m src.analysis.build_asset_specific_cost_figures \
      --td3-report-dir outputs/tables/asset_specific_cost_full_final_report \
      --benchmark-comparison-dir outputs/tables/asset_specific_cost_benchmark_comparison \
      --statistical-validation-dir outputs/tables/asset_specific_cost_statistical_validation \
      --white-reality-check-dir outputs/tables/asset_specific_cost_white_reality_check \
      --regime-analysis-dir outputs/tables/asset_specific_cost_regime_analysis \
      --mandate-profile-dir outputs/tables/asset_specific_cost_mandate_profile_comparison \
      --output-dir outputs/figures/asset_specific_cost_final

Build the paper:

    cd paper
    make

or:

    cd paper
    tectonic main.tex

## Academic Disclaimer

This is research code. It is not production trading software, financial advice, or an investment recommendation.

There is no deployable alpha claim here.

Any empirical claim in this project must be backed by reproducible experiments, chronological out-of-sample testing, benchmark comparison, sensitivity analysis, statistical validation, regime analysis, and audit checks.
