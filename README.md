# Robust TD3 Portfolio Allocation under Realistic Trading Frictions

This repository contains Master's thesis research code for TD3-based Deep Reinforcement Learning in portfolio allocation.

The project does not try to market a trading bot. It asks a narrower question: can a continuous-action TD3 portfolio policy become credible once transaction costs, cash assumptions, concentration, drawdown, turnover, benchmarks, regimes, and statistical uncertainty are treated seriously?

The final corrected answer is cautious. TD3 is competitive and sometimes top-ranked, but it does not pass statistical superiority tests against clean benchmarks. The strongest contribution is the evaluation framework and the corrected protocol, not a fake alpha claim.

## What This Project Tests

- TD3 for weekly long-only portfolio allocation.
- Financial, macro, volatility, and hybrid feature families.
- Asset-specific transaction costs.
- Synthetic zero-return `CASH` versus a BIL short-term Treasury ETF cash proxy.
- Deterministic benchmark strategies under matching cost assumptions.
- Bootstrap validation, White Reality Check, regime analysis, mandate profiles, and Pareto analysis.

## Why This Is Not Just a Toy Backtest

The final corrected experiments use:

- walk-forward-style evaluation;
- 10 random seeds;
- 4 folds;
- 60 training episodes;
- 800 TD3 histories per cash assumption;
- 14 deterministic benchmarks per cash assumption;
- explicit transaction costs and turnover diagnostics;
- max-weight caps as TD3 structural constraints;
- canonical mandate profiles;
- bootstrap statistical validation and White Reality Check;
- regime, mandate-profile, and Pareto reporting layers.

Generated outputs are not fully committed to git. The repository keeps code, tests, docs, and audit tooling; large experiment outputs live under `outputs/` or an external output directory such as `~/Projects/portfolio_drl_outputs`.

## Final Corrected Protocol

| Component | Final corrected choice |
|---|---|
| Assets | `SPY`, `TLT`, `GLD`, `BTC-USD`, `CASH` |
| Portfolio | weekly, long-only, fully invested |
| Asset-specific costs | `SPY/TLT/GLD`: 2 bps, `BTC-USD`: 10 bps |
| Synthetic CASH | zero return, 0 bps cost |
| BIL-CASH robustness | BIL proxy return, 2 bps cost on `CASH` sleeve |
| Reward | net-return-first: full transaction costs enter financial net return |
| Risk shaping | drawdown penalty active; turnover and concentration evaluated through diagnostics/mandates |
| Actions | cap-consistent behavior action, executed action, replay action, target smoothing, and actor-loss critic evaluation |
| Exploration | behavior-policy exploration noise during training only |
| Macro data | clean vintage/as-of FRED macro with required sidecar metadata |
| Mandate constraints | max drawdown, max annualized volatility, min effective assets, max average turnover |

`max_weight` is not an official mandate constraint. It is a structural TD3 training/evaluation intervention used to control degenerate concentration.

## Feature Families

Each state specification is trained as a separate TD3 policy under the same protocol.

- `V2_reference_full`: rich financial/reference state.
- `V3_real_macro_vintage_clean_no_dxy`: clean real-time/as-of macro state.
- `V4_real_garch_current`: rolling fitted GARCH volatility state.
- `V5_no_volatility_block`: no-volatility ablation.
- `V6_financial_state`: parsimonious financial state.
- `V7_real_macro_vintage_clean_no_dxy_garch`: clean macro plus GARCH.
- `V8_ewma_garch_vol_current`: EWMA/GARCH volatility hybrid.

The clean macro specification uses `DGS10`, `DGS2`, `VIX`, and `CPI`. DXY is excluded because no full-window fresh true-vintage dollar proxy was available for 2015-2026 without fallback, discontinuation, or current-vintage relabeling.

The corrected clean macro pipeline requires traceability sidecar metadata. CPI YoY is computed on monthly CPI before weekly alignment. The old CPI 12-week momentum feature was removed from the clean final specification.

V6 heuristic probability-like variables were renamed as scores. Mechanical duplicate/self-reference features were removed. PCA was audited but not added as a default protocol step.

## Main Results

Different reporting layers answer different questions:

- TD3-only cap sensitivity asks which TD3 variant is best.
- Combined ranking asks how selected TD3 policies compare with benchmarks.
- White Reality Check asks whether searched TD3 superiority is statistically supported.
- Mandate/Pareto analysis asks whether strategies survive practical constraints.

### A. TD3-Only Cap Sensitivity

Under synthetic zero-CASH:

- winner: `V5_no_volatility_block_cap_0p50`
- mandate-aware score: `0.601124`
- robust score: `0.696702`

Under BIL-CASH robustness:

- winner: `V8_ewma_garch_vol_current_cap_0p70`
- mandate-aware score: `0.660435`
- robust score: `0.749958`

The cash-return assumption materially changes TD3 model selection.

### B. TD3 + Benchmark Ranking

Under synthetic zero-CASH:

- best overall: `V3_real_macro_vintage_clean_no_dxy_cap_0p70`

Under BIL-CASH robustness:

- best overall: `V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80`

Best benchmark in both settings:

- `trend_spy_cash_12p`

These rankings are useful, but they are not statistical superiority evidence.

### C. Statistical Validation

Top TD3 versus `trend_spy_cash_12p`:

| Cash assumption | Sharpe delta | Bootstrap CI | P(TD3 beats) | WRC p-value |
|---|---:|---:|---:|---:|
| Zero-CASH | `0.1559` | `[-0.6011, 0.9767]` | `0.629` | `0.7136` |
| BIL-CASH | `0.1170` | `[-0.7172, 0.9963]` | `0.588` | `0.6767` |

The conclusion is direct: TD3 does not survive statistical superiority tests. No statistical dominance claim should be made.

### D. Regime, Mandate, and Pareto Results

Regime analysis shows selected TD3 competitiveness, not broad dominance. Benchmarks still win important slices.

Constraint-first and mandate-profile analysis use canonical hard constraints:

- maximum drawdown;
- maximum annualized volatility;
- minimum effective number of assets;
- maximum average turnover.

No strategy passes all hard canonical mandate filters. This is uncomfortable, but useful: the final result is not that TD3 solved the mandate problem. The result is that corrected TD3 policies can be competitive research candidates, while practical constraints remain binding.

## Interpretation

The corrected protocol changed the story.

Earlier scalar-cost results suggested a clean macro TD3 candidate was the main leader. After asset-specific costs, cap-consistent actions, net-return-first reward, corrected macro alignment, benchmark regeneration, and robustness checks, the preferred model depends on the cash assumption and reporting layer.

The honest summary:

- TD3 is competitive.
- TD3 sometimes ranks first.
- TD3 does not statistically dominate clean benchmarks.
- Cash assumptions matter.
- Regime behavior matters.
- The evaluation framework is the main contribution.

## Repository Structure

```text
configs/      experiment configuration
src/          data, environment, models, training, analysis, risk, backtests
tests/        unit and integration tests
scripts/      standalone audit/data/report helpers
docs/         research notes and design documents
paper/        manuscript files, figures, bibliography
outputs/      generated artifacts; not fully versioned
```

## How To Reproduce

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the test suite:

```bash
.venv/bin/python -m unittest discover tests
```

Generate the audit pack:

```bash
.venv/bin/python scripts/create_tfm_audit_pack.py \
  --external-outputs-dir ~/Projects/portfolio_drl_outputs \
  --output-dir tfm_audit_pack \
  --zip-name tfm_audit_pack.zip
```

The heavy final-corrected experiments are not rerun by this README. They are generated under `outputs/` or the external output directory and summarized by the audit pack.

## Audit Status

Latest audit pack:

- 22 audit checks passed.
- Zero-CASH histories: 800/800.
- BIL-CASH histories: 800/800.
- Zero-CASH benchmarks: 14.
- BIL-CASH benchmarks: 14.
- `compileall` OK.
- unittest suite OK: 1,258 tests in the latest recorded full run.

## Important Caveats

- This is research code, not production trading software.
- It is not financial advice or an investment recommendation.
- There is no deployable alpha claim.
- BIL-CASH is a robustness proxy, not exact cash execution.
- Asset-specific costs are broker/exchange-style proxies; they do not model taxes, custody frictions, withdrawal fees, market impact, or transfer delays.
- Statistical validation and White Reality Check do not support a superiority claim.
- Data, cash, cost, and execution assumptions matter.

## Future Work

- Calibrated probability features for the financial-state block using logistic regression, Platt scaling, isotonic calibration, Brier score, and reliability curves.
- Larger asset universes.
- Stricter slippage and execution modeling.
- Paper-trading or live-forward validation.
- Alternative continuous-control RL agents.
- More formal treatment of benchmark uncertainty and regime transitions.
