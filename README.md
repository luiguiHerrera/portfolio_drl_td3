# Portfolio DRL TD3 - Realistic Evaluation of DRL Portfolio Claims

This repository contains a research case study on TD3 for cross-asset portfolio
allocation. The point is not to present TD3 as a trading system. The point is to
test whether deep reinforcement learning portfolio claims survive a stricter
evaluation protocol: realistic transaction costs, explicit cash assumptions,
matched deterministic benchmarks, out-of-sample validation, and statistical
checks for searched strategy performance.

The current paper is titled:

> Evaluating DRL Portfolio Claims under Realistic Cross-Asset Frictions: A TD3
> Case Study with Costs, Cash, Matched Benchmarks, and Statistical Validation

The final answer is cautious. Selected TD3 candidates can be competitive under
mandate-aware and robust diagnostic rankings, but the evidence does not support
a clean statistical superiority claim over deterministic benchmarks. Bootstrap
Sharpe-difference intervals cross zero. White Reality Check does not support a
strong alpha claim. The main contribution is the evaluation discipline, not a
deployable trading signal.

## Why This Project Exists

DRL finance results can look stronger than they are when the evaluation setup is
too loose. Common problems include weak benchmarks, ambiguous cash handling,
simplified transaction costs, model-search bias, limited seed testing, and
little statistical validation.

This repo stress-tests a TD3 portfolio allocator under those conditions. It asks
what remains after the backtest is forced through a more skeptical evidence
stack.

## Research Question

Can a disciplined evaluation protocol distinguish statistically credible and
practically feasible DRL portfolio performance from apparent backtest strength?

## What Is Being Tested

The learning model is TD3, used as a continuous-action portfolio allocator. At a
weekly decision frequency, the agent chooses long-only portfolio weights over a
compact cross-asset universe:

| Asset | Role in the allocation problem |
| --- | --- |
| `SPY` | Equity / growth risk |
| `TLT` | Duration / interest-rate risk |
| `GLD` | Real safe-haven / hard-asset exposure |
| `BTC-USD` | Digital alternative / speculative convexity |
| `CASH` / `BIL` | Defensive liquidity / optionality |

The project evaluates two cash protocols:

- `Zero-CASH`: synthetic zero-return cash sleeve with 0 bps cash transaction
  cost.
- `BIL-CASH`: short-term Treasury ETF proxy for cash, with 2 bps cash
  transaction cost.

The corrected cost schedule uses asset-specific transaction costs:

- `SPY`, `TLT`, `GLD`: 2 bps
- `BTC-USD`: 10 bps
- `CASH`: 0 bps under Zero-CASH, 2 bps under BIL-CASH

## Evaluation Stack

The final paper separates ranking, statistical credibility, and practical
feasibility. Those are different claims.

The evaluation stack includes:

- out-of-sample walk-forward evaluation
- multiple random seeds and folds
- asset-specific transaction costs
- explicit Zero-CASH and BIL-CASH assumptions
- regenerated deterministic benchmarks under matching assumptions
- bootstrap Sharpe-difference intervals
- White Reality Check for searched candidate performance
- mandate and Pareto feasibility analysis
- regime dependence analysis
- execution-spread stress tests
- training-budget convergence checks

## Benchmarks

The benchmark set is deliberately not weak. It includes static allocations,
single-asset buy-and-hold exposures, momentum rules, risk-off rules, risk
parity, and rolling Markowitz-style optimizers.

Deterministic benchmarks used in the final corrected comparison include:

- `60_40_SPY_TLT`
- `BuyHold_BTC-USD`
- `BuyHold_GLD`
- `BuyHold_SPY`
- `BuyHold_TLT`
- `Equal_Weight`
- `Equal_Weight_Risky`
- `defensive_risk_off_12p`
- `momentum_winner_12p`
- `risk_adjusted_momentum_winner_12p_12p`
- `rolling_markowitz_long_only_52p`
- `rolling_markowitz_min_variance_52p`
- `rolling_risk_parity_inverse_vol_12p`
- `trend_spy_cash_12p`

## Main Findings

The findings are intentionally modest:

- TD3 can be competitive under some mandate-aware and robust diagnostic
  rankings.
- TD3 does not establish statistical superiority over the deterministic
  benchmarks in the final validation layer.
- `trend_spy_cash_12p` remains a serious benchmark comparator in both cash
  protocols.
- Cash assumptions matter. Zero-CASH and BIL-CASH select different preferred
  TD3 candidates.
- Costs and execution-spread assumptions materially affect interpretation.
- No strategy passes all hard canonical mandate filters in the final
  constraint-first analysis.
- More training does not automatically improve the selected candidates; the
  convergence check does not show that the 60-episode protocol is obviously
  undertrained.
- The value of the repo is the evaluation protocol and the traceable research
  workflow, not a claim that TD3 is a reliable trading edge.

## What This Repo Demonstrates

For reviewers, quant researchers, and recruiters, the useful part of this repo
is the research discipline around the model:

- PyTorch TD3 implementation for continuous portfolio weights
- portfolio environment design with feasible action handling
- net-return-first accounting with transaction costs
- feature-family experimentation across financial, macro, volatility, and
  hybrid states
- walk-forward validation across seeds and folds
- deterministic benchmark regeneration under matched assumptions
- bootstrap and White Reality Check validation
- mandate-aware, robust-score, Pareto, regime, and execution-friction reporting
- evidence traceability documentation for paper tables and figures
- a paper workflow that links narrative claims back to source files

## What This Repo Does Not Claim

This repository does not claim:

- that TD3 reliably beats deterministic portfolio benchmarks
- that the result is statistically dominant after model search
- that the code is a production trading system
- that this is investment advice
- that the asset universe is a complete global allocation universe
- that the backtest models taxes, custody, intraday liquidity, order-book depth,
  or full market impact
- that the results have been live-forward validated

## Repository Map

```text
configs/      experiment configuration files
data/         raw, interim, and processed data files
docs/         research notes, paper framing, and evidence traceability
notebooks/    notebook placeholder area
outputs/      generated experiment and report artifacts
paper/        LaTeX manuscript, rendered PDF, references, and paper notes
scripts/      data, recovery, and robustness helper scripts
src/          implementation code
tests/        unittest coverage for data, env, TD3, benchmarks, and reports
```

Important implementation areas:

```text
src/models/        actor, critic, TD3 agent
src/env/           portfolio environment
src/train/         TD3 training and exploration
src/data/          datasets, features, macro data, walk-forward splits
src/backtest/      benchmarks, metrics, policy evaluation
src/analysis/      final reports, audits, statistical validation, robustness
src/costs/         spread-cost utilities
src/risk/          mandate profiles and penalties
```

## Paper And Evidence Traceability

Because this project went through several experiment iterations, final paper
claims are documented through source-map and audit files under `docs/`. Readers
who want to verify table or figure lineage should start with:

- `docs/final_output_source_map.md`
- `docs/final_paper_full_audit.md`

Some heavy experiment artifacts are not intended to be regenerated from the
README. The paper and audit docs identify the relevant final evidence files.

## How To Read The Paper

The polished research narrative is in:

- `paper/main.pdf`
- `paper/main.tex`

The paper explains the motivation, methodology, results, and interpretation.
The docs explain traceability:

- `docs/paper_results_narrative.md`
- `docs/paper_reporting_layers.md`
- `docs/paper_claim_vs_evidence.md`
- `docs/final_output_source_map.md`
- `docs/output_lineage_audit.md`

Read the paper for the argument. Read the source-map docs when checking whether
a table or claim is supported by final evidence.

## Reproducibility And Usage

The repository has a Python dependency file:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the test suite:

```bash
.venv/bin/python -m unittest discover tests
```

The paper is available directly at `paper/main.pdf`, with the LaTeX source in
`paper/main.tex`.

Heavy final experiments are documented through the paper and evidence
traceability files under `docs/`. They are not presented here as a one-command
production pipeline.

## License And Use

This is research code for portfolio-allocation evaluation. It is not financial
advice, not a trading recommendation, and not a production execution system.
