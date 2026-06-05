# Results Narrative

This note gives the order and logic for the results section. The section should read as a sequence of increasingly strict filters, not as a chronological story of fixes.

## A. TD3-Only Cap Sensitivity

Start inside the TD3 universe. Report standard metrics first: annualized return, annualized volatility, Sharpe, max drawdown, turnover, effective assets, and average max weight. Then report robust and mandate-aware scores as diagnostics.

Key message:

- Zero-CASH TD3-only winner: `V5_no_volatility_block_cap_0p50`.
- BIL-CASH TD3-only winner: `V8_ewma_garch_vol_current_cap_0p70`.
- Caps materially change behavior and selection.
- This layer does not answer whether TD3 beats benchmarks.

## B. Combined TD3 + Benchmark Ranking

Next combine selected TD3 candidates with deterministic benchmarks regenerated under the same cost and cash assumptions. This is the first layer that makes TD3 compete with clean benchmark strategies.

Key message:

- Zero-CASH combined winner: `V3_real_macro_vintage_clean_no_dxy_cap_0p70`.
- BIL-CASH combined winner: `V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80`.
- Best benchmark in both combined settings: `trend_spy_cash_12p`.
- TD3 is competitive, but the preferred TD3 variant depends on the cash assumption.

## C. Statistical Validation

Bootstrap and White Reality Check are the hard statistical judge. They should appear after rankings, because they test whether the observed ranking advantage survives uncertainty and model search.

Key message:

- Bootstrap intervals do not establish clear TD3 superiority over clean benchmarks.
- White Reality Check does not support searched TD3 dominance.
- Do not phrase top-ranked TD3 as statistically superior.

## D. Regime Analysis

Regime analysis asks whether performance is broad or concentrated in particular market environments.

Key message:

- TD3 can be competitive in some regimes.
- Benchmark strength remains important in specific regimes.
- Regime behavior supports caution, not all-weather dominance.

## E. Mandate / Pareto

Mandate and Pareto analysis should use hard canonical mandate filters first, then standard metric rankings among feasible or non-dominated strategies. Custom scores can be reported afterward.

Key message:

- Official mandate filters use max drawdown, annualized volatility, effective assets, and turnover.
- Max weight is a structural training/evaluation cap, not an official mandate constraint.
- No strategy passes all hard canonical mandate filters.
- Some TD3 policies remain Pareto-competitive.

## F. Execution-Spread Robustness

Spread robustness is a post-training stress test. It should not create new winners or alter the main ranking.

Key message:

- Selected TD3 histories are sensitive to additional spread costs.
- Under stress spreads, selected TD3 strategies degrade more than `trend_spy_cash_12p`.
- This strengthens execution realism and weakens any temptation to overclaim.

## G. Training-Budget Convergence

Training-budget convergence checks whether 60 episodes looks obviously undertrained.

Key message:

- The 5-seed convergence check does not support a required longer-budget rerun.
- 100 and 150 episodes do not materially improve Sharpe for selected candidates.
- Longer budgets often increase turnover or reduce Sharpe.
- A 10-seed extension is optional publication-grade confirmation, not a prerequisite for the TFM narrative.

## Writing Rules

- Put standard metrics before custom scores.
- Treat robust and mandate-aware scores as diagnostics.
- Treat bootstrap and White Reality Check as the statistical gatekeeper.
- Do not turn spread robustness or convergence robustness into model-selection layers.
- Do not imply statistical superiority.
- Do not narrate the project as a sequence of bugs fixed.
