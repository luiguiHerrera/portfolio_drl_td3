# Publication plan

This note translates the structure of the reference portfolio-construction
paper into a roadmap for this TD3 portfolio-allocation manuscript.

## What the reference paper does well

- Opens with a clear problem: prediction helps, but prediction error creates
  losses; diversification is the control mechanism.
- Uses a compact workflow figure before the empirical results.
- Separates the empirical story into data, method, parameter sensitivity,
  transaction costs, portfolio metrics, and cumulative-return plots.
- Reports many tables, but each table has a specific job: data description,
  model comparison, threshold sensitivity, risk-return metrics, and costs.
- Uses figures to make the result readable quickly: cumulative returns,
  threshold sensitivity, and model contrasts.

## How to adapt that standard here

The paper should not simply add tables. It should use tables and figures to
make the constrained-TD3 claim auditable:

1. Position the contribution against portfolio ML and DRL work.
2. Show the workflow and timing convention visually.
3. Document the asset universe, macro state, exclusions, and diagnostics.
4. Separate model families from final selected candidates.
5. Report cap sensitivity as a core empirical result, not an appendix detail.
6. Compare against clean benchmarks and aggressive high-drawdown references.
7. Keep the bootstrap result visible so the claim remains publishable and
   honest.
8. Use regime figures to show where the strategy is and is not robust.

## Target table and figure set

Core tables:

- Positioning relative to closely related literature, especially TD3/DRL
  portfolio papers rather than generic ML studies.
- Investor mandate profiles and their quantitative limits.
- Data, assets, and feature families.
- Model families and selected caps.
- Main mandate-aware ranking.
- Cap-sensitivity summary.
- Clean benchmark comparison.
- Bootstrap validation.

Core figures:

- Empirical workflow and timing discipline.
- Mandate score vs. max drawdown.
- Effective assets vs. mandate score.
- Robust score vs. max drawdown.
- Regime mandate heatmap.
- Regime winners bar chart.

## Claim discipline

Allowed:

- Max-weight constraints materially reduce degenerate TD3 concentration.
- The clean no-DXY real-time/as-of macro specification is the leading
  mandate-aware constrained TD3 candidate.
- V7 clean no-DXY + GARCH is competitive but does not outperform the simpler V3
  clean macro model.
- Bootstrap validation does not establish statistically clear superiority over
  clean benchmarks.

Avoid:

- "TD3 beats the market."
- "The model proves alpha."
- "GARCH improves the policy."
- "DXY is part of the clean real-time macro result."
- Any claim that bootstrap validation proves benchmark dominance.

## Next writing steps

- Expand the related-work section with 12--20 carefully selected citations.
- Move some dense tables to an appendix once the main text becomes too heavy.
- Add an equity-curve figure if generated consistently from the final histories.
- Add a reproducibility appendix listing scripts, output directories, seeds,
  folds, and exact model identifiers.
- Decide target venue/style before final formatting, because table density and
  appendix conventions vary by journal.
- Expand the positioning table again once the final related-work set is frozen;
  the current comparison should remain focused on papers with similar portfolio
  construction, DRL, transaction-cost, risk-aversion, or constraint elements.
