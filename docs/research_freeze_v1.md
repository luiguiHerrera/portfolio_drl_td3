# Research Freeze v1

Date: 2026-05-28

This document freezes the current research state for the constrained TD3 portfolio allocation project. It is intended to prevent scope drift while the results are written up and interpreted.

## Current Leading Candidate

The current leading constrained TD3 candidate is:

- `V3_real_macro_vintage_clean_no_dxy_cap_0.50`

This candidate is the top mandate-aware constrained TD3 result in the current experimental set.

## Key Result

`V3_real_macro_vintage_clean_no_dxy_cap_0.50` is the strongest TD3 candidate under the mandate-aware evaluation layer.

The result does not imply that unconstrained TD3 dominates benchmarks. The central empirical pattern is that unconstrained TD3 policies tend toward degenerate concentration, while max-weight constrained TD3 variants become materially more competitive.

## Macro Specification

The leading V3 clean macro specification uses real-time/as-of FRED vintage macro data for:

- `DGS10`
- `DGS2`
- `VIX`
- `CPI`

The dollar proxy is excluded. No full-window fresh true-vintage dollar proxy was available for the 2015-2026 protocol window without fallback, discontinuation, or current-vintage relabeling.

This clean no-DXY specification is preferred over earlier macro variants because it avoids fallback macro series and avoids presenting a discontinued or revised dollar proxy as real-time/as-of evidence.

## V7 Clean No-DXY + GARCH

`V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50` is competitive, but it does not outperform the simpler V3 clean no-DXY candidate.

This supports the current interpretation that adding real fitted GARCH features to the clean macro state does not automatically improve TD3 policy quality. The simpler constrained V3 clean no-DXY candidate remains the leading mandate-aware result.

## Central Scientific Claim

The central claim is:

> Max-weight constraints materially reduce degenerate TD3 concentration and make TD3 more credible under mandate-aware portfolio evaluation.

The contribution is not that TD3 universally beats benchmarks. The contribution is that TD3 becomes more viable when the allocation problem is constrained in a way that reflects realistic portfolio mandate limits.

## Evidence Base

The current freeze is based on:

- cap sensitivity experiments across TD3 candidates;
- repeated seeds under the 60 episode x 10 seed protocol;
- comparison against the benchmark suite;
- mandate-aware scoring;
- statistical validation using available return histories;
- regime analysis using existing per-period histories.

The leading result is not a single isolated backtest. It is supported by cross-candidate cap sensitivity and by reporting layers designed to check robustness, concentration, drawdown behavior, and benchmark-relative performance.

## Cautions

The bootstrap validation does not show clear statistical superiority against the clean benchmark set.

The result is regime-sensitive. Constrained TD3 is competitive, but it is not universally dominant across all regimes.

The current findings are research evidence only. They are not a production trading claim, investment recommendation, or claim of deployable alpha.

## Allowed Claims

The following claims are allowed under this freeze:

- Unconstrained TD3 remains fragile and tends toward excessive concentration.
- Max-weight constraints materially improve TD3 allocation behavior.
- `V3_real_macro_vintage_clean_no_dxy_cap_0.50` is the current leading mandate-aware constrained TD3 candidate.
- The clean V3 macro specification uses real-time/as-of FRED vintage data for `DGS10`, `DGS2`, `VIX`, and `CPI`.
- The dollar proxy is excluded because no full-window fresh true-vintage dollar proxy exists for 2015-2026 without fallback or discontinuation.
- V7 clean no-DXY + GARCH is competitive but does not outperform the simpler V3 clean no-DXY candidate.
- Constrained TD3 is competitive under mandate-aware evaluation, but benchmark comparisons remain essential.

## Prohibited Claims

The following claims are prohibited under this freeze:

- TD3 universally dominates the benchmark suite.
- Unconstrained TD3 is a satisfactory final model.
- The clean leading V3 specification includes DXY.
- The macro data are full real-time vintage including a dollar proxy.
- Bootstrap validation proves clear statistical superiority over clean benchmarks.
- The result is a production trading strategy.
- The result is financial advice or an investment recommendation.
- Adding more econometric features necessarily improves TD3 policy quality.

## Next Work

The next work should focus on:

- writing the thesis/report narrative;
- refining the statistical validation interpretation;
- interpreting regime behavior carefully;
- preparing figures and tables for the paper/TFM structure;
- documenting limitations and benchmark-relative caveats.

Further model expansion is out of scope for this freeze.

## Freeze Rule

No new model families should be added unless explicitly opened as a new research branch.

The current phase is closed for model expansion. The project should now prioritize interpretation, validation, writing, and presentation.
