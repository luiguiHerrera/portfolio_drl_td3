# Final Corrected TD3 Paper Draft Change Log

## Surgical Supervisor-Facing Polish Pass

Applied a targeted polish pass for supervisor review:

- Changed the main title from the more informal reality-check question to `Robust Evaluation of TD3 Portfolio Allocation under Realistic Cross-Asset Frictions`.
- Preserved the previous framing as a subtitle: `A Cross-Asset Reality Check under Costs, Cash, and Statistical Validation`.
- Clarified first abstract use of BIL-CASH as a short-term Treasury ETF proxy for remunerated cash.
- Renamed Table 6's transaction-cost column from `Mean transaction cost` to `Mean weekly transaction cost`; values were unchanged.
- Renamed `Recommended Figures` to `Manuscript Development Plan: Recommended Figures` so the section reads as paper-development guidance rather than final manuscript content.
- Reduced a small amount of repeated statistical-superiority wording in the statistical validation section without weakening the central caution.

## Metrics Filled

Filled Table 6, "TD3-only selected candidates and standard metrics", with reported test-split metrics for the two TD3-only selected candidates:

| Cash assumption | Candidate | Source file |
| --- | --- | --- |
| Zero-CASH | `V5_no_volatility_block_cap_0p50` | `outputs/tables/final_corrected_limited_td3_60ep_10seeds/cap_sensitivity_all_results.csv` |
| BIL-CASH | `V8_ewma_garch_vol_current_cap_0p70` | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds/cap_sensitivity_all_results.csv` |

Metrics added:

- annualized return;
- annualized volatility;
- Sharpe;
- maximum drawdown;
- average turnover;
- effective number of assets;
- mean transaction cost;
- mandate-aware score;
- robust score.

Also expanded the combined ranking table with standard metrics for:

| Cash assumption | Strategy | Source file |
| --- | --- | --- |
| Zero-CASH | `V3_real_macro_vintage_clean_no_dxy_cap_0p70` | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_benchmark_comparison/final_corrected_zero_cash_combined_ranking.csv` |
| Zero-CASH | `trend_spy_cash_12p` | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_benchmark_comparison/final_corrected_zero_cash_combined_ranking.csv` |
| BIL-CASH | `V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80` | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_benchmark_comparison/final_corrected_bil_cash_combined_ranking.csv` |
| BIL-CASH | `trend_spy_cash_12p` | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_benchmark_comparison/final_corrected_bil_cash_combined_ranking.csv` |

## Citation Placeholders Resolved

Replaced several broad citation placeholders with references already available in `paper/references.bib`:

- Portfolio theory: Markowitz (1952), Sharpe (1966), Merton (1969, 1971).
- Continuous-control RL and TD3: Silver et al. (2014), Lillicrap et al. (2015), Fujimoto et al. (2018).
- DRL portfolio allocation: Moody and Saffell (2001), Jiang et al. (2017), Almahdi and Yang (2017), Liu et al. (2020), Zhao et al. (2023), Jiang et al. (2024, 2025).
- Diversification-oriented ML portfolio construction: Chen et al. (2022).
- Data-snooping and backtest-overfitting controls: White (2000), Bailey and López de Prado (2014), López de Prado (2020).

## Gold/Bitcoin Citation Gap Resolved

Resolved the remaining gold/Bitcoin safe-haven citation gap by adding and citing:

- Bouri et al. (2017), on Bitcoin hedge, safe-haven, and diversifier properties.
- Henriques and Sadorsky (2018), on whether Bitcoin can replace gold in an investment portfolio.
- Guesmi et al. (2019), on portfolio diversification with Bitcoin and related hedging/spillover evidence.

Updated the supervisor-facing draft's citation-status section so it no longer lists the Bitcoin/gold safe-haven references as pending, and removed the stale open item asking to verify the remaining safe-haven placeholder.

No citation placeholders remain in the supervisor-facing draft.

## Repetition Reduced

Reduced repeated cautionary wording around statistical dominance while keeping the central thesis intact. The draft still clearly states that:

- TD3 remains competitive under the corrected protocol;
- statistical validation does not support superiority over clean benchmarks;
- custom robust and mandate-aware scores are diagnostic rather than standalone academic proof;
- the contribution is an evaluation framework, not a deployable alpha claim.

## Sections Materially Improved

- Abstract: tightened contribution statement and reduced repetitive negative framing.
- Introduction: added the supervisor-facing empirical finding sentence distinguishing ranking competitiveness from statistical dominance.
- Related Literature: sharpened the claim that TD3, DRL portfolio allocation, transaction costs, risk-aware objectives, statistical validation, and crypto/gold portfolio questions already exist; the contribution is the combined falsification-oriented evaluation stack.
- Results: filled missing standard metrics and added combined ranking metrics.
- Mandate/Pareto: reframed hard mandate infeasibility as a feasibility warning rather than a simple failure.
- Discussion and Conclusion: reduced repeated phrasing and kept the interpretation compact.
- Open Items: updated to reflect that the main TD3-only standard metrics have now been filled.

## Unresolved Items

- Decide whether the supervisor draft should include figures now or remain table-first.
- Confirm the professor's preferred TFM format and citation style.
- Decide whether the final title should emphasize "Reality Check" or "Robust Evaluation."
