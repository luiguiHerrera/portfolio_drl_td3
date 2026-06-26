# Output Lineage Audit

This audit records detailed notes from the final-output organization pass. It
does not modify empirical CSVs or move/delete outputs.

## Scope

Inspected:

- `docs/research_log.md`
- `docs/paper_results_narrative.md`
- `docs/paper_reporting_layers.md`
- `docs/paper_claim_vs_evidence.md`
- `paper/main.tex`
- final external output roots under `/Users/thiagoherrera/Projects/portfolio_drl_outputs/`
- repo-local output folders under `outputs/tables/` and `outputs/figures/`

## Main Findings

1. The final evidence stack is mostly traceable to the external
   `final_corrected_*` roots listed in `docs/final_output_source_map.md`.
2. Table 7 has split lineage: Zero-CASH TD3-only selection uses the repo-local
   `outputs/tables/final_corrected_limited_td3_60ep_10seeds/`, while BIL-CASH
   TD3-only selection uses the external
   `final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds/`.
3. The Zero-CASH benchmark-comparison metadata explicitly names
   `outputs/tables/final_corrected_limited_td3_60ep_10seeds` as its TD3 input,
   so that repo-local folder is canonical only as an upstream component.
4. Table 8 values match the external combined-ranking CSVs after rounding.
5. Table 8 should be interpreted as mandate-aware combined ranking. In
   Zero-CASH, `V3_real_macro_vintage_clean_no_dxy_cap_0p70` is rank 1 by
   mandate-aware score, while `V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80`
   has higher Sharpe and is the WRC best candidate by mean differential.
6. Table 9 values match the final bootstrap and WRC files after rounding, but
   the WRC p-values are candidate-set/search-adjusted results. The Zero-CASH WRC
   p-value is not a pair-specific p-value for the V3 row.
7. Figure 2 now uses the final split Zero-CASH and BIL-CASH regime-winner
   sources under `/Users/thiagoherrera/Projects/portfolio_drl_outputs/`.
   The previous repo-local non-split source has been superseded in the current
   paper.
8. Table 10 is supported by the final external constraint/Pareto and
   mandate-profile folders.
9. Section 7.3 execution-spread robustness matches the final external
   execution-spread folder.
10. Section 7.4 training-budget convergence matches the final external
    convergence folder.

## Detailed Verification Notes

### Table 7 TD3-Only Candidate Selection

Zero-CASH row:

- Source: `outputs/tables/final_corrected_limited_td3_60ep_10seeds/cap_sensitivity_all_results.csv`
- Row: `V5_no_volatility_block_cap_0p50`
- Verified values:
  - annualized_return `0.0667285458250577` -> paper `0.0667`
  - annualized_volatility `0.1156123336805923` -> paper `0.1156`
  - sharpe `0.5686403910199787` -> paper `0.5686`
  - max_drawdown `-0.1206362879433997` -> paper `-0.1206`
  - average_turnover `0.419475576098095` -> paper `0.4195`
  - average_effective_number_of_assets `2.809332442509802` -> paper `2.8093`
  - mean_transaction_cost `0.0001151414080626` -> paper `0.000115`
  - mandate_aware_score `0.6011242192614834` -> paper `0.601124`
  - robust_score `0.6967018827278302` -> paper `0.696702`

BIL-CASH row:

- Source: `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds/cap_sensitivity_all_results.csv`
- Row: `V8_ewma_garch_vol_current_cap_0p70`
- Verified values:
  - annualized_return `0.07311290017609237` -> paper `0.0731`
  - annualized_volatility `0.10604668359235567` -> paper `0.1060`
  - sharpe `1.025294552073214` -> paper `1.0253`
  - max_drawdown `-0.1066407291516199` -> paper `-0.1066`
  - average_turnover `0.3450304786512213` -> paper `0.3450`
  - average_effective_number_of_assets `2.004438919725421` -> paper `2.0044`
  - mean_transaction_cost `0.00010320090401739251` -> paper `0.000103`
  - mandate_aware_score `0.6604353942543875` -> paper `0.660435`
  - robust_score `0.7499582773034212` -> paper `0.749958`

### Table 8 Combined Ranking

Zero-CASH source:

- `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_benchmark_comparison/final_corrected_zero_cash_combined_ranking.csv`

Verified rows:

- `V3_real_macro_vintage_clean_no_dxy_cap_0p70`: rank_mandate_aware `1`,
  rank_robust `3`, rank_sharpe `10`, annualized_return `0.0868668949379369`,
  annualized_volatility `0.1143426519290021`, sharpe `0.9233923384992592`,
  max_drawdown `-0.104001577768405`, mandate_aware_score
  `0.6605601145650347`, robust_score `0.7473019931544701`.
- `trend_spy_cash_12p`: rank_mandate_aware `6`, annualized_return
  `0.09786621846484`, annualized_volatility `0.1135626356435402`, sharpe
  `0.8801689872941877`, max_drawdown `-0.1782271833966934`,
  mandate_aware_score `0.4830790259640067`, robust_score
  `0.6168656756015484`.

BIL-CASH source:

- `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_benchmark_comparison/final_corrected_bil_cash_combined_ranking.csv`

Verified rows:

- `V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80`: rank_mandate_aware
  `1`, rank_robust `1`, rank_sharpe `6`, annualized_return
  `0.1064663447943072`, annualized_volatility `0.1270114499567047`, sharpe
  `1.141530430627092`, max_drawdown `-0.1030030423587755`,
  mandate_aware_score `0.6901969314744756`, robust_score
  `0.7797346249003407`.
- `trend_spy_cash_12p`: rank_mandate_aware `6`, annualized_return
  `0.1024331081806664`, annualized_volatility `0.1135428121881919`, sharpe
  `0.9169357263463784`, max_drawdown `-0.1730060265977576`,
  mandate_aware_score `0.4777841370787463`, robust_score
  `0.6041771929925795`.

Interpretation note:

- `TD3 winner` in the paper means winner by recomputed mandate-aware score in
  the combined ranking, not necessarily winner by Sharpe or by WRC search test.

### Table 9 Statistical Validation

Bootstrap sources:

- Zero-CASH: `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_statistical_validation/statistical_validation_pairwise_bootstrap.csv`
- BIL-CASH: `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_statistical_validation/statistical_validation_pairwise_bootstrap.csv`

WRC sources:

- Zero-CASH: `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_white_reality_check/white_reality_check_summary.csv`
- BIL-CASH: `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_white_reality_check/white_reality_check_summary.csv`

Verified values:

- Zero-CASH bootstrap row `V3_real_macro_vintage_clean_no_dxy_cap_0p70` vs
  `trend_spy_cash_12p`, metric `sharpe`: mean_delta `0.15586196925719`,
  lower_5pct_delta `-0.6011003083777723`, upper_95pct_delta
  `0.9767249637812803`, probability_candidate_beats `0.629`.
- Zero-CASH WRC row for benchmark `trend_spy_cash_12p`: p_value
  `0.7136431784107946`, best_candidate_by_mean_diff
  `V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80`.
- BIL-CASH bootstrap row `V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80`
  vs `trend_spy_cash_12p`, metric `sharpe`: mean_delta
  `0.11703210644801496`, lower_5pct_delta `-0.7171659783381924`,
  upper_95pct_delta `0.9963327447686309`, probability_candidate_beats
  `0.588`.
- BIL-CASH WRC row for benchmark `trend_spy_cash_12p`: p_value
  `0.6766616691654173`, best_candidate_by_mean_diff
  `V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80`.

Unresolved ambiguity:

- The paper row labels can be read as pair-specific comparisons, but WRC is a
  candidate-set test. A future paper edit should label the WRC column as
  candidate-set WRC p-value or split bootstrap and WRC into separate rows.

### Figure 2 Regime Analysis

Current paper sources:

- `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_regime_analysis/regime_winners_summary.csv`
- `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_regime_analysis/regime_winners_summary.csv`

Status:

- Final split Zero-CASH and BIL-CASH regime evidence.
- Current Figure 2 now uses these final split regime winners after label
  mapping.

Superseded previous source:

- `outputs/tables/asset_specific_cost_regime_analysis/regime_winners_summary.csv`

Interpretation note:

- The previous non-split repo-local source remains historical and should not be
  used for final cash-specific regime claims.

### Table 10 Feasibility

Sources:

- `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_constraint_pareto/constraint_pareto_summary.md`
- `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_constraint_pareto/constraint_pareto_summary.md`
- `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_mandate_profile_comparison/mandate_profile_winners.csv`
- `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_mandate_profile_comparison/mandate_profile_winners.csv`

Verified:

- Both final constraint/Pareto summaries report no feasible strategies for
  conservative, moderate, or aggressive hard filters.
- Both final mandate-profile winner files show rolling min-var as conservative
  and moderate overall winner.
- BIL-CASH mandate-profile winners show `V3_real_macro_vintage_clean_no_dxy_cap_0p70`
  as the aggressive overall winner and best TD3 candidate.

### Section 7.3 Execution-Spread Robustness

Source:

- `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_execution_spread_robustness/execution_spread_degradation_summary.csv`

Verified stress-spread rows:

- Zero-CASH `V3_real_macro_vintage_clean_no_dxy_cap_0p70`: delta_return_vs_base
  `-0.013936993311305157`, delta_sharpe_vs_base `-0.13211322869618303`.
- BIL-CASH `V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80`:
  delta_return_vs_base `-0.012811862394794815`, delta_sharpe_vs_base
  `-0.11803585477393141`.
- Zero-CASH `trend_spy_cash_12p`: delta_sharpe_vs_base
  `-0.013180736401825599`.
- BIL-CASH `trend_spy_cash_12p`: delta_sharpe_vs_base
  `-0.01318208371514562`.

### Section 7.4 Training-Budget Convergence

Sources:

- `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_training_budget_convergence/training_budget_convergence_summary.csv`
- `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_training_budget_convergence/training_budget_convergence_summary.md`

Verified:

- Summary states 4 candidates, 4 budgets, 5 seeds, 4 folds = 320 histories.
- The summary states the 5-seed check does not support rerunning the main
  protocol with longer budgets.
- Three selected candidate/cap pairs have best Sharpe at 60 episodes:
  BIL-CASH V7, BIL-CASH V8, Zero-CASH V3.
- Zero-CASH V5 has best Sharpe at 30 episodes.
- No row supports 60-episode undertraining evidence.

## Organization Recommendation

Recommended future organization, if the user explicitly approves moving files:

- Keep external final roots under a `canonical_final/` index or manifest.
- Move repo-local historical outputs into an archive namespace such as
  `outputs/archive_pre_final_corrected/`.
- Keep `outputs/tables/final_corrected_limited_td3_60ep_10seeds/` available as
  a named upstream input to Zero-CASH final reports.
- Add an `outputs/README.md` or `outputs/LINEAGE.md` that points to
  `docs/final_output_source_map.md`.
- Do not move or delete any outputs as part of paper editing.

## Unresolved Ambiguities

1. Table 9 WRC labeling should be clarified in a later paper edit because the
   p-value is candidate-set evidence, not necessarily pair-specific evidence for
   the strategy named in the comparison cell.
2. Table 5's 800-history-per-cash-assumption statement should remain tied to
   cap-sensitivity metadata. The benchmark-comparison metadata only reports 5
   selected TD3 candidates and 14 benchmarks.
3. Some repo-local folders use names like `final_*` or `final_corrected_*` but
   are not automatically canonical. Metadata lineage, not folder naming, should
   decide final status.
