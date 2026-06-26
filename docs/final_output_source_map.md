# Final Output Source Map

This document is the source-of-truth map for the current final paper evidence.
It is an audit and organization aid, not a rewrite of the paper.

## Canonical Source-of-Truth Folders

Treat these external output roots as the canonical final evidence for paper-level
claims unless a later audit explicitly supersedes them:

| Evidence layer | Canonical folder |
| --- | --- |
| Zero-CASH benchmark-matched ranking | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_benchmark_comparison/` |
| BIL-CASH benchmark-matched ranking | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_benchmark_comparison/` |
| Zero-CASH bootstrap validation | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_statistical_validation/` |
| BIL-CASH bootstrap validation | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_statistical_validation/` |
| Zero-CASH White Reality Check | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_white_reality_check/` |
| BIL-CASH White Reality Check | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_white_reality_check/` |
| Zero-CASH regime analysis | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_regime_analysis/` |
| BIL-CASH regime analysis | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_regime_analysis/` |
| Training-budget convergence | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_training_budget_convergence/` |
| Execution-spread robustness | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_execution_spread_robustness/` |
| Zero-CASH constraint/Pareto | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_constraint_pareto/` |
| BIL-CASH constraint/Pareto | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_constraint_pareto/` |
| Zero-CASH mandate profiles | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_mandate_profile_comparison/` |
| BIL-CASH mandate profiles | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_mandate_profile_comparison/` |
| BIL-CASH TD3-only cap sensitivity | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds/` |

Special case: Zero-CASH TD3-only cap sensitivity currently traces to the
repo-local folder `outputs/tables/final_corrected_limited_td3_60ep_10seeds/`.
The external Zero-CASH benchmark metadata explicitly lists this folder as its
TD3 input. Treat it as a canonical upstream component for the Zero-CASH TD3-only
selection and benchmark-comparison reports, but do not treat similarly named
repo-local folders as final without this explicit lineage.

## Paper Table, Figure, and Claim Source Map

| Paper item | Current value or summary | Exact source path | Source status | Match status |
| --- | --- | --- | --- | --- |
| Table 1, research gap | Conceptual literature-positioning table. | `paper/main.tex` and cited literature in `paper/references.bib`. | Paper-authored conceptual content. | Not an empirical output. |
| Table 2, asset universe | SPY, TLT, GLD, BTC-USD, CASH mapped to economic sleeves. | `paper/main.tex`; data inputs are `data/processed/returns_weekly_latest.csv` and BIL proxy variant where applicable. | Design specification. | Not a numeric output table. |
| Table 3, cash protocols | Zero-CASH: 0 return / 0 bps; BIL-CASH: BIL proxy / 2 bps. | `paper/main.tex`; BIL benchmark metadata in `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_benchmark_comparison/final_corrected_bil_cash_benchmark_comparison_metadata.json`. | Final design specification. | Matches metadata for BIL-CASH cost; Zero-CASH cost matches Zero-CASH metadata. |
| Cost schedule claim | SPY/TLT/GLD 2 bps, BTC-USD 10 bps, CASH 0 bps under Zero-CASH and 2 bps under BIL-CASH. | Zero-CASH and BIL-CASH benchmark metadata JSON files in the canonical benchmark folders. | Final. | Matches metadata. |
| Table 4, feature families | V2-V8 candidate descriptions. | `paper/main.tex`; historical experiment docs and configs. | Design/narrative layer. | Not a direct final output table. |
| Figure 1, protocol overview | TikZ evidence-system diagram. | `paper/main.tex`. | Paper-authored conceptual figure. | Not an empirical output. |
| Table 5, final corrected protocol | Weekly, 10 seeds, 4 folds, 60 episodes, 800 TD3 histories per cash assumption, 14 benchmarks per cash assumption. | Benchmark metadata JSON files; TD3 cap-sensitivity metadata in `outputs/tables/final_corrected_limited_td3_60ep_10seeds/cap_sensitivity_metadata.json` and `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds/cap_sensitivity_metadata.json`. | Final/mostly final. | Matches inspected benchmark metadata for 14 benchmarks and 5 selected TD3 candidates. The 800-history count is a protocol aggregate claim and should be kept tied to the cap-sensitivity metadata. |
| Table 6, reporting layers | Layer definitions and interpretation boundaries. | `docs/paper_reporting_layers.md`; `paper/main.tex`. | Final narrative guidance. | Matches existing reporting-layer docs. |
| Table 7, TD3-only selected candidates | Zero-CASH winner: `V5_no_volatility_block_cap_0p50`, ann. ret. 0.0667, vol. 0.1156, Sharpe 0.5686, max DD -0.1206, turnover 0.4195, eff. assets 2.8093, weekly cost 0.000115, mandate-aware 0.601124, robust 0.696702. BIL-CASH winner: `V8_ewma_garch_vol_current_cap_0p70`, ann. ret. 0.0731, vol. 0.1060, Sharpe 1.0253, max DD -0.1066, turnover 0.3450, eff. assets 2.0044, weekly cost 0.000103, mandate-aware 0.660435, robust 0.749958. | Zero-CASH: `outputs/tables/final_corrected_limited_td3_60ep_10seeds/cap_sensitivity_all_results.csv` and `cap_sensitivity_summary.csv`. BIL-CASH: `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds/cap_sensitivity_all_results.csv` and `cap_sensitivity_summary.csv`. | Zero-CASH: final upstream component but repo-local; BIL-CASH: final external. | Values match after rounding. |
| Table 8, combined ranking | Zero-CASH mandate-aware winner: `V3_real_macro_vintage_clean_no_dxy_cap_0p70`, ann. ret. 0.0869, vol. 0.1143, Sharpe 0.9234, max DD -0.1040, mandate-aware 0.6606, robust 0.7473. Zero-CASH benchmark comparator: `trend_spy_cash_12p`, ann. ret. 0.0979, vol. 0.1136, Sharpe 0.8802, max DD -0.1782, mandate-aware 0.4831, robust 0.6169. BIL-CASH mandate-aware/robust winner: `V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80`, ann. ret. 0.1065, vol. 0.1270, Sharpe 1.1415, max DD -0.1030, mandate-aware 0.6902, robust 0.7797. BIL-CASH benchmark comparator: `trend_spy_cash_12p`, ann. ret. 0.1024, vol. 0.1135, Sharpe 0.9169, max DD -0.1730, mandate-aware 0.4778, robust 0.6042. | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_benchmark_comparison/final_corrected_zero_cash_combined_ranking.csv`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_benchmark_comparison/final_corrected_bil_cash_combined_ranking.csv`. | Final. | Values match after rounding. Interpretation must distinguish roles: V3 is Zero-CASH mandate-aware winner, not best Sharpe or WRC best-by-mean-diff; V7 is BIL-CASH mandate-aware and robust winner; `trend_spy_cash_12p` is the best benchmark comparator by mandate-aware score in both files. |
| Table 9, bootstrap and WRC | Zero-CASH bootstrap Sharpe delta 0.1559, CI [-0.6011, 0.9767], P(beats) 0.629; WRC p-value 0.7136. BIL-CASH bootstrap Sharpe delta 0.1170, CI [-0.7172, 0.9963], P(beats) 0.588; WRC p-value 0.6767. | Bootstrap: `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_statistical_validation/statistical_validation_pairwise_bootstrap.csv`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_statistical_validation/statistical_validation_pairwise_bootstrap.csv`. WRC: `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_white_reality_check/white_reality_check_summary.csv`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_white_reality_check/white_reality_check_summary.csv`. | Final. | Numeric values match after rounding. Ambiguity: Zero-CASH bootstrap row is `V3` vs `trend_spy_cash_12p`, while the Zero-CASH WRC p-value is a candidate-set result whose `best_candidate_by_mean_diff` is `V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80`. The table is valid only if the WRC value is interpreted as search-adjusted candidate-set evidence, not as the p-value for the named V3 pair. |
| Figure 2, regime winners | Split Zero-CASH and BIL-CASH regime-winner table with mixed TD3/benchmark/momentum-style winners. | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_regime_analysis/regime_winners_summary.csv`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_regime_analysis/regime_winners_summary.csv`. | Final. | Values match the split final sources after label mapping. |
| Table 10, practical feasibility | No strategy passes all hard filters; rolling min-var dominates conservative/moderate profiles; V3 appears as best TD3/aggressive candidate in BIL-CASH; selected candidates remain Pareto-competitive. | Hard filters/Pareto: `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_constraint_pareto/constraint_pareto_summary.md`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_constraint_pareto/constraint_pareto_summary.md`. Mandate profiles: `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_mandate_profile_comparison/mandate_profile_winners.csv`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_mandate_profile_comparison/mandate_profile_winners.csv`. | Final. | Summary matches sources. |
| Section 7.3, execution-spread robustness | Stress spreads: Zero-CASH V3 ann. return delta -0.0139, Sharpe delta -0.1321; BIL-CASH V7 ann. return delta -0.0128, Sharpe delta -0.1180; Trend SPY/CASH Sharpe delta about -0.0132. | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_execution_spread_robustness/execution_spread_degradation_summary.csv`; summary in `execution_spread_summary.md`. | Final. | Values match after rounding. |
| Section 7.4, training-budget convergence | 320 histories across 4 candidates, 4 budgets, 5 seeds, 4 folds; zero rows support 60-episode undertraining; 3 selected pairs best at 60 episodes; V5 Zero-CASH peaks at 30 episodes. | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_training_budget_convergence/training_budget_convergence_summary.csv`; `training_budget_convergence_summary.md`; `training_budget_convergence_metadata.json`. | Final. | Values match summary. |
| Table 11, claim versus evidence | Cautious claim map: competitiveness yes; statistical superiority no; cash matters yes; undertraining no evidence; execution matters yes; hard mandate feasibility no; Pareto-competitive yes; custom scores insufficient; benchmarks not weak. | `docs/paper_claim_vs_evidence.md`; final sources listed above for each claim. | Final narrative synthesis. | Supported by verified final sources. |

## Superseded or Historical Output Folders

Do not delete these folders, but do not use them as final paper evidence unless a
future audit explicitly promotes a file and records why.

### Directly Superseded by Final External Corrected Reports

| Repo-local folder | Superseded by |
| --- | --- |
| `outputs/tables/asset_specific_cost_benchmark_comparison/` | Final Zero-CASH and BIL-CASH benchmark-comparison folders. |
| `outputs/tables/asset_specific_cost_statistical_validation/` | Final Zero-CASH and BIL-CASH statistical-validation folders. |
| `outputs/tables/asset_specific_cost_white_reality_check/` | Final Zero-CASH and BIL-CASH WRC folders. |
| `outputs/tables/asset_specific_cost_constraint_pareto/` | Final Zero-CASH and BIL-CASH constraint/Pareto folders. |
| `outputs/tables/asset_specific_cost_mandate_profile_comparison/` | Final Zero-CASH and BIL-CASH mandate-profile folders. |
| `outputs/tables/asset_specific_cost_regime_analysis/` | Final Zero-CASH and BIL-CASH regime-analysis folders. This repo-local non-split regime summary was previously used for Figure 2 but has been superseded in the current paper. |
| `outputs/tables/final_corrected_cash_robustness_comparison/` | Final BIL-CASH and Zero-CASH benchmark-comparison and validation folders. |

### Historical Candidate-Development or Intermediate Reports

Treat these as historical development evidence, not final paper evidence:

- `outputs/tables/asset_specific_cost_full_final_report/`
- `outputs/tables/asset_specific_cost_full_final_report_manual/`
- `outputs/tables/asset_specific_cost_limited_final_report/`
- `outputs/tables/asset_specific_cost_manual_raw_report/`
- `outputs/tables/asset_specific_cost_full_final_candidates_60ep_10seeds/`
- `outputs/tables/asset_specific_cost_v3_clean_no_dxy_60ep_10seeds/`
- `outputs/tables/asset_specific_cost_v4_garch_60ep_10seeds/`
- `outputs/tables/asset_specific_cost_v7_clean_no_dxy_garch_60ep_10seeds/`
- `outputs/tables/asset_specific_cost_v7_full_grid_60ep_10seeds/`
- `outputs/tables/asset_specific_cost_v8_full_grid_60ep_10seeds/`
- `outputs/tables/final_constrained_td3_report_60ep_10seeds/`
- `outputs/tables/final_constrained_td3_report_with_v3_clean_no_dxy_v4_v7_v8_60ep_10seeds/`
- `outputs/tables/final_constrained_td3_report_with_v3_clean_no_dxy_v7_clean_garch_v4_v7_v8_60ep_10seeds/`
- `outputs/tables/final_constrained_td3_report_with_v3_seeded_60ep_10seeds/`
- `outputs/tables/final_constrained_td3_report_with_v3_v4_60ep_10seeds/`
- `outputs/tables/final_constrained_td3_report_with_v3_v4_v7_60ep_10seeds/`
- `outputs/tables/final_constrained_td3_report_with_v3_v4_v7_v8_60ep_10seeds/`
- `outputs/tables/final_constrained_td3_report_with_v3_vintage_v4_v7_v8_60ep_10seeds/`
- `outputs/tables/mandate_profile_comparison_final/`
- `outputs/tables/statistical_validation_final_v3_clean_no_dxy/`
- `outputs/tables/statistical_validation_final_v3_clean_no_dxy_v7_clean_garch/`
- `outputs/tables/statistical_validation_final_v3_v4/`
- `outputs/tables/regime_analysis_final_v3_clean_no_dxy/`
- `outputs/tables/regime_analysis_final_v3_clean_no_dxy_v7_clean_garch/`
- `outputs/tables/regime_analysis_final_v3_v4/`
- `outputs/tables/white_reality_check_final/`
- `outputs/tables/transaction_cost_sensitivity_final/`

### Protocol, Smoke, and Exploration Runs

All `outputs/tables/*smoke*`, `outputs/tables/protocol_*`,
`outputs/tables/feature_set_comparison_*`, and
`outputs/tables/cap_sensitivity_experiment_*` folders should be treated as
development or exploration runs unless this source map names a specific file as
canonical.

### Figures

The folders under `outputs/figures/` appear historical. The current paper's
protocol figure is embedded in `paper/main.tex`, and the current regime figure
is an inline LaTeX table. Do not use old figure PNG/PDF files as final evidence
without a fresh source-map entry.

## Rules for Future Paper Edits

1. Do not edit `paper/main.tex` numeric results unless the exact source file and
   row are identified first.
2. Use the external `final_corrected_*` roots above for final paper evidence.
3. Never mix Zero-CASH and BIL-CASH outputs in the same claim unless the claim
   is explicitly a cross-cash comparison.
4. Distinguish mandate-aware ranking, robust ranking, Sharpe ranking, bootstrap
   pairwise validation, and WRC candidate-set validation. They answer different
   questions and can select different strategies.
5. Treat robust and mandate-aware scores as diagnostics, not statistical proof.
6. Treat WRC p-values as candidate-set/model-search evidence. Do not attach a
   WRC p-value to a named pair unless the WRC source row supports that exact
   interpretation.
7. Treat execution-spread and training-budget convergence as robustness layers,
   not model-selection layers.
8. Keep Figure 2 tied to the split external Zero-CASH and BIL-CASH regime
   folders for final cash-specific regime evidence.
9. Do not delete, move, or rewrite output folders during paper editing. Archive
   movement should be a separate explicit operation.
10. When a repo-local folder has a `final_*` name, require metadata lineage to a
    canonical external report before treating it as final.

## Recommended Documentation Updates

- Update `docs/paper_results_narrative.md` to name the canonical output roots
  for each layer.
- Update `docs/paper_reporting_layers.md` with a short warning that WRC is a
  candidate-set test and may not name the same strategy as the mandate-aware
  ranking winner.
- Update `docs/paper_claim_vs_evidence.md` with references to this source map
  and the final external folders.
- Add an archive index if output folders are moved later; do not move them in
  this pass.
