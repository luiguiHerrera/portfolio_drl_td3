# Final Paper Full Audit

## Executive verdict

**Ready after minor wording cleanup.** I found no blocking numeric mismatch between `paper/main.tex` and the final source map. The latest Figure 2 correction is reflected in `docs/final_output_source_map.md`, and the current paper uses the final split Zero-CASH and BIL-CASH regime outputs rather than the historical non-split repo-local regime table.

**Blocking issues:** none.

**Minor issue before final submission:** Table 9's numeric values match the final bootstrap and White Reality Check outputs, but the paper should make explicit that the WRC p-values are candidate-set/search-adjusted evidence against the benchmark, not necessarily pair-specific p-values for the displayed TD3 row. This matters most for Zero-CASH because the WRC file's best candidate by mean differential is V7, while the displayed bootstrap pair is V3 versus Trend SPY/CASH.

**PDF check limitation:** `paper/main.pdf` is newer than `paper/main.tex` in the local filesystem, but text/layout extraction could not be performed in this environment because `pdfinfo`, `pdftotext`, `pypdf`, `qpdf`, `mutool`, and Ghostscript were unavailable. This audit therefore verifies `paper/main.tex` and source files directly, plus PDF artifact freshness, not rendered PDF text.

## Table and figure audit matrix

| Paper item | Current claim/value in paper | Exact source file path | Source status | Value matches source? | Interpretation accurate? | Action required |
|---|---|---|---|---|---|---|
| Abstract and framing claims | TD3 candidates can be competitive, but statistically conservative tests do not establish reliable superiority over the best benchmark comparator; cash treatment and benchmark choice matter. | `paper/main.tex`; final evidence paths listed below. | Conceptual synthesis over final sources | Yes | Yes; claims are cautious and do not overstate significance. | None |
| Table 1: research gap and literature positioning | Positions the paper against prior RL portfolio work, transaction-cost studies, cash proxies, and statistical validation gaps. | `paper/main.tex`; literature references in bibliography. | Conceptual | Not numeric | Yes; not an empirical output table. | None |
| Table 2: asset universe and economic sleeves | SPY, TLT, GLD, BTC-USD, CASH/BIL proxy mapped to economic sleeves and cost assumptions. | `data/processed/returns_weekly_latest.csv`; `data/processed/returns_weekly_latest_cash_bil_proxy.csv`; paper protocol text. | Final upstream dataset/protocol | Yes, at protocol level | Yes; this is a design/protocol table, not a historical experiment output. | None |
| Table 3: cash protocols | Zero-CASH uses synthetic zero-return cash with zero trading cost; BIL-CASH uses BIL proxy with trading cost. | `data/processed/returns_weekly_latest.csv`; `data/processed/returns_weekly_latest_cash_bil_proxy.csv`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_benchmark_comparison/final_corrected_zero_cash_benchmark_comparison_metadata.json`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_benchmark_comparison/final_corrected_bil_cash_benchmark_comparison_metadata.json` | Final upstream/protocol | Yes | Yes | None |
| Table 4: TD3 feature families | V1-V8 feature-family definitions, including clean macro, GARCH, and macro/GARCH variants. | `paper/main.tex`; TD3 experiment naming in final selected outputs. | Protocol/conceptual | Not a numeric result table | Yes | None |
| Figure 1: protocol overview | Experimental protocol flow from data, candidate generation, benchmark comparison, validation, and interpretation. | `paper/main.tex`; `docs/paper_reporting_layers.md` | Conceptual | Not numeric | Yes | None |
| Table 5: final corrected experimental protocol | 5 feature candidates x 4 caps x 10 seeds x 4 folds = 800 histories per cash assumption; benchmark comparison uses 5 selected TD3 candidates plus 14 benchmarks; validation layers use bootstrap/WRC, regime, spread, convergence, and feasibility analyses. | `outputs/tables/final_corrected_limited_td3_60ep_10seeds/cap_sensitivity_metadata.json`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds/cap_sensitivity_metadata.json`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_benchmark_comparison/final_corrected_zero_cash_benchmark_comparison_metadata.json`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_benchmark_comparison/final_corrected_bil_cash_benchmark_comparison_metadata.json` | Final and final upstream | Yes. Both cap-sensitivity metadata files report `expected_histories=800` and `found_histories=800`; benchmark metadata reports 5 selected TD3 candidates and 14 benchmarks. | Yes; "800 histories per cash assumption" is verified as the cap-sensitivity protocol aggregate. | None |
| Table 6: reporting layers | Separates within-TD3 screening, benchmark comparison, statistical validation, regime, feasibility, spread, convergence, and claim-versus-evidence layers. | `docs/paper_reporting_layers.md`; `paper/main.tex` | Conceptual/reporting protocol | Not numeric | Mostly yes. The paper table is accurate. Supporting doc has minor stale phrasing around "best benchmark." | Non-blocking doc cleanup |
| Table 7: TD3-only selected candidates | Zero-CASH V5: return 0.0667, volatility 0.1156, Sharpe 0.5686, max drawdown -0.1206, turnover 0.4195, effective assets 2.8093, cost 0.000115, mandate-aware 0.6011, robust 0.6967. BIL-CASH V8: return 0.0731, volatility 0.1060, Sharpe 1.0253, max drawdown -0.1066, turnover 0.3450, effective assets 2.0044, cost 0.000103, mandate-aware 0.6604, robust 0.7500. | Zero-CASH: `outputs/tables/final_corrected_limited_td3_60ep_10seeds/cap_sensitivity_all_results.csv`; BIL-CASH: `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds/cap_sensitivity_all_results.csv`; Zero-CASH canonical link verified by `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_benchmark_comparison/final_corrected_zero_cash_benchmark_comparison_metadata.json` naming `td3_dir=outputs/tables/final_corrected_limited_td3_60ep_10seeds`. | Final upstream; Zero-CASH is repo-local but canonical upstream for final corrected benchmark comparison | Yes | Yes; this is within-TD3 screening, not a benchmark superiority claim. | None |
| Table 8: selected learning candidates and clean benchmark comparator | Zero-CASH selected TD3 V3: return 0.0869, volatility 0.1143, Sharpe 0.9234, max drawdown -0.1040, mandate-aware 0.6606, robust 0.7473; Trend SPY/CASH comparator: return 0.0979, volatility 0.1136, Sharpe 0.8802, max drawdown -0.1782, mandate-aware 0.4831, robust 0.6169. BIL-CASH selected TD3 V7: return 0.1065, volatility 0.1270, Sharpe 1.1415, max drawdown -0.1030, mandate-aware 0.6902, robust 0.7797; Trend SPY/CASH comparator: return 0.1024, volatility 0.1135, Sharpe 0.9169, max drawdown -0.1730, mandate-aware 0.4778, robust 0.6042. | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_benchmark_comparison/final_corrected_zero_cash_combined_ranking.csv`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_benchmark_comparison/final_corrected_bil_cash_combined_ranking.csv` | Final | Yes | Yes. Current wording distinguishes mandate-aware, robust, Sharpe, and benchmark-comparator roles. It does not call Trend SPY/CASH the unconditional "best benchmark" and does not call the selected TD3 row an unconditional "TD3 winner." | None |
| Table 9: bootstrap Sharpe differences and White Reality Check | Zero-CASH V3 vs Trend SPY/CASH: mean Sharpe difference 0.1559, 95% CI [-0.6011, 0.9767], probability 0.629, WRC p-value 0.7136. BIL-CASH V7 vs Trend SPY/CASH: mean Sharpe difference 0.1170, 95% CI [-0.7172, 0.9963], probability 0.588, WRC p-value 0.6767. | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_statistical_validation/statistical_validation_pairwise_bootstrap.csv`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_statistical_validation/statistical_validation_pairwise_bootstrap.csv`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_white_reality_check/white_reality_check_summary.csv`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_white_reality_check/white_reality_check_summary.csv` | Final | Yes | Mostly. The bootstrap rows are pair-specific. The WRC p-values are search-adjusted/candidate-set evidence against Trend SPY/CASH. Zero-CASH WRC's best candidate by mean differential is V7, not the displayed V3 row. | Wording tweak |
| Figure 2: regime winners | Split Zero-CASH and BIL-CASH regime-winner panels by metric and regime. | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_regime_analysis/regime_winners_summary.csv`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_regime_analysis/regime_winners_summary.csv` | Final | Yes, after label mapping from source strategy IDs to display labels. | Yes. Current paper no longer relies on the historical non-split `outputs/tables/asset_specific_cost_regime_analysis/` source. | None |
| Table 10: practical feasibility interpretation | No strategy satisfies all conservative/moderate/aggressive feasibility filters in either cash setting. Mandate-profile winners are rolling min-variance for conservative/moderate Zero-CASH and BIL-CASH; rolling min-variance for aggressive Zero-CASH; TD3 V3 for aggressive BIL-CASH. | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_constraint_pareto/constraint_feasibility_summary.csv`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_constraint_pareto/constraint_feasibility_summary.csv`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_mandate_profile_comparison/mandate_profile_winners.csv`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_mandate_profile_comparison/mandate_profile_winners.csv` | Final | Yes | Yes. The table accurately distinguishes infeasibility under strict filters from profile-based ranking. | None |
| Table 11: claim versus evidence | States that TD3 is competitive but not statistically proven superior; cash assumption changes interpretation; benchmark comparator remains strong; deployment feasibility is limited under strict filters; additional training-budget checks do not overturn conclusions. | `docs/paper_claim_vs_evidence.md`; final source files for Tables 7-10 and Sections 7.3-7.4 | Final synthesis/conceptual | Yes | Yes. Current table reflects cautious corrected interpretation. Supporting doc has minor stale phrasing that can be cleaned later. | None for paper; non-blocking doc cleanup |
| Section 7.3: execution-spread robustness | Stress spread reduces Sharpe by -0.1321 for Zero-CASH selected TD3 V3 and -0.1180 for BIL-CASH selected TD3 V7; Trend SPY/CASH Sharpe degradation is about -0.0132 under both cash assumptions. | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_execution_spread_robustness/execution_spread_degradation_summary.csv` | Final | Yes. Source deltas: Zero TD3 stress return -0.013937 and Sharpe -0.132113; BIL TD3 stress return -0.012812 and Sharpe -0.118036; Trend comparator Sharpe deltas about -0.01318. | Yes | None |
| Section 7.4: training-budget convergence | 320 additional histories: 4 episode budgets x 4 selected candidate/cash cases x 5 seeds x 4 folds. No case has a 60-episode Sharpe above every larger budget. Best Sharpe budget is 60 for Zero-CASH V3, BIL-CASH V7, and BIL-CASH V8; Zero-CASH V5 peaks at 30 episodes. | `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_training_budget_convergence/training_budget_convergence_metadata.json`; `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_training_budget_convergence/training_budget_convergence_summary.md` | Final | Yes | Yes; the conclusion that extra training does not reverse the economic interpretation is supported. | None |
| Introduction, discussion, and conclusion | Claims emphasize conditional competitiveness, lack of statistically decisive superiority, cash-proxy sensitivity, and feasibility constraints. | `paper/main.tex`; final evidence paths above | Final synthesis | Yes | Yes; no claim appears to exceed the corrected evidence. | None |

## Detailed notes

### Figure 2 lineage

The current paper and source map now use the final split regime files:

- `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_zero_cash_regime_analysis/regime_winners_summary.csv`
- `/Users/thiagoherrera/Projects/portfolio_drl_outputs/final_corrected_bil_cash_regime_analysis/regime_winners_summary.csv`

The previous historical non-split source, `outputs/tables/asset_specific_cost_regime_analysis/`, is no longer a paper source and should remain documented only as superseded historical output.

### Table 8 interpretation

Table 8 now uses the corrected final combined-ranking sources and avoids the earlier interpretation drift. The Zero-CASH selected TD3 row is a top mandate-aware candidate and robust/highly ranked candidate, not an unconditional winner across all diagnostics. BIL-CASH V7 is both mandate-aware and robust rank 1 in the final combined ranking. Trend SPY/CASH is treated as the clean benchmark comparator used for statistical validation, not as the best benchmark on every metric.

### Table 9 WRC wording risk

The numbers in Table 9 match the final sources. The remaining issue is interpretive labeling. The pairwise bootstrap values are for the displayed TD3-versus-Trend pairs. The WRC p-values come from candidate-set/search-adjusted tests against `trend_spy_cash_12p`. In the Zero-CASH WRC file, the best candidate by mean differential is `V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80`, while the displayed bootstrap row is `V3_real_macro_vintage_clean_no_dxy_cap_0p70` versus Trend SPY/CASH.

This is not a numeric mismatch, but the paper should add a short note so readers do not interpret the WRC p-value as a strictly pair-specific p-value for the displayed Zero-CASH V3 bootstrap row.

### Table 5 protocol aggregate

The "800 histories per cash assumption" statement is verified against cap-sensitivity metadata:

- Zero-CASH metadata reports `expected_histories=800` and `found_histories=800`.
- BIL-CASH metadata reports `expected_histories=800` and `found_histories=800`.

The final benchmark metadata also confirms 5 selected TD3 candidates and 14 benchmark comparators for each cash assumption.

### Table 7 canonical upstream status

The Zero-CASH Table 7 source is repo-local, but it is not a stale historical table. The final corrected Zero-CASH benchmark-comparison metadata names `outputs/tables/final_corrected_limited_td3_60ep_10seeds` as its selected TD3 source directory, so the repo-local Zero-CASH Table 7 values are canonical upstream for the final corrected comparison.

### Supporting documentation freshness

`docs/final_output_source_map.md` and `docs/output_lineage_audit.md` are up to date after the Figure 2 correction. Two supporting narrative docs contain minor phrasing that could confuse future edits:

- `docs/paper_reporting_layers.md` still uses language that can sound like Trend SPY/CASH is the "best benchmark" rather than the clean benchmark comparator.
- `docs/paper_claim_vs_evidence.md` uses broad phrasing such as TD3 candidates "rank first" in corrected combined rankings; this should be aligned with the paper's more careful mandate-aware/robust language.

These are non-blocking for the paper if `docs/final_output_source_map.md` remains the source of truth.

## Required edits

No blocking numeric correction is required.

Recommended small paper wording edit before final submission:

1. Add a note near Table 9 clarifying the WRC evidence:

   ```tex
   The White Reality Check p-values are candidate-set/search-adjusted tests against the Trend SPY/CASH benchmark comparator; the pairwise bootstrap rows report the displayed TD3-versus-benchmark pairs.
   ```

   If space allows, the Zero-CASH-specific nuance can be made explicit:

   ```tex
   In Zero-CASH, the WRC best candidate by mean differential is the V7 macro+GARCH capped variant, while the displayed bootstrap row reports the selected V3 clean-macro candidate against Trend SPY/CASH.
   ```

## Non-blocking cleanup

- Update `docs/paper_reporting_layers.md` so the combined/validation layer calls Trend SPY/CASH the clean benchmark comparator, not the "best benchmark."
- Update `docs/paper_claim_vs_evidence.md` so any "rank first" language is scoped to mandate-aware or robust diagnostic rankings rather than implying unconditional dominance.
- If rendered PDF layout confidence is needed, run a PDF text/layout check in an environment with Poppler or `pypdf` installed. The local `paper/main.pdf` artifact appears fresh relative to `paper/main.tex`, but this environment could not extract rendered text.
