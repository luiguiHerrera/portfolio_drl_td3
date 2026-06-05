# Reporting Layers

The final corrected project contains several reporting layers because each layer answers a different question. A different winner in a different layer is not automatically a contradiction.

| Layer | Question answered | Main result | What it does NOT prove |
| --- | --- | --- | --- |
| TD3-only cap sensitivity | Which TD3 specification and cap performs best inside the TD3 candidate universe? | Zero-CASH: `V5_no_volatility_block_cap_0p50`. BIL-CASH: `V8_ewma_garch_vol_current_cap_0p70`. | It does not compare TD3 against deterministic benchmarks. |
| Combined TD3 + benchmark ranking | How do selected TD3 candidates rank against regenerated deterministic benchmarks under the same cost model? | Zero-CASH best overall: `V3_real_macro_vintage_clean_no_dxy_cap_0p70`. BIL-CASH best overall: `V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80`. Best benchmark in both settings: `trend_spy_cash_12p`. | It does not establish statistical superiority. |
| Bootstrap / White Reality Check | Does the top searched TD3 candidate statistically outperform clean benchmarks after sampling uncertainty and model search? | No statistical superiority is supported. | It does not invalidate TD3 competitiveness or stabilization. It only rejects overclaiming. |
| Regime analysis | Is performance broad-based or concentrated in particular market regimes? | TD3 can lead in some slices, but benchmark strength remains regime-specific and important. | It does not prove all-weather dominance. |
| Mandate / Pareto analysis | Which strategies remain credible under hard mandate filters and standard multi-metric tradeoffs? | No strategy passes all hard canonical mandate filters. Some TD3 strategies remain Pareto-competitive. | It does not make custom mandate-aware scores primary evidence. |
| Execution-spread robustness | How sensitive are selected histories to additional bid-ask spread assumptions? | Selected TD3 strategies degrade more than `trend_spy_cash_12p` under stress spread assumptions. | It is not a new model-selection layer and does not create new final winners. |
| Training-budget convergence | Does the 60-episode training budget look obviously undertrained? | The 5-seed check does not show that 60 episodes is undertrained. Longer budgets often reduce Sharpe or increase turnover. | It does not prove optimal training length; a 10-seed extension is optional confirmation. |

## Why Multiple Winners Are Not Contradictions

The project has multiple winners because the word "best" changes meaning across layers. TD3-only cap sensitivity asks which TD3 variant is strongest inside the TD3 universe. Combined ranking asks whether that TD3 variant still ranks well against deterministic benchmarks. Bootstrap and White Reality Check ask whether the apparent edge survives statistical uncertainty and model search. Mandate and Pareto analysis ask whether the strategy is practically feasible under hard constraints. These are complementary filters, not competing stories.
