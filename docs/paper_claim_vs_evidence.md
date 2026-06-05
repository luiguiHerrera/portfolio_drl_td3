# Claim Versus Evidence

| Claim | Evidence | Supported? | Interpretation for the paper |
| --- | --- | --- | --- |
| TD3 is competitive. | TD3 candidates rank first in the corrected combined Zero-CASH and BIL-CASH rankings, and several TD3 variants remain Pareto-competitive. | Yes. | TD3 can be a serious research candidate under realistic frictions. |
| TD3 statistically dominates benchmarks. | Bootstrap confidence intervals include zero and White Reality Check p-values do not support superiority. | No. | Do not claim statistical dominance. |
| Cash assumption matters. | Zero-CASH and BIL-CASH select different TD3-only winners and different combined winners. | Yes. | Cash modeling is not a detail; it changes model selection. |
| 60 episodes is undertrained. | Training-budget convergence shows no material Sharpe improvement at 100 or 150 episodes for selected candidates; longer budgets often degrade Sharpe or increase turnover. | No evidence. | The 60-episode protocol can remain the main corrected protocol; 10-seed convergence is optional confirmation. |
| Execution assumptions matter. | Stress spread robustness degrades selected TD3 strategies more than `trend_spy_cash_12p`. | Yes. | Execution realism should be treated as a robustness layer, not ignored. |
| Hard mandate feasibility is achieved. | Constraint-first analysis shows no strategy passes all hard canonical mandate filters. | No. | The paper should distinguish competitiveness from strict mandate feasibility. |
| TD3 remains Pareto-competitive. | Some TD3 strategies remain on or near the Pareto frontier under standard metrics and practical diagnostics. | Yes. | TD3 is not dismissed; it remains plausible but constrained. |
| Custom scores are sufficient academic evidence. | Robust and mandate-aware scores are useful diagnostics, but statistical validation and standard metrics remain necessary. | No. | Custom scores should be secondary, not the main proof. |
| Benchmarks are weak strawmen. | `trend_spy_cash_12p`, Equal Weight, BuyHold_GLD, and other deterministic rules remain strong in several layers. | No. | Benchmarks are meaningful comparators and make the evaluation harder. |

## Paper Interpretation

The evidence supports a cautious claim: TD3 can remain competitive under a corrected, friction-aware allocation protocol, but it does not statistically dominate clean benchmarks and does not satisfy every hard mandate constraint. That is not a failed result. It is the central empirical finding.
