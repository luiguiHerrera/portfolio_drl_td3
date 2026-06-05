# Literature Positioning

This note is deliberately sharp. The paper should not pretend that every component is novel.

## What Is Not Novel

- TD3 is not novel. The algorithm is established in continuous-control reinforcement learning through Fujimoto et al. (2018), with roots in deterministic policy gradients and DDPG.
- Portfolio choice is not novel. The paper sits downstream of Markowitz (1952), Sharpe (1966), and Merton (1969, 1971).
- DRL portfolio allocation is not novel. Prior work includes recurrent reinforcement learning, model-free portfolio management, FinRL-style trading frameworks, and TD3 portfolio-selection studies.
- Multi-asset allocation is not novel. Cross-asset portfolio construction and risk sleeves are standard in applied asset allocation [citation needed].
- Crypto/gold portfolio analysis is not novel. There is existing literature on Bitcoin, gold, safe-haven behavior, and alternative assets [citation needed].
- Transaction costs and risk-aware rewards are not novel. They appear throughout trading and DRL portfolio literature.
- Statistical data-snooping control is not novel. White (2000), Deflated Sharpe ideas, and bootstrap validation already exist.

## What May Still Be Novel

The available gap is not a new TD3 architecture. The gap is the combined evaluation stack:

- TD3 portfolio allocation under asset-specific transaction-cost-aware training;
- explicit Zero-CASH and BIL-CASH cash-assumption robustness;
- regenerated deterministic benchmarks under matching cost models;
- clean macro vintage/as-of feature discipline;
- cap-consistent action handling;
- bootstrap and White Reality Check validation;
- hard mandate filters and Pareto analysis;
- regime analysis;
- execution-spread stress;
- training-budget convergence.

The novelty is therefore methodological discipline. The paper asks whether TD3 remains credible after common backtest objections are imposed simultaneously.

## Likely Reviewer Attacks

1. "The asset universe is too small."

Response: correct, and intentional. The universe is a compact risk-sleeve laboratory, not a complete institutional portfolio.

2. "TD3 does not statistically beat benchmarks."

Response: correct. That is part of the finding, not a hidden weakness.

3. "Custom mandate-aware scores are arbitrary."

Response: they are diagnostic. The paper also reports standard metrics, hard mandate filters, Pareto dominance, bootstrap, and White Reality Check.

4. "The transaction-cost model is still approximate."

Response: correct. The corrected protocol uses asset-specific broker-style costs and an additional spread robustness layer, but it does not claim exact execution.

5. "The macro data pipeline is fragile."

Response: the final clean macro path excludes DXY, requires traceability metadata, and corrects CPI YoY before weekly alignment. Claims should stay within that audited scope.

6. "Sixty episodes may be too few."

Response: the convergence robustness check does not show material longer-budget improvement for selected candidate-cap pairs. A 10-seed extension is optional confirmation, not required for the main thesis.

## Honest Defense

The strongest defense is not that TD3 wins. The defense is that the paper refuses to hide the tests that weaken the claim. If TD3 remains competitive after realistic frictions, benchmark regeneration, statistical validation, mandate filters, execution stress, and convergence checks, that is a useful result even without statistical dominance.

## Citation Notes

Existing project references already cover Markowitz, Sharpe, Merton, deterministic policy gradients, DDPG, TD3, several portfolio DRL papers, GARCH, Deflated Sharpe, Lopez de Prado, and White Reality Check. Broader Bitcoin/gold safe-haven and cross-asset allocation references should be added only after exact sources are verified.
