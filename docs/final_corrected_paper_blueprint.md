# Final Corrected Paper Blueprint

## Title Options

1. Robust TD3 Portfolio Allocation under Realistic Trading Frictions
2. A Falsification-Oriented Evaluation of TD3 for Cross-Asset Portfolio Allocation
3. TD3 Portfolio Allocation under Costs, Cash Assumptions, Benchmarks, and Mandate Constraints
4. When Does TD3 Remain Credible? A Corrected Cross-Asset Portfolio Evaluation

## Paper Thesis In One Paragraph

This paper does not show that TD3 statistically dominates portfolio benchmarks. It shows that when TD3 is evaluated under a realistic cross-asset protocol with asset-specific costs, explicit cash assumptions, regenerated benchmarks, statistical validation, mandate constraints, regime analysis, execution-spread stress, and training-budget convergence, TD3 remains competitive but not statistically superior. The contribution is therefore a robust evaluation framework and a cautionary interpretation of DRL portfolio claims, not a deployable alpha claim.

## Abstract Skeleton

This paper evaluates whether TD3-based dynamic portfolio allocation remains credible under realistic financial evaluation conditions. The protocol uses a compact cross-asset universe of SPY, TLT, GLD, BTC-USD, and CASH, with long-only weekly allocation, asset-specific transaction costs, explicit Zero-CASH and BIL-CASH assumptions, regenerated deterministic benchmarks, walk-forward evaluation, multiple seeds, bootstrap validation, White Reality Check, hard mandate filters, Pareto analysis, regime analysis, execution-spread robustness, and training-budget convergence. TD3 remains competitive and sometimes ranks first, but statistical validation does not establish superiority over clean benchmarks. The results show that cash assumptions, execution costs, and practical mandate constraints materially affect model selection. The paper contributes a falsification-oriented evaluation framework rather than a deployable alpha claim.

## Section Structure

### 1. Introduction

- Research question: can TD3 allocation remain credible under realistic portfolio conditions?
- State the negative guardrail: the paper does not claim market-beating alpha.
- Preview the corrected conclusion: competitive, not statistically dominant.

### 2. Related Literature

- Portfolio theory: Markowitz, Sharpe, Merton.
- Continuous-control RL: deterministic policy gradient, DDPG, TD3.
- DRL portfolio allocation and TD3 portfolio-selection work.
- Transaction costs, risk-aware rewards, statistical validation, data-snooping control.
- Position the gap as combined evaluation discipline.

### 3. Protocol

- Asset universe and risk sleeves.
- Timing convention.
- TD3 candidates and feature families.
- Clean macro vintage/as-of construction.
- Asset-specific transaction costs.
- Zero-CASH and BIL-CASH robustness paths.
- Action consistency and net-return-first reward.
- Deterministic benchmarks.

### 4. Evaluation Layers

- Standard metrics.
- TD3-only cap sensitivity.
- Combined TD3 + benchmark ranking.
- Bootstrap and White Reality Check.
- Regime analysis.
- Mandate/Pareto analysis.
- Execution-spread robustness.
- Training-budget convergence.

### 5. Results

- Present results in the order above.
- Standard metrics first, diagnostic scores second.
- Keep statistical validation visible.
- Explain multiple winners by layer.

### 6. Discussion

- TD3 competitiveness.
- No statistical dominance.
- Cash assumption sensitivity.
- Execution sensitivity.
- Mandate feasibility limits.
- What the result says about DRL portfolio claims.

### 7. Limitations and Future Work

- Compact asset universe.
- Approximate execution model.
- No live trading.
- No calibrated probability features yet.
- Possible 10-seed convergence extension.
- More assets and richer slippage/market-impact modeling.

### 8. Conclusion

- Restate the falsification-oriented contribution.
- TD3 remains competitive, but the corrected evidence does not support superiority claims.

## Main Table Plan

Maximum six main tables:

1. Protocol summary.
2. Asset universe / risk sleeves.
3. Feature families.
4. TD3-only selected models with standard metrics first.
5. Combined TD3 + benchmark + statistical validation.
6. Practical robustness summary: mandate/Pareto, spread, convergence.

Avoid a main table filled mostly with mandate-feasibility zeros. Put detailed pass/fail matrices in the appendix.

## Main Figure Plan

Maximum six main figures:

1. Experimental pipeline.
2. TD3 selected models: Sharpe vs max drawdown or standard metric view.
3. Combined ranking TD3 vs benchmark.
4. Bootstrap Sharpe delta confidence intervals with zero line.
5. Regime winner heatmap.
6. Execution-spread degradation.

## Appendix Plan

- Candidate registry and feature definitions.
- Macro vintage/as-of traceability and CPI YoY correction.
- Asset-specific transaction cost details.
- Action consistency and reward correction details.
- Full cap-sensitivity tables.
- Full benchmark list.
- Bootstrap and White Reality Check implementation details.
- Regime definitions.
- Mandate/Pareto pass/fail matrices.
- Execution-spread scenarios.
- Training-budget convergence details.
- Audit-pack checklist.

## Narrative Guardrails

- Do not frame the paper as "TD3 beats the market."
- Do not frame the paper as "I tried many things and fixed many issues."
- Do not make action consistency a standalone contribution.
- Do not make custom scores the primary evidence.
- Do not mix Zero-CASH and BIL-CASH results as if they are the same protocol.
- Do not treat spread robustness or convergence robustness as new model-selection layers.
- Do not claim statistical significance where WRC/bootstrap do not support it.

## Preferred Narrative

The paper should read as a falsification sequence. TD3 first has to survive realistic costs and corrected action handling. It then has to compete with benchmarks under matching cost assumptions. It then has to pass statistical validation, regime analysis, mandate filters, execution-spread stress, and convergence checks. The final answer is deliberately cautious: TD3 remains competitive, but not statistically dominant.
