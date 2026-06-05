# Contribution Statement

## One-Paragraph Contribution

This paper contributes a falsification-oriented evaluation framework for TD3-based dynamic portfolio allocation. The contribution is not the use of TD3, Bitcoin, macro variables, transaction costs, turnover, cash, constraints, or benchmarks in isolation. The contribution is the simultaneous evaluation of TD3 under a realistic cross-asset protocol with asset-specific transaction costs, explicit cash assumptions, regenerated deterministic benchmarks, statistical validation, mandate constraints, regime analysis, execution-spread robustness, and training-budget convergence. The resulting evidence is deliberately conservative: TD3 remains competitive, but it does not statistically dominate clean benchmarks.

## Explicit Contributions

1. A compact cross-asset TD3 allocation protocol with long-only portfolio weights, explicit CASH, asset-specific transaction costs, and separate Zero-CASH and BIL-CASH robustness specifications.

2. A corrected TD3 training and evaluation stack that aligns behavior actions, executed actions, replay-buffer actions, target policy smoothing, and actor-loss critic evaluation under active max-weight caps.

3. A clean macro vintage/as-of specification that excludes DXY, requires traceability metadata, removes incorrectly constructed CPI 12-week momentum from the clean final macro state, and uses correctly constructed CPI YoY before weekly alignment.

4. A layered evaluation design that separates TD3-only candidate selection, combined TD3-versus-benchmark ranking, bootstrap validation, White Reality Check, regime analysis, mandate/Pareto feasibility, execution-spread robustness, and training-budget convergence.

5. A conservative interpretation framework in which non-significance is treated as evidence about the limits of DRL portfolio claims rather than as a failed attempt to market an alpha strategy.

## What The Paper Does Not Claim

- It does not claim that TD3 beats the market.
- It does not claim statistical superiority over deterministic benchmarks.
- It does not claim a deployable trading strategy.
- It does not claim that custom robust or mandate-aware scores are sufficient academic evidence.
- It does not claim that the asset universe is a complete global portfolio.
- It does not claim exact broker execution or complete market-impact modeling.

## Why Non-Significance Is Informative

In a DRL portfolio setting, a non-significant result is not empty. It shows what remains after the model is forced through realistic frictions and validation layers. A high-ranking TD3 policy that does not survive bootstrap or White Reality Check validation is still useful evidence: it demonstrates that apparent performance can be fragile once benchmark strength, data-snooping risk, transaction costs, cash assumptions, and regime dependence are made explicit.

## Why This Is Stronger Than A Simple "TD3 Wins" Claim

A simple "TD3 wins" result would be easier to state but weaker scientifically. It would invite obvious objections about overfitting, weak benchmarks, data snooping, transaction costs, cash treatment, concentration, and regime dependence. This project instead asks whether TD3 remains credible after those objections are built into the protocol. The answer is more modest and more defensible: TD3 remains competitive, but not statistically dominant.
