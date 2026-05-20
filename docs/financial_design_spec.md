# Financial Design Specification

## 1. Financial Problem

This project studies dynamic multi-asset allocation across:

- SPY
- TLT
- GLD
- BTC-USD
- CASH

The research objective is to evaluate whether a TD3-based deep reinforcement
learning agent can learn an economically meaningful dynamic allocation policy
across equity, bonds, gold, bitcoin, and cash under realistic mandate
constraints.

The model is not evaluated only by raw performance or Sharpe. It is evaluated
by its ability to manage:

- risk regimes
- defensive allocations
- turnover
- cash exposure
- concentration
- drawdown
- robustness versus transparent benchmark strategies

The project does not assume TD3 should beat benchmarks. Benchmark comparison
is used as a falsification test, not as a marketing target.

## 2. Asset Roles

SPY is the equity risk and growth engine.

TLT is duration exposure and a potential defensive bond allocation.

GLD is a real asset, crisis hedge, and alternative defensive asset.

BTC-USD is a high-convexity risk asset.

CASH is a defensive allocation and waiting asset. It is not an alpha asset.

## 3. Decision Hierarchy

The allocation problem should be interpreted as a sequence:

1. Decide risk-on versus risk-off.
2. Select the dominant return engine.
3. Select the defensive asset when needed.
4. Control implementation: turnover, drawdown, cash, and concentration.

The model should be evaluated as a sequential allocator, not as a black-box
return maximizer. A good policy should make financially understandable
allocation decisions before it produces a good summary statistic.

## 4. Mandate Profiles

Mandate profiles already exist in the codebase and should govern reward and
constraint choices. Exact values should be read from the mandate profile
definitions.

Conceptually, the profiles are:

- aggressive
- balanced/moderate
- defensive

Different mandate profiles imply different acceptable levels of:

- volatility
- drawdown
- cash exposure
- turnover
- concentration
- risk-asset exposure

A policy that is acceptable for an aggressive mandate may be unacceptable for a
moderate or defensive mandate. The mandate should be defined before tuning
reward parameters.

## 5. Parameter Philosophy

Lambdas should not be optimized just to win a backtest. They should translate
financial preferences into the reward.

`normal_cash_max` should reflect mandate cash tolerance.

`turnover_free_band` should reflect acceptable weekly reallocation.

`cash_penalty_weight` should prevent cash traps, not eliminate valid defensive
cash.

Concentration penalties should not punish justified dominant exposure.

Drawdown penalties should protect mandate consistency, not force the model
into permanent defensiveness.

Parameters should be financially justified before being empirically tuned. If
a parameter cannot be explained in financial terms, it should not become part
of the main experiment.

## 6. Evaluation Criteria

Model quality cannot be judged only by Sharpe.

Evaluation must include:

- Deflated Sharpe / robust_score
- Sortino
- Calmar
- maximum drawdown
- turnover
- cash discipline
- concentration quality
- dominant-asset regret
- benchmark comparison
- walk-forward consistency

A useful TD3 policy should be judged by financial behavior, not only by a
higher final equity curve.

## 7. Current Research Position

Current working evidence says:

- V2 remains the clean reference.
- Full V5 appears overloaded.
- V5 dynamic CASH improves cash discipline but does not beat simple benchmarks.
- Feature-block ablation suggests momentum/trend signals are most useful for
  dominant-asset timing.
- The next modeling work should focus on parsimonious financial state design,
  not more penalty tuning.

These are working conclusions, not final proof. They describe the current
evidence and should remain open to stronger validation.

## 8. Next Modeling Direction

The next stage should build a V6 financial state.

Momentum/trend should be central.

Regime probabilities can be added, but only if they are interpretable.

Forward-looking volatility can be used if it supports risk adjustment or
regime scoring.

Cointegration and LSTM are not current priorities for this asset universe.

The next stage should test whether a simpler, financially structured state
improves decision quality before adding more model complexity.
