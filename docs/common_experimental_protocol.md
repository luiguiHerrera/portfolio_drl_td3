# Common Experimental Protocol

## 1. Purpose

This protocol defines the official comparison rules for future model and
benchmark experiments. Its purpose is to avoid comparing stale, inconsistent,
or non-investable results. Any reported comparison should make clear whether it
follows this protocol.

## 2. Asset Universe

The standard asset universe is:

- SPY
- TLT
- GLD
- BTC-USD
- CASH

CASH is synthetic zero-return unless a later experiment explicitly documents a
cash yield assumption.

## 3. Data Frequency

All standard experiments use weekly returns. All strategies in the same
comparison must use the same aligned return matrix, asset ordering, and
evaluation dates.

## 4. Train / Validation / Test Structure

The default validation design is expanding walk-forward validation. Training
windows expand through time, validation follows training, and test follows
validation.

Feature warmup may shorten early training windows. Validation and test windows
should remain aligned across candidates whenever possible so that differences
reflect strategy behavior rather than date coverage.

## 5. Timing Convention

All investable strategies must follow the same decision timing:

- information available through `t-1`
- weights selected for period `t`
- realized return at `t` applied after weights are chosen
- no same-period signal leakage

Signals that use rolling returns, volatility, covariance, drawdown, or regimes
must be lagged before they determine weights for the evaluated return period.

## 6. Initial Weights

At the first evaluated period, previous weights should default to equal weights
over the evaluated asset universe. This matches the TD3 environment reset
convention. Any different initial allocation must be explicitly documented.

## 7. Turnover Convention

Turnover is:

```text
sum(abs(w_t - w_{t-1}))
```

The same convention must be used for TD3 policies and benchmark strategies.

## 8. Transaction Cost Convention

Transaction cost is:

```text
transaction_cost = transaction_cost_rate * turnover
```

Net financial return is:

```text
financial_net_return = portfolio_return - transaction_cost
```

Final comparable rankings should use net returns whenever a strategy has
defined weights and turnover. Gross-only references may be reported, but they
must not be treated as fully cost-comparable.

## 9. Static Benchmarks

The standard static benchmarks are:

- BuyHold_SPY
- BuyHold_TLT
- BuyHold_GLD
- BuyHold_BTC-USD
- Equal_Weight
- Equal_Weight_Risky
- 60/40 SPY-TLT

Single-asset buy-and-hold benchmarks are gross references unless evaluated
through a common weight-strategy evaluator with an explicit initial allocation,
turnover, and cost convention. Equal-weight and 60/40 benchmarks are
net-cost comparable only when implemented through the common evaluator using
the protocol turnover and transaction-cost rules.

## 10. Dynamic Benchmarks

The current dynamic benchmarks are:

- `momentum_winner_12p`
- `risk_adjusted_momentum_winner_12p_12p`
- `trend_spy_cash_12p`
- `defensive_risk_off_12p`

These benchmarks must be signal-lagged. The signal at `t` must be computed
from information available through `t-1`, then the selected weight is applied
to realized return `t`.

## 11. Rolling Benchmarks

The first rolling risk parity benchmark is `rolling_risk_parity_inverse_vol`.
It is rolling inverse-volatility risk parity, not full equal-risk-contribution
optimization. Weights are proportional to inverse realized volatility estimated
from past returns only. CASH is excluded by default. If CASH is included in a
special run, its zero or near-zero volatility must be handled with an explicit
volatility floor and reported clearly.

The first rolling Markowitz benchmark is `rolling_markowitz_long_only`. It is a
long-only constrained rolling mean-variance benchmark using rolling historical
mean and covariance estimates. It is not a full-sample Markowitz portfolio and
not an oracle benchmark. The default version excludes CASH, applies a maximum
weight constraint, adds covariance ridge regularization, and falls back to
inverse-volatility weights if the optimizer fails for a rebalance date.

Future benchmark families are reserved for:

- constrained rolling Markowitz

Minimum requirements:

- rolling windows use only past data
- no full-sample covariance
- no future returns
- same timing convention as TD3
- same turnover and transaction-cost convention
- constraints explicitly documented

Constraints may include long-only weights, max weight, asset eligibility,
volatility targeting, cash bounds, or turnover controls. Any constraint must be
reported with the benchmark result.

## 12. TD3 Candidate Models

Current TD3 candidates for the next protocol run are:

- `V2_reference_full`
- `V5_no_volatility_block`
- `V6_financial_state`

V6 is an experimental candidate, not a proven winner. It should be compared
against V2, V5_no_volatility, and transparent benchmarks before any stronger
claim is made.

## 13. Evaluation Metrics

Primary financial metrics:

- annualized return
- annualized volatility
- Sharpe
- Sortino
- Calmar
- maximum drawdown
- cumulative return

Robustness metrics:

- robust Sharpe
- Deflated Sharpe Ratio
- robust_score

Behavioral diagnostics:

- turnover
- average max weight
- effective number of assets
- cash above 10%
- unjustified cash
- dominant-asset regret
- hit rate
- rule comparison

Model quality should be judged by financial behavior and robustness, not only
by final equity value or Sharpe.

## 14. DSR Aggregation Policy

Pooled DSR may be reported for transparency, but it should not drive final
selection when folds or seeds contain overlapping dates.

Composite `robust_score` should use the following DSR method order:

1. `median_run`
2. `date_averaged`
3. `pooled`
4. `fallback_from_sharpe`

Warning: pooled DSR can overstate evidence when folds or seeds contain
overlapping dates. Median run-level DSR is the preferred input when available.

## 15. Model Selection Rule

A TD3 candidate cannot be called superior unless it:

- beats simple benchmarks on net-return metrics
- is not dominated on drawdown
- has stable seed and fold behavior
- has acceptable turnover and transaction costs
- improves behavioral diagnostics without relying on one lucky fold
- survives conservative DSR and robust_score aggregation

If a TD3 policy improves timing diagnostics but not aggregate risk-adjusted
performance, the conclusion should separate those findings.

## 16. Staleness Policy

The following outputs should be treated as stale for final comparison:

- dynamic benchmark output produced before the benchmark first-period turnover
  convention was aligned with TD3
- robust_score output produced before conservative DSR aggregation
- any experiment using `lambda_sharpe` as if it were an active reward parameter

Stale outputs may be used as research history, but not as final model-selection
evidence.

## 17. Reproducibility Checklist

Each future experiment report must include:

- config path
- data path
- feature version
- seeds
- folds
- transaction cost rate
- turnover convention
- DSR method
- git commit hash if available
- tests status

Reports should also identify whether benchmark returns are gross references or
net-cost comparable results.

The official benchmark-only comparison runner is
`src/experiments/run_protocol_benchmark_comparison.py`. Static benchmarks in
that runner are evaluated as explicit weight strategies so they are net-cost
comparable under the protocol.

The official combined reporting entry point for TD3 candidates and benchmarks
is `src/experiments/run_protocol_td3_comparison.py`. It is a comparison and
ingestion layer, not a long-training orchestrator. TD3 candidate results should
be generated under the protocol and then combined with the benchmark suite for
common reporting.

## 18. Next Implementation Steps

The next implementation steps are:

- revalidate V2, V5_no_volatility, and V6 under this protocol
- update README only after the protocol has been used
