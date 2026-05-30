# Asset-Specific Transaction Cost Design

This document specifies a future design for implementing asset-specific transaction costs inside the portfolio environment and TD3 training protocol.

It is a design note only. It does not change the current experiments, final rankings, or paper claims.

## 1. Current Cost Model

The current environment uses a scalar proportional turnover cost.

At each step, the environment computes total turnover as:

```text
turnover_t = sum_i abs(w_t,i - w_{t-1,i})
```

The realized transaction cost is then:

```text
transaction_cost_t = transaction_cost_rate * turnover_t
```

The environment subtracts this cost from the gross portfolio return to obtain the financial net return used in evaluation:

```text
financial_net_return_t = gross_return_t - transaction_cost_t
```

The same turnover and cost diagnostics are also made available to the reward calculation and reporting layers. The reward can include transaction-cost and turnover-related penalties depending on the active reward configuration.

Current limitations:

- all assets share the same per-turnover cost;
- ETF, crypto, and CASH trades are treated too similarly;
- BTC trading is not separately penalized despite higher explicit trading costs;
- CASH changes are charged like risky asset changes unless handled indirectly;
- broker minimum commissions, spreads, market impact, taxes, and order-routing details are not modeled;
- the model is suitable as a first-order proportional cost, but not as a realistic asset-specific execution approximation.

## 2. Proposed Asset-Specific Cost Model

Use an asset-level cost vector.

Asset classes:

- `SPY`, `TLT`, `GLD`: ETF assets
- `BTC-USD`: crypto asset
- `CASH`: zero-cost synthetic cash

Costs are applied to the one-way traded notional implied by changes in portfolio weights.

Let:

- `w_t,i` be the target weight for asset `i` at time `t`;
- `w_{t-1,i}` be the previous portfolio weight for asset `i`;
- `cost_bps_i` be the cost for asset `i`, converted from basis points to decimal.

Then:

```text
transaction_cost_t = sum_i cost_bps_i * abs(w_t,i - w_{t-1,i})
net_return_t = gross_return_t - transaction_cost_t
```

Unit convention:

```text
cost_decimal_i = cost_bps_i / 10,000
```

Example:

- 2 bps = `0.0002`
- 18 bps = `0.0018`
- CASH = `0.0`

This preserves the same turnover-based structure as the current model, but replaces one scalar cost with an asset-specific vector.

## 3. IBKR-Inspired Assumptions

The proposed defaults should be broker-inspired approximations, not exact execution simulation.

Suggested baseline proxy:

- ETF assets (`SPY`, `TLT`, `GLD`): 2 bps per one-way traded notional
- `BTC-USD`: 18 bps per one-way traded notional
- `CASH`: 0 bps

Suggested stress proxy:

- ETF assets: 5 bps
- `BTC-USD`: 30 bps
- `CASH`: 0 bps

Motivation:

- IBKR Pro tiered US stock/ETF pricing starts at 0.0035 USD/share for lower monthly share volume tiers, with a minimum commission per order.
- IBKR fixed US stock/ETF pricing is 0.005 USD/share, with a minimum commission and a trade-value cap.
- Crypto commissions through IBKR/Paxos/Zero Hash are published as percentage-of-trade-value rates that are materially higher than liquid ETF commissions.

Important caveat:

This backtest operates on weekly portfolio weights, not routed orders. Therefore, the asset-specific model remains an approximation.

Not fully modeled:

- minimum commissions;
- bid-ask spreads;
- market impact;
- taxes;
- order routing;
- partial fills;
- intraday execution timing;
- share-level rounding;
- venue-specific fees and rebates.

The correct paper language should be “asset-specific transaction-cost-aware training” or “broker-inspired transaction-cost approximation,” not “exact IBKR execution.”

## 4. Implementation Locations

### Portfolio Environment

Primary implementation location:

- `src/env/portfolio_env.py`

The environment should accept either:

- existing scalar `transaction_cost_rate`; or
- new asset-specific cost mapping/vector.

Backward compatibility must be preserved. If no asset-specific vector is provided, the environment should behave exactly as it does now.

Suggested environment behavior:

```text
if asset_specific_transaction_costs is provided:
    cost_t = sum_i cost_i * abs(w_t,i - w_{t-1,i})
else:
    cost_t = scalar_transaction_cost_rate * sum_i abs(w_t,i - w_{t-1,i})
```

The environment should report:

- total transaction cost;
- total turnover;
- asset-level turnover if available;
- asset-level transaction-cost contribution if enabled;
- transaction-cost mode: `scalar` or `asset_specific`.

### Config Schema

Likely locations:

- `configs/config.yaml`
- `configs/empirical_long_history.yaml`
- `src/utils/config.py`

Suggested config fields:

```yaml
environment:
  transaction_cost_rate: 0.001
  transaction_cost_mode: scalar
  asset_transaction_cost_bps:
    SPY: 2.0
    TLT: 2.0
    GLD: 2.0
    BTC-USD: 18.0
    CASH: 0.0
```

Rules:

- `transaction_cost_mode: scalar` uses existing behavior.
- `transaction_cost_mode: asset_specific` requires a complete asset-cost mapping.
- Missing assets should fail fast.
- Negative costs should fail validation.
- CASH should be allowed only as zero or explicitly documented if nonzero.

### Reward Calculation

The reward should consume the realized transaction cost reported by the environment.

Do not duplicate transaction-cost logic inside the reward.

Expected behavior:

- environment computes realized asset-specific cost;
- reward receives or references realized cost diagnostics;
- transaction-cost reward penalty remains semantically aligned with actual environment cost.

If `lambda_transaction_cost` remains active, it should penalize the realized cost under whichever mode is active.

### Benchmark Cost Handling

Benchmark runner logic should be reviewed if asset-specific costs become part of the training protocol.

Likely affected areas:

- benchmark history generation;
- rolling benchmark turnover calculation;
- buy-and-hold initial allocation cost;
- rebalanced equal-weight and dynamic benchmark cost assumptions.

To preserve comparability, benchmarks should use the same transaction-cost mode as TD3 when producing final comparison tables.

### Reporting Outputs

Affected reporting tables should include:

- `transaction_cost_mode`;
- scalar cost rate, if applicable;
- asset-specific cost vector, if applicable;
- average transaction cost;
- asset-level cost contributions, if available;
- average turnover;
- cost sensitivity caveat.

Existing historical outputs should not be overwritten. Asset-specific runs should write to new output directories.

## 5. Required Tests

Minimum unit tests:

- zero turnover produces zero transaction cost;
- CASH has zero cost;
- BTC cost is higher than ETF cost for equal one-way traded notional;
- equal asset cost vector reproduces the old scalar proportional model;
- net return is reduced by the asset-specific transaction cost;
- config validation rejects missing asset costs;
- config validation rejects negative costs;
- backward compatibility: scalar cost mode produces identical results to current behavior;
- diagnostics report transaction-cost mode correctly;
- reward receives the realized environment transaction cost, not a duplicated calculation.

Benchmark tests:

- buy-and-hold initial trade cost is computed consistently;
- rebalanced benchmarks use asset-specific cost when enabled;
- CASH transitions are not charged when CASH cost is zero.

## 6. Experiment Plan

### Phase 1: Unit Tests Only

Implement environment/config support and tests.

Do not run TD3 training in this phase.

### Phase 2: Smoke Run

Run a short smoke protocol with asset-specific costs enabled.

Suggested smoke:

- one candidate;
- one seed;
- few episodes;
- one or two folds;
- verify diagnostics and output tables.

### Phase 3: Limited Candidate Retraining

Retrain only the most relevant candidates:

- `V3_real_macro_vintage_clean_no_dxy` with cap `0.50`
- `V3_real_macro_vintage_clean_no_dxy` with cap `0.60`
- `V7_real_macro_vintage_clean_no_dxy_garch` with cap `0.50`
- `V4_real_garch_current` with cap `0.50`

Use the same fold/seed protocol as the current final report.

### Phase 4: Conditional Expansion

Only expand to all candidates if rankings materially change.

Expansion triggers:

- leading candidate changes;
- BTC allocation changes materially;
- turnover-heavy candidates are meaningfully penalized;
- mandate-aware ranking shifts enough to affect paper conclusions.

If rankings do not materially change, keep the broader candidate set frozen.

## 7. Risks

Main risks:

- results are no longer directly comparable to existing final reports;
- final tables may need regeneration;
- turnover-heavy strategies may be reduced or reranked;
- BTC allocations may become less attractive;
- learned policies may change because the cost signal changes during training;
- benchmark comparisons must be regenerated under the same cost model;
- the main conclusion may remain unchanged, making the implementation useful but not central.

The most important methodological risk is mixing scalar-cost-trained TD3 policies with asset-specific-cost-trained benchmark comparisons or vice versa. Final claims require consistent cost assumptions across TD3 and benchmarks.

## 8. Paper Impact

If implemented, the paper should describe the result as:

> asset-specific transaction-cost-aware TD3 training

or:

> broker-inspired asset-specific transaction-cost approximation

If not implemented, the current paper remains a transaction-cost sensitivity study only.

Current status without this implementation:

- the training protocol uses scalar proportional turnover costs;
- the reporting layer tests IBKR-inspired cost sensitivity ex post;
- conclusions should not claim that TD3 was trained under asset-specific broker costs.

If implemented and rerun, the paper must clearly separate:

- scalar-cost final report;
- asset-specific-cost retraining report;
- reporting-only transaction cost sensitivity.

No paper claim should imply exact broker execution.
