# Execution Spread Robustness Design

## Purpose

This layer adds an optional execution-friction robustness check on top of already-generated strategy histories.

It does not retrain TD3, does not modify the portfolio environment, and does not change the final corrected model-selection results. It asks a narrower question: how sensitive are selected TD3 and benchmark histories to additional bid-ask spread assumptions?

## Cost Model

The module models top-of-book bid-ask half-spread costs.

For quote data with:

- `timestamp`
- `asset`
- `bid`
- `ask`

the proportional half-spread is:

```text
mid = (bid + ask) / 2
half_spread = (ask - bid) / (2 * mid)
```

The spread cost applied to a rebalance is:

```text
spread_cost = sum_i(abs(target_weight_i - drifted_weight_i) * asset_half_spread_i)
```

`CASH` always has zero spread in this layer.

## Weekly Aggregation

Two weekly quote aggregations are supported:

- weekly mean half-spread;
- weekly close/last-observed half-spread.

The reporting layer can also use existing `asset_turnover_*` columns in strategy histories. In that case, the stored one-way asset turnover is multiplied by the scenario half-spread.

## Proxy Mode

If quote-level bid/ask data is unavailable, the report uses explicit scenario assumptions:

- `base_no_extra_spread`
- `institutional_clean_spread`
- `conservative_spread`
- `stress_spread`

Proxy spreads may optionally scale with rolling volatility through:

```text
dynamic_half_spread = base_half_spread * volatility_regime_multiplier
```

These proxy spreads are robustness assumptions. They are not calibrated execution estimates.

## Scope

This is not:

- market impact;
- order-book simulation;
- tax modeling;
- broker routing simulation;
- a new model-selection layer.

It is a post-training reporting layer that makes execution realism more explicit without rewriting the final corrected protocol.

## Outputs

The report writes:

- `execution_spread_strategy_metrics.csv`
- `execution_spread_degradation_summary.csv`
- `execution_spread_summary.md`
- `execution_spread_metadata.json`

The metadata records scenario assumptions, source histories, warnings, and the fact that no retraining occurred.
