# Asset Universe Justification

The empirical universe is deliberately compact. It is not intended to represent a complete global multi-asset portfolio. It is a small set of economically distinct risk sleeves that makes the allocation problem interpretable, auditable, and hard enough to test whether TD3 learns useful dynamic allocation behavior under realistic frictions.

## Risk Sleeves

| Asset | Sleeve | Economic role |
| --- | --- | --- |
| SPY | Equity / growth risk | Broad U.S. equity exposure and the main pro-cyclical growth sleeve. |
| TLT | Duration / interest-rate risk | Long-duration U.S. Treasury exposure. It can hedge some equity drawdowns, but it is not risk-free and can suffer large losses when rates rise. |
| GLD | Hard asset / safe-haven proxy | Gold exposure, used as a hard-asset and crisis-hedge sleeve. It is not a cash substitute and does not have the same risk profile as Treasury duration. |
| BTC-USD | Digital alternative / speculative convexity | A high-volatility alternative asset with asymmetric upside and large idiosyncratic risk. It is not assumed to be digital gold. |
| CASH | Defensive / optionality sleeve | A risk-reduction allocation sleeve. In the main protocol it is synthetic zero-return cash; in robustness checks it is replaced by a BIL short-term Treasury ETF proxy. |

## Why This Universe Is Compact

The goal is not to maximize asset coverage. The goal is to test a continuous-control allocation agent across a small set of sleeves with different economic behavior:

- growth risk through equities;
- interest-rate and duration risk through long Treasuries;
- hard-asset and safe-haven exposure through gold;
- speculative alternative risk through Bitcoin;
- explicit defensive optionality through cash.

A larger universe would add realism, but it would also make interpretation harder. For this thesis, the compact universe makes the falsification exercise clearer: if TD3 cannot remain credible in this small, economically interpretable cross-asset setting, stronger claims in a broader universe would be premature.

## Gold and Bitcoin Are Different Sleeves

GLD and BTC-USD are not treated as interchangeable. Gold is included as a hard-asset and safe-haven proxy with a long institutional history. Bitcoin is included as a digital alternative asset with high volatility, policy uncertainty, liquidity episodes, and idiosyncratic adoption risk. The experiment allows TD3 to allocate to both, but it does not assume that Bitcoin is "digital gold" or that the two assets hedge the same states of the world.

## TLT Is Not Risk-Free

TLT represents long-duration Treasury exposure, not a risk-free asset. It can behave defensively in some equity drawdowns, but it embeds substantial interest-rate risk. The 2022 rate shock is exactly the type of environment in which treating long bonds as risk-free would be misleading. This is why CASH remains a separate allocation sleeve.

## Why CASH Is Explicit

CASH is included because avoiding risk is an allocation decision. A long-only fully invested agent without a defensive sleeve must always express risk through risky assets. The main protocol uses synthetic zero-return CASH to isolate allocation behavior. A robustness protocol replaces CASH with BIL proxy returns to test whether the cash-return assumption materially changes the selected TD3 specification.

## Limitations

This universe is intentionally incomplete:

- it is U.S.-centric;
- it excludes credit spreads and corporate bonds;
- it excludes real estate;
- it excludes broad commodities beyond gold;
- BTC-USD is highly idiosyncratic;
- BIL is an investable short-term Treasury proxy, not perfect cash;
- transaction-cost and spread assumptions remain approximations.

## Paper Paragraph

The asset universe is designed as a compact cross-asset risk-sleeve laboratory rather than a complete global portfolio. SPY represents equity growth risk, TLT represents duration and interest-rate risk, GLD represents a hard-asset and safe-haven sleeve, BTC-USD represents a speculative digital alternative sleeve, and CASH represents defensive optionality. This structure is intentionally small enough to keep allocation behavior interpretable while still forcing the TD3 agent to choose among economically distinct sources of risk. The design does not assume that Bitcoin is digital gold, does not treat long-duration Treasuries as risk-free, and does not claim to span the full opportunity set available to an institutional investor.
