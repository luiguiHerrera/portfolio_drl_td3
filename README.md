# Falsification-Oriented Evaluation of DRL Portfolio Allocation

*A cross-asset TD3 case study with trading costs, explicit cash assumptions,
matched benchmarks, and statistical controls.*

Financial DRL results often depend on convenient evaluation choices hidden
behind the final Sharpe ratio. I built this project to make those choices
explicit and test whether the conclusions survived them.

TD3 is the case study, not the contribution. It remained competitive in a few
descriptive comparisons, but it did not win the full benchmark universe.
Several favorable conclusions also disappeared after I corrected the date
alignment and the aggregation across training seeds.

**Paper:** [PDF](paper/main.pdf) · [LaTeX](paper/main.tex) ·
[validation notes](paper/README.md)

## Findings

| Question | Result |
| --- | --- |
| Does TD3 lead the combined ranking? | No. Buy-and-hold GLD ranks first under both cash protocols. |
| Is any TD3 candidate descriptively competitive? | Yes, within limits. V3 ranks fourth by mandate-aware score under Zero-CASH and V7 seventh under BIL-CASH. Both rank tenth by canonical Sharpe. |
| Is statistical superiority established? | No. Both paired bootstrap ranges include zero, and White Reality Check does not reject its null. |
| Does an aggregate TD3 candidate pass a tested mandate? | No. Three deterministic benchmarks pass the aggressive profile. |
| Does cash treatment affect selection? | Yes. The highest mandate-aware TD3 changes from V3 to V7. |
| Are any TD3 candidates Pareto-relevant? | Conditionally. V3/V4 remain on both Zero-CASH frontiers; under BIL-CASH, V3 remains on both and V8 on the full frontier only. |
| Do execution assumptions matter? | Yes. Added spread assumptions weaken selected TD3 histories more than Trend SPY/CASH. |
| Was 60 episodes obviously insufficient? | No clear evidence. Longer budgets do not consistently improve the selected cases. |

“Not statistically superior” does not mean “useless,” and Pareto membership
does not mean “best.” The evidence supports limited descriptive claims, not
deployable alpha or universal DRL superiority.

## Experiment

| Item | Design |
| --- | --- |
| Assets | `SPY`, `TLT`, `GLD`, `BTC-USD`, and `CASH`/`BIL` |
| Portfolio decision | Weekly, long-only weights with concentration constraints |
| TD3 grid | 5 feature families × 4 caps × 10 seeds × 4 walk-forward folds |
| Scale | 800 TD3 histories per cash protocol; they share one market record and are not independent samples |
| Final comparison | 5 selected TD3 candidates and 14 deterministic rule-based benchmarks |
| Cash and costs | Zero-CASH or a BIL proxy; asset-specific transaction costs from 0 to 10 bps |
| Matched window | 228 Fridays, 2022-01-07 through 2026-05-15 |

The benchmarks cover buy-and-hold, equal weight, 60/40, momentum, risk-off,
risk parity, rolling Markowitz, and Trend SPY/CASH. The evidence stack is:

```text
candidate search → walk-forward tests → matched ranking → bootstrap/WRC
→ mandate filters → Pareto/regime analysis → execution/training diagnostics
```

A high rank is only a ranking result. It does not settle statistical credibility
or practical feasibility.

## Two corrections that changed the result

### Temporal alignment

The original combined table mixed TD3 and benchmark metrics from different
evaluation horizons. The corrected pipeline intersects every strategy on the
same 228 Friday observations. Rankings, scores, mandate checks, and Pareto
membership now come only from those matched histories.

### Seed aggregation

Averaging ten policy return paths by date created a synthetic diversified path
and reduced volatility by roughly 18–34%. It was neither the expected outcome
of training one policy nor an implemented ten-agent portfolio. The final method
concatenates four out-of-sample folds inside each seed, calculates metrics on
each complete seed history, and then aggregates those metrics across ten seeds.
Average and median return paths remain diagnostics only.

The reporting also separates canonical Sharpe, the paired report's historical
`CAGR / annualized volatility` ratio, and the White Reality Check statistic on
weekly return differentials. Their formulas and estimands are recorded in the
[seed-aggregated metadata](outputs/paper_seed_aggregated_comparison/metadata/methodology.json).

## Repository map

| Path | Contents |
| --- | --- |
| [`paper/`](paper/) | Manuscript, PDF, figures, and build notes |
| [`src/models/`](src/models/), [`src/env/`](src/env/), [`src/train/`](src/train/) | TD3 agent, portfolio environment, and training logic |
| [`src/experiments/`](src/experiments/) | Walk-forward, seed, feature, and concentration experiments |
| [`src/backtest/`](src/backtest/), [`src/analysis/`](src/analysis/) | Benchmarks, metrics, statistical tests, constraints, Pareto, and audits |
| [`scripts/`](scripts/), [`tests/`](tests/) | Rebuild entry points, validators, and regression tests |
| [`outputs/paper_aligned_comparison/`](outputs/paper_aligned_comparison/) | Exact-date histories and comparison lineage |
| [`outputs/paper_seed_aggregated_comparison/`](outputs/paper_seed_aggregated_comparison/) | Per-seed metrics, rankings, diagnostics, metadata, and paper fragments |

## Reproduction

Create the environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Validate committed results

With the packaged output directories present, these checks do not need the
external source histories:

```bash
PYTHONPATH=. .venv/bin/python scripts/validate_paper_aligned_comparison.py \
  --output-dir outputs/paper_aligned_comparison

PYTHONPATH=. .venv/bin/python scripts/validate_paper_seed_aggregated_comparison.py \
  --output-dir outputs/paper_seed_aggregated_comparison
```

The validators fail on index mismatches, missing or duplicate observations,
incorrect aggregation order, or downstream rankings that do not match their
packaged inputs.

### Rebuild from original sources

Full regeneration requires an external history bundle that is not fully stored
in this repository. Supply a root containing the BIL TD3 histories, both
benchmark-history sets, and the referenced statistical outputs:

```bash
PYTHONPATH=. .venv/bin/python scripts/build_paper_aligned_comparison.py \
  --repo-root . \
  --external-root /path/to/portfolio_drl_outputs

PYTHONPATH=. .venv/bin/python scripts/build_paper_seed_aggregated_comparison.py \
  --repo-root . \
  --external-root /path/to/portfolio_drl_outputs
```

The second build validates and reuses the aligned package. Neither command
retrains TD3.

### Focused tests and paper build

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover \
  -s tests -p 'test_paper_aligned_comparison.py' -v

PYTHONPATH=. .venv/bin/python -m unittest discover \
  -s tests -p 'test_paper_seed_aggregated_comparison.py' -v

tectonic -o paper paper/main.tex
```

For the full argument and evidence trail, see the [paper](paper/main.pdf), its
[source](paper/main.tex), the [build notes](paper/README.md), and the
[final evidence audit](docs/final_paper_full_audit.md).

## Implementation

- The experiment runner crosses feature families, concentration caps, folds,
  and seeds without treating repeated histories as independent market samples.
- The alignment layer requires identical timestamps for every strategy before
  recomputing metrics or scores.
- The portfolio environment applies long-only constraints and charges
  asset-specific costs on weight changes.
- The benchmark and analysis code separates ranking from block-bootstrap,
  White Reality Check, mandate, Pareto, regime, and execution evidence.
- Generated packages record source hashes, formulas, estimands, and operation
  order; validators and tests check the paper's dependency chain.

## Limitations

- The universe is compact, mainly U.S.-centric, and observed over one market
  history. Seeds are training repetitions, not new market samples.
- TD3 is the only primary DRL algorithm tested. Deterministic benchmarks have
  no comparable training-seed distribution.
- Mandate thresholds are test profiles, not universal investor rules.
- Spread stress is a proxy. There is no market-impact model, order-book
  simulation, capacity analysis, or live-forward validation.
- Some source histories needed for full regeneration remain external.

This is research code, not investment advice or a production trading system.

I did not build the evaluation to make TD3 win. I built it to find out which
claims survived after removing the assumptions that made the result look
better. Fewer did.
