# Paper draft and aligned reproducibility package

This folder contains the LaTeX source, bibliography, paper-owned figures, and
rendered PDF for the TD3 portfolio-allocation paper.

Build from the repository root:

```bash
tectonic -o paper paper/main.tex
```

The output is `paper/main.pdf`. `latexmk -pdf -outdir=paper paper/main.tex` is
an alternative when the required TeX packages are installed.

The source is `paper/main.tex`; references are in `paper/references.bib`. Three
figures are generated inline by LaTeX. The two external PDF figures used by the
paper are owned by `paper/figures/`:

```text
execution_spread_sharpe_degradation.pdf
training_budget_sharpe_convergence.pdf
```

The primary benchmark comparison, combined ranking, scores, hard mandates,
mandate profiles, and Pareto results used by the paper are packaged under:

```text
outputs/paper_seed_aggregated_comparison/
```

They use the exact 228 Friday timestamps from 2022-01-07 through 2026-05-15 for
all 19 strategies in each cash protocol. TD3 metrics are calculated on each of
ten complete OOS seed histories before their arithmetic mean is used for the
ranking. The package contains all seed histories and metrics, dispersion
summaries, average/median-path diagnostics, rankings, mandates, Pareto tables,
source hashes, methodology, an aggregation audit, and LaTeX fragments used
directly by Tables 8 and 9.

Regenerate and validate from the repository root with the external canonical
source root supplied explicitly:

```bash
PYTHONPATH=. .venv/bin/python scripts/build_paper_seed_aggregated_comparison.py \
  --repo-root . \
  --external-root /path/to/portfolio_drl_outputs

PYTHONPATH=. .venv/bin/python scripts/validate_paper_seed_aggregated_comparison.py \
  --output-dir outputs/paper_seed_aggregated_comparison
```

The new package reuses and validates the preserved exact-date prerequisite at
`outputs/paper_aligned_comparison/`; it does not regenerate the general
alignment audit. The scripts never select `tail(228)`, forward-fill dates,
extrapolate benchmarks, overwrite source histories, train TD3, or alter
candidate hyperparameters.

For verification after generation, no external histories are required. The
validator recomputes every per-seed return metric, checks that aggregation
occurs after metric calculation, and rebuilds rankings, mandates, and Pareto
membership from the packaged evidence.

Exact source-level regeneration still requires the external BIL candidate test
histories and both cash-specific benchmark-history directories. The existing
pairwise bootstrap, White Reality Check, regime, spread-stress, and
training-budget result packages also remain external; their reported figures
were not regenerated in the alignment task.

The previous 228-versus-581/593-week horizon mismatch is resolved. The date-wise
average TD3 return path remains available only as a synthetic diversification
diagnostic; it is not treated as expected performance or as a deployable
ensemble. The pairwise bootstrap's legacy field named “Sharpe” is labeled by its
actual estimator, the CAGR-to-annualized-volatility ratio. White Reality Check
is documented separately as a centered bootstrap of weekly return
differentials and does not use that ratio.
