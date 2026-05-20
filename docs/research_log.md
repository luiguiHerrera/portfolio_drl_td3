# Research Log - Portfolio DRL TD3

## Purpose

This log records technical decisions, methodological reasoning, experiments,
results, and pending risks for the Portfolio DRL TD3 research process. It is
intended to support later writing of the TFM and any related publication by
preserving why changes were made, how they were tested, what was observed, and
which limitations remain unresolved.

## Templates

### Template A - Technical Development Entry

- Date:
- Commit / working state:
- Modules affected:
- Problem detected:
- Decision implemented:
- Methodological justification:
- Tests executed:
- Result:
- Implication for TFM/publication:
- Pending items:

### Template B - Methodological Experiment Entry

- Date:
- Research question:
- Initial hypothesis:
- Experiment:
- Observed result:
- Financial interpretation:
- Next decision:
- Interpretation risk:
- How it will be reported:

## Initial Entries

### TD3 Timing Fix

- Date: 2026-05-12
- Commit / working state: Working state; exact commit to be filled when logged
  against version control.
- Modules affected: Data preparation, environment interaction, training, and
  evaluation workflow.
- Problem detected: The timing convention needed to ensure that the agent did
  not observe information from the same period whose return it was trying to
  capture.
- Decision implemented: Features are shifted by one period, so the agent
  observes information available through the previous period. Current action
  weights are then applied to the current realized return.
- Methodological justification: This preserves chronological causality and
  reduces look-ahead bias in the portfolio allocation experiment.
- Tests executed: Unit test suite to be recorded for the specific working
  state.
- Result: Timing semantics align the state, action, and realized return more
  closely with an investable decision process.
- Implication for TFM/publication: The methodology section should explicitly
  describe the decision timing convention and its role in avoiding look-ahead
  bias.
- Pending items: Confirm timing behavior in future walk-forward and empirical
  experiment designs.

### Financial Net Evaluation Fix

- Date: 2026-05-12
- Commit / working state: Working state; exact commit to be filled when logged
  against version control.
- Modules affected: Environment accounting, evaluation metrics, and comparison
  workflow.
- Problem detected: Financial portfolio accounting and learning reward needed
  to be clearly separated.
- Decision implemented: Portfolio value and reported performance metrics use
  financial net returns after transaction costs. The learning reward remains a
  separate configurable training signal.
- Methodological justification: Reported performance should correspond to
  realized financial outcomes, while the reward can include additional shaping
  terms for learning stability and risk control.
- Tests executed: Unit test suite to be recorded for the specific working
  state.
- Result: Evaluation now distinguishes investable net performance from the
  reward used for policy optimization.
- Implication for TFM/publication: Results tables should identify metrics as
  net financial performance and explain that reward design is not identical to
  portfolio return accounting.
- Pending items: Maintain this distinction when adding plots, reports, and
  additional benchmark workflows.

### Risk-Aware Reward Implementation

- Date: 2026-05-12
- Commit / working state: Working state; exact commit to be filled when logged
  against version control.
- Modules affected: Reward function, environment, configuration, and training
  workflow.
- Problem detected: A pure return-oriented objective may encourage policies
  with excessive trading, concentration, or drawdown exposure.
- Decision implemented: The reward function includes configurable penalties for
  transaction cost, turnover, concentration, and drawdown.
- Methodological justification: These terms encode economically relevant
  frictions and risk preferences while preserving a continuous-action portfolio
  optimization framework.
- Tests executed: Unit test suite to be recorded for the specific working
  state.
- Result: Reward shaping is configurable and can be studied through sensitivity
  analysis.
- Implication for TFM/publication: The reward specification should be presented
  as a methodological design choice, with ablation or sensitivity analysis used
  to assess its effect.
- Pending items: Evaluate robustness across penalty weights, random seeds, and
  market regimes before drawing empirical conclusions.

### Individual Buy-and-Hold Benchmarks

- Date: 2026-05-12
- Commit / working state: Working state; exact commit to be filled when logged
  against version control.
- Modules affected: Benchmark comparison and experiment summary workflow.
- Problem detected: A TD3 policy may become highly concentrated, making it
  necessary to test whether performance reflects dynamic allocation skill or
  simple exposure to a single strong asset.
- Decision implemented: Individual buy-and-hold benchmarks were added for each
  asset, enabling comparison against simple single-asset exposures.
- Methodological justification: Single-asset buy-and-hold benchmarks provide a
  stricter reference point when learned allocations concentrate in one asset.
- Tests executed: Unit test suite to be recorded for the specific working
  state.
- Result: Comparison outputs can identify the best individual buy-and-hold
  benchmark and quantify the agent's differences versus it.
- Implication for TFM/publication: Empirical claims should compare the agent not
  only to diversified baselines but also to the strongest simple asset exposure.
- Pending items: Extend reporting to include robustness across seeds and
  alternative time splits.

### Latest Baseline Net Reward Grid With Individual Buy-Hold Comparison

- Date: 2026-05-12
- Research question: Does the current baseline net reward configuration produce
  test performance that exceeds the best individual buy-and-hold asset
  benchmark?
- Initial hypothesis: Risk-aware TD3 may improve risk-adjusted performance if
  dynamic allocation adds value beyond static exposure to the strongest asset.
- Experiment: Baseline net reward grid with individual buy-and-hold benchmark
  comparison.
- Observed result: Experiment E achieved test Sharpe 2.7338 and test
  cumulative return 58.51%. The best individual buy-and-hold benchmark was
  `buy_hold_GLD`, with Sharpe 1.6856. The agent outperformed this benchmark by
  +1.0481 Sharpe.
- Financial interpretation: The result is consistent with the possibility that
  the learned policy added value beyond static gold exposure in this run.
- Next decision: Treat Experiment E as a candidate configuration for further
  robustness testing.
- Interpretation risk: The result is preliminary and may be sensitive to random
  seed, time split, transaction cost assumptions, reward penalties, or asset
  regime. It should not be interpreted as established empirical superiority.
- How it will be reported: Report as a preliminary single-run result requiring
  seed sensitivity and robustness analysis before any stronger conclusion is
  made.

### Seed Sensitivity for Experiment E

- Date: 2026-05-12
- Research question: Is the best current TD3 configuration stable across
  random seeds?
- Initial hypothesis: The previously observed strong result for experiment E
  may represent either a genuinely robust policy or a seed-sensitive outcome.
- Experiment: Ran seed sensitivity for experiment E with episodes = 100,
  batch_size = 64, actor_learning_rate = 0.0003, critic_learning_rate =
  0.0003, seeds = [7, 21, 42, 73, 101], and base config =
  `configs/local/smoke_risk_aware.yaml`.
- Observed result: mean_test_agent_sharpe = 0.2395,
  std_test_agent_sharpe = 0.9826, min_test_agent_sharpe = -0.9369,
  max_test_agent_sharpe = 1.7501, mean_test_agent_cumulative_return = 2.46%,
  win_rate_vs_best_individual_buyhold_by_sharpe = 20%, and
  win_rate_best_policy_agent = 0%. The best individual buy-and-hold benchmark
  was `buy_hold_GLD` across the runs shown. Only seed 101 produced a positive
  Sharpe advantage versus the best individual buy-and-hold benchmark.
- Financial interpretation: The strong result previously observed for
  experiment E is not stable across seeds. The TD3 implementation can produce
  competitive policies, but the current training setup does not provide robust
  empirical evidence of stable superiority.
- Next decision: Do not claim empirical superiority yet. The next step should
  focus on improving stability and validation design rather than adding more ad
  hoc reward penalties.
- Interpretation risk: A five-seed sample is still limited, and results may
  depend on the selected date range, asset universe, and reward configuration.
  However, the observed dispersion is large enough to reject any strong
  robustness claim at this stage.
- How it will be reported: As a robustness check showing that preliminary
  out-of-sample success is seed-sensitive and requires further methodological
  refinement before making performance claims.

## Entry X — Robust Seed Sensitivity for Experiment E

**Date:** 2026-05-12

**Configuration:**
- Base config: `configs/local/smoke_risk_aware.yaml`
- Episodes: 100
- Batch size: 64
- Actor learning rate: 0.0003
- Critic learning rate: 0.0003
- Seeds: 7, 21, 42, 73, 101

**Main robust summary:**
- Mean test Sharpe: 0.0520
- Standard deviation test Sharpe: 0.7905
- Robust Sharpe score 0.5: -0.3432
- Robust Sharpe score 1.0: -0.7385
- Mean test Sortino: 0.0704
- Robust Sortino score 0.5: -0.6285
- Mean Information Ratio vs equal-weight rebalanced net: -0.9199
- Robust Information Ratio score 0.5: -1.4453
- Mean CAPM alpha vs SPY: -0.1387
- Robust CAPM alpha score 0.5: -0.2780
- Positive Sharpe rate: 0.40
- Positive CAPM alpha rate: 0.40
- Positive Information Ratio rate: 0.20
- Win rate as best policy: 0.00
- Win rate vs best individual buy-and-hold: 0.00

**Interpretation:**
The robust seed sensitivity analysis confirms that Experiment E is not stable across random seeds. Although some seeds produce positive Sharpe and alpha, the robust scores are negative once dispersion is penalized. The agent does not become the best policy in any seed and does not beat the best individual buy-and-hold benchmark by Sharpe.

**Research implication:**
The current TD3 setup should not be presented as empirically superior. Its value at this stage is methodological: the framework now detects instability, seed dependence, benchmark weakness, and lack of robust alpha generation.

## Entry X — Hyperparameter × Seed Robustness Grid

**Date:** 2026-05-13

**Objective:**  
Evaluate TD3 hyperparameter configurations across multiple random seeds using robust performance scores, instead of selecting models from single-seed results.

**Setup:**
- Base config: `configs/local/smoke_risk_aware.yaml`
- Experiments: A-H
- Seeds: 7, 21, 42, 73, 101
- Output: `outputs/tables/td3_hyperparameter_seed_grid_AH_5seeds/`

**Main result:**  
The best-ranked configuration by robust Sharpe was Experiment H:

- Description: `higher_learning_rate`
- Episodes: 100
- Batch size: 32
- Actor learning rate: 0.0005
- Critic learning rate: 0.0005
- Mean test Sharpe: 0.5830
- Standard deviation test Sharpe: 0.5889
- Robust Sharpe score 0.5: 0.2886
- Robust Sharpe score 1.0: -0.0059
- Mean Sortino: 0.9692
- Robust Sortino score 0.5: 0.3509
- Mean Information Ratio vs equal-weight rebalanced net: -0.4100
- Robust Information Ratio score 0.5: -0.7678
- Mean CAPM alpha vs SPY: 0.0367
- Robust CAPM alpha score 0.5: -0.0933
- Positive Sharpe rate: 0.80
- Positive Sortino rate: 0.80
- Positive CAPM alpha rate: 0.60
- Positive Information Ratio rate: 0.20
- Win rate as best policy: 0.00
- Win rate vs best individual buy-and-hold by Sharpe: 0.00

**Interpretation:**  
Experiment H is the strongest relative candidate under the current design, but the evidence is still not sufficient to claim robust superiority. Its robust Sharpe score under a mild penalty is positive, but the stronger robust Sharpe score is approximately zero. Information ratio and robust CAPM alpha remain negative, and the agent never wins against all benchmark policies.

**Research implication:**  
The next stage should use Experiment H as the reference TD3 configuration for further methodological improvements, but not as a final empirical result. The current framework is now useful because it identifies instability, benchmark weakness, and seed sensitivity directly.

## Entry X — Walk-Forward Validation for Experiment H

**Date:** 2026-05-13

**Objective:**  
Evaluate the current reference TD3 configuration, Experiment H, using explicit
chronological walk-forward validation instead of relying on one fixed
train/validation/test split.

**Setup:**
- Base config: `configs/local/smoke_risk_aware.yaml`
- Output directory: `outputs/tables/walk_forward_H_100/`
- Seed: 42
- Episodes: 100
- Batch size: 32
- Actor learning rate: 0.0005
- Critic learning rate: 0.0005
- Folds:
  - F1: train 2021-2022, validate 2023H1, test 2023H2
  - F2: train 2021H2-2023H1, validate 2023H2, test 2024H1
  - F3: train 2022-2023, validate 2024H1, test 2024H2

**Fold-level results:**
- F1: cumulative return 2.61%, Sharpe 0.4112, Sortino 0.5348,
  Information Ratio vs equal-weight rebalanced net -1.1544, CAPM alpha 0.0402,
  max drawdown -9.84%, Sharpe rank 7, best policy
  `equal_weight_gross`, best individual buy-and-hold `buy_hold_BTC-USD`,
  Sharpe difference vs best individual buy-and-hold -1.3822, average turnover
  0.3016, and average effective number of assets 1.0358.
- F2: cumulative return 24.11%, Sharpe 3.2960, Sortino 9.9386,
  Information Ratio vs equal-weight rebalanced net 1.9193, CAPM alpha 0.3859,
  max drawdown -4.22%, Sharpe rank 1, best policy `agent`, best individual
  buy-and-hold `buy_hold_SPY`, Sharpe difference vs best individual
  buy-and-hold 0.2858, average turnover 0.6807, and average effective number
  of assets 1.1958.
- F3: cumulative return 2.03%, Sharpe 0.2970, Sortino 0.1558,
  Information Ratio vs equal-weight rebalanced net -0.7910, CAPM alpha
  -0.1078, max drawdown -21.48%, Sharpe rank 7, best policy
  `equal_weight_gross`, best individual buy-and-hold `buy_hold_BTC-USD`,
  Sharpe difference vs best individual buy-and-hold -1.6164, average turnover
  0.8514, and average effective number of assets 1.0782.

**Aggregate summary:**
- Mean test Sharpe: 1.3347
- Standard deviation test Sharpe: 1.6995
- Minimum test Sharpe: 0.2970
- Maximum test Sharpe: 3.2960
- Robust Sharpe score 0.5: 0.4850
- Robust Sharpe score 1.0: -0.3647
- Mean test Sortino: 3.5431
- Mean Information Ratio vs equal-weight rebalanced net: -0.0087
- Mean CAPM alpha vs SPY: 0.1061
- Mean cumulative return: 9.58%
- Mean max drawdown: -11.85%
- Positive Sharpe rate: 100%
- Positive Sortino rate: 100%
- Positive CAPM alpha rate: 66.67%
- Positive Information Ratio rate: 33.33%
- Win rate as best policy: 33.33%
- Win rate vs best individual buy-and-hold by Sharpe: 33.33%
- Mean average turnover: 0.6112
- Mean average effective number of assets: 1.1033

**Interpretation:**  
The walk-forward workflow now provides a more realistic chronological
validation design than a single train/validation/test split. The agent achieved
positive Sharpe in all three test folds, which suggests some potential signal.
However, the evidence is not yet robust enough to claim empirical superiority.
Performance is dominated by F2, while F1 and F3 are positive but weak. The
robust Sharpe score with a 0.5 penalty is positive, but the score with a 1.0
penalty is negative.

The mean Information Ratio versus the equal-weight rebalanced net benchmark is
approximately zero and slightly negative. The agent was the best policy in only
one of three folds and beat the best individual buy-and-hold benchmark by
Sharpe in only one of three folds. The average effective number of assets is
close to 1, indicating that the policy remains highly concentrated.

**Research implication:**  
This result supports using walk-forward validation as the main empirical
validation framework, but it also highlights regime dependence and
concentration risk. The next research step should avoid blind tuning and use
walk-forward validation together with seed robustness. Further work should
consider improving objective stability, expanding the number of folds, and
eventually adding stronger baselines such as risk parity and Markowitz before
making performance claims.

**Interpretation risk:**  
This is the first serious walk-forward run for Experiment H, but it uses one
seed and three folds. It is stronger than a single split, yet still not enough
to establish robust out-of-sample superiority.

## Entry X — Feature Set V2 and Date Boundary Fix

**Date:** 2026-05-13

**Research question:**  
Can the state representation be enriched with more financially meaningful
return-derived features without breaking the original V1 pipeline or
introducing look-ahead/date-boundary leakage?

**Decision implemented:**  
Added opt-in Feature Set V2 through a shared feature factory. V1 remains the
default. Added date-boundary clipping after computed returns to prevent weekly
resampling labels from exceeding `data.end_date`.

**Methodological justification:**  
Feature Set V2 expands the state space with momentum, volatility, EWMA
volatility, beta/correlation versus market asset, rolling drawdown, and simple
market-regime indicators. Clipping computed returns after resampling prevents
evaluation windows from silently exceeding configured boundaries.

**Tests executed:**  
The full unit test suite passed: 366 tests OK.

**Observed result:**  
A smoke dataset check with `configs/local/smoke_feature_v2.yaml` produced 49
features and no missing values. `max_return_index` was corrected to
2024-12-27, which is <= 2024-12-31.

**Interpretation risk:**  
This is not performance evidence. It only validates pipeline correctness and
feature availability.

**Next decision:**  
After commit, run walk-forward and walk-forward-seed validation using V2 before
any tuning or performance claim.

## Entry X — Feature Set V3 Local Macro Smoke Validation

**Date:** 2026-05-13

**Research question:**  
Can local macro features be added end-to-end to the configured dataset pipeline
without introducing missing values, date leakage, or live-data dependencies?

**Decision implemented:**  
Added V3 local macro CSV plumbing and validated it with a synthetic weekly macro
fixture covering 2020-01-03 through 2024-12-27.

**Technical result:**  
The V3 macro smoke workflow produced `train_features` shape (165, 61),
`validation_features` shape (35, 61), and `test_features` shape (37, 61).
`missing_values_zero = True`, and the final test date was 2024-12-27.

**Methodological interpretation:**  
The pipeline can now carry macro/regime variables into the state representation
in a reproducible way, but this does not prove better performance.

**Safeguards:**  
No live downloads were added. Macro alignment uses forward fill only, with no
backfill. Unavailable rolling macro regimes remain NaN and are removed
downstream by `dropna`. The external one-period feature shift still applies.

**Tests:**  
The full unit test suite passed: 398 tests OK.

**Risk:**  
The macro fixture is synthetic and must not be interpreted as real macro
evidence. Real macro data still needs proper sourcing, release-lag treatment,
and validation.

**Next step:**  
Build and validate a real macro dataset with release-date awareness before
using V3 for empirical claims.

## Entry X — Feature Set Comparison and Decision to Test Real Macro Data

**Date:** 2026-05-13

**Research question:**  
Do richer risk/regime features and synthetic macro features improve TD3
robustness over V1 under identical walk-forward seed validation?

**Experiment:**  
Compared V1, V2, and V3_macro using the same benchmark pipeline, reward
definition, TD3 hyperparameters, three walk-forward folds, and five seeds. Each
run used 100 episodes, batch size 32, actor learning rate 0.0005, and critic
learning rate 0.0005. Outputs were written under
`outputs/tables/feature_set_comparison_V1_V2_V3_macro_3folds_5seeds_100ep/`.

**Result:**  
V2 ranked first by robust Sharpe with 0.5 dispersion penalty:

- V2: robust Sharpe 0.5 = 0.7125, mean Sharpe = 1.0949, standard deviation =
  0.7647, robust Sharpe 1.0 = 0.3301, mean CAPM alpha = 0.1430, robust CAPM
  alpha 0.5 = 0.0096, positive Sharpe rate = 86.67%, best-policy win rate =
  6.67%, and win rate versus the best individual buy-and-hold benchmark =
  13.33%.
- V3_macro: robust Sharpe 0.5 = 0.0111, mean Sharpe = 0.6065, standard
  deviation = 1.1908, robust Sharpe 1.0 = -0.5844, mean CAPM alpha = 0.1259,
  robust CAPM alpha 0.5 = -0.0172, positive Sharpe rate = 66.67%,
  best-policy win rate = 0.00%, and win rate versus the best individual
  buy-and-hold benchmark = 6.67%.
- V1: robust Sharpe 0.5 = -0.6491, mean Sharpe = 0.0423, standard deviation =
  1.3827, robust Sharpe 1.0 = -1.3404, mean CAPM alpha = -0.0447, robust CAPM
  alpha 0.5 = -0.1786, positive Sharpe rate = 46.67%, best-policy win rate =
  0.00%, and win rate versus the best individual buy-and-hold benchmark =
  6.67%.

**Financial interpretation:**  
V2 appears to add useful risk/regime information relative to V1 under this
controlled comparison. Synthetic V3_macro did not improve on V2.

**Caution:**  
V3_macro used synthetic macro data, so this result is not evidence against real
macro variables. It only shows that adding the current synthetic macro inputs
did not improve this run.

**Benchmark warning:**  
Information ratios remained weak or negative, and benchmark win rates remained
low. This is not evidence of robust TD3 superiority.

**Decision:**  
Do not change the reward function yet. First test V3 with real macro variables,
because the current V3 result may be contaminated by synthetic macro data.

**Proposed real macro variables:**  
`DGS10`, `DGS2`, `VIXCLS`, `DTWEXBGS`, and `CPIAUCSL`.

**Methodological safeguard:**  
The real macro implementation must avoid look-ahead bias. CPI requires
release-date or lag treatment before it can be used as a state variable. No
backfill should be used.

**Next step:**  
Build a local real macro CSV pipeline and rerun the feature set comparison
before updating README or committing the feature set comparison runner.

## Entry X — Real Local Macro Fixture and Feature Set Comparison

**Date:** 2026-05-13

**Research question:**  
Does V3 with local real macro fixture improve TD3 robustness over V2 under
identical walk-forward seed validation?

**Infrastructure:**  
Added a local macro builder from raw CSV files only. It performs weekly Friday
alignment with last available observations, uses forward fill only, applies a
simple four-week CPI availability lag before weekly alignment, and does not use
live downloads, APIs, FRED calls, yfinance calls, or backfill.

**Macro builder smoke result:**  
The weekly macro fixture produced shape (257, 5), start date 2020-01-31, end
date 2024-12-27, columns `DGS10`, `DGS2`, `VIX`, `DXY`, and `CPI`, and 0
missing values.

**V3 prepared dataset smoke:**  
The configured V3 dataset produced 61 features with no missing values.
`train_returns` had shape (128, 5), `validation_returns` shape (27, 5), and
`test_returns` shape (28, 5). The final test date was 2024-12-27.

**Experiment:**  
Compared V1, V2, and V3_real_macro_fixture using three walk-forward folds,
five seeds, 100 episodes, batch size 32, actor learning rate 0.0005, and critic
learning rate 0.0005. Outputs were written under
`outputs/tables/feature_set_comparison_V1_V2_V3_real_macro_3folds_5seeds_100ep/`.

**Result:**  
V2 ranked first by robust Sharpe with 0.5 dispersion penalty:

- V2: robust Sharpe 0.5 = 0.3821, mean Sharpe = 0.8979, robust Sharpe 1.0 =
  -0.1338, mean Information Ratio = -0.6097, mean CAPM alpha = 0.0639,
  positive Sharpe rate = 86.67%, and mean effective number of assets = 1.1143.
- V3_real_macro_fixture: robust Sharpe 0.5 = 0.1670, mean Sharpe = 0.5339,
  robust Sharpe 1.0 = -0.1999, mean Information Ratio = -0.8449, mean CAPM
  alpha = -0.0134, positive Sharpe rate = 86.67%, and mean effective number of
  assets = 1.1162.
- V1: robust Sharpe 0.5 = -0.2126, mean Sharpe = 0.2543, robust Sharpe 1.0 =
  -0.6796, mean Information Ratio = -1.2933, mean CAPM alpha = -0.0440,
  positive Sharpe rate = 60.00%, and mean effective number of assets = 1.1533.

**Financial interpretation:**  
V2 remains the strongest current representation. The real macro fixture
improved over V1 but did not add enough value beyond V2 in this controlled
comparison.

**Benchmark warning:**  
`win_rate_best_policy_agent = 0.0` and
`win_rate_vs_best_individual_buyhold_by_sharpe = 0.0` across all three feature
sets. Information ratios remain negative, so benchmark-relative performance is
still weak.

**Decision:**  
Keep V2 as the reference feature set. Do not change the reward function yet.
The next methodological priority should be benchmark and concentration
analysis, or feature ablation, rather than adding more macro complexity
blindly.

**Risk:**  
The raw macro fixture is still a local fixture, not a fully audited real-time
macro release database. CPI lag is a simple approximation, not a full
release-calendar model.

## Entry X — Real FRED Macro Long-History Comparison

**Date:** 2026-05-13

**Research question:**  
Does V3 with a real local FRED macro CSV improve TD3 robustness over V1 and V2
under identical long-history walk-forward seed validation?

**Macro data preparation:**  
Prepared local FRED-style macro CSVs for `DGS10`, `DGS2`, `VIXCLS`,
`DTWEXBGS`, and `CPIAUCSL` as a standalone acquisition step, separate from the
training and evaluation pipeline. The processed weekly macro CSV had shape
(522, 5), start date 2015-01-02, end date 2024-12-27, and 0 missing values.

**V3 dataset smoke:**  
The long-history V3 dataset preparation produced 61 features with no missing
values. The resulting split sizes were 347 train observations, 74 validation
observations, and 76 test observations.

**Experiment:**  
Compared V1, V2, and V3_real_macro using three walk-forward folds, five seeds,
and 100 training episodes per fold-seed run.

**Result:**  
The robust Sharpe scores with 0.5 dispersion penalty were:

- V3_real_macro: 0.570887
- V2: 0.560976
- V1: -0.073884

**Interpretation:**  
V3_real_macro is competitive and marginally ranks first in this comparison.
However, the difference versus V2 is too small to claim robust superiority.
This is a useful signal for continued testing, not a final empirical result.

**Benchmark warning:**  
`win_rate_best_policy_agent = 0.0` and
`win_rate_vs_best_individual_buyhold_by_sharpe = 0.0` across all feature sets.
The agent still does not beat the benchmark set consistently.

**Concentration warning:**  
The mean effective number of assets remains near 1.1, so concentration is now
the main methodological issue to diagnose before changing the objective or
adding more feature complexity.

**Decision:**  
Do not change the reward function yet. The next step is concentration
diagnostics.

## Entry X — Mandate Risk Diagnostics and Concentration Interpretation

**Date:** 2026-05-13

**Purpose:**  
Evaluate whether the learned TD3 policies comply with client-style risk
mandates instead of judging concentration as automatically wrong.

**Implementation:**  
Added `src/analysis/mandate_risk_diagnostics.py` and
`tests/test_mandate_risk_diagnostics.py`.

**Tests:**  
`python3 -m unittest tests/test_mandate_risk_diagnostics.py` ran 24 tests OK.
`python3 -m unittest discover tests` ran 450 tests OK.

**Real smoke:**  
Applied the diagnostics to 45 paired test metrics/diagnostics files from
`outputs/tables/feature_set_comparison_V1_V2_V3_real_macro_2015_2024_3folds_5seeds_100ep`.

**Moderate mandate limits:**  
`max_drawdown_limit = -0.20`, `max_volatility_limit = 0.25`,
`max_weight_limit = 0.80`, `min_effective_assets = 1.25`, and
`max_turnover_limit = 0.75`.

**Moderate result:**  
The moderate mandate produced `mandate_pass_rate = 0.0`,
`drawdown_pass_rate = 0.8889`, `volatility_pass_rate = 0.5778`,
`final_weight_pass_rate = 0.1556`, `effective_assets_pass_rate = 0.1111`, and
`turnover_pass_rate = 0.7111`. Mean Sharpe was 0.8169, mean max drawdown was
-0.1108, and mean average effective number of assets was 1.1405.

**Aggressive result:**  
Under an aggressive mandate, `mandate_pass_rate = 0.9778`.

**Corrected dominant assets:**  
- GLD: 14 observations, rate 0.3111, mean final weight 0.9713.
- BTC-USD: 11 observations, rate 0.2444, mean final weight 0.8651.
- SPY: 11 observations, rate 0.2444, mean final weight 0.9364.
- CASH: 5 observations, rate 0.1111, mean final weight 0.9344.
- TLT: 4 observations, rate 0.0889, mean final weight 0.8747.

**Interpretation:**  
The policy is highly concentrated, but not always in the same asset. It rotates
dominant exposure across assets. Under the current moderate limits, the learned
policies are not appropriate for a moderate mandate mainly because of
concentration and low effective number of assets, not because of drawdown.
Under an aggressive mandate, most observations pass.

**Methodological statement:**  
Individual buy-and-hold benchmarks remain useful references, but they are not
the only success criterion. A client mandate can rationally accept lower return
than a concentrated buy-and-hold asset if risk limits are respected.

**Next decision:**  
Do not change the reward function yet. The next step should compare mandate
compliance by feature set and then evaluate whether max-weight constraints or
concentration penalties are needed for moderate/client mandates.

## Entry X — Policy History Export and Policy Behavior Diagnostics

**Date:** 2026-05-13

**Purpose:**  
Move from aggregate diagnostics to per-period policy behavior analysis.

**Implementation:**  
Added policy history export for validation and test evaluations. Added the
Policy Behavior Diagnostics module to analyze dominant assets, concentration,
transitions, holding periods, conditional performance, and future regime
attribution.

**Tests:**  
`python3 -m unittest discover tests` ran 478 tests OK.

**Smoke result:**  
`outputs/tables/policy_history_export_smoke/test_policy_history.csv` had shape
(30, 14), with 0 missing values. Columns included `date`,
`portfolio_return`, `financial_net_return`, `portfolio_value`, `drawdown`,
`turnover`, `transaction_cost`, `max_weight`, `cash_weight`, `weight_SPY`,
`weight_TLT`, `weight_GLD`, `weight_BTC-USD`, and `weight_CASH`.

**Feature set comparison rerun with policy history:**  
The rerun output was written to
`outputs/tables/feature_set_comparison_V1_V2_V3_real_macro_2015_2024_policy_history`.
Ranking by robust Sharpe with 0.5 dispersion penalty:

- V3_real_macro: 0.3016
- V2: 0.1477
- V1: -0.1341

**Policy behavior by feature set:**  
Mean dominant weight:

- V1: 0.9434
- V2: 0.9456
- V3_real_macro: 0.9527

Mean effective number of assets:

- V1: 1.1377
- V2: 1.1342
- V3_real_macro: 1.1116

High concentration 90% rate:

- V1: 0.8179
- V2: 0.8359
- V3_real_macro: 0.8487

**Dominant asset distribution:**  
V1 total counts: BTC-USD = 118, SPY = 98, GLD = 67, CASH = 64, and TLT = 43.
V2 total counts: BTC-USD = 113, CASH = 92, TLT = 69, GLD = 64, and SPY = 52.
V3_real_macro total counts: BTC-USD = 155, GLD = 98, CASH = 78, SPY = 43, and
TLT = 16.

**Interpretation:**  
The learned policy is structurally concentrated. It behaves more like
dominant-asset selection than diversified allocation. However, the dominant
asset changes across observations, so it is not simply static all-in BTC.
V3_real_macro ranks best in robust Sharpe in the policy-history rerun, but it
is also the most concentrated of the three feature sets.

**Methodological caution:**  
Concentration should still not be judged as automatically wrong. The next
question is whether dominant-asset rotation is associated with meaningful
financial regimes or whether it is unstable/noisy.

**Next decision:**  
Do not change the reward function or add max-weight constraints yet. The next
step is regime attribution by merging or exporting evaluation features with
policy history.

## Entry X — Regime Attribution for Dominant-Asset Policy Behavior

**Date:** 2026-05-13

**Purpose:**  
Evaluate whether the learned concentrated policies rotate dominant assets in
ways associated with financial regimes.

**Implementation:**  
Added `src/analysis/regime_attribution.py` and
`tests/test_regime_attribution.py`. The module merges `policy_history` with
raw, unnormalized features and computes regime attribution by dominant asset.

**Tests:**  
`python3 -m unittest tests/test_regime_attribution.py` ran 17 tests OK.
`python3 -m unittest discover tests` ran 495 tests OK.

**Source experiment:**  
`outputs/tables/feature_set_comparison_V1_V2_V3_real_macro_2015_2024_policy_history`.

**Aggregated outputs:**  
- `outputs/tables/regime_attribution_by_feature_set_real_macro_2015_2024/regime_attribution_summary_by_feature_set_asset.csv`
- `outputs/tables/regime_attribution_by_feature_set_real_macro_2015_2024/dominant_asset_summary_by_feature_set_asset.csv`

**Dominant asset distribution:**  
V2 total counts: BTC-USD = 113, CASH = 92, TLT = 69, GLD = 64, and SPY = 52.
V3_real_macro total counts: BTC-USD = 155, GLD = 98, CASH = 78, SPY = 43, and
TLT = 16.

**Key V3 regime attribution:**  
BTC-USD dominance had 155 observations, with `market_trend_regime = 0.9354`,
`market_defensive_regime = 0.7079`, `macro_high_vix_regime = 0.5747`,
`macro_inverted_yield_curve_regime = 0.7432`, and
`macro_strong_dollar_regime = 0.4649`.

GLD dominance had 98 observations, with `market_high_vol_regime = 0.5612`,
`market_risk_off_regime = 0.3715`, `macro_high_vix_regime = 0.2948`, and
`macro_strong_dollar_regime = 0.8257`.

CASH dominance had 78 observations, with `market_high_vol_regime = 0.5911`,
`macro_high_vix_regime = 0.4808`, and `macro_strong_dollar_regime = 0.6953`.

SPY dominance had 43 observations, with `market_trend_regime = 0.6667`,
`market_risk_off_regime = 0.5944`, `macro_high_vix_regime = 0.8183`,
`macro_inverted_yield_curve_regime = 1.0`, and
`macro_strong_dollar_regime = 1.0`.

TLT dominance had 16 observations, with `market_trend_regime = 1.0`,
`macro_high_vix_regime = 0.2083`, and `macro_yield_curve_10y_2y = -0.3608`.

**Interpretation:**  
The model behaves like a dominant-asset switching policy rather than a
diversified allocation policy. Some regime associations appear economically
plausible, especially BTC-USD dominance during positive trend regimes and CASH
dominance during more volatile or strong-dollar environments. However, the
macro attribution is not clean enough to claim that the model learned a robust
macroeconomic allocation rule. Some variables, such as inflation pressure, do
not discriminate because they are effectively constant across the tested
period.

**Methodological caution:**  
Regime attribution is descriptive, not causal. It does not prove that the agent
uses these variables correctly; it only shows associations between dominant
allocation choices and observed state variables.

**Next decision:**  
Before changing the reward function or imposing max-weight constraints, compare
TD3 against simple dynamic allocation benchmarks such as 12-week momentum
winner, risk-adjusted momentum, trend-following, and risk-off defensive rules.

## Entry X — Dynamic Allocation Benchmarks and TD3 Hurdle Comparison

**Date:** 2026-05-13

**Purpose:**  
Since TD3 learned highly concentrated dominant-asset policies, compare it
against simple dynamic allocation rules before changing the reward function or
adding portfolio constraints.

**Implementation:**  
Added `scripts/download_market_data.py` to create reproducible local weekly
returns at `data/processed/returns_weekly_2015_2024.csv`. Added
`src/backtest/dynamic_allocation_benchmarks.py` for diagnostic dynamic
allocation rules. Added tests in `tests/test_download_market_data.py` and
`tests/test_dynamic_allocation_benchmarks.py`.

**Local market dataset:**  
The local processed return dataset had shape (521, 5), start date 2015-01-09,
end date 2024-12-27, columns SPY, TLT, GLD, BTC-USD, and CASH, and 0 missing
values.

**Benchmarks:**  
- `momentum_winner_12p`
- `risk_adjusted_momentum_winner_12p_12p`
- `trend_spy_cash_12p`
- `defensive_risk_off_12p`

**Safe ratio flags:**  
Added `sortino_ratio_is_finite`, `sortino_ratio_is_extreme`,
`calmar_ratio_is_finite`, `calmar_ratio_is_infinite`, and
`max_drawdown_is_zero`. These flags are designed to prevent over-interpretation
of extreme Sortino or infinite Calmar values when downside deviation or maximum
drawdown is near zero. The raw ratios are still preserved.

**Tests:**  
`python3 -m unittest tests/test_download_market_data.py` ran 11 tests OK.
`python3 -m unittest tests/test_dynamic_allocation_benchmarks.py` ran 28 tests
OK. `python3 -m unittest discover tests` ran 534 tests OK.

**Dynamic benchmark aggregate:**  
`defensive_risk_off_12p`: mean cumulative return = 0.074091, mean Sharpe =
1.336054, mean Sortino = 2.551050, mean max drawdown = -0.073294, worst max
drawdown = -0.118148, and mean average turnover = 0.192308.

`momentum_winner_12p`: mean cumulative return = 0.291238, mean Sharpe =
1.375670, mean Sortino = 3.858278, mean max drawdown = -0.155668, worst max
drawdown = -0.192170, and mean average turnover = 0.500000.

`risk_adjusted_momentum_winner_12p_12p`: mean cumulative return = 0.065992,
mean Sharpe = 0.607083, mean Sortino = 1.067393, mean max drawdown =
-0.085640, worst max drawdown = -0.129004, and mean average turnover =
0.397436.

`trend_spy_cash_12p`: mean cumulative return = 0.084668, mean Sharpe =
1.609587, mean Sortino = 2.902973, mean max drawdown = -0.052766, worst max
drawdown = -0.056565, and mean average turnover = 0.141026.

**TD3 vs dynamic benchmark table:**  
`momentum_winner_12p` (dynamic benchmark): mean cumulative return = 0.291238,
mean Sharpe = 1.375670, robust Sharpe 0.5 = 1.122988, mean Sortino = 3.858278,
mean max drawdown = -0.155668, worst max drawdown = -0.192170, and mean
average turnover = 0.500000.

`trend_spy_cash_12p` (dynamic benchmark): mean cumulative return = 0.084668,
mean Sharpe = 1.609587, robust Sharpe 0.5 = 1.015885, mean Sortino = 2.902973,
mean max drawdown = -0.052766, worst max drawdown = -0.056565, and mean
average turnover = 0.141026.

`defensive_risk_off_12p` (dynamic benchmark): mean cumulative return =
0.074091, mean Sharpe = 1.336054, robust Sharpe 0.5 = 0.533350, mean Sortino =
2.551050, mean max drawdown = -0.073294, worst max drawdown = -0.118148, and
mean average turnover = 0.192308.

`risk_adjusted_momentum_winner_12p_12p` (dynamic benchmark): mean cumulative
return = 0.065992, mean Sharpe = 0.607083, robust Sharpe 0.5 = 0.382539, mean
Sortino = 1.067393, mean max drawdown = -0.085640, worst max drawdown =
-0.129004, and mean average turnover = 0.397436.

`V3_real_macro` (TD3): mean cumulative return = 0.081678, mean Sharpe =
0.563611, robust Sharpe 0.5 = 0.301614, mean Sortino = 1.018962, mean max
drawdown = -0.134771, worst max drawdown = -0.208422, mean average turnover =
0.421601, and mean effective number of assets = 1.111574.

`V2` (TD3): mean cumulative return = 0.094096, mean Sharpe = 0.552743, robust
Sharpe 0.5 = 0.147718, mean Sortino = 1.442779, mean max drawdown = -0.101749,
worst max drawdown = -0.158292, mean average turnover = 0.651826, and mean
effective number of assets = 1.134161.

`V1` (TD3): mean cumulative return = 0.042415, mean Sharpe = 0.387797, robust
Sharpe 0.5 = -0.134123, mean Sortino = 1.099904, mean max drawdown =
-0.127345, worst max drawdown = -0.251822, mean average turnover = 0.718174,
and mean effective number of assets = 1.137702.

**Interpretation:**  
The simple dynamic rules currently outperform TD3 on risk-adjusted metrics.
`momentum_winner_12p` dominates return and robust Sharpe.
`trend_spy_cash_12p` dominates Sharpe and drawdown control. TD3 remains useful
as a research object, but it has not yet demonstrated incremental value over
simple dynamic allocation rules.

**Methodological caution:**  
The comparison is still based on three walk-forward test folds and should not
be overgeneralized. However, these dynamic benchmarks are now the minimum
hurdle for future TD3 and reward-design improvements.

**Next decision:**  
Do not add complexity blindly. The next research step should be reward or
mandate redesign only if it can improve against these dynamic benchmark
hurdles.

## Entry X — Mandate-Aware Reward Infrastructure and Diagnostics

**Date:** 2026-05-19

**Purpose:**  
Prepare mandate-aware reward experiments without treating concentration as an
automatic failure. The goal is to separate useful concentrated allocation from
reward gaming, passive cash behavior, and mandate-incompatible exposure.

**Fresh-market runner fix:**  
The fresh-market workflow now updates market data first, writes a local returns
snapshot, and then trains from a generated config that points to that snapshot.
The generated config uses the actual snapshot end date instead of silently
clipping back to the base config's stale historical `data.end_date`.

The live smoke requested data through `requested_end_date = 2026-05-19`. The
refreshed local returns snapshot ended at `market_data_end = 2026-05-15`, had
shape [593, 5], and had 0 missing values.
`snapshot_end_used_in_generated_config = 2026-05-15`.

**Mandate-aware reward infrastructure:**  
Added mandate profiles and pure mandate penalty components, then integrated
the mandate penalty into the active reward as an opt-in additive term:

```text
reward_new = reward_old - lambda_mandate * mandate_penalty
```

Default behavior remains unchanged unless `reward.use_mandate_penalty = true`.
This is infrastructure for controlled experiments, not a conclusion that the
mandate reward is calibrated.

**Mandate smoke:**  
Using `data/processed/returns_weekly_latest.csv`, `training.episodes = 3`, and
a moderate mandate penalty configuration, the first saved smoke comparison
showed:

- `baseline_no_mandate`: test Sharpe = 1.5066, test cumulative return =
  0.7694, max drawdown = -0.1688, average turnover = 0.3551, and average
  effective number of assets = 1.0663.
- `mandate_moderate`: test Sharpe = -0.4136, test cumulative return =
  -0.0009, max drawdown = -0.0009, average turnover = 0.0200, and average
  effective number of assets = 1.0037.

This first smoke suggested that the mandate penalty could reduce turnover and
drawdown but might also create a passive defensive allocation. A later rerun
concentrated mostly in GLD and improved concentration-quality metrics versus
equal weight across horizons. Therefore these smoke runs are behavior evidence,
not performance evidence.

**Mandate penalty component means:**  
For the rerun `mandate_moderate` test policy history, the mean penalty
components were:

- `mandate_penalty = 0.402616`
- `drawdown_breach = 0.000000`
- `volatility_breach = 0.006086`
- `max_weight_breach = 0.196417`
- `effective_assets_breach = 0.242750`
- `turnover_breach = 0.009509`

The penalty was driven mainly by max-weight concentration and low effective
number of assets, not drawdown.

**Concentration quality diagnostics:**  
The concentration quality diagnostic evaluates whether the dominant asset was
subsequently rewarded by realized returns.

`baseline_no_mandate`:

- Horizon 1: best-rate = 0.1954, beats equal weight = 0.5287, excess vs equal
  weight = 0.0008.
- Horizon 4: best-rate = 0.1667, beats equal weight = 0.5357, excess vs equal
  weight = 0.0031.
- Horizon 12: best-rate = 0.0395, beats equal weight = 0.5132, excess vs equal
  weight = -0.0065.

`mandate_moderate`:

- Horizon 1: best-rate = 0.3563, beats equal weight = 0.6207, excess vs equal
  weight = 0.0041.
- Horizon 4: best-rate = 0.4405, beats equal weight = 0.6548, excess vs equal
  weight = 0.0169.
- Horizon 12: best-rate = 0.5789, beats equal weight = 0.7105, excess vs equal
  weight = 0.0677.

In this rerun, concentration in GLD looked more justified ex post than the
baseline's dominant choices. This does not prove robust performance, but it
does show why concentration should not be rejected mechanically.

**Cash allocation diagnostics:**  
Added a cash allocation diagnostic to distinguish valid defensive cash from a
cash trap. With `normal_cash_max = 0.10`, the latest smoke produced:

- Baseline max cash = 0.000085.
- Mandate max cash = 0.027275.
- Cash above normal rate = 0.0 for both strategies.

The current smoke does not show a cash trap, so no cash cap should be added
yet. If future policies hide in CASH, the diagnostic can test whether that
cash exposure occurred during risk-off states and what the forward opportunity
cost was.

**Tests:**  
After the diagnostics and reward-debug export updates,
`python3 -m unittest discover tests` ran 658 tests OK.

**Interpretation:**  
Do not treat concentration as automatically bad. The latest evidence says the
cash trap is not present in the saved smoke, and GLD concentration may be
justified in this short run. But this is still a small smoke experiment, not
robust performance evidence.

**Next decision:**  
Run a controlled mandate-reward mini-grid before making more penalty changes.
The next experiment should vary mandate penalty weights deliberately and
compare not only Sharpe and drawdown, but also mandate breaches, concentration
quality, cash allocation quality, and dynamic benchmark hurdles.

## Entry X — Multi-Seed Mandate Reward Mini-Grid

**Date:** 2026-05-19

**Purpose:**  
Test whether the first promising mandate penalty region survives a slightly
more controlled seed sweep before changing reward defaults or adding new
constraints.

**Experiment:**  
Used `data/processed/returns_weekly_latest.csv`, `episodes = 20`, and seeds
`[7, 42, 101]`. The grid compared `baseline_no_mandate` against balanced
moderate mandate penalties with `lambda_mandate` values of 0.001, 0.003,
0.005, and 0.0075.

**Result:**  
`baseline_no_mandate`: robust Sharpe 0.5 = 0.1779, mean Sharpe = 0.4736, mean
return = 0.1409, mean drawdown = -0.2423, turnover = 0.3548, and effective
assets = 1.0618.

`moderate_balanced_lambda_0075`: robust Sharpe 0.5 = 0.1280, mean Sharpe =
0.4748, mean return = 0.1325, mean drawdown = -0.2560, turnover = 0.6135, and
effective assets = 1.2568.

`moderate_balanced_lambda_005`: robust Sharpe 0.5 = 0.0899.

`moderate_balanced_lambda_003`: robust Sharpe 0.5 = -0.0169.

`moderate_balanced_lambda_001`: robust Sharpe 0.5 = -0.1980.

**Interpretation:**  
The mandate reward changes behavior, but it does not yet improve robust
performance. The baseline still wins on robust Sharpe. `lambda_0075` is the
closest mandate-aware setting, but it remains below the baseline. `lambda_001`
showed the clearest cash-trap behavior. Higher lambda values improved
effective assets, but did not clearly improve dominant-asset forward quality.

**Decision:**  
The mandate-aware reward should remain experimental, not default. The next
research step should not be random penalty tuning. It should be a clearer
hypothesis about when concentration is justified and when mandate penalties
should relax or activate.

## Entry X — V2 vs V4 GARCH-Style Feature Comparison

**Date:** 2026-05-19

**Purpose:**  
Test whether adding deterministic GARCH-style volatility features improves TD3
behavior relative to the current V2 reference feature set.

**Feature definition:**  
V4 is defined as V2 plus deterministic GARCH-style volatility features. The
filter is not a full estimated GARCH model. It uses fixed parameters and
lagged returns to produce conditional-volatility-style state variables.

**Experiment:**  
Used `data/processed/returns_weekly_latest.csv`, `episodes = 20`, seeds
`[7, 42, 101]`, baseline reward with mandate penalty disabled, and compared
feature versions V2 and V4.

**Aggregate result:**  
`V2`: mean test Sharpe = 0.7625, standard deviation = 0.1256, robust Sharpe
0.5 = 0.6998, mean cumulative return = 0.2735, mean max drawdown = -0.2595,
worst max drawdown = -0.4872, mean turnover = 0.5064, mean max weight =
0.9332, mean effective assets = 1.1574, and mean final cash weight = 0.2903.

`V4`: mean test Sharpe = 0.0278, standard deviation = 0.1291, robust Sharpe
0.5 = -0.0367, mean cumulative return = -0.0616, mean max drawdown = -0.4594,
worst max drawdown = -0.4612, mean turnover = 0.3534, mean max weight =
0.9657, mean effective assets = 1.0911, and mean final cash weight was
approximately 0.0000.

**Concentration quality:**  
V2 had better long-horizon concentration quality than V4. At horizon 12, V2
had dominant-best-asset rate = 0.1316, beats-equal-weight rate = 0.3596, and
mean excess versus equal weight = -0.0450. V4 had dominant-best-asset rate =
0.1974, beats-equal-weight rate = 0.3246, and mean excess versus equal weight
= -0.0574. V4 was more concentrated and did not improve forward selection
quality.

**Interpretation:**  
V4 did not improve TD3 behavior relative to V2. V2 remains the reference
feature set. The deterministic GARCH-style features should remain
experimental.

**Decision:**  
Do not implement estimated real GARCH yet unless a clearer hypothesis is
formed. The next step should be either deeper V2 analysis or a more targeted
volatility-regime hypothesis, not adding GARCH complexity blindly.

## Entry X — V2 vs V5 Regime and Correlation Feature Comparison

**Date:** 2026-05-19

**Purpose:**  
Test whether regime and dynamic-correlation state variables improve TD3
behavior relative to V2 after the V4 GARCH-style feature set failed to improve
the model.

**Feature definition:**  
V5 is defined as V2 plus return-derived regime and correlation features.

**Experiment:**  
Used `data/processed/returns_weekly_latest.csv`, `episodes = 20`, seeds
`[7, 42, 101]`, baseline reward with mandate penalty disabled, and compared
feature versions V2 and V5.

**Aggregate result:**  
`V5`: mean test Sharpe = 0.7461, standard deviation = 0.4423, robust Sharpe
0.5 = 0.5249, mean cumulative return = 0.2389, mean max drawdown = -0.2681,
worst max drawdown = -0.4126, mean turnover = 0.4780, mean max weight =
0.9499, mean effective assets = 1.1269, and mean final cash weight = 0.3333.

`V2`: mean test Sharpe = 0.3547, standard deviation = 0.1969, robust Sharpe
0.5 = 0.2562, mean cumulative return = 0.0988, mean max drawdown = -0.2553,
worst max drawdown = -0.3848, mean turnover = 0.5778, mean max weight =
0.9469, mean effective assets = 1.1249, and mean final cash weight = 0.0000.

**Concentration quality:**  
V5 improved long-horizon concentration quality relative to V2. At horizon 12,
V2 had beats-equal-weight rate = 0.3596, mean excess versus equal weight =
-0.0467, and mean dominant-asset rank = 3.4605. V5 had beats-equal-weight rate
= 0.4298, mean excess versus equal weight = -0.0218, and mean dominant-asset
rank = 3.2237.

**Cash diagnostics:**  
V5 strongly increased cash allocation. Its mean cash weight was roughly
0.54-0.56 and cash was above the 10% normal band in roughly 57%-59% of
observations. V2 mean cash was roughly 0.095-0.100, with cash above the 10%
band around 12% of observations.

**Interpretation:**  
V5 improves robust Sharpe versus V2 in this controlled 3-seed comparison and
also improves long-horizon concentration quality. However, V5 relies heavily
on CASH, so the result is promising but not clean.

**Decision:**  
The next step is not reward tuning. The next step is to test whether high CASH
exposure happens during V5 risk-off states. If high cash is mostly aligned
with `risk_off_state`, V5 may be learning useful regime defense. If high cash
occurs outside `risk_off_state`, V5 may be exploiting a cash trap. V5 is more
promising than V4 GARCH-style features, but it is not ready to become the
default.

## Entry X — V5 Cash Risk-Off Attribution

**Date:** 2026-05-19

**Purpose:**  
Test whether V5 high CASH exposure is aligned with the raw V5 `risk_off_state`.

**Output path:**  
`outputs/tables/feature_set_comparison_V2_V5_regime_latest_3seeds_20ep/diagnostics/v5_cash_risk_off_attribution`

**Tests:**  
`python3 -m unittest discover tests` ran 737 tests OK.

**Summary:**  
`V5_seed_101`: mean cash = 0.0604, high-cash rate = 0.0909, risk-off rate =
0.1023, and share of high-cash observations in risk-off = 0.0000.

`V5_seed_42`: mean cash = 0.6844, high-cash rate = 0.7159, risk-off rate =
0.1023, and share of high-cash observations in risk-off = 0.0794.

`V5_seed_7`: mean cash = 0.8569, high-cash rate = 0.8977, risk-off rate =
0.1023, and share of high-cash observations in risk-off = 0.1013.

**By-state interpretation:**  
For seed 42, mean cash outside risk-off was 0.6986. For seed 7, mean cash
outside risk-off was 0.8541. Therefore most high CASH exposure occurs outside
V5 `risk_off_state`.

**Interpretation:**  
V5 improved robust Sharpe versus V2, but the improvement is contaminated by
cash-trap behavior. V5 is promising but not clean. V2 remains the cleaner
reference feature set.

**Decision:**  
V5 should not become the default until CASH is constrained or penalized
conditionally. The next step should be a conditional CASH rule: normal cash
band around 0%-10%, with higher cash allowed only when risk-off state is
justified. Do not implement estimated real GARCH yet.

## Entry X — V5 Turnover Penalty Smoke

**Date:** 2026-05-19

**Purpose:**  
Test whether the current linear turnover penalty was limiting V5 policy
behavior. The concern was that the agent may be learning that moving is worse
than being wrong, especially because transaction costs already penalize
turnover.

**Experiment:**  
Used V5 features, `data/processed/returns_weekly_latest.csv`,
`episodes = 10`, seeds `[7, 42, 101]`, with mandate penalty disabled and cash
penalty disabled.

**Turnover modes tested:**  
`linear` current behavior, `none`, `excess_linear` with free band 0.10,
`excess_linear` with free band 0.20, and `excess_quadratic` with free band
0.10.

**Result:**  
`V5_turnover_free_band_020`: robust Sharpe = 1.1846, mean Sharpe = 1.3739,
average turnover = 0.3231, mean turnover penalty = 0.000124, and cash above
10% rate = 0.2511.

`V5_turnover_free_band_010`: robust Sharpe = 1.0920, mean Sharpe = 1.4257,
average turnover = 0.2653, mean turnover penalty = 0.000119, and cash above
10% rate = 0.2987.

`V5_turnover_none`: robust Sharpe = 0.7853, mean Sharpe = 0.9712, average
turnover = 0.4871, mean turnover penalty = 0.000000, and cash above 10% rate =
0.3593.

`V5_turnover_linear_current`: robust Sharpe = 0.5150, mean Sharpe = 1.0697,
average turnover = 0.4641, mean turnover penalty = 0.000232, and cash above
10% rate = 0.3247.

`V5_turnover_quadratic_010`: robust Sharpe = 0.3648, mean Sharpe = 0.4863,
average turnover = 0.4901, mean turnover penalty = 0.000541, and cash above
10% rate = 0.2165.

**Policy history export:**  
Turnover reward debug fields are now saved in `policy_history` when available:
`turnover_penalty`, `turnover_penalty_mode`, `turnover_free_band`, and
`turnover_excess`.

**Interpretation:**  
The linear turnover penalty appears too blunt. Removing the turnover penalty
entirely is not the answer; it worsened cash exposure. A free-band turnover
penalty, especially `free_band = 0.20`, looks more promising. The quadratic
0.10 version reduced cash exposure but hurt performance.

**Decision:**  
The current candidate for future experiments is `excess_linear` turnover with
`turnover_free_band = 0.20`. It should remain experimental, not default, until
tested across stronger folds and seeds.

## Entry X — V5 Dynamic CASH Penalty Weight Sweep

**Date:** 2026-05-20

**Purpose:**  
Test whether a dynamic conditional CASH penalty using raw V5 `risk_off_state`
can reduce unjustified high CASH exposure. Also test whether lower cash
penalty weights allow justified CASH allocation during risk-off states.

**Experiment:**  
Used V5 features, raw auxiliary regime features from
`build_v5_regime_auxiliary_features`, `data/processed/returns_weekly_latest.csv`,
`episodes = 10`, seeds `[7, 42, 101]`, `turnover_penalty_mode =
excess_linear`, and `turnover_free_band = 0.20`. Mandate penalty was disabled.
The cash penalty was dynamic through `reward.cash_risk_off_column =
risk_off_state`. Auxiliary regime features were used only for reward/business
rules and were not neural observations.

**Aggregate result:**  
`dynamic_cash_weight_025`: robust Sharpe = 1.8846, mean Sharpe = 2.1214,
mean return = 1.6936, mean drawdown = -0.1562, turnover = 0.2161, effective
assets = 1.0324, mean cash = 0.0000, and cash above 10% rate = 0.0000.

`dynamic_cash_weight_010`: robust Sharpe = 1.6186, mean Sharpe = 1.7135,
mean return = 1.6038, mean drawdown = -0.1846, turnover = 0.1449, effective
assets = 1.0138, mean cash = 0.0000, and cash above 10% rate = 0.0000.

`dynamic_cash_weight_0025`: robust Sharpe = 0.9734, mean Sharpe = 1.0944,
mean return = 0.2969, mean drawdown = -0.1000, turnover = 0.4652, effective
assets = 1.2340, mean cash = 0.0085, and cash above 10% rate = 0.0173.

`dynamic_cash_weight_005`: robust Sharpe = 0.6558, mean Sharpe = 0.9153,
mean return = 0.2763, mean drawdown = -0.1676, turnover = 0.5189, effective
assets = 1.2769, mean cash = 0.1941, and cash above 10% rate = 0.2251.

`no_cash_penalty`: robust Sharpe = 0.9432, mean Sharpe = 1.1675, mean return
= 0.2667, mean drawdown = -0.1254, turnover = 0.4351, effective assets =
1.2500, mean cash = 0.1504, and cash above 10% rate = 0.2035.

**Cash attribution:**  
The V5 risk-off rate was 0.0649 across runs. `dynamic_cash_weight_025` and
`dynamic_cash_weight_010` eliminated high cash. Lower weights did not create
clearly justified high CASH exposure: share of high-cash observations in
risk-off was 0.0000 for weight 0.0025 and only 0.0321 for weight 0.005.

**Concentration quality, horizon 12:**  
`dynamic_cash_weight_010`: best-asset rate = 0.4978, beats-equal rate =
0.6190, excess versus equal weight = 0.1198, and mean rank = 2.3377.

`dynamic_cash_weight_025`: best-asset rate = 0.4199, beats-equal rate =
0.5758, excess versus equal weight = 0.0861, and mean rank = 2.3810.

`no_cash_penalty`: best-asset rate = 0.0823, beats-equal rate = 0.2944,
excess versus equal weight = -0.0302, and mean rank = 3.0736.

**Interpretation:**  
The dynamic CASH penalty successfully reduces unjustified high CASH exposure.
However, the current `risk_off_state` does not appear to generate justified
high CASH allocation; high CASH mostly disappears instead of concentrating in
risk-off periods. Weight 0.010 has the best horizon-12 concentration quality.
Weight 0.025 has the strongest robust Sharpe but remains highly concentrated.

**Decision:**  
Both 0.010 and 0.025 should be treated as experimental candidates, not
defaults. The next step is to evaluate them with more seeds and episodes, and
inspect which assets receive the displaced allocation.

## Entry X — V5 Dynamic CASH Walk-Forward Validation

**Date:** 2026-05-20

**Purpose:**  
Test whether the dynamic CASH penalty candidates survive across expanding
walk-forward folds, and resolve the disagreement between fixed validation and
fixed test results.

**Experiment:**  
Output folder:
`outputs/tables/v5_dynamic_cash_walk_forward_30ep_5seeds`. The run used V5
features, raw auxiliary V5 regime features, `episodes = 30`, seeds
`[7, 21, 42, 84, 101]`, `turnover_penalty_mode = excess_linear`,
`turnover_free_band = 0.20`, and mandate penalty disabled.

Candidates:

- `V5_tfb020_no_cash_penalty`
- `V5_tfb020_dynamic_cash_weight_010`
- `V5_tfb020_dynamic_cash_weight_025`

**Actual folds:**  
F1: train 2015-04-03 to 2020-12-25, validation 2021-01-01 to 2021-12-31,
test 2022-01-07 to 2022-12-30; n_train = 300, n_val = 53, n_test = 52.

F2: train 2015-04-03 to 2021-12-31, validation 2022-01-07 to 2022-12-30,
test 2023-01-06 to 2023-12-29; n_train = 353, n_val = 52, n_test = 52.

F3: train 2015-04-03 to 2022-12-30, validation 2023-01-06 to 2023-12-29,
test 2024-01-05 to 2024-12-27; n_train = 405, n_val = 52, n_test = 52.

F4: train 2015-04-03 to 2023-12-29, validation 2024-01-05 to 2024-12-27,
test 2025-01-03 to 2026-05-15; n_train = 457, n_val = 52, n_test = 72.

**Overall test aggregate:**  
`V5_tfb020_dynamic_cash_weight_025`: robust Sharpe = -0.2246, mean Sharpe =
0.4265, mean return = 0.0703, mean drawdown = -0.2649, worst drawdown =
-0.6329, turnover = 0.6007, effective assets = 1.1204, and cash above 10%
rate = 0.0000.

`V5_tfb020_no_cash_penalty`: robust Sharpe = -0.4139, mean Sharpe = 0.1583,
mean return = -0.0152, mean drawdown = -0.2640, worst drawdown = -0.6697,
turnover = 0.5912, effective assets = 1.1339, and cash above 10% rate =
0.1932.

`V5_tfb020_dynamic_cash_weight_010`: robust Sharpe = -0.5597, mean Sharpe =
-0.0206, mean return = -0.0623, mean drawdown = -0.2867, worst drawdown =
-0.6303, turnover = 0.5799, effective assets = 1.1219, and cash above 10%
rate = 0.1097.

**Fold-level winners:**  
F1: `dynamic_cash_weight_010`, robust Sharpe = -1.4782.

F2: `dynamic_cash_weight_025`, robust Sharpe = 1.4430.

F3: `dynamic_cash_weight_025`, robust Sharpe = 1.1929.

F4: `no_cash_penalty`, robust Sharpe = -0.0951.

**Cash attribution:**  
`dynamic_cash_weight_025` eliminated high cash and unjustified cash.
`dynamic_cash_weight_010` reduced high cash, but left unjustified cash excess
= 0.0828. `no_cash_penalty` had unjustified cash excess = 0.1316.

**Concentration quality, horizon 12:**  
`dynamic_cash_weight_025`: best-rate = 0.1897, beats equal weight = 0.4340,
excess versus equal weight = -0.0174, and mean rank = 3.0037.

`no_cash_penalty`: best-rate = 0.1607, beats equal weight = 0.3830, excess
versus equal weight = -0.0237, and mean rank = 3.2388.

`dynamic_cash_weight_010`: best-rate = 0.1556, beats equal weight = 0.3545,
excess versus equal weight = -0.0299, and mean rank = 3.2758.

**Win rates versus no cash:**  
`dynamic_cash_weight_025`: robust Sharpe win rate = 0.75, cash reduction rate
= 1.00, and H12 quality improvement rate = 0.50.

`dynamic_cash_weight_010`: robust Sharpe win rate = 0.25, cash reduction rate
= 1.00, and H12 quality improvement rate = 0.00.

**Interpretation:**  
Walk-forward evidence favors `dynamic_cash_weight_025` over 0.010. The 0.025
candidate beats no-cash in 3 of 4 test folds and fully removes unjustified
high CASH. The 0.010 candidate is not robust enough and should be dropped for
now.

However, no candidate is cleanly robust in absolute terms. Overall robust
Sharpe remains negative and F1 is weak across all candidates. The 0.025
candidate remains experimental, not a default.

**Decision:**  
Next step: compare 0.025 against V2/V5 baselines and traditional benchmarks
before claiming improvement.

## Entry X — V5 Dynamic CASH Candidate Versus V2 and Benchmarks

**Date:** 2026-05-20

**Purpose:**  
Compare the current experimental candidate, `V5_dynamic_cash_025`, against
`V2_reference`, `V5_no_cash_penalty`, and traditional benchmarks. The goal was
to determine whether the dynamic CASH penalty candidate is a real improvement
or only an internal V5 improvement.

**Experiment:**  
Output folder:
`outputs/tables/v2_v5_dynamic_cash_benchmark_comparison_30ep_5seeds`.

Setup: `episodes = 30`, seeds `[7, 21, 42, 84, 101]`, and the same expanding
walk-forward folds as the prior run.

`V5_dynamic_cash_025` used V5 features, `turnover_penalty_mode =
excess_linear`, `turnover_free_band = 0.20`, dynamic CASH penalty enabled,
`cash_penalty_weight = 0.025`, `cash_risk_off_column = risk_off_state`, and
auxiliary raw V5 regime features. `V5_no_cash_penalty` used the same V5 and
turnover setup without the CASH penalty. `V2_reference` used V2 features and
the original/current V2 reference reward. Benchmarks included `BuyHold_GLD`,
`BuyHold_SPY`, `Equal_Weight`, `60_40_SPY_TLT`, `BuyHold_BTC-USD`, and
`BuyHold_TLT`.

**Actual folds:**  
F1: train 2015-04-03 to 2020-12-25, validation 2021-01-01 to 2021-12-31,
test 2022-01-07 to 2022-12-30; counts 300 / 53 / 52.

F2: train 2015-04-03 to 2021-12-31, validation 2022-01-07 to 2022-12-30,
test 2023-01-06 to 2023-12-29; counts 353 / 52 / 52.

F3: train 2015-04-03 to 2022-12-30, validation 2023-01-06 to 2023-12-29,
test 2024-01-05 to 2024-12-27; counts 405 / 52 / 52.

F4: train 2015-04-03 to 2023-12-29, validation 2024-01-05 to 2024-12-27,
test 2025-01-03 to 2026-05-15; counts 457 / 52 / 72.

**Overall test aggregate:**  
`BuyHold_GLD`: robust Sharpe = 0.7178, mean Sharpe = 1.1618, mean return =
0.2776, mean drawdown = -0.1199, and worst drawdown = -0.1735.

`BuyHold_SPY`: robust Sharpe = 0.4094, mean Sharpe = 1.0359, mean return =
0.1521, mean drawdown = -0.1362, and worst drawdown = -0.2248.

`Equal_Weight`: robust Sharpe = -0.0259, mean Sharpe = 0.9224, mean return =
0.1370, mean drawdown = -0.1124, and worst drawdown = -0.2513.

`60_40_SPY_TLT`: robust Sharpe = -0.1566, mean Sharpe = 0.5183, mean return =
0.0550, mean drawdown = -0.1265, and worst drawdown = -0.2483.

`V5_dynamic_cash_025`: robust Sharpe = -0.2246, mean Sharpe = 0.4265, mean
return = 0.0703, mean drawdown = -0.2649, worst drawdown = -0.6329, turnover
= 0.6007, effective assets = 1.1204, and cash above 10% rate = 0.0000.

`V2_reference`: robust Sharpe = -0.2653, mean Sharpe = 0.3379, mean return =
0.0538, mean drawdown = -0.2391, worst drawdown = -0.5333, turnover = 0.5440,
effective assets = 1.1192, and cash above 10% rate = 0.2187.

`BuyHold_BTC-USD`: robust Sharpe = -0.3141, mean Sharpe = 0.5025, mean return
= 0.4927, mean drawdown = -0.3776, and worst drawdown = -0.6430.

`V5_no_cash_penalty`: robust Sharpe = -0.4139, mean Sharpe = 0.1583, mean
return = -0.0152, mean drawdown = -0.2640, worst drawdown = -0.6697, turnover
= 0.5912, effective assets = 1.1339, and cash above 10% rate = 0.1932.

**Validation:**  
Top validation robust Sharpe was `BuyHold_SPY = 0.6225`. Among DRL strategies,
`V5_dynamic_cash_025 = -0.1395`, `V2_reference = -0.3291`, and
`V5_no_cash_penalty = -0.4089`.

**Fold winners by test robust Sharpe:**  
F1: `BuyHold_GLD`, robust Sharpe = 0.0147.

F2: `Equal_Weight`, robust Sharpe = 2.1161.

F3: `Equal_Weight_Risky`, robust Sharpe = 2.2754.

F4: `BuyHold_GLD`, robust Sharpe = 1.9948.

**Win rates for `V5_dynamic_cash_025`:**  
Against `V2_reference`: robust Sharpe wins = 2/4, return wins = 2/4, and
drawdown wins = 2/4.

Against `V5_no_cash_penalty`: robust Sharpe wins = 3/4, return wins = 3/4,
and drawdown wins = 2/4.

Against `Equal_Weight`: robust Sharpe wins = 1/4, return wins = 2/4, and
drawdown wins = 0/4.

Against `60_40_SPY_TLT`: robust Sharpe wins = 2/4, return wins = 2/4, and
drawdown wins = 1/4.

Against `BuyHold_SPY`: robust Sharpe wins = 0/4, return wins = 2/4, and
drawdown wins = 0/4.

**Cash attribution:**  
`V5_dynamic_cash_025`: mean cash = 0.0001, high cash rate = 0.0000,
risk-off rate = 0.1851, high cash in risk-off = 0.0000, and unjustified cash
excess = 0.0000.

`V5_no_cash_penalty`: mean cash = 0.1635, high cash rate = 0.1932, risk-off
rate = 0.1851, high cash in risk-off = 0.0284, and unjustified cash excess =
0.1316.

**Concentration quality, horizon 12:**  
`V5_dynamic_cash_025`: best-asset rate = 0.1897, beats-equal rate = 0.4340,
excess versus equal weight = -0.0174, and mean rank = 3.0037.

`V5_no_cash_penalty`: best-asset rate = 0.1607, beats-equal rate = 0.3830,
excess versus equal weight = -0.0237, and mean rank = 3.2388.

`V2_reference`: best-asset rate = 0.1471, beats-equal rate = 0.3548, excess
versus equal weight = -0.0306, and mean rank = 3.3519.

**Interpretation:**  
`V5_dynamic_cash_025` is an internal improvement over `V5_no_cash_penalty`,
especially on cash discipline. It removes high CASH exposure and improves
horizon-12 concentration quality relative to both `V5_no_cash_penalty` and
`V2_reference`.

It is not clearly better than `V2_reference`: it only wins 2 of 4 folds
against V2 on robust Sharpe, return, and drawdown. It is also not competitive
against traditional benchmarks. GLD and SPY dominate robust Sharpe, while
Equal Weight wins key folds.

The DRL candidate improves mandate discipline, but it does not yet deliver
superior out-of-sample performance. Therefore, `V5_dynamic_cash_025` remains
an experimental candidate, not a final/default model.

**Next step:**  
Implement robust score / DSR as an evaluation layer, not as a training reward.
Then re-rank all candidates and benchmarks using a composite robustness
metric.

## Entry X — Robust Score and Deflated Sharpe Evaluation Layer

**Date:** 2026-05-20

**Purpose:**  
Move beyond ranking strategies only by Sharpe or robust Sharpe. The new
evaluation layer adds a composite score that balances statistical robustness,
downside risk, drawdown control, stability, and mandate discipline. Deflated
Sharpe Ratio is used to reduce the risk of selecting a strategy because it
looked good after multiple experiments.

**Implementation:**  
Added `src/analysis/robust_score.py` and `tests/test_robust_score.py`.

Core functions added:

- `compute_annualized_sharpe`
- `compute_probabilistic_sharpe_ratio`
- `compute_deflated_sharpe_ratio`
- `estimate_expected_max_sharpe`
- `normalize_metric_series`
- `compute_discipline_score`
- `compute_composite_robust_score`

DSR uses the Bailey / Lopez de Prado expected maximum Sharpe adjustment.
Default `n_trials_effective = 25`, with sensitivity reported for
`n_trials = 10, 25, 50`. This is an evaluation layer only, not a training
reward.

**Composite robust_score weights:**  
`DSR_score = 0.30`, `Sortino_score = 0.20`, `Calmar_score = 0.20`,
`Drawdown_score = 0.15`, `Stability_score = 0.10`, and
`Discipline_score = 0.05`.

**Important interpretation:**  
DSR is the main statistical robustness component, but it is not the whole
selection criterion. The score still reflects downside risk, drawdown,
stability, and mandate discipline. I deliberately did not overweight DSR above
0.30, to avoid turning the metric into a single-author bias or another
Sharpe-only proxy.

**Ranking output:**  
`outputs/tables/v2_v5_dynamic_cash_benchmark_comparison_30ep_5seeds/robust_score_ranking.csv`

1. `BuyHold_GLD`: robust score = 0.8013, `dsr_n10 = 0.8605`,
   `dsr_n25 = 0.7454`, and `dsr_n50 = 0.6483`.
2. `Equal_Weight_Risky`: robust score = 0.7117 and `dsr_n25 = 0.3805`.
3. `Equal_Weight`: robust score = 0.6939 and `dsr_n25 = 0.3805`.
4. `BuyHold_SPY`: robust score = 0.6255 and `dsr_n25 = 0.3653`.
5. `V2_reference`: robust score = 0.4957 and `dsr_n25 = 0.0674`.
6. `BuyHold_BTC-USD`: robust score = 0.4750 and `dsr_n25 = 0.1682`.
7. `V5_dynamic_cash_025`: robust score = 0.4544 and `dsr_n25 = 0.0439`.
8. `60_40_SPY_TLT`: robust score = 0.4477 and `dsr_n25 = 0.1181`.
9. `V5_no_cash_penalty`: robust score = 0.3439 and `dsr_n25 = 0.0047`.
10. `BuyHold_TLT`: robust score = 0.2263 and `dsr_n25 = 0.0008`.

**Interpretation:**  
The formal DSR adjustment lowers absolute DSR values, as expected. The main
conclusion does not change: traditional benchmarks still dominate,
`V2_reference` remains above `V5_dynamic_cash_025`, and
`V5_dynamic_cash_025` still improves over `V5_no_cash_penalty`.

`V5_dynamic_cash_025` improves mandate and cash discipline, but it does not
yet beat V2 or traditional benchmarks. The robust score does not artificially
rescue the DRL candidate, which is methodologically healthy.

**Warning note:**  
DSR uses the Bailey / Lopez de Prado expected maximum Sharpe adjustment with
`n_trials_effective = 25`. Sensitivity is computed for `n_trials = 10, 25, 50`.

## Entry X — TD3 Decision Attribution Versus Simple Rules

**Date:** 2026-05-20

**Purpose:**  
Diagnose why TD3 does not beat simple dynamic allocation rules. The goal was
to move beyond aggregate Sharpe and inspect whether the learned dominant-asset
choices add value ex post.

**Implementation:**  
Added `src/analysis/decision_attribution.py` and
`tests/test_decision_attribution.py`. This is a pure analysis layer. It does
not change reward, TD3 architecture, environment dynamics, training, or
evaluation mechanics.

**Output:**  
`outputs/tables/v2_v5_dynamic_cash_benchmark_comparison_30ep_5seeds/decision_attribution`

Files written:

- `dominant_asset_regret_summary.csv`
- `td3_vs_rule_choice_summary.csv`
- `dominant_asset_hit_rate_by_horizon.csv`
- `regret_by_regime.csv`
- `decision_attribution_warnings.txt`

**Tests:**  
`python3 -m unittest tests/test_decision_attribution.py` ran 9 tests OK.
`python3 -m unittest discover tests` ran 802 tests OK.

**Regret summary:**  
`V2_reference`:

- Horizon 1: mean regret = 0.0385, best-hit rate = 0.1982, beats equal =
  0.4917, and excess versus equal = -0.0017.
- Horizon 4: mean regret = 0.0860, best-hit rate = 0.1932, beats equal =
  0.4536, and excess versus equal = -0.0072.
- Horizon 12: mean regret = 0.1978, best-hit rate = 0.1471, beats equal =
  0.3548, and excess versus equal = -0.0306.

`V5_dynamic_cash_025`:

- Horizon 1: mean regret = 0.0392, best-hit rate = 0.2170, beats equal =
  0.5053, and excess versus equal = -0.0023.
- Horizon 4: mean regret = 0.0853, best-hit rate = 0.1916, beats equal =
  0.4938, and excess versus equal = -0.0064.
- Horizon 12: mean regret = 0.1846, best-hit rate = 0.1897, beats equal =
  0.4340, and excess versus equal = -0.0174.

`V5_no_cash_penalty`:

- Horizon 1: mean regret = 0.0379, best-hit rate = 0.2097, beats equal =
  0.5077, and excess versus equal = -0.0010.
- Horizon 4: mean regret = 0.0856, best-hit rate = 0.2003, beats equal =
  0.4638, and excess versus equal = -0.0068.
- Horizon 12: mean regret = 0.1909, best-hit rate = 0.1607, beats equal =
  0.3830, and excess versus equal = -0.0237.

**TD3 versus simple rules, horizon 12:**  
For `V5_dynamic_cash_025`:

- Versus `defensive_risk_off_12p`: overlap = 0.2351, TD3 minus rule =
  -0.0143, and win rate = 0.3253.
- Versus `momentum_winner_12p`: overlap = 0.1921, TD3 minus rule = -0.0381,
  and win rate = 0.3469.
- Versus `risk_adjusted_momentum_winner_12p_12p`: overlap = 0.2267, TD3 minus
  rule = -0.0355, and win rate = 0.3267.
- Versus `trend_spy_cash_12p`: overlap = 0.1954, TD3 minus rule = -0.0118,
  and win rate = 0.3639.

**Regime error findings, horizon 12:**  
For `V5_dynamic_cash_025`:

- Outside `risk_off_state`: mean regret = 0.1851, best-hit rate = 0.1911,
  beats equal = 0.4425, and excess versus equal = -0.0153.
- Inside `risk_off_state`: mean regret = 0.1796, best-hit rate = 0.1838,
  beats equal = 0.4000, and excess versus equal = -0.0305.

**Interpretation:**  
All TD3 variants have low future-best-asset hit rates, especially at horizon
12. `V5_dynamic_cash_025` improves long-horizon decision quality versus
`V2_reference` and `V5_no_cash_penalty`, but it still loses to simple dynamic
rules on forward-return comparison.

Overlap with simple rules is low, so the agent is not merely copying those
rules. But its different decisions are not paying enough. The weakest
comparison is against momentum-style rules at horizon 12. The remaining
problem is less about CASH discipline and more about dominant-asset timing and
selection quality. Risk-off grouping does not show a clean story that V5 fixes
risk-off decisions.

**Research implication:**  
Do not keep tuning CASH, turnover, or mandate penalties blindly. The next
modeling work should target dominant-asset timing and state representation.
Future experiments should test whether the agent can learn or outperform
simple momentum and trend signals, rather than just adding more penalties.

## Entry X — Feature-Block Ablation for Dominant-Asset Timing

**Date:** 2026-05-20

**Purpose:**  
Diagnose which V5 feature blocks help or hurt dominant-asset timing. The
experiment tested whether the full V5 state representation is useful, or
whether simpler momentum/trend representations work better. This continues the
investigation after decision attribution showed that TD3 loses mainly on
dominant-asset timing versus simple dynamic rules.

**Implementation:**  
Added `src/experiments/run_feature_block_ablation.py` and
`tests/test_run_feature_block_ablation.py`. This is an experiment and analysis
runner only. It does not change reward, TD3 architecture, environment
dynamics, or training logic.

**Output:**  
`outputs/tables/v5_feature_block_ablation_timing_30ep_5seeds`

**Tests:**  
`python3 -m unittest discover tests` ran 808 tests OK.

**Feature variants:**  
- `V2_reference_full`: 49 features.
- `V5_full_dynamic_cash_025`: 67 features.
- `V5_no_momentum_block`: 52 features.
- `V5_no_volatility_block`: 48 features.
- `V5_no_drawdown_block`: 60 features.
- `V5_no_correlation_block`: 49 features.
- `V5_no_regime_block`: 54 features.
- `V5_momentum_only_or_minimal_momentum_regime`: 20 features.

**Overall test aggregate:**  
`V5_no_volatility_block`: mean Sharpe = 0.6632, robust Sharpe = 0.0621,
return = 0.1245, max drawdown = -0.2054, turnover = 0.5121, and effective
assets = 1.0918.

`V5_no_regime_block`: mean Sharpe = 0.3289, robust Sharpe = -0.2208, return =
-0.0119, max drawdown = -0.2612, turnover = 0.4869, and effective assets =
1.0943.

`V2_reference_full`: mean Sharpe = 0.2962, robust Sharpe = -0.2376, return =
0.0290, max drawdown = -0.2401, turnover = 0.5468, and effective assets =
1.1145.

`V5_full_dynamic_cash_025`: mean Sharpe = -0.0302, robust Sharpe = -0.4637,
return = -0.0521, max drawdown = -0.2924, turnover = 0.5106, and effective
assets = 1.0890.

**Robust score ranking:**  
1. `V5_no_volatility_block`: robust score = 0.6484 and `dsr_n25 = 0.2255`.
2. `V5_no_momentum_block`: robust score = 0.4951 and `dsr_n25 = 0.0580`.
3. `V5_momentum_only_or_minimal_momentum_regime`: robust score = 0.3970 and
   `dsr_n25 = 0.3626`.
4. `V2_reference_full`: robust score = 0.3866 and `dsr_n25 = 0.0485`.
8. `V5_full_dynamic_cash_025`: robust score = 0.1669 and `dsr_n25 = 0.0034`.

**Decision attribution, horizon 12:**  
`V5_momentum_only_or_minimal_momentum_regime`: mean regret = 0.1392,
best-hit rate = 0.2910, beats equal = 0.5165, and excess versus equal =
0.0280.

`V5_no_volatility_block`: mean regret = 0.1686, best-hit rate = 0.1818,
beats equal = 0.4920, and excess versus equal = -0.0014.

`V2_reference_full`: mean regret = 0.1952, best-hit rate = 0.1338, beats equal
= 0.3501, and excess versus equal = -0.0280.

`V5_full_dynamic_cash_025`: mean regret = 0.1970, best-hit rate = 0.1702,
beats equal = 0.3780, and excess versus equal = -0.0298.

**Versus simple rules, horizon 12:**  
`V5_momentum_only_or_minimal_momentum_regime`: TD3 minus
`momentum_winner_12p` = 0.0074, win rate versus momentum = 0.3149, TD3 minus
`trend_spy_cash_12p` = 0.0336, and win rate versus trend = 0.4535.

`V5_no_volatility_block`: TD3 minus `momentum_winner_12p` = -0.0220, win rate
versus momentum = 0.3867, TD3 minus `trend_spy_cash_12p` = 0.0042, and win
rate versus trend = 0.3363.

`V2_reference_full`: TD3 minus `momentum_winner_12p` = -0.0486, win rate
versus momentum = 0.3162, TD3 minus `trend_spy_cash_12p` = -0.0224, and win
rate versus trend = 0.3085.

`V5_full_dynamic_cash_025`: TD3 minus `momentum_winner_12p` = -0.0504, win
rate versus momentum = 0.2988, TD3 minus `trend_spy_cash_12p` = -0.0242, and
win rate versus trend = 0.3202.

**Cash and concentration:**  
Dynamic CASH generally worked across V5 ablations. `V2_reference_full` still
had high cash exposure, with cash above 10% = 0.1828 and unjustified cash
excess = 0.1184. The best horizon-12 dominant-asset timing came from
`V5_momentum_only_or_minimal_momentum_regime`.

**Interpretation:**  
The full V5 state representation appears overloaded. The most useful timing
signal is momentum/trend, not the full V5 block.
`V5_momentum_only_or_minimal_momentum_regime` gives the clearest
dominant-asset timing improvement. `V5_no_volatility_block` gives the best
composite robust score and aggregate risk/return profile.

Several ablations beat both `V2_reference_full` and
`V5_full_dynamic_cash_025`. Full V5 dynamic CASH is no longer the strongest
candidate. The next controlled validation should compare
`V5_no_volatility_block` and
`V5_momentum_only_or_minimal_momentum_regime` against V2, full V5, and
benchmarks.

**Research implication:**  
Do not add more CASH, turnover, or mandate penalty tuning now. Do not move yet
to GARCH, cointegration, LSTM, or imitation learning. The next modeling work
should target parsimonious momentum/trend state design and dominant-asset
timing quality.

## Entry X — V6 Financial State Features and First Validation

**Date:** 2026-05-20

**Purpose:**  
Validate the new V6 financial state representation before committing it as an
experimental candidate. The objective was to test whether a more parsimonious
financial state, built from momentum/trend, regime probabilities, volatility
proxies, and defensive attractiveness, improves over V2 and recent V5
ablations without adding more reward tuning.

**Implementation:**  
Added `src/data/features_v6.py`, updated feature factory/config support for
`features.version = v6`, and added `tests/test_features_v6.py`.

V6 is opt-in and does not change V1-V5 behavior. No reward, TD3 architecture,
environment dynamics, or training logic were changed.

**V6 feature structure:**  
- Momentum/trend: 24 features.
- Cross-sectional momentum/ranks/winners: 16 features.
- Risk regime probabilities: 5 features.
- Volatility proxies: 16 features.
- Defensive attractiveness: 7 features.
- Total: 68 features.

**Smoke checks:**  
Using `data/processed/returns_weekly_latest.csv`, the V6 prepared dataset
produced:

- `train_features_shape = (371, 68)`
- `validation_features_shape = (79, 68)`
- `test_features_shape = (80, 68)`
- `missing_value_count = 0`
- `first_aligned_date = 2016-03-25`
- `last_aligned_date = 2026-05-15`

Additional checks:

- No CASH momentum, trend, winner, or volatility proxy columns were created.
- Probability features stayed within `[0, 1]`.
- No duplicated columns or obvious redundant naming were detected.

**Validation output:**  
`outputs/tables/v6_financial_state_validation_30ep_5seeds`

**Setup:**  
- `returns_path = data/processed/returns_weekly_latest.csv`
- Episodes: 30
- Seeds: `[7, 21, 42, 84, 101]`
- Expanding walk-forward folds.
- V6 train starts later than V2/V5 because of rolling/z-score warmup:
  - F1: 249 / 53 / 52
  - F2: 302 / 52 / 52
  - F3: 354 / 52 / 52
  - F4: 406 / 52 / 72
- V6 used dynamic cash permission through raw shifted `cash_permission_score`
  as auxiliary `cash_risk_off_column`.
- V5 references used `risk_off_state`.
- V2 used no dynamic cash penalty.

**Overall test results:**  

`BuyHold_GLD`:
- Robust Sharpe = 0.7111
- Mean Sharpe = 1.1556
- Return = 0.2764
- Max drawdown = -0.1199

`BuyHold_SPY`:
- Robust Sharpe = 0.4040
- Mean Sharpe = 1.0294
- Return = 0.1509
- Max drawdown = -0.1362

`trend_spy_cash_12p`:
- Robust Sharpe = 0.1132
- Mean Sharpe = 0.9152
- Return = 0.0905
- Max drawdown = -0.0641

`V6_financial_state`:
- Robust Sharpe = 0.0248
- Mean Sharpe = 0.5312
- Return = 0.1878
- Max drawdown = -0.2401

`V5_no_volatility_block`:
- Robust Sharpe = -0.1155
- Mean Sharpe = 0.4480
- Return = 0.0935
- Max drawdown = -0.1836

`V2_reference_full`:
- Robust Sharpe = -0.1296
- Mean Sharpe = 0.3836
- Return = 0.0994
- Max drawdown = -0.2306

`V5_momentum_only`:
- Robust Sharpe = -0.2761
- Mean Sharpe = 0.3125
- Return = 0.1628
- Max drawdown = -0.3024

**Robust score ranking after conservative DSR aggregation:**  

The first robust-score report used pooled DSR across folds and seeds. A later
audit showed that this overstated the statistical evidence because overlapping
fold/seed returns were treated as too many independent observations.

For V6:

- `pooled_dsr_n25 = 0.6254`
- `median_run_dsr_n25 = 0.0768`
- `date_averaged_dsr_n25 = 0.2385`
- `dsr_method = median_run`

Pooled DSR remains reported for transparency, but composite `robust_score` now
uses median run-level DSR when available.

Corrected robust-score ranking:

`Equal_Weight`:
- `robust_score = 0.7019`

`Equal_Weight_Risky`:
- `robust_score = 0.6974`

`BuyHold_GLD`:
- `robust_score = 0.6315`

`trend_spy_cash_12p`:
- `robust_score = 0.6168`

`BuyHold_SPY`:
- `robust_score = 0.5792`

`V5_no_volatility_block`:
- `robust_score = 0.4271`

`V6_financial_state`:
- `robust_score = 0.4215`

`V2_reference_full`:
- `robust_score = 0.4158`

**Decision attribution at horizon 12:**  

`V5_momentum_only`:
- Regret = 0.1534
- Hit rate = 0.2725
- Beats equal = 0.4838
- Excess vs equal = 0.0138

`V6_financial_state`:
- Regret = 0.1581
- Hit rate = 0.2349
- Beats equal = 0.4921
- Excess vs equal = 0.0091

`V5_no_volatility_block`:
- Regret = 0.1743
- Hit rate = 0.1703
- Beats equal = 0.4910
- Excess vs equal = -0.0071

`V2_reference_full`:
- Regret = 0.1999
- Hit rate = 0.1462
- Beats equal = 0.3437
- Excess vs equal = -0.0327

**Versus simple rules at horizon 12:**  

`V6_financial_state`:
- TD3 minus `momentum_winner_12p` = -0.0116
- Win rate vs momentum = 0.3410
- TD3 minus `trend_spy_cash_12p` = 0.0147
- Win rate vs trend = 0.3825

`V2_reference_full`:
- TD3 minus `momentum_winner_12p` = -0.0533
- Win rate vs momentum = 0.3128
- TD3 minus `trend_spy_cash_12p` = -0.0271
- Win rate vs trend = 0.2927

**Cash discipline:**  

`V6_financial_state`:
- Mean cash = 0.0014
- Cash above 10% = 0.0028
- Unjustified cash = 0.0011

`V2_reference_full`:
- Mean cash = 0.1536
- Cash above 10% = 0.1856
- Unjustified cash = 0.1332

**Seed summary:**  
V6 mean Sharpe was positive for all five seeds, but robust seed scores were
mixed:

- Seed 7: Sharpe = 0.5518, robust = 0.1286
- Seed 21: Sharpe = 1.0367, robust = 0.6069
- Seed 42: Sharpe = 0.2405, robust = -0.2944
- Seed 84: Sharpe = 0.3137, robust = -0.4198
- Seed 101: Sharpe = 0.5134, robust = -0.0406

V6 is not just one lucky seed, but it remains fold/regime sensitive. F2/F3 were
stronger, while F1/F4 were weaker.

**Interpretation:**  
V6 improves over V2 on cash discipline, dominant-asset regret, and
rule-comparison quality. It also improves the financial structure of the state
representation by centering the model on momentum/trend, interpretable regime
probabilities, volatility proxies, and defensive attractiveness.

However, after correcting DSR aggregation, V6 no longer clearly ranks above
`V5_no_volatility_block` on composite `robust_score`. The corrected ranking is
more conservative:

- `V5_no_volatility_block`: `robust_score = 0.4271`
- `V6_financial_state`: `robust_score = 0.4215`
- `V2_reference_full`: `robust_score = 0.4158`

This means V6 remains a valid experimental candidate, but not a clear empirical
winner.

V6 does not beat the simple benchmark set. Traditional benchmarks still
dominate fold-level winners and top robust-score ranks. Therefore, V6 is worth
committing as a candidate feature set, but not as evidence of TD3 benchmark
superiority.

The DSR audit is methodologically important: the initially high V6 DSR was
overstated by pooled fold/seed returns. Pooled DSR remains useful as a
diagnostic, but final model selection should rely on median run-level DSR or
another conservative aggregation.

**Research implication:**  
V6 supports the financial design direction: a parsimonious state centered on
momentum/trend, interpretable regime probabilities, volatility proxies, and
defensive attractiveness is more coherent than the overloaded full V5 state.

The corrected DSR aggregation also shows why model selection must be
conservative. V6 improves several behavioral diagnostics, but it is not yet a
robust winner over the strongest DRL alternatives or simple benchmarks.

The next step should not be reward penalty tuning. It should be the common
experimental protocol: reward/config cleanup, benchmark leakage audit, rolling
risk parity/Markowitz benchmarks, and revalidation of V2/V6 under identical
conditions.

**Tests:**  
`python3 -m unittest discover tests` ran 823 tests OK.

## Entry X — Reward Configuration Semantics Cleanup

**Date:** 2026-05-20

**Purpose:**  
Clean the active reward/configuration semantics before continuing with the
common experimental protocol. The objective was to remove misleading inactive
parameters and confirm that evaluation metrics such as DSR and `robust_score`
do not affect training.

**Audit finding:**  
`lambda_sharpe` was present in active YAML configs and test fixtures, but it
was not used by `src/rewards/reward.py` or `src/env/portfolio_env.py`.

This created methodological ambiguity because the config suggested that Sharpe
was part of the reward, while the actual reward calculation did not use it.

**Decision implemented:**  
Removed `lambda_sharpe` from tracked active configs and test fixtures. Added
config validation so unsupported reward keys fail clearly. `lambda_sharpe` is
now rejected as an active reward field.

**Confirmed active reward semantics:**  
The active base reward remains financially interpretable and uses:

- `lambda_return`
- `lambda_transaction_cost`
- `lambda_turnover`
- `lambda_concentration`
- `lambda_drawdown`

Optional extensions remain opt-in:

- mandate penalty
- cash risk-off penalty
- turnover penalty modes

No TD3 architecture, environment dynamics, training loop, or reward formula
math was changed.

**DSR / robust_score audit:**  
DSR and `robust_score` were confirmed as evaluation/reporting-only. They appear
in analysis/reporting code, not in reward, environment, or training modules.

This preserves the distinction between:

- training reward: used to optimize the agent;
- evaluation metrics: used to compare strategies after training.

**Files modified:**  
- `configs/config.yaml`
- `configs/empirical_long_history.yaml`
- `src/utils/config.py`
- reward/config-related tests and fixtures.

**Tests:**  
- `python3 -m unittest tests/test_config.py`: 62 tests OK
- `python3 -m unittest tests/test_reward.py`: 14 tests OK
- `python3 -m unittest discover tests`: 830 tests OK

**Research implication:**  
The project now has cleaner reward semantics. Future experiments should not
interpret `lambda_sharpe` as part of the reward, and any new reward term must
be explicitly implemented, validated, and documented.

This supports the common experimental protocol phase by reducing hidden
configuration ambiguity before revalidating V2, V6, and future benchmarks.

## Entry X — Benchmark Timing and Cost-Comparability Audit

**Date:** 2026-05-20

**Purpose:**  
Audit dynamic benchmark timing, leakage risk, turnover convention, and
transaction-cost comparability before adding new rolling benchmarks or
revalidating TD3 candidates under a common experimental protocol.

**Audit result:**  
No look-ahead leakage was found in the dynamic benchmark rules:

- `momentum_winner_12p`
- `risk_adjusted_momentum_winner_12p_12p`
- `trend_spy_cash_12p`
- `defensive_risk_off_12p`

The dynamic rules were already signal-lagged: rolling signals are computed,
selected assets are shifted by one period, weights at `t` are applied to
realized returns at `t`.

**Issue found:**  
A cost-comparability issue was found in the dynamic benchmark evaluator.
First-period turnover assumed previous weights were zero. This differed from
TD3, where `PortfolioEnv` starts from equal weights at reset.

This was not a look-ahead bug, but it made dynamic benchmark transaction costs
not fully comparable with TD3.

**Fix implemented:**  
Updated `evaluate_weight_strategy` so previous weights default to equal weights,
matching TD3's reset convention.

Added comparison-friendly benchmark history columns:

- `portfolio_return`
- `financial_net_return`
- `transaction_cost`
- `turnover`
- `portfolio_value`
- `drawdown`
- `weight_*`

Existing `gross_return` and `net_return` aliases remain.

**Static benchmark note:**  
`Equal_Weight` and individual buy-and-hold helpers do not use future returns.
However, current gross static references do not model transaction costs or
turnover, so they are investable references but not fully net-cost comparable
with TD3 unless evaluated through a common weight-strategy evaluator.

**Staleness implication:**  
Dynamic benchmark outputs produced before this fix should be considered stale
because first-period turnover and transaction cost changed.

Static gross benchmark outputs are unchanged, but should be interpreted as
gross references, not fully net-cost comparable TD3 benchmarks.

**Tests added:**  
Regression tests now cover:

- future winner not selected early;
- trend rule reacts one period later;
- weights are lagged relative to signals;
- TD3-style first-period turnover convention;
- transaction costs reduce net return exactly;
- benchmark history includes comparison columns;
- benchmark timing audit summary output.

**Tests:**  
- `python3 -m unittest tests/test_dynamic_allocation_benchmarks.py`: 36 tests OK
- `python3 -m unittest discover tests`: 838 tests OK

**Research implication:**  
The dynamic benchmark timing is clean, but previous dynamic benchmark result
tables should not be used as final evidence after the cost-convention fix.

The next common-protocol step should evaluate TD3 candidates and benchmarks
through a unified net-return comparison layer, including rolling risk parity and
rolling Markowitz baselines.

## Entry X — Common Benchmark Protocol Runner Smoke Test

**Date:** 2026-05-20

**Purpose:**  
Validate the new benchmark-only protocol runner before using it as the
comparison base for TD3 candidates.

**Implementation summary:**  
Added a benchmark-only runner for the common experimental protocol. The runner
evaluates static and dynamic benchmarks on the same aligned return matrix,
using the same timing, turnover, and transaction-cost conventions.

Static benchmarks are now evaluated as explicit weight strategies inside this
runner, so they are net-cost comparable rather than only gross references.

**Benchmarks included:**  

- `BuyHold_SPY`
- `BuyHold_TLT`
- `BuyHold_GLD`
- `BuyHold_BTC-USD`
- `Equal_Weight`
- `Equal_Weight_Risky`
- `60_40_SPY_TLT`
- `momentum_winner_12p`
- `risk_adjusted_momentum_winner_12p_12p`
- `trend_spy_cash_12p`
- `defensive_risk_off_12p`
- `rolling_risk_parity_inverse_vol_12p`
- `rolling_markowitz_long_only_52p`
- `rolling_markowitz_min_variance_52p`

**New rolling baselines:**  

`rolling_risk_parity_inverse_vol_12p` was implemented as rolling
inverse-volatility risk parity, not full equal-risk-contribution optimization.

`rolling_markowitz_long_only_52p` was implemented as a constrained rolling
historical mean-variance benchmark using only past data, long-only weights,
ridge covariance regularization, and a max-weight constraint.

`rolling_markowitz_min_variance_52p` was also added as a constrained rolling
minimum-variance variant.

**Protocol conventions:**  

All benchmark histories are evaluated through the common strategy evaluator:

- information through `t-1`
- weights selected for `t`
- realized return applied at `t`
- previous weights at first evaluated period default to equal weight
- turnover = `sum(abs(w_t - w_{t-1}))`
- transaction cost = `transaction_cost_rate * turnover`
- financial net return = `portfolio_return - transaction_cost`

**Smoke test with real data:**  
The runner was executed on:

`data/processed/returns_weekly_latest.csv`

Output directory:

`outputs/tables/protocol_benchmark_comparison_smoke`

The smoke test produced:

- `benchmark_metrics_table.csv`
- `benchmark_comparison_summary.csv`
- `benchmark_diagnostics.csv`
- 14 benchmark history files under `histories/`

All benchmark history files included the required comparison columns:

- `portfolio_return`
- `financial_net_return`
- `transaction_cost`
- `turnover`
- `portfolio_value`
- `drawdown`
- `weight_*`

**Smoke result:**  
The benchmark runner produced coherent outputs. No missing history columns were
detected. Rolling risk parity, constrained rolling Markowitz, and constrained
minimum-variance benchmarks all appeared in the summary tables.

The smoke outputs should be treated as functional validation, not final
model-selection evidence.

**Tests:**  

- `python3 -m unittest tests/test_run_protocol_benchmark_comparison.py`: 7 tests OK
- `python3 -m unittest tests/test_dynamic_allocation_benchmarks.py`: 55 tests OK
- `python3 -m unittest discover tests`: 864 tests OK

**Research implication:**  
The project now has a common benchmark suite that is timing-aware,
net-cost-comparable, and suitable as the baseline layer for the next TD3
candidate comparison.

The next step is to create a TD3 protocol comparison runner that evaluates
V2, `V5_no_volatility_block`, and V6 against this benchmark suite under the
same protocol.

## Entry X — TD3 Protocol Comparison Runner Smoke Test

**Date:** 2026-05-20

**Purpose:**  
Validate the unified TD3 protocol comparison runner as the official combined
reporting layer for benchmark and TD3 candidate comparisons.

**Implementation summary:**  
Added a TD3 protocol comparison runner as an ingestion/reporting layer, not a
TD3 training orchestrator.

The runner reuses the benchmark protocol runner and can combine benchmark
metrics with ingested TD3 candidate results under the same reporting schema.

**Output files:**  

The smoke run produced:

- `protocol_comparison_metrics.csv`
- `protocol_comparison_summary.csv`
- `protocol_comparison_diagnostics.csv`
- `protocol_model_selection_table.csv`
- `benchmark_metrics_table.csv`
- `td3_candidate_metrics_table.csv`
- `protocol_metadata.json`
- `histories/`

**Smoke run:**  

Command:

```text
python3 -m src.experiments.run_protocol_td3_comparison \
  --returns-path data/processed/returns_weekly_latest.csv \
  --output-dir outputs/tables/protocol_td3_comparison_smoke \
  --smoke

  **Real ingestion note:**  
The real ingestion test successfully combined fresh protocol benchmarks with
TD3 candidate rows ingested from the previous V6 validation experiment.

This validates the reporting infrastructure, not final TD3 superiority. In this
combined table, benchmark rows do not yet include protocol-level robust_score
fields, so `protocol_model_selection_table.csv` should not be interpreted as a
final model ranking. Financial metrics such as Sharpe, drawdown, turnover, and
net-return comparability still show that simple benchmarks remain very strong.

A final model-selection run should either compute robust_score for benchmarks
inside the combined protocol runner or clearly separate benchmark financial
ranking from TD3 robust-score ranking.

## Entry X — Benchmark Robust Score Added to Protocol TD3 Comparison

**Date:** 2026-05-20

**Purpose:**  
Fix the combined protocol ranking so benchmarks and TD3 candidates both carry
`robust_score` / DSR fields when possible.

**Issue:**  
The first combined TD3 protocol comparison correctly ingested TD3 robust-score
fields, but regenerated benchmark rows had missing `robust_score`. This could
make TD3 appear above benchmarks in `protocol_model_selection_table.csv` simply
because benchmarks lacked the robust-score field.

**Fix implemented:**  
Updated the TD3 protocol comparison runner so regenerated benchmark rows receive
robust-score and DSR fields from their benchmark histories.

Benchmark robust-score method:

- `single_history_date_averaged_dsr`
- uses benchmark `financial_net_return` when available
- sets benchmark `dsr_method = date_averaged`
- leaves benchmark run-level DSR fields as `NaN`, because these are
  deterministic single-history benchmark runs, not fold/seed TD3 runs

TD3 robust-score fields remain ingested from the candidate experiment output,
using the conservative DSR aggregation already implemented.

**Real comparison result:**  
Output folder:

`outputs/tables/protocol_td3_comparison_real_test_with_benchmark_robust_score`

After benchmark robust scores were included, TD3 no longer floated above
benchmarks because of missing benchmark `robust_score`.

Top combined robust-score ranks:

- `momentum_winner_12p`: `robust_score = 0.8575`
- `Equal_Weight_Risky`: `robust_score = 0.8123`
- `Equal_Weight`: `robust_score = 0.8004`
- `risk_adjusted_momentum_winner_12p_12p`: `robust_score = 0.7789`
- `rolling_markowitz_long_only_52p`: `robust_score = 0.7677`
- `rolling_risk_parity_inverse_vol_12p`: `robust_score = 0.7355`

**Interpretation:**  
The combined ranking is now more honest. Simple and dynamic benchmarks dominate
the top robust-score ranks in this run. TD3 candidates remain useful for
research comparison, but there is still no evidence of TD3 benchmark
superiority under the current protocol.

Important methodological caveat: benchmark robust scores are computed from
single deterministic histories using date-averaged DSR, while TD3 robust scores
come from fold/seed candidate outputs using median-run DSR. This difference is
now explicit in `dsr_method` and metadata.

**Tests:**  

- `python3 -m unittest tests/test_run_protocol_td3_comparison.py`: 17 tests OK
- `python3 -m unittest tests/test_run_protocol_benchmark_comparison.py`: 7 tests OK
- `python3 -m unittest discover tests`: 881 tests OK

**Research implication:**  
The protocol comparison runner can now compare benchmarks and TD3 candidates
with a shared robust-score reporting layer. This closes the main ranking
fairness issue in the combined protocol infrastructure.

## Entry X — Protocol-Pure TD3 Revalidation

**Date:** 2026-05-20

**Purpose:**  
Run a clean protocol-pure TD3 revalidation under the current common experimental
protocol, instead of relying only on historical candidate outputs.

**Setup:**  

Candidates:

- `V2_reference_full`
- `V5_no_volatility_block`
- `V6_financial_state`

Run configuration:

- `returns_path = data/processed/returns_weekly_latest.csv`
- episodes = 30
- seeds = `[7, 21, 42, 84, 101]`
- expanding walk-forward
- transaction cost = 0.001
- cleaned reward/config semantics
- `lambda_sharpe_present_or_active = false`
- `robust_score_training_usage = evaluation_only`

Output directory:

`outputs/tables/protocol_pure_td3_revalidation_30ep_5seeds`

Combined protocol comparison output:

`outputs/tables/protocol_pure_td3_comparison_30ep_5seeds`

**TD3 test aggregate:**  

`V2_reference_full`:

- mean Sharpe = 0.3682
- robust Sharpe = -0.2670
- robust_score = 0.6297
- max drawdown = -0.2033
- average turnover = 0.6070

`V5_no_volatility_block`:

- mean Sharpe = 0.1810
- robust Sharpe = -0.3995
- robust_score = 0.1178
- max drawdown = -0.2288
- average turnover = 0.6238

`V6_financial_state`:

- mean Sharpe = 0.4683
- robust Sharpe = 0.0164
- robust_score = 0.4619
- max drawdown = -0.2401
- average turnover = 0.4232

**TD3 interpretation:**  
V6 delivered the highest Sharpe among TD3 candidates in the protocol-pure run.
However, V2 ranked higher by composite `robust_score`, mainly because its
drawdown, Sortino/Calmar components, and conservative DSR aggregation produced
a stronger composite profile.

Therefore:

- Best TD3 by Sharpe: `V6_financial_state`
- Best TD3 by robust_score: `V2_reference_full`
- Weakest TD3 in this protocol-pure run: `V5_no_volatility_block`

**Combined benchmark comparison:**  
After adding benchmark robust scores to the combined protocol runner, benchmarks
dominated the top robust-score ranks:

- `momentum_winner_12p`: `robust_score = 0.8575`
- `Equal_Weight_Risky`: `robust_score = 0.8123`
- `Equal_Weight`: `robust_score = 0.8004`
- `risk_adjusted_momentum_winner_12p_12p`: `robust_score = 0.7789`
- `rolling_markowitz_long_only_52p`: `robust_score = 0.7677`
- `rolling_risk_parity_inverse_vol_12p`: `robust_score = 0.7355`

The best TD3 candidate by robust_score, `V2_reference_full`, ranked below these
benchmarks.

**Conclusion:**  
Under the current common protocol, TD3 does not demonstrate benchmark
superiority. The protocol-pure run supports a conservative conclusion: TD3
candidates are research-relevant, but simple and dynamic benchmark strategies
remain stronger under the current evaluation design.

**Research implication:**  
The project should not claim that TD3 beats traditional or transparent dynamic
benchmarks. The stronger contribution is methodological: a protocol-aware DRL
portfolio evaluation framework with investable timing, transaction costs,
walk-forward validation, conservative DSR aggregation, and benchmark
comparability.

Future work should focus on explaining why TD3 fails to dominate, improving
state/reward design only with clear hypotheses, and avoiding benchmark
overfitting.

**Tests:**  

- `python3 -m unittest tests/test_run_protocol_pure_td3_revalidation.py`: 5 tests OK
- `python3 -m unittest tests/test_run_protocol_td3_comparison.py`: 17 tests OK
- `python3 -m unittest discover tests`: 886 tests OK