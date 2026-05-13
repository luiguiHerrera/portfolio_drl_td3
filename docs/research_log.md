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
