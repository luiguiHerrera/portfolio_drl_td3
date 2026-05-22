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

## Entry X — Robust Score Bias Audit and Recovery-Based Mandate-Aware Evaluation Design

**Date:** 2026-05-20

**Purpose:**  
Audit whether the current `robust_score` is biased toward momentum-like
strategies, high DSR, high Sharpe-like behavior, and insufficiently penalizes
drawdown, turnover, and concentration.

The motivation came from the protocol-pure TD3 comparison, where
`momentum_winner_12p` ranked first by `robust_score` despite a maximum drawdown
above 50%. This raised a methodological concern: a strategy can be attractive
under a performance-oriented risk-return score while still being unsuitable for
a realistic portfolio mandate.

---

### Robust score bias audit

**Implementation:**  
Added a separate audit/reporting layer without changing production
`robust_score` logic.

Files created:

- `src/analysis/audit_robust_score_bias.py`
- `tests/test_audit_robust_score_bias.py`

Audit outputs:

- `robust_score_component_audit.csv`
- `robust_score_rank_sensitivity.csv`
- `robust_score_drawdown_turnover_flags.csv`
- `robust_score_method_comparison.csv`

**Current robust_score weights:**  

- DSR: 0.30
- Sortino: 0.20
- Calmar: 0.20
- Drawdown: 0.15
- Stability: 0.10
- Discipline: 0.05

**Findings:**  
The current `robust_score` is materially driven by DSR, Sortino, and Calmar.
Together, these components represent 70% of the composite score.

Drawdown is only min-max normalized and has no hard cap. Turnover and
concentration enter weakly through the 5% discipline component. Effective assets
affect discipline, but average max weight is not directly scored.

As a result, concentrated or high-turnover strategies can still rank highly if
their DSR, Sortino, and Calmar components are strong.

**Momentum result:**  
`momentum_winner_12p` ranks first mainly because of very high:

- `dsr_score = 0.9873`
- `sortino_score = 1.0000`
- `calmar_score = 0.9819`

However, it also has:

- `max_drawdown = -0.5127`

This implies that the current `robust_score` should not be interpreted as a
mandate-aware score. A drawdown of approximately 50% requires a subsequent gain
of approximately 100% on the remaining capital to recover to breakeven, which is
not acceptable for a base multi-asset portfolio mandate.

**DSR method concern:**  
Benchmark rows using `date_averaged` DSR showed much higher DSR scores than TD3
rows using `median_run` DSR.

Observed means:

- `date_averaged` benchmark rows: mean `dsr_score = 0.8058`
- `median_run` TD3 rows: mean `dsr_score = 0.0647`

This is not necessarily a code bug, but it confirms that benchmark and TD3 DSR
methods are not identical evidence structures.

**Sensitivity result:**  
Alternative scoring variants showed that rankings are sensitive to the DSR and
mandate assumptions:

- Removing DSR moved TD3 candidates upward; `V2_reference_full` became rank 1
  in the audit recomputation.
- Adding a drawdown hard cap penalized high-drawdown momentum and Markowitz-like
  strategies.
- A mandate-style score moved risk parity and minimum-variance Markowitz toward
  the top, and improved V6's relative position.

**Conclusion of the audit:**  
No implementation bug was found in production `robust_score`. The issue is
scoring design.

The current `robust_score` is best interpreted as a performance-oriented
robustness score, not a mandate-aware patrimonial score.

---

### Mandate-aware design decision

A separate `mandate_aware_score` was created on top of the existing
`robust_score`. The original `robust_score` was not replaced.

The objective is to distinguish between:

1. `performance_robust_score`: performance-oriented risk-return robustness.
2. `mandate_aware_score`: suitability under realistic portfolio drawdown
   constraints.

This prevents the evaluation framework from confusing aggressive performance
with investable robustness.

---

### Drawdown mandate buckets

Since `max_drawdown` is represented as a negative number, the mandate buckets
are:

- `clean_mandate`:  
  `max_drawdown >= -0.20`

- `eligible_yellow`:  
  `-0.25 <= max_drawdown < -0.20`

- `eligible_red`:  
  `-0.30 <= max_drawdown < -0.25`

- `not_eligible`:  
  `max_drawdown < -0.30`

These buckets define mandate eligibility only. They do not directly define the
score multiplier.

The interpretation is:

1. Clean mandate strategies: drawdown up to 20%.
2. Eligible but penalized strategies: drawdown between 20% and 30%.
3. Not eligible for the base mandate: drawdown above 30%.

This avoids treating high-drawdown momentum strategies as superior mandate
candidates solely because they rank highly under performance-oriented metrics.

---

### Recovery-based multiplier

The initial mandate-aware design considered discrete step multipliers by
drawdown bucket. This was replaced by a recovery-based continuous multiplier,
which is more directly grounded in drawdown recovery asymmetry.

Let:

`abs_dd = abs(max_drawdown)`

The recovery return required to return to breakeven is:

`recovery_required = abs_dd / (1 - abs_dd)`

The drawdown multiplier is:

`drawdown_multiplier = max(0, 1 - recovery_required)`

The mandate-aware score is:

`mandate_aware_score = robust_score * drawdown_multiplier`

For strategies classified as `not_eligible`, `mandate_aware_score` is forced to
zero regardless of the formula-based multiplier.

This separates two concepts:

1. `mandate_bucket`: eligibility classification.
2. `drawdown_multiplier`: continuous penalty based on recovery asymmetry.

Examples:

- DD = -1% → recovery required ≈ 1.01% → multiplier ≈ 0.9899
- DD = -20% → recovery required = 25.0% → multiplier = 0.7500
- DD = -25% → recovery required = 33.3% → multiplier ≈ 0.6667
- DD = -30% → recovery required = 42.9% → multiplier ≈ 0.5714
- DD < -30% → not eligible for the base mandate → mandate-aware score = 0

This avoids arbitrary step penalties while keeping the base drawdown mandate
explicit.

---

### Mandate-aware implementation

Files created:

- `src/analysis/mandate_aware_score.py`
- `tests/test_mandate_aware_score.py`

Generated outputs:

- `outputs/tables/mandate_aware_score/mandate_aware_ranking.csv`
- `outputs/tables/mandate_aware_score/mandate_bucket_summary.csv`
- `outputs/tables/mandate_aware_score/mandate_eligibility_flags.csv`

The scoring layer does not modify:

- production `robust_score`
- TD3 architecture
- reward function
- environment dynamics
- training logic
- benchmark logic

It is a separate reporting/evaluation layer.

---

### Recovery-based mandate-aware ranking

After applying the recovery-based mandate-aware score, the top mandate-aware
strategies were:

- `BuyHold_GLD`: `mandate_aware_score = 0.5244`
- `trend_spy_cash_12p`: `mandate_aware_score = 0.4841`
- `V2_reference_full`: `mandate_aware_score = 0.4691`
- `rolling_markowitz_min_variance_52p`: `mandate_aware_score = 0.4533`
- `defensive_risk_off_12p`: `mandate_aware_score = 0.4414`
- `rolling_risk_parity_inverse_vol_12p`: `mandate_aware_score = 0.4262`
- `60_40_SPY_TLT`: `mandate_aware_score = 0.3848`
- `V6_financial_state`: `mandate_aware_score = 0.3159`
- `V5_no_volatility_block`: `mandate_aware_score = 0.0829`
- `momentum_winner_12p`: `mandate_aware_score = 0.0000`

The best clean mandate strategies were:

- `BuyHold_GLD`
- `trend_spy_cash_12p`

The strongest TD3 candidate under the recovery-based mandate-aware score was:

- `V2_reference_full`

V6 remained eligible, but ranked below several more conservative benchmark
strategies.

The following strategies were classified as not eligible for the base mandate:

- `momentum_winner_12p`
- `Equal_Weight_Risky`
- `Equal_Weight`
- `risk_adjusted_momentum_winner_12p_12p`
- `rolling_markowitz_long_only_52p`
- `BuyHold_SPY`
- `BuyHold_BTC-USD`
- `BuyHold_TLT`

---

### Interpretation

The mandate-aware layer does not claim that high-drawdown strategies are invalid
in absolute terms. It classifies them as outside the base mandate.

Under the performance-oriented robust score, momentum-style strategies can rank
highly because of strong DSR, Sortino, and Calmar components.

Under the recovery-based mandate-aware score, strategies with excessive
drawdowns are either continuously penalized or excluded from the base mandate if
their maximum drawdown exceeds 30%.

This makes the evaluation more consistent with a real-world multi-asset mandate,
where drawdown depth affects investor survival, recovery burden, behavioral
risk, and capital preservation.

---

### Research implication

The project should report both rankings:

- performance robust ranking
- mandate-aware ranking

This allows momentum-style strategies to be recognized for strong performance
while clearly marking them as unsuitable for the base mandate when their
drawdown violates realistic portfolio constraints.

The strongest claim is not that TD3 dominates benchmarks. The stronger and more
honest contribution is that the framework separates:

1. raw performance robustness,
2. mandate eligibility,
3. drawdown recovery burden,
4. and real-world investability.

Under this framework, TD3 does not dominate the full benchmark universe, but
`V2_reference_full` becomes a competitive mandate-aware candidate, ranking third
overall under the recovery-based mandate-aware score.

---

### Tests

Robust score bias audit:

- `python3 -m unittest tests/test_audit_robust_score_bias.py`: 5 tests OK
- `python3 -m unittest tests/test_robust_score.py`: 19 tests OK
- `python3 -m unittest discover tests`: 891 tests OK

Mandate-aware scoring layer:

- `python3 -m unittest tests/test_mandate_aware_score.py`: 7 tests OK
- `python3 -m unittest discover tests`: 898 tests OK

## Entry X — Protocol-Pure TD3 Revalidation: 60 Episodes × 10 Seeds

**Date:** 2026-05-21

**Purpose:**  
Run a larger protocol-pure TD3 revalidation to test whether the 30ep × 5seeds
results were stable across more seeds and a longer training budget.

The goal was to answer two questions:

1. Whether the relative ranking of V2, V5, and V6 was stable across more seeds.
2. Whether 30 episodes had materially undertrained the TD3 candidates.

**Setup:**  

Candidates:

- `V2_reference_full`
- `V5_no_volatility_block`
- `V6_financial_state`

Run configuration:

- `returns_path = data/processed/returns_weekly_latest.csv`
- episodes = 60
- seeds = `[7, 21, 42, 84, 101, 123, 202, 303, 404, 505]`
- expanding walk-forward
- transaction cost = 0.001
- current common experimental protocol
- current cleaned reward/config semantics
- current recovery-based mandate-aware scoring layer

Output directories:

- `outputs/tables/protocol_pure_td3_revalidation_60ep_10seeds`
- `outputs/tables/protocol_pure_td3_comparison_60ep_10seeds`
- `outputs/tables/mandate_aware_score_60ep_10seeds`

The run was executed after commit:

`303d932`

**Tests before run:**  

- `python3 -m unittest discover tests`: 898 tests OK

---

### TD3 aggregate results

Test split aggregate:

`V2_reference_full`:

- mean Sharpe = 0.2076
- robust Sharpe = -0.2643
- cumulative return = 0.0297
- annualized return = 0.0154
- annualized volatility = 0.2099
- max drawdown = -0.1953
- average turnover = 0.6065
- effective assets = 1.0834
- mean cash weight = 0.2387
- worst max drawdown = -0.5717

`V5_no_volatility_block`:

- mean Sharpe = 0.4874
- robust Sharpe = -0.0828
- cumulative return = 0.0626
- annualized return = 0.0489
- annualized volatility = 0.2757
- max drawdown = -0.2360
- average turnover = 0.6246
- effective assets = 1.0869
- mean cash weight ≈ 0.0000
- worst max drawdown = -0.6341

`V6_financial_state`:

- mean Sharpe = 0.2669
- robust Sharpe = -0.2723
- cumulative return = 0.1280
- annualized return = 0.1389
- annualized volatility = 0.3409
- max drawdown = -0.2885
- average turnover = 0.4860
- effective assets = 1.0660
- mean cash weight = 0.0004
- worst max drawdown = -0.6247

---

### TD3 robust score ranking

The TD3-only robust score ranking changed relative to the earlier 30ep × 5seeds
run:

1. `V5_no_volatility_block`: `robust_score = 0.5469`
2. `V2_reference_full`: `robust_score = 0.2975`
3. `V6_financial_state`: `robust_score = 0.2944`

This suggests that V6 did not strengthen with additional episodes/seeds, while
V5 improved meaningfully under the larger run.

---

### Combined protocol comparison

Benchmarks continued to dominate the performance-oriented robust score ranking.

Top performance robust strategies:

- `momentum_winner_12p`: `robust_score = 0.8575`
- `Equal_Weight_Risky`: `robust_score = 0.8123`
- `Equal_Weight`: `robust_score = 0.8004`
- `risk_adjusted_momentum_winner_12p_12p`: `robust_score = 0.7789`
- `rolling_markowitz_long_only_52p`: `robust_score = 0.7677`
- `rolling_risk_parity_inverse_vol_12p`: `robust_score = 0.7355`
- `BuyHold_GLD`: `robust_score = 0.6967`
- `rolling_markowitz_min_variance_52p`: `robust_score = 0.6870`
- `trend_spy_cash_12p`: `robust_score = 0.6362`
- `BuyHold_SPY`: `robust_score = 0.6324`

The best TD3 candidate in this larger run, `V5_no_volatility_block`, ranked
below several benchmark strategies under the performance-oriented robust score.

---

### Recovery-based mandate-aware ranking

After applying the recovery-based mandate-aware score, the top mandate-aware
strategies were:

- `BuyHold_GLD`: `mandate_aware_score = 0.5244`
- `trend_spy_cash_12p`: `mandate_aware_score = 0.4841`
- `rolling_markowitz_min_variance_52p`: `mandate_aware_score = 0.4533`
- `defensive_risk_off_12p`: `mandate_aware_score = 0.4414`
- `rolling_risk_parity_inverse_vol_12p`: `mandate_aware_score = 0.4262`
- `60_40_SPY_TLT`: `mandate_aware_score = 0.3848`
- `V5_no_volatility_block`: `mandate_aware_score = 0.3780`
- `V2_reference_full`: `mandate_aware_score = 0.2253`
- `V6_financial_state`: `mandate_aware_score = 0.1750`

The best TD3 candidate under the 60ep × 10seeds mandate-aware ranking was:

- `V5_no_volatility_block`

This differs from the earlier 30ep × 5seeds result, where `V2_reference_full`
was the strongest TD3 candidate under the mandate-aware layer.

---

### Interpretation

The larger run changed the TD3 ranking:

- `V5_no_volatility_block` became the strongest TD3 candidate by both
  performance-oriented robust score and recovery-based mandate-aware score.
- `V2_reference_full` remained the cleanest TD3 candidate by drawdown bucket,
  but its performance profile weakened.
- `V6_financial_state` did not improve with more episodes/seeds and ranked last
  among the three TD3 candidates under mandate-aware scoring.

However, all TD3 candidates remained highly concentrated, with effective assets
close to one. This means the learned policies are behaving closer to
single-asset selectors than diversified portfolio allocators.

The larger run therefore strengthens two conclusions:

1. TD3 does not dominate the benchmark universe under the current protocol.
2. The main TD3 behavioral issue is structural concentration, not merely
   insufficient training episodes.

---

### Research implication

The 60ep × 10seeds revalidation provides a more reliable stress test than the
initial 30ep × 5seeds run.

The result does not justify claiming TD3 benchmark superiority. Instead, it
suggests that further work should focus on understanding and controlling the
concentration behavior of learned TD3 policies.

This motivates the next audit: whether the active reward configuration,
especially transaction-cost and turnover penalties combined with zero
concentration penalty, is indirectly allowing or encouraging excessive
concentration.

**Tests:**  

- `python3 -m unittest discover tests`: 898 tests OK before the overnight run


## Entry X — Reward Incentive Audit: Turnover, Transaction Costs, and Concentration

**Date:** 2026-05-21

**Purpose:**  
Audit whether the active reward configuration is indirectly encouraging
excessive concentration through transaction-cost and turnover penalties.

The concern was that if transaction costs and turnover penalties discourage
rebalancing, while concentration penalties are zero or weak, TD3 may learn to
concentrate in one asset and remain there. This would reduce costs but would not
necessarily represent a useful portfolio allocation signal.

**Implementation:**  
Added a reporting-only reward incentive audit. No reward, TD3 architecture,
environment dynamics, training logic, `robust_score`, `mandate_aware_score`,
README, or docs were modified by the audit itself.

Files created:

- `src/analysis/audit_reward_incentives.py`
- `tests/test_audit_reward_incentives.py`

Generated outputs:

- `outputs/tables/reward_incentive_audit/reward_concentration_turnover_audit.csv`
- `outputs/tables/reward_incentive_audit/reward_lazy_concentration_flags.csv`
- `outputs/tables/reward_incentive_audit/reward_candidate_behavior_summary.csv`
- `outputs/tables/reward_incentive_audit/reward_30ep_vs_60ep_comparison.csv`

**Inspected files:**  

- `src/rewards/reward.py`
- `src/env/portfolio_env.py`
- `tests/test_reward.py`
- `configs/config.yaml`
- `configs/empirical_long_history.yaml`
- `src/experiments/run_protocol_pure_td3_revalidation.py`

**Active reward terms:**  

- `lambda_return`: active; rewards period portfolio return.
- `lambda_transaction_cost`: active; pushes reward toward net return and discourages trading.
- `lambda_turnover`: active; separately penalizes turnover and can discourage rebalancing.
- `lambda_concentration`: active if configured, but current main configs use `0.0`.
- `lambda_drawdown`: active; mild in the empirical config.
- Cash risk-off penalty: active for V5/V6 protocol candidates.
- Mandate penalty: available but disabled in protocol-pure revalidation.
- Turnover modes: V2 uses `linear`; V5/V6 use `excess_linear` with `turnover_free_band = 0.20`.

**Finding:**  
`lambda_transaction_cost` and `lambda_turnover` plausibly encourage portfolio
stickiness. However, the 60ep × 10seeds evidence is not a clean lazy
low-turnover concentration story.

All TD3 candidates remain extremely concentrated, but V2 and V5 also show high
turnover:

- `V2_reference_full`: turnover = 0.6065, effective assets = 1.0834, robust_score = 0.2975
- `V5_no_volatility_block`: turnover = 0.6246, effective assets = 1.0869, robust_score = 0.5469
- `V6_financial_state`: turnover = 0.4860, effective assets = 1.0660, robust_score = 0.2944

This suggests that the agent is not simply holding one concentrated allocation
to avoid trading costs. Instead, it is often rotating between highly concentrated
allocations.

**Diagnostic classification:**  
All three TD3 candidates were flagged as `lazy_concentration_candidate` by the
diagnostic layer because concentration is extreme and performance/stability is
not strong enough to justify it.

None were flagged as `justified_concentration_candidate`.

These flags are diagnostic, not final truth.

**30ep × 5seeds vs 60ep × 10seeds:**  

- Concentration persisted and worsened slightly for all candidates.
- Turnover persisted.
- V2 turnover remained roughly high.
- V5 turnover remained roughly high.
- V6 turnover increased.
- The best TD3 candidate changed from `V2_reference_full` at 30ep × 5seeds to
  `V5_no_volatility_block` at 60ep × 10seeds.
- Higher episodes/seeds did not reduce concentration.

**Interpretation:**  
The current issue is not only transaction-cost-induced inactivity. The broader
issue is structural concentration in the learned TD3 policies.

The reward and action behavior currently allow the agent to behave closer to a
single-asset selector than a diversified portfolio allocator.

**Research implication:**  
A strong concentration penalty should not be activated by default, because it
may force artificial diversification and obscure whether concentration is
sometimes useful.

The next appropriate step is a controlled soft concentration experiment:

- baseline: `lambda_concentration = 0.0`
- soft: `lambda_concentration = 0.01`
- moderate: `lambda_concentration = 0.03`
- aggressive diagnostic only: `lambda_concentration = 0.05`

The goal is not to force diversification. The goal is to test whether soft
concentration pressure improves mandate-aware behavior without destroying
performance.

**Tests:**  

- `python3 -m unittest tests/test_reward.py`: 14 tests OK
- `python3 -m unittest tests/test_audit_reward_incentives.py`: 5 tests OK
- `python3 -m unittest discover tests`: 903 tests OK

### Role-aware concentration audit update

The reward incentive audit was updated to distinguish structural benchmark
concentration from learned TD3 concentration.

This was necessary because not all concentration has the same interpretation.
For example, `BuyHold_GLD` has effective assets close to one by design. It is a
single-asset benchmark, so its concentration is structural and should not be
classified as lazy learned behavior.

The updated audit now classifies strategy roles and concentration origin:

- `BuyHold_GLD`, `BuyHold_SPY`, `BuyHold_BTC-USD`, and `BuyHold_TLT` are
  classified as `structural_concentration_benchmark`.
- `momentum_winner_12p` and
  `risk_adjusted_momentum_winner_12p_12p` are also treated as structurally
  concentrated or winner-take-all benchmarks.
- `V2_reference_full`, `V5_no_volatility_block`, and `V6_financial_state` are
  classified as learned TD3 allocators with learned extreme concentration.

The audit no longer labels structurally concentrated benchmarks as lazy.
`BuyHold_GLD` is now correctly treated as a structurally concentrated benchmark,
with the reason:

`Structural single-asset benchmark; concentration expected by design.`

The audit also now exposes explicit return fields, including:

- cumulative return
- annualized return
- annualized volatility
- Sharpe
- Sortino
- Calmar
- robust_score
- mandate_aware_score

This makes the concentration diagnosis more transparent because concentration
can only be judged relative to return, risk, drawdown, turnover, and stability.

Current diagnostic thresholds:

- high concentration: effective assets `< 1.5`
- extreme concentration: effective assets `< 1.2`
- high turnover: turnover `> 0.50`
- low-turnover high concentration: turnover `< 0.25` and effective assets `< 1.5`

A learned concentration candidate is only considered justified if it satisfies:

- Sharpe `>= 0.75`
- robust_score `>= 0.60`
- mandate_aware_score `>= 0.40`, if available
- max_drawdown `>= -0.25`
- turnover `<= 0.50`
- no extreme validation-test Sharpe gap

Under these thresholds, no TD3 candidate qualifies as justified concentration.

The 30ep × 5seeds vs 60ep × 10seeds comparison showed:

- `V2_reference_full`: robust_score fell by `-0.3322`; effective assets fell by
  `-0.0549`; turnover remained high.
- `V5_no_volatility_block`: robust_score rose by `+0.4291`; effective assets
  fell by `-0.0457`; turnover stayed high.
- `V6_financial_state`: robust_score fell by `-0.1675`; effective assets fell
  by `-0.0219`; turnover rose by `+0.0628`.

Interpretation:

The concentration problem is not simply that some strategies are concentrated.
Structural benchmark concentration is expected. The relevant concern is that
the learned TD3 allocators converge toward extreme concentration despite having
the ability to diversify, and the current evidence is not strong enough to
classify that learned concentration as justified.

**Updated tests:**

- `python3 -m unittest tests/test_audit_reward_incentives.py`: 6 tests OK
- `python3 -m unittest tests/test_reward.py`: 14 tests OK
- `python3 -m unittest discover tests`: 904 tests OK

## Entry X — Controlled Soft Concentration Penalty Experiment

**Date:** 2026-05-21

**Purpose:**  
Test whether a soft positive `lambda_concentration` can reduce learned extreme
concentration in the strongest TD3 candidate without destroying performance or
mandate-aware behavior.

The experiment was run only as an isolated experimental layer. It did not
modify the default config, TD3 architecture, reward implementation,
environment, training logic, `robust_score`, or `mandate_aware_score`.

**Setup:**  

Candidate:

- `V5_no_volatility_block`

Grid:

- `lambda_concentration = 0.00`
- `lambda_concentration = 0.01`
- `lambda_concentration = 0.03`
- `lambda_concentration = 0.05`

Run configuration:

- episodes = 60
- seeds = `[7, 21, 42, 84, 101]`
- folds = 4
- current protocol-pure training stack
- output directory:
  `outputs/tables/concentration_penalty_experiment_v5_60ep_5seeds`

**Baseline result:**  

`lambda_concentration = 0.00`:

- robust_score = 0.7088
- mandate_aware_score = 0.5505
- Sharpe = 0.5348
- annualized_return = 0.1179
- max_drawdown = -0.1825
- average_turnover = 0.6154
- effective assets = 1.0783
- average max weight = 0.9672

The baseline remains an extreme learned concentration case.

**Grid results:**  

`lambda_concentration = 0.01`:

- effective assets increased by 0.0606
- average max weight decreased by 0.0260
- robust_score fell by 0.2382
- mandate_aware_score fell by 0.2063
- turnover increased by 0.1401
- decision label: `no_behavioral_improvement`

`lambda_concentration = 0.03`:

- effective assets increased by 0.3024
- average max weight decreased by 0.1177
- robust_score fell by 0.3239
- mandate_aware_score fell by 0.2749
- turnover increased by 0.1926
- decision label: `diversifies_but_hurts_performance`

`lambda_concentration = 0.05`:

- effective assets increased by 0.6444
- average max weight decreased by 0.2267
- robust_score fell by 0.6251
- mandate_aware_score fell by 0.4929
- turnover increased by 0.1729
- decision label: `diversifies_but_hurts_performance`

**Interpretation:**  
A direct concentration penalty successfully increased diversification, but it
did not improve the candidate's mandate-aware behavior. Instead, it materially
reduced `robust_score` and `mandate_aware_score`, while increasing turnover.

This suggests that V5's concentration is not fixed by a simple soft
concentration penalty. The learned policy appears to rely on concentrated
positions for its performance, and forcing diversification through this reward
term leads to worse risk-adjusted and mandate-aware outcomes.

**Research implication:**  
A strong or default `lambda_concentration` should not be activated based on the
current evidence.

The result rejects the simplistic solution:

`extreme concentration -> add concentration penalty by default`

A more careful next step is needed. Possible directions include:

- testing whether concentration constraints should be handled at the action or
  allocation layer rather than only through reward penalties;
- designing a conditional concentration penalty that distinguishes justified
  concentration from unstable concentration;
- reviewing action geometry and policy output behavior;
- comparing against explicit max-weight constrained baselines.

**Tests:**  

- `python3 -m unittest discover tests`: 911 tests OK before the experiment


## Entry X — Experiment-Only Max-Weight Cap Test

**Date:** 2026-05-21

**Purpose:**  
Test whether learned TD3 concentration is better controlled through an
allocation/action constraint rather than through a direct reward penalty.

This experiment was motivated by the previous concentration penalty grid, where
positive `lambda_concentration` increased diversification mechanically but
reduced `robust_score`, reduced `mandate_aware_score`, and increased turnover.

The new hypothesis was that concentration may be better controlled by limiting
the maximum portfolio weight directly.

**Implementation:**  
Added an experiment-only max-weight cap runner.

Files created:

- `src/experiments/run_max_weight_cap_experiment.py`
- `tests/test_run_max_weight_cap_experiment.py`

The runner does not modify:

- default config behavior
- global `PortfolioEnv` behavior
- TD3 architecture
- production `robust_score`
- `mandate_aware_score`
- README

The cap is implemented through an experiment-only `CappedPortfolioEnv` and pure
projection utilities:

- `project_weights_to_max_cap(weights, max_weight)`
- `apply_max_weight_cap_to_action(weights, max_weight)`

The projection enforces:

- non-negative weights
- weights sum to one
- maximum weight less than or equal to the cap

The uncapped baseline leaves weights unchanged.

**Experiment setup:**  

Candidate:

- `V5_no_volatility_block`

Grid:

- uncapped
- max weight cap = 0.80
- max weight cap = 0.70
- max weight cap = 0.60

Run configuration:

- episodes = 60
- seeds = `[7, 21, 42, 84, 101]`
- folds = 4
- output directory:
  `outputs/tables/max_weight_cap_experiment_v5_60ep_5seeds`

**Main result:**  

Uncapped baseline:

- robust_score = 0.1599
- mandate_aware_score = 0.1207
- annualized_return = 0.0776
- Sharpe = 0.5858
- max_drawdown = -0.1965
- average_turnover = 0.6712
- effective assets = 1.1015
- average max weight = 0.9565

Max-weight cap 0.80:

- robust_score = 0.3077
- mandate_aware_score = 0.2297
- annualized_return = 0.1134
- Sharpe = 0.6121
- max_drawdown = -0.2021
- average_turnover = 0.6105
- effective assets = 1.5652
- average max weight = 0.7855
- decision label: `reduces_concentration_and_turnover`

Max-weight cap 0.70:

- robust_score = 0.2899
- mandate_aware_score = 0.2222
- annualized_return = 0.0672
- Sharpe = 0.5765
- max_drawdown = -0.1894
- average_turnover = 0.4553
- effective assets = 1.9501
- average max weight = 0.6928
- decision label: `reduces_concentration_and_turnover`

Max-weight cap 0.60:

- robust_score = 0.7257
- mandate_aware_score = 0.5960
- annualized_return = 0.1235
- Sharpe = 0.8858
- max_drawdown = -0.1516
- average_turnover = 0.3618
- effective assets = 2.4645
- average max weight = 0.5978
- decision label: `reduces_concentration_and_turnover`

**Interpretation:**  
Within this paired max-weight cap experiment, the 0.60 cap produced the best
result. It increased diversification, reduced maximum weight, reduced turnover,
improved drawdown, improved Sharpe, improved `robust_score`, and improved
`mandate_aware_score`.

This contrasts sharply with the direct `lambda_concentration` experiment, where
greater diversification came at the cost of lower performance and higher
turnover.

Therefore, the evidence suggests that concentration control is more promising
as an allocation/action constraint than as a direct reward penalty.

**Baseline equivalence audit:**  
Because the uncapped baseline in the max-weight cap experiment differed from
previous V5 baselines, an additional audit was performed.

Verified:

- `max_weight_cap = None` does not patch `PortfolioEnv`.
- `apply_max_weight_cap_to_action(weights, None)` returns weights unchanged.
- `CappedPortfolioEnv` with `max_weight_cap = None` behaves identically to
  normal `PortfolioEnv` for a synthetic step.
- The uncapped cap-run config matched the concentration-penalty baseline config
  for F1 seed 7.
- Fold dates also matched.

Observed checks:

- `config_equal = True`
- `folds_equal = True`

No max-weight cap runner bug was found in the uncapped path.

The baseline difference is therefore interpreted as training/evaluation
stochasticity and experiment-level robust score normalization, not as cap leakage
into the uncapped run.

**Caveat:**  
The max-weight cap result should be interpreted internally within its paired
experiment. It should not be treated as directly interchangeable with older
baselines from separate runs.

If this result becomes central to the thesis, it should be repeated with a
larger paired validation, for example 60ep × 10seeds or 100ep × 10seeds.

**Research implication:**  
A default concentration penalty should not be activated based on current
evidence.

A max-weight cap is a more promising path for controlling TD3 concentration
because it directly constrains the allocation space rather than asking the
reward function to indirectly punish concentration.

Future experiments should test whether max-weight caps remain beneficial across:

- more seeds
- other TD3 candidates
- alternative cap levels
- final protocol comparison against benchmarks

**Tests:**  

Initial max-weight cap runner:

- `python3 -m unittest tests/test_run_max_weight_cap_experiment.py`: 10 tests OK
- `python3 -m unittest discover tests`: 921 tests OK

Baseline equivalence audit:

- `python3 -m unittest tests/test_run_max_weight_cap_experiment.py`: 13 tests OK
- `python3 -m unittest discover tests`: 924 tests OK

## Entry X — Paired Max-Weight Cap Validation: V5 60 Episodes × 10 Seeds

**Date:** 2026-05-21

**Purpose:**  
Validate whether a direct max-weight allocation constraint can control learned
TD3 concentration better than a direct reward concentration penalty.

Previous experiments showed that increasing `lambda_concentration` made the
portfolio more diversified, but reduced `robust_score`, reduced
`mandate_aware_score`, and increased turnover.

The new hypothesis was that concentration should be controlled directly at the
allocation/action layer rather than indirectly through the reward.

**Setup:**  

Candidate:

- `V5_no_volatility_block`

Experiment:

- uncapped baseline
- max weight cap = 0.60

Run configuration:

- episodes = 60
- seeds = `[7, 21, 42, 84, 101, 123, 202, 303, 404, 505]`
- folds = 4
- output directory:
  `outputs/tables/max_weight_cap_experiment_v5_60ep_10seeds_cap060`

The experiment was paired: uncapped and capped variants were produced within the
same runner and should be interpreted relative to each other.

**Uncapped baseline:**  

- annualized_return = 0.0407
- Sharpe = 0.3557
- robust_score = 0.1256
- mandate_aware_score = 0.0870
- max_drawdown = -0.2350
- average_turnover = 0.6149
- effective assets = 1.0819
- average max weight = 0.9651

**Max-weight cap 0.60:**  

- annualized_return = 0.0702
- Sharpe = 0.6833
- robust_score = 0.7021
- mandate_aware_score = 0.5627
- max_drawdown = -0.1656
- average_turnover = 0.3758
- effective assets = 2.4667
- average max weight = 0.5983

**Delta versus uncapped:**  

- annualized_return improved by 0.0295
- Sharpe improved by 0.3276
- robust_score improved by 0.5765
- mandate_aware_score improved by 0.4757
- max_drawdown improved by 0.0694
- average_turnover fell by 0.2391
- effective assets increased by 1.3848
- average max weight fell by 0.3667

Decision label:

- `reduces_concentration_and_turnover`

**Interpretation:**  
The 0.60 max-weight cap dominated the uncapped baseline inside this paired
validation. It improved diversification, reduced average max weight, reduced
turnover, improved drawdown, improved Sharpe, improved `robust_score`, improved
`mandate_aware_score`, and improved annualized return.

This contrasts with the `lambda_concentration` experiment, where diversification
came with lower performance and higher turnover.

**Research implication:**  
For V5, controlling concentration through a direct allocation constraint appears
more promising than adding a concentration penalty to the reward.

This does not yet justify making the cap a global default. The result should be
treated as a candidate protocol enhancement requiring additional validation
across other TD3 candidates and against the benchmark suite.

Next validation step:

- test cap 0.60 on `V2_reference_full`
- test cap 0.60 on `V6_financial_state`
- compare capped TD3 candidates against uncapped TD3 and benchmarks under the
  common protocol

  ## Entry X — Max-Weight Cap Validation Across V2 and V6

**Date:** 2026-05-21

**Purpose:**  
Test whether the promising V5 max-weight cap result generalizes to other TD3
candidates.

Previous evidence showed that `max_weight_cap = 0.60` improved V5 relative to
its uncapped paired baseline. The next question was whether this was specific to
V5 or whether the cap addresses a broader TD3 concentration problem.

**Setup:**  

Candidates:

- `V2_reference_full`
- `V6_financial_state`

Experiment:

- uncapped baseline
- max weight cap = 0.60

Run configuration:

- episodes = 60
- seeds = `[7, 21, 42, 84, 101, 123, 202, 303, 404, 505]`
- folds = 4

Output directories:

- `outputs/tables/max_weight_cap_experiment_v2_60ep_10seeds_cap060`
- `outputs/tables/max_weight_cap_experiment_v6_60ep_10seeds_cap060`

---

### V2 result

Uncapped baseline:

- annualized_return = -0.0007
- Sharpe = 0.0336
- robust_score = 0.1104
- mandate_aware_score = 0.0692
- max_drawdown = -0.2721
- average_turnover = 0.6456
- effective assets = 1.0963
- average max weight = 0.9594

Max-weight cap 0.60:

- annualized_return = 0.0911
- Sharpe = 0.7613
- robust_score = 0.6877
- mandate_aware_score = 0.5473
- max_drawdown = -0.1696
- average_turnover = 0.3683
- effective assets = 2.4616
- average max weight = 0.5972

Delta versus uncapped:

- annualized_return improved by 0.0918
- Sharpe improved by 0.7278
- robust_score improved by 0.5773
- mandate_aware_score improved by 0.4781
- max_drawdown improved by 0.1026
- average_turnover fell by 0.2772
- effective assets increased by 1.3652
- average max weight fell by 0.3622

Decision label:

- `reduces_concentration_and_turnover`

---

### V6 result

Uncapped baseline:

- annualized_return = 0.1868
- Sharpe = 0.3276
- robust_score = 0.1169
- mandate_aware_score = 0.0716
- max_drawdown = -0.2793
- average_turnover = 0.4643
- effective assets = 1.0658
- average max weight = 0.9720

Max-weight cap 0.60:

- annualized_return = 0.1340
- Sharpe = 0.4904
- robust_score = 0.6541
- mandate_aware_score = 0.5039
- max_drawdown = -0.1867
- average_turnover = 0.3016
- effective assets = 2.4638
- average max weight = 0.5981

Delta versus uncapped:

- annualized_return fell by 0.0528
- Sharpe improved by 0.1627
- robust_score improved by 0.5372
- mandate_aware_score improved by 0.4323
- max_drawdown improved by 0.0926
- average_turnover fell by 0.1626
- effective assets increased by 1.3980
- average max weight fell by 0.3740

Decision label:

- `reduces_concentration_and_turnover`

---

### Interpretation

The 0.60 max-weight cap improved both V2 and V6 in the dimensions most relevant
to the concentration problem:

- effective assets increased materially
- average max weight fell to approximately 60%
- turnover decreased
- max drawdown improved
- robust_score improved
- mandate_aware_score improved

For V2, the cap also improved annualized return substantially. For V6, the cap
reduced annualized return but still improved Sharpe, drawdown, turnover,
robust_score, and mandate_aware_score.

Together with the previous V5 result, this suggests that the learned TD3
concentration problem is not candidate-specific. A max-weight allocation
constraint appears to be a more promising concentration-control mechanism than
a direct concentration penalty in the reward.

**Research implication:**  
`max_weight_cap = 0.60` should be promoted to a candidate protocol enhancement
for further comparison, but not yet made a global default.

The next step is to compare capped TD3 candidates against uncapped TD3
candidates and the full benchmark suite under the common protocol.

## Entry X — Capped TD3 vs Benchmarks Protocol Comparison

**Date:** 2026-05-21

**Purpose:**  
Compare capped and uncapped TD3 candidates against the full benchmark suite
under the common protocol.

Previous experiments showed that `max_weight_cap = 0.60` improved V2, V5, and
V6 relative to their uncapped paired baselines. This entry evaluates whether
the capped TD3 candidates are also competitive against benchmarks.

**Implementation:**  
Added a reporting-only capped-vs-uncapped TD3 protocol comparison layer.

Files created:

- `src/experiments/run_capped_td3_protocol_comparison.py`
- `tests/test_run_capped_td3_protocol_comparison.py`

Output directory:

`outputs/tables/capped_td3_protocol_comparison_60ep_10seeds_cap060`

Output files:

- `capped_td3_vs_benchmarks_summary.csv`
- `capped_td3_pairwise_deltas.csv`
- `capped_td3_mandate_ranking.csv`
- `capped_td3_performance_ranking.csv`
- `capped_td3_protocol_metadata.json`

The runner is reporting-only. It regenerates protocol benchmarks, loads the
capped TD3 experiment summaries, combines them, adds mandate-aware ranking
fields, and writes pairwise capped-vs-uncapped deltas. It does not train TD3 or
modify reward, environment, or training logic.

**Top performance robust ranking:**  

The highest `robust_score` strategies remained aggressive benchmark strategies:

- `momentum_winner_12p`: `robust_score = 0.8575`, `mandate_score = 0.0000`, `max_drawdown = -0.5127`
- `Equal_Weight_Risky`: `robust_score = 0.8123`, `mandate_score = 0.0000`, `max_drawdown = -0.3672`
- `Equal_Weight`: `robust_score = 0.8004`, `mandate_score = 0.0000`, `max_drawdown = -0.3044`
- `risk_adjusted_momentum_winner_12p_12p`: `robust_score = 0.7789`, `mandate_score = 0.0000`, `max_drawdown = -0.5303`
- `rolling_markowitz_long_only_52p`: `robust_score = 0.7677`, `mandate_score = 0.0000`, `max_drawdown = -0.5380`

However, these top robust-score benchmarks are not eligible under the base
mandate because their drawdowns exceed the mandate threshold.

The best capped TD3 candidates by `robust_score` were:

- `V5_cap_0.60`: `robust_score = 0.7021`, `mandate_score = 0.5627`, `max_drawdown = -0.1656`
- `V2_cap_0.60`: `robust_score = 0.6877`, `mandate_score = 0.5473`, `max_drawdown = -0.1696`
- `V6_cap_0.60`: `robust_score = 0.6541`, `mandate_score = 0.5039`, `max_drawdown = -0.1867`

**Top mandate-aware ranking:**  

Under the recovery-based mandate-aware score, capped TD3 candidates ranked at
the top:

- `V5_cap_0.60`: `mandate_aware_score = 0.5627`
- `V2_cap_0.60`: `mandate_aware_score = 0.5473`
- `BuyHold_GLD`: `mandate_aware_score = 0.5244`
- `V6_cap_0.60`: `mandate_aware_score = 0.5039`
- `trend_spy_cash_12p`: `mandate_aware_score = 0.4841`
- `rolling_markowitz_min_variance_52p`: `mandate_aware_score = 0.4533`
- `defensive_risk_off_12p`: `mandate_aware_score = 0.4414`
- `rolling_risk_parity_inverse_vol_12p`: `mandate_aware_score = 0.4262`
- `60_40_SPY_TLT`: `mandate_aware_score = 0.3848`

**Best capped TD3 candidate:**  

The best capped TD3 candidate was:

- `V5_cap_0.60`

by mandate-aware score.

**Clean benchmark comparison:**  
`V5_cap_0.60` and `V2_cap_0.60` beat the best clean benchmark,
`BuyHold_GLD`, by mandate-aware score.

They do not beat the highest robust-score benchmarks overall, because those are
high-drawdown, non-eligible momentum-style benchmarks.

**Pairwise cap conclusions:**  

- `V2`: `cap_dominates_uncapped`
- `V5`: `cap_dominates_uncapped`
- `V6`: `cap_improves_mandate_but_hurts_return`

**Interpretation:**  
This is the strongest evidence so far that the TD3 framework becomes materially
more competitive when concentration is controlled through a direct allocation
constraint rather than through a reward penalty.

The uncapped TD3 candidates were not competitive because they learned extreme
single-asset-like concentration. With `max_weight_cap = 0.60`, the capped TD3
candidates improved diversification, drawdown behavior, turnover, and
mandate-aware ranking.

The result does not imply that TD3 dominates all benchmarks. Aggressive
momentum-style benchmarks still dominate the performance-oriented robust score.
However, once realistic drawdown mandate constraints are applied, capped TD3
becomes competitive and ranks above the best clean benchmark in this run.

**Research implication:**  
The project narrative should distinguish three layers:

1. Unconstrained TD3 does not dominate benchmarks and suffers from learned
   extreme concentration.
2. Direct concentration penalties in the reward diversify mechanically but hurt
   performance and increase turnover.
3. A direct max-weight allocation constraint, especially `max_weight_cap = 0.60`,
   materially improves TD3 mandate-aware behavior.

Therefore, `max_weight_cap = 0.60` should be treated as a candidate protocol
enhancement for the constrained TD3 portfolio allocation framework, not as a
global default yet.

Further validation should test whether the capped TD3 result remains stable
under broader seeds, alternative caps, and final paper-level robustness checks.

**Tests:**  

- `python3 -m unittest tests/test_run_capped_td3_protocol_comparison.py`: 9 tests OK
- `python3 -m unittest discover tests`: 933 tests OK

## Entry X — Executive Results Report: Capped TD3 vs Benchmarks

**Date:** 2026-05-21

**Purpose:**  
Create a paper-ready executive reporting layer for the capped TD3 protocol
comparison.

The goal was to summarize the final comparison across:

- aggressive benchmarks
- mandate-eligible benchmarks
- uncapped TD3 candidates
- capped TD3 candidates

**Implementation:**  
Added:

- `src/analysis/build_executive_results_report.py`
- `tests/test_build_executive_results_report.py`

Input directory:

- `outputs/tables/capped_td3_protocol_comparison_60ep_10seeds_cap060`

Output directory:

- `outputs/tables/executive_results_report_60ep_10seeds_cap060`

Output files:

- `executive_main_ranking.csv`
- `executive_mandate_eligible_ranking.csv`
- `executive_non_eligible_strategies.csv`
- `executive_td3_cap_impact.csv`
- `executive_strategy_groups_summary.csv`
- `executive_results_summary.md`

The module is reporting-only. It does not modify TD3 architecture, reward,
environment, training logic, `robust_score`, or `mandate_aware_score`.

**Top robust-score result:**  
The highest `robust_score` strategies remained aggressive benchmarks:

- `momentum_winner_12p`
- `Equal_Weight_Risky`
- `Equal_Weight`
- `risk_adjusted_momentum_winner_12p_12p`
- `rolling_markowitz_long_only_52p`

However, these strategies had high drawdowns and received zero mandate-aware
score under the current mandate filter.

**Top mandate-aware result:**  

The top mandate-eligible strategies were:

- `V5_cap_0.60`: `mandate_aware_score = 0.5627`
- `V2_cap_0.60`: `mandate_aware_score = 0.5473`
- `BuyHold_GLD`: `mandate_aware_score = 0.5244`
- `V6_cap_0.60`: `mandate_aware_score = 0.5039`
- `trend_spy_cash_12p`: `mandate_aware_score = 0.4841`

**TD3 cap impact:**  

- `V2_reference_full`: `cap_dominates_uncapped`
- `V5_no_volatility_block`: `cap_dominates_uncapped`
- `V6_financial_state`: `cap_improves_mandate_but_hurts_return`

The cap improved mandate-aware score, robust score, drawdown behavior, turnover,
and effective diversification across all three TD3 candidates.

**Core generated claim:**  

> TD3 does not dominate benchmarks in unconstrained form, but a max-weight
> constrained TD3 variant becomes competitive under a mandate-aware evaluation
> layer.

**Interpretation:**  
This result provides the cleanest current narrative for the project. The main
contribution is not that unconstrained TD3 dominates simple benchmarks. It does
not. The stronger and more defensible result is that TD3 requires realistic
portfolio constraints to become competitive under a mandate-aware investment
framework.

Aggressive benchmarks still dominate performance-oriented `robust_score`, but
they fail mandate-aware evaluation due to large drawdowns. Capped TD3 variants
rank at the top once drawdown recovery and mandate eligibility are considered.

**Caveat:**  
This evidence remains conditional on the current asset universe, sample window,
feature construction, TD3 implementation, cap level, and mandate-aware scoring
design. It should not be generalized as a universal TD3 dominance claim.

**Tests:**  

- `python3 -m unittest tests/test_build_executive_results_report.py`: 7 tests OK
- `python3 -m unittest discover tests`: 940 tests OK

## Entry X — Executive Results Consistency Audit

**Date:** 2026-05-21

**Purpose:**  
Audit whether the executive capped TD3 comparison is methodologically consistent
before using it as a results-layer finding.

**Implementation:**  
Added:

- `src/analysis/audit_executive_results_consistency.py`
- `tests/test_audit_executive_results_consistency.py`

Output directory:

- `outputs/tables/executive_results_consistency_audit_60ep_10seeds_cap060`

Output files:

- `executive_consistency_checks.csv`
- `executive_consistency_issues.csv`
- `executive_consistency_summary.md`

**Audit result:**  

Verdict:

- usable with caveats

Checks:

- total checks = 12
- pass = 11
- warning = 1
- fail = 0

Passed checks included:

- TD3 rows use `split == test` only
- benchmarks use the same returns/protocol
- mandate-aware formula consistency
- drawdown bucket consistency
- no duplicate strategy rows
- metric column consistency
- correct TD3 source folders
- no train/validation rows in executive report
- not-eligible strategies have zero mandate-aware score
- eligible strategies have nonzero mandate-aware score
- V5/V2 capped are genuinely above BuyHold_GLD by mandate-aware score

**Main caveat:**  
The only warning is that benchmark DSR uses `date_averaged`, while TD3 DSR uses
`median_run`. This is documented in metadata, so the comparison is usable, but
the robust-score component should be interpreted with this caveat.

**Validated key result:**  

- `V5_cap_0.60`: `mandate_aware_score = 0.562728`
- `V2_cap_0.60`: `mandate_aware_score = 0.547303`
- `BuyHold_GLD`: `mandate_aware_score = 0.524407`

Therefore, `V5_cap_0.60` and `V2_cap_0.60` are genuinely above `BuyHold_GLD`
by mandate-aware score in the final executive output.

**Interpretation:**  
The executive comparison is reliable enough for reporting, provided the DSR
aggregation caveat is disclosed. The main claim should remain mandate-aware and
constraint-focused, not a broad claim that TD3 dominates all benchmarks by
performance robust score.

**Tests:**  

- `python3 -m unittest tests/test_audit_executive_results_consistency.py`: 7 tests OK
- `python3 -m unittest discover tests`: 947 tests OK

## Entry X — Mandate-Aware Score Sensitivity Analysis

**Date:** 2026-05-21**

**Purpose:**  
Test whether the capped TD3 mandate-aware result depends too strongly on the
specific drawdown mandate thresholds used in the base evaluation.

**Implementation:**  
Added a reporting-only mandate-aware score sensitivity layer.

Files created:

- `src/analysis/mandate_score_sensitivity.py`
- `tests/test_mandate_score_sensitivity.py`

Input directory:

- `outputs/tables/capped_td3_protocol_comparison_60ep_10seeds_cap060`

Output directory:

- `outputs/tables/mandate_score_sensitivity_60ep_10seeds_cap060`

Output files:

- `mandate_score_sensitivity_all_scenarios.csv`
- `mandate_score_sensitivity_top10_by_scenario.csv`
- `mandate_score_sensitivity_td3_focus.csv`
- `mandate_score_sensitivity_summary.csv`
- `mandate_score_sensitivity_summary.md`

**Sensitivity scenarios:**  

Three mandate scenarios were tested:

- `strict`
- `base`
- `flexible`

Each scenario recalculates mandate buckets and the recovery-based
mandate-aware score. No model is retrained.

**Result:**  

The top five strategies were stable across all three scenarios:

1. `V5_cap_0.60`
2. `V2_cap_0.60`
3. `BuyHold_GLD`
4. `V6_cap_0.60`
5. `trend_spy_cash_12p`

`V5_cap_0.60` and `V2_cap_0.60` remained above `BuyHold_GLD` under all tested
mandate scenarios.

Summary:

- strict: best strategy = `V5_cap_0.60`; best benchmark = `BuyHold_GLD`; TD3 beats benchmark = True
- base: best strategy = `V5_cap_0.60`; best benchmark = `BuyHold_GLD`; TD3 beats benchmark = True
- flexible: best strategy = `V5_cap_0.60`; best benchmark = `BuyHold_GLD`; TD3 beats benchmark = True

Under the strict scenario, `V5_cap_0.60` and `V2_cap_0.60` move from
`clean_mandate` to `eligible_yellow`, but remain above `BuyHold_GLD`.

**Interpretation:**  
The capped TD3 result is not strongly dependent on the base mandate threshold.
The ranking is stable across strict, base, and flexible mandate scenarios.

Because the recovery-based multiplier is continuous, bucket changes do not alter
the score unless a strategy becomes `not_eligible`. Therefore, this sensitivity
test mainly verifies that the capped TD3 strategies remain eligible and remain
above the best clean benchmark under reasonable mandate definitions.

**Core implication:**  
The claim that capped TD3 becomes competitive under a mandate-aware evaluation
layer is more robust after this sensitivity check.

**Tests:**  

- `python3 -m unittest tests/test_mandate_score_sensitivity.py`: 8 tests OK
- `python3 -m unittest discover tests`: 955 tests OK

## Entry X — Full Max-Weight Cap Sensitivity: 60 Episodes × 10 Seeds

**Date:** 2026-05-22

**Purpose:**  
Test whether the previous `max_weight_cap = 0.60` result was a cherry-picked cap
level or whether the benefit comes more generally from imposing a reasonable
maximum-weight allocation constraint.

**Implementation:**  
Added and ran a full cap sensitivity experiment.

Files created:

- `src/experiments/run_cap_sensitivity_experiment.py`
- `tests/test_run_cap_sensitivity_experiment.py`

Output directory:

- `outputs/tables/cap_sensitivity_experiment_60ep_10seeds`

Output files:

- `cap_sensitivity_all_results.csv`
- `cap_sensitivity_pairwise_deltas.csv`
- `cap_sensitivity_best_caps.csv`
- `cap_sensitivity_summary.csv`
- `cap_sensitivity_summary.md`
- `cap_sensitivity_metadata.json`
- `per_candidate/`

Run configuration:

- candidates:
  - `V2_reference_full`
  - `V5_no_volatility_block`
  - `V6_financial_state`
- caps:
  - `uncapped`
  - `0.50`
  - `0.60`
  - `0.70`
  - `0.80`
- episodes = 60
- seeds = `[7, 21, 42, 84, 101, 123, 202, 303, 404, 505]`

Runtime:

- started: `2026-05-22 00:28:46 CEST`
- ended: `2026-05-22 04:14:49 CEST`
- total runtime: approximately `3h 46m`

**Summary:**  

| Candidate | Best mandate cap | Best mandate score | Best robust cap | Best robust score | Interpretation |
|---|---:|---:|---:|---:|---|
| `V2_reference_full` | `0.50` | `0.5482` | `0.50` | `0.6553` | `threshold_sensitive` |
| `V5_no_volatility_block` | `0.70` | `0.5294` | `0.70` | `0.6641` | `stable_cap_benefit` |
| `V6_financial_state` | `0.50` | `0.5492` | `0.50` | `0.6757` | `threshold_sensitive` |

**Top result per candidate:**  

- `V2_reference_full`: `cap_0.50` dominates uncapped.
- `V5_no_volatility_block`: `cap_0.70` dominates uncapped.
- `V6_financial_state`: `cap_0.50` dominates uncapped.

**Does 0.60 remain best?**  

No. In the broader grid:

- V2 best cap is `0.50`, not `0.60`.
- V5 best cap is `0.70`, not `0.60`.
- V6 best cap is `0.50`, not `0.60`.

**Do capped versions consistently beat uncapped?**  

- V5: yes. All tested caps beat uncapped on both `robust_score` and
  `mandate_aware_score`.
- V6: yes. All tested caps beat uncapped on both `robust_score` and
  `mandate_aware_score`.
- V2: mixed. Caps `0.50` and `0.70` beat uncapped; caps `0.60` and `0.80`
  underperform uncapped on `robust_score` and `mandate_aware_score`.

All capped variants improved turnover and effective assets versus uncapped
across all three candidates.

**Interpretation:**  
The result does not support claiming that `max_weight_cap = 0.60` is universally
optimal. Instead, the stronger conclusion is that a maximum-weight allocation
constraint improves TD3 behavior, while the best cap level is
candidate-sensitive.

This reduces the cherry-picking concern around the original `0.60` result. The
benefit appears to come from imposing a reasonable allocation constraint, not
from one isolated cap value.

**Research implication:**  
The paper/TFM narrative should avoid claiming that `0.60` is the definitive cap.
A better claim is:

> A max-weight allocation constraint materially improves TD3 behavior, although
> the optimal cap level is candidate-sensitive.

This supports using max-weight constrained TD3 as the more realistic candidate
framework for final comparison, rather than unconstrained TD3.

**Tests:**  

- `python3 -m unittest tests/test_run_cap_sensitivity_experiment.py`: 10 tests OK
- `python3 -m unittest discover tests`: 965 tests OK

## Entry X — Final Best-Constrained TD3 Report

**Date:** 2026-05-22

**Purpose:**  
Build the final reporting layer comparing the best constrained TD3 variants
against uncapped TD3, cap 0.60 reference variants, and the full benchmark suite.

This report incorporates the full cap sensitivity experiment, where the best cap
was selected by mandate-aware score for each TD3 candidate.

**Implementation:**  
Added:

- `src/analysis/build_final_constrained_td3_report.py`
- `tests/test_build_final_constrained_td3_report.py`

Output directory:

- `outputs/tables/final_constrained_td3_report_60ep_10seeds`

Output files:

- `final_constrained_td3_main_ranking.csv`
- `final_constrained_td3_mandate_ranking.csv`
- `final_constrained_td3_selected_candidates.csv`
- `final_constrained_td3_vs_benchmarks.csv`
- `final_constrained_td3_summary.md`
- `final_constrained_td3_metadata.json`

**Selected best caps:**  

| Candidate | Selected cap | Mandate-aware score | Robust score | Max drawdown |
|---|---:|---:|---:|---:|
| `V6_financial_state` | `0.50` | `0.5492` | `0.6757` | `-0.1576` |
| `V2_reference_full` | `0.50` | `0.5482` | `0.6553` | `-0.1405` |
| `V5_no_volatility_block` | `0.70` | `0.5294` | `0.6641` | `-0.1686` |

**Top mandate-aware ranking:**  

1. `V6_cap_0.50`: `0.5492`
2. `V2_cap_0.50`: `0.5482`
3. `V5_cap_0.70`: `0.5294`
4. `BuyHold_GLD`: `0.5244`
5. `trend_spy_cash_12p`: `0.4841`

**Top robust-score ranking:**  

The top robust-score strategies remain aggressive benchmarks:

1. `momentum_winner_12p`
2. `Equal_Weight_Risky`
3. `Equal_Weight`
4. `risk_adjusted_momentum_winner_12p_12p`
5. `rolling_markowitz_long_only_52p`

However, these aggressive strategies are mandate-ineligible due to high
drawdowns.

**Key comparison:**  

Best constrained TD3:

- `V6_cap_0.50`

Best clean benchmark:

- `BuyHold_GLD`

Comparison by mandate-aware score:

- `V6_cap_0.50`: `0.5492`
- `BuyHold_GLD`: `0.5244`
- Difference: `+0.0248`

`V6_cap_0.50` also beats `trend_spy_cash_12p` by mandate-aware score:

- `V6_cap_0.50`: `0.5492`
- `trend_spy_cash_12p`: `0.4841`
- Difference: `+0.0651`

**Interpretation:**  
Unconstrained TD3 does not dominate the benchmark suite. However, TD3 with an
empirically selected max-weight constraint becomes competitive under
mandate-aware evaluation and can outperform the best clean benchmark in this
experimental setting.

The optimal cap is candidate-sensitive:

- V6 prefers `0.50`
- V2 prefers `0.50`
- V5 prefers `0.70`

Therefore, the result should not be framed as "`0.60` is the best cap." The
stronger and more defensible conclusion is that max-weight constraints improve
TD3 behavior, while the optimal cap level depends on the candidate.

**Final defensible claim:**  

> Unconstrained TD3 does not dominate the benchmark suite. However, TD3 with an
> empirically selected max-weight constraint becomes competitive under
> mandate-aware evaluation and can outperform the best clean benchmark in this
> experimental setting. The optimal cap is candidate-sensitive.

**Tests:**  

- `python3 -m unittest tests/test_build_final_constrained_td3_report.py`: 8 tests OK
- `python3 -m unittest discover tests`: 973 tests OK

## Entry X — V3 Real Macro Full Revalidation

**Date:** 2026-05-22

**Purpose:**  
Evaluate `V3_real_macro_current`, the current-window macro-enhanced TD3
candidate, under the same protocol used for the other TD3 candidates.

**Setup:**  

Candidate:

- `V3_real_macro_current`

Configuration:

- episodes = 60
- seeds = `[7, 21, 42, 84, 101, 123, 202, 303, 404, 505]`
- folds = 4
- returns file = `data/processed/returns_weekly_latest.csv`
- macro file = `data/processed/macro_weekly_latest.csv`

The macro dataset covers the full current returns window and is loaded locally.
No macro data is downloaded inside training.

**Test results:**  

- mean Sharpe = `0.2155`
- robust Sharpe 0.5 = `-0.3652`
- cumulative return = `0.0321`
- annualized return = `0.0219`
- annualized volatility = `0.1860`
- max drawdown = `-0.1920`
- worst max drawdown = `-0.4790`
- average turnover = `0.3078`
- effective assets = `1.0464`
- average max weight = `0.9802`
- robust score = `0.3748`
- DSR method = `median_run`

**Benchmark comparison:**  

`V3_real_macro_current` underperformed the main benchmark set by robust score:

- `V3_real_macro_current`: `0.3748`
- `BuyHold_GLD`: `0.6967`
- `trend_spy_cash_12p`: `0.6362`
- `60_40_SPY_TLT`: `0.6084`
- `rolling_risk_parity_inverse_vol_12p`: `0.7355`

**Interpretation:**  
V3 is now technically valid as a current-window macro candidate, but the full
60ep × 10seeds revalidation does not show that macro features improve TD3
performance under the current protocol.

The candidate also remains highly concentrated:

- effective assets = `1.0464`
- average max weight = `0.9802`

Therefore, V3 should be documented as a valid but weak candidate unless further
cap-constrained testing changes the result.

**Caveat:**  
CPI still uses a conservative four-week lag rather than a full real-time
release-calendar or vintage-data treatment.

**Decision:**  
Do not promote V3 as a final unconstrained candidate. Consider cap testing only
as a secondary robustness check, not as the next priority.

## Entry X — V3 Real Macro Cap Sensitivity

**Date:** 2026-05-22

**Purpose:**  
Evaluate whether `V3_real_macro_current` remains weak after applying the same
max-weight cap sensitivity framework used for V2, V5, and V6.

The uncapped V3 full run showed weak performance and extreme concentration.
However, previous TD3 experiments showed that unconstrained TD3 candidates often
fail because they learn near single-asset allocations. Therefore, V3 should not
be rejected only from its uncapped result.

**Setup:**  

Candidate:

- `V3_real_macro_current`

Cap grid:

- `uncapped`
- `0.50`
- `0.60`
- `0.70`
- `0.80`

Configuration:

- episodes = 60
- seeds = `[7, 21, 42, 84, 101, 123, 202, 303, 404, 505]`

Output directory:

- `outputs/tables/cap_sensitivity_experiment_v3_60ep_10seeds`

**Implementation note:**  
Updated the cap experiment wrapper so the selected candidate is passed into the
feature context builder. This ensures `V3_real_macro_current` uses the guarded
macro candidate path instead of defaulting to the V2/V5/V6 feature context.

Modified:

- `src/experiments/run_max_weight_cap_experiment.py`
- `tests/test_run_max_weight_cap_experiment.py`

**Results:**  

| cap | robust_score | mandate_aware_score | Sharpe | annualized_return | max_drawdown | worst_drawdown | turnover | effective_assets | average_max_weight |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uncapped | 0.1287 | 0.0982 | 0.3291 | 0.0182 | -0.1918 | -0.5082 | 0.3827 | 1.0577 | 0.9754 |
| 0.50 | 0.7309 | 0.6287 | 0.9418 | 0.1000 | -0.1227 | -0.3191 | 0.1921 | 3.1471 | 0.4999 |
| 0.60 | 0.5265 | 0.4419 | 0.6795 | 0.0739 | -0.1384 | -0.3443 | 0.2084 | 2.4777 | 0.5987 |
| 0.70 | 0.3304 | 0.2703 | 0.4482 | 0.0384 | -0.1538 | -0.3375 | 0.3010 | 1.9462 | 0.6964 |
| 0.80 | 0.3872 | 0.3165 | 0.5943 | 0.0449 | -0.1544 | -0.5265 | 0.3172 | 1.5529 | 0.7922 |

Best V3 cap:

- by mandate-aware score: `0.50`
- by robust score: `0.50`

**Interpretation:**  
`V3_real_macro_current_cap_0.50` materially improves over uncapped V3 across
robust score, mandate-aware score, Sharpe, annualized return, max drawdown,
turnover, effective assets, and average max weight.

Against the previous best constrained TD3 candidates, V3 now appears highly
competitive:

- `V3_cap_0.50`: mandate-aware `0.6287`, robust `0.7309`
- `V6_cap_0.50`: mandate-aware `0.5492`, robust `0.6757`
- `V2_cap_0.50`: mandate-aware `0.5482`, robust `0.6553`
- `V5_cap_0.70`: mandate-aware `0.5294`, robust `0.6641`

This suggests that macro features may become useful once the TD3 allocation
space is constrained.

**Caveat:**  
The V3 macro dataset uses current-vintage macro data and a conservative CPI
four-week lag, not a full real-time vintage or release-calendar macro database.

A further consistency audit is needed because the uncapped V3 robust score in
the cap sensitivity run differs from the standalone V3 protocol revalidation.
The capped result should therefore be treated as promising but not final until
the baseline equivalence is checked.

**Tests:**  

- `python3 -m unittest tests/test_run_max_weight_cap_experiment.py`: 14 tests OK
- `python3 -m unittest tests/test_run_cap_sensitivity_experiment.py`: 10 tests OK
- `python3 -m unittest discover tests`: 986 tests OK

## Entry X — ReplayBuffer Seed Reproducibility Fix

**Date:** 2026-05-22

**Purpose:**  
Fix the reproducibility issue identified during the V3 uncapped baseline
equivalence audit.

The audit found that the standalone V3 protocol run and the V3 cap-sensitivity
uncapped baseline were methodologically comparable but not numerically
equivalent. One reason was that they were independent stochastic training
realizations. Source inspection found that `ReplayBuffer` supports seeding, but
the TD3 training paths were not passing `training_config["seed"]` into the
buffer.

**Implementation:**  
Updated both TD3 training paths so `ReplayBuffer(...)` receives the configured
training seed.

Modified:

- `src/train/train_td3.py`
- `src/experiments/run_feature_block_ablation.py`
- `tests/test_train_td3.py`
- `tests/test_run_feature_block_ablation.py`

No TD3 architecture, reward, environment dynamics, robust score,
mandate-aware score, README, or experiment outputs were changed.

**Tests:**  

- `python3 -m unittest tests/test_replay_buffer.py`: 17 OK
- `python3 -m unittest tests/test_train_td3.py`: 13 OK
- `python3 -m unittest tests/test_run_protocol_pure_td3_revalidation.py`: 10 OK
- `python3 -m unittest tests/test_run_feature_block_ablation.py`: 7 OK
- `python3 -m unittest discover tests`: 993 OK

**Reproducibility smoke:**  

Two identical in-memory TD3 runs with the same seed/config produced:

- `episode_logs_match = True`
- `replay_sample_match = True`

**Interpretation:**  
This does not retroactively make previous experiment outputs bitwise
equivalent, but it strengthens reproducibility for all future TD3 protocol and
cap-sensitivity runs.

The V3 cap-sensitivity result remains usable as an internally paired cap-grid
result, with the documented caveat that previous standalone and cap-grid
robust scores came from different scoring universes and independent stochastic
runs.

## Entry X — V3 Seeded Cap Sensitivity Rerun

**Date:** 2026-05-22

**Purpose:**  
Rerun the V3 cap sensitivity experiment after fixing ReplayBuffer seeding, to
check whether the promising V3 capped result remains stable under improved
training reproducibility.

**Setup:**  

Candidate:

- `V3_real_macro_current`

Cap grid:

- `uncapped`
- `0.50`
- `0.60`
- `0.70`
- `0.80`

Configuration:

- episodes = 60
- seeds = `[7, 21, 42, 84, 101, 123, 202, 303, 404, 505]`
- output directory:
  `outputs/tables/cap_sensitivity_experiment_v3_60ep_10seeds_seeded`

**Result:**  

| cap | robust_score | mandate_aware_score | Sharpe | annualized_return | max_drawdown | worst_drawdown | turnover | effective_assets | average_max_weight |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uncapped | 0.1372 | 0.1032 | 0.1777 | -0.0198 | -0.1984 | -0.4871 | 0.3628 | 1.0499 | 0.9785 |
| 0.50 | 0.6868 | 0.5769 | 0.6233 | 0.0611 | -0.1379 | -0.4083 | 0.1920 | 3.1472 | 0.4999 |
| 0.60 | 0.7010 | 0.5907 | 0.7987 | 0.0849 | -0.1360 | -0.4048 | 0.2461 | 2.4727 | 0.5985 |
| 0.70 | 0.3457 | 0.2763 | 0.4485 | 0.0311 | -0.1671 | -0.4544 | 0.2479 | 1.9496 | 0.6955 |
| 0.80 | 0.1852 | 0.1444 | 0.2638 | 0.0029 | -0.1806 | -0.4177 | 0.3037 | 1.5555 | 0.7913 |

Best cap:

- by mandate-aware score: `0.60`
- by robust score: `0.60`
- by max drawdown: `0.60`
- by turnover: `0.50`
- by effective assets: `0.50`

**Interpretation:**  
The seeded rerun confirms that capped V3 remains materially stronger than
uncapped V3. All tested caps improved mandate-aware score, robust score,
turnover, and effective assets versus the uncapped baseline.

The best cap changed from the previous unseeded run:

- previous best: `0.50`
- seeded best: `0.60`

This reinforces that the exact cap level is somewhat stochastic and
candidate-sensitive. However, the broader conclusion is stable: V3 benefits
substantially from a max-weight constraint.

After the seeded rerun, `V3_real_macro_current_cap_0.60` is currently the
strongest constrained TD3 candidate by mandate-aware score and robust score
among the evaluated TD3 variants.

**Caveat:**  
V3 still uses current-vintage macro data and a conservative CPI lag
approximation. It is not a full real-time vintage macro implementation.

**Decision:**  
Use `V3_real_macro_current_cap_0.60` as the current leading constrained TD3
candidate in the next final comparison report.

## Entry X — Final Constrained TD3 Report with Seeded V3

**Date:** 2026-05-22

**Purpose:**  
Update the final constrained TD3 report to include the seeded V3 cap sensitivity
result after the ReplayBuffer seeding fix.

**Implementation:**  
Updated:

- `src/analysis/build_final_constrained_td3_report.py`
- `tests/test_build_final_constrained_td3_report.py`

The report builder now accepts an optional:

- `--v3-cap-sensitivity-dir`

This allows the final constrained report to include the seeded
`V3_real_macro_current` cap sensitivity result while preserving compatibility
with the previous report when no V3 directory is provided.

Output directory:

- `outputs/tables/final_constrained_td3_report_with_v3_seeded_60ep_10seeds`

**Selected best constrained TD3 candidates:**  

| strategy | cap | mandate-aware | robust | max drawdown | turnover | effective assets |
|---|---:|---:|---:|---:|---:|---:|
| `V3_cap_0.60` | `0.60` | `0.5907` | `0.7010` | `-0.1360` | `0.2461` | `2.4727` |
| `V6_cap_0.50` | `0.50` | `0.5492` | `0.6757` | `-0.1576` | `0.2370` | `3.1407` |
| `V2_cap_0.50` | `0.50` | `0.5482` | `0.6553` | `-0.1405` | `0.3555` | `3.1109` |
| `V5_cap_0.70` | `0.70` | `0.5294` | `0.6641` | `-0.1686` | `0.4169` | `1.9457` |

**Key comparisons:**  

Against `BuyHold_GLD`:

- `V3_cap_0.60`: mandate-aware `0.5907`, robust `0.7010`
- `BuyHold_GLD`: mandate-aware `0.5244`, robust `0.6967`

Against `trend_spy_cash_12p`:

- `V3_cap_0.60`: mandate-aware `0.5907`, robust `0.7010`
- `trend_spy_cash_12p`: mandate-aware `0.4841`, robust `0.6362`

`V3_cap_0.60` beats both by mandate-aware score and also slightly beats
`BuyHold_GLD` and `trend_spy_cash_12p` by robust score.

However, it does not beat the aggressive high-drawdown benchmarks by robust
score. Strategies such as `momentum_winner_12p`, `Equal_Weight_Risky`, and
`Equal_Weight` still rank higher by robust score, but they are not
mandate-eligible due to drawdown.

**Updated final claim:**  

> After adding real macro features and applying a max-weight constraint, V3
> becomes the strongest constrained TD3 candidate in the current protocol.
> Unconstrained TD3 remains weak, but constrained TD3 with macro features is
> competitive under mandate-aware evaluation. This result remains subject to the
> macro vintage/release-timing caveat.

**Caveat:**  
V3 uses current-vintage macro data and a conservative CPI lag approximation. It
is not yet a full real-time vintage or release-calendar macro implementation.

**Tests:**  

- `python3 -m unittest tests/test_build_final_constrained_td3_report.py`: 14 OK
- `python3 -m unittest discover tests`: 999 OK