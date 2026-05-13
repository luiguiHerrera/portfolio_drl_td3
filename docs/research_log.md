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
