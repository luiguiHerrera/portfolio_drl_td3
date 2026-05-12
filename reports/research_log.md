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
