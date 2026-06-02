# Macro Vintage Alignment Audit

Date: 2026-06-02

## Scope

This audit checks whether the clean macro candidates are genuinely safe to use
as real-time/as-of macro specifications before final corrected reruns.

Audited candidates:

- `V3_real_macro_vintage_clean_no_dxy`
- `V7_real_macro_vintage_clean_no_dxy_garch`

Both candidates use:

- macro input: `data/processed/macro_weekly_realtime_clean_latest.csv`
- macro metadata sidecar:
  `outputs/tables/v3_macro_realtime_clean_validation/v3_macro_realtime_series_metadata.csv`

The clean dataset contains:

- `DGS10`
- `DGS2`
- `VIX`
- `CPI`

It excludes `DXY`.

## Files Inspected

Core data construction:

- `src/data/build_realtime_macro_dataset.py`
- `src/analysis/validate_v3_macro_realtime.py`
- `src/data/macro_loader.py`

Feature construction:

- `src/data/features_v3.py`
- `src/data/features_v7.py`
- `src/data/feature_factory.py`
- `src/experiments/run_protocol_pure_td3_revalidation.py`
- `src/experiments/run_feature_block_ablation.py`
- `src/data/prepare_dataset.py`

Validation/tests:

- `tests/test_build_realtime_macro_dataset.py`
- `tests/test_validate_v3_macro_realtime.py`
- `tests/test_features_v3.py`
- `tests/test_features_v7.py`
- `tests/test_run_protocol_pure_td3_revalidation.py`

Processed files:

- `data/processed/macro_weekly_realtime_clean_latest.csv`
- `outputs/tables/v3_macro_realtime_clean_validation/v3_macro_realtime_series_metadata.csv`
- `outputs/tables/v3_macro_realtime_clean_validation/v3_macro_realtime_leakage_checks.csv`
- `outputs/tables/v3_macro_realtime_clean_validation/v3_macro_realtime_freshness_checks.csv`

## Macro Inputs Used

`V3_real_macro_vintage_clean_no_dxy` is wired in
`src/experiments/run_protocol_pure_td3_revalidation.py` with:

- `features.version = v3`
- `macro_path = data/processed/macro_weekly_realtime_clean_latest.csv`
- `macro_date_column = date`
- `macro_source = realtime_asof_no_dxy_no_fallback`
- `dollar_proxy = excluded`

`V7_real_macro_vintage_clean_no_dxy_garch` is wired with the same macro file
and source metadata, plus rolling fitted GARCH features.

No macro download occurs inside TD3 training.

## Date Fields Found

The processed macro CSV used by training contains only:

- `date`
- `DGS10`
- `DGS2`
- `VIX`
- `CPI`

It does not contain traceability fields such as `observation_date_used`,
`as_of_date`, `realtime_start_used`, `release_date`, or `availability_date`.

The metadata sidecar contains:

- `date`
- `series_id`
- `feature_name`
- `output_name`
- `title`
- `source`
- `conceptual_role`
- `frequency`
- `observation_date_used`
- `as_of_date`
- `realtime_start_used`
- `realtime_end_used`
- `vintage_method`
- `true_vintage_data_available`
- `fallback_method`
- `fallback_used`
- `realtime_end_parsed`

There is no explicit BLS/FRED release-calendar `release_date` column.
Availability is represented by FRED realtime/as-of vintage selection.

## Series-Level Status

From the clean validation metadata:

| Series | Source series | Frequency | Vintage method | Fallback | Latest observation at 2026-05-15 | Assessment |
|---|---:|---|---|---:|---|---|
| `DGS10` | `DGS10` | daily | `fred_api_asof` | false | 2026-05-14 | as-of safe for level value |
| `DGS2` | `DGS2` | daily | `fred_api_asof` | false | 2026-05-14 | as-of safe for level value |
| `VIX` | `VIXCLS` | daily | `fred_api_asof` | false | 2026-05-15 | as-of safe for level value |
| `CPI` | `CPIAUCSL` | monthly | `fred_api_asof` | false | 2026-04-01 | as-of safe by FRED vintage, but no explicit release-date column |

Validation checks passed:

- `as_of_date <= weekly date`
- `observation_date_used <= weekly date`
- `realtime_start_used <= weekly date`
- weekly date inside vintage interval
- endpoint freshness
- no fallback rows
- no missing values

## Alignment Method

The realtime macro builder selects the latest observation known as of each
weekly return date. For FRED API rows, it requests observations at weekly
vintage/as-of dates and selects observations with `observation_date <= weekly
date`.

After loading the processed CSV, `src/data/features_v3.py` aligns macro values
with:

```python
aligned_macro_data = sorted_macro_data.reindex(target_index, method="ffill")
```

For the clean realtime file this forward fill is mostly redundant because the
file is already weekly on the return index, but it would be unsafe if the input
index were an economic observation date rather than an availability/as-of date.

The project-wide training alignment then applies:

```python
features_available_before_return = raw_features.shift(1).dropna()
```

This means a return period at date `t` uses the feature row from `t-1`.
For already-as-of weekly macro values this is conservative.

## CPI Audit

### Is CPI Indexed by Observation Month or Availability Date?

The processed training CSV is indexed by weekly as-of feature date. The sidecar
metadata shows the economic CPI observation month separately in
`observation_date_used`.

Example:

- weekly feature date: 2020-04-10
- selected CPI observation date: 2020-03-01
- as-of date: 2020-04-10
- realtime start used: 2020-04-10

So the clean processed value is not simply indexed by the observed CPI month.
It is weekly as-of selected.

### Is CPI Release-Date Safe?

Partially.

The current builder uses FRED realtime/as-of vintage dates, which is a valid
as-of availability mechanism for the value returned by FRED at a weekly
decision date. The validation confirms that CPI values are not selected with a
future `realtime_start_used` or future `observation_date_used`.

However, there is no explicit `release_date` or BLS release-calendar field.
Therefore the current dataset is best described as FRED real-time/as-of safe,
not as a fully release-calendar audited CPI implementation.

### Is `macro_cpi_momentum_12p` YoY Inflation?

No.

`src/data/features_v3.py` computes:

```python
cpi_momentum = aligned_macro_data["CPI"].pct_change(12)
```

This calculation happens after CPI has been aligned to a weekly index.
Therefore `pct_change(12)` means twelve weekly feature rows, not twelve monthly
CPI observations.

The feature should be interpreted as a 12-week CPI momentum/proxy feature. It
should not be described as YoY inflation.

If YoY inflation is intended, it should be computed on the monthly CPI series
before weekly as-of alignment, using only values available as of each decision
date.

## Candidate Assessment

| Candidate | Level macro values | CPI release traceability | CPI derived feature | Overall status |
|---|---|---|---|---|
| `V3_real_macro_vintage_clean_no_dxy` | as-of safe via FRED vintage metadata | no explicit release-date field | 12-week momentum, not YoY | partially safe but not auditable enough as final evidence |
| `V7_real_macro_vintage_clean_no_dxy_garch` | same macro path as V3 plus validated GARCH | same CPI limitation | same CPI limitation | partially safe but not auditable enough as final evidence |

This is not a reason to drop V3/V7. It is a reason to fix and harden the clean
macro pipeline before final corrected reruns.

## Recommendation for Final Rerun

Do not exclude V3/V7 as the easy solution.

Recommended status:

> Current clean macro is partially safe but not auditable enough. Add
> traceability metadata/tests and correct or relabel the CPI derived feature
> before final corrected reruns. Then keep V3/V7 as intended final candidates.

Required fixes before rerun:

1. Preserve auditable date fields in either the processed macro feature file or
   an explicitly required sidecar:
   - `decision_date` / weekly `feature_date`
   - `observation_date_used`
   - `as_of_date`
   - `realtime_start_used` / `vintage_date`
   - `realtime_end_used`
   - optional explicit `release_date` or `availability_date`

2. Make feature construction require this provenance for
   `realtime_asof_no_dxy_no_fallback` candidates, or validate that the sidecar
   exactly matches the processed weekly macro file.

3. For CPI, choose one of two clean specifications:
   - compute true monthly YoY CPI change before weekly alignment, using only
     as-of available monthly CPI values; or
   - keep the current weekly calculation but rename/document it as
     `macro_cpi_momentum_12w` or `macro_cpi_momentum_12p_weekly_proxy`.

4. Add tests that fail if:
   - CPI is used before its FRED realtime/as-of availability date;
   - a weekly `pct_change(12)` CPI feature is labeled as YoY;
   - clean macro features lack auditable date fields or matching sidecar
     metadata;
   - `feature_date` is earlier than `availability_date`;
   - a decision date uses information not available before that decision.

5. Update validation outputs to explicitly state:
   - DGS10/DGS2/VIX are daily FRED as-of/market-observable proxies;
   - CPI is monthly FRED as-of, not a release-calendar table unless release
     dates are added;
   - DXY is excluded with no fallback.

## Final Audit Conclusion

True as-of safety for the macro level values is supported by the current
metadata sidecar and validation checks.

Full final-paper auditability is not yet strong enough because the processed
training CSV does not carry trace fields and CPI lacks an explicit release-date
column.

The current CPI momentum feature is a twelve-week weekly-aligned momentum
proxy, not YoY inflation.

V3/V7 should remain in the final corrected rerun plan, but the clean macro
pipeline should be corrected or hardened first. Existing macro-based outputs
should be treated as provisional until that correction is complete.
