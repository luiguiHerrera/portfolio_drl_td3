"""Build local weekly macro datasets for Feature Set V3.

This module intentionally reads only local CSV files. It does not download
macro data or call external APIs.
"""

from pathlib import Path

import pandas as pd


RAW_TO_OUTPUT_COLUMNS = {
    "DGS10": "DGS10",
    "DGS2": "DGS2",
    "VIXCLS": "VIX",
    "DTWEXBGS": "DXY",
    "CPIAUCSL": "CPI",
}

DAILY_SERIES = ("DGS10", "DGS2", "VIXCLS", "DTWEXBGS")
CPI_SERIES = "CPIAUCSL"


def build_weekly_macro_dataset(
    raw_macro_dir: str,
    output_path: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    weekly_frequency: str = "W-FRI",
    cpi_lag_weeks: int = 4,
) -> pd.DataFrame:
    """Build one weekly macro DataFrame from local raw macro CSV files."""
    _validate_builder_inputs(weekly_frequency, cpi_lag_weeks)

    raw_dir = Path(raw_macro_dir)
    weekly_series = {}

    for raw_name in DAILY_SERIES:
        series = load_raw_macro_series(raw_dir, raw_name)
        output_name = RAW_TO_OUTPUT_COLUMNS[raw_name]
        weekly_series[output_name] = align_daily_series_to_weekly(
            series,
            weekly_frequency,
        )

    cpi = load_raw_macro_series(raw_dir, CPI_SERIES)
    weekly_series[RAW_TO_OUTPUT_COLUMNS[CPI_SERIES]] = (
        align_cpi_to_weekly_with_lag(
            cpi,
            weekly_frequency=weekly_frequency,
            cpi_lag_weeks=cpi_lag_weeks,
        )
    )

    macro = pd.concat(
        _reindex_to_common_weekly_index(weekly_series, weekly_frequency),
        axis=1,
    )
    macro = macro.sort_index()

    if start_date is not None:
        macro = macro.loc[macro.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        macro = macro.loc[macro.index <= pd.Timestamp(end_date)]

    macro = macro.dropna()
    if macro.empty:
        raise ValueError("Weekly macro dataset is empty after alignment and filtering.")

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        macro.to_csv(destination, index_label="date")

    return macro


def _reindex_to_common_weekly_index(
    weekly_series: dict[str, pd.Series],
    weekly_frequency: str,
) -> dict[str, pd.Series]:
    starts = [series.index.min() for series in weekly_series.values()]
    ends = [series.index.max() for series in weekly_series.values()]
    common_index = pd.date_range(
        start=min(starts),
        end=max(ends),
        freq=weekly_frequency,
    )

    return {
        name: series.reindex(common_index).ffill()
        for name, series in weekly_series.items()
    }


def load_raw_macro_series(raw_macro_dir: str | Path, series_name: str) -> pd.Series:
    """Load one raw local macro CSV with date,value columns."""
    path = Path(raw_macro_dir) / f"{series_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw macro file: {path}")

    raw = pd.read_csv(path)
    required_columns = {"date", "value"}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Raw macro file {path} is missing columns: {missing}")

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["date", "value"])
    if raw.empty:
        raise ValueError(f"Raw macro file {path} has no usable observations.")

    raw = raw.set_index("date").sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]

    series = raw["value"].rename(series_name)
    if series.empty:
        raise ValueError(f"Raw macro file {path} has no usable observations.")

    return series


def align_daily_series_to_weekly(
    series: pd.Series,
    weekly_frequency: str = "W-FRI",
) -> pd.Series:
    """Align daily or market-frequency observations to weekly Fridays."""
    _validate_weekly_frequency(weekly_frequency)

    return series.sort_index().resample(weekly_frequency).last().ffill()


def align_cpi_to_weekly_with_lag(
    series: pd.Series,
    weekly_frequency: str = "W-FRI",
    cpi_lag_weeks: int = 4,
) -> pd.Series:
    """Align CPI to weekly Fridays after applying a simple availability lag.

    CPI is monthly and should not be treated as known on the observation date.
    This implementation shifts availability forward by a configurable number of
    weeks before weekly alignment. It is a conservative approximation, not a
    full release-calendar model.
    """
    _validate_weekly_frequency(weekly_frequency)
    _validate_cpi_lag_weeks(cpi_lag_weeks)

    lagged = series.sort_index().copy()
    lagged.index = lagged.index + pd.DateOffset(weeks=cpi_lag_weeks)

    return lagged.resample(weekly_frequency).last().ffill()


def _validate_builder_inputs(
    weekly_frequency: str,
    cpi_lag_weeks: int,
) -> None:
    _validate_weekly_frequency(weekly_frequency)
    _validate_cpi_lag_weeks(cpi_lag_weeks)


def _validate_weekly_frequency(weekly_frequency: str) -> None:
    if not isinstance(weekly_frequency, str) or not weekly_frequency:
        raise ValueError("weekly_frequency must be a non-empty string.")


def _validate_cpi_lag_weeks(cpi_lag_weeks: int) -> None:
    if not isinstance(cpi_lag_weeks, int) or cpi_lag_weeks < 0:
        raise ValueError("cpi_lag_weeks must be an integer >= 0.")
