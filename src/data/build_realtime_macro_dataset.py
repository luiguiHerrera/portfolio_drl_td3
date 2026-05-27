"""Build a weekly real-time vintage macro dataset for V3.

This module is an explicit data-acquisition/preparation step. It is not used
inside TD3 training. The preferred input is ALFRED/FRED vintage observations
with ``realtime_start`` and ``realtime_end`` fields. If local raw vintage files
are unavailable, the CLI requires ``FRED_API_KEY`` and downloads vintage
observations from the FRED API.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.analysis.validate_v3_macro_current import load_returns_csv


FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_VINTAGE_DATES_URL = "https://api.stlouisfed.org/fred/series/vintagedates"
DEFAULT_RETURNS_PATH = "data/processed/returns_weekly_latest.csv"
DEFAULT_OUTPUT_PATH = "data/processed/macro_weekly_realtime_latest.csv"
DEFAULT_METADATA_OUTPUT = (
    "outputs/tables/v3_macro_realtime_validation/"
    "v3_macro_realtime_series_metadata.csv"
)
DEFAULT_RAW_VINTAGE_DIR = "data/raw/macro_realtime"
FRED_OPEN_END_DATE = "9999-12-31"


@dataclass(frozen=True)
class MacroSeriesConfig:
    output_name: str
    series_id: str
    allow_current_vintage_fallback: bool = False


SERIES_CONFIGS = (
    MacroSeriesConfig("DGS10", "DGS10"),
    MacroSeriesConfig("DGS2", "DGS2"),
    MacroSeriesConfig("VIX", "VIXCLS"),
    MacroSeriesConfig("DXY", "DTWEXBGS", allow_current_vintage_fallback=True),
    MacroSeriesConfig("CPI", "CPIAUCSL"),
)


def build_realtime_macro_dataset(
    returns_path: str = DEFAULT_RETURNS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    metadata_output: str = DEFAULT_METADATA_OUTPUT,
    raw_vintage_dir: str = DEFAULT_RAW_VINTAGE_DIR,
    api_key: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build weekly macro values known as of each weekly return date."""
    returns = load_returns_csv(returns_path)
    weekly_dates = returns.index
    raw_dir = Path(raw_vintage_dir)
    api_key = api_key or os.environ.get("FRED_API_KEY")

    macro_columns: dict[str, pd.Series] = {}
    provenance_frames: list[pd.DataFrame] = []
    observation_start = str((weekly_dates.min() - pd.DateOffset(years=2)).date())
    observation_end = str(weekly_dates.max().date())

    for config in SERIES_CONFIGS:
        if verbose:
            print(f"Building real-time macro series: {config.output_name} ({config.series_id})")
        local_path = raw_dir / f"{config.series_id}.csv"
        if local_path.exists():
            records, source = load_or_fetch_vintage_records(
                config=config,
                raw_vintage_dir=raw_dir,
                api_key=api_key,
                observation_start=observation_start,
                observation_end=observation_end,
            )
            selected = select_asof_weekly_values(records, weekly_dates, config, source)
        else:
            if not api_key:
                raise RuntimeError(
                    "FRED_API_KEY is required because no local raw vintage file "
                    f"was found for {config.series_id} at {local_path}. Provide "
                    "FRED_API_KEY or place ALFRED/FRED vintage CSVs with "
                    "observation_date/date, value, realtime_start, and "
                    "realtime_end columns in the raw vintage directory."
                )
            try:
                selected = fetch_fred_weekly_asof_values(
                    config=config,
                    api_key=api_key,
                    weekly_dates=weekly_dates,
                    observation_start=observation_start,
                )
            except RuntimeError as exc:
                if not config.allow_current_vintage_fallback:
                    raise
                selected = fetch_current_vintage_observation_asof_fallback(
                    config=config,
                    api_key=api_key,
                    weekly_dates=weekly_dates,
                    observation_start=observation_start,
                    fallback_reason=str(exc),
                )
        macro_columns[config.output_name] = selected.set_index("date")["value"]
        provenance_frames.append(selected)

    macro = pd.DataFrame(macro_columns, index=weekly_dates).sort_index()
    if macro.isna().any().any():
        missing = macro.isna().sum()
        raise ValueError(
            "Real-time macro dataset has missing values after as-of selection: "
            f"{missing[missing > 0].to_dict()}"
        )

    provenance = pd.concat(provenance_frames, ignore_index=True)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    macro.to_csv(destination, index_label="date")

    metadata_destination = Path(metadata_output)
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    provenance.to_csv(metadata_destination, index=False)

    return {
        "macro": macro,
        "metadata": provenance,
        "output_path": str(destination),
        "metadata_output": str(metadata_destination),
        "fred_api_key_found": bool(api_key),
    }


def load_or_fetch_vintage_records(
    config: MacroSeriesConfig,
    raw_vintage_dir: Path,
    api_key: str | None,
    observation_start: str,
    observation_end: str,
) -> tuple[pd.DataFrame, str]:
    """Load local raw vintage records or fetch them with FRED_API_KEY."""
    local_path = raw_vintage_dir / f"{config.series_id}.csv"
    if local_path.exists():
        return normalize_vintage_records(pd.read_csv(local_path), config.series_id), "local_raw_vintage"

    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is required because no local raw vintage file was found "
            f"for {config.series_id} at {local_path}. Provide FRED_API_KEY or "
            "place ALFRED/FRED vintage CSVs with observation_date/date, value, "
            "realtime_start, and realtime_end columns in the raw vintage directory."
        )

    records = fetch_fred_vintage_observations(
        series_id=config.series_id,
        api_key=api_key,
        observation_start=observation_start,
        observation_end=observation_end,
    )
    raw_vintage_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(local_path, index=False)
    return records, "fred_api_vintage"


def fetch_fred_vintage_observations(
    series_id: str,
    api_key: str,
    observation_start: str,
    observation_end: str,
    session: requests.Session | None = None,
    page_limit: int = 100000,
) -> pd.DataFrame:
    """Fetch FRED vintage observations using output_type=2 pagination."""
    client = session or requests.Session()
    vintage_dates = fetch_fred_vintage_dates(
        series_id=series_id,
        api_key=api_key,
        observation_start=observation_start,
        observation_end=observation_end,
        session=client,
    )
    if not vintage_dates:
        raise ValueError(f"FRED returned no vintage dates for {series_id}.")

    frames: list[pd.DataFrame] = []
    for vintage_chunk in _chunks(vintage_dates, 100):
        frames.extend(
            _fetch_observation_chunk(
                client=client,
                series_id=series_id,
                api_key=api_key,
                observation_start=observation_start,
                observation_end=observation_end,
                vintage_dates=vintage_chunk,
                page_limit=page_limit,
            )
        )

    if not frames:
        raise ValueError(f"FRED returned no vintage observations for {series_id}.")
    raw_observations = pd.concat(frames, ignore_index=True)
    if "realtime_start" not in raw_observations.columns:
        raw_observations = normalize_fred_output_type2_wide(
            raw_observations,
            series_id=series_id,
            vintage_dates=vintage_dates,
        )
    return normalize_vintage_records(raw_observations, series_id)


def fetch_fred_weekly_asof_values(
    config: MacroSeriesConfig,
    api_key: str,
    weekly_dates: pd.DatetimeIndex,
    observation_start: str,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch latest observation known as of each weekly date using FRED realtime."""
    client = session or requests.Session()
    vintage_dates = [str(date.date()) for date in weekly_dates]
    frames: list[pd.DataFrame] = []
    for vintage_chunk in _chunks(vintage_dates, 50):
        frames.extend(
            _fetch_observation_chunk(
                client=client,
                series_id=config.series_id,
                api_key=api_key,
                observation_start=observation_start,
                observation_end=str(weekly_dates.max().date()),
                vintage_dates=vintage_chunk,
                page_limit=100000,
            )
        )
    if not frames:
        raise ValueError(f"FRED returned no weekly as-of observations for {config.series_id}.")
    raw = pd.concat(frames, ignore_index=True)
    records = normalize_fred_output_type2_wide(
        raw,
        series_id=config.series_id,
        vintage_dates=vintage_dates,
    )
    records = normalize_vintage_records(records, config.series_id)
    return select_asof_weekly_values(records, weekly_dates, config, "fred_api_asof")


def fetch_current_vintage_observation_asof_fallback(
    config: MacroSeriesConfig,
    api_key: str,
    weekly_dates: pd.DatetimeIndex,
    observation_start: str,
    fallback_reason: str,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch current-vintage observations and use observation dates only.

    This is not true vintage data. It is used only for explicitly allowed
    series that are unavailable in ALFRED. Metadata flags every row.
    """
    client = session or requests.Session()
    raw = fetch_current_fred_observations(
        config=config,
        api_key=api_key,
        observation_start=observation_start,
        observation_end=str(weekly_dates.max().date()),
        session=client,
    )
    series = raw.set_index("observation_date")["value"].sort_index()
    rows: list[dict[str, Any]] = []
    for date in weekly_dates:
        available = series.loc[series.index <= date]
        if available.empty:
            value = pd.NA
            observation_date = pd.NaT
        else:
            value = float(available.iloc[-1])
            observation_date = available.index[-1]
        rows.append(
            {
                "date": date,
                "series_id": config.series_id,
                "output_name": config.output_name,
                "value": value,
                "observation_date_used": observation_date,
                "as_of_date": date,
                "realtime_start_used": date,
                "realtime_end_used": str(date.date()),
                "vintage_method": "current_vintage_observation_asof",
                "true_vintage_data_available": False,
                "fallback_method": "current_vintage_observation_asof",
                "fallback_used": True,
                "fallback_reason": fallback_reason,
                "source": "fred_current_vintage_fallback",
            }
        )
    return pd.DataFrame(rows)


def fetch_current_fred_observations(
    config: MacroSeriesConfig,
    api_key: str,
    observation_start: str,
    observation_end: str,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch current-vintage FRED observations for an explicit fallback series."""
    client = session or requests.Session()
    params = {
        "series_id": config.series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
        "observation_end": observation_end,
        "sort_order": "asc",
        "limit": 100000,
    }
    try:
        response = client.get(FRED_OBSERVATIONS_URL, params=params, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        detail = _sanitize_request_error(exc, api_key)
        raise RuntimeError(
            "FRED current-vintage fallback request failed for "
            f"{config.series_id}: {exc.__class__.__name__}. "
            "The request URL and API key are intentionally omitted."
            f"{detail}"
        ) from None
    observations = response.json().get("observations", [])
    frame = pd.DataFrame(observations)
    if frame.empty:
        raise ValueError(f"FRED returned no current observations for {config.series_id}.")
    frame = frame.rename(columns={"date": "observation_date"})
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"].replace(".", pd.NA), errors="coerce")
    frame = frame.dropna(subset=["observation_date", "value"])
    if frame.empty:
        raise ValueError(
            f"FRED current observations for {config.series_id} have no usable rows."
        )
    return frame.loc[:, ["observation_date", "value"]]


def _select_latest_numeric_observation(
    observations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for observation in observations:
        value = pd.to_numeric(observation.get("value"), errors="coerce")
        if pd.notna(value):
            selected = observation.copy()
            selected["value"] = value
            return selected
    return None


def fetch_fred_vintage_dates(
    series_id: str,
    api_key: str,
    observation_start: str,
    observation_end: str,
    session: requests.Session | None = None,
    page_limit: int = 10000,
) -> list[str]:
    """Fetch vintage dates for a FRED series with pagination."""
    client = session or requests.Session()
    offset = 0
    vintage_dates: list[str] = []
    while True:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "realtime_start": observation_start,
            "realtime_end": observation_end,
            "limit": page_limit,
            "offset": offset,
        }
        try:
            response = client.get(FRED_VINTAGE_DATES_URL, params=params, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = _sanitize_request_error(exc, api_key)
            raise RuntimeError(
                "FRED vintage-date request failed for "
                f"{series_id}: {exc.__class__.__name__}. "
                "The request URL and API key are intentionally omitted."
                f"{detail}"
            ) from None
        payload = response.json()
        dates = payload.get("vintage_dates", [])
        vintage_dates.extend(str(date) for date in dates)
        offset += len(dates)
        total_count = int(payload.get("count", offset))
        if not dates or offset >= total_count:
            break
    return vintage_dates


def _fetch_observation_chunk(
    client: requests.Session,
    series_id: str,
    api_key: str,
    observation_start: str,
    observation_end: str,
    vintage_dates: list[str],
    page_limit: int,
) -> list[pd.DataFrame]:
    offset = 0
    frames: list[pd.DataFrame] = []
    while True:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": observation_start,
            "observation_end": observation_end,
            "vintage_dates": ",".join(vintage_dates),
            "output_type": 2,
            "sort_order": "asc",
            "limit": page_limit,
            "offset": offset,
        }
        try:
            response = client.get(FRED_OBSERVATIONS_URL, params=params, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = _sanitize_request_error(exc, api_key)
            raise RuntimeError(
                "FRED API request failed for "
                f"{series_id}: {exc.__class__.__name__}. "
                "The request URL and API key are intentionally omitted."
                f"{detail}"
            ) from None
        payload = response.json()
        observations = payload.get("observations", [])
        if not observations:
            break
        frames.append(pd.DataFrame(observations))
        offset += len(observations)
        total_count = int(payload.get("count", offset))
        if offset >= total_count:
            break
    return frames


def _chunks(values: list[str], chunk_size: int) -> list[list[str]]:
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def normalize_vintage_records(raw: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Normalize local/API vintage records to typed columns."""
    frame = raw.copy()
    if "observation_date" not in frame.columns:
        if "date" in frame.columns:
            frame = frame.rename(columns={"date": "observation_date"})
        else:
            raise KeyError(
                f"Vintage records for {series_id} need observation_date or date."
            )
    required = {"observation_date", "value", "realtime_start", "realtime_end"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            f"Vintage records for {series_id} are missing columns: "
            f"{', '.join(sorted(missing))}"
        )

    frame = frame.loc[:, ["observation_date", "value", "realtime_start", "realtime_end"]]
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    frame["realtime_start"] = pd.to_datetime(frame["realtime_start"], errors="coerce")
    frame["realtime_end_raw"] = frame["realtime_end"].astype(str)
    frame["realtime_end"] = frame["realtime_end"].map(_parse_realtime_end)
    frame["value"] = pd.to_numeric(frame["value"].replace(".", pd.NA), errors="coerce")
    frame = frame.dropna(
        subset=["observation_date", "value", "realtime_start", "realtime_end"]
    )
    frame = frame.sort_values(["observation_date", "realtime_start"])
    if frame.empty:
        raise ValueError(f"Vintage records for {series_id} have no usable rows.")
    return frame


def normalize_fred_output_type2_wide(
    raw: pd.DataFrame,
    series_id: str,
    vintage_dates: list[str],
) -> pd.DataFrame:
    """Convert FRED output_type=2 wide vintage columns to long records."""
    if "date" not in raw.columns:
        raise KeyError(f"FRED output_type=2 records for {series_id} need a date column.")
    value_columns = [
        column
        for column in raw.columns
        if isinstance(column, str) and column.startswith(f"{series_id}_")
    ]
    if not value_columns:
        raise KeyError(
            f"FRED output_type=2 records for {series_id} have no {series_id}_YYYYMMDD columns."
        )
    vintage_end_by_start = _build_vintage_end_map(vintage_dates)
    long = raw.melt(
        id_vars=["date"],
        value_vars=value_columns,
        var_name="vintage_column",
        value_name="value",
    )
    long["realtime_start"] = long["vintage_column"].map(
        lambda value: _parse_vintage_column_date(str(value), series_id)
    )
    long["realtime_end"] = long["realtime_start"].map(vintage_end_by_start)
    return long.rename(columns={"date": "observation_date"}).loc[
        :,
        ["observation_date", "value", "realtime_start", "realtime_end"],
    ]


def _build_vintage_end_map(vintage_dates: list[str]) -> dict[pd.Timestamp, str]:
    parsed = sorted(pd.Timestamp(date) for date in vintage_dates)
    result: dict[pd.Timestamp, str] = {}
    for index, start in enumerate(parsed):
        if index + 1 < len(parsed):
            result[start] = str((parsed[index + 1] - pd.Timedelta(days=1)).date())
        else:
            result[start] = FRED_OPEN_END_DATE
    return result


def _parse_vintage_column_date(column: str, series_id: str) -> pd.Timestamp:
    suffix = column.removeprefix(f"{series_id}_")
    return pd.to_datetime(suffix, format="%Y%m%d")


def select_asof_weekly_values(
    records: pd.DataFrame,
    weekly_dates: pd.DatetimeIndex,
    config: MacroSeriesConfig,
    source: str,
) -> pd.DataFrame:
    """Select the latest observation available for each weekly as-of date."""
    rows: list[dict[str, Any]] = []
    for date in weekly_dates:
        available = records[
            (records["observation_date"] <= date)
            & (records["realtime_start"] <= date)
            & (records["realtime_end"] >= date)
        ]
        if available.empty:
            rows.append(
                {
                    "date": date,
                    "series_id": config.series_id,
                    "output_name": config.output_name,
                    "value": pd.NA,
                    "observation_date_used": pd.NaT,
                    "as_of_date": date,
                    "realtime_start_used": pd.NaT,
                    "realtime_end_used": pd.NaT,
                    "vintage_method": source,
                    "true_vintage_data_available": _is_true_vintage_source(source),
                    "fallback_method": "",
                    "fallback_used": False,
                    "source": source,
                }
            )
            continue
        selected = available.sort_values(["observation_date", "realtime_start"]).iloc[-1]
        rows.append(
            {
                "date": date,
                "series_id": config.series_id,
                "output_name": config.output_name,
                "value": float(selected["value"]),
                "observation_date_used": selected["observation_date"],
                "as_of_date": date,
                "realtime_start_used": selected["realtime_start"],
                "realtime_end_used": selected["realtime_end_raw"],
                "vintage_method": source,
                "true_vintage_data_available": _is_true_vintage_source(source),
                "fallback_method": "",
                "fallback_used": False,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def _parse_realtime_end(value: object) -> pd.Timestamp:
    text = str(value)
    if text == FRED_OPEN_END_DATE:
        return pd.Timestamp.max.normalize()
    return pd.Timestamp(value)


def _is_true_vintage_source(source: str) -> bool:
    return source in {"fred_api_vintage", "fred_api_asof", "local_raw_vintage"}


def _sanitize_request_error(exc: requests.RequestException, api_key: str) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return ""
    status = getattr(response, "status_code", "unknown")
    text = getattr(response, "text", "") or ""
    if api_key:
        text = text.replace(api_key, "[redacted]")
    text = text.replace("\n", " ").strip()
    if len(text) > 300:
        text = text[:300] + "..."
    return f" Status={status}. Response={text}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build weekly real-time vintage macro data for V3.",
    )
    parser.add_argument("--returns-path", default=DEFAULT_RETURNS_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metadata-output", default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--raw-vintage-dir", default=DEFAULT_RAW_VINTAGE_DIR)
    args = parser.parse_args()

    try:
        result = build_realtime_macro_dataset(
            returns_path=args.returns_path,
            output_path=args.output_path,
            metadata_output=args.metadata_output,
            raw_vintage_dir=args.raw_vintage_dir,
            verbose=True,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    macro = result["macro"]
    metadata = result["metadata"]
    print("output_path:", result["output_path"])
    print("metadata_output:", result["metadata_output"])
    print("fred_api_key_found:", result["fred_api_key_found"])
    print("coverage:", macro.index.min(), "to", macro.index.max())
    print("shape:", macro.shape)
    print("missing_values:", int(macro.isna().sum().sum()))
    print("series:")
    print(
        metadata.groupby(["output_name", "series_id", "vintage_method", "source"])
        .size()
        .reset_index(name="rows")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
