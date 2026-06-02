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
import time
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
FRED_REQUEST_RETRIES = 5
FRED_RATE_LIMIT_SLEEP_SECONDS = 20.0


@dataclass(frozen=True)
class MacroSeriesConfig:
    output_name: str
    series_id: str
    title: str
    data_source: str
    conceptual_role: str
    frequency: str
    note: str = ""


SERIES_CONFIGS = (
    MacroSeriesConfig(
        "DGS10",
        "DGS10",
        "10-Year Treasury Constant Maturity Rate",
        "FRED/Federal Reserve H.15",
        "long_rate",
        "daily",
    ),
    MacroSeriesConfig(
        "DGS2",
        "DGS2",
        "2-Year Treasury Constant Maturity Rate",
        "FRED/Federal Reserve H.15",
        "short_rate",
        "daily",
    ),
    MacroSeriesConfig(
        "VIX",
        "VIXCLS",
        "CBOE Volatility Index: VIX",
        "FRED/CBOE",
        "equity_volatility_proxy",
        "daily",
    ),
    MacroSeriesConfig(
        "DXY",
        "DTWEXBGS",
        "Nominal Broad U.S. Dollar Index",
        "FRED/Federal Reserve H.10",
        "dollar_strength_proxy",
        "daily",
        "This is not ICE DXY/USDX; it is the Fed nominal broad trade-weighted U.S. dollar index.",
    ),
    MacroSeriesConfig(
        "CPI",
        "CPIAUCSL",
        "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average",
        "FRED/Bureau of Labor Statistics",
        "inflation_proxy",
        "monthly",
    ),
)


def build_realtime_macro_dataset(
    returns_path: str = DEFAULT_RETURNS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    metadata_output: str = DEFAULT_METADATA_OUTPUT,
    raw_vintage_dir: str = DEFAULT_RAW_VINTAGE_DIR,
    api_key: str | None = None,
    dollar_series_id: str | None = None,
    dollar_column_name: str | None = None,
    exclude_series: tuple[str, ...] | list[str] | None = None,
    require_no_fallback: bool = False,
    vintage_chunk_size: int = 50,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build weekly macro values known as of each weekly return date."""
    if vintage_chunk_size < 1:
        raise ValueError("vintage_chunk_size must be >= 1.")
    returns = load_returns_csv(returns_path)
    weekly_dates = returns.index
    raw_dir = Path(raw_vintage_dir)
    api_key = api_key or os.environ.get("FRED_API_KEY")
    series_configs = resolve_series_configs(
        dollar_series_id=dollar_series_id,
        dollar_column_name=dollar_column_name,
        exclude_series=exclude_series,
    )

    macro_columns: dict[str, pd.Series] = {}
    provenance_frames: list[pd.DataFrame] = []
    observation_start = str((weekly_dates.min() - pd.DateOffset(years=2)).date())
    observation_end = str(weekly_dates.max().date())

    preflight_fred_vintage_coverage(
        series_configs=series_configs,
        api_key=api_key,
        weekly_dates=weekly_dates,
        observation_start=observation_start,
        observation_end=observation_end,
        raw_vintage_dir=raw_dir,
        verbose=verbose,
    )

    for config in series_configs:
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
            selected, records = fetch_fred_weekly_asof_values(
                config=config,
                api_key=api_key,
                weekly_dates=weekly_dates,
                observation_start=observation_start,
                vintage_chunk_size=vintage_chunk_size,
                verbose=verbose,
                return_records=True,
            )
        validate_asof_endpoint_usability(
            selected=selected,
            config=config,
            required_start=weekly_dates.min(),
            required_end=weekly_dates.max(),
        )
        macro_columns[config.output_name] = selected.set_index("date")["value"]
        provenance_frames.append(selected)
        if config.output_name == "CPI":
            cpi_yoy = build_cpi_yoy_asof_metadata(
                selected=selected,
                records=records,
                config=config,
                source=selected["vintage_method"].dropna().astype(str).iloc[0],
            )
            macro_columns["cpi_yoy_asof"] = cpi_yoy.set_index("date")["value"]
            provenance_frames.append(cpi_yoy)

    macro = pd.DataFrame(macro_columns, index=weekly_dates).sort_index()
    if macro.isna().any().any():
        missing = macro.isna().sum()
        raise ValueError(
            "Real-time macro dataset has missing values after as-of selection: "
            f"{missing[missing > 0].to_dict()}"
        )

    provenance = pd.concat(provenance_frames, ignore_index=True)
    if require_no_fallback:
        fallback_count = int(provenance["fallback_used"].astype(bool).sum())
        true_vintage_all = bool(
            provenance["true_vintage_data_available"].astype(bool).all()
        )
        if fallback_count > 0 or not true_vintage_all:
            raise ValueError(
                "Real-time macro build requires true vintage data with no fallback, "
                f"but found fallback rows={fallback_count}, "
                f"true_vintage_all={true_vintage_all}."
            )

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


def preflight_fred_vintage_coverage(
    series_configs: tuple[MacroSeriesConfig, ...],
    api_key: str | None,
    weekly_dates: pd.DatetimeIndex,
    observation_start: str,
    observation_end: str,
    raw_vintage_dir: Path,
    verbose: bool = False,
) -> None:
    """Check online vintage coverage before downloading observations."""
    if not api_key:
        return
    client = requests.Session()
    for config in series_configs:
        local_path = raw_vintage_dir / f"{config.series_id}.csv"
        if local_path.exists():
            continue
        if verbose:
            print(
                f"Preflight vintage coverage: {config.output_name} ({config.series_id})",
                flush=True,
            )
        vintage_dates = fetch_fred_vintage_dates(
            series_id=config.series_id,
            api_key=api_key,
            observation_start=observation_start,
            observation_end=observation_end,
            session=client,
        )
        validate_vintage_coverage(
            series_id=config.series_id,
            vintage_dates=vintage_dates,
            required_start=weekly_dates.min(),
            required_end=weekly_dates.max(),
        )


def resolve_series_configs(
    dollar_series_id: str | None = None,
    dollar_column_name: str | None = None,
    exclude_series: tuple[str, ...] | list[str] | None = None,
) -> tuple[MacroSeriesConfig, ...]:
    """Return macro series configs with an optional explicit dollar proxy."""
    excluded = {str(value).strip() for value in (exclude_series or []) if str(value).strip()}
    if dollar_series_id is None and dollar_column_name is None:
        return tuple(
            config
            for config in SERIES_CONFIGS
            if config.output_name not in excluded and config.series_id not in excluded
        )
    dollar_series_id = dollar_series_id or "DTWEXBGS"
    dollar_column_name = dollar_column_name or "DXY"
    if dollar_series_id != "DTWEXBGS":
        raise ValueError(
            "Only DTWEXBGS is currently supported as the real-time vintage "
            "dollar-strength proxy."
        )
    if dollar_column_name != "DXY":
        raise ValueError("The dollar-strength output column must remain DXY.")

    configs = []
    for config in SERIES_CONFIGS:
        if config.output_name == "DXY":
            configs.append(
                MacroSeriesConfig(
                    output_name=dollar_column_name,
                    series_id=dollar_series_id,
                    title="Nominal Broad U.S. Dollar Index",
                    data_source="FRED/Federal Reserve H.10",
                    conceptual_role="dollar_strength_proxy",
                    frequency="daily",
                    note=(
                        "This is not ICE DXY/USDX; it is the Fed nominal broad "
                        "trade-weighted U.S. dollar index."
                    ),
                )
            )
        else:
            configs.append(config)
    return tuple(
        config
        for config in configs
        if config.output_name not in excluded and config.series_id not in excluded
    )


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

    record_frames: list[pd.DataFrame] = []
    for vintage_chunk in _chunks(vintage_dates, 100):
        raw_frames = (
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
        if not raw_frames:
            continue
        raw_chunk = pd.concat(raw_frames, ignore_index=True)
        if "realtime_start" not in raw_chunk.columns:
            raw_chunk = normalize_fred_output_type2_wide(
                raw_chunk,
                series_id=series_id,
                vintage_dates=vintage_chunk,
            )
        record_frames.append(normalize_vintage_records(raw_chunk, series_id))

    if not record_frames:
        raise ValueError(f"FRED returned no vintage observations for {series_id}.")
    return pd.concat(record_frames, ignore_index=True)


def fetch_fred_weekly_asof_values(
    config: MacroSeriesConfig,
    api_key: str,
    weekly_dates: pd.DatetimeIndex,
    observation_start: str,
    session: requests.Session | None = None,
    vintage_chunk_size: int = 50,
    verbose: bool = False,
    return_records: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch latest observation known as of each weekly date using FRED realtime."""
    if vintage_chunk_size < 1:
        raise ValueError("vintage_chunk_size must be >= 1.")
    client = session or requests.Session()
    vintage_dates = [str(date.date()) for date in weekly_dates]
    available_vintage_dates = fetch_fred_vintage_dates(
        series_id=config.series_id,
        api_key=api_key,
        observation_start=observation_start,
        observation_end=str(weekly_dates.max().date()),
        session=client,
    )
    validate_vintage_coverage(
        series_id=config.series_id,
        vintage_dates=available_vintage_dates,
        required_start=weekly_dates.min(),
        required_end=weekly_dates.max(),
    )
    record_frames: list[pd.DataFrame] = []
    raw_row_count = 0
    vintage_chunks = _chunks(vintage_dates, vintage_chunk_size)
    for index, vintage_chunk in enumerate(vintage_chunks, start=1):
        if verbose:
            print(
                f"  {config.output_name} ({config.series_id}) as-of chunk "
                f"{index}/{len(vintage_chunks)}: {vintage_chunk[0]} to {vintage_chunk[-1]}",
                flush=True,
            )
        raw_frames = (
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
        if not raw_frames:
            continue
        raw_chunk = pd.concat(raw_frames, ignore_index=True)
        raw_row_count += len(raw_chunk)
        chunk_records = normalize_fred_output_type2_wide(
            raw_chunk,
            series_id=config.series_id,
            vintage_dates=vintage_chunk,
        )
        record_frames.append(normalize_vintage_records(chunk_records, config.series_id))
    if not record_frames:
        raise ValueError(f"FRED returned no weekly as-of observations for {config.series_id}.")
    if verbose:
        print(
            f"  {config.output_name} ({config.series_id}) fetched raw rows: {raw_row_count}",
            flush=True,
        )
    records = pd.concat(record_frames, ignore_index=True)
    if verbose:
        print(
            f"  {config.output_name} ({config.series_id}) normalized vintage rows: {len(records)}",
            flush=True,
        )
    selected = select_asof_weekly_values(records, weekly_dates, config, "fred_api_asof")
    if verbose:
        print(
            f"  {config.output_name} ({config.series_id}) selected weekly rows: {len(selected)}",
            flush=True,
        )
    if return_records:
        return selected, records
    return selected


def build_cpi_yoy_asof_metadata(
    selected: pd.DataFrame,
    records: pd.DataFrame,
    config: MacroSeriesConfig,
    source: str,
) -> pd.DataFrame:
    """Compute monthly CPI YoY from same-vintage observations before weekly use."""
    if config.output_name != "CPI":
        raise ValueError("build_cpi_yoy_asof_metadata only supports CPI.")
    normalized_records = records.copy()
    normalized_records["observation_date"] = pd.to_datetime(
        normalized_records["observation_date"],
        errors="coerce",
    )
    normalized_records["realtime_start"] = pd.to_datetime(
        normalized_records["realtime_start"],
        errors="coerce",
    )
    normalized_records["realtime_end"] = pd.to_datetime(
        normalized_records["realtime_end"],
        errors="coerce",
    )
    normalized_records["value"] = pd.to_numeric(normalized_records["value"], errors="coerce")
    common_metadata = {
        **_common_series_metadata(
            MacroSeriesConfig(
                output_name="cpi_yoy_asof",
                series_id=config.series_id,
                title="CPI year-over-year change computed from as-of monthly CPI observations",
                data_source=config.data_source,
                conceptual_role="inflation_yoy_asof",
                frequency="monthly",
                note=(
                    "Computed as CPI_t / CPI_{t-12 months} - 1 using same-as-of "
                    "FRED vintage observations before weekly alignment. This is "
                    "FRED realtime/as-of safe, not explicit BLS release-calendar audited."
                ),
            )
        ),
        "transformation_applied": "monthly_yoy_before_weekly_alignment",
    }
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        date = pd.Timestamp(row["date"]).normalize()
        current_observation_date = pd.Timestamp(row["observation_date_used"]).normalize()
        lagged_observation_date = current_observation_date - pd.DateOffset(years=1)
        lagged = normalized_records[
            (normalized_records["observation_date"].dt.normalize() == lagged_observation_date)
            & (normalized_records["realtime_start"] <= date)
            & (normalized_records["realtime_end"] >= date)
        ].sort_values(["observation_date", "realtime_start"])
        if lagged.empty:
            value = pd.NA
            lagged_value = pd.NA
            lagged_realtime_start = pd.NaT
            lagged_realtime_end = pd.NaT
        else:
            lagged_row = lagged.iloc[-1]
            lagged_value = float(lagged_row["value"])
            value = float(row["value"]) / lagged_value - 1.0 if lagged_value != 0.0 else pd.NA
            lagged_realtime_start = lagged_row["realtime_start"]
            lagged_realtime_end = lagged_row.get("realtime_end_raw", lagged_row["realtime_end"])
        rows.append(
            {
                **common_metadata,
                "date": date,
                "value": value,
                "observation_date_used": current_observation_date,
                "as_of_date": date,
                "realtime_start_used": row["realtime_start_used"],
                "realtime_end_used": row["realtime_end_used"],
                "vintage_method": source,
                "true_vintage_data_available": bool(row["true_vintage_data_available"]),
                "fallback_method": "",
                "fallback_used": False,
                "current_cpi_observation_date_used": current_observation_date,
                "current_cpi_as_of_date": date,
                "current_cpi_realtime_start_used": row["realtime_start_used"],
                "lagged_12m_cpi_observation_date_used": lagged_observation_date,
                "lagged_12m_cpi_as_of_date": date,
                "lagged_12m_cpi_realtime_start_used": lagged_realtime_start,
                "lagged_12m_cpi_realtime_end_used": lagged_realtime_end,
            }
        )
    frame = pd.DataFrame(rows)
    if frame["value"].isna().any():
        missing_dates = frame.loc[frame["value"].isna(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(
            "Unable to compute as-of CPI YoY for weekly dates because same-vintage "
            f"12-month lagged CPI observations are missing: {missing_dates[:10]}"
        )
    return frame


def validate_vintage_coverage(
    series_id: str,
    vintage_dates: list[str],
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
) -> None:
    """Fail clearly if FRED true-vintage coverage cannot span the protocol window."""
    if not vintage_dates:
        raise ValueError(f"FRED returned no vintage dates for {series_id}.")
    parsed = pd.to_datetime(pd.Series(vintage_dates), errors="coerce").dropna()
    if parsed.empty:
        raise ValueError(f"FRED returned no parseable vintage dates for {series_id}.")
    first_vintage = parsed.min()
    if first_vintage > pd.Timestamp(required_start):
        raise ValueError(
            f"FRED true-vintage coverage for {series_id} starts on "
            f"{first_vintage.date()}, after required start "
            f"{pd.Timestamp(required_start).date()}. No fallback is allowed."
        )


def validate_asof_endpoint_usability(
    selected: pd.DataFrame,
    config: MacroSeriesConfig,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
) -> None:
    """Require true as-of rows at both endpoints and fresh endpoint observations."""
    if selected.empty:
        raise ValueError(f"No selected as-of rows for {config.series_id}.")
    frame = selected.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["observation_date_used"] = pd.to_datetime(
        frame["observation_date_used"], errors="coerce"
    )
    required_start = pd.Timestamp(required_start).normalize()
    required_end = pd.Timestamp(required_end).normalize()
    indexed = frame.set_index(frame["date"].dt.normalize())
    for label, date in (("required_start", required_start), ("required_end", required_end)):
        if date not in indexed.index:
            raise ValueError(
                f"{config.series_id} has no as-of row at {label} {date.date()}."
            )
        row = indexed.loc[date]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        if pd.isna(row["value"]) or pd.isna(row["observation_date_used"]):
            raise ValueError(
                f"{config.series_id} has no usable true as-of value at "
                f"{label} {date.date()}."
            )
        if not bool(row.get("true_vintage_data_available", False)):
            raise ValueError(
                f"{config.series_id} is not marked as true vintage at "
                f"{label} {date.date()}."
            )

    end_row = indexed.loc[required_end]
    if isinstance(end_row, pd.DataFrame):
        end_row = end_row.iloc[-1]
    latest_observation_date = pd.Timestamp(end_row["observation_date_used"]).normalize()
    tolerance_days = freshness_tolerance_days(config.frequency)
    age_days = int((required_end - latest_observation_date).days)
    if age_days > tolerance_days:
        raise ValueError(
            f"{config.series_id} latest observation as of {required_end.date()} is "
            f"{latest_observation_date.date()}, age {age_days} days, exceeding "
            f"{tolerance_days}-day freshness tolerance for {config.frequency} data. "
            "Discontinued/stale series cannot be used as full-window proxies."
        )


def freshness_tolerance_days(frequency: str) -> int:
    """Return endpoint freshness tolerance in calendar days."""
    normalized = str(frequency).lower().strip()
    if normalized in {"daily", "weekly"}:
        return 30
    if normalized == "monthly":
        return 90
    raise ValueError(f"Unsupported macro series frequency: {frequency}")


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
        payload = _fred_get_json_with_retry(
            client=client,
            url=FRED_VINTAGE_DATES_URL,
            params=params,
            api_key=api_key,
            series_id=series_id,
            request_name="vintage-date",
        )
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
        payload = _fred_get_json_with_retry(
            client=client,
            url=FRED_OBSERVATIONS_URL,
            params=params,
            api_key=api_key,
            series_id=series_id,
            request_name="observation",
        )
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


def _fred_get_json_with_retry(
    client: requests.Session,
    url: str,
    params: dict[str, Any],
    api_key: str,
    series_id: str,
    request_name: str,
    retries: int = FRED_REQUEST_RETRIES,
    rate_limit_sleep_seconds: float = FRED_RATE_LIMIT_SLEEP_SECONDS,
) -> dict[str, Any]:
    """Fetch FRED JSON with conservative rate-limit retry handling."""
    last_exc: requests.RequestException | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_exc = exc
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if status == 429 and attempt < retries:
                sleep_seconds = rate_limit_sleep_seconds * attempt
                print(
                    f"FRED {request_name} request for {series_id} hit rate limit; "
                    f"retrying in {sleep_seconds:.0f}s "
                    f"({attempt}/{retries}).",
                    flush=True,
                )
                time.sleep(sleep_seconds)
                continue
            detail = _sanitize_request_error(exc, api_key)
            raise RuntimeError(
                f"FRED {request_name} request failed for "
                f"{series_id}: {exc.__class__.__name__}. "
                "The request URL and API key are intentionally omitted."
                f"{detail}"
            ) from None
    if last_exc is not None:
        detail = _sanitize_request_error(last_exc, api_key)
        raise RuntimeError(
            f"FRED {request_name} request failed for {series_id} after "
            f"{retries} attempts. The request URL and API key are intentionally "
            f"omitted.{detail}"
        ) from None
    raise RuntimeError(f"FRED {request_name} request failed for {series_id}.")


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
    if source == "fred_api_asof":
        return _select_exact_vintage_weekly_values(records, weekly_dates, config, source)

    rows: list[dict[str, Any]] = []
    common_metadata = _common_series_metadata(config)
    for date in weekly_dates:
        available = records[
            (records["observation_date"] <= date)
            & (records["realtime_start"] <= date)
            & (records["realtime_end"] >= date)
        ]
        if available.empty:
            rows.append(
                {
                    **common_metadata,
                    "date": date,
                    "value": pd.NA,
                    "observation_date_used": pd.NaT,
                    "as_of_date": date,
                    "realtime_start_used": pd.NaT,
                    "realtime_end_used": pd.NaT,
                    "vintage_method": source,
                    "true_vintage_data_available": _is_true_vintage_source(source),
                    "fallback_method": "",
                    "fallback_used": False,
                }
            )
            continue
        selected = available.sort_values(["observation_date", "realtime_start"]).iloc[-1]
        rows.append(
            {
                **common_metadata,
                "date": date,
                "value": float(selected["value"]),
                "observation_date_used": selected["observation_date"],
                "as_of_date": date,
                "realtime_start_used": selected["realtime_start"],
                "realtime_end_used": selected["realtime_end_raw"],
                "vintage_method": source,
                "true_vintage_data_available": _is_true_vintage_source(source),
                "fallback_method": "",
                "fallback_used": False,
            }
        )
    return pd.DataFrame(rows)


def _select_exact_vintage_weekly_values(
    records: pd.DataFrame,
    weekly_dates: pd.DatetimeIndex,
    config: MacroSeriesConfig,
    source: str,
) -> pd.DataFrame:
    """Fast path for FRED output_type=2 calls made exactly at weekly as-of dates."""
    rows: list[dict[str, Any]] = []
    common_metadata = _common_series_metadata(config)
    grouped = {
        pd.Timestamp(key).normalize(): frame
        for key, frame in records.groupby(records["realtime_start"].dt.normalize())
    }
    for date in weekly_dates:
        date = pd.Timestamp(date).normalize()
        available = grouped.get(date)
        if available is not None:
            available = available[available["observation_date"] <= date]
        if available is None or available.empty:
            rows.append(
                {
                    **common_metadata,
                    "date": date,
                    "value": pd.NA,
                    "observation_date_used": pd.NaT,
                    "as_of_date": date,
                    "realtime_start_used": pd.NaT,
                    "realtime_end_used": pd.NaT,
                    "vintage_method": source,
                    "true_vintage_data_available": _is_true_vintage_source(source),
                    "fallback_method": "",
                    "fallback_used": False,
                }
            )
            continue
        selected = available.sort_values(["observation_date", "realtime_start"]).iloc[-1]
        rows.append(
            {
                **common_metadata,
                "date": date,
                "value": float(selected["value"]),
                "observation_date_used": selected["observation_date"],
                "as_of_date": date,
                "realtime_start_used": selected["realtime_start"],
                "realtime_end_used": selected["realtime_end_raw"],
                "vintage_method": source,
                "true_vintage_data_available": _is_true_vintage_source(source),
                "fallback_method": "",
                "fallback_used": False,
            }
        )
    return pd.DataFrame(rows)


def _common_series_metadata(config: MacroSeriesConfig) -> dict[str, str]:
    return {
        "series_id": config.series_id,
        "feature_name": config.output_name,
        "output_name": config.output_name,
        "title": config.title,
        "source": config.data_source,
        "conceptual_role": config.conceptual_role,
        "frequency": config.frequency,
        "note": config.note,
        "transformation_applied": "asof_level",
    }


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
    parser.add_argument("--dollar-series-id", default=None)
    parser.add_argument("--dollar-column-name", default=None)
    parser.add_argument(
        "--exclude-series",
        default="",
        help="Comma-separated output names or FRED series IDs to exclude, e.g. DXY.",
    )
    parser.add_argument("--require-no-fallback", action="store_true")
    parser.add_argument("--vintage-chunk-size", type=int, default=50)
    args = parser.parse_args()

    try:
        result = build_realtime_macro_dataset(
            returns_path=args.returns_path,
            output_path=args.output_path,
            metadata_output=args.metadata_output,
            raw_vintage_dir=args.raw_vintage_dir,
            dollar_series_id=args.dollar_series_id,
            dollar_column_name=args.dollar_column_name,
            exclude_series=tuple(
                value.strip()
                for value in args.exclude_series.split(",")
                if value.strip()
            ),
            require_no_fallback=args.require_no_fallback,
            vintage_chunk_size=args.vintage_chunk_size,
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
