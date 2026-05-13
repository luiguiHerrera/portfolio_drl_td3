"""Download local FRED macro CSVs and build the weekly V3 macro dataset.

This script is an explicit data-acquisition step. Training, feature
construction, evaluation, and experiment runners should consume only the local
CSV outputs produced here.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import ssl
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.build_macro_dataset import build_weekly_macro_dataset


FRED_GRAPH_CSV_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SERIES_IDS = ("DGS10", "DGS2", "VIXCLS", "DTWEXBGS", "CPIAUCSL")


def build_fred_csv_url(
    series_id: str,
    observation_start: str,
    observation_end: str,
) -> str:
    """Build a FRED graph CSV endpoint URL for one series."""
    query = urlencode(
        {
            "id": series_id,
            "observation_start": observation_start,
            "observation_end": observation_end,
        }
    )
    return f"{FRED_GRAPH_CSV_BASE_URL}?{query}"


def normalize_fred_csv(raw_csv_text: str, series_id: str) -> pd.DataFrame:
    """Normalize a FRED graph CSV payload to date,value columns."""
    raw = pd.read_csv(StringIO(raw_csv_text), na_values=["."])

    date_column = _detect_date_column(raw)
    if series_id not in raw.columns:
        raise KeyError(f"FRED CSV is missing expected series column: {series_id}")

    normalized = raw.loc[:, [date_column, series_id]].rename(
        columns={date_column: "date", series_id: "value"}
    )
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "value"]).sort_values("date")

    if normalized.empty:
        raise ValueError(f"FRED CSV for {series_id} has no usable observations.")

    return normalized.loc[:, ["date", "value"]]


def download_fred_series(
    series_id: str,
    output_dir: str,
    observation_start: str,
    observation_end: str,
) -> Path:
    """Download and save one FRED series as a local date,value CSV."""
    url = build_fred_csv_url(series_id, observation_start, observation_end)
    raw_csv_text = _download_text(url)

    normalized = normalize_fred_csv(raw_csv_text, series_id)

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / f"{series_id}.csv"
    normalized.to_csv(output_path, index=False)

    return output_path


def download_all_fred_macro_series(
    output_dir: str = "data/raw/macro",
    observation_start: str = "2015-01-01",
    observation_end: str = "2024-12-31",
) -> list[Path]:
    """Download all required macro series to local raw CSV files."""
    return [
        download_fred_series(
            series_id=series_id,
            output_dir=output_dir,
            observation_start=observation_start,
            observation_end=observation_end,
        )
        for series_id in SERIES_IDS
    ]


def build_processed_macro_after_download(
    raw_macro_dir: str = "data/raw/macro",
    output_path: str = "data/processed/macro_weekly_2015_2024.csv",
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    cpi_lag_weeks: int = 4,
) -> pd.DataFrame:
    """Build the processed weekly macro CSV from local raw macro files."""
    return build_weekly_macro_dataset(
        raw_macro_dir=raw_macro_dir,
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
        cpi_lag_weeks=cpi_lag_weeks,
    )


def main() -> None:
    raw_paths = download_all_fred_macro_series()
    macro = build_processed_macro_after_download()
    processed_path = Path("data/processed/macro_weekly_2015_2024.csv")

    print("raw_file_paths:")
    for path in raw_paths:
        print(path)
    print("processed_output_path:", processed_path)
    print("shape:", macro.shape)
    print("start:", macro.index.min())
    print("end:", macro.index.max())
    print("columns:", macro.columns.tolist())
    print("missing_values:", int(macro.isna().sum().sum()))


def _detect_date_column(raw: pd.DataFrame) -> str:
    if "observation_date" in raw.columns:
        return "observation_date"
    if "date" in raw.columns:
        return "date"
    raise KeyError("FRED CSV is missing a date column.")


def _download_text(url: str) -> str:
    context = _ssl_context()
    with urlopen(url, timeout=30, context=context) as response:
        return response.read().decode("utf-8")


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi
    except ImportError:
        return None

    return ssl.create_default_context(cafile=certifi.where())


if __name__ == "__main__":
    main()
