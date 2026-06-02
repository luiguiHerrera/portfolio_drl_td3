"""Local macro data loading utilities for Feature Set V3."""

from pathlib import Path

import pandas as pd


REQUIRED_CLEAN_MACRO_METADATA_COLUMNS = {
    "date",
    "series_id",
    "feature_name",
    "output_name",
    "source",
    "observation_date_used",
    "as_of_date",
    "realtime_start_used",
    "realtime_end_used",
    "vintage_method",
    "fallback_used",
    "transformation_applied",
}

REQUIRED_CPI_YOY_METADATA_COLUMNS = {
    "current_cpi_observation_date_used",
    "current_cpi_as_of_date",
    "current_cpi_realtime_start_used",
    "lagged_12m_cpi_observation_date_used",
    "lagged_12m_cpi_as_of_date",
    "lagged_12m_cpi_realtime_start_used",
}


def load_macro_data_from_csv(
    path: str,
    date_column: str = "date",
) -> pd.DataFrame:
    """Load macro observations from a local CSV file."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Macro CSV file not found: {path}")

    macro_data = pd.read_csv(csv_path)
    if date_column not in macro_data.columns:
        raise KeyError(f"Macro CSV is missing date column: {date_column}")

    macro_data[date_column] = pd.to_datetime(macro_data[date_column], errors="coerce")
    macro_data = macro_data.dropna(subset=[date_column])
    macro_data = macro_data.set_index(date_column).sort_index()
    macro_data = macro_data[~macro_data.index.duplicated(keep="last")]

    value_columns = macro_data.columns
    macro_data = macro_data.loc[:, value_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    macro_data = macro_data.dropna(axis=1, how="all")
    if len(macro_data.columns) == 0:
        raise ValueError("Macro CSV has no usable macro columns.")
    if macro_data.empty:
        raise ValueError("Macro CSV produced an empty macro DataFrame.")

    return macro_data


def default_macro_metadata_path(macro_path: str) -> str:
    """Return the default sidecar metadata path for a processed macro CSV."""
    path = Path(macro_path)
    if path.name == "macro_weekly_realtime_clean_latest.csv":
        return (
            "outputs/tables/v3_macro_realtime_clean_validation/"
            "v3_macro_realtime_series_metadata.csv"
        )
    return str(path.with_name(path.stem + "_series_metadata.csv"))


def load_clean_realtime_macro_data_from_csv(
    path: str,
    metadata_path: str | None = None,
    date_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load clean realtime macro data and require an auditable metadata sidecar."""
    macro = load_macro_data_from_csv(path, date_column=date_column)
    metadata_location = metadata_path or default_macro_metadata_path(path)
    metadata = load_clean_realtime_macro_metadata(metadata_location)
    validate_clean_realtime_macro_metadata(macro, metadata)
    return macro, metadata


def load_clean_realtime_macro_metadata(path: str) -> pd.DataFrame:
    """Load and type clean realtime macro sidecar metadata."""
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Clean realtime macro metadata sidecar not found: {path}")
    metadata = pd.read_csv(metadata_path)
    missing = REQUIRED_CLEAN_MACRO_METADATA_COLUMNS.difference(metadata.columns)
    if missing:
        raise KeyError(
            "Clean realtime macro metadata is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )
    date_columns = [
        "date",
        "observation_date_used",
        "as_of_date",
        "realtime_start_used",
        "current_cpi_observation_date_used",
        "current_cpi_as_of_date",
        "current_cpi_realtime_start_used",
        "lagged_12m_cpi_observation_date_used",
        "lagged_12m_cpi_as_of_date",
        "lagged_12m_cpi_realtime_start_used",
    ]
    for column in date_columns:
        if column in metadata.columns:
            metadata[column] = pd.to_datetime(metadata[column], errors="coerce")
    return metadata


def validate_clean_realtime_macro_metadata(
    macro: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    """Validate clean no-DXY realtime/as-of macro traceability."""
    if "DXY" in macro.columns or (metadata["output_name"].astype(str) == "DXY").any():
        raise ValueError("Clean realtime no-DXY macro specification must not include DXY.")
    if "cpi_yoy_asof" not in macro.columns:
        raise ValueError("Clean realtime macro data must include cpi_yoy_asof.")
    if macro.index.has_duplicates:
        raise ValueError("Clean realtime macro dates must be unique.")
    macro_dates = pd.DatetimeIndex(macro.index).normalize()
    metadata_dates = pd.DatetimeIndex(metadata["date"]).normalize()
    missing_dates = set(macro_dates) - set(metadata_dates)
    if missing_dates:
        sample = sorted(pd.Timestamp(date).strftime("%Y-%m-%d") for date in list(missing_dates)[:5])
        raise ValueError(f"Clean realtime macro metadata missing processed dates: {sample}")

    fallback_used = metadata["fallback_used"].astype(str).str.lower().isin({"true", "1", "yes"})
    if fallback_used.any():
        raise ValueError("Clean realtime macro metadata contains fallback_used rows.")

    date_safety_checks = {
        "as_of_date": metadata["as_of_date"] <= metadata["date"],
        "observation_date_used": metadata["observation_date_used"] <= metadata["date"],
        "realtime_start_used": metadata["realtime_start_used"] <= metadata["date"],
    }
    for name, mask in date_safety_checks.items():
        failures = int((~mask.fillna(False)).sum())
        if failures:
            raise ValueError(
                f"Clean realtime macro metadata has {failures} rows with {name} after decision date."
            )

    cpi_yoy = metadata[metadata["output_name"].astype(str) == "cpi_yoy_asof"]
    if cpi_yoy.empty:
        raise ValueError("Clean realtime macro metadata must include cpi_yoy_asof rows.")
    cpi_yoy_dates = pd.DatetimeIndex(cpi_yoy["date"]).normalize()
    missing_cpi_yoy_dates = set(macro_dates) - set(cpi_yoy_dates)
    if missing_cpi_yoy_dates:
        sample = sorted(
            pd.Timestamp(date).strftime("%Y-%m-%d")
            for date in list(missing_cpi_yoy_dates)[:5]
        )
        raise ValueError(
            "Clean realtime macro metadata missing cpi_yoy_asof trace rows "
            f"for processed dates: {sample}"
        )
    missing_cpi_yoy = REQUIRED_CPI_YOY_METADATA_COLUMNS.difference(metadata.columns)
    if missing_cpi_yoy:
        raise KeyError(
            "Clean realtime CPI YoY metadata is missing required columns: "
            f"{', '.join(sorted(missing_cpi_yoy))}"
        )
    transformations = set(cpi_yoy["transformation_applied"].dropna().astype(str))
    if transformations != {"monthly_yoy_before_weekly_alignment"}:
        raise ValueError(
            "cpi_yoy_asof must use transformation_applied="
            "monthly_yoy_before_weekly_alignment."
        )
    cpi_checks = {
        "current_cpi_as_of_date": cpi_yoy["current_cpi_as_of_date"] <= cpi_yoy["date"],
        "current_cpi_observation_date_used": (
            cpi_yoy["current_cpi_observation_date_used"] <= cpi_yoy["date"]
        ),
        "current_cpi_realtime_start_used": (
            cpi_yoy["current_cpi_realtime_start_used"] <= cpi_yoy["date"]
        ),
        "lagged_12m_cpi_as_of_date": cpi_yoy["lagged_12m_cpi_as_of_date"] <= cpi_yoy["date"],
        "lagged_12m_cpi_observation_date_used": (
            cpi_yoy["lagged_12m_cpi_observation_date_used"] <= cpi_yoy["date"]
        ),
        "lagged_12m_cpi_realtime_start_used": (
            cpi_yoy["lagged_12m_cpi_realtime_start_used"] <= cpi_yoy["date"]
        ),
    }
    for name, mask in cpi_checks.items():
        failures = int((~mask.fillna(False)).sum())
        if failures:
            raise ValueError(
                f"Clean realtime CPI YoY metadata has {failures} rows with {name} after decision date."
            )
