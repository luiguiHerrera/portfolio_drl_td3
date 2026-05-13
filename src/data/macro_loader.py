"""Local macro data loading utilities for Feature Set V3."""

from pathlib import Path

import pandas as pd


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
