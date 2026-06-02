"""Download local market CSVs and build weekly returns for empirical analysis.

This script is a standalone data-acquisition step. Training, feature
construction, evaluation, and experiment runners should consume local outputs
instead of downloading market data during analysis.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_ASSETS = ("SPY", "TLT", "GLD", "BTC-USD", "CASH")
SYNTHETIC_ASSETS = {"CASH"}
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = date.today().isoformat()
DEFAULT_RAW_DIR = "data/raw/market"
DEFAULT_OUTPUT_PATH = "data/processed/returns_weekly_latest.csv"
CASH_RETURN_MODEL_ZERO = "zero"
CASH_RETURN_MODEL_BIL_PROXY = "bil_proxy"
DEFAULT_CASH_PROXY_ASSET = "BIL"
ZERO_CASH_TRANSACTION_COST_BPS = 0.0
BIL_PROXY_CASH_TRANSACTION_COST_BPS = 2.0
DEFAULT_CASH_PROXY_METADATA_PATH = (
    "outputs/tables/cash_proxy_robustness/cash_proxy_metadata.json"
)


def normalize_price_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw price data to a sorted DatetimeIndex with a close column."""
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("raw must be a pandas DataFrame.")
    if raw.empty:
        raise ValueError("raw price data must not be empty.")

    working = raw.copy()
    if "date" not in working.columns:
        if isinstance(working.index, pd.DatetimeIndex):
            working = working.reset_index().rename(columns={working.index.name or "index": "date"})
        elif "Date" in working.columns:
            working = working.rename(columns={"Date": "date"})
        else:
            raise KeyError("raw price data must include a date column.")

    close_column = _detect_close_column(working)
    normalized = working.loc[:, ["date", close_column]].rename(columns={close_column: "close"})
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "close"])
    normalized = normalized.sort_values("date")
    normalized = normalized.drop_duplicates(subset=["date"], keep="last")

    if normalized.empty:
        raise ValueError("raw price data has no usable observations.")

    return normalized.set_index("date").loc[:, ["close"]]


def compute_weekly_returns_from_prices(
    price_data_by_asset: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    weekly_frequency: str = "W-FRI",
    cash_return_model: str = CASH_RETURN_MODEL_ZERO,
    cash_proxy_asset: str = DEFAULT_CASH_PROXY_ASSET,
) -> pd.DataFrame:
    """Build weekly returns from local per-asset close-price DataFrames."""
    cash_return_model = _validate_cash_return_model(cash_return_model)
    if not price_data_by_asset:
        raise ValueError("price_data_by_asset must not be empty.")
    if not weekly_frequency:
        raise ValueError("weekly_frequency must be a non-empty string.")

    risky_assets = [asset for asset in price_data_by_asset if asset not in SYNTHETIC_ASSETS]
    if cash_return_model == CASH_RETURN_MODEL_BIL_PROXY and cash_proxy_asset not in risky_assets:
        risky_assets.append(cash_proxy_asset)
    if not risky_assets:
        raise ValueError("price_data_by_asset must contain at least one risky asset.")

    weekly_prices_by_asset = {}
    for asset in risky_assets:
        raw_prices = price_data_by_asset[asset]
        if raw_prices.empty:
            raise ValueError(f"No usable price data for asset: {asset}")
        prices = normalize_price_data(raw_prices)
        if prices.empty:
            raise ValueError(f"No usable price data for asset: {asset}")
        weekly_prices_by_asset[asset] = prices["close"].resample(weekly_frequency).last()

    weekly_prices = pd.DataFrame(weekly_prices_by_asset)
    returns = weekly_prices.pct_change()
    returns = returns.loc[returns.index >= pd.Timestamp(start_date)]
    returns = returns.loc[returns.index <= pd.Timestamp(end_date)]
    returns = returns.dropna(subset=risky_assets)

    if returns.empty:
        raise ValueError("Processed weekly returns are empty.")

    if "CASH" in price_data_by_asset or "CASH" in DEFAULT_ASSETS:
        if cash_return_model == CASH_RETURN_MODEL_ZERO:
            returns["CASH"] = 0.0
        elif cash_return_model == CASH_RETURN_MODEL_BIL_PROXY:
            if cash_proxy_asset not in returns.columns:
                raise KeyError(
                    f"Missing weekly returns for CASH proxy asset: {cash_proxy_asset}"
                )
            returns["CASH"] = returns[cash_proxy_asset]
        else:
            raise ValueError(f"Unsupported cash_return_model: {cash_return_model}")

    ordered_columns = [asset for asset in DEFAULT_ASSETS if asset in returns.columns]
    remaining_columns = [
        asset
        for asset in returns.columns
        if asset not in ordered_columns and asset != cash_proxy_asset
    ]
    returns = returns.loc[:, [*ordered_columns, *remaining_columns]]

    return returns


def download_asset_price_data(
    asset: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download one risky asset from yfinance and normalize it to date,close rows."""
    raw = yf.download(
        tickers=asset,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"No price data returned for asset: {asset}")

    close = _extract_yfinance_close(raw, asset)
    normalized = close.rename("close").reset_index()
    normalized = normalized.rename(columns={normalized.columns[0]: "date"})

    return normalize_price_data(normalized).reset_index()


def write_market_data_outputs(
    assets: tuple[str, ...] | list[str] = DEFAULT_ASSETS,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    raw_dir: str = DEFAULT_RAW_DIR,
    output_path: str = DEFAULT_OUTPUT_PATH,
    cash_return_model: str = CASH_RETURN_MODEL_ZERO,
    cash_proxy_asset: str = DEFAULT_CASH_PROXY_ASSET,
    metadata_output_path: str | None = None,
) -> dict:
    """Download local raw market CSVs and write the processed weekly returns CSV."""
    cash_return_model = _validate_cash_return_model(cash_return_model)
    raw_destination = Path(raw_dir)
    raw_destination.mkdir(parents=True, exist_ok=True)

    price_data_by_asset = {}
    raw_paths = {}
    download_assets = list(assets)
    if cash_return_model == CASH_RETURN_MODEL_BIL_PROXY and cash_proxy_asset not in download_assets:
        download_assets.append(cash_proxy_asset)
    for asset in download_assets:
        if asset in SYNTHETIC_ASSETS:
            continue

        prices = download_asset_price_data(asset, start_date, end_date)
        if prices.empty:
            raise ValueError(f"No usable price data for asset: {asset}")

        raw_path = raw_destination / f"{asset}.csv"
        prices.to_csv(raw_path, index=False)
        raw_paths[asset] = str(raw_path)
        price_data_by_asset[asset] = prices

    if "CASH" in assets:
        price_data_by_asset["CASH"] = pd.DataFrame()

    returns = compute_weekly_returns_from_prices(
        price_data_by_asset,
        start_date=start_date,
        end_date=end_date,
        cash_return_model=cash_return_model,
        cash_proxy_asset=cash_proxy_asset,
    )

    processed_path = Path(output_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    returns.reset_index(names="date").to_csv(processed_path, index=False)
    metadata = build_cash_proxy_metadata(
        returns=returns,
        assets=assets,
        output_path=str(processed_path),
        cash_return_model=cash_return_model,
        cash_proxy_asset=cash_proxy_asset,
        raw_paths=raw_paths,
    )
    metadata_path = None
    if metadata_output_path:
        metadata_destination = Path(metadata_output_path)
        metadata_destination.parent.mkdir(parents=True, exist_ok=True)
        metadata_destination.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        metadata_path = str(metadata_destination)

    return {
        "raw_paths": raw_paths,
        "processed_path": str(processed_path),
        "returns": returns,
        "metadata": metadata,
        "metadata_path": metadata_path,
    }


def build_cash_proxy_metadata(
    returns: pd.DataFrame,
    assets: tuple[str, ...] | list[str],
    output_path: str,
    cash_return_model: str,
    cash_proxy_asset: str,
    raw_paths: dict[str, str],
) -> dict:
    """Build metadata describing the CASH return convention."""
    cash_transaction_cost_bps = _cash_transaction_cost_bps_for_model(
        cash_return_model
    )
    cash_model_note = (
        "CASH is synthetic zero-return in the main protocol."
        if cash_return_model == CASH_RETURN_MODEL_ZERO
        else (
            "CASH is mapped to weekly returns of the BIL short-term Treasury ETF "
            "proxy for this robustness protocol and should use ETF-like "
            "transaction costs."
        )
    )
    return {
        "output_path": output_path,
        "assets": list(assets),
        "cash_asset_name": "CASH",
        "cash_return_model": cash_return_model,
        "cash_proxy_asset": cash_proxy_asset if cash_return_model != CASH_RETURN_MODEL_ZERO else None,
        "cash_transaction_cost_bps_default": cash_transaction_cost_bps,
        "recommended_asset_transaction_cost_bps": {
            "SPY": 2.0,
            "TLT": 2.0,
            "GLD": 2.0,
            "BTC-USD": 10.0,
            "CASH": cash_transaction_cost_bps,
        },
        "cash_model_note": cash_model_note,
        "main_protocol_cash_note": "CASH is zero-return synthetic in the main protocol.",
        "robustness_protocol_cash_note": (
            "CASH is BIL short-term Treasury ETF proxy in the BIL robustness protocol; "
            "it is an investable ETF-like cash-yield sleeve, not frictionless "
            "synthetic cash."
        ),
        "raw_proxy_path": raw_paths.get(cash_proxy_asset),
        "start": str(returns.index.min().date()) if not returns.empty else None,
        "end": str(returns.index.max().date()) if not returns.empty else None,
        "n_rows": int(len(returns)),
        "cash_missing_values": int(returns["CASH"].isna().sum()) if "CASH" in returns else None,
        "cash_nonzero_count": int((returns["CASH"] != 0.0).sum()) if "CASH" in returns else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download local market data and build weekly returns."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--cash-return-model",
        choices=[CASH_RETURN_MODEL_ZERO, CASH_RETURN_MODEL_BIL_PROXY],
        default=CASH_RETURN_MODEL_ZERO,
    )
    parser.add_argument("--cash-proxy-asset", default=DEFAULT_CASH_PROXY_ASSET)
    parser.add_argument("--metadata-output-path", default=None)
    args = parser.parse_args()

    result = write_market_data_outputs(
        start_date=args.start_date,
        end_date=args.end_date,
        raw_dir=args.raw_dir,
        output_path=args.output_path,
        cash_return_model=args.cash_return_model,
        cash_proxy_asset=args.cash_proxy_asset,
        metadata_output_path=args.metadata_output_path,
    )
    returns = result["returns"]

    print("raw_file_paths:")
    for asset, path in result["raw_paths"].items():
        print(f"{asset}: {path}")
    print("processed_output_path:", result["processed_path"])
    if result["metadata_path"]:
        print("metadata_output_path:", result["metadata_path"])
    print("cash_return_model:", result["metadata"]["cash_return_model"])
    print("cash_proxy_asset:", result["metadata"]["cash_proxy_asset"])
    print("shape:", returns.shape)
    print("start:", returns.index.min())
    print("end:", returns.index.max())
    print("columns:", returns.columns.tolist())
    print("missing_values:", int(returns.isna().sum().sum()))


def _detect_close_column(raw: pd.DataFrame) -> str:
    for candidate in ("close", "Close", "Adj Close", "adj_close"):
        if candidate in raw.columns:
            return candidate
    raise KeyError("raw price data must include a close column.")


def _extract_yfinance_close(raw: pd.DataFrame, asset: str) -> pd.Series:
    if isinstance(raw.columns, pd.MultiIndex):
        if ("Adj Close", asset) in raw.columns:
            return raw[("Adj Close", asset)]
        if ("Close", asset) in raw.columns:
            return raw[("Close", asset)]
        if "Adj Close" in raw.columns.get_level_values(0):
            return raw["Adj Close"].iloc[:, 0]
        if "Close" in raw.columns.get_level_values(0):
            return raw["Close"].iloc[:, 0]
    if "Adj Close" in raw.columns:
        return raw["Adj Close"]
    if "Close" in raw.columns:
        return raw["Close"]

    raise KeyError(f"Downloaded data for {asset} is missing a close price column.")


def _validate_cash_return_model(value: str) -> str:
    model = str(value).strip()
    if model not in {CASH_RETURN_MODEL_ZERO, CASH_RETURN_MODEL_BIL_PROXY}:
        raise ValueError("cash_return_model must be one of: zero, bil_proxy.")
    return model


def _cash_transaction_cost_bps_for_model(cash_return_model: str) -> float:
    model = _validate_cash_return_model(cash_return_model)
    if model == CASH_RETURN_MODEL_BIL_PROXY:
        return BIL_PROXY_CASH_TRANSACTION_COST_BPS
    return ZERO_CASH_TRANSACTION_COST_BPS


if __name__ == "__main__":
    main()
