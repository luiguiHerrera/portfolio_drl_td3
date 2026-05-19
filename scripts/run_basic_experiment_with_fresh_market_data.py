"""Run a basic experiment from a freshly prepared local market-data snapshot.

This script makes the data refresh explicit:
1. download/build local market returns;
2. write a generated config that points to the local returns CSV;
3. run the normal basic experiment from that generated config;
4. record metadata about the snapshot used.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_market_data import (  # noqa: E402
    DEFAULT_END_DATE,
    DEFAULT_RAW_DIR,
    DEFAULT_START_DATE,
    write_market_data_outputs,
)
from src.experiments.run_basic_experiment import run_basic_experiment  # noqa: E402


def build_config_with_returns_path(
    base_config: dict,
    returns_path: str,
    date_column: str = "date",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Return a copy of config that reads returns from a local CSV snapshot."""
    generated_config = deepcopy(base_config)
    generated_config.setdefault("data", {})
    generated_config["data"]["returns_path"] = returns_path
    generated_config["data"]["returns_date_column"] = date_column
    if start_date is not None:
        generated_config["data"]["start_date"] = start_date
    if end_date is not None:
        generated_config["data"]["end_date"] = end_date

    return generated_config


def write_generated_config(
    config: dict,
    output_dir: str,
    experiment_name: str,
) -> str:
    """Write a generated experiment config to the run output directory."""
    destination = Path(output_dir) / experiment_name
    destination.mkdir(parents=True, exist_ok=True)
    generated_config_path = destination / f"{experiment_name}_generated_config.yaml"
    with generated_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return str(generated_config_path)


def summarize_returns_snapshot(returns_path: str) -> dict:
    """Summarize a processed returns CSV snapshot."""
    path = Path(returns_path)
    if not path.exists():
        raise FileNotFoundError(f"Returns snapshot not found: {returns_path}")

    returns = pd.read_csv(path)
    if "date" not in returns.columns:
        raise KeyError("Returns snapshot must include a date column.")
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
    returns = returns.dropna(subset=["date"])

    asset_columns = [column for column in returns.columns if column != "date"]
    if not asset_columns:
        raise ValueError("Returns snapshot must include at least one asset column.")

    return {
        "market_data_start": str(returns["date"].min().date()),
        "market_data_end": str(returns["date"].max().date()),
        "market_data_shape": [int(returns.shape[0]), int(returns.shape[1] - 1)],
        "assets": asset_columns,
        "missing_values": int(returns[asset_columns].isna().sum().sum()),
    }


def run_basic_experiment_with_fresh_market_data(
    base_config_path: str,
    output_dir: str,
    experiment_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    raw_market_dir: str = DEFAULT_RAW_DIR,
    returns_output_path: str = "data/processed/returns_weekly_latest.csv",
    respect_config_end_date: bool = False,
) -> dict:
    """Refresh local market data, run the basic experiment, and save metadata."""
    base_config = _load_yaml_config(base_config_path)
    data_config = base_config.get("data", {})
    selected_start_date = str(
        start_date
        if start_date is not None
        else data_config.get("start_date", DEFAULT_START_DATE)
    )
    if end_date is not None:
        selected_end_date = str(end_date)
        respected_config_end_date = False
    elif respect_config_end_date:
        selected_end_date = str(data_config.get("end_date", DEFAULT_END_DATE))
        respected_config_end_date = True
    else:
        selected_end_date = str(DEFAULT_END_DATE)
        respected_config_end_date = False
    assets = data_config.get("assets")

    try:
        refresh_result = write_market_data_outputs(
            assets=assets,
            start_date=selected_start_date,
            end_date=selected_end_date,
            raw_dir=raw_market_dir,
            output_path=returns_output_path,
        )
    except Exception as exc:
        raise RuntimeError("Market data refresh failed; experiment was not run.") from exc

    returns_path = refresh_result["processed_path"]
    snapshot_summary = summarize_returns_snapshot(returns_path)
    generated_config = build_config_with_returns_path(
        base_config,
        returns_path=returns_path,
        date_column="date",
        start_date=snapshot_summary["market_data_start"],
        end_date=snapshot_summary["market_data_end"],
    )
    generated_config_path = write_generated_config(
        generated_config,
        output_dir=output_dir,
        experiment_name=experiment_name,
    )
    experiment_result = run_basic_experiment(generated_config_path)

    metadata = {
        "base_config_path": base_config_path,
        "generated_config_path": generated_config_path,
        "returns_path": returns_path,
        "requested_start_date": selected_start_date,
        "requested_end_date": selected_end_date,
        "respected_config_end_date": respected_config_end_date,
        "snapshot_end_used_in_generated_config": snapshot_summary["market_data_end"],
        **snapshot_summary,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = _write_metadata(
        metadata,
        output_dir=output_dir,
        experiment_name=experiment_name,
    )

    return {
        "experiment_result": experiment_result,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "generated_config_path": generated_config_path,
        "returns_path": returns_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh local market data and run a basic TD3 experiment."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--raw-market-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--returns-output-path", default="data/processed/returns_weekly_latest.csv")
    parser.add_argument(
        "--respect-config-end-date",
        action="store_true",
        help="Use config data.end_date when --end-date is not provided.",
    )
    args = parser.parse_args()

    result = run_basic_experiment_with_fresh_market_data(
        base_config_path=args.config,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        start_date=args.start_date,
        end_date=args.end_date,
        raw_market_dir=args.raw_market_dir,
        returns_output_path=args.returns_output_path,
        respect_config_end_date=args.respect_config_end_date,
    )

    print("generated_config_path:", result["generated_config_path"])
    print("returns_path:", result["returns_path"])
    print("metadata_path:", result["metadata_path"])
    print("metadata:", result["metadata"])


def _load_yaml_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Base config must contain a YAML mapping.")

    return config


def _write_metadata(
    metadata: dict,
    output_dir: str,
    experiment_name: str,
) -> str:
    destination = Path(output_dir) / experiment_name
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "fresh_market_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return str(metadata_path)


if __name__ == "__main__":
    main()
