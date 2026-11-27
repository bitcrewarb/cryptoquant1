"""Immutable versioned Parquet storage for OHLCV datasets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from crypto_quant_lab.app.config import get_settings
from crypto_quant_lab.data.exchanges.base import CandleRow

OHLCV_SCHEMA = {
    "open_time_ms": pl.Int64,
    "close_time_ms": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "quote_volume": pl.Float64,
    "trades_count": pl.Int64,
    "taker_buy_volume": pl.Float64,
    "taker_buy_quote_volume": pl.Float64,
}


def rows_to_polars(rows: Sequence[CandleRow]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=OHLCV_SCHEMA)
    return pl.DataFrame(
        {
            "open_time_ms": [r.open_time_ms for r in rows],
            "close_time_ms": [r.close_time_ms for r in rows],
            "open": [r.open for r in rows],
            "high": [r.high for r in rows],
            "low": [r.low for r in rows],
            "close": [r.close for r in rows],
            "volume": [r.volume for r in rows],
            "quote_volume": [r.quote_volume for r in rows],
            "trades_count": [r.trades_count for r in rows],
            "taker_buy_volume": [r.taker_buy_volume for r in rows],
            "taker_buy_quote_volume": [r.taker_buy_quote_volume for r in rows],
        },
        schema=OHLCV_SCHEMA,
    )


def dataset_parquet_paths(
    exchange: str,
    market_type: str,
    symbol: str,
    timeframe: str,
    version: int,
) -> tuple[Path, Path]:
    base = get_settings().parquet_dir / exchange / market_type / symbol.upper() / timeframe
    base.mkdir(parents=True, exist_ok=True)
    raw_path = base / f"raw_v{version}.parquet"
    norm_path = base / f"normalized_v{version}.parquet"
    return raw_path, norm_path


def write_parquet(df: pl.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp_path, compression="zstd")
    tmp_path.replace(path)
    return path.stat().st_size


def read_parquet(path: str | Path) -> pl.DataFrame:
    p = Path(path)
    if not p.exists():
        return pl.DataFrame(schema=OHLCV_SCHEMA)
    return pl.read_parquet(p)
