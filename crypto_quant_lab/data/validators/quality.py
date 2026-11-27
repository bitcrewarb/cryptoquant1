"""OHLCV data quality validation, gap detection, outlier detection, and non-destructive repair."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl

from crypto_quant_lab.data.exchanges.base import TIMEFRAME_MS


@dataclass
class DataQualityReport:
    symbol: str
    timeframe: str
    total_candles: int
    start_time_utc: str | None
    end_time_utc: str | None
    missing_candles: int
    missing_intervals: list[dict[str, Any]]
    duplicate_candles: int
    invalid_ohlc_candles: int
    non_monotonic_timestamps: int
    outlier_candles: int
    quality_status: str  # verified | warnings | failed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_ohlcv_dataframe(df: pl.DataFrame, symbol: str, timeframe: str) -> DataQualityReport:
    """Validate OHLCV Polars DataFrame for gaps, duplicates, UTC alignment, OHLC invariants, and outliers."""
    if df.is_empty():
        return DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_candles=0,
            start_time_utc=None,
            end_time_utc=None,
            missing_candles=0,
            missing_intervals=[],
            duplicate_candles=0,
            invalid_ohlc_candles=0,
            non_monotonic_timestamps=0,
            outlier_candles=0,
            quality_status="failed",
        )

    tf_ms = TIMEFRAME_MS[timeframe]
    times = df["open_time_ms"].to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    vols = df["volume"].to_numpy()

    # Duplicates
    _, counts = np.unique(times, return_counts=True)
    duplicate_candles = int(np.sum(counts - 1))

    # Non-monotonic check
    diffs = np.diff(times)
    non_monotonic = int(np.sum(diffs <= 0))

    # Missing intervals (sort unique times first)
    unique_times = np.unique(times)
    missing_intervals: list[dict[str, Any]] = []
    missing_candles = 0
    if len(unique_times) > 1:
        u_diffs = np.diff(unique_times)
        gap_indices = np.where(u_diffs > tf_ms)[0]
        for idx in gap_indices:
            gap_start = int(unique_times[idx] + tf_ms)
            gap_end = int(unique_times[idx + 1] - tf_ms)
            n_miss = int((unique_times[idx + 1] - unique_times[idx]) // tf_ms) - 1
            if n_miss > 0:
                missing_candles += n_miss
                if len(missing_intervals) < 50:
                    missing_intervals.append(
                        {
                            "from_ms": gap_start,
                            "to_ms": gap_end,
                            "from_utc": datetime.fromtimestamp(gap_start / 1000, tz=timezone.utc).isoformat(),
                            "to_utc": datetime.fromtimestamp(gap_end / 1000, tz=timezone.utc).isoformat(),
                            "missing_count": n_miss,
                        }
                    )

    # OHLC invariants
    max_oc = np.maximum(opens, closes)
    min_oc = np.minimum(opens, closes)
    invalid_mask = (
        (highs < max_oc - 1e-12)
        | (lows > min_oc + 1e-12)
        | (lows <= 0)
        | (opens <= 0)
        | (closes <= 0)
        | (vols < 0)
        | ~np.isfinite(opens)
        | ~np.isfinite(highs)
        | ~np.isfinite(lows)
        | ~np.isfinite(closes)
    )
    invalid_ohlc = int(np.sum(invalid_mask))

    # Robust return outlier detection via Median Absolute Deviation (MAD)
    outlier_candles = 0
    if len(closes) > 30:
        valid_closes = np.where(closes > 0, closes, np.nan)
        log_ret = np.diff(np.log(valid_closes))
        finite_ret = log_ret[np.isfinite(log_ret)]
        if len(finite_ret) > 10:
            med = float(np.median(finite_ret))
            mad = float(np.median(np.abs(finite_ret - med)))
            if mad > 1e-12:
                robust_z = 0.6745 * np.abs(finite_ret - med) / mad
                outlier_candles = int(np.sum(robust_z > 12.0))

    start_utc = datetime.fromtimestamp(int(unique_times[0]) / 1000, tz=timezone.utc).isoformat()
    end_utc = datetime.fromtimestamp(int(unique_times[-1]) / 1000, tz=timezone.utc).isoformat()

    if invalid_ohlc > 0 or non_monotonic > 0:
        status = "failed"
    elif missing_candles > 0 or duplicate_candles > 0 or outlier_candles > 0:
        status = "warnings"
    else:
        status = "verified"

    return DataQualityReport(
        symbol=symbol,
        timeframe=timeframe,
        total_candles=len(df),
        start_time_utc=start_utc,
        end_time_utc=end_utc,
        missing_candles=missing_candles,
        missing_intervals=missing_intervals,
        duplicate_candles=duplicate_candles,
        invalid_ohlc_candles=invalid_ohlc,
        non_monotonic_timestamps=non_monotonic,
        outlier_candles=outlier_candles,
        quality_status=status,
    )


def repair_ohlcv_dataframe(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """Create a normalized, deduplicated, gap-repaired OHLCV dataset (never overwrites raw file)."""
    if df.is_empty():
        return df

    tf_ms = TIMEFRAME_MS[timeframe]
    cleaned = (
        df.filter(
            (pl.col("open") > 0)
            & (pl.col("high") > 0)
            & (pl.col("low") > 0)
            & (pl.col("close") > 0)
            & (pl.col("volume") >= 0)
        )
        .unique(subset=["open_time_ms"], keep="last")
        .sort("open_time_ms")
        .with_columns(
            [
                pl.max_horizontal("high", "open", "close").alias("high"),
                pl.min_horizontal("low", "open", "close").alias("low"),
            ]
        )
    )

    if cleaned.height < 2:
        return cleaned

    min_t = int(cleaned["open_time_ms"].min())  # type: ignore[arg-type]
    max_t = int(cleaned["open_time_ms"].max())  # type: ignore[arg-type]
    expected_grid = pl.DataFrame(
        {"open_time_ms": list(range(min_t, max_t + 1, tf_ms))}
    )
    joined = expected_grid.join(cleaned, on="open_time_ms", how="left")

    # Carry forward close price for zero-volume synthetic gap bars, flag missing via volume=0
    joined = joined.with_columns(
        [
            pl.col("close").forward_fill().alias("_ff_close"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("open").is_null()).then(pl.col("_ff_close")).otherwise(pl.col("open")).alias("open"),
            pl.when(pl.col("high").is_null()).then(pl.col("_ff_close")).otherwise(pl.col("high")).alias("high"),
            pl.when(pl.col("low").is_null()).then(pl.col("_ff_close")).otherwise(pl.col("low")).alias("low"),
            pl.when(pl.col("close").is_null()).then(pl.col("_ff_close")).otherwise(pl.col("close")).alias("close"),
            pl.col("volume").fill_null(0.0),
            pl.col("quote_volume").fill_null(0.0),
            pl.col("trades_count").fill_null(0),
            pl.col("taker_buy_volume").fill_null(0.0),
            pl.col("taker_buy_quote_volume").fill_null(0.0),
            pl.when(pl.col("close_time_ms").is_null())
            .then(pl.col("open_time_ms") + tf_ms - 1)
            .otherwise(pl.col("close_time_ms"))
            .alias("close_time_ms"),
        ]
    ).drop("_ff_close")

    return joined
