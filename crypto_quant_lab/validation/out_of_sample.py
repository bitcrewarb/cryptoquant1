"""Chronological Train / Validation / Untouched Out-of-Sample dataset separation (zero shuffling)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from crypto_quant_lab.core.exceptions import LookAheadBiasError


@dataclass(frozen=True)
class ChronologicalSplits:
    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame
    train_range: tuple[int, int]
    val_range: tuple[int, int]
    test_range: tuple[int, int]


def split_chronological(
    df: pl.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
) -> ChronologicalSplits:
    """Split financial time series strictly chronologically without shuffling."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6 or train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("Train, validation, and test ratios must be positive and sum to 1.0")

    ordered = df.sort("open_time_ms")
    n = ordered.height
    if n < 30:
        raise ValueError(f"Dataset too small ({n} bars) for chronological 3-way split")

    i_train = int(n * train_ratio)
    i_val = i_train + int(n * val_ratio)

    train_df = ordered.slice(0, i_train)
    val_df = ordered.slice(i_train, i_val - i_train)
    test_df = ordered.slice(i_val, n - i_val)

    # Verify zero temporal overlap / look-ahead leakage
    if int(train_df["open_time_ms"].max()) >= int(val_df["open_time_ms"].min()):  # type: ignore[arg-type]
        raise LookAheadBiasError("Temporal overlap detected between train and validation splits")
    if int(val_df["open_time_ms"].max()) >= int(test_df["open_time_ms"].min()):  # type: ignore[arg-type]
        raise LookAheadBiasError("Temporal overlap detected between validation and test splits")

    return ChronologicalSplits(
        train=train_df,
        validation=val_df,
        test=test_df,
        train_range=(int(train_df["open_time_ms"].min()), int(train_df["open_time_ms"].max())),  # type: ignore[arg-type]
        val_range=(int(val_df["open_time_ms"].min()), int(val_df["open_time_ms"].max())),  # type: ignore[arg-type]
        test_range=(int(test_df["open_time_ms"].min()), int(test_df["open_time_ms"].max())),  # type: ignore[arg-type]
    )
