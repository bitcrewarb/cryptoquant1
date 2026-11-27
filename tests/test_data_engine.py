"""Data quality validation, repair (non-destructive), Parquet storage, and real Binance API ingestion tests."""

from __future__ import annotations

import polars as pl
import pytest

from crypto_quant_lab.data.downloaders.service import download_and_store_dataset, seed_default_assets
from crypto_quant_lab.data.exchanges.binance import BinanceAdapter
from crypto_quant_lab.data.storage.parquet_store import read_parquet, write_parquet
from crypto_quant_lab.data.validators.quality import repair_ohlcv_dataframe, validate_ohlcv_dataframe
from crypto_quant_lab.database.session import get_session_factory
from tests.conftest import make_synthetic_ohlcv


def test_data_quality_validation_detects_gaps_duplicates_invalid_ohlc_and_outliers() -> None:
    df = make_synthetic_ohlcv(n=100, seed=5)
    pdf = df.to_pandas()

    # Inject duplicate timestamp, missing gap (drop rows 20..22), invalid OHLC (row 50), and huge outlier (row 80)
    pdf.loc[50, "high"] = pdf.loc[50, "low"] - 10.0  # invalid OHLC invariant
    pdf.loc[80, "close"] = pdf.loc[80, "close"] * 10.0  # massive outlier
    pdf.loc[80, "high"] = max(pdf.loc[80, "high"], pdf.loc[80, "close"])
    dup_row = pdf.iloc[[10]].copy()
    pdf = pdf.drop(index=[20, 21, 22])
    pdf = pdf._append(dup_row, ignore_index=True)

    corrupted = pl.from_pandas(pdf)
    report = validate_ohlcv_dataframe(corrupted, symbol="BTCUSDT", timeframe="1h")
    assert report.missing_candles == 3
    assert report.duplicate_candles == 1
    assert report.invalid_ohlc_candles >= 1
    assert report.non_monotonic_timestamps >= 1
    assert report.outlier_candles >= 1
    assert report.quality_status == "failed"

    # Repair creates clean monotonic gap-filled dataset without touching original DataFrame
    repaired = repair_ohlcv_dataframe(corrupted, timeframe="1h")
    rep_report = validate_ohlcv_dataframe(repaired, symbol="BTCUSDT", timeframe="1h")
    assert rep_report.missing_candles == 0
    assert rep_report.duplicate_candles == 0
    assert rep_report.invalid_ohlc_candles == 0
    assert rep_report.non_monotonic_timestamps == 0
    assert corrupted.height != repaired.height


def test_parquet_roundtrip(tmp_path) -> None:
    df = make_synthetic_ohlcv(n=80, seed=9)
    p = tmp_path / "sample.parquet"
    size = write_parquet(df, p)
    assert size > 0
    loaded = read_parquet(p)
    assert loaded.height == df.height
    assert loaded.columns == df.columns


@pytest.mark.integration
def test_real_binance_spot_and_futures_download(isolated_env) -> None:
    """Verify real public Binance Spot & USDT-M Futures historical ingestion, pagination, and derivatives endpoints."""
    adapter = BinanceAdapter()
    try:
        spot_rows = adapter.download_klines_paginated("BTCUSDT", "1h", market_type="spot", max_candles=150)
        assert len(spot_rows) >= 120
        assert spot_rows[0].open_time_ms < spot_rows[-1].open_time_ms

        fut_rows = adapter.download_klines_paginated("BTCUSDT", "1h", market_type="futures", max_candles=120)
        assert len(fut_rows) >= 100

        funding = adapter.fetch_funding_rates("BTCUSDT", limit=20)
        assert len(funding) > 0
        oi = adapter.fetch_open_interest_history("BTCUSDT", period="1h", limit=10)
        assert len(oi) > 0
        prem = adapter.fetch_premium_index("BTCUSDT")
        assert prem["mark_price"] > 0
        ob = adapter.fetch_order_book_snapshot("BTCUSDT", market_type="futures", limit=10)
        assert ob["collection_mode"] == "live_snapshot"
        assert ob["best_ask"] >= ob["best_bid"] > 0
    finally:
        adapter.close()

    # Persist via service into SQLite + Parquet
    factory = get_session_factory()
    with factory() as db:
        seed_default_assets(db)
        ds = download_and_store_dataset(db, symbol="BTCUSDT", timeframe="1h", market_type="spot", max_candles=200)
        assert ds.candle_count >= 180
        assert ds.quality_status in ("verified", "warnings")
        stored_df = read_parquet(ds.parquet_path)
        assert stored_df.height == ds.candle_count
