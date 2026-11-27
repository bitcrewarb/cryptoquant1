"""Data management service orchestrating historical downloads, validation, versioning, and derivatives sync."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import polars as pl
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from crypto_quant_lab.data.exchanges.base import DEFAULT_SYMBOLS
from crypto_quant_lab.data.exchanges.binance import BinanceAdapter
from crypto_quant_lab.data.storage.parquet_store import (
    dataset_parquet_paths,
    read_parquet,
    rows_to_polars,
    write_parquet,
)
from crypto_quant_lab.data.validators.quality import repair_ohlcv_dataframe, validate_ohlcv_dataframe
from crypto_quant_lab.database.models import Asset, FundingRate, MarketDataset, OpenInterest, OrderBookSnapshot


def seed_default_assets(db: Session) -> list[Asset]:
    """Ensure initial BTCUSDT, ETHUSDT, SOLUSDT assets exist for Binance spot and futures."""
    created_or_existing: list[Asset] = []
    defaults = [
        ("BTCUSDT", "BTC", "USDT", "spot", 0.01, 0.00001, 5.0, 10.0, 10.0),
        ("ETHUSDT", "ETH", "USDT", "spot", 0.01, 0.0001, 5.0, 10.0, 10.0),
        ("SOLUSDT", "SOL", "USDT", "spot", 0.01, 0.001, 5.0, 10.0, 10.0),
        ("BTCUSDT", "BTC", "USDT", "futures", 0.1, 0.001, 5.0, 2.0, 5.0),
        ("ETHUSDT", "ETH", "USDT", "futures", 0.01, 0.001, 5.0, 2.0, 5.0),
        ("SOLUSDT", "SOL", "USDT", "futures", 0.001, 0.01, 5.0, 2.0, 5.0),
    ]
    for symbol, base, quote, mkt, tick, step, min_not, maker_bps, taker_bps in defaults:
        existing = db.scalar(
            select(Asset).where(
                Asset.symbol == symbol,
                Asset.exchange == "binance",
                Asset.market_type == mkt,
            )
        )
        if existing is None:
            existing = Asset(
                symbol=symbol,
                base_asset=base,
                quote_asset=quote,
                exchange="binance",
                market_type=mkt,
                tick_size=tick,
                step_size=step,
                min_notional=min_not,
                maker_fee_bps=maker_bps,
                taker_fee_bps=taker_bps,
            )
            db.add(existing)
            db.flush()
        created_or_existing.append(existing)
    db.commit()
    return created_or_existing


def register_symbol(db: Session, symbol: str, market_type: str = "spot") -> Asset:
    """Dynamically add a new Binance symbol without code changes."""
    symbol = symbol.strip().upper()
    existing = db.scalar(
        select(Asset).where(
            Asset.symbol == symbol,
            Asset.exchange == "binance",
            Asset.market_type == market_type,
        )
    )
    if existing:
        return existing

    adapter = BinanceAdapter()
    try:
        info = adapter.fetch_exchange_symbol_info(symbol, market_type=market_type)
    finally:
        adapter.close()

    asset = Asset(
        symbol=info["symbol"],
        base_asset=info["base_asset"],
        quote_asset=info["quote_asset"],
        exchange="binance",
        market_type=market_type,
        tick_size=info["tick_size"],
        step_size=info["step_size"],
        min_notional=info["min_notional"],
        maker_fee_bps=2.0 if market_type == "futures" else 10.0,
        taker_fee_bps=5.0 if market_type == "futures" else 10.0,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def download_and_store_dataset(
    db: Session,
    symbol: str,
    timeframe: str,
    market_type: str = "spot",
    max_candles: int | None = 5000,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    progress_cb: Any = None,
) -> MarketDataset:
    """Download historical OHLCV from Binance, validate, write immutable raw + normalized Parquet, and record dataset."""
    symbol = symbol.strip().upper()
    asset = db.scalar(
        select(Asset).where(
            Asset.symbol == symbol,
            Asset.exchange == "binance",
            Asset.market_type == market_type,
        )
    )
    if asset is None:
        if symbol in DEFAULT_SYMBOLS:
            seed_default_assets(db)
            asset = db.scalar(
                select(Asset).where(
                    Asset.symbol == symbol,
                    Asset.exchange == "binance",
                    Asset.market_type == market_type,
                )
            )
        if asset is None:
            asset = register_symbol(db, symbol, market_type=market_type)

    latest = db.scalar(
        select(MarketDataset)
        .where(
            MarketDataset.symbol == symbol,
            MarketDataset.exchange == "binance",
            MarketDataset.market_type == market_type,
            MarketDataset.timeframe == timeframe,
        )
        .order_by(desc(MarketDataset.version))
    )
    next_version = (latest.version + 1) if latest else 1

    adapter = BinanceAdapter()
    try:
        rows = adapter.download_klines_paginated(
            symbol=symbol,
            timeframe=timeframe,
            market_type=market_type,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            max_candles=max_candles,
            progress_cb=progress_cb,
        )
    finally:
        adapter.close()

    raw_df = rows_to_polars(rows)
    raw_path, norm_path = dataset_parquet_paths("binance", market_type, symbol, timeframe, next_version)

    # Raw historical data is written immutably for this version
    write_parquet(raw_df, raw_path)

    report = validate_ohlcv_dataframe(raw_df, symbol=symbol, timeframe=timeframe)
    # Normalized dataset sorted & deduplicated without silently dropping provenance
    norm_df = raw_df.unique(subset=["open_time_ms"], keep="last").sort("open_time_ms") if not raw_df.is_empty() else raw_df
    storage_bytes = write_parquet(norm_df, norm_path)

    start_dt = (
        datetime.fromtimestamp(int(norm_df["open_time_ms"].min()) / 1000, tz=timezone.utc)  # type: ignore[arg-type]
        if not norm_df.is_empty()
        else None
    )
    end_dt = (
        datetime.fromtimestamp(int(norm_df["open_time_ms"].max()) / 1000, tz=timezone.utc)  # type: ignore[arg-type]
        if not norm_df.is_empty()
        else None
    )

    dataset = MarketDataset(
        asset_id=asset.id,
        symbol=symbol,
        exchange="binance",
        market_type=market_type,
        timeframe=timeframe,
        version=next_version,
        provenance=f"binance_{market_type}_klines",
        parquet_path=str(norm_path),
        raw_parquet_path=str(raw_path),
        start_time=start_dt,
        end_time=end_dt,
        candle_count=norm_df.height,
        missing_candles=report.missing_candles,
        duplicate_candles=report.duplicate_candles,
        outlier_candles=report.outlier_candles,
        quality_status=report.quality_status,
        quality_report_json=report.to_dict(),
        storage_bytes=storage_bytes,
        last_synced_at=datetime.now(timezone.utc),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def update_existing_dataset(db: Session, dataset_id: int) -> MarketDataset:
    """Incrementally sync new candles from latest stored timestamp into a new dataset version."""
    ds = db.get(MarketDataset, dataset_id)
    if ds is None:
        raise ValueError("Dataset not found")
    existing_df = read_parquet(ds.parquet_path)
    last_open = int(existing_df["open_time_ms"].max()) if not existing_df.is_empty() else None  # type: ignore[arg-type]

    adapter = BinanceAdapter()
    try:
        new_rows = adapter.download_klines_paginated(
            symbol=ds.symbol,
            timeframe=ds.timeframe,
            market_type=ds.market_type,
            start_time_ms=(last_open + 1) if last_open is not None else None,
            max_candles=5000 if last_open is None else None,
        )
    finally:
        adapter.close()

    new_df = rows_to_polars(new_rows)
    merged = pl.concat([existing_df, new_df], how="vertical") if not existing_df.is_empty() else new_df
    merged = merged.unique(subset=["open_time_ms"], keep="last").sort("open_time_ms") if not merged.is_empty() else merged

    next_version = ds.version + 1
    raw_path, norm_path = dataset_parquet_paths(ds.exchange, ds.market_type, ds.symbol, ds.timeframe, next_version)
    write_parquet(merged, raw_path)
    report = validate_ohlcv_dataframe(merged, symbol=ds.symbol, timeframe=ds.timeframe)
    storage_bytes = write_parquet(merged, norm_path)

    new_ds = MarketDataset(
        asset_id=ds.asset_id,
        symbol=ds.symbol,
        exchange=ds.exchange,
        market_type=ds.market_type,
        timeframe=ds.timeframe,
        version=next_version,
        provenance=f"{ds.provenance}_incremental_update",
        parquet_path=str(norm_path),
        raw_parquet_path=str(raw_path),
        start_time=(
            datetime.fromtimestamp(int(merged["open_time_ms"].min()) / 1000, tz=timezone.utc)  # type: ignore[arg-type]
            if not merged.is_empty()
            else None
        ),
        end_time=(
            datetime.fromtimestamp(int(merged["open_time_ms"].max()) / 1000, tz=timezone.utc)  # type: ignore[arg-type]
            if not merged.is_empty()
            else None
        ),
        candle_count=merged.height,
        missing_candles=report.missing_candles,
        duplicate_candles=report.duplicate_candles,
        outlier_candles=report.outlier_candles,
        quality_status=report.quality_status,
        quality_report_json=report.to_dict(),
        storage_bytes=storage_bytes,
        last_synced_at=datetime.now(timezone.utc),
    )
    db.add(new_ds)
    db.commit()
    db.refresh(new_ds)
    return new_ds


def validate_and_repair_dataset(db: Session, dataset_id: int, repair: bool = False) -> MarketDataset:
    ds = db.get(MarketDataset, dataset_id)
    if ds is None:
        raise ValueError("Dataset not found")
    df = read_parquet(ds.parquet_path)
    if repair:
        repaired_df = repair_ohlcv_dataframe(df, timeframe=ds.timeframe)
        next_version = ds.version + 1
        _, norm_path = dataset_parquet_paths(ds.exchange, ds.market_type, ds.symbol, ds.timeframe, next_version)
        storage_bytes = write_parquet(repaired_df, norm_path)
        report = validate_ohlcv_dataframe(repaired_df, symbol=ds.symbol, timeframe=ds.timeframe)
        repaired_ds = MarketDataset(
            asset_id=ds.asset_id,
            symbol=ds.symbol,
            exchange=ds.exchange,
            market_type=ds.market_type,
            timeframe=ds.timeframe,
            version=next_version,
            provenance=f"{ds.provenance}_repaired_v{ds.version}",
            parquet_path=str(norm_path),
            raw_parquet_path=ds.raw_parquet_path,
            start_time=ds.start_time,
            end_time=ds.end_time,
            candle_count=repaired_df.height,
            missing_candles=report.missing_candles,
            duplicate_candles=report.duplicate_candles,
            outlier_candles=report.outlier_candles,
            quality_status=report.quality_status,
            quality_report_json=report.to_dict(),
            storage_bytes=storage_bytes,
            last_synced_at=datetime.now(timezone.utc),
        )
        db.add(repaired_ds)
        db.commit()
        db.refresh(repaired_ds)
        return repaired_ds

    report = validate_ohlcv_dataframe(df, symbol=ds.symbol, timeframe=ds.timeframe)
    ds.missing_candles = report.missing_candles
    ds.duplicate_candles = report.duplicate_candles
    ds.outlier_candles = report.outlier_candles
    ds.quality_status = report.quality_status
    ds.quality_report_json = report.to_dict()
    db.commit()
    db.refresh(ds)
    return ds


def sync_derivatives_for_symbol(db: Session, symbol: str, period: str = "1h") -> dict[str, Any]:
    """Download real Binance Futures funding rates, open interest, and live orderbook snapshot."""
    symbol = symbol.strip().upper()
    adapter = BinanceAdapter()
    try:
        funding_rows = adapter.fetch_funding_rates(symbol, limit=500)
        oi_rows = adapter.fetch_open_interest_history(symbol, period=period, limit=200)
        ls_rows = adapter.fetch_long_short_ratio(symbol, period=period, limit=200)
        prem = adapter.fetch_premium_index(symbol)
        ob = adapter.fetch_order_book_snapshot(symbol, market_type="futures", limit=20)
    finally:
        adapter.close()

    ls_map = {r["timestamp_ms"]: r["long_short_ratio"] for r in ls_rows}

    inserted_funding = 0
    for f in funding_rows:
        exists = db.scalar(
            select(FundingRate.id).where(
                FundingRate.symbol == symbol,
                FundingRate.exchange == "binance",
                FundingRate.funding_time_ms == f["funding_time_ms"],
            )
        )
        if not exists:
            db.add(
                FundingRate(
                    symbol=symbol,
                    exchange="binance",
                    funding_time_ms=f["funding_time_ms"],
                    funding_rate=f["funding_rate"],
                    mark_price=f["mark_price"],
                    index_price=prem["index_price"],
                    source="binance_fapi_public",
                )
            )
            inserted_funding += 1

    inserted_oi = 0
    for item in oi_rows:
        exists = db.scalar(
            select(OpenInterest.id).where(
                OpenInterest.symbol == symbol,
                OpenInterest.exchange == "binance",
                OpenInterest.period == item["period"],
                OpenInterest.timestamp_ms == item["timestamp_ms"],
            )
        )
        if not exists:
            db.add(
                OpenInterest(
                    symbol=symbol,
                    exchange="binance",
                    period=item["period"],
                    timestamp_ms=item["timestamp_ms"],
                    open_interest=item["open_interest"],
                    open_interest_value=item["open_interest_value"],
                    long_short_ratio=ls_map.get(item["timestamp_ms"]),
                    source="binance_fapi_public",
                )
            )
            inserted_oi += 1

    db.add(
        OrderBookSnapshot(
            symbol=symbol,
            exchange="binance",
            market_type="futures",
            collection_mode="live_snapshot",
            timestamp_ms=ob["timestamp_ms"],
            best_bid=ob["best_bid"],
            best_ask=ob["best_ask"],
            spread_bps=ob["spread_bps"],
            bid_depth_10=ob["bid_depth_10"],
            ask_depth_10=ob["ask_depth_10"],
            imbalance=ob["imbalance"],
            levels_json={"bids": ob["bids"], "asks": ob["asks"]},
        )
    )
    db.commit()
    return {
        "symbol": symbol,
        "funding_inserted": inserted_funding,
        "open_interest_inserted": inserted_oi,
        "premium_index": prem,
        "order_book_snapshot": {
            "best_bid": ob["best_bid"],
            "best_ask": ob["best_ask"],
            "spread_bps": ob["spread_bps"],
            "imbalance": ob["imbalance"],
        },
    }
