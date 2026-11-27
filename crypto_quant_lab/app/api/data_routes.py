"""Market Data Center API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from crypto_quant_lab.app.dependencies import require_user
from crypto_quant_lab.data.downloaders.service import (
    download_and_store_dataset,
    register_symbol,
    sync_derivatives_for_symbol,
    update_existing_dataset,
    validate_and_repair_dataset,
)
from crypto_quant_lab.data.exchanges.base import SUPPORTED_TIMEFRAMES
from crypto_quant_lab.data.storage.parquet_store import read_parquet
from crypto_quant_lab.database.models import (
    Asset,
    FundingRate,
    MarketDataset,
    OpenInterest,
    OrderBookSnapshot,
    User,
)
from crypto_quant_lab.database.session import get_db
from crypto_quant_lab.workers.job_runner import create_and_dispatch_job

router = APIRouter(prefix="/api/data", tags=["market-data"])


class RegisterAssetRequest(BaseModel):
    symbol: str = Field(..., min_length=4, max_length=24)
    market_type: str = Field(default="spot", pattern="^(spot|futures)$")


class DownloadDatasetRequest(BaseModel):
    symbol: str = Field(..., min_length=4, max_length=24)
    timeframe: str = Field(default="1h")
    market_type: str = Field(default="spot", pattern="^(spot|futures)$")
    max_candles: int = Field(default=3000, ge=100, le=50000)
    background: bool = Field(default=True)


@router.get("/assets")
def list_assets(db: Session = Depends(get_db), _user: User = Depends(require_user)) -> list[dict[str, Any]]:
    assets = list(db.scalars(select(Asset).order_by(Asset.symbol, Asset.market_type)).all())
    return [
        {
            "id": a.id,
            "symbol": a.symbol,
            "base_asset": a.base_asset,
            "quote_asset": a.quote_asset,
            "exchange": a.exchange,
            "market_type": a.market_type,
            "tick_size": a.tick_size,
            "step_size": a.step_size,
            "maker_fee_bps": a.maker_fee_bps,
            "taker_fee_bps": a.taker_fee_bps,
        }
        for a in assets
    ]


@router.post("/assets")
def add_asset(
    payload: RegisterAssetRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    try:
        asset = register_symbol(db, payload.symbol, market_type=payload.market_type)
        return {"id": asset.id, "symbol": asset.symbol, "market_type": asset.market_type}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/datasets")
def list_datasets(db: Session = Depends(get_db), _user: User = Depends(require_user)) -> list[dict[str, Any]]:
    datasets = list(db.scalars(select(MarketDataset).order_by(desc(MarketDataset.created_at))).all())
    return [
        {
            "id": d.id,
            "symbol": d.symbol,
            "market_type": d.market_type,
            "timeframe": d.timeframe,
            "version": d.version,
            "candle_count": d.candle_count,
            "missing_candles": d.missing_candles,
            "duplicate_candles": d.duplicate_candles,
            "outlier_candles": d.outlier_candles,
            "quality_status": d.quality_status,
            "storage_bytes": d.storage_bytes,
            "start_time": d.start_time.isoformat() if d.start_time else None,
            "end_time": d.end_time.isoformat() if d.end_time else None,
            "last_synced_at": d.last_synced_at.isoformat() if d.last_synced_at else None,
            "provenance": d.provenance,
        }
        for d in datasets
    ]


@router.post("/download")
def download_dataset(
    payload: DownloadDatasetRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    if payload.timeframe not in SUPPORTED_TIMEFRAMES:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {payload.timeframe}")
    if payload.background:
        job = create_and_dispatch_job(
            db,
            job_type="data_download",
            config=payload.model_dump(),
            total_items=payload.max_candles,
        )
        return {"mode": "background", "job_uid": job.job_uid, "job_id": job.id, "status": job.status}

    ds = download_and_store_dataset(
        db=db,
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        market_type=payload.market_type,
        max_candles=payload.max_candles,
    )
    return {
        "mode": "sync",
        "dataset_id": ds.id,
        "symbol": ds.symbol,
        "timeframe": ds.timeframe,
        "candle_count": ds.candle_count,
        "quality_status": ds.quality_status,
    }


@router.post("/datasets/{dataset_id}/update")
def update_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    try:
        ds = update_existing_dataset(db, dataset_id)
        return {"dataset_id": ds.id, "version": ds.version, "candle_count": ds.candle_count, "quality_status": ds.quality_status}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/datasets/{dataset_id}/validate")
def validate_dataset(
    dataset_id: int,
    repair: bool = False,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    try:
        ds = validate_and_repair_dataset(db, dataset_id, repair=repair)
        return {
            "dataset_id": ds.id,
            "version": ds.version,
            "quality_status": ds.quality_status,
            "missing_candles": ds.missing_candles,
            "quality_report": ds.quality_report_json,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/derivatives/{symbol}")
def sync_derivatives(
    symbol: str,
    period: str = "1h",
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    try:
        return sync_derivatives_for_symbol(db, symbol, period=period)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/derivatives/{symbol}")
def get_derivatives_summary(
    symbol: str,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    sym = symbol.upper()
    fr = list(db.scalars(select(FundingRate).where(FundingRate.symbol == sym).order_by(desc(FundingRate.funding_time_ms)).limit(50)).all())
    oi = list(db.scalars(select(OpenInterest).where(OpenInterest.symbol == sym).order_by(desc(OpenInterest.timestamp_ms)).limit(50)).all())
    ob = db.scalar(select(OrderBookSnapshot).where(OrderBookSnapshot.symbol == sym).order_by(desc(OrderBookSnapshot.timestamp_ms)))
    return {
        "symbol": sym,
        "funding_rates": [{"t": f.funding_time_ms, "rate": f.funding_rate, "mark": f.mark_price} for f in reversed(fr)],
        "open_interest": [{"t": o.timestamp_ms, "oi": o.open_interest, "oi_val": o.open_interest_value, "ls_ratio": o.long_short_ratio} for o in reversed(oi)],
        "latest_order_book": {
            "collection_mode": ob.collection_mode,
            "best_bid": ob.best_bid,
            "best_ask": ob.best_ask,
            "spread_bps": ob.spread_bps,
            "imbalance": ob.imbalance,
            "timestamp_ms": ob.timestamp_ms,
        }
        if ob
        else None,
    }


@router.get("/datasets/{dataset_id}/ohlcv")
def get_dataset_ohlcv(
    dataset_id: int,
    limit: int = 300,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    ds = db.get(MarketDataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = read_parquet(ds.parquet_path).tail(min(limit, 2000))
    return {
        "dataset_id": ds.id,
        "symbol": ds.symbol,
        "timeframe": ds.timeframe,
        "candles": df.to_dicts(),
    }


@router.delete("/datasets/{dataset_id}")
def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    from pathlib import Path

    from crypto_quant_lab.database.models import MarketRegime, StrategyExperiment

    ds = db.get(MarketDataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    in_use = db.scalar(select(StrategyExperiment.id).where(StrategyExperiment.dataset_id == dataset_id).limit(1))
    if in_use is not None:
        raise HTTPException(
            status_code=409,
            detail="Dataset is referenced by existing strategy experiments and cannot be deleted.",
        )
    db.query(MarketRegime).filter(MarketRegime.dataset_id == dataset_id).delete()
    parquet_paths = [ds.parquet_path, ds.raw_parquet_path]
    db.delete(ds)
    db.commit()
    for p in parquet_paths:
        if p:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
    return {"deleted_id": dataset_id}

