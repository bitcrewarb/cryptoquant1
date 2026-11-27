"""API routes for Quant Research, Strategy Discovery, Backtesting, Validation, Portfolio, Paper & Live Trading, and Exports."""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from crypto_quant_lab.app.dependencies import require_user
from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.data.storage.parquet_store import read_parquet
from crypto_quant_lab.database.models import (
    BacktestResult,
    ExchangeConfiguration,
    MarketDataset,
    MarketRegime,
    MonteCarloResult,
    PaperOrder,
    PaperPosition,
    PaperTrade,
    PortfolioAllocation,
    PortfolioDefinition,
    ResearchJob,
    StrategyDefinition,
    StrategyExperiment,
    StrategyVersion,
    SystemLog,
    TradeRecord,
    User,
    ValidationResult,
    WalkForwardResult,
)
from crypto_quant_lab.database.session import get_db
from crypto_quant_lab.indicators.statistical import (
    autocorrelation_analysis,
    engle_granger_cointegration,
    garch_11_volatility,
    hurst_exponent,
    kalman_filter_1d,
    pearson_spearman_correlation,
    shannon_entropy,
    stationarity_tests,
    volatility_clustering_score,
)
from crypto_quant_lab.indicators.volume import volume_profile
from crypto_quant_lab.live_trading.adapter import (
    check_exchange_connectivity,
    save_exchange_credentials,
    toggle_live_trading,
    trigger_emergency_kill_switch,
)
from crypto_quant_lab.paper_trading.engine import (
    deploy_paper_strategy,
    force_close_paper_position,
    step_paper_trading_cycle,
    stop_paper_strategy,
)
from crypto_quant_lab.portfolio.engine import construct_portfolio
from crypto_quant_lab.regimes.detector import detect_market_regimes
from crypto_quant_lab.risk.manager import RiskConfig
from crypto_quant_lab.strategies.evaluator import IndicatorCache, evaluate_strategy
from crypto_quant_lab.strategies.grammar import STRATEGY_FAMILIES, StrategySpec, validate_strategy_spec
from crypto_quant_lab.workers.job_runner import (
    create_and_dispatch_job,
    dispatch_job,
    load_funding_rates_map,
    run_validation_suite_sync,
)

router = APIRouter(prefix="/api", tags=["quant-research"])


class AnalyzeDatasetRequest(BaseModel):
    dataset_id: int
    tail_bars: int = Field(default=500, ge=100, le=5000)
    use_hmm: bool = Field(default=False)


class CointegrationRequest(BaseModel):
    dataset_a_id: int
    dataset_b_id: int
    tail_bars: int = Field(default=500, ge=60, le=5000)


class DiscoveryStartRequest(BaseModel):
    dataset_id: int
    candidate_count: int = Field(default=500, ge=10, le=50000)
    algorithm: str = Field(default="random_search")
    families: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    seed: int = Field(default=42)
    persist_top_n: int = Field(default=50, ge=5, le=500)


class BacktestRunRequest(BaseModel):
    experiment_id: int
    dataset_id: int | None = None
    engine_type: str = Field(default="event_driven", pattern="^(vectorized|event_driven)$")
    initial_capital: float = Field(default=100_000.0, ge=100.0)
    maker_fee_bps: float = Field(default=10.0, ge=0.0, le=100.0)
    taker_fee_bps: float = Field(default=10.0, ge=0.0, le=100.0)
    slippage_bps: float = Field(default=3.0, ge=0.0, le=100.0)
    spread_bps: float = Field(default=2.0, ge=0.0, le=100.0)
    leverage: float = Field(default=1.0, ge=1.0, le=20.0)
    start_date: str | None = None
    end_date: str | None = None


class ValidationRunRequest(BaseModel):
    experiment_id: int
    mc_simulations: int = Field(default=1000, ge=100, le=10000)
    wf_windows: int = Field(default=5, ge=3, le=12)
    wf_mode: str = Field(default="rolling", pattern="^(rolling|expanding|anchored)$")
    n_trials_attempted: int = Field(default=100, ge=1, le=100000)
    unlock_final_test: bool = Field(default=False)


class BatchValidationRequest(BaseModel):
    experiment_ids: list[int] = Field(..., min_length=1, max_length=25)
    mc_simulations: int = Field(default=1000, ge=100, le=10000)
    wf_windows: int = Field(default=5, ge=3, le=12)
    wf_mode: str = Field(default="rolling", pattern="^(rolling|expanding|anchored)$")
    n_trials_attempted: int = Field(default=100, ge=1, le=100000)


class PortfolioConstructRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    experiment_ids: list[int] = Field(..., min_length=1, max_length=25)
    allocation_method: str = Field(default="risk_parity")
    initial_capital: float = Field(default=100_000.0, ge=1000.0)
    max_single_weight: float = Field(default=0.40, ge=0.05, le=1.0)


class PaperDeployRequest(BaseModel):
    experiment_id: int
    symbol: str = Field(default="BTCUSDT")
    timeframe: str = Field(default="1h")
    market_type: str = Field(default="spot", pattern="^(spot|futures)$")
    allocated_capital: float = Field(default=10_000.0, ge=100.0)


class LiveCredentialRequest(BaseModel):
    exchange: str = Field(default="binance")
    market_type: str = Field(default="spot", pattern="^(spot|futures)$")
    label: str = Field(..., min_length=2, max_length=60)
    api_key: str = Field(..., min_length=8)
    api_secret: str = Field(..., min_length=8)
    permissions_scope: str = Field(default="read_only", pattern="^(read_only|spot_trade|futures_trade)$")
    max_capital_usdt: float = Field(default=0.0, ge=0.0)
    max_exposure_pct: float = Field(default=10.0, ge=1.0, le=100.0)


class LiveToggleRequest(BaseModel):
    enable: bool
    confirmation_phrase: str = ""


@router.get("/research/catalog")
def research_catalog(_user: User = Depends(require_user)) -> dict[str, Any]:
    return {
        "trend": ["SMA", "EMA", "WMA", "HMA", "VWAP", "Anchored VWAP", "ADX (+DI/-DI)", "Supertrend", "Donchian Channels", "Ichimoku Cloud", "Linear Regression Slope"],
        "momentum": ["RSI", "Stochastic RSI", "MACD", "CCI", "ROC", "Momentum", "Williams %R", "Ultimate Oscillator", "TRIX"],
        "volatility": ["ATR", "Historical Volatility", "Bollinger Bands", "Keltner Channels", "Donchian Width", "Parkinson Volatility", "Garman-Klass Volatility", "Volatility Percentile", "Volatility Regime"],
        "volume": ["OBV", "Volume SMA", "Volume Z-Score", "Volume Profile (POC/VAH/VAL)", "Cumulative Volume Delta", "Relative Volume", "Buying/Selling Pressure", "Volume Imbalance"],
        "statistical": ["Z-Score", "Rolling Mean/Std", "Pearson/Spearman Corr", "ACF/PACF", "ADF Stationarity", "KPSS Stationarity", "Engle-Granger Cointegration", "Linear Regression Residuals", "1D Kalman Filter", "Hurst Exponent", "Shannon Entropy", "Volatility Clustering", "GARCH(1,1) MLE"],
        "price_action": ["Causal Swing High/Low", "HH/HL/LH/LL", "Breakouts & Breakdowns", "Dynamic S/R", "Trendline Slopes", "Inside Bars", "Engulfing", "Pin Bars", "Range Expansion", "Failed Breakouts"],
        "smc": ["External/Internal Structure", "BOS", "CHoCH", "MSS", "EQH/EQL", "BSL/SSL Liquidity Sweeps", "Order Blocks", "Breaker Blocks", "Mitigation Zones", "FVG & Inverse FVG", "Displacement", "Premium/Discount Dealing Range"],
        "ict": ["Configurable Sessions & Kill Zones", "Optimal Trade Entry (0.618-0.786)", "Liquidity Raid", "Judas Swing", "Opening Range", "PDH/PDL & PWH/PWL", "Daily/Weekly Bias", "Power of Three", "SMT Divergence", "Consequent Encroachment"],
        "regimes": ["Strong Bull", "Weak Bull", "Strong Bear", "Weak Bear", "Sideways", "High Vol", "Low Vol", "Vol Expansion", "Vol Compression", "Transitional"],
        "strategy_families": list(STRATEGY_FAMILIES),
    }


@router.post("/research/analyze")
def analyze_dataset(
    payload: AnalyzeDatasetRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    ds = db.get(MarketDataset, payload.dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = read_parquet(ds.parquet_path).tail(payload.tail_bars)
    if df.height < 60:
        raise HTTPException(status_code=400, detail="Dataset must have at least 60 candles for quantitative analysis")

    ctx = IndicatorCache(df, timeframe=ds.timeframe)
    c = ctx.close
    log_ret = np.diff(np.log(np.maximum(c, 1e-12)))

    stat_res = stationarity_tests(log_ret)
    hurst = hurst_exponent(c)
    ent = shannon_entropy(log_ret)
    v_clust = volatility_clustering_score(log_ret)
    garch = garch_11_volatility(log_ret)
    acf_pacf = autocorrelation_analysis(log_ret, nlags=15)
    vp = volume_profile(ctx.high, ctx.low, c, ctx.volume)
    kf = kalman_filter_1d(c)
    ema20 = ctx.get_ema(20)
    ema50 = ctx.get_ema(50)
    rsi14 = ctx.get_rsi(14)
    smc = ctx.get_smc()
    pa = ctx.get_pa()
    ict = ctx.get_ict()
    regimes = detect_market_regimes(ctx.high, ctx.low, c, use_hmm=payload.use_hmm)

    # Persist latest regime distribution summary
    db.query(MarketRegime).filter(MarketRegime.dataset_id == ds.id).delete()
    step_persist = max(1, ctx.n // 200)
    for idx in range(0, ctx.n, step_persist):
        db.add(
            MarketRegime(
                dataset_id=ds.id,
                open_time_ms=int(ctx.open_time_ms[idx]),
                primary_regime=regimes["primary_regime"][idx],
                trend_regime=regimes["trend_regime"][idx],
                volatility_regime=regimes["volatility_regime"][idx],
                confidence=float(regimes["confidence"][idx]),
            )
        )
    db.commit()

    def _clean(arr: np.ndarray) -> list[float | None]:
        return [round(float(x), 4) if np.isfinite(x) else None for x in arr]

    return {
        "symbol": ds.symbol,
        "timeframe": ds.timeframe,
        "bars": ctx.n,
        "statistics": {
            "adf_pvalue": round(stat_res["adf_pvalue"], 5),
            "is_stationary_returns": stat_res["is_stationary_adf"],
            "kpss_pvalue": round(stat_res["kpss_pvalue"], 5),
            "hurst_exponent": round(hurst, 4),
            "shannon_entropy": round(ent, 4),
            "volatility_clustering": round(v_clust, 4),
            "garch_alpha": round(garch["alpha"], 4),
            "garch_beta": round(garch["beta"], 4),
            "garch_persistence": round(garch.get("persistence", 0.0), 4),
            "volume_profile": {"poc": round(float(vp["poc"]), 2), "vah": round(float(vp["vah"]), 2), "val": round(float(vp["val"]), 2)},
            "acf": acf_pacf["acf"][:10],
        },
        "signal_counts": {
            "pa_breakouts_up": int(np.sum(pa["breakout_up"])),
            "pa_pin_bars": int(np.sum(pa["bullish_pin_bar"]) + np.sum(pa["bearish_pin_bar"])),
            "smc_bos": int(np.sum(smc["bos_bull"]) + np.sum(smc["bos_bear"])),
            "smc_choch": int(np.sum(smc["choch_bull"]) + np.sum(smc["choch_bear"])),
            "smc_fvg": int(np.sum(smc["fvg_bull"]) + np.sum(smc["fvg_bear"])),
            "smc_liquidity_sweeps": int(np.sum(smc["bsl_sweep"]) + np.sum(smc["ssl_sweep"])),
            "ict_judas_swings": int(np.sum(ict["judas_swing_bull"]) + np.sum(ict["judas_swing_bear"])),
            "ict_liquidity_raids": int(np.sum(ict["liquidity_raid_bull"]) + np.sum(ict["liquidity_raid_bear"])),
        },
        "regime_distribution": regimes["distribution"],
        "series": {
            "timestamps_ms": [int(t) for t in ctx.open_time_ms],
            "close": _clean(c),
            "ema20": _clean(ema20),
            "ema50": _clean(ema50),
            "kalman": _clean(kf),
            "rsi14": _clean(rsi14),
            "regimes": regimes["primary_regime"],
            "garch_vol": _clean(garch["conditional_vol"]),
        },
    }


@router.post("/research/cointegration")
def cointegration_check(
    payload: CointegrationRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    ds_a = db.get(MarketDataset, payload.dataset_a_id)
    ds_b = db.get(MarketDataset, payload.dataset_b_id)
    if ds_a is None or ds_b is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df_a = read_parquet(ds_a.parquet_path).select(["open_time_ms", "close"]).rename({"close": "close_a"})
    df_b = read_parquet(ds_b.parquet_path).select(["open_time_ms", "close"]).rename({"close": "close_b"})
    joined = df_a.join(df_b, on="open_time_ms", how="inner").sort("open_time_ms").tail(payload.tail_bars)
    if joined.height < 40:
        raise HTTPException(status_code=400, detail="Insufficient overlapping timestamps between the two datasets")
    a = joined["close_a"].to_numpy()
    b = joined["close_b"].to_numpy()
    coint_res = engle_granger_cointegration(a, b)
    corr_res = pearson_spearman_correlation(np.diff(np.log(a)), np.diff(np.log(b)))
    return {
        "pair": f"{ds_a.symbol} / {ds_b.symbol}",
        "overlapping_bars": joined.height,
        "cointegration": coint_res,
        "return_correlation": corr_res,
    }


@router.post("/discovery/start")
def start_discovery(
    payload: DiscoveryStartRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    ds = db.get(MarketDataset, payload.dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if ds.market_type == "spot" and any(d in ("short_only", "long_short") for d in payload.directions):
        raise HTTPException(status_code=400, detail="Short selling directions are not allowed on spot datasets")
    job = create_and_dispatch_job(
        db,
        job_type="strategy_discovery",
        config=payload.model_dump(),
        total_items=payload.candidate_count,
    )
    return {"job_id": job.id, "job_uid": job.job_uid, "status": job.status}


@router.get("/jobs")
def list_jobs(
    job_type: str | None = None,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> list[dict[str, Any]]:
    stmt = select(ResearchJob)
    if job_type:
        stmt = stmt.where(ResearchJob.job_type == job_type)
    jobs = list(db.scalars(stmt.order_by(desc(ResearchJob.created_at)).limit(40)).all())
    return [
        {
            "id": j.id,
            "job_uid": j.job_uid,
            "job_type": j.job_type,
            "status": j.status,
            "total_items": j.total_items,
            "completed_items": j.completed_items,
            "failed_items": j.failed_items,
            "progress_pct": round(j.progress_pct, 1),
            "error_message": j.error_message,
            "config": j.config_json,
            "checkpoint": j.checkpoint_json,
            "created_at": j.created_at.isoformat(),
        }
        for j in jobs
    ]


@router.get("/jobs/{job_id}/results")
def get_job_results(
    job_id: int,
    limit: int = Query(default=30, le=200),
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    job = db.get(ResearchJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    exps = list(
        db.scalars(
            select(StrategyExperiment)
            .where(StrategyExperiment.research_job_id == job_id)
            .order_by(desc(StrategyExperiment.composite_score))
            .limit(limit)
        ).all()
    )
    items: list[dict[str, Any]] = []
    for exp in exps:
        sdef = db.get(StrategyDefinition, exp.strategy_id)
        ds = db.get(MarketDataset, exp.dataset_id)
        bt = db.scalar(
            select(BacktestResult)
            .where(BacktestResult.experiment_id == exp.id)
            .order_by(desc(BacktestResult.id))
        )
        if not sdef or not ds or not bt:
            continue
        items.append(
            {
                "experiment_id": exp.id,
                "experiment_uid": exp.experiment_uid,
                "strategy_name": sdef.name,
                "family": sdef.family,
                "direction": sdef.direction,
                "symbol": ds.symbol,
                "timeframe": ds.timeframe,
                "validation_tier": exp.validation_tier,
                "sharpe_ratio": round(bt.sharpe_ratio, 3),
                "profit_factor": round(bt.profit_factor, 2),
                "win_rate": round(bt.win_rate, 1),
                "max_drawdown_pct": round(bt.max_drawdown_pct, 2),
                "return_pct": round(bt.return_on_capital_pct, 2),
                "total_trades": bt.total_trades,
                "composite_score": round(exp.composite_score, 2),
            }
        )
    return {
        "job_id": job.id,
        "job_uid": job.job_uid,
        "status": job.status,
        "count": len(items),
        "results": items,
    }


@router.post("/jobs/{job_id}/pause")
def pause_job(job_id: int, db: Session = Depends(get_db), _user: User = Depends(require_user)) -> dict[str, Any]:
    job = db.get(ResearchJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in ("queued", "running"):
        job.status = "paused"
        db.commit()
    return {"id": job.id, "status": job.status}


@router.post("/jobs/{job_id}/resume")
def resume_job(job_id: int, db: Session = Depends(get_db), _user: User = Depends(require_user)) -> dict[str, Any]:
    job = db.get(ResearchJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in ("paused", "failed"):
        job.status = "queued"
        job.error_message = None
        db.commit()
        dispatch_job(job.id)
    return {"id": job.id, "status": job.status}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, db: Session = Depends(get_db), _user: User = Depends(require_user)) -> dict[str, Any]:
    job = db.get(ResearchJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in ("queued", "running", "paused"):
        job.status = "cancelled"
        db.commit()
    return {"id": job.id, "status": job.status}


@router.post("/backtest/run")
def run_on_demand_backtest(
    payload: BacktestRunRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    exp = db.get(StrategyExperiment, payload.experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    ds_id = payload.dataset_id or exp.dataset_id
    ds = db.get(MarketDataset, ds_id)
    strat_def = db.get(StrategyDefinition, exp.strategy_id)
    strat_ver = db.get(StrategyVersion, exp.strategy_version_id)
    if ds is None or strat_def is None or strat_ver is None:
        raise HTTPException(status_code=404, detail="Strategy or dataset missing")

    df = read_parquet(ds.parquet_path).sort("open_time_ms")
    if payload.start_date:
        start_ms = int(datetime.fromisoformat(payload.start_date).replace(tzinfo=timezone.utc).timestamp() * 1000)
        df = df.filter(df["open_time_ms"] >= start_ms)
    if payload.end_date:
        end_ms = int(datetime.fromisoformat(payload.end_date).replace(tzinfo=timezone.utc).timestamp() * 1000)
        df = df.filter(df["open_time_ms"] <= end_ms)

    if df.height < 40:
        raise HTTPException(status_code=400, detail="Insufficient candles in selected date range")

    spec = StrategySpec.from_dicts(strat_def.name, strat_def.rule_json, strat_ver.params_json, strat_ver.version)
    try:
        validate_strategy_spec(spec, market_type=ds.market_type)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ctx = IndicatorCache(df, timeframe=ds.timeframe)
    funding_map = load_funding_rates_map(db, ds.symbol, ds.market_type)
    costs = ExecutionCostConfig(
        market_type=ds.market_type,
        maker_fee_bps=payload.maker_fee_bps,
        taker_fee_bps=payload.taker_fee_bps,
        slippage_bps=payload.slippage_bps,
        spread_bps=payload.spread_bps,
        leverage=payload.leverage,
    )
    bt_out = evaluate_strategy(
        spec,
        ctx,
        engine_type=payload.engine_type,
        initial_capital=payload.initial_capital,
        costs=costs,
        funding_rates_map=funding_map,
    )
    m = bt_out.metrics

    bt_row = BacktestResult(
        experiment_id=exp.id,
        engine_type=payload.engine_type,
        split_segment="full",
        net_profit=float(m["net_profit"]),
        gross_profit=float(m["gross_profit"]),
        gross_loss=float(m["gross_loss"]),
        profit_factor=float(m["profit_factor"]),
        win_rate=float(m["win_rate"]),
        expectancy=float(m["expectancy"]),
        sharpe_ratio=float(m["sharpe_ratio"]),
        sortino_ratio=float(m["sortino_ratio"]),
        calmar_ratio=float(m["calmar_ratio"]),
        max_drawdown_pct=float(m["max_drawdown_pct"]),
        avg_drawdown_pct=float(m["avg_drawdown_pct"]),
        recovery_factor=float(m["recovery_factor"]),
        ulcer_index=float(m["ulcer_index"]),
        return_on_capital_pct=float(m["return_on_capital_pct"]),
        annualized_return_pct=float(m["annualized_return_pct"]),
        total_trades=int(m["total_trades"]),
        exposure_pct=float(m["exposure_pct"]),
        metrics_json=m,
        equity_curve_json=bt_out.equity_curve,
        regime_breakdown_json=m.get("regime_breakdown", {}),
    )
    db.add(bt_row)
    db.flush()
    for tr in bt_out.trades[:1000]:
        db.add(
            TradeRecord(
                backtest_id=bt_row.id,
                side=tr.side,
                entry_time_ms=tr.entry_time_ms,
                exit_time_ms=tr.exit_time_ms,
                entry_price=tr.entry_price,
                exit_price=tr.exit_price,
                quantity=tr.quantity,
                gross_pnl=tr.gross_pnl,
                fee_paid=tr.fee_paid,
                slippage_paid=tr.slippage_paid,
                funding_paid=tr.funding_paid,
                net_pnl=tr.net_pnl,
                return_pct=tr.return_pct,
                exit_reason=tr.exit_reason,
                regime_at_entry=tr.regime_at_entry,
            )
        )
    db.commit()

    return {
        "backtest_id": bt_row.id,
        "strategy_name": strat_def.name,
        "symbol": ds.symbol,
        "timeframe": ds.timeframe,
        "engine_type": payload.engine_type,
        "metrics": m,
        "equity_curve": bt_out.equity_curve,
        "trades": [
            {
                "side": t.side,
                "entry_time_ms": t.entry_time_ms,
                "exit_time_ms": t.exit_time_ms,
                "entry_price": round(t.entry_price, 4),
                "exit_price": round(t.exit_price, 4),
                "net_pnl": round(t.net_pnl, 2),
                "return_pct": round(t.return_pct, 2),
                "exit_reason": t.exit_reason,
                "regime": t.regime_at_entry,
            }
            for t in bt_out.trades[:200]
        ],
    }


@router.post("/validation/run")
def run_validation(
    payload: ValidationRunRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    try:
        return run_validation_suite_sync(
            db=db,
            experiment_id=payload.experiment_id,
            mc_simulations=payload.mc_simulations,
            wf_windows=payload.wf_windows,
            wf_mode=payload.wf_mode,
            n_trials_attempted=payload.n_trials_attempted,
            unlock_final_test=payload.unlock_final_test,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/validation/batch")
def run_validation_batch(
    payload: BatchValidationRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    job = create_and_dispatch_job(
        db,
        job_type="validation_suite",
        config=payload.model_dump(),
        total_items=len(payload.experiment_ids),
    )
    return {"job_id": job.id, "job_uid": job.job_uid, "status": job.status, "total_items": job.total_items}


@router.get("/leaderboard")
def get_leaderboard(
    symbol: str | None = None,
    timeframe: str | None = None,
    family: str | None = None,
    validation_tier: str | None = None,
    min_sharpe: float | None = None,
    min_trades: int = 0,
    max_drawdown: float | None = None,
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> list[dict[str, Any]]:
    exps = list(db.scalars(select(StrategyExperiment).order_by(desc(StrategyExperiment.composite_score)).limit(500)).all())
    rows: list[dict[str, Any]] = []
    for exp in exps:
        sdef = db.get(StrategyDefinition, exp.strategy_id)
        ds = db.get(MarketDataset, exp.dataset_id)
        bt = db.scalar(
            select(BacktestResult)
            .where(BacktestResult.experiment_id == exp.id)
            .order_by(desc(BacktestResult.id))
        )
        val = db.scalar(
            select(ValidationResult)
            .where(ValidationResult.experiment_id == exp.id)
            .order_by(desc(ValidationResult.id))
        )
        if not sdef or not ds or not bt:
            continue
        if symbol and ds.symbol != symbol:
            continue
        if timeframe and ds.timeframe != timeframe:
            continue
        if family and sdef.family != family:
            continue
        if validation_tier and exp.validation_tier != validation_tier:
            continue
        if min_sharpe is not None and bt.sharpe_ratio < min_sharpe:
            continue
        if bt.total_trades < min_trades:
            continue
        if max_drawdown is not None and bt.max_drawdown_pct > max_drawdown:
            continue

        rows.append(
            {
                "experiment_id": exp.id,
                "experiment_uid": exp.experiment_uid,
                "strategy_name": sdef.name,
                "family": sdef.family,
                "direction": sdef.direction,
                "symbol": ds.symbol,
                "timeframe": ds.timeframe,
                "market_type": ds.market_type,
                "validation_tier": exp.validation_tier,
                "composite_score": exp.composite_score,
                "sharpe_ratio": bt.sharpe_ratio,
                "profit_factor": bt.profit_factor,
                "win_rate": bt.win_rate,
                "max_drawdown_pct": bt.max_drawdown_pct,
                "return_pct": bt.return_on_capital_pct,
                "total_trades": bt.total_trades,
                "pbo_score": val.pbo_score if val else None,
                "deflated_sharpe": val.deflated_sharpe if val else None,
                "passed_gates": val.passed_gates if val else False,
            }
        )
    tier_rank = {"VALIDATED": 3, "MONTE_CARLO_TESTED": 2, "OOS_TESTED": 1, "BACKTESTED": 0}
    rows.sort(
        key=lambda r: (tier_rank.get(r["validation_tier"], 0), r["composite_score"], r["sharpe_ratio"]),
        reverse=True,
    )
    return rows[:limit]


@router.get("/experiments/{experiment_id}/export.json")
def export_experiment_json(
    experiment_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> Response:
    exp = db.get(StrategyExperiment, experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    sdef = db.get(StrategyDefinition, exp.strategy_id)
    sver = db.get(StrategyVersion, exp.strategy_version_id)
    ds = db.get(MarketDataset, exp.dataset_id)
    bts = list(db.scalars(select(BacktestResult).where(BacktestResult.experiment_id == exp.id)).all())
    val = db.scalar(select(ValidationResult).where(ValidationResult.experiment_id == exp.id).order_by(desc(ValidationResult.id)))
    mc = db.scalar(select(MonteCarloResult).where(MonteCarloResult.experiment_id == exp.id).order_by(desc(MonteCarloResult.id)))
    wf = db.scalar(select(WalkForwardResult).where(WalkForwardResult.experiment_id == exp.id).order_by(desc(WalkForwardResult.id)))

    payload = {
        "experiment_uid": exp.experiment_uid,
        "software_version": exp.software_version,
        "random_seed": exp.random_seed,
        "search_algorithm": exp.search_algorithm,
        "validation_tier": exp.validation_tier,
        "composite_score": exp.composite_score,
        "created_at": exp.created_at.isoformat(),
        "dataset": {"id": ds.id, "symbol": ds.symbol, "timeframe": ds.timeframe, "version": ds.version, "provenance": ds.provenance} if ds else None,
        "strategy": {"uid": sdef.strategy_uid, "name": sdef.name, "family": sdef.family, "rules": sdef.rule_json, "params": sver.params_json if sver else {}} if sdef else None,
        "backtests": [{"engine": b.engine_type, "split": b.split_segment, "metrics": b.metrics_json} for b in bts],
        "validation": val.details_json if val else None,
        "monte_carlo": mc.summary_json if mc else None,
        "walk_forward": {"mode": wf.mode, "consistency_pct": wf.consistency_pct, "windows": wf.windows_json} if wf else None,
    }
    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{exp.experiment_uid}.json"'},
    )


@router.get("/experiments/{experiment_id}/export.csv")
def export_experiment_csv(
    experiment_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> Response:
    exp = db.get(StrategyExperiment, experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    bt = db.scalar(select(BacktestResult).where(BacktestResult.experiment_id == exp.id).order_by(desc(BacktestResult.id)))
    if bt is None:
        raise HTTPException(status_code=404, detail="No backtest results found")
    trades = list(db.scalars(select(TradeRecord).where(TradeRecord.backtest_id == bt.id)).all())

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["experiment_uid", "side", "entry_time_ms", "exit_time_ms", "entry_price", "exit_price", "quantity", "gross_pnl", "fee_paid", "slippage_paid", "funding_paid", "net_pnl", "return_pct", "exit_reason", "regime"])
    for t in trades:
        writer.writerow([exp.experiment_uid, t.side, t.entry_time_ms, t.exit_time_ms, t.entry_price, t.exit_price, t.quantity, t.gross_pnl, t.fee_paid, t.slippage_paid, t.funding_paid, t.net_pnl, t.return_pct, t.exit_reason, t.regime_at_entry])

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{exp.experiment_uid}_trades.csv"'},
    )


@router.post("/portfolio/construct")
def build_portfolio(
    payload: PortfolioConstructRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    names: list[str] = []
    return_series: list[np.ndarray] = []
    valid_ids: list[int] = []

    for eid in payload.experiment_ids:
        exp = db.get(StrategyExperiment, eid)
        if exp is None:
            continue
        sdef = db.get(StrategyDefinition, exp.strategy_id)
        ds = db.get(MarketDataset, exp.dataset_id)
        if sdef is None or ds is None:
            continue
        sver = db.get(StrategyVersion, exp.strategy_version_id)
        if sver is None:
            continue
        df = read_parquet(ds.parquet_path).tail(1500)
        ctx = IndicatorCache(df, timeframe=ds.timeframe)
        spec = StrategySpec.from_dicts(sdef.name, sdef.rule_json, sver.params_json, sver.version)
        bt = evaluate_strategy(spec, ctx, engine_type="vectorized")
        valid_ids.append(eid)
        names.append(f"{sdef.name} ({ds.symbol} {ds.timeframe})")
        return_series.append(bt.bar_returns)

    if not valid_ids:
        raise HTTPException(status_code=400, detail="No valid strategy experiments found")

    res = construct_portfolio(
        strategy_ids=valid_ids,
        strategy_names=names,
        return_series_list=return_series,
        allocation_method=payload.allocation_method,
        initial_capital=payload.initial_capital,
        max_single_weight=payload.max_single_weight,
    )

    existing = db.scalar(select(PortfolioDefinition).where(PortfolioDefinition.name == payload.name))
    if existing:
        db.delete(existing)
        db.flush()

    port = PortfolioDefinition(
        name=payload.name,
        allocation_method=payload.allocation_method,
        initial_capital=payload.initial_capital,
        risk_config_json={"max_single_weight": payload.max_single_weight},
        metrics_json=res["metrics"],
        correlation_matrix_json={
            "labels": names,
            "return_corr": res["return_correlation_matrix"],
            "drawdown_corr": res["drawdown_correlation_matrix"],
        },
    )
    db.add(port)
    db.flush()
    for alloc in res["allocations"]:
        db.add(
            PortfolioAllocation(
                portfolio_id=port.id,
                experiment_id=alloc["experiment_id"],
                weight=alloc["weight"],
                risk_contribution=alloc["risk_contribution_pct"],
                cluster_id=alloc["cluster_id"],
            )
        )
    db.commit()
    return {"portfolio_id": port.id, "name": port.name, **res}


@router.post("/paper/deploy")
def api_paper_deploy(
    payload: PaperDeployRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    try:
        pos = deploy_paper_strategy(
            db,
            experiment_id=payload.experiment_id,
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            market_type=payload.market_type,
            allocated_capital=payload.allocated_capital,
        )
        return {"position_id": pos.id, "symbol": pos.symbol, "is_active": pos.is_active}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/paper/step")
def api_paper_step(
    max_daily_loss_pct: float = 5.0,
    max_portfolio_drawdown_pct: float = 20.0,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    cfg = RiskConfig(max_daily_loss_pct=max_daily_loss_pct, max_portfolio_drawdown_pct=max_portfolio_drawdown_pct)
    return step_paper_trading_cycle(db, risk_config=cfg)


@router.post("/paper/positions/{position_id}/close")
def api_paper_close(position_id: int, db: Session = Depends(get_db), _user: User = Depends(require_user)) -> dict[str, Any]:
    pos = force_close_paper_position(db, position_id)
    return {"position_id": pos.id, "side": pos.side, "realized_pnl": round(pos.realized_pnl, 2)}


@router.post("/paper/positions/{position_id}/stop")
def api_paper_stop(position_id: int, db: Session = Depends(get_db), _user: User = Depends(require_user)) -> dict[str, Any]:
    pos = stop_paper_strategy(db, position_id)
    return {"position_id": pos.id, "is_active": pos.is_active}


@router.post("/live/credentials")
def api_live_save_credentials(
    payload: LiveCredentialRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    cfg = save_exchange_credentials(
        db=db,
        exchange=payload.exchange,
        market_type=payload.market_type,
        label=payload.label,
        api_key=payload.api_key,
        api_secret=payload.api_secret,
        permissions_scope=payload.permissions_scope,
        max_capital_usdt=payload.max_capital_usdt,
        max_exposure_pct=payload.max_exposure_pct,
    )
    return {"id": cfg.id, "label": cfg.label, "live_trading_enabled": cfg.live_trading_enabled}


@router.post("/live/{config_id}/toggle")
def api_live_toggle(
    config_id: int,
    payload: LiveToggleRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_user),
) -> dict[str, Any]:
    try:
        cfg = toggle_live_trading(db, config_id, payload.enable, payload.confirmation_phrase)
        return {"id": cfg.id, "live_trading_enabled": cfg.live_trading_enabled}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/live/kill-switch")
def api_live_kill_switch(db: Session = Depends(get_db), _user: User = Depends(require_user)) -> dict[str, str]:
    trigger_emergency_kill_switch(db)
    return {"status": "halted"}


@router.get("/live/{config_id}/check")
def api_live_check(config_id: int, db: Session = Depends(get_db), _user: User = Depends(require_user)) -> dict[str, Any]:
    return check_exchange_connectivity(db, config_id)
