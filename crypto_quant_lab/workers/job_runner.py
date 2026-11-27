"""Persistent background job orchestrator supporting pause, resume, cancellation, checkpointing, and restart recovery."""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from crypto_quant_lab import __version__
from crypto_quant_lab.app.config import get_settings
from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.core.logging import log_system_event
from crypto_quant_lab.data.downloaders.service import download_and_store_dataset
from crypto_quant_lab.data.storage.parquet_store import read_parquet
from crypto_quant_lab.database.models import (
    BacktestResult,
    FundingRate,
    MarketDataset,
    MonteCarloResult,
    ResearchJob,
    StrategyDefinition,
    StrategyExperiment,
    StrategyVersion,
    TradeRecord,
    ValidationResult,
    WalkForwardResult,
)
from crypto_quant_lab.database.session import get_session_factory
from crypto_quant_lab.strategies.evaluator import IndicatorCache, evaluate_strategy
from crypto_quant_lab.strategies.generator import stream_candidate_strategies
from crypto_quant_lab.strategies.genetic import run_genetic_discovery
from crypto_quant_lab.strategies.grammar import StrategySpec
from crypto_quant_lab.strategies.optimizer import run_optuna_optimization
from crypto_quant_lab.validation.monte_carlo import run_monte_carlo_simulation
from crypto_quant_lab.validation.out_of_sample import split_chronological
from crypto_quant_lab.validation.overfitting import (
    compute_deflated_sharpe_ratio,
    parameter_stability_analysis,
)
from crypto_quant_lab.validation.ranking import evaluate_and_rank_strategy
from crypto_quant_lab.validation.stress_testing import run_stress_tests
from crypto_quant_lab.validation.walk_forward import run_walk_forward_analysis

logger = logging.getLogger("crypto_quant_lab.workers")

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cql-worker")
_active_locks: dict[int, threading.Lock] = {}


def load_funding_rates_map(db: Session, symbol: str, market_type: str) -> dict[int, float] | None:
    """Load stored historical FundingRate rows snapped to exact 8h UTC settlement boundaries."""
    if market_type != "futures":
        return None
    rows = list(
        db.scalars(
            select(FundingRate).where(FundingRate.symbol == symbol.upper()).order_by(FundingRate.funding_time_ms)
        ).all()
    )
    if not rows:
        return None
    out: dict[int, float] = {}
    eight_h = 28_800_000
    for r in rows:
        ts = int(r.funding_time_ms)
        snapped = round(ts / eight_h) * eight_h
        if abs(ts - snapped) <= 60_000:
            out[snapped] = float(r.funding_rate)
        else:
            out[ts] = float(r.funding_rate)
    return out


def create_and_dispatch_job(db: Session, job_type: str, config: dict[str, Any], total_items: int = 1) -> ResearchJob:
    job = ResearchJob(
        job_uid=f"JOB-{uuid.uuid4().hex[:10].upper()}",
        job_type=job_type,
        status="queued",
        config_json=config,
        checkpoint_json={"offset": 0},
        total_items=max(1, total_items),
        completed_items=0,
        failed_items=0,
        progress_pct=0.0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    dispatch_job(job.id)
    return job


def dispatch_job(job_id: int) -> None:
    settings = get_settings()
    if settings.celery_enabled:
        from crypto_quant_lab.workers.celery_app import run_research_job_celery

        async_res = run_research_job_celery.delay(job_id)
        factory = get_session_factory()
        with factory() as db:
            job = db.get(ResearchJob, job_id)
            if job:
                job.celery_task_id = str(async_res.id)
                db.commit()
    else:
        _executor.submit(execute_job_by_id, job_id)


def recover_interrupted_jobs() -> int:
    """Resume any queued or running research jobs that were interrupted by a server restart."""
    factory = get_session_factory()
    recovered = 0
    with factory() as db:
        stale_jobs = list(
            db.scalars(select(ResearchJob).where(ResearchJob.status.in_(["queued", "running"]))).all()
        )
        for job in stale_jobs:
            log_system_event(db, "WARNING", "workers.recovery", f"Recovering interrupted job {job.job_uid} ({job.job_type})")
            dispatch_job(job.id)
            recovered += 1
    return recovered


def _persist_experiment_result(
    db: Session,
    job_id: int | None,
    dataset: MarketDataset,
    spec: StrategySpec,
    bt_out: Any,
    algorithm: str,
    seed: int,
    split_segment: str,
    engine_type: str = "vectorized",
    save_trades: bool = False,
    n_trials_attempted: int = 1,
) -> StrategyExperiment:
    """Persist unique strategy definition, version, experiment, and backtest metrics cleanly."""
    ast_hash = spec.ast_hash()
    s_def = db.scalar(select(StrategyDefinition).where(StrategyDefinition.ast_hash == ast_hash))
    if s_def is None:
        s_def = StrategyDefinition(
            strategy_uid=f"STR-{uuid.uuid4().hex[:10].upper()}",
            name=spec.name,
            family=spec.family,
            direction=spec.direction,
            ast_hash=ast_hash,
            rule_json=spec.to_rule_dict(),
            description=f"Auto-generated {spec.family} ({spec.direction}) strategy",
        )
        db.add(s_def)
        db.flush()

    p_hash = spec.param_hash()
    s_ver = db.scalar(
        select(StrategyVersion).where(
            StrategyVersion.strategy_id == s_def.id,
            StrategyVersion.param_hash == p_hash,
        )
    )
    if s_ver is None:
        s_ver = StrategyVersion(
            strategy_id=s_def.id,
            version=spec.version,
            params_json=spec.to_params_dict(),
            param_hash=p_hash,
        )
        db.add(s_ver)
        db.flush()

    exp = StrategyExperiment(
        experiment_uid=f"EXP-{uuid.uuid4().hex[:10].upper()}",
        research_job_id=job_id,
        strategy_id=s_def.id,
        strategy_version_id=s_ver.id,
        dataset_id=dataset.id,
        search_algorithm=algorithm,
        random_seed=seed,
        software_version=__version__,
        backtest_config_json={
            "engine_type": engine_type,
            "market_type": dataset.market_type,
            "split_segment": split_segment,
            "n_trials_attempted": max(1, int(n_trials_attempted)),
        },
        validation_config_json={},
        validation_tier="BACKTESTED",
        composite_score=round(float(bt_out.metrics.get("sharpe_ratio", 0.0)), 4),
        status="completed",
    )
    db.add(exp)
    db.flush()

    m = bt_out.metrics
    bt_row = BacktestResult(
        experiment_id=exp.id,
        engine_type=engine_type,
        split_segment=split_segment,
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

    if save_trades and bt_out.trades:
        for tr in bt_out.trades[:2000]:
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
    return exp


def execute_job_by_id(job_id: int) -> None:
    """Execute a queued/resumed ResearchJob with database checkpointing and pause/cancel responsiveness."""
    factory = get_session_factory()
    with factory() as db:
        job = db.get(ResearchJob, job_id)
        if job is None or job.status in ("completed", "cancelled", "paused"):
            return
        job.status = "running"
        if job.started_at is None:
            job.started_at = datetime.now(timezone.utc)
        db.commit()

    try:
        with factory() as db:
            job = db.get(ResearchJob, job_id)
            if job is None:
                return
            jtype = job.job_type
            cfg = dict(job.config_json)

        if jtype == "data_download":
            _run_download_job(job_id, cfg)
        elif jtype == "strategy_discovery":
            _run_discovery_job(job_id, cfg)
        elif jtype == "validation_suite":
            _run_validation_job(job_id, cfg)
        else:
            raise ValueError(f"Unknown job type: {jtype}")

    except Exception as exc:
        logger.exception("Research job %s failed", job_id)
        with factory() as db:
            job = db.get(ResearchJob, job_id)
            if job and job.status not in ("paused", "cancelled"):
                job.status = "failed"
                job.error_message = str(exc)[:1000]
                db.commit()
                log_system_event(db, "ERROR", "workers.job_runner", f"Job {job.job_uid} failed: {exc}")


def _run_download_job(job_id: int, cfg: dict[str, Any]) -> None:
    factory = get_session_factory()
    with factory() as db:
        def _prog(done: int, target: int | None) -> None:
            j = db.get(ResearchJob, job_id)
            if j:
                j.completed_items = done
                if target:
                    j.total_items = target
                    j.progress_pct = min(99.0, done / max(1, target) * 100.0)
                db.commit()

        ds = download_and_store_dataset(
            db=db,
            symbol=cfg["symbol"],
            timeframe=cfg["timeframe"],
            market_type=cfg.get("market_type", "spot"),
            max_candles=cfg.get("max_candles", 5000),
            progress_cb=_prog,
        )
        j = db.get(ResearchJob, job_id)
        if j:
            j.status = "completed"
            j.completed_items = ds.candle_count
            j.total_items = ds.candle_count
            j.progress_pct = 100.0
            j.completed_at = datetime.now(timezone.utc)
            j.checkpoint_json = {"dataset_id": ds.id}
            db.commit()


def _run_discovery_job(job_id: int, cfg: dict[str, Any]) -> None:
    """
    Execute large-scale strategy discovery & chronological train-split backtesting.
    Streams candidates in batches without loading all 20,000+ objects into RAM,
    supports pause/resume via checkpoint offset, and persists top strategies + aggregate telemetry.
    """
    factory = get_session_factory()
    dataset_id = int(cfg["dataset_id"])
    candidate_count = int(cfg.get("candidate_count", 500))
    algorithm = cfg.get("algorithm", "random_search")
    families = cfg.get("families") or None
    directions = cfg.get("directions") or None
    categories = set(cfg["categories"]) if cfg.get("categories") else None
    seed = int(cfg.get("seed", 42))
    persist_top_n = int(cfg.get("persist_top_n", 100))

    with factory() as db:
        ds = db.get(MarketDataset, dataset_id)
        if ds is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        job = db.get(ResearchJob, job_id)
        offset = int((job.checkpoint_json or {}).get("offset", 0)) if job else 0
        funding_map = load_funding_rates_map(db, ds.symbol, ds.market_type)

    if ds.market_type == "spot":
        directions = ["long_only"]

    df = read_parquet(ds.parquet_path)
    splits = split_chronological(df)
    train_ctx = IndicatorCache(splits.train, timeframe=ds.timeframe)
    costs = ExecutionCostConfig(
        market_type=ds.market_type,
        maker_fee_bps=2.0 if ds.market_type == "futures" else 10.0,
        taker_fee_bps=5.0 if ds.market_type == "futures" else 10.0,
    )

    # For genetic or optuna algorithms
    if algorithm in ("genetic_algorithm", "evolutionary"):
        gens = max(2, min(10, candidate_count // 20))
        pop_size = max(10, candidate_count // gens)
        results = run_genetic_discovery(
            train_ctx,
            population_size=pop_size,
            generations=gens,
            families=families,
            directions=directions,
            seed=seed,
            costs=costs,
            funding_rates_map=funding_map,
        )
        with factory() as db:
            ds_db = db.get(MarketDataset, dataset_id)
            for spec, _ in results[:persist_top_n]:
                bt_out = evaluate_strategy(spec, train_ctx, engine_type="vectorized", costs=costs, funding_rates_map=funding_map)
                _persist_experiment_result(
                    db, job_id, ds_db, spec, bt_out, algorithm, seed, "train",
                    save_trades=True, n_trials_attempted=max(candidate_count, len(results)),
                )
            j = db.get(ResearchJob, job_id)
            if j:
                j.completed_items = len(results)
                j.total_items = len(results)
                j.progress_pct = 100.0
                j.status = "completed"
                j.completed_at = datetime.now(timezone.utc)
            db.commit()
        return

    if algorithm == "optuna":
        results = run_optuna_optimization(
            train_ctx,
            n_trials=candidate_count,
            families=families,
            directions=directions,
            seed=seed,
            costs=costs,
            funding_rates_map=funding_map,
        )
        with factory() as db:
            ds_db = db.get(MarketDataset, dataset_id)
            for spec, _ in results[:persist_top_n]:
                bt_out = evaluate_strategy(spec, train_ctx, engine_type="vectorized", costs=costs, funding_rates_map=funding_map)
                _persist_experiment_result(
                    db, job_id, ds_db, spec, bt_out, algorithm, seed, "train",
                    save_trades=True, n_trials_attempted=max(candidate_count, len(results)),
                )
            j = db.get(ResearchJob, job_id)
            if j:
                j.completed_items = len(results)
                j.total_items = len(results)
                j.progress_pct = 100.0
                j.status = "completed"
                j.completed_at = datetime.now(timezone.utc)
            db.commit()
        return

    # Streaming batch evaluation (random_search or grid_search) supporting 20,000+ candidates
    stream = stream_candidate_strategies(
        count=candidate_count,
        families=families,
        directions=directions,
        algorithm=algorithm,
        seed=seed,
        allowed_categories=categories,
    )

    batch_size = 100
    top_heap: list[tuple[float, StrategySpec, Any]] = []
    processed = 0
    failed = 0

    for idx, spec in enumerate(stream):
        if idx < offset:
            processed = idx + 1
            continue

        # Check pause/cancel every 25 strategies
        if idx % 25 == 0:
            with factory() as db:
                j = db.get(ResearchJob, job_id)
                if j is None or j.status in ("paused", "cancelled"):
                    if j:
                        j.checkpoint_json = {"offset": idx}
                        db.commit()
                    return

        try:
            bt_out = evaluate_strategy(spec, train_ctx, engine_type="vectorized", costs=costs, funding_rates_map=funding_map)
            m = bt_out.metrics
            trades_n = int(m["total_trades"])
            score = float(m["sharpe_ratio"]) - 0.02 * float(m["max_drawdown_pct"]) if trades_n >= 5 else -999.0
            if len(top_heap) < persist_top_n:
                top_heap.append((score, spec, bt_out))
                top_heap.sort(key=lambda x: x[0])
            elif score > top_heap[0][0]:
                top_heap[0] = (score, spec, bt_out)
                top_heap.sort(key=lambda x: x[0])
        except Exception:
            failed += 1

        processed = idx + 1
        if processed % batch_size == 0 or processed == candidate_count:
            with factory() as db:
                j = db.get(ResearchJob, job_id)
                if j:
                    j.completed_items = processed
                    j.failed_items = failed
                    j.progress_pct = round(min(99.5, processed / max(1, candidate_count) * 100.0), 2)
                    j.checkpoint_json = {"offset": processed}
                    db.commit()

    # Persist top performing candidates & record total attempted count for DSR multiple-testing adjustment
    top_heap.sort(key=lambda x: x[0], reverse=True)
    with factory() as db:
        ds_db = db.get(MarketDataset, dataset_id)
        for rank_idx, (_, spec, bt_out) in enumerate(top_heap):
            _persist_experiment_result(
                db,
                job_id,
                ds_db,
                spec,
                bt_out,
                algorithm,
                seed,
                "train",
                save_trades=(rank_idx < 20),
                n_trials_attempted=max(candidate_count, processed),
            )
        j = db.get(ResearchJob, job_id)
        if j and j.status not in ("paused", "cancelled"):
            j.completed_items = processed
            j.failed_items = failed
            j.progress_pct = 100.0
            j.status = "completed"
            j.completed_at = datetime.now(timezone.utc)
            j.checkpoint_json = {"offset": processed, "persisted_top": len(top_heap)}
        db.commit()


def run_validation_suite_sync(
    db: Session,
    experiment_id: int,
    mc_simulations: int = 1000,
    wf_windows: int = 5,
    wf_mode: str = "rolling",
    n_trials_attempted: int = 100,
    unlock_final_test: bool = False,
) -> dict[str, Any]:
    """Run full statistical validation suite (OOS, Walk-Forward, Monte Carlo, CSCV PBO, DSR, Stress, Ranking)."""
    exp = db.get(StrategyExperiment, experiment_id)
    if exp is None:
        raise ValueError(f"Experiment {experiment_id} not found")
    ds = db.get(MarketDataset, exp.dataset_id)
    strat_def = db.get(StrategyDefinition, exp.strategy_id)
    strat_ver = db.get(StrategyVersion, exp.strategy_version_id)
    if ds is None or strat_def is None or strat_ver is None:
        raise ValueError("Missing strategy or dataset records")

    spec = StrategySpec.from_dicts(strat_def.name, strat_def.rule_json, strat_ver.params_json, strat_ver.version)
    funding_map = load_funding_rates_map(db, ds.symbol, ds.market_type)
    df = read_parquet(ds.parquet_path)
    splits = split_chronological(df)
    costs = ExecutionCostConfig(
        market_type=ds.market_type,
        maker_fee_bps=2.0 if ds.market_type == "futures" else 10.0,
        taker_fee_bps=5.0 if ds.market_type == "futures" else 10.0,
    )

    # 1. Validation OOS slice (never touches final test unless explicitly unlocked)
    val_df = splits.test if unlock_final_test else splits.validation
    val_segment_name = "test" if unlock_final_test else "validation"
    val_ctx = IndicatorCache(val_df, timeframe=ds.timeframe)
    oos_out = evaluate_strategy(spec, val_ctx, engine_type="event_driven", costs=costs, funding_rates_map=funding_map)

    db.add(
        BacktestResult(
            experiment_id=exp.id,
            engine_type="event_driven",
            split_segment=val_segment_name,
            net_profit=float(oos_out.metrics["net_profit"]),
            gross_profit=float(oos_out.metrics["gross_profit"]),
            gross_loss=float(oos_out.metrics["gross_loss"]),
            profit_factor=float(oos_out.metrics["profit_factor"]),
            win_rate=float(oos_out.metrics["win_rate"]),
            expectancy=float(oos_out.metrics["expectancy"]),
            sharpe_ratio=float(oos_out.metrics["sharpe_ratio"]),
            sortino_ratio=float(oos_out.metrics["sortino_ratio"]),
            calmar_ratio=float(oos_out.metrics["calmar_ratio"]),
            max_drawdown_pct=float(oos_out.metrics["max_drawdown_pct"]),
            avg_drawdown_pct=float(oos_out.metrics["avg_drawdown_pct"]),
            recovery_factor=float(oos_out.metrics["recovery_factor"]),
            ulcer_index=float(oos_out.metrics["ulcer_index"]),
            return_on_capital_pct=float(oos_out.metrics["return_on_capital_pct"]),
            annualized_return_pct=float(oos_out.metrics["annualized_return_pct"]),
            total_trades=int(oos_out.metrics["total_trades"]),
            exposure_pct=float(oos_out.metrics["exposure_pct"]),
            metrics_json=oos_out.metrics,
            equity_curve_json=oos_out.equity_curve,
            regime_breakdown_json=oos_out.metrics.get("regime_breakdown", {}),
        )
    )

    # 2. Walk-Forward on train+validation (keeping final test untouched unless unlocked)
    wf_df = df if unlock_final_test else df.slice(0, splits.train.height + splits.validation.height)
    wf_res = run_walk_forward_analysis(wf_df, spec, timeframe=ds.timeframe, mode=wf_mode, num_windows=wf_windows, costs=costs)
    db.add(
        WalkForwardResult(
            experiment_id=exp.id,
            mode=wf_res["mode"],
            num_windows=wf_res["num_windows"],
            consistency_pct=wf_res["consistency_pct"],
            avg_oos_sharpe=wf_res["avg_oos_sharpe"],
            wf_efficiency_ratio=wf_res["wf_efficiency_ratio"],
            windows_json=wf_res["windows"],
        )
    )

    # 3. Monte Carlo simulation on OOS trades
    mc_res = run_monte_carlo_simulation(
        oos_out.trades,
        simulations=mc_simulations,
        method="trade_permutation",
        seed=exp.random_seed,
    )
    db.add(
        MonteCarloResult(
            experiment_id=exp.id,
            method=mc_res["method"],
            simulations=mc_res["simulations"],
            median_return_pct=mc_res["median_return_pct"],
            p05_return_pct=mc_res["p05_return_pct"],
            p95_return_pct=mc_res["p95_return_pct"],
            median_max_dd_pct=mc_res["median_max_dd_pct"],
            p95_max_dd_pct=mc_res["p95_max_dd_pct"],
            ruin_probability=mc_res["ruin_probability"],
            summary_json=mc_res,
        )
    )

    # 4. Overfitting controls: Parameter Stability, CSCV PBO, Deflated Sharpe Ratio
    stab_res = parameter_stability_analysis(val_ctx, spec, costs=costs)
    dsr_res = compute_deflated_sharpe_ratio(
        observed_sharpe=float(oos_out.metrics["sharpe_ratio"]),
        bar_returns=oos_out.bar_returns,
        n_trials=max(1, n_trials_attempted),
    )

    # 5. Stress Tests
    stress_res = run_stress_tests(val_df, spec, timeframe=ds.timeframe, base_costs=costs)

    # 6. Weighted Ranking & Hard Rejection Gates
    ranking = evaluate_and_rank_strategy(
        oos_metrics=oos_out.metrics,
        wf_result=wf_res,
        mc_result=mc_res,
        stability_result=stab_res,
        stress_result=stress_res,
        dsr_result=dsr_res,
    )

    db.add(
        ValidationResult(
            experiment_id=exp.id,
            oos_sharpe=float(oos_out.metrics["sharpe_ratio"]),
            oos_return_pct=float(oos_out.metrics["return_on_capital_pct"]),
            oos_max_drawdown_pct=float(oos_out.metrics["max_drawdown_pct"]),
            pbo_score=float(stab_res["cscv_pbo"]["pbo"]),
            deflated_sharpe=float(dsr_res["dsr"]),
            param_stability_score=float(stab_res["stability_score"]),
            stress_survival_rate=float(stress_res["survival_rate"]),
            passed_gates=bool(ranking["passed_gates"]),
            rejection_reasons_json=ranking["rejection_reasons"],
            details_json={
                "dsr": dsr_res,
                "stability": stab_res,
                "stress": stress_res,
                "ranking": ranking,
                "split_segment": val_segment_name,
            },
        )
    )

    exp.validation_tier = "VALIDATED" if ranking["passed_gates"] else "MONTE_CARLO_TESTED"
    exp.composite_score = float(ranking["composite_score"])
    exp.validation_config_json = {
        "mc_simulations": mc_simulations,
        "wf_windows": wf_windows,
        "wf_mode": wf_mode,
        "n_trials_attempted": n_trials_attempted,
        "split_segment": val_segment_name,
    }
    db.commit()

    return {
        "experiment_id": exp.id,
        "validation_tier": exp.validation_tier,
        "oos_metrics": oos_out.metrics,
        "walk_forward": wf_res,
        "monte_carlo": mc_res,
        "stability": stab_res,
        "dsr": dsr_res,
        "stress": stress_res,
        "ranking": ranking,
    }


def _run_validation_job(job_id: int, cfg: dict[str, Any]) -> None:
    factory = get_session_factory()
    exp_ids: list[int] = [int(x) for x in cfg.get("experiment_ids", [])]
    with factory() as db:
        for i, eid in enumerate(exp_ids):
            j = db.get(ResearchJob, job_id)
            if j is None or j.status in ("paused", "cancelled"):
                return
            run_validation_suite_sync(
                db=db,
                experiment_id=eid,
                mc_simulations=int(cfg.get("mc_simulations", 1000)),
                wf_windows=int(cfg.get("wf_windows", 5)),
                wf_mode=cfg.get("wf_mode", "rolling"),
                n_trials_attempted=int(cfg.get("n_trials_attempted", 100)),
                unlock_final_test=bool(cfg.get("unlock_final_test", False)),
            )
            j.completed_items = i + 1
            j.progress_pct = round((i + 1) / max(1, len(exp_ids)) * 100.0, 2)
            db.commit()

        j = db.get(ResearchJob, job_id)
        if j and j.status not in ("paused", "cancelled"):
            j.status = "completed"
            j.progress_pct = 100.0
            j.completed_at = datetime.now(timezone.utc)
            db.commit()
