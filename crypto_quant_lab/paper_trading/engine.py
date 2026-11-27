"""Persistent Paper Trading Engine connected to Binance public market data."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.core.exceptions import RiskViolationError
from crypto_quant_lab.core.logging import log_system_event
from crypto_quant_lab.data.exchanges.binance import BinanceAdapter
from crypto_quant_lab.data.storage.parquet_store import rows_to_polars
from crypto_quant_lab.database.models import (
    PaperOrder,
    PaperPosition,
    PaperTrade,
    StrategyDefinition,
    StrategyExperiment,
    StrategyVersion,
)
from crypto_quant_lab.risk.manager import RiskConfig, RiskManager
from crypto_quant_lab.strategies.evaluator import IndicatorCache, evaluate_condition
from crypto_quant_lab.strategies.grammar import StrategySpec, validate_strategy_spec

logger = logging.getLogger("crypto_quant_lab.paper_trading")


def deploy_paper_strategy(
    db: Session,
    experiment_id: int,
    symbol: str,
    timeframe: str = "1h",
    market_type: str = "spot",
    allocated_capital: float = 10_000.0,
) -> PaperPosition:
    """Register or reactivate a strategy experiment for live paper trading."""
    exp = db.get(StrategyExperiment, experiment_id)
    if exp is None:
        raise ValueError(f"Experiment {experiment_id} not found")
    strat_def = db.get(StrategyDefinition, exp.strategy_id)
    strat_ver = db.get(StrategyVersion, exp.strategy_version_id)
    if strat_def is not None and strat_ver is not None:
        spec = StrategySpec.from_dicts(strat_def.name, strat_def.rule_json, strat_ver.params_json, strat_ver.version)
        validate_strategy_spec(spec, market_type=market_type)

    existing = db.scalar(
        select(PaperPosition).where(
            PaperPosition.experiment_id == experiment_id,
            PaperPosition.symbol == symbol.upper(),
        )
    )
    if existing:
        existing.is_active = True
        existing.timeframe = timeframe
        existing.market_type = market_type
        if existing.allocated_capital <= 0:
            existing.allocated_capital = allocated_capital
        db.commit()
        db.refresh(existing)
        return existing

    pos = PaperPosition(
        experiment_id=experiment_id,
        symbol=symbol.upper(),
        timeframe=timeframe,
        market_type=market_type,
        is_active=True,
        side="FLAT",
        allocated_capital=allocated_capital,
        peak_equity=allocated_capital,
    )
    db.add(pos)
    db.commit()
    db.refresh(pos)
    log_system_event(
        db,
        "INFO",
        "paper_trading",
        f"Deployed strategy experiment {exp.experiment_uid} to paper trading on {symbol.upper()} ({timeframe})",
        {"experiment_id": experiment_id, "symbol": symbol.upper(), "timeframe": timeframe},
    )
    return pos


def stop_paper_strategy(db: Session, position_id: int) -> PaperPosition:
    pos = db.get(PaperPosition, position_id)
    if pos is None:
        raise ValueError("Paper position not found")
    pos.is_active = False
    db.commit()
    db.refresh(pos)
    return pos


def step_paper_trading_cycle(db: Session, risk_config: RiskConfig | None = None) -> dict[str, Any]:
    """
    Execute one synchronous paper trading cycle across all active paper positions:
    - Fetches latest real Binance public klines & ticker prices.
    - Updates mark-to-market unrealized P&L and drawdown.
    - Processes only NEW closed candles (idempotent across restarts, zero duplicate trades).
    - Enforces RiskManager limits before opening virtual orders.
    """
    active_positions = list(db.scalars(select(PaperPosition).where(PaperPosition.is_active.is_(True))).all())
    if not active_positions:
        return {"active_strategies": 0, "orders_executed": 0, "updated": 0}

    adapter = BinanceAdapter()
    risk_mgr = RiskManager(risk_config or RiskConfig())
    orders_executed = 0
    errors: list[str] = []

    try:
        # Portfolio-level risk snapshot
        all_open = sum(1 for p in active_positions if p.side != "FLAT")
        total_equity = sum(p.allocated_capital + p.realized_pnl + p.unrealized_pnl for p in active_positions)
        total_peak = sum(p.peak_equity for p in active_positions)

        for pos in active_positions:
            try:
                exp = db.get(StrategyExperiment, pos.experiment_id)
                if exp is None:
                    continue
                strat_def = db.get(StrategyDefinition, exp.strategy_id)
                strat_ver = db.get(StrategyVersion, exp.strategy_version_id)
                if strat_def is None or strat_ver is None:
                    continue

                spec = StrategySpec.from_dicts(strat_def.name, strat_def.rule_json, strat_ver.params_json, strat_ver.version)
                cost_cfg = ExecutionCostConfig(
                    market_type=pos.market_type,
                    maker_fee_bps=2.0 if pos.market_type == "futures" else 10.0,
                    taker_fee_bps=5.0 if pos.market_type == "futures" else 10.0,
                )

                # Fetch latest 250 candles + live mark price from Binance public API
                rows = adapter.fetch_klines_page(pos.symbol, pos.timeframe, market_type=pos.market_type, limit=250)
                live_price = adapter.fetch_latest_price(pos.symbol, market_type=pos.market_type)
                if len(rows) < 60:
                    continue

                now_ms = int(time.time() * 1000)
                closed_rows = [r for r in rows if r.close_time_ms <= now_ms]
                if len(closed_rows) < 55:
                    closed_rows = rows[:-1]
                latest_closed = closed_rows[-1]

                # Update mark price & unrealized P&L
                pos.mark_price = live_price
                if pos.side == "LONG":
                    pos.unrealized_pnl = (live_price - pos.entry_price) * pos.quantity
                elif pos.side == "SHORT":
                    pos.unrealized_pnl = (pos.entry_price - live_price) * pos.quantity
                else:
                    pos.unrealized_pnl = 0.0

                cur_eq = pos.allocated_capital + pos.realized_pnl + pos.unrealized_pnl
                if cur_eq > pos.peak_equity:
                    pos.peak_equity = cur_eq
                if pos.peak_equity > 0:
                    dd = (pos.peak_equity - cur_eq) / pos.peak_equity * 100.0
                    pos.max_drawdown_pct = max(pos.max_drawdown_pct, dd)

                # Check live SL/TP breach even intra-candle
                if pos.side != "FLAT":
                    sl_hit = (pos.side == "LONG" and pos.stop_loss and live_price <= pos.stop_loss) or (
                        pos.side == "SHORT" and pos.stop_loss and live_price >= pos.stop_loss
                    )
                    tp_hit = (pos.side == "LONG" and pos.take_profit and live_price >= pos.take_profit) or (
                        pos.side == "SHORT" and pos.take_profit and live_price <= pos.take_profit
                    )
                    if sl_hit or tp_hit:
                        reason = "stop_loss" if sl_hit else "take_profit"
                        _close_paper_position(db, pos, live_price, latest_closed.open_time_ms, cost_cfg, reason)
                        orders_executed += 1
                        db.commit()
                        continue

                # Only evaluate new signal if a NEW closed candle arrived (idempotent on restart)
                if latest_closed.open_time_ms <= pos.last_processed_candle_ms:
                    db.commit()
                    continue

                df = rows_to_polars(closed_rows)
                ctx = IndicatorCache(df, timeframe=pos.timeframe)
                atr_val = float(ctx.get_atr(spec.atr_period)[-1])
                if not (atr_val > 0):
                    atr_val = latest_closed.close * 0.015

                long_sig = bool(evaluate_condition(spec.long_entry, ctx)[-1]) if spec.direction != "short_only" else False
                long_exit_sig = bool(evaluate_condition(spec.long_exit, ctx)[-1]) if spec.long_exit else False
                short_sig = bool(evaluate_condition(spec.short_entry, ctx)[-1]) if spec.direction != "long_only" else False
                short_exit_sig = bool(evaluate_condition(spec.short_exit, ctx)[-1]) if spec.short_exit else False

                if pos.side == "LONG" and long_exit_sig:
                    _close_paper_position(db, pos, live_price, latest_closed.open_time_ms, cost_cfg, "signal_exit")
                    orders_executed += 1
                elif pos.side == "SHORT" and short_exit_sig:
                    _close_paper_position(db, pos, live_price, latest_closed.open_time_ms, cost_cfg, "signal_exit")
                    orders_executed += 1
                elif pos.side == "FLAT":
                    chosen: str | None = None
                    if long_sig and not short_sig:
                        chosen = "LONG"
                    elif short_sig and not long_sig:
                        chosen = "SHORT"

                    if chosen is not None:
                        qty = risk_mgr.calculate_position_size(
                            equity=cur_eq,
                            price=live_price,
                            atr_value=atr_val,
                            sl_atr_mult=spec.sl_atr_mult,
                        )
                        if qty * live_price >= 5.0:
                            try:
                                risk_mgr.validate_new_order(
                                    equity=total_equity,
                                    peak_equity=total_peak,
                                    daily_pnl=0.0,
                                    open_positions_count=all_open,
                                    asset_exposure_notional=pos.quantity * live_price,
                                    order_notional=qty * live_price,
                                    consecutive_losses=0,
                                )
                                _open_paper_position(
                                    db,
                                    pos,
                                    chosen,
                                    live_price,
                                    qty,
                                    atr_val,
                                    spec,
                                    latest_closed.open_time_ms,
                                    cost_cfg,
                                )
                                orders_executed += 1
                                all_open += 1
                            except RiskViolationError as rve:
                                log_system_event(db, "WARNING", "paper_trading.risk", str(rve), {"position_id": pos.id})

                pos.last_processed_candle_ms = latest_closed.open_time_ms
                db.commit()
            except Exception as exc:
                db.rollback()
                errors.append(f"{pos.symbol}: {exc}")
                logger.warning("Paper cycle error on %s: %s", pos.symbol, exc)
    finally:
        adapter.close()

    return {
        "active_strategies": len(active_positions),
        "orders_executed": orders_executed,
        "errors": errors,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _open_paper_position(
    db: Session,
    pos: PaperPosition,
    side: str,
    raw_price: float,
    qty: float,
    atr_val: float,
    spec: StrategySpec,
    candle_ms: int,
    cost_cfg: ExecutionCostConfig,
) -> None:
    idem_key = f"OPEN:{pos.experiment_id}:{pos.symbol}:{candle_ms}:{side}"
    fill_px, slip_u = cost_cfg.effective_entry_price(raw_price, side)
    fee = cost_cfg.commission(qty * fill_px)
    order = PaperOrder(
        order_uid=f"ORD-{uuid.uuid4().hex[:12].upper()}",
        experiment_id=pos.experiment_id,
        symbol=pos.symbol,
        side="BUY" if side == "LONG" else "SELL",
        order_type="MARKET",
        quantity=qty,
        fill_price=fill_px,
        fee_paid=fee,
        slippage_paid=slip_u * qty,
        status="FILLED",
        candle_open_time_ms=candle_ms,
        idempotency_key=idem_key,
    )
    try:
        db.add(order)
        db.flush()
    except IntegrityError:
        db.rollback()
        return

    stop_dist = max(spec.sl_atr_mult * atr_val, fill_px * 0.002)
    tp_dist = max(spec.tp_atr_mult * atr_val, fill_px * 0.003)

    pos.side = side
    pos.quantity = qty
    pos.entry_price = fill_px
    pos.mark_price = raw_price
    pos.realized_pnl -= fee
    pos.unrealized_pnl = 0.0
    pos.stop_loss = (fill_px - stop_dist) if side == "LONG" else (fill_px + stop_dist)
    pos.take_profit = (fill_px + tp_dist) if side == "LONG" else (fill_px - tp_dist)
    pos.opened_at = datetime.now(timezone.utc)
    log_system_event(
        db,
        "INFO",
        "paper_trading.execution",
        f"Opened virtual {side} {qty:.6f} {pos.symbol} @ {fill_px:.4f}",
        {"order_uid": order.order_uid, "fee": fee},
    )


def _close_paper_position(
    db: Session,
    pos: PaperPosition,
    raw_price: float,
    candle_ms: int,
    cost_cfg: ExecutionCostConfig,
    reason: str,
) -> None:
    if pos.side == "FLAT" or pos.quantity <= 0:
        return
    idem_key = f"CLOSE:{pos.experiment_id}:{pos.symbol}:{candle_ms}:{pos.side}:{reason}"
    fill_px, slip_u = cost_cfg.effective_exit_price(raw_price, pos.side)
    fee = cost_cfg.commission(pos.quantity * fill_px)
    order = PaperOrder(
        order_uid=f"ORD-{uuid.uuid4().hex[:12].upper()}",
        experiment_id=pos.experiment_id,
        symbol=pos.symbol,
        side="SELL" if pos.side == "LONG" else "BUY",
        order_type="MARKET",
        quantity=pos.quantity,
        fill_price=fill_px,
        fee_paid=fee,
        slippage_paid=slip_u * pos.quantity,
        status="FILLED",
        candle_open_time_ms=candle_ms,
        idempotency_key=idem_key,
    )
    try:
        db.add(order)
        db.flush()
    except IntegrityError:
        db.rollback()
        return

    gross = (fill_px - pos.entry_price) * pos.quantity if pos.side == "LONG" else (pos.entry_price - fill_px) * pos.quantity
    slip_total = slip_u * pos.quantity
    net = gross - fee

    entry_ms = int(pos.opened_at.timestamp() * 1000) if pos.opened_at else candle_ms
    db.add(
        PaperTrade(
            experiment_id=pos.experiment_id,
            symbol=pos.symbol,
            side=pos.side,
            entry_time_ms=entry_ms,
            exit_time_ms=int(time.time() * 1000),
            entry_price=pos.entry_price,
            exit_price=fill_px,
            quantity=pos.quantity,
            gross_pnl=gross,
            fees_and_slippage=fee + slip_total,
            net_pnl=net,
            exit_reason=reason,
        )
    )
    pos.realized_pnl += net
    pos.unrealized_pnl = 0.0
    pos.side = "FLAT"
    pos.quantity = 0.0
    pos.entry_price = 0.0
    pos.stop_loss = None
    pos.take_profit = None
    pos.opened_at = None
    log_system_event(
        db,
        "INFO",
        "paper_trading.execution",
        f"Closed virtual position {pos.symbol} ({reason}) @ {fill_px:.4f} | Net PnL: {net:.2f}",
        {"order_uid": order.order_uid, "net_pnl": net},
    )


def force_close_paper_position(db: Session, position_id: int) -> PaperPosition:
    pos = db.get(PaperPosition, position_id)
    if pos is None:
        raise ValueError("Paper position not found")
    if pos.side == "FLAT":
        return pos
    adapter = BinanceAdapter()
    try:
        px = adapter.fetch_latest_price(pos.symbol, market_type=pos.market_type)
    finally:
        adapter.close()
    cost_cfg = ExecutionCostConfig(market_type=pos.market_type)
    _close_paper_position(db, pos, px, int(time.time() * 1000), cost_cfg, "manual_close")
    db.commit()
    db.refresh(pos)
    return pos
