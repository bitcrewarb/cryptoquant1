"""Event-driven bar-by-bar execution simulator with limit/stop orders, structural SL/TP, partial TP, BE moves, futures liquidation, funding, and risk limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.backtesting.metrics import SimulatedTrade, compute_performance_metrics
from crypto_quant_lab.backtesting.vectorized.engine import BacktestOutput, _apply_funding_step
from crypto_quant_lab.core.exceptions import RiskViolationError
from crypto_quant_lab.data.exchanges.base import TIMEFRAME_MS


@dataclass
class EventOrder:
    side: str               # LONG | SHORT
    order_type: str         # MARKET | LIMIT | STOP
    target_qty: float
    filled_qty: float = 0.0
    limit_price: float | None = None
    stop_price: float | None = None
    sl_dist: float = 0.0
    tp_dist: float = 0.0
    sl_abs: float | None = None
    tp_abs: float | None = None
    created_idx: int = 0
    ttl_bars: int = 3


def run_event_driven_backtest(
    open_time_ms: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    long_entry: np.ndarray,
    long_exit: np.ndarray,
    short_entry: np.ndarray,
    short_exit: np.ndarray,
    atr_arr: np.ndarray,
    regimes: list[str] | None = None,
    timeframe: str = "1h",
    initial_capital: float = 100_000.0,
    risk_per_trade_pct: float = 1.0,
    sl_atr_mult: float = 2.0,
    tp_atr_mult: float = 3.0,
    trailing_stop_atr: float | None = None,
    max_holding_bars: int | None = None,
    order_type: str = "MARKET",
    costs: ExecutionCostConfig | None = None,
    allow_long: bool = True,
    allow_short: bool = True,
    max_daily_loss_pct: float = 5.0,
    max_portfolio_dd_pct: float = 30.0,
    max_consecutive_losses_halt: int = 10,
    funding_rates_map: dict[int, float] | None = None,
    long_limit_prices: np.ndarray | None = None,
    short_limit_prices: np.ndarray | None = None,
    long_sl_prices: np.ndarray | None = None,
    short_sl_prices: np.ndarray | None = None,
    long_tp_prices: np.ndarray | None = None,
    short_tp_prices: np.ndarray | None = None,
    entry_ttl_bars: int = 5,
    partial_tp_ratio: float = 0.0,
    partial_tp_rr: float = 1.5,
    move_to_be_after_tp1: bool = False,
    session_exit: np.ndarray | None = None,
) -> BacktestOutput:
    """
    Detailed event-driven simulator:
    - Consumes Market, Limit (including structural zone/OTE limits with TTL expiry), and Stop entry orders.
    - Supports structural SL (sweep wick / zone invalidation), structural TP (opposing liquidity / R-multiple),
      partial take-profit with proportional cost/funding conservation, move-to-breakeven, trailing stops, and session exits.
    - Enforces isolated futures liquidation, exchange min-notional, leverage cap, and risk manager circuit breakers.
    - Evaluates entry-bar SL/TP/liquidation right after fill; conservative SL-first resolution on ambiguous OHLC bars.
    """
    cost_cfg = costs or ExecutionCostConfig()
    if cost_cfg.market_type == "spot" and allow_short and np.any(short_entry):
        raise RiskViolationError("Short selling is not permitted on spot market_type='spot'")

    tf_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)

    t_ms = np.asarray(open_time_ms, dtype=np.int64)
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    v = np.asarray(volume, dtype=np.float64)
    n = len(c)

    if n < 5:
        eq = np.full(max(n, 1), initial_capital, dtype=np.float64)
        metrics = compute_performance_metrics([], eq, t_ms, initial_capital, tf_ms, 0)
        metrics["funding_source"] = "none"
        return BacktestOutput(metrics=metrics, trades=[], equity_curve=[], raw_equity_array=eq, bar_returns=np.zeros(max(0, n - 1)))

    equity = initial_capital
    peak_equity = initial_capital
    eq_arr = np.empty(n, dtype=np.float64)
    eq_arr[0] = equity

    pending_order: EventOrder | None = None
    pos_side: str | None = None
    pos_qty = 0.0
    raw_entry_price = 0.0
    fill_entry_price = 0.0
    entry_idx = 0
    entry_time = 0
    entry_fee = 0.0
    entry_slip = 0.0
    funding_acc = 0.0
    sl_price = 0.0
    tp_price = 0.0
    tp1_price: float | None = None
    tp1_taken = False
    liq_price: float | None = None
    entry_regime = "unknown"
    bars_in_pos = 0

    consec_losses = 0
    trading_halted = False
    cur_day = int(t_ms[0] // 86_400_000)
    day_start_equity = equity

    used_hist_funding = False
    used_fallback_funding = False

    trades: list[SimulatedTrade] = []

    def _close_trade(exit_raw: float, exit_reason: str, bar_i: int, close_fraction: float = 1.0) -> None:
        nonlocal pos_side, pos_qty, entry_fee, entry_slip, funding_acc, equity, consec_losses, liq_price, tp1_price, tp1_taken
        frac = float(np.clip(close_fraction, 0.0, 1.0))
        if pos_side is None or pos_qty <= 0 or frac <= 0:
            return
        exec_qty = pos_qty * frac
        is_maker_exit = exit_reason in ("take_profit", "partial_tp1")
        a_cur = atr_arr[bar_i - 1] if bar_i > 0 and bar_i - 1 < len(atr_arr) else None
        fill_exit, slip_u = cost_cfg.effective_exit_price(
            exit_raw, pos_side, is_maker=is_maker_exit, atr_val=a_cur, order_qty=exec_qty, bar_volume=float(v[bar_i])
        )
        exit_fee = cost_cfg.commission(exec_qty * fill_exit, is_maker=is_maker_exit)
        exit_slip = slip_u * exec_qty
        alloc_entry_fee = entry_fee * frac
        alloc_entry_slip = entry_slip * frac
        alloc_funding = funding_acc * frac

        total_slip = alloc_entry_slip + exit_slip
        total_fee = alloc_entry_fee + exit_fee

        if pos_side == "LONG":
            raw_gross = (exit_raw - raw_entry_price) * exec_qty
            slipped_pnl = (fill_exit - fill_entry_price) * exec_qty
        else:
            raw_gross = (raw_entry_price - exit_raw) * exec_qty
            slipped_pnl = (fill_entry_price - fill_exit) * exec_qty

        net_pnl = raw_gross - total_fee - total_slip - alloc_funding
        equity += slipped_pnl - exit_fee

        if frac >= 0.999999:
            if net_pnl <= 0:
                consec_losses += 1
            else:
                consec_losses = 0

        trades.append(
            SimulatedTrade(
                side=pos_side,
                entry_idx=entry_idx,
                exit_idx=bar_i,
                entry_time_ms=entry_time,
                exit_time_ms=int(t_ms[bar_i]),
                entry_price=raw_entry_price,
                exit_price=exit_raw,
                quantity=exec_qty,
                gross_pnl=raw_gross,
                fee_paid=total_fee,
                slippage_paid=total_slip,
                funding_paid=alloc_funding,
                net_pnl=net_pnl,
                return_pct=net_pnl / max(exec_qty * raw_entry_price, 1e-9) * 100.0,
                exit_reason=exit_reason,
                regime_at_entry=entry_regime,
            )
        )

        if frac >= 0.999999:
            pos_side = None
            pos_qty = 0.0
            entry_fee = 0.0
            entry_slip = 0.0
            funding_acc = 0.0
            liq_price = None
            tp1_price = None
            tp1_taken = False
        else:
            pos_qty -= exec_qty
            entry_fee -= alloc_entry_fee
            entry_slip -= alloc_entry_slip
            funding_acc -= alloc_funding

    def _init_sl_tp(side: str, fill_px: float, sl_dist: float, tp_dist: float, sl_abs: float | None, tp_abs: float | None) -> None:
        nonlocal sl_price, tp_price, tp1_price, tp1_taken, liq_price
        if side == "LONG":
            sl_price = sl_abs if (sl_abs is not None and np.isfinite(sl_abs) and sl_abs < fill_px) else (fill_px - sl_dist)
            tp_price = tp_abs if (tp_abs is not None and np.isfinite(tp_abs) and tp_abs > fill_px) else (fill_px + tp_dist)
            eff_sl_dist = max(fill_px - sl_price, fill_px * 0.002)
            tp1_price = (fill_px + partial_tp_rr * eff_sl_dist) if 0.0 < partial_tp_ratio < 1.0 else None
        else:
            sl_price = sl_abs if (sl_abs is not None and np.isfinite(sl_abs) and sl_abs > fill_px) else (fill_px + sl_dist)
            tp_price = tp_abs if (tp_abs is not None and np.isfinite(tp_abs) and tp_abs < fill_px) else (fill_px - tp_dist)
            eff_sl_dist = max(sl_price - fill_px, fill_px * 0.002)
            tp1_price = (fill_px - partial_tp_rr * eff_sl_dist) if 0.0 < partial_tp_ratio < 1.0 else None
        tp1_taken = False
        liq_price = cost_cfg.liquidation_price(fill_px, side)

    def _check_entry_bar_sl_tp(bar_i: int, fill_anchor: float) -> None:
        nonlocal bars_in_pos, pending_order, sl_price, tp1_taken
        if pos_side is None:
            return
        entry_exit_raw: float | None = None
        entry_exit_reason = ""
        if pos_side == "LONG":
            if liq_price is not None and l[bar_i] <= liq_price:
                entry_exit_raw = min(fill_anchor, liq_price)
                entry_exit_reason = "liquidation"
            elif l[bar_i] <= sl_price:
                entry_exit_raw = min(fill_anchor, sl_price)
                entry_exit_reason = "stop_loss"
            elif h[bar_i] >= tp_price:
                entry_exit_raw = max(fill_anchor, tp_price)
                entry_exit_reason = "take_profit"
        else:
            if liq_price is not None and h[bar_i] >= liq_price:
                entry_exit_raw = max(fill_anchor, liq_price)
                entry_exit_reason = "liquidation"
            elif h[bar_i] >= sl_price:
                entry_exit_raw = max(fill_anchor, sl_price)
                entry_exit_reason = "stop_loss"
            elif l[bar_i] <= tp_price:
                entry_exit_raw = min(fill_anchor, tp_price)
                entry_exit_reason = "take_profit"

        if entry_exit_raw is not None and entry_exit_raw > 0:
            bars_in_pos += 1
            pending_order = None
            _close_trade(entry_exit_raw, entry_exit_reason, bar_i)
            return

        # Partial TP1 on entry bar (only if full SL/TP didn't fire)
        if not tp1_taken and tp1_price is not None and 0.0 < partial_tp_ratio < 1.0:
            if pos_side == "LONG" and h[bar_i] >= tp1_price:
                _close_trade(max(fill_anchor, tp1_price), "partial_tp1", bar_i, close_fraction=partial_tp_ratio)
                tp1_taken = True
                if move_to_be_after_tp1:
                    sl_price = max(sl_price, fill_entry_price)
            elif pos_side == "SHORT" and l[bar_i] <= tp1_price:
                _close_trade(min(fill_anchor, tp1_price), "partial_tp1", bar_i, close_fraction=partial_tp_ratio)
                tp1_taken = True
                if move_to_be_after_tp1:
                    sl_price = min(sl_price, fill_entry_price)

    for i in range(1, n):
        day = int(t_ms[i] // 86_400_000)
        if day != cur_day:
            cur_day = day
            day_start_equity = equity

        # Risk Manager checks
        dd_from_peak = ((peak_equity - equity) / peak_equity * 100.0) if peak_equity > 0 else 0.0
        daily_loss = ((day_start_equity - equity) / day_start_equity * 100.0) if day_start_equity > 0 else 0.0
        if dd_from_peak >= max_portfolio_dd_pct or consec_losses >= max_consecutive_losses_halt:
            trading_halted = True

        # 1. Exact UTC 00:00/08:00/16:00 funding payment on open futures position
        if pos_side is not None and entry_idx < i:
            f_cost, uh, uf = _apply_funding_step(
                int(t_ms[i - 1]), int(t_ms[i]), pos_side, pos_qty, o[i], cost_cfg, funding_rates_map
            )
            if f_cost != 0.0 or uh or uf:
                equity -= f_cost
                funding_acc += f_cost
                used_hist_funding = used_hist_funding or uh
                used_fallback_funding = used_fallback_funding or uf

        # 2. Trailing Stop update using prior bar close/extremes (causal)
        if pos_side is not None and trailing_stop_atr is not None and trailing_stop_atr > 0:
            a_prev = atr_arr[i - 1]
            if np.isfinite(a_prev) and a_prev > 0:
                if pos_side == "LONG":
                    sl_price = max(sl_price, h[i - 1] - trailing_stop_atr * a_prev)
                else:
                    sl_price = min(sl_price, l[i - 1] + trailing_stop_atr * a_prev)

        # 3. Position Exit check (conservative Liquidation -> SL -> TP1/TP2 -> Signal -> Session -> Time)
        if pos_side is not None:
            bars_in_pos += 1
            exit_raw: float | None = None
            exit_reason = ""

            if pos_side == "LONG":
                if liq_price is not None and l[i] <= liq_price:
                    exit_raw = min(o[i], liq_price)
                    exit_reason = "liquidation"
                elif l[i] <= sl_price:
                    exit_raw = min(o[i], sl_price)
                    exit_reason = "stop_loss"
                elif h[i] >= tp_price:
                    exit_raw = max(o[i], tp_price)
                    exit_reason = "take_profit"
                elif not tp1_taken and tp1_price is not None and 0.0 < partial_tp_ratio < 1.0 and h[i] >= tp1_price:
                    _close_trade(max(o[i], tp1_price), "partial_tp1", i, close_fraction=partial_tp_ratio)
                    tp1_taken = True
                    if move_to_be_after_tp1:
                        sl_price = max(sl_price, fill_entry_price)
                elif long_exit[i - 1]:
                    exit_raw = o[i]
                    exit_reason = "signal_exit"
                elif session_exit is not None and bool(session_exit[i - 1]):
                    exit_raw = o[i]
                    exit_reason = "session_exit"
                elif max_holding_bars is not None and (i - entry_idx) >= max_holding_bars:
                    exit_raw = o[i]
                    exit_reason = "time_exit"
            else:
                if liq_price is not None and h[i] >= liq_price:
                    exit_raw = max(o[i], liq_price)
                    exit_reason = "liquidation"
                elif h[i] >= sl_price:
                    exit_raw = max(o[i], sl_price)
                    exit_reason = "stop_loss"
                elif l[i] <= tp_price:
                    exit_raw = min(o[i], tp_price)
                    exit_reason = "take_profit"
                elif not tp1_taken and tp1_price is not None and 0.0 < partial_tp_ratio < 1.0 and l[i] <= tp1_price:
                    _close_trade(min(o[i], tp1_price), "partial_tp1", i, close_fraction=partial_tp_ratio)
                    tp1_taken = True
                    if move_to_be_after_tp1:
                        sl_price = min(sl_price, fill_entry_price)
                elif short_exit[i - 1]:
                    exit_raw = o[i]
                    exit_reason = "signal_exit"
                elif session_exit is not None and bool(session_exit[i - 1]):
                    exit_raw = o[i]
                    exit_reason = "session_exit"
                elif max_holding_bars is not None and (i - entry_idx) >= max_holding_bars:
                    exit_raw = o[i]
                    exit_reason = "time_exit"

            if exit_raw is not None and exit_raw > 0:
                _close_trade(exit_raw, exit_reason, i)
                pending_order = None

        # 4. Process pending order (supports Limit/Stop + TTL expiry + volume partial fills)
        if pending_order is not None:
            if (i - pending_order.created_idx) >= pending_order.ttl_bars:
                pending_order = None
            else:
                fill_raw: float | None = None
                is_maker = False
                if pending_order.order_type == "MARKET":
                    fill_raw = o[i]
                elif pending_order.order_type == "LIMIT" and pending_order.limit_price is not None:
                    lp = pending_order.limit_price
                    if pending_order.side == "LONG" and l[i] <= lp:
                        fill_raw = min(o[i], lp)
                        is_maker = True
                    elif pending_order.side == "SHORT" and h[i] >= lp:
                        fill_raw = max(o[i], lp)
                        is_maker = True
                elif pending_order.order_type == "STOP" and pending_order.stop_price is not None:
                    sp = pending_order.stop_price
                    if pending_order.side == "LONG" and h[i] >= sp:
                        fill_raw = max(o[i], sp)
                    elif pending_order.side == "SHORT" and l[i] <= sp:
                        fill_raw = min(o[i], sp)

                if fill_raw is not None and fill_raw > 0:
                    a_cur = atr_arr[i - 1] if np.isfinite(atr_arr[i - 1]) else None
                    max_bar_qty = max(v[i] * cost_cfg.max_volume_participation, 1e-6) if v[i] > 0 else pending_order.target_qty
                    remaining = pending_order.target_qty - pending_order.filled_qty
                    exec_qty = min(remaining, max_bar_qty)
                    fill_px, slip_u = cost_cfg.effective_entry_price(
                        fill_raw, pending_order.side, is_maker=is_maker, atr_val=a_cur, order_qty=exec_qty, bar_volume=float(v[i])
                    )
                    if exec_qty * fill_px >= 5.0:
                        fee = cost_cfg.commission(exec_qty * fill_px, is_maker=is_maker)
                        equity -= fee
                        newly_opened = pos_side is None
                        if newly_opened:
                            pos_side = pending_order.side
                            pos_qty = exec_qty
                            raw_entry_price = fill_raw
                            fill_entry_price = fill_px
                            entry_idx = i
                            entry_time = int(t_ms[i])
                            entry_fee = fee
                            entry_slip = slip_u * exec_qty
                            funding_acc = 0.0
                            entry_regime = regimes[i - 1] if regimes and i - 1 < len(regimes) else "unknown"
                            _init_sl_tp(
                                pos_side,
                                fill_px,
                                pending_order.sl_dist,
                                pending_order.tp_dist,
                                pending_order.sl_abs,
                                pending_order.tp_abs,
                            )
                        else:
                            total_q = pos_qty + exec_qty
                            raw_entry_price = (raw_entry_price * pos_qty + fill_raw * exec_qty) / total_q
                            fill_entry_price = (fill_entry_price * pos_qty + fill_px * exec_qty) / total_q
                            pos_qty = total_q
                            entry_fee += fee
                            entry_slip += slip_u * exec_qty

                        pending_order.filled_qty += exec_qty
                        if pending_order.filled_qty >= pending_order.target_qty - 1e-12:
                            pending_order = None

                        if newly_opened:
                            _check_entry_bar_sl_tp(i, fill_raw)
                    else:
                        pending_order = None

        # 5. Generate new entry order from signal at close[i-1]
        if (
            pos_side is None
            and pending_order is None
            and not trading_halted
            and daily_loss < max_daily_loss_pct
            and equity > 10.0
        ):
            want_long = allow_long and bool(long_entry[i - 1])
            want_short = allow_short and bool(short_entry[i - 1])
            chosen_side: str | None = None
            if want_long and not want_short:
                chosen_side = "LONG"
            elif want_short and not want_long:
                chosen_side = "SHORT"

            if chosen_side is not None:
                a_val = atr_arr[i - 1] if np.isfinite(atr_arr[i - 1]) and atr_arr[i - 1] > 0 else o[i] * 0.015

                # Determine effective order type & structural limit price if provided
                lim_px: float | None = None
                if chosen_side == "LONG" and long_limit_prices is not None and np.isfinite(long_limit_prices[i - 1]):
                    lim_px = float(long_limit_prices[i - 1])
                elif chosen_side == "SHORT" and short_limit_prices is not None and np.isfinite(short_limit_prices[i - 1]):
                    lim_px = float(short_limit_prices[i - 1])

                eff_order_type = "LIMIT" if lim_px is not None else order_type
                if eff_order_type == "LIMIT" and lim_px is None:
                    lim_px = c[i - 1] * (0.999 if chosen_side == "LONG" else 1.001)
                stp_px = (h[i - 1] if chosen_side == "LONG" else l[i - 1]) if eff_order_type == "STOP" else None

                anchor_px = lim_px if (eff_order_type == "LIMIT" and lim_px is not None) else o[i]

                sl_abs: float | None = None
                if chosen_side == "LONG" and long_sl_prices is not None and np.isfinite(long_sl_prices[i - 1]) and long_sl_prices[i - 1] < anchor_px:
                    sl_abs = float(long_sl_prices[i - 1])
                    sl_dist = max(anchor_px - sl_abs, anchor_px * 0.002)
                elif chosen_side == "SHORT" and short_sl_prices is not None and np.isfinite(short_sl_prices[i - 1]) and short_sl_prices[i - 1] > anchor_px:
                    sl_abs = float(short_sl_prices[i - 1])
                    sl_dist = max(sl_abs - anchor_px, anchor_px * 0.002)
                else:
                    sl_dist = max(sl_atr_mult * a_val, anchor_px * 0.002)

                tp_abs: float | None = None
                if chosen_side == "LONG" and long_tp_prices is not None and np.isfinite(long_tp_prices[i - 1]) and long_tp_prices[i - 1] > anchor_px:
                    tp_abs = float(long_tp_prices[i - 1])
                    tp_dist = max(tp_abs - anchor_px, anchor_px * 0.003)
                elif chosen_side == "SHORT" and short_tp_prices is not None and np.isfinite(short_tp_prices[i - 1]) and short_tp_prices[i - 1] < anchor_px:
                    tp_abs = float(short_tp_prices[i - 1])
                    tp_dist = max(anchor_px - tp_abs, anchor_px * 0.003)
                else:
                    tp_dist = max(tp_atr_mult * a_val, anchor_px * 0.003)

                risk_amt = equity * (risk_per_trade_pct / 100.0)
                raw_qty = risk_amt / sl_dist
                max_notional = equity * cost_cfg.leverage
                if raw_qty * anchor_px > max_notional:
                    raw_qty = max_notional / anchor_px

                if raw_qty * anchor_px >= 5.0:
                    order = EventOrder(
                        side=chosen_side,
                        order_type=eff_order_type,
                        target_qty=raw_qty,
                        limit_price=lim_px,
                        stop_price=stp_px,
                        sl_dist=sl_dist,
                        tp_dist=tp_dist,
                        sl_abs=sl_abs,
                        tp_abs=tp_abs,
                        created_idx=i,
                        ttl_bars=max(1, int(entry_ttl_bars)),
                    )
                    if eff_order_type == "MARKET":
                        raw_px = o[i]
                        max_bar_qty = max(v[i] * cost_cfg.max_volume_participation, 1e-6) if v[i] > 0 else raw_qty
                        exec_qty = min(raw_qty, max_bar_qty)
                        fill_px, slip_u = cost_cfg.effective_entry_price(
                            raw_px, chosen_side, atr_val=a_val, order_qty=exec_qty, bar_volume=float(v[i])
                        )
                        fee = cost_cfg.commission(exec_qty * fill_px)
                        equity -= fee
                        pos_side = chosen_side
                        pos_qty = exec_qty
                        raw_entry_price = raw_px
                        fill_entry_price = fill_px
                        entry_idx = i
                        entry_time = int(t_ms[i])
                        entry_fee = fee
                        entry_slip = slip_u * exec_qty
                        funding_acc = 0.0
                        entry_regime = regimes[i - 1] if regimes and i - 1 < len(regimes) else "unknown"
                        _init_sl_tp(pos_side, fill_px, sl_dist, tp_dist, sl_abs, tp_abs)
                        if exec_qty < raw_qty - 1e-12:
                            order.filled_qty = exec_qty
                            pending_order = order
                        _check_entry_bar_sl_tp(i, raw_px)
                    elif eff_order_type == "LIMIT" and lim_px is not None:
                        # Check immediate limit touch on bar i itself!
                        hit_now = (chosen_side == "LONG" and l[i] <= lim_px) or (chosen_side == "SHORT" and h[i] >= lim_px)
                        if hit_now:
                            fill_raw = min(o[i], lim_px) if chosen_side == "LONG" else max(o[i], lim_px)
                            max_bar_qty = max(v[i] * cost_cfg.max_volume_participation, 1e-6) if v[i] > 0 else raw_qty
                            exec_qty = min(raw_qty, max_bar_qty)
                            fill_px, slip_u = cost_cfg.effective_entry_price(
                                fill_raw, chosen_side, is_maker=True, atr_val=a_val, order_qty=exec_qty, bar_volume=float(v[i])
                            )
                            fee = cost_cfg.commission(exec_qty * fill_px, is_maker=True)
                            equity -= fee
                            pos_side = chosen_side
                            pos_qty = exec_qty
                            raw_entry_price = fill_raw
                            fill_entry_price = fill_px
                            entry_idx = i
                            entry_time = int(t_ms[i])
                            entry_fee = fee
                            entry_slip = slip_u * exec_qty
                            funding_acc = 0.0
                            entry_regime = regimes[i - 1] if regimes and i - 1 < len(regimes) else "unknown"
                            _init_sl_tp(pos_side, fill_px, sl_dist, tp_dist, sl_abs, tp_abs)
                            if exec_qty < raw_qty - 1e-12:
                                order.filled_qty = exec_qty
                                pending_order = order
                            _check_entry_bar_sl_tp(i, fill_raw)
                        else:
                            pending_order = order
                    else:
                        pending_order = order

        mtm = ((c[i] - fill_entry_price) * pos_qty) if pos_side == "LONG" else (((fill_entry_price - c[i]) * pos_qty) if pos_side == "SHORT" else 0.0)
        eq_arr[i] = max(0.0, equity + mtm)
        if eq_arr[i] > peak_equity:
            peak_equity = eq_arr[i]

    if pos_side is not None:
        _close_trade(c[-1], "end_of_dataset", n - 1)
        eq_arr[-1] = max(0.0, equity)

    metrics = compute_performance_metrics(trades, eq_arr, t_ms, initial_capital, tf_ms, bars_in_pos)
    if cost_cfg.market_type == "futures":
        if used_hist_funding and used_fallback_funding:
            metrics["funding_source"] = "mixed"
        elif used_hist_funding:
            metrics["funding_source"] = "historical_series"
        elif used_fallback_funding:
            metrics["funding_source"] = "flat_fallback"
        else:
            metrics["funding_source"] = "historical_series" if funding_rates_map else "none"
    else:
        metrics["funding_source"] = "none"

    step = max(1, n // 400)
    curve = [{"t": int(t_ms[idx]), "equity": round(float(eq_arr[idx]), 2)} for idx in range(0, n, step)]
    if curve and curve[-1]["t"] != int(t_ms[-1]):
        curve.append({"t": int(t_ms[-1]), "equity": round(float(eq_arr[-1]), 2)})

    bar_rets = np.diff(eq_arr) / np.where(eq_arr[:-1] > 1e-9, eq_arr[:-1], initial_capital)
    return BacktestOutput(
        metrics=metrics,
        trades=trades,
        equity_curve=curve,
        raw_equity_array=eq_arr,
        bar_returns=bar_rets,
    )
