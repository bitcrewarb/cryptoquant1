"""High-throughput vectorized/array-indexed backtesting engine for candidate screening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.backtesting.metrics import SimulatedTrade, compute_performance_metrics
from crypto_quant_lab.core.exceptions import RiskViolationError
from crypto_quant_lab.data.exchanges.base import TIMEFRAME_MS

EIGHT_HOURS_MS = 28_800_000


@dataclass
class BacktestOutput:
    metrics: dict[str, Any]
    trades: list[SimulatedTrade]
    equity_curve: list[dict[str, Any]]
    raw_equity_array: np.ndarray
    bar_returns: np.ndarray


def _apply_funding_step(
    prev_t_ms: int,
    cur_t_ms: int,
    pos_side: str,
    qty: float,
    mark_px: float,
    cost_cfg: ExecutionCostConfig,
    funding_rates_map: dict[int, float] | None,
) -> tuple[float, bool, bool]:
    """Compute funding paid across any UTC 00:00/08:00/16:00 boundaries in (prev_t_ms, cur_t_ms]."""
    if cost_cfg.market_type != "futures" or cur_t_ms <= prev_t_ms:
        return 0.0, False, False

    first_f = ((prev_t_ms // EIGHT_HOURS_MS) + 1) * EIGHT_HOURS_MS
    total_f = 0.0
    used_hist = False
    used_fallback = False

    f_ts = first_f
    while f_ts <= cur_t_ms:
        if funding_rates_map is not None and f_ts in funding_rates_map:
            rate = float(funding_rates_map[f_ts])
            used_hist = True
        else:
            rate = float(cost_cfg.funding_rate_per_8h)
            used_fallback = True
        notional = qty * mark_px
        total_f += (notional * rate) if pos_side == "LONG" else (-notional * rate)
        f_ts += EIGHT_HOURS_MS

    return total_f, used_hist, used_fallback


def run_vectorized_backtest(
    open_time_ms: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
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
    max_holding_bars: int | None = None,
    costs: ExecutionCostConfig | None = None,
    allow_long: bool = True,
    allow_short: bool = True,
    funding_rates_map: dict[int, float] | None = None,
    volume: np.ndarray | None = None,
) -> BacktestOutput:
    """
    Strictly causal screening backtester:
    - Signal evaluated at close of bar `i-1` executes at open of bar `i` (no look-ahead).
    - Entry-bar SL/TP is evaluated right after fill on bar `i`.
    - Conservative ambiguous OHLC rule: if both SL and TP fall inside `[low[i], high[i]]`,
      STOP LOSS is assumed to trigger first.
    - Spot markets reject short selling.
    - Futures funding accrues at exact UTC 00:00 / 08:00 / 16:00 timestamps.
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
    v = np.asarray(volume, dtype=np.float64) if volume is not None and len(volume) == len(c) else None
    n = len(c)

    if n < 5:
        eq = np.full(max(n, 1), initial_capital, dtype=np.float64)
        metrics = compute_performance_metrics([], eq, t_ms, initial_capital, tf_ms, 0)
        metrics["funding_source"] = "none"
        return BacktestOutput(metrics=metrics, trades=[], equity_curve=[], raw_equity_array=eq, bar_returns=np.zeros(max(0, n - 1)))

    equity = initial_capital
    eq_arr = np.empty(n, dtype=np.float64)
    eq_arr[0] = equity

    pos_side: str | None = None
    qty = 0.0
    raw_entry_price = 0.0
    fill_entry_price = 0.0
    entry_idx = 0
    entry_time = 0
    entry_fee = 0.0
    entry_slip = 0.0
    funding_acc = 0.0
    sl_price = 0.0
    tp_price = 0.0
    liq_price: float | None = None
    entry_regime = "unknown"
    bars_in_pos = 0

    used_hist_funding = False
    used_fallback_funding = False

    trades: list[SimulatedTrade] = []

    def _close_trade(exit_raw: float, exit_reason: str, bar_i: int) -> None:
        nonlocal pos_side, qty, funding_acc, equity, liq_price
        is_maker_exit = exit_reason == "take_profit"
        a_cur = atr_arr[bar_i - 1] if bar_i > 0 and bar_i - 1 < len(atr_arr) else None
        v_cur = float(v[bar_i]) if v is not None else None
        fill_exit, slip_unit = cost_cfg.effective_exit_price(
            exit_raw, pos_side or "LONG", is_maker=is_maker_exit, atr_val=a_cur, order_qty=qty, bar_volume=v_cur
        )
        exit_fee = cost_cfg.commission(qty * fill_exit, is_maker=is_maker_exit)
        exit_slip = slip_unit * qty
        total_slip = entry_slip + exit_slip
        total_fee = entry_fee + exit_fee

        if pos_side == "LONG":
            raw_gross = (exit_raw - raw_entry_price) * qty
            slipped_pnl = (fill_exit - fill_entry_price) * qty
        else:
            raw_gross = (raw_entry_price - exit_raw) * qty
            slipped_pnl = (fill_entry_price - fill_exit) * qty

        net_pnl = raw_gross - total_fee - total_slip - funding_acc
        equity += slipped_pnl - exit_fee
        notional_entry = max(qty * raw_entry_price, 1e-9)

        trades.append(
            SimulatedTrade(
                side=pos_side or "LONG",
                entry_idx=entry_idx,
                exit_idx=bar_i,
                entry_time_ms=entry_time,
                exit_time_ms=int(t_ms[bar_i]),
                entry_price=raw_entry_price,
                exit_price=exit_raw,
                quantity=qty,
                gross_pnl=raw_gross,
                fee_paid=total_fee,
                slippage_paid=total_slip,
                funding_paid=funding_acc,
                net_pnl=net_pnl,
                return_pct=net_pnl / notional_entry * 100.0,
                exit_reason=exit_reason,
                regime_at_entry=entry_regime,
            )
        )
        pos_side = None
        qty = 0.0
        funding_acc = 0.0
        liq_price = None

    for i in range(1, n):
        # 1. Exact UTC 00:00/08:00/16:00 funding payment on open futures positions held from prior bar
        if pos_side is not None and entry_idx < i:
            f_cost, uh, uf = _apply_funding_step(
                int(t_ms[i - 1]), int(t_ms[i]), pos_side, qty, o[i], cost_cfg, funding_rates_map
            )
            if f_cost != 0.0 or uh or uf:
                equity -= f_cost
                funding_acc += f_cost
                used_hist_funding = used_hist_funding or uh
                used_fallback_funding = used_fallback_funding or uf

        # 2. Check exits on bar `i` if in position from a prior bar
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
                elif long_exit[i - 1]:
                    exit_raw = o[i]
                    exit_reason = "signal_exit"
                elif max_holding_bars is not None and (i - entry_idx) >= max_holding_bars:
                    exit_raw = o[i]
                    exit_reason = "time_exit"
            else:  # SHORT
                if liq_price is not None and h[i] >= liq_price:
                    exit_raw = max(o[i], liq_price)
                    exit_reason = "liquidation"
                elif h[i] >= sl_price:
                    exit_raw = max(o[i], sl_price)
                    exit_reason = "stop_loss"
                elif l[i] <= tp_price:
                    exit_raw = min(o[i], tp_price)
                    exit_reason = "take_profit"
                elif short_exit[i - 1]:
                    exit_raw = o[i]
                    exit_reason = "signal_exit"
                elif max_holding_bars is not None and (i - entry_idx) >= max_holding_bars:
                    exit_raw = o[i]
                    exit_reason = "time_exit"

            if exit_raw is not None and exit_raw > 0:
                _close_trade(exit_raw, exit_reason, i)

        # 3. Check new entry at open[i] from signal at close[i-1]
        if pos_side is None and equity > 10.0:
            want_long = allow_long and bool(long_entry[i - 1])
            want_short = allow_short and bool(short_entry[i - 1])
            chosen_side: str | None = None
            if want_long and not want_short:
                chosen_side = "LONG"
            elif want_short and not want_long:
                chosen_side = "SHORT"

            if chosen_side is not None:
                a_val = atr_arr[i - 1]
                if not np.isfinite(a_val) or a_val <= 0:
                    a_val = o[i] * 0.015
                stop_dist = max(sl_atr_mult * a_val, o[i] * 0.002)
                tp_dist = max(tp_atr_mult * a_val, o[i] * 0.003)

                raw_px = o[i]
                risk_amt = equity * (risk_per_trade_pct / 100.0)
                raw_qty = risk_amt / stop_dist
                max_notional = equity * cost_cfg.leverage
                if raw_qty * raw_px > max_notional:
                    raw_qty = max_notional / raw_px
                v_cur = float(v[i]) if v is not None else None
                fill_entry, slip_unit = cost_cfg.effective_entry_price(
                    raw_px, chosen_side, atr_val=a_val, order_qty=raw_qty, bar_volume=v_cur
                )

                if raw_qty > 0 and raw_qty * fill_entry >= 5.0:
                    entry_fee = cost_cfg.commission(raw_qty * fill_entry)
                    equity -= entry_fee
                    pos_side = chosen_side
                    qty = raw_qty
                    raw_entry_price = raw_px
                    fill_entry_price = fill_entry
                    entry_idx = i
                    entry_time = int(t_ms[i])
                    entry_slip = slip_unit * qty
                    funding_acc = 0.0
                    liq_price = cost_cfg.liquidation_price(fill_entry, pos_side)
                    entry_regime = regimes[i - 1] if regimes and i - 1 < len(regimes) else "unknown"
                    if pos_side == "LONG":
                        sl_price = fill_entry - stop_dist
                        tp_price = fill_entry + tp_dist
                    else:
                        sl_price = fill_entry + stop_dist
                        tp_price = fill_entry - tp_dist

                    # 4. Immediate entry-bar liquidation / SL / TP resolution (conservative first)
                    entry_exit_raw: float | None = None
                    entry_exit_reason = ""
                    if pos_side == "LONG":
                        if liq_price is not None and l[i] <= liq_price:
                            entry_exit_raw = min(raw_px, liq_price)
                            entry_exit_reason = "liquidation"
                        elif l[i] <= sl_price:
                            entry_exit_raw = min(raw_px, sl_price)
                            entry_exit_reason = "stop_loss"
                        elif h[i] >= tp_price:
                            entry_exit_raw = max(raw_px, tp_price)
                            entry_exit_reason = "take_profit"
                    else:
                        if liq_price is not None and h[i] >= liq_price:
                            entry_exit_raw = max(raw_px, liq_price)
                            entry_exit_reason = "liquidation"
                        elif h[i] >= sl_price:
                            entry_exit_raw = max(raw_px, sl_price)
                            entry_exit_reason = "stop_loss"
                        elif l[i] <= tp_price:
                            entry_exit_raw = min(raw_px, tp_price)
                            entry_exit_reason = "take_profit"

                    if entry_exit_raw is not None and entry_exit_raw > 0:
                        bars_in_pos += 1
                        _close_trade(entry_exit_raw, entry_exit_reason, i)

        # Mark-to-market equity at bar close
        if pos_side == "LONG":
            mtm = (c[i] - fill_entry_price) * qty
        elif pos_side == "SHORT":
            mtm = (fill_entry_price - c[i]) * qty
        else:
            mtm = 0.0
        eq_arr[i] = max(0.0, equity + mtm)

    # Close any open position cleanly at final bar close for accounting
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

    # Downsample equity curve to max 400 points for lightweight JSON/Plotly rendering
    step = max(1, n // 400)
    curve = [
        {"t": int(t_ms[idx]), "equity": round(float(eq_arr[idx]), 2)}
        for idx in range(0, n, step)
    ]
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
