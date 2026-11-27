"""Comprehensive quantitative performance metrics (all 26 required metrics + regime breakdown)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SimulatedTrade:
    side: str               # LONG | SHORT
    entry_idx: int
    exit_idx: int
    entry_time_ms: int
    exit_time_ms: int
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fee_paid: float
    slippage_paid: float
    funding_paid: float
    net_pnl: float
    return_pct: float
    exit_reason: str
    regime_at_entry: str = "unknown"


def _max_consecutive(flags: list[bool]) -> int:
    best = 0
    cur = 0
    for f in flags:
        if f:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def compute_performance_metrics(
    trades: list[SimulatedTrade],
    equity_curve: np.ndarray,
    timestamps_ms: np.ndarray,
    initial_capital: float,
    timeframe_ms: int,
    bars_in_position: int = 0,
) -> dict[str, Any]:
    """Compute all 26 required quantitative metrics accurately from equity curve & trade ledger."""
    n_bars = max(1, len(equity_curve))
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) == 0:
        eq = np.array([initial_capital], dtype=np.float64)

    net_pnls = np.array([t.net_pnl for t in trades], dtype=np.float64)
    gross_pnls = np.array([t.gross_pnl for t in trades], dtype=np.float64)
    wins = net_pnls[net_pnls > 0]
    losses = net_pnls[net_pnls < 0]

    total_trades = len(trades)
    net_profit = float(eq[-1] - initial_capital)
    gross_profit = float(np.sum(net_pnls[net_pnls > 0])) if total_trades else 0.0
    gross_loss = float(abs(np.sum(net_pnls[net_pnls < 0]))) if total_trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else (99.0 if gross_profit > 0 else 0.0)
    profit_factor = min(profit_factor, 99.0)

    win_rate = (len(wins) / total_trades * 100.0) if total_trades else 0.0
    loss_rate = (len(losses) / total_trades * 100.0) if total_trades else 0.0
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss = float(abs(np.mean(losses))) if len(losses) else 0.0
    expectancy = float(np.mean(net_pnls)) if total_trades else 0.0

    # Drawdown series
    running_peak = np.maximum.accumulate(eq)
    dd_pct = np.where(running_peak > 0, (running_peak - eq) / running_peak * 100.0, 0.0)
    max_drawdown_pct = float(np.max(dd_pct)) if len(dd_pct) else 0.0
    avg_drawdown_pct = float(np.mean(dd_pct[dd_pct > 0])) if np.any(dd_pct > 0) else 0.0
    ulcer_index = float(np.sqrt(np.mean(dd_pct**2))) if len(dd_pct) else 0.0

    return_on_capital_pct = (net_profit / initial_capital * 100.0) if initial_capital > 0 else 0.0

    # Time & Annualization
    if len(timestamps_ms) >= 2 and timestamps_ms[-1] > timestamps_ms[0]:
        total_days = (float(timestamps_ms[-1] - timestamps_ms[0])) / 86_400_000.0
    else:
        total_days = (n_bars * timeframe_ms) / 86_400_000.0
    total_days = max(total_days, 1e-6)
    years = total_days / 365.0

    final_ratio = max(eq[-1] / initial_capital, 1e-9) if initial_capital > 0 else 1.0
    annualized_return_pct = (float(final_ratio ** (1.0 / max(years, 1e-4)) - 1.0) * 100.0) if years >= 0.01 else return_on_capital_pct
    annualized_return_pct = float(np.clip(annualized_return_pct, -100.0, 10000.0))

    # Bar-level returns for Sharpe & Sortino
    bar_rets = np.zeros(max(1, len(eq) - 1), dtype=np.float64)
    if len(eq) > 1:
        denom = np.where(eq[:-1] > 1e-9, eq[:-1], initial_capital)
        bar_rets = np.diff(eq) / denom

    bars_per_year = (365.0 * 86_400_000.0) / max(timeframe_ms, 60_000)
    mean_ret = float(np.mean(bar_rets))
    std_ret = float(np.std(bar_rets, ddof=1)) if len(bar_rets) > 1 else 0.0
    sharpe_ratio = (mean_ret / std_ret * float(np.sqrt(bars_per_year))) if std_ret > 1e-12 else 0.0

    downside = bar_rets[bar_rets < 0]
    down_std = float(np.sqrt(np.mean(downside**2))) if len(downside) > 0 else 0.0
    sortino_ratio = (mean_ret / down_std * float(np.sqrt(bars_per_year))) if down_std > 1e-12 else (sharpe_ratio if mean_ret > 0 else 0.0)

    calmar_ratio = (annualized_return_pct / max_drawdown_pct) if max_drawdown_pct > 1e-6 else (10.0 if annualized_return_pct > 0 else 0.0)
    recovery_factor = (return_on_capital_pct / max_drawdown_pct) if max_drawdown_pct > 1e-6 else (10.0 if return_on_capital_pct > 0 else 0.0)

    trade_frequency = total_trades / total_days
    holding_bars = [max(1, t.exit_idx - t.entry_idx) for t in trades]
    avg_holding_bars = float(np.mean(holding_bars)) if holding_bars else 0.0
    avg_holding_hours = avg_holding_bars * (timeframe_ms / 3_600_000.0)

    max_consecutive_wins = _max_consecutive([t.net_pnl > 0 for t in trades])
    max_consecutive_losses = _max_consecutive([t.net_pnl <= 0 for t in trades])

    exposure_pct = (bars_in_position / n_bars * 100.0) if n_bars > 0 else 0.0
    total_traded_notional = sum(abs(t.quantity * t.entry_price) + abs(t.quantity * t.exit_price) for t in trades)
    turnover = (total_traded_notional / initial_capital) if initial_capital > 0 else 0.0

    long_trades = [t for t in trades if t.side == "LONG"]
    short_trades = [t for t in trades if t.side == "SHORT"]
    long_perf = {
        "trades": len(long_trades),
        "net_pnl": float(sum(t.net_pnl for t in long_trades)),
        "win_rate": (sum(1 for t in long_trades if t.net_pnl > 0) / len(long_trades) * 100.0) if long_trades else 0.0,
    }
    short_perf = {
        "trades": len(short_trades),
        "net_pnl": float(sum(t.net_pnl for t in short_trades)),
        "win_rate": (sum(1 for t in short_trades if t.net_pnl > 0) / len(short_trades) * 100.0) if short_trades else 0.0,
    }

    # Regime breakdown
    regime_stats: dict[str, dict[str, Any]] = {}
    for t in trades:
        reg = t.regime_at_entry or "unknown"
        bucket = regime_stats.setdefault(reg, {"trades": 0, "wins": 0, "net_pnl": 0.0})
        bucket["trades"] += 1
        if t.net_pnl > 0:
            bucket["wins"] += 1
        bucket["net_pnl"] += float(t.net_pnl)
    for reg, bucket in regime_stats.items():
        bucket["win_rate"] = round(bucket["wins"] / bucket["trades"] * 100.0, 2) if bucket["trades"] else 0.0
        bucket["net_pnl"] = round(bucket["net_pnl"], 2)

    return {
        "net_profit": round(net_profit, 4),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "profit_factor": round(profit_factor, 4),
        "win_rate": round(win_rate, 2),
        "loss_rate": round(loss_rate, 2),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "expectancy": round(expectancy, 4),
        "sharpe_ratio": round(float(np.clip(sharpe_ratio, -50.0, 50.0)), 4),
        "sortino_ratio": round(float(np.clip(sortino_ratio, -50.0, 50.0)), 4),
        "calmar_ratio": round(float(np.clip(calmar_ratio, -50.0, 50.0)), 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "avg_drawdown_pct": round(avg_drawdown_pct, 4),
        "recovery_factor": round(float(np.clip(recovery_factor, -50.0, 50.0)), 4),
        "ulcer_index": round(ulcer_index, 4),
        "return_on_capital_pct": round(return_on_capital_pct, 4),
        "annualized_return_pct": round(annualized_return_pct, 4),
        "total_trades": total_trades,
        "trade_frequency_per_day": round(trade_frequency, 4),
        "avg_holding_bars": round(avg_holding_bars, 2),
        "avg_holding_hours": round(avg_holding_hours, 2),
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "exposure_pct": round(exposure_pct, 2),
        "turnover": round(turnover, 4),
        "long_performance": long_perf,
        "short_performance": short_perf,
        "regime_breakdown": regime_stats,
        "gross_pnl_before_costs": round(float(np.sum(gross_pnls)) if total_trades else 0.0, 4),
        "total_fees": round(float(sum(t.fee_paid for t in trades)), 4),
        "total_slippage": round(float(sum(t.slippage_paid for t in trades)), 4),
        "total_funding": round(float(sum(t.funding_paid for t in trades)), 4),
    }
