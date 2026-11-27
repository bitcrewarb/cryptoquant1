"""Stress testing engine evaluating execution fragility under adverse market regimes."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.backtesting.vectorized.engine import run_vectorized_backtest
from crypto_quant_lab.strategies.evaluator import IndicatorCache, evaluate_condition
from crypto_quant_lab.strategies.grammar import StrategySpec


def run_stress_tests(
    df: pl.DataFrame,
    spec: StrategySpec,
    timeframe: str = "1h",
    base_costs: ExecutionCostConfig | None = None,
) -> dict[str, Any]:
    """Stress test strategy across elevated fees, slippage, wider spreads, delayed fills, missing bars, and volatility spikes."""
    cfg = base_costs or ExecutionCostConfig()
    ctx = IndicatorCache(df, timeframe=timeframe)
    allow_long = spec.direction in ("long_only", "long_short")
    allow_short = spec.direction in ("short_only", "long_short")

    le = evaluate_condition(spec.long_entry, ctx) if allow_long else np.zeros(ctx.n, dtype=bool)
    lx = evaluate_condition(spec.long_exit, ctx) if allow_long else np.zeros(ctx.n, dtype=bool)
    se = evaluate_condition(spec.short_entry, ctx) if allow_short else np.zeros(ctx.n, dtype=bool)
    sx = evaluate_condition(spec.short_exit, ctx) if allow_short else np.zeros(ctx.n, dtype=bool)
    atr_arr = ctx.get_atr(spec.atr_period)
    regimes = ctx.get_regimes()["primary_regime"]

    scenarios: list[dict[str, Any]] = []

    def _run_case(name: str, cost_override: ExecutionCostConfig, shift_delay: int = 0, drop_pct: float = 0.0, vol_shock: float = 1.0) -> None:
        o, h, l, c, t = ctx.open.copy(), ctx.high.copy(), ctx.low.copy(), ctx.close.copy(), ctx.open_time_ms.copy()
        le_c, lx_c, se_c, sx_c = le.copy(), lx.copy(), se.copy(), sx.copy()
        atr_c = atr_arr.copy()
        reg_c = list(regimes)

        if shift_delay > 0 and ctx.n > shift_delay:
            le_c = np.roll(le_c, shift_delay)
            le_c[:shift_delay] = False
            se_c = np.roll(se_c, shift_delay)
            se_c[:shift_delay] = False

        if drop_pct > 0 and ctx.n > 50:
            rng = np.random.default_rng(101)
            keep_mask = rng.random(ctx.n) >= drop_pct
            o, h, l, c, t = o[keep_mask], h[keep_mask], l[keep_mask], c[keep_mask], t[keep_mask]
            le_c, lx_c, se_c, sx_c = le_c[keep_mask], lx_c[keep_mask], se_c[keep_mask], sx_c[keep_mask]
            atr_c = atr_c[keep_mask]
            reg_c = [reg_c[idx] for idx, keep in enumerate(keep_mask) if keep]

        if vol_shock > 1.0:
            mid = (h + l) / 2.0
            half_rng = (h - l) / 2.0 * vol_shock
            h = np.maximum(mid + half_rng, np.maximum(o, c))
            l = np.maximum(1e-6, np.minimum(mid - half_rng, np.minimum(o, c)))

        out = run_vectorized_backtest(
            open_time_ms=t,
            open_=o,
            high=h,
            low=l,
            close=c,
            long_entry=le_c,
            long_exit=lx_c,
            short_entry=se_c,
            short_exit=sx_c,
            atr_arr=atr_c,
            regimes=reg_c,
            timeframe=timeframe,
            risk_per_trade_pct=spec.risk_per_trade_pct,
            sl_atr_mult=spec.sl_atr_mult,
            tp_atr_mult=spec.tp_atr_mult,
            max_holding_bars=spec.max_holding_bars,
            costs=cost_override,
            allow_long=allow_long,
            allow_short=allow_short,
        )
        m = out.metrics
        survived = m["net_profit"] > 0 and m["max_drawdown_pct"] < 45.0
        scenarios.append(
            {
                "scenario": name,
                "return_pct": m["return_on_capital_pct"],
                "sharpe": m["sharpe_ratio"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "trades": m["total_trades"],
                "survived": survived,
            }
        )

    _run_case("2x Transaction Fees", ExecutionCostConfig(**{**cfg.__dict__, "maker_fee_bps": cfg.maker_fee_bps * 2, "taker_fee_bps": cfg.taker_fee_bps * 2}))
    _run_case("3x Slippage & Spread", ExecutionCostConfig(**{**cfg.__dict__, "slippage_bps": cfg.slippage_bps * 3, "spread_bps": cfg.spread_bps * 3}))
    _run_case("1-Bar Execution Delay", cfg, shift_delay=1)
    _run_case("3% Missing Candles", cfg, drop_pct=0.03)
    _run_case("1.5x Volatility Shock", ExecutionCostConfig(**{**cfg.__dict__, "slippage_bps": cfg.slippage_bps * 2}), vol_shock=1.5)

    survival_rate = sum(1 for s in scenarios if s["survived"]) / len(scenarios) if scenarios else 0.0
    return {
        "survival_rate": round(survival_rate, 4),
        "scenarios": scenarios,
    }
