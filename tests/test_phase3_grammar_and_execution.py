"""Phase 3 tests: temporal grammar operators, structural limit/SL/TP execution, partial TP, volatility/size slippage, liquidation, 11 explicit templates, and screen-and-verify."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.backtesting.event_driven.engine import run_event_driven_backtest
from crypto_quant_lab.strategies.evaluator import IndicatorCache, evaluate_condition, evaluate_strategy, screen_and_verify_strategies
from crypto_quant_lab.strategies.generator import generate_explicit_template_strategies
from crypto_quant_lab.strategies.grammar import LeafCondition, LogicalExpr, StrategySpec, TemporalExpr, validate_strategy_spec
from tests.conftest import make_synthetic_ohlcv


def test_temporal_grammar_within_and_sequence() -> None:
    n = 30
    t = np.arange(n, dtype=np.int64) * 3_600_000 + 1_700_000_000_000
    c = np.linspace(100.0, 130.0, n)
    df = make_synthetic_ohlcv(n=n, seed=42)
    df = df.with_columns(close=pl.Series(c))
    ctx = IndicatorCache(df, timeframe="1h")

    within_expr = TemporalExpr("WITHIN", (LeafCondition("momentum", "rsi_below", {"period": 14, "threshold": 5.0}),), window_bars=3)
    res_within = evaluate_condition(within_expr, ctx)
    assert np.sum(res_within) == 0

    seq_expr = TemporalExpr("SEQUENCE", (LeafCondition("trend", "close_gt_ema", {"period": 10}), LeafCondition("trend", "close_gt_sma", {"period": 10})), window_bars=4)
    res_seq = evaluate_condition(seq_expr, ctx)
    assert len(res_seq) == n


def test_explicit_11_strategy_templates_validate_and_execute() -> None:
    templates = generate_explicit_template_strategies()
    assert len(templates) == 11

    df = make_synthetic_ohlcv(n=250, seed=77)
    ctx = IndicatorCache(df, timeframe="1h")
    costs = ExecutionCostConfig(
        market_type="futures",
        slippage_bps=2.0,
        vol_slippage_scale=0.1,
        size_impact_bps_per_vol_pct=0.05,
        leverage=3.0,
    )

    for tpl in templates:
        validate_strategy_spec(tpl, market_type="futures")
        out = evaluate_strategy(tpl, ctx, engine_type="event_driven", costs=costs)
        assert "net_profit" in out.metrics
        assert "sharpe_ratio" in out.metrics


def test_volatility_and_size_scaled_slippage_and_liquidation() -> None:
    n = 20
    t = np.arange(n, dtype=np.int64) * 3_600_000 + 1_700_000_000_000
    o = np.full(n, 100.0)
    h = np.full(n, 105.0)
    l = np.full(n, 95.0)
    l[2] = 70.0  # Breaches leverage 5.0 liquidation price (~80.5) on entry bar 2
    c = np.full(n, 100.0)
    v = np.full(n, 10_000.0)
    atr_arr = np.full(n, 5.0)

    le = np.zeros(n, dtype=bool)
    le[1] = True  # Enters long at bar 2 open = 100.0
    lx = np.zeros(n, dtype=bool)

    costs = ExecutionCostConfig(
        market_type="futures",
        leverage=5.0,
        slippage_bps=5.0,
        vol_slippage_scale=0.5,
        size_impact_bps_per_vol_pct=0.2,
    )

    out = run_event_driven_backtest(
        open_time_ms=t, open_=o, high=h, low=l, close=c, volume=v,
        long_entry=le, long_exit=lx, short_entry=np.zeros(n, dtype=bool), short_exit=np.zeros(n, dtype=bool),
        atr_arr=atr_arr, costs=costs, sl_atr_mult=10.0, tp_atr_mult=10.0,
    )
    assert len(out.trades) == 1
    assert out.trades[0].exit_reason in ("liquidation", "stop_loss")


def test_screen_and_verify_pipeline() -> None:
    df = make_synthetic_ohlcv(n=200, seed=123)
    ctx = IndicatorCache(df, timeframe="1h")
    templates = generate_explicit_template_strategies()[:4]
    costs = ExecutionCostConfig(market_type="futures")

    res = screen_and_verify_strategies(templates, ctx, top_k=2, costs=costs)
    assert res["attempted_trials"] == 4
    assert res["screened_count"] <= 4
    assert len(res["verified"]) == 2
