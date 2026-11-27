"""Backtesting accuracy, OHLC SL/TP conservatism, validation, overfitting, portfolio, and risk tests."""

from __future__ import annotations

import numpy as np
import pytest

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.backtesting.event_driven.engine import run_event_driven_backtest
from crypto_quant_lab.backtesting.vectorized.engine import run_vectorized_backtest
from crypto_quant_lab.core.exceptions import RiskViolationError
from crypto_quant_lab.portfolio.engine import construct_portfolio
from crypto_quant_lab.risk.manager import RiskConfig, RiskManager
from crypto_quant_lab.strategies.evaluator import IndicatorCache
from crypto_quant_lab.strategies.grammar import LeafCondition, StrategySpec
from crypto_quant_lab.validation.monte_carlo import run_monte_carlo_simulation
from crypto_quant_lab.validation.out_of_sample import split_chronological
from crypto_quant_lab.validation.overfitting import (
    benjamini_hochberg_fdr,
    compute_cscv_pbo,
    compute_deflated_sharpe_ratio,
    parameter_stability_analysis,
)
from crypto_quant_lab.validation.ranking import HardRejectionGates, evaluate_and_rank_strategy
from crypto_quant_lab.validation.stress_testing import run_stress_tests
from crypto_quant_lab.validation.walk_forward import run_walk_forward_analysis
from tests.conftest import make_synthetic_ohlcv


def test_conservative_ohlc_sl_tp_ambiguous_candle_resolution() -> None:
    """
    If a single OHLC bar's [low, high] spans BOTH the stop-loss AND the take-profit,
    both backtesting engines MUST conservatively assume STOP LOSS hit first.
    """
    n = 10
    t = np.arange(n, dtype=np.int64) * 3_600_000 + 1_700_000_000_000
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l = np.full(n, 99.0)
    c = np.full(n, 100.0)
    v = np.full(n, 10_000.0)
    atr_arr = np.full(n, 1.0)

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    # Signal at bar 2 -> enters LONG at bar 3 open (100.0); SL ~ 98.0, TP ~ 102.0
    long_entry[2] = True
    # Bar 4 violently spans BOTH low=90 and high=115!
    l[4] = 90.0
    h[4] = 115.0

    costs = ExecutionCostConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, slippage_bps=0.0, spread_bps=0.0)
    vec_out = run_vectorized_backtest(
        t, o, h, l, c, long_entry, long_exit, short_entry, short_exit, atr_arr,
        sl_atr_mult=2.0, tp_atr_mult=2.0, costs=costs,
    )
    ev_out = run_event_driven_backtest(
        t, o, h, l, c, v, long_entry, long_exit, short_entry, short_exit, atr_arr,
        sl_atr_mult=2.0, tp_atr_mult=2.0, costs=costs,
    )

    assert len(vec_out.trades) == 1
    assert vec_out.trades[0].exit_reason == "stop_loss"
    assert vec_out.trades[0].net_pnl < 0

    assert len(ev_out.trades) == 1
    assert ev_out.trades[0].exit_reason == "stop_loss"
    assert ev_out.trades[0].net_pnl < 0


def test_fee_slippage_and_funding_impact() -> None:
    n = 30
    t = np.arange(n, dtype=np.int64) * 3_600_000 + 1_700_000_000_000
    o = np.linspace(100.0, 129.0, n)
    c = o + 0.8
    h = c + 0.3
    l = o - 0.3
    v = np.full(n, 50_000.0)
    atr_arr = np.full(n, 2.0)

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)
    le[2] = True
    lx[25] = True

    zero_cost = ExecutionCostConfig(market_type="futures", maker_fee_bps=0.0, taker_fee_bps=0.0, slippage_bps=0.0, spread_bps=0.0, funding_rate_per_8h=0.0)
    real_cost = ExecutionCostConfig(market_type="futures", maker_fee_bps=2.0, taker_fee_bps=5.0, slippage_bps=5.0, spread_bps=2.0, funding_rate_per_8h=0.0005)

    r_zero = run_event_driven_backtest(t, o, h, l, c, v, le, lx, se, sx, atr_arr, sl_atr_mult=5.0, tp_atr_mult=20.0, costs=zero_cost)
    r_real = run_event_driven_backtest(t, o, h, l, c, v, le, lx, se, sx, atr_arr, sl_atr_mult=5.0, tp_atr_mult=20.0, costs=real_cost)

    assert r_zero.metrics["net_profit"] > r_real.metrics["net_profit"]
    assert r_real.metrics["total_fees"] > 0
    assert r_real.metrics["total_slippage"] > 0
    assert r_real.metrics["total_funding"] > 0
    # Verify required 26+ metric keys exist
    for key in (
        "net_profit", "gross_profit", "gross_loss", "profit_factor", "win_rate", "loss_rate",
        "avg_win", "avg_loss", "expectancy", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "max_drawdown_pct", "avg_drawdown_pct", "recovery_factor", "ulcer_index",
        "return_on_capital_pct", "annualized_return_pct", "total_trades",
        "trade_frequency_per_day", "avg_holding_bars", "avg_holding_hours",
        "max_consecutive_wins", "max_consecutive_losses", "exposure_pct", "turnover",
    ):
        assert key in r_real.metrics


def test_chronological_validation_overfitting_portfolio_and_risk() -> None:
    df = make_synthetic_ohlcv(n=500, seed=31, drift=0.0008, vol=0.015)
    splits = split_chronological(df, 0.6, 0.2, 0.2)
    assert splits.train_range[1] < splits.val_range[0]
    assert splits.val_range[1] < splits.test_range[0]

    spec = StrategySpec(
        name="trend_test",
        family="trend_following",
        direction="long_only",
        long_entry=LeafCondition("trend", "ema_gt_ema", {"fast": 10, "slow": 30}),
        long_exit=LeafCondition("trend", "ema_cross_below", {"fast": 10, "slow": 30}),
        short_entry=None,
        short_exit=None,
        sl_atr_mult=2.0,
        tp_atr_mult=3.0,
    )

    val_ctx = IndicatorCache(splits.validation, timeframe="1h")
    oos_out = run_event_driven_backtest(
        val_ctx.open_time_ms, val_ctx.open, val_ctx.high, val_ctx.low, val_ctx.close, val_ctx.volume,
        val_ctx.get_ema(10) > val_ctx.get_ema(30),
        np.zeros(val_ctx.n, dtype=bool),
        np.zeros(val_ctx.n, dtype=bool),
        np.zeros(val_ctx.n, dtype=bool),
        val_ctx.get_atr(14),
    )

    wf = run_walk_forward_analysis(df.slice(0, splits.train.height + splits.validation.height), spec, num_windows=4, mode="rolling")
    assert wf["num_windows"] == 4

    mc = run_monte_carlo_simulation(oos_out.trades, simulations=1000, method="trade_permutation", seed=42)
    assert mc["simulations"] == 1000

    stab = parameter_stability_analysis(val_ctx, spec)
    assert 0.0 <= stab["stability_score"] <= 1.0
    assert 0.0 <= stab["cscv_pbo"]["pbo"] <= 1.0

    # CSCV PBO directly on random matrix
    rng = np.random.default_rng(42)
    pbo_res = compute_cscv_pbo(rng.normal(0.0, 0.01, size=(200, 8)), n_partitions=8)
    assert pbo_res["combinations"] == 70
    assert 0.0 <= pbo_res["pbo"] <= 1.0

    # DSR & BH-FDR
    dsr = compute_deflated_sharpe_ratio(1.5, oos_out.bar_returns, n_trials=200)
    assert 0.0 <= dsr["dsr"] <= 1.0
    qvals = benjamini_hochberg_fdr([0.001, 0.04, 0.25, 0.80])
    assert len(qvals) == 4 and qvals[0] <= qvals[-1]

    # Stress testing
    stress = run_stress_tests(splits.validation, spec)
    assert len(stress["scenarios"]) == 5

    # Hard rejection gates
    bad_rank = evaluate_and_rank_strategy(
        oos_metrics={"total_trades": 2, "sharpe_ratio": -1.0, "max_drawdown_pct": 60.0, "profit_factor": 0.4},
        wf_result=wf,
        mc_result=mc,
        stability_result=stab,
        stress_result=stress,
        dsr_result={"dsr": 0.05},
        gates=HardRejectionGates(),
    )
    assert bad_rank["passed_gates"] is False
    assert len(bad_rank["rejection_reasons"]) >= 3

    # Portfolio Construction (all 6 methods)
    r1 = rng.normal(0.0005, 0.01, size=200)
    r2 = rng.normal(0.0004, 0.015, size=200)
    r3 = rng.normal(0.0003, 0.008, size=200)
    for method in ("equal_weight", "volatility_weighted", "risk_parity", "min_variance", "sharpe_weighted", "max_dd_constrained"):
        port = construct_portfolio([1, 2, 3], ["S1", "S2", "S3"], [r1, r2, r3], allocation_method=method, max_single_weight=0.6)
        w_sum = sum(a["weight"] for a in port["allocations"])
        assert abs(w_sum - 1.0) < 0.02

    # RiskManager enforcement
    rm = RiskManager(RiskConfig(max_daily_loss_pct=5.0, max_portfolio_drawdown_pct=20.0, max_open_positions=2, emergency_halt=False))
    with pytest.raises(RiskViolationError):
        rm.validate_new_order(equity=75_000, peak_equity=100_000, daily_pnl=0, open_positions_count=0, asset_exposure_notional=0, order_notional=1000, consecutive_losses=0)
    with pytest.raises(RiskViolationError):
        rm.validate_new_order(equity=100_000, peak_equity=100_000, daily_pnl=-6000, open_positions_count=0, asset_exposure_notional=0, order_notional=1000, consecutive_losses=0)
