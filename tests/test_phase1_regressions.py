"""Phase 1 regression tests: each test reproduces a confirmed bug before the fix and verifies the fix."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.backtesting.event_driven.engine import run_event_driven_backtest
from crypto_quant_lab.backtesting.vectorized.engine import run_vectorized_backtest
from crypto_quant_lab.core.exceptions import RiskViolationError, StrategyGrammarError
from crypto_quant_lab.ict.engine import compute_ict_features
from crypto_quant_lab.smc.engine import compute_smc_features
from crypto_quant_lab.strategies.evaluator import IndicatorCache, evaluate_condition, evaluate_strategy
from crypto_quant_lab.strategies.generator import build_family_condition_pools
from crypto_quant_lab.strategies.grammar import STRATEGY_FAMILIES, LeafCondition, StrategySpec, validate_strategy_spec
from tests.conftest import make_synthetic_ohlcv


def _make_ict_session_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hand-built 48-hour fixture in America/New_York with controlled Asian range, Judas sweep, and PO3 expansion."""
    ny = ZoneInfo("America/New_York")
    start_ny = datetime(2025, 1, 6, 18, 0, tzinfo=ny)
    start_ms = int(start_ny.astimezone(timezone.utc).timestamp() * 1000)
    n = 48
    t = start_ms + np.arange(n, dtype=np.int64) * 3_600_000
    o = np.full(n, 101.0)
    h = np.full(n, 101.6)
    l = np.full(n, 100.4)
    c = np.full(n, 101.0)

    # Bar 2..5 = 20:00..23:00 NY Day 1 Asian session -> tight range [100.0, 102.0]
    for i in range(2, 6):
        o[i], h[i], l[i], c[i] = 101.0, 102.0, 100.0, 101.0

    # Bar 6 = 00:00 NY Day 2 open = 101.0
    o[6], h[6], l[6], c[6] = 101.0, 101.4, 100.6, 101.1

    # Bar 9 = 03:00 NY (London KZ): sweeps Asian low 100.0 down to 98.8, closes strongly back above 102.2
    o[9], h[9], l[9], c[9] = 100.2, 102.4, 98.8, 102.2

    # Bar 15 = 09:00 NY (NY AM KZ): strong bullish displacement above Asian high 102.0 -> PO3 bull
    o[15], h[15], l[15], c[15] = 101.2, 105.5, 101.0, 105.2

    # Bar 26..29 = 20:00..23:00 NY Day 2 Asian session -> tight range [104.0, 106.0]
    for i in range(26, 30):
        o[i], h[i], l[i], c[i] = 105.0, 106.0, 104.0, 105.0

    # Bar 30 = 00:00 NY Day 3 open = 105.0
    o[30], h[30], l[30], c[30] = 105.0, 105.3, 104.7, 104.9

    # Bar 33 = 03:00 NY (London KZ): sweeps Asian high 106.0 up to 107.2, reverses below open 105.0 -> Judas bear
    o[33], h[33], l[33], c[33] = 105.8, 107.2, 103.6, 103.8

    # Bar 39 = 09:00 NY (NY AM KZ): strong bearish displacement below Asian low 104.0 -> PO3 bear
    o[39], h[39], l[39], c[39] = 104.8, 105.0, 100.5, 100.8

    return t, o, h, l, c


def test_ict_judas_swing_and_po3_fire_with_carried_asian_range() -> None:
    t, o, h, l, c = _make_ict_session_fixture()
    ict = compute_ict_features(t, o, h, l, c)
    assert np.sum(ict["judas_swing_bull"]) >= 1
    assert np.sum(ict["judas_swing_bear"]) >= 1
    assert np.sum(ict["po3_expansion_bull"]) >= 1
    assert np.sum(ict["po3_expansion_bear"]) >= 1


def test_smc_and_ict_zone_lifecycle_expiry_and_sanity_rates() -> None:
    df = make_synthetic_ohlcv(n=1000, seed=42, vol=0.018)
    t = df["open_time_ms"].to_numpy()
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()

    smc = compute_smc_features(o, h, l, c, max_zone_age_bars=80, min_fvg_atr_mult=0.2)
    ict = compute_ict_features(t, o, h, l, c)

    bull_brk_rate = float(np.mean(smc["in_bull_breaker"]))
    bear_brk_rate = float(np.mean(smc["in_bear_breaker"]))
    ce_rate = float(np.mean(ict["consequent_encroachment_touch"]))

    # Previously ~40% and ~41%; with lifecycle/mitigation/expiry these must be selective (<15%)
    assert bull_brk_rate < 0.15, f"in_bull_breaker rate too high: {bull_brk_rate:.3f}"
    assert bear_brk_rate < 0.15, f"in_bear_breaker rate too high: {bear_brk_rate:.3f}"
    assert ce_rate < 0.15, f"consequent_encroachment_touch rate too high: {ce_rate:.3f}"

    # Verify FVG minimum ATR filter suppresses tiny sub-threshold gaps
    smc_strict = compute_smc_features(o, h, l, c, min_fvg_atr_mult=2.5)
    assert np.sum(smc_strict["fvg_bull"]) + np.sum(smc_strict["fvg_bear"]) <= np.sum(smc["fvg_bull"]) + np.sum(smc["fvg_bear"])


def test_vectorized_entry_bar_sl_tp_conservative() -> None:
    n = 10
    t = np.arange(n, dtype=np.int64) * 3_600_000 + 1_700_000_000_000
    o = np.full(n, 100.0)
    h = np.full(n, 100.5)
    l = np.full(n, 99.5)
    c = np.full(n, 100.0)
    v = np.full(n, 10_000.0)
    atr_arr = np.full(n, 1.0)

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    # Signal at bar 2 -> fill at open[3]=100.0 (SL=98, TP=102); bar 3 itself hits BOTH SL & TP!
    le[2] = True
    l[3] = 95.0
    h[3] = 105.0

    costs = ExecutionCostConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, slippage_bps=0.0, spread_bps=0.0)
    vec_out = run_vectorized_backtest(t, o, h, l, c, le, lx, se, sx, atr_arr, sl_atr_mult=2.0, tp_atr_mult=2.0, costs=costs)
    ev_out = run_event_driven_backtest(t, o, h, l, c, v, le, lx, se, sx, atr_arr, sl_atr_mult=2.0, tp_atr_mult=2.0, costs=costs)

    assert len(vec_out.trades) == 1
    assert vec_out.trades[0].exit_idx == 3
    assert vec_out.trades[0].exit_reason == "stop_loss"
    assert vec_out.trades[0].net_pnl < 0

    assert len(ev_out.trades) == 1
    assert ev_out.trades[0].exit_idx == 3
    assert ev_out.trades[0].exit_reason == "stop_loss"
    assert ev_out.trades[0].net_pnl < 0


def test_funding_timestamps_series_and_engine_parity() -> None:
    # Start at 03:00 UTC (2023-11-14T03:00:00Z = 1699930800000) so index % 8 != UTC 00/08/16!
    start_ms = int(datetime(2023, 11, 14, 3, 0, tzinfo=timezone.utc).timestamp() * 1000)
    n = 20
    t = start_ms + np.arange(n, dtype=np.int64) * 3_600_000
    o = np.full(n, 100.0)
    h = np.full(n, 100.5)
    l = np.full(n, 99.5)
    c = np.full(n, 100.0)
    v = np.full(n, 1_000_000.0)
    atr_arr = np.full(n, 2.0)

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)
    le[0] = True   # Enters at bar 1 = 04:00 UTC
    lx[15] = True  # Exits at bar 16 = 19:00 UTC (crosses 08:00 UTC and 16:00 UTC)

    ts_08 = int(datetime(2023, 11, 14, 8, 0, tzinfo=timezone.utc).timestamp() * 1000)
    ts_16 = int(datetime(2023, 11, 14, 16, 0, tzinfo=timezone.utc).timestamp() * 1000)
    ts_11 = int(datetime(2023, 11, 14, 11, 0, tzinfo=timezone.utc).timestamp() * 1000)
    funding_map = {ts_08: 0.001, ts_16: 0.002, ts_11: 0.999}  # 11:00 UTC must NEVER be used

    costs = ExecutionCostConfig(
        market_type="futures",
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
        funding_rate_per_8h=0.0001,
    )
    vec = run_vectorized_backtest(
        t, o, h, l, c, le, lx, se, sx, atr_arr, sl_atr_mult=2.0, tp_atr_mult=5.0, costs=costs, funding_rates_map=funding_map
    )
    ev = run_event_driven_backtest(
        t, o, h, l, c, v, le, lx, se, sx, atr_arr, sl_atr_mult=2.0, tp_atr_mult=5.0, costs=costs,
        max_daily_loss_pct=100.0, max_portfolio_dd_pct=100.0, max_consecutive_losses_halt=1000,
        funding_rates_map=funding_map,
    )

    # qty = (100_000 * 1% = 1000) / (2 * 2 = 4) = 250 units -> notional = 25_000
    # Funding at 08:00 = 25_000 * 0.001 = 25.0; at 16:00 = 25_000 * 0.002 = 50.0 -> total = 75.0
    assert len(vec.trades) == 1
    assert len(ev.trades) == 1
    assert vec.trades[0].funding_paid == pytest.approx(75.0, abs=1e-6)
    assert ev.trades[0].funding_paid == pytest.approx(75.0, abs=1e-6)
    assert vec.trades[0].net_pnl == pytest.approx(ev.trades[0].net_pnl, abs=1e-6)
    assert vec.raw_equity_array[-1] == pytest.approx(ev.raw_equity_array[-1], abs=1e-6)
    assert vec.metrics["funding_source"] == "historical_series"
    assert ev.metrics["funding_source"] == "historical_series"


def test_spot_market_disallows_short_selling() -> None:
    df = make_synthetic_ohlcv(n=120, seed=5)
    ctx = IndicatorCache(df, timeframe="1h")
    short_spec = StrategySpec(
        name="spot_short_invalid",
        family="trend_following",
        direction="short_only",
        long_entry=None,
        long_exit=None,
        short_entry=LeafCondition("trend", "ema_lt_ema", {"fast": 10, "slow": 30}),
        short_exit=None,
    )
    with pytest.raises(StrategyGrammarError):
        validate_strategy_spec(short_spec, market_type="spot")

    with pytest.raises(StrategyGrammarError):
        evaluate_strategy(short_spec, ctx, engine_type="vectorized", costs=ExecutionCostConfig(market_type="spot"))

    t = ctx.open_time_ms
    se = np.zeros(ctx.n, dtype=bool)
    se[15] = True
    with pytest.raises(RiskViolationError):
        run_vectorized_backtest(
            t, ctx.open, ctx.high, ctx.low, ctx.close,
            np.zeros(ctx.n, dtype=bool), np.zeros(ctx.n, dtype=bool), se, np.zeros(ctx.n, dtype=bool),
            ctx.get_atr(14), costs=ExecutionCostConfig(market_type="spot"), allow_short=True,
        )


def test_htf_trend_timestamp_aligned_and_prefix_invariant() -> None:
    # Build 1h series starting at 01:00 UTC with a 2-hour missing gap at index 10 so index-modulo breaks
    start_ms = int(datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc).timestamp() * 1000)
    n = 120
    times = []
    cur = start_ms
    for i in range(n):
        times.append(cur)
        cur += 3 * 3_600_000 if i == 10 else 3_600_000
    t = np.array(times, dtype=np.int64)
    c = np.linspace(100.0, 140.0, n)
    df = pl.DataFrame({
        "open_time_ms": t,
        "open": c - 0.2,
        "high": c + 0.5,
        "low": c - 0.5,
        "close": c,
        "volume": np.full(n, 1000.0),
    })
    ctx_full = IndicatorCache(df, timeframe="1h")
    htf_full = ctx_full.get_htf_trend(factor=4, ema_period=3)

    # Prefix invariance even with missing candle gap & future shock
    k = 80
    pdf = df.to_pandas()
    pdf.loc[k:, "close"] *= 5.0
    ctx_pref = IndicatorCache(df.slice(0, k), timeframe="1h")
    ctx_shock = IndicatorCache(pl.from_pandas(pdf), timeframe="1h")
    np.testing.assert_array_equal(ctx_pref.get_htf_trend(4, 3), ctx_shock.get_htf_trend(4, 3)[:k])

    # Because prices strictly trend upward, completed 4h UTC candles must establish +1 trend
    assert htf_full[-1] == 1
    # Bar 2 is 03:00 UTC (closes 00:00-04:00 #1), Bar 6 is 07:00 UTC (closes 04:00-08:00 #2),
    # Bar 10 is 11:00 UTC (closes 08:00-12:00 #3). With ema_period=3, bar 10 is first non-zero!
    # (Index-modulo factor=4 would wait until index 11 = 14:00 UTC!)
    assert htf_full[9] == 0
    assert htf_full[10] == 1


def test_all_generator_signals_are_alive_on_realistic_fixture() -> None:
    # 6 months of hourly data (~4320 bars) with drift=0 and realistic volatility
    df = make_synthetic_ohlcv(n=4320, seed=77, drift=0.0, vol=0.015)
    ctx = IndicatorCache(df, timeframe="1h")

    seen_leaves: dict[str, LeafCondition] = {}
    for fam in STRATEGY_FAMILIES:
        for side in ("long", "short"):
            p1, p2, p3 = build_family_condition_pools(fam, side)
            for leaf in p1 + p2 + p3:
                key = f"{leaf.category}:{leaf.signal}:{sorted(leaf.params.items())}"
                seen_leaves[key] = leaf

    dead = []
    for key, leaf in seen_leaves.items():
        sig = evaluate_condition(leaf, ctx)
        if int(np.sum(sig)) == 0:
            dead.append(key)

    assert not dead, f"Dead generator signals with 0 True bars: {dead}"


def test_trade_pnl_slippage_fee_funding_hand_computed_consistency() -> None:
    n = 12
    t0 = int(datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc).timestamp() * 1000)
    t = t0 + np.arange(n, dtype=np.int64) * 3_600_000
    o = np.full(n, 100.0)
    h = np.full(n, 100.2)
    l = np.full(n, 99.8)
    c = np.full(n, 100.0)
    v = np.full(n, 1_000_000.0)
    atr_arr = np.full(n, 5.0)

    # Signal at bar 1 -> enters LONG at bar 2 open (03:00 UTC) @ raw=100.0
    # Holds across 08:00 UTC (bar 7) -> exits at bar 9 open (10:00 UTC) @ raw=110.0
    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)
    le[1] = True
    lx[8] = True
    o[9] = 110.0
    h[9] = 110.2
    l[9] = 109.8
    c[9] = 110.0

    costs = ExecutionCostConfig(
        market_type="futures",
        maker_fee_bps=10.0,
        taker_fee_bps=10.0,     # 0.10% fee
        slippage_bps=4.0,
        spread_bps=2.0,         # total impact = (4 + 2*0.5)/10000 = 5 bps = 0.0005
        funding_rate_per_8h=0.0002,
    )

    for engine_fn, extra in [
        (run_vectorized_backtest, {}),
        (run_event_driven_backtest, {"volume": v, "max_daily_loss_pct": 100.0, "max_portfolio_dd_pct": 100.0, "max_consecutive_losses_halt": 1000}),
    ]:
        kwargs = dict(
            open_time_ms=t, open_=o, high=h, low=l, close=c,
            long_entry=le, long_exit=lx, short_entry=se, short_exit=sx,
            atr_arr=atr_arr, sl_atr_mult=2.0, tp_atr_mult=10.0,
            initial_capital=100_000.0, risk_per_trade_pct=1.0, costs=costs,
        )
        kwargs.update(extra)
        out = engine_fn(**kwargs)
        assert len(out.trades) == 1
        tr = out.trades[0]

        # Hand check: stop_dist = 2.0 * 5.0 = 10.0 -> qty = 1000 / 10 = 100.0
        # raw_entry = 100.0, raw_exit = 110.0 -> gross_pnl = (110 - 100) * 100 = 1000.0
        assert tr.quantity == pytest.approx(100.0, abs=1e-6)
        assert tr.entry_price == pytest.approx(100.0, abs=1e-6)
        assert tr.exit_price == pytest.approx(110.0, abs=1e-6)
        assert tr.gross_pnl == pytest.approx(1000.0, abs=1e-6)

        # Entry slip = 100 * 0.0005 * 100 = 5.0 (fill_entry = 100.05)
        # Exit slip = 110 * 0.0005 * 100 = 5.5 (fill_exit = 109.945)
        # Slippage total = 10.5
        assert tr.slippage_paid == pytest.approx(10.5, abs=1e-6)

        # Fees: entry = 100 * 100.05 * 0.001 = 10.005; exit = 100 * 109.945 * 0.001 = 10.9945 -> 20.9995
        assert tr.fee_paid == pytest.approx(20.9995, abs=1e-6)

        # Funding at 08:00 UTC (o[7]=100.0): 100 * 100.0 * 0.0002 = 2.0
        assert tr.funding_paid == pytest.approx(2.0, abs=1e-6)

        # Exact identity: net_pnl = gross_pnl - fee_paid - slippage_paid - funding_paid
        expected_net = tr.gross_pnl - tr.fee_paid - tr.slippage_paid - tr.funding_paid
        assert tr.net_pnl == pytest.approx(expected_net, abs=1e-6)
        assert out.raw_equity_array[-1] - 100_000.0 == pytest.approx(tr.net_pnl, abs=1e-6)
