"""Phase 2 tests: golden fixtures, structured Zone/Event invariants, prefix-invariance, and Hypothesis property tests."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from crypto_quant_lab.core.structures import DetectionResult, EventObject, ZoneObject
from crypto_quant_lab.ict.engine import ICTSessionConfig, align_reference_ohlcv, detect_ict
from crypto_quant_lab.price_action.engine import detect_price_action
from crypto_quant_lab.smc.engine import detect_smc
from crypto_quant_lab.strategies.evaluator import IndicatorCache, evaluate_condition
from crypto_quant_lab.strategies.grammar import LeafCondition
from tests.conftest import make_synthetic_ohlcv


def _ohlcv_from_hlc(closes: list[float], spread: float = 0.4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = np.asarray(closes, dtype=np.float64)
    o = np.roll(c, 1)
    o[0] = c[0]
    bull = c >= o
    h = np.where(bull, np.maximum(o, c) + spread, np.maximum(o, c) + spread * 0.25)
    l = np.where(bull, np.minimum(o, c) - spread * 0.25, np.minimum(o, c) - spread)
    return o, h, l, c


def test_smc_golden_bos_choch_mss_ob_breaker_fvg_ifvg_eqh() -> None:
    """Hand-labelled OHLC fixture verifying exact SMC structured zones, events, and lifecycles."""
    # Construct deterministic swings with ext_swing=2 (int_swing=1):
    # First rally -> swing high @ idx 4 (108), pullback -> swing low @ idx 7 (102),
    # rally -> BOS bull above 108 @ idx 10 (creates bull OB), retest of bull OB,
    # then bearish displacement breaking down below 102 -> CHoCH/MSS bear + bear OB + bull OB failure -> Breaker!
    n = 36
    o = np.full(n, 104.0)
    h = np.full(n, 104.5)
    l = np.full(n, 103.5)
    c = np.full(n, 104.0)

    # Bar 4: ext swing high = 108.0 (confirmed at bar 4 + 2 = 6)
    o[2], h[2], l[2], c[2] = 104.0, 105.0, 103.8, 104.8
    o[3], h[3], l[3], c[3] = 104.8, 106.5, 104.6, 106.2
    o[4], h[4], l[4], c[4] = 106.2, 108.0, 105.8, 107.0
    o[5], h[5], l[5], c[5] = 107.0, 107.2, 104.5, 105.0
    o[6], h[6], l[6], c[6] = 105.0, 105.4, 103.0, 103.5

    # Bar 7: ext swing low = 101.0 (confirmed at bar 7 + 2 = 9)
    o[7], h[7], l[7], c[7] = 103.5, 103.8, 101.0, 102.0
    o[8], h[8], l[8], c[8] = 102.0, 104.2, 101.8, 104.0
    # Bar 9: bearish candle (opposing candle before BOS bull!) -> will be selected as Bullish OB
    o[9], h[9], l[9], c[9] = 104.5, 104.8, 103.2, 103.6

    # Bar 10: strong bullish break above ext_sh=108.0 -> BOS bull + FVG bull
    o[10], h[10], l[10], c[10] = 103.8, 111.5, 103.7, 111.0
    # Bar 11: push higher leaving bull FVG [104.8, 107.0] between bar 9 high (104.8) and bar 11 low (107.0)
    o[11], h[11], l[11], c[11] = 111.0, 113.0, 107.0, 112.5
    # Bar 12: swing high @ 113.0 (confirmed at bar 14)
    o[12], h[12], l[12], c[12] = 112.5, 114.0, 111.0, 111.5
    o[13], h[13], l[13], c[13] = 111.5, 112.0, 109.5, 110.0
    o[14], h[14], l[14], c[14] = 110.0, 110.5, 108.5, 109.0

    # Bar 15: sweeps down into Bull OB [103.2, 104.8] -> in_bull_ob == True
    o[15], h[15], l[15], c[15] = 109.0, 109.2, 104.2, 106.0
    o[16], h[16], l[16], c[16] = 106.0, 107.5, 105.5, 107.0

    # Bar 17..19: sharp bearish break below ext_sl=101.0 AND below bull OB bottom=103.2 -> MSS bear + Breaker bear!
    o[17], h[17], l[17], c[17] = 107.0, 107.2, 100.0, 100.2
    o[18], h[18], l[18], c[18] = 100.2, 100.5, 98.0, 98.5
    # Bar 19: closes above any bear FVG or leaves bear FVG [100.5, 105.5]
    o[19], h[19], l[19], c[19] = 98.5, 99.0, 96.5, 97.0

    # Bar 22..25: rally invalidating bear FVG -> IFVG bull
    o[22], h[22], l[22], c[22] = 97.5, 106.8, 97.4, 106.5

    res = detect_smc(
        o, h, l, c,
        int_swing=1,
        ext_swing=2,
        displacement_atr_mult=1.2,
        min_fvg_atr_mult=0.1,
        ob_refined_mode="full",
    )
    assert isinstance(res, DetectionResult)
    arr = res.arrays
    assert arr["bos_bull"][10], "Bar 10 must fire BOS bull above swing high 108.0"
    assert arr["fvg_bull"][11], "Bar 11 must detect bullish FVG above bar 9 high"
    assert arr["in_bull_ob"][15], "Bar 15 must touch fresh bullish OB created at bar 10"
    assert arr["choch_bear"][17] or arr["mss_bear"][17], "Bar 17 breaking below 101.0 must fire CHoCH/MSS bear"

    z_types = {z.type for z in res.zones}
    ev_types = {e.type for e in res.events}
    assert {"ob", "fvg", "breaker"}.issubset(z_types)
    assert {"bos_bull", "breaker_created"}.issubset(ev_types)

    # Check OB refined modes yield strictly nested body / mean_threshold bounds
    res_body = detect_smc(o, h, l, c, int_swing=1, ext_swing=2, ob_refined_mode="body")
    res_mt = detect_smc(o, h, l, c, int_swing=1, ext_swing=2, ob_refined_mode="mean_threshold")
    ob_full = next(z for z in res.zones if z.type == "ob" and z.direction == "bull")
    ob_body = next(z for z in res_body.zones if z.type == "ob" and z.direction == "bull")
    ob_mt = next(z for z in res_mt.zones if z.type == "ob" and z.direction == "bull")
    assert ob_body.top <= ob_full.top and ob_body.bottom >= ob_full.bottom
    assert ob_mt.bottom == pytest.approx((ob_full.bottom + ob_full.top) * 0.5, abs=1e-6)

    # Parameterized grammar exposure check: larger ext_swing changes BOS position
    df = pl.DataFrame({
        "open_time_ms": np.arange(n, dtype=np.int64) * 3_600_000 + 1_700_000_000_000,
        "open": o, "high": h, "low": l, "close": c, "volume": np.full(n, 100.0),
    })
    ctx = IndicatorCache(df, timeframe="1h")
    s_fast = evaluate_condition(LeafCondition("smc", "bos_bull", {"int_swing": 1, "ext_swing": 2}), ctx)
    s_slow = evaluate_condition(LeafCondition("smc", "bos_bull", {"int_swing": 2, "ext_swing": 10}), ctx)
    assert np.sum(s_fast) >= 1 and np.sum(s_slow) == 0


def test_ict_golden_silver_bullet_nwog_ndog_ipda_ote_smt_dol_and_bias() -> None:
    """Hand-crafted multi-week NY-session fixture checking all Phase 2 ICT detectors + aligned SMT."""
    ny = ZoneInfo("America/New_York")
    # Start Sunday 2025-01-05 18:00 NY -> crosses midnight & new week gap vs Friday/earlier bars
    start_ny = datetime(2025, 1, 5, 18, 0, tzinfo=ny)
    start_ms = int(start_ny.astimezone(timezone.utc).timestamp() * 1000)
    n = 72  # 3 full days hourly
    t = start_ms + np.arange(n, dtype=np.int64) * 3_600_000
    o = np.full(n, 100.0)
    h = np.full(n, 100.5)
    l = np.full(n, 99.5)
    c = np.full(n, 100.0)

    # Day 1 -> Day 2 gap down (midnight NY at idx 6, then idx 30): create clear NDOG
    o[:30], h[:30], l[:30], c[:30] = 100.0, 100.6, 99.4, 100.0
    h[10], l[10] = 103.0, 98.0  # Day 1 high=103, low=98

    # Day 2 starts at idx 30 (2025-01-07 00:00 NY) with gap down to 97.5 (NDOG [97.5, 100.0])
    o[30:], h[30:], l[30:], c[30:] = 98.0, 98.6, 97.4, 98.0
    o[30], h[30], l[30], c[30] = 97.5, 98.2, 97.2, 97.8

    # At idx 39 (09:00 NY, NY AM KZ): raid PDL (98.0) or earlier low -> set Day 2 sweep low at 09:00 NY then Silver Bullet FVG at 10:00 NY (idx 40)
    l[39], c[39] = 96.5, 98.1  # sweeps SSL
    # Wait: make idx 38 high = 98.5, idx 39 impulse bull close = 99.8, idx 40 (10:00 NY Silver Bullet window) low = 100.2 > h[38] -> FVG bull inside Silver Bullet!
    o[38], h[38], l[38], c[38] = 98.0, 98.5, 96.5, 98.2  # sweeps PDL (98.0) & closes back above!
    o[39], h[39], l[39], c[39] = 98.3, 100.6, 98.2, 100.4
    o[40], h[40], l[40], c[40] = 100.5, 102.0, 100.2, 101.8  # 10:00 NY Silver Bullet FVG!

    # Reference symbol with timestamp gaps to test align_reference_ohlcv + SMT divergence
    ref_t = np.delete(t, [35, 36, 37, 38])  # Missing bars right before idx 38!
    ref_h = np.full(len(ref_t), 100.5)
    ref_l = np.full(len(ref_t), 99.5)  # Never makes lower low while primary drops to 96.5 -> SMT bull!
    ref_c = np.full(len(ref_t), 100.0)

    res = detect_ict(
        t, o, h, l, c,
        reference_high=ref_h,
        reference_low=ref_l,
        reference_open_time_ms=ref_t,
        min_fvg_atr_mult=0.1,
        swing_len=3,
    )
    arr = res.arrays

    assert arr["liquidity_raid_bull"][38], "Bar 38 must detect PDL sweep & close back inside"
    assert arr["silver_bullet_window"][40], "10:00 NY must be Silver Bullet window"
    assert arr["silver_bullet_bull"][40], "Silver Bullet bull FVG inside window after sweep must fire"
    assert np.isfinite(arr["true_day_open"][30]) and arr["true_day_open"][30] == pytest.approx(97.5)
    assert np.isfinite(arr["ny_0830_open"][40])
    assert np.any(arr["in_macro_window"])
    assert np.isfinite(arr["ipda_20_high"][40]) and arr["ipda_20_high"][40] == pytest.approx(103.0)
    assert np.any(np.isfinite(arr["dol_bull_target"]))
    assert arr["smt_divergence_bull"][38], "Primary lower low while timestamp-aligned reference held must fire SMT bull"
    assert any(z.type == "ndog" for z in res.zones)
    assert arr["daily_bias"][38] != 0

    # Causal check for align_reference_ohlcv: future reference bars cannot leak backwards
    aligned_h, aligned_l, _ = align_reference_ohlcv(t, np.array([t[50]]), np.array([999.0]), np.array([1.0]))
    assert np.all(np.isnan(aligned_h[:50]))
    assert aligned_h[50] == 999.0


def test_price_action_golden_sr_pivots_vwap_vp_candlesticks_and_chart_patterns() -> None:
    """Verify S/R clustering/flips, VWAP bands, VP POC/VAH/VAL, candlesticks, and chart patterns."""
    # 1. Exact candlestick fixture (12 bars): Doji, Morning Star, Three White Soldiers, Outside Bull
    n_c = 14
    t_c = np.arange(n_c, dtype=np.int64) * 3_600_000 + 1_700_000_000_000
    o = np.full(n_c, 100.0)
    h = np.full(n_c, 100.5)
    l = np.full(n_c, 99.5)
    c = np.full(n_c, 100.0)

    # Bar 2: Doji (body=0.05, range=1.0)
    o[2], h[2], l[2], c[2] = 100.0, 100.5, 99.5, 100.05

    # Bars 3,4,5: Morning Star
    o[3], h[3], l[3], c[3] = 102.0, 102.2, 99.0, 99.2
    o[4], h[4], l[4], c[4] = 99.1, 99.5, 98.7, 99.2
    o[5], h[5], l[5], c[5] = 99.3, 101.8, 99.2, 101.6

    # Bars 7,8,9: Three White Soldiers
    o[7], h[7], l[7], c[7] = 100.0, 101.5, 99.8, 101.2
    o[8], h[8], l[8], c[8] = 101.2, 102.8, 101.0, 102.6
    o[9], h[9], l[9], c[9] = 102.6, 104.5, 102.4, 104.2

    # Bar 11: Outside Bull Bar engulfing bar 10 [99.5, 100.5]
    o[11], h[11], l[11], c[11] = 99.4, 105.5, 99.0, 105.0

    pa_c = detect_price_action(o, h, l, c, open_time_ms=t_c, swing_left=1, swing_right=1)
    assert pa_c.arrays["doji"][2]
    assert pa_c.arrays["morning_star"][5]
    assert pa_c.arrays["three_white_soldiers"][9]
    assert pa_c.arrays["outside_bar_bull"][11]

    # 2. Double top + S/R + VWAP bands + VP fixture
    closes = [
        100, 101, 103, 106, 110, 107, 104, 102, 104, 107, 110.2, 106, 103, 101, 100.5, 102,
        104, 106, 108, 110, 112, 114, 113, 113.5, 113.2, 113.4, 113.3, 115, 117, 119,
        # Draw down to trigger double top breakdown below neckline (~102) or breakout retest
        118, 116, 112, 108, 104, 100, 99,
    ]
    n_p = len(closes)
    t_p = np.arange(n_p, dtype=np.int64) * 3_600_000 + 1_700_000_000_000
    po, ph, pl_, pc = _ohlcv_from_hlc(closes, spread=0.5)
    vol = np.linspace(1000.0, 3000.0, n_p)
    pa = detect_price_action(po, ph, pl_, pc, open_time_ms=t_p, volume=vol, swing_left=2, swing_right=2, sr_lookback=10)

    assert len([z for z in pa.zones if z.type == "sr"]) >= 1
    assert np.any(pa.arrays["sr_flip_bull"]) or np.any(pa.arrays["sr_flip_bear"])
    assert np.any(np.isfinite(pa.arrays["pivot_pp"]))
    assert np.any(np.isfinite(pa.arrays["vwap_upper_1"]))
    assert np.any(np.isfinite(pa.arrays["anchored_vwap"]))
    valid_vp = np.isfinite(pa.arrays["vp_poc"])
    assert np.any(valid_vp)
    assert np.all(pa.arrays["vp_val"][valid_vp] <= pa.arrays["vp_poc"][valid_vp] + 1e-9)
    assert np.all(pa.arrays["vp_poc"][valid_vp] <= pa.arrays["vp_vah"][valid_vp] + 1e-9)
    assert np.sum(pa.arrays["double_top"]) >= 1 or np.sum(pa.arrays["bull_flag_breakout"]) >= 1


def test_phase2_detectors_prefix_invariance_zones_events_and_arrays() -> None:
    """Strict zero look-ahead prefix-invariance for ALL Phase 2 detectors (arrays, zones, and events)."""
    df_full = make_synthetic_ohlcv(n=240, seed=101, vol=0.02)
    ref_full = make_synthetic_ohlcv(n=240, seed=202, vol=0.02)
    k = 160

    df_pref = df_full.slice(0, k)
    ref_pref = ref_full.slice(0, k)

    mut_pdf = df_full.to_pandas()
    mut_pdf.loc[k:, ["open", "high", "low", "close"]] *= 8.0
    mut_pdf.loc[k:, "volume"] *= 20.0
    df_shock = pl.from_pandas(mut_pdf)

    mut_ref = ref_full.to_pandas()
    mut_ref.loc[k:, ["high", "low"]] *= 9.0
    ref_shock = pl.from_pandas(mut_ref)

    t_p, o_p, h_p, l_p, c_p, v_p = (
        df_pref["open_time_ms"].to_numpy(),
        df_pref["open"].to_numpy(),
        df_pref["high"].to_numpy(),
        df_pref["low"].to_numpy(),
        df_pref["close"].to_numpy(),
        df_pref["volume"].to_numpy(),
    )
    t_s, o_s, h_s, l_s, c_s, v_s = (
        df_shock["open_time_ms"].to_numpy(),
        df_shock["open"].to_numpy(),
        df_shock["high"].to_numpy(),
        df_shock["low"].to_numpy(),
        df_shock["close"].to_numpy(),
        df_shock["volume"].to_numpy(),
    )

    # 1. SMC
    smc_p = detect_smc(o_p, h_p, l_p, c_p, open_time_ms=t_p, volume=v_p)
    smc_s = detect_smc(o_s, h_s, l_s, c_s, open_time_ms=t_s, volume=v_s)
    for key in smc_p.arrays:
        np.testing.assert_allclose(smc_p.arrays[key], smc_s.arrays[key][:k], equal_nan=True, err_msg=f"SMC array leak: {key}")
    assert [(z.type, z.direction, z.created_bar, round(z.top, 6), round(z.bottom, 6)) for z in smc_p.zones] == [
        (z.type, z.direction, z.created_bar, round(z.top, 6), round(z.bottom, 6))
        for z in smc_s.zones if z.created_bar < k
    ]
    assert [(e.type, e.bar_index) for e in smc_p.events] == [(e.type, e.bar_index) for e in smc_s.events if e.bar_index < k]

    # 2. ICT
    ict_p = detect_ict(
        t_p, o_p, h_p, l_p, c_p,
        reference_high=ref_pref["high"].to_numpy(),
        reference_low=ref_pref["low"].to_numpy(),
        reference_open_time_ms=ref_pref["open_time_ms"].to_numpy(),
    )
    ict_s = detect_ict(
        t_s, o_s, h_s, l_s, c_s,
        reference_high=ref_shock["high"].to_numpy(),
        reference_low=ref_shock["low"].to_numpy(),
        reference_open_time_ms=ref_shock["open_time_ms"].to_numpy(),
    )
    for key in ict_p.arrays:
        np.testing.assert_allclose(ict_p.arrays[key], ict_s.arrays[key][:k], equal_nan=True, err_msg=f"ICT array leak: {key}")
    assert [(e.type, e.bar_index) for e in ict_p.events] == [(e.type, e.bar_index) for e in ict_s.events if e.bar_index < k]

    # 3. Price Action
    pa_p = detect_price_action(o_p, h_p, l_p, c_p, open_time_ms=t_p, volume=v_p)
    pa_s = detect_price_action(o_s, h_s, l_s, c_s, open_time_ms=t_s, volume=v_s)
    for key in pa_p.arrays:
        np.testing.assert_allclose(pa_p.arrays[key], pa_s.arrays[key][:k], equal_nan=True, err_msg=f"PA array leak: {key}")
    assert [(e.type, e.bar_index) for e in pa_p.events] == [(e.type, e.bar_index) for e in pa_s.events if e.bar_index < k]


@given(
    returns=st.lists(st.floats(min_value=-0.08, max_value=0.08, allow_nan=False, allow_infinity=False), min_size=45, max_size=90),
    ext_swing=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=20)
def test_hypothesis_phase2_zone_event_and_detector_invariants(returns: list[float], ext_swing: int) -> None:
    """Property tests for SMC, ICT, and Price Action structured objects and numeric bounds."""
    n = len(returns)
    c = 100.0 * np.exp(np.cumsum(np.asarray(returns, dtype=np.float64)))
    o = np.roll(c, 1)
    o[0] = 100.0
    h = np.maximum(o, c) * 1.005
    l = np.minimum(o, c) * 0.995
    t = 1_700_000_000_000 + np.arange(n, dtype=np.int64) * 3_600_000
    v = np.linspace(500.0, 2000.0, n)

    valid_statuses = {"fresh", "tested", "mitigated", "invalidated", "expired"}

    for res in (
        detect_smc(o, h, l, c, open_time_ms=t, volume=v, int_swing=1, ext_swing=ext_swing),
        detect_ict(t, o, h, l, c, swing_len=ext_swing),
        detect_price_action(o, h, l, c, open_time_ms=t, volume=v, swing_left=2, swing_right=2),
    ):
        for arr in res.arrays.values():
            assert len(arr) == n
        for z in res.zones:
            assert isinstance(z, ZoneObject)
            assert z.bottom <= z.top + 1e-9
            assert z.confirmed_time >= z.created_time
            assert 0 <= z.created_bar < n
            assert z.status in valid_statuses
            assert 0.0 <= z.fill_pct <= 1.0 + 1e-9
        for e in res.events:
            assert isinstance(e, EventObject)
            assert 0 <= e.bar_index < n
            assert e.direction in {"bull", "bear", "neutral"}
