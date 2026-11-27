"""
Objective, strictly causal Support/Resistance, Price Action & Chart Pattern engine (zero look-ahead).

Returns structured ZoneObject and EventObject instances via `detect_price_action(...)`
as the single source of truth for strategies, signal engine, and charts, while deriving
screener arrays in `compute_price_action_features(...)`.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from crypto_quant_lab.core.structures import DetectionResult, EventObject, ZoneObject
from crypto_quant_lab.indicators.volatility import atr


def _safe_times(open_time_ms: np.ndarray | None, n: int) -> np.ndarray:
    if open_time_ms is not None and len(open_time_ms) == n:
        return np.asarray(open_time_ms, dtype=np.int64)
    return np.arange(n, dtype=np.int64) * 3_600_000


def _round_step_for_price(px: float) -> float:
    if px <= 0 or not np.isfinite(px):
        return 1.0
    exp = math.floor(math.log10(px))
    base = 10.0 ** max(exp - 1, -4)
    if px / base >= 50:
        return base * 5.0
    return base


def _rolling_vp_poc_vah_val(
    h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray, bins: int = 16
) -> tuple[float, float, float]:
    p_min = float(np.min(l))
    p_max = float(np.max(h))
    if p_max <= p_min or np.sum(v) <= 0:
        px = float(c[-1]) if len(c) else 0.0
        return px, px, px
    edges = np.linspace(p_min, p_max, bins + 1)
    hist = np.zeros(bins, dtype=np.float64)
    tp = (h + l + c) / 3.0
    bin_idx = np.clip(np.digitize(tp, edges) - 1, 0, bins - 1)
    np.add.at(hist, bin_idx, v)
    poc_idx = int(np.argmax(hist))
    poc_px = float((edges[poc_idx] + edges[poc_idx + 1]) * 0.5)
    target_v = 0.70 * float(np.sum(hist))
    lo_i = poc_idx
    hi_i = poc_idx
    acc = float(hist[poc_idx])
    while acc < target_v and (lo_i > 0 or hi_i < bins - 1):
        lv = hist[lo_i - 1] if lo_i > 0 else -1.0
        rv = hist[hi_i + 1] if hi_i < bins - 1 else -1.0
        if rv >= lv and hi_i < bins - 1:
            hi_i += 1
            acc += float(hist[hi_i])
        elif lo_i > 0:
            lo_i -= 1
            acc += float(hist[lo_i])
        else:
            break
    return poc_px, float(edges[hi_i + 1]), float(edges[lo_i])


def detect_price_action(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_time_ms: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    swing_left: int = 3,
    swing_right: int = 3,
    sr_lookback: int = 50,
    atr_period: int = 14,
    cluster_tol_atr: float = 0.6,
    vp_lookback: int = 120,
    max_sr_zones: int = 12,
) -> DetectionResult:
    """
    Strictly causal Price Action, S/R clustering, Pivot, VWAP-band, Volume Profile,
    Candlestick, and Chart Pattern detector.
    A swing point at bar `j` is ONLY confirmed at bar `i = j + swing_right`.
    """
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    n = len(c)
    times = _safe_times(open_time_ms, n)
    vol = np.asarray(volume, dtype=np.float64) if volume is not None and len(volume) == n else np.ones(n, dtype=np.float64)

    swing_left = max(1, int(swing_left))
    swing_right = max(1, int(swing_right))

    atr_vals = atr(h, l, c, period=atr_period)

    # Swings & structure
    swing_high_confirmed = np.zeros(n, dtype=bool)
    swing_low_confirmed = np.zeros(n, dtype=bool)
    last_swing_high = np.full(n, np.nan, dtype=np.float64)
    last_swing_low = np.full(n, np.nan, dtype=np.float64)
    prev_swing_high = np.full(n, np.nan, dtype=np.float64)
    prev_swing_low = np.full(n, np.nan, dtype=np.float64)
    hh = np.zeros(n, dtype=bool)
    hl = np.zeros(n, dtype=bool)
    lh = np.zeros(n, dtype=bool)
    ll = np.zeros(n, dtype=bool)

    # Rolling & clustered S/R + flips + round numbers + pivots
    resistance = np.full(n, np.nan, dtype=np.float64)
    support = np.full(n, np.nan, dtype=np.float64)
    at_sr_support = np.zeros(n, dtype=bool)
    at_sr_resistance = np.zeros(n, dtype=bool)
    sr_flip_bull = np.zeros(n, dtype=bool)
    sr_flip_bear = np.zeros(n, dtype=bool)
    near_round_number = np.zeros(n, dtype=bool)

    pivot_pp = np.full(n, np.nan, dtype=np.float64)
    pivot_r1 = np.full(n, np.nan, dtype=np.float64)
    pivot_s1 = np.full(n, np.nan, dtype=np.float64)
    pivot_r2 = np.full(n, np.nan, dtype=np.float64)
    pivot_s2 = np.full(n, np.nan, dtype=np.float64)

    # VWAP + Bands + Anchored VWAP
    vwap_arr = np.full(n, np.nan, dtype=np.float64)
    vwap_upper_1 = np.full(n, np.nan, dtype=np.float64)
    vwap_lower_1 = np.full(n, np.nan, dtype=np.float64)
    vwap_upper_2 = np.full(n, np.nan, dtype=np.float64)
    vwap_lower_2 = np.full(n, np.nan, dtype=np.float64)
    anchored_vwap_arr = np.full(n, np.nan, dtype=np.float64)
    vwap_band_bounce_bull = np.zeros(n, dtype=bool)
    vwap_band_bounce_bear = np.zeros(n, dtype=bool)
    above_anchored_vwap = np.zeros(n, dtype=bool)

    # Volume profile (POC/VAH/VAL) & HTF levels
    vp_poc = np.full(n, np.nan, dtype=np.float64)
    vp_vah = np.full(n, np.nan, dtype=np.float64)
    vp_val = np.full(n, np.nan, dtype=np.float64)
    at_vp_poc = np.zeros(n, dtype=bool)
    at_vp_vah = np.zeros(n, dtype=bool)
    at_vp_val = np.zeros(n, dtype=bool)
    htf_resistance = np.full(n, np.nan, dtype=np.float64)
    htf_support = np.full(n, np.nan, dtype=np.float64)

    # Breakouts & trendlines
    breakout_up = np.zeros(n, dtype=bool)
    breakdown_down = np.zeros(n, dtype=bool)
    failed_breakout_up = np.zeros(n, dtype=bool)
    failed_breakdown_down = np.zeros(n, dtype=bool)
    trendline_slope_bull = np.zeros(n, dtype=np.float64)
    trendline_slope_bear = np.zeros(n, dtype=np.float64)
    trendline_break_retest_bull = np.zeros(n, dtype=bool)
    trendline_break_retest_bear = np.zeros(n, dtype=bool)
    breakout_retest_bull = np.zeros(n, dtype=bool)
    breakout_retest_bear = np.zeros(n, dtype=bool)

    # Candlestick patterns
    inside_bar = np.zeros(n, dtype=bool)
    bullish_engulfing = np.zeros(n, dtype=bool)
    bearish_engulfing = np.zeros(n, dtype=bool)
    bullish_pin_bar = np.zeros(n, dtype=bool)
    bearish_pin_bar = np.zeros(n, dtype=bool)
    range_expansion = np.zeros(n, dtype=bool)
    doji = np.zeros(n, dtype=bool)
    morning_star = np.zeros(n, dtype=bool)
    evening_star = np.zeros(n, dtype=bool)
    three_white_soldiers = np.zeros(n, dtype=bool)
    three_black_crows = np.zeros(n, dtype=bool)
    outside_bar_bull = np.zeros(n, dtype=bool)
    outside_bar_bear = np.zeros(n, dtype=bool)

    # Chart patterns
    double_top = np.zeros(n, dtype=bool)
    double_bottom = np.zeros(n, dtype=bool)
    head_and_shoulders = np.zeros(n, dtype=bool)
    inverse_head_and_shoulders = np.zeros(n, dtype=bool)
    triangle_breakout_up = np.zeros(n, dtype=bool)
    triangle_breakout_down = np.zeros(n, dtype=bool)
    bull_flag_breakout = np.zeros(n, dtype=bool)
    bear_flag_breakout = np.zeros(n, dtype=bool)
    rising_wedge_breakdown = np.zeros(n, dtype=bool)
    falling_wedge_breakout = np.zeros(n, dtype=bool)

    all_zones: list[ZoneObject] = []
    all_events: list[EventObject] = []
    z_seq = 0
    ev_seq = 0

    def _add_zone(ztype: str, direction: str, bottom: float, top: float, bar_i: int, src: str, strength: float = 1.0, meta: dict[str, Any] | None = None) -> ZoneObject:
        nonlocal z_seq
        z_seq += 1
        ts = int(times[bar_i])
        z = ZoneObject(
            id=f"pa_{ztype}_{direction}_{z_seq}",
            type=ztype,
            direction=direction,
            top=max(float(bottom), float(top)),
            bottom=min(float(bottom), float(top)),
            created_bar=bar_i,
            created_time=ts,
            confirmed_time=ts,
            status="fresh",
            source_event=src,
            strength=float(strength),
            meta=dict(meta or {}),
        )
        all_zones.append(z)
        return z

    def _add_event(etype: str, direction: str, bar_i: int, px: float, level: float | None = None, linked: list[str] | None = None) -> EventObject:
        nonlocal ev_seq
        ev_seq += 1
        ev = EventObject(
            id=f"pa_ev_{etype}_{ev_seq}",
            type=etype,
            direction=direction,
            time=int(times[bar_i]),
            bar_index=bar_i,
            price=float(px),
            level=float(level) if level is not None and np.isfinite(level) else None,
            linked_zone_ids=list(linked or []),
        )
        all_events.append(ev)
        return ev

    # State tracking
    cur_sh, prev_sh = np.nan, np.nan
    cur_sl, prev_sl = np.nan, np.nan
    swing_highs_hist: list[tuple[int, float]] = []   # (bar_idx_of_swing, price)
    swing_lows_hist: list[tuple[int, float]] = []
    sr_zones: list[ZoneObject] = []

    # Daily VWAP + pivots state
    cur_day = None
    day_hi, day_lo, day_cl = -np.inf, np.inf, np.nan
    prev_day_pp, prev_day_r1, prev_day_s1, prev_day_r2, prev_day_s2 = np.nan, np.nan, np.nan, np.nan, np.nan
    cum_pv, cum_v, cum_p2v = 0.0, 0.0, 0.0
    avwap_pv, avwap_v = 0.0, 0.0
    avwap_active = False

    # Recent breakout levels for breakout-retest tracking
    recent_up_break: tuple[float, int] | None = None
    recent_dn_break: tuple[float, int] | None = None
    # Recent trendline breaks for break-retest
    recent_tl_bull_break: tuple[float, float, int] | None = None  # (slope, intercept, break_bar)
    recent_tl_bear_break: tuple[float, float, int] | None = None

    last_vp = (np.nan, np.nan, np.nan)

    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    for i in range(n):
        ts = int(times[i])
        dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        d_key = dt.date()
        a = atr_vals[i] if np.isfinite(atr_vals[i]) and atr_vals[i] > 0 else max(h[i] - l[i], 1e-9)

        # Daily UTC reset for pivots & VWAP bands
        tp = (h[i] + l[i] + c[i]) / 3.0
        vi = max(float(vol[i]), 1e-9)
        if cur_day != d_key:
            if cur_day is not None and np.isfinite(day_cl):
                pp = (day_hi + day_lo + day_cl) / 3.0
                prev_day_pp = pp
                prev_day_r1 = 2.0 * pp - day_lo
                prev_day_s1 = 2.0 * pp - day_hi
                prev_day_r2 = pp + (day_hi - day_lo)
                prev_day_s2 = pp - (day_hi - day_lo)
            cur_day = d_key
            day_hi, day_lo, day_cl = h[i], l[i], c[i]
            cum_pv, cum_v, cum_p2v = 0.0, 0.0, 0.0
        else:
            day_hi = max(day_hi, h[i])
            day_lo = min(day_lo, l[i])
            day_cl = c[i]

        pivot_pp[i] = prev_day_pp
        pivot_r1[i] = prev_day_r1
        pivot_s1[i] = prev_day_s1
        pivot_r2[i] = prev_day_r2
        pivot_s2[i] = prev_day_s2
        if np.isfinite(prev_day_r1):
            htf_resistance[i] = prev_day_r1
            htf_support[i] = prev_day_s1

        cum_pv += tp * vi
        cum_v += vi
        cum_p2v += (tp * tp) * vi
        vw = cum_pv / cum_v if cum_v > 0 else tp
        var = max(0.0, (cum_p2v / cum_v) - vw * vw) if cum_v > 0 else 0.0
        sd = math.sqrt(var)
        vwap_arr[i] = vw
        vwap_upper_1[i] = vw + sd
        vwap_lower_1[i] = vw - sd
        vwap_upper_2[i] = vw + 2.0 * sd
        vwap_lower_2[i] = vw - 2.0 * sd
        if sd > 0 and l[i] <= vwap_lower_1[i] and c[i] > vwap_lower_1[i]:
            vwap_band_bounce_bull[i] = True
        if sd > 0 and h[i] >= vwap_upper_1[i] and c[i] < vwap_upper_1[i]:
            vwap_band_bounce_bear[i] = True

        # Causal swings at j = i - swing_right
        if i >= swing_left + swing_right:
            j = i - swing_right
            win_h = h[j - swing_left : j + swing_right + 1]
            win_l = l[j - swing_left : j + swing_right + 1]

            if h[j] == np.max(win_h) and np.sum(win_h == h[j]) == 1:
                swing_high_confirmed[i] = True
                prev_sh = cur_sh
                cur_sh = float(h[j])
                swing_highs_hist.append((j, cur_sh))
                swing_highs_hist = swing_highs_hist[-12:]
                avwap_pv, avwap_v, avwap_active = 0.0, 0.0, True
                if np.isfinite(prev_sh):
                    if cur_sh > prev_sh:
                        hh[i] = True
                    elif cur_sh < prev_sh:
                        lh[i] = True

                # Cluster into S/R resistance zone (or merge within cluster_tol_atr * a)
                tol = cluster_tol_atr * a
                merged = False
                for z in sr_zones:
                    if abs(cur_sh - z.midpoint) <= tol:
                        z.bottom = min(z.bottom, cur_sh - 0.15 * a)
                        z.top = max(z.top, cur_sh + 0.15 * a)
                        z.touches += 1
                        z.strength += 0.5
                        merged = True
                        break
                if not merged:
                    sr_zones.append(_add_zone("sr", "bear", cur_sh - 0.25 * a, cur_sh + 0.25 * a, i, "SWING_HIGH", strength=1.0, meta={"role": "resistance"}))
                    sr_zones = sr_zones[-max_sr_zones:]

            if l[j] == np.min(win_l) and np.sum(win_l == l[j]) == 1:
                swing_low_confirmed[i] = True
                prev_sl = cur_sl
                cur_sl = float(l[j])
                swing_lows_hist.append((j, cur_sl))
                swing_lows_hist = swing_lows_hist[-12:]
                avwap_pv, avwap_v, avwap_active = 0.0, 0.0, True
                if np.isfinite(prev_sl):
                    if cur_sl > prev_sl:
                        hl[i] = True
                    elif cur_sl < prev_sl:
                        ll[i] = True

                tol = cluster_tol_atr * a
                merged = False
                for z in sr_zones:
                    if abs(cur_sl - z.midpoint) <= tol:
                        z.bottom = min(z.bottom, cur_sl - 0.15 * a)
                        z.top = max(z.top, cur_sl + 0.15 * a)
                        z.touches += 1
                        z.strength += 0.5
                        merged = True
                        break
                if not merged:
                    sr_zones.append(_add_zone("sr", "bull", cur_sl - 0.25 * a, cur_sl + 0.25 * a, i, "SWING_LOW", strength=1.0, meta={"role": "support"}))
                    sr_zones = sr_zones[-max_sr_zones:]

        if avwap_active:
            avwap_pv += tp * vi
            avwap_v += vi
            anchored_vwap_arr[i] = avwap_pv / avwap_v
            above_anchored_vwap[i] = bool(c[i] > anchored_vwap_arr[i])

        last_swing_high[i] = cur_sh
        prev_swing_high[i] = prev_sh
        last_swing_low[i] = cur_sl
        prev_swing_low[i] = prev_sl

        # S/R Zone touch & role reversal (flip) tracking
        for z in sr_zones:
            if i <= z.created_bar:
                continue
            if z.direction == "bear":  # currently resistance
                if c[i] > z.top and c[i - 1] <= z.top:
                    z.direction = "bull"
                    z.meta["role"] = "support_flipped"
                    z.status = "tested"
                    sr_flip_bull[i] = True
                    _add_event("sr_flip_bull", "bull", i, c[i], level=z.midpoint, linked=[z.id])
                elif l[i] <= z.top and h[i] >= z.bottom:
                    at_sr_resistance[i] = True
                    z.touches += 1
            else:  # currently support
                if c[i] < z.bottom and c[i - 1] >= z.bottom:
                    z.direction = "bear"
                    z.meta["role"] = "resistance_flipped"
                    z.status = "tested"
                    sr_flip_bear[i] = True
                    _add_event("sr_flip_bear", "bear", i, c[i], level=z.midpoint, linked=[z.id])
                elif l[i] <= z.top and h[i] >= z.bottom:
                    at_sr_support[i] = True
                    z.touches += 1

        # Round number proximity
        r_step = _round_step_for_price(c[i])
        nearest_round = round(c[i] / r_step) * r_step
        if abs(c[i] - nearest_round) <= 0.25 * a:
            near_round_number[i] = True

        # Rolling causal Volume Profile (recompute every 5 bars or when window just warmed up)
        if i >= 20:
            if i == 20 or i % 5 == 0 or not np.isfinite(last_vp[0]):
                ws = max(0, i - vp_lookback + 1)
                last_vp = _rolling_vp_poc_vah_val(h[ws : i + 1], l[ws : i + 1], c[ws : i + 1], vol[ws : i + 1])
            vp_poc[i], vp_vah[i], vp_val[i] = last_vp
            at_vp_poc[i] = bool(l[i] - 0.1 * a <= vp_poc[i] <= h[i] + 0.1 * a)
            at_vp_vah[i] = bool(l[i] - 0.1 * a <= vp_vah[i] <= h[i] + 0.1 * a)
            at_vp_val[i] = bool(l[i] - 0.1 * a <= vp_val[i] <= h[i] + 0.1 * a)

        # Rolling causal support/resistance & breakouts + breakout-retest
        if i >= sr_lookback:
            resistance[i] = float(np.max(h[i - sr_lookback : i]))
            support[i] = float(np.min(l[i - sr_lookback : i]))
        if i >= sr_lookback + 1:
            if np.isfinite(resistance[i]):
                if c[i] > resistance[i] and c[i - 1] <= resistance[i]:
                    breakout_up[i] = True
                    recent_up_break = (float(resistance[i]), i)
                    _add_event("breakout_up", "bull", i, c[i], level=resistance[i])
                if h[i] > resistance[i] and c[i] < resistance[i]:
                    failed_breakout_up[i] = True
                    _add_event("failed_breakout_up", "bear", i, h[i], level=resistance[i])
            if np.isfinite(support[i]):
                if c[i] < support[i] and c[i - 1] >= support[i]:
                    breakdown_down[i] = True
                    recent_dn_break = (float(support[i]), i)
                    _add_event("breakdown_down", "bear", i, c[i], level=support[i])
                if l[i] < support[i] and c[i] > support[i]:
                    failed_breakdown_down[i] = True
                    _add_event("failed_breakdown_down", "bull", i, l[i], level=support[i])

        if recent_up_break is not None:
            lvl, bbar = recent_up_break
            if 1 <= (i - bbar) <= 12:
                if l[i] <= lvl + 0.3 * a and c[i] > lvl:
                    breakout_retest_bull[i] = True
                    recent_up_break = None
            elif (i - bbar) > 12:
                recent_up_break = None

        if recent_dn_break is not None:
            lvl, bbar = recent_dn_break
            if 1 <= (i - bbar) <= 12:
                if h[i] >= lvl - 0.3 * a and c[i] < lvl:
                    breakout_retest_bear[i] = True
                    recent_dn_break = None
            elif (i - bbar) > 12:
                recent_dn_break = None

        # Causal trendlines & break-retest
        if i >= 1 and np.isfinite(last_swing_low[i]) and np.isfinite(prev_swing_low[i]):
            trendline_slope_bull[i] = (last_swing_low[i] - prev_swing_low[i]) / max(last_swing_low[i], 1e-12)
        if i >= 1 and np.isfinite(last_swing_high[i]) and np.isfinite(prev_swing_high[i]):
            trendline_slope_bear[i] = (last_swing_high[i] - prev_swing_high[i]) / max(last_swing_high[i], 1e-12)

        if len(swing_highs_hist) >= 2:
            (x1, y1), (x2, y2) = swing_highs_hist[-2], swing_highs_hist[-1]
            if x2 > x1 and y2 < y1:  # Descending resistance trendline
                slope = (y2 - y1) / (x2 - x1)
                proj = y2 + slope * (i - x2)
                prev_proj = y2 + slope * ((i - 1) - x2)
                if c[i] > proj and c[i - 1] <= prev_proj:
                    recent_tl_bull_break = (slope, y2 - slope * x2, i)
                elif recent_tl_bull_break is not None:
                    slp, icp, bbar = recent_tl_bull_break
                    if 1 <= (i - bbar) <= 10:
                        tl_val = slp * i + icp
                        if l[i] <= tl_val + 0.3 * a and c[i] > tl_val:
                            trendline_break_retest_bull[i] = True
                            recent_tl_bull_break = None

        if len(swing_lows_hist) >= 2:
            (x1, y1), (x2, y2) = swing_lows_hist[-2], swing_lows_hist[-1]
            if x2 > x1 and y2 > y1:  # Ascending support trendline
                slope = (y2 - y1) / (x2 - x1)
                proj = y2 + slope * (i - x2)
                prev_proj = y2 + slope * ((i - 1) - x2)
                if c[i] < proj and c[i - 1] >= prev_proj:
                    recent_tl_bear_break = (slope, y2 - slope * x2, i)
                elif recent_tl_bear_break is not None:
                    slp, icp, bbar = recent_tl_bear_break
                    if 1 <= (i - bbar) <= 10:
                        tl_val = slp * i + icp
                        if h[i] >= tl_val - 0.3 * a and c[i] < tl_val:
                            trendline_break_retest_bear[i] = True
                            recent_tl_bear_break = None

        # Candlestick Patterns
        if i >= 1:
            inside_bar[i] = bool(h[i] <= h[i - 1] and l[i] >= l[i - 1])
            bullish_engulfing[i] = bool(c[i - 1] < o[i - 1] and c[i] > o[i] and o[i] <= c[i - 1] and c[i] >= o[i - 1])
            bearish_engulfing[i] = bool(c[i - 1] > o[i - 1] and c[i] < o[i] and o[i] >= c[i - 1] and c[i] <= o[i - 1])
            outside_bar_bull[i] = bool(h[i] > h[i - 1] and l[i] < l[i - 1] and c[i] > o[i])
            outside_bar_bear[i] = bool(h[i] > h[i - 1] and l[i] < l[i - 1] and c[i] < o[i])

        doji[i] = bool(body[i] <= 0.12 * rng[i])
        bullish_pin_bar[i] = bool(lower_wick[i] >= 2.0 * body[i] and upper_wick[i] <= 0.35 * rng[i])
        bearish_pin_bar[i] = bool(upper_wick[i] >= 2.0 * body[i] and lower_wick[i] <= 0.35 * rng[i])
        if np.isfinite(atr_vals[i]) and atr_vals[i] > 0:
            range_expansion[i] = bool((h[i] - l[i]) > 1.8 * atr_vals[i] and body[i] > 0.6 * (h[i] - l[i]))

        if i >= 2:
            # Morning / Evening Star
            b0 = abs(c[i - 2] - o[i - 2])
            b1 = abs(c[i - 1] - o[i - 1])
            b2 = abs(c[i] - o[i])
            if c[i - 2] < o[i - 2] and b0 > 0.4 * rng[i - 2] and b1 < 0.35 * b0 and c[i] > o[i] and c[i] > (o[i - 2] + c[i - 2]) * 0.5:
                morning_star[i] = True
            if c[i - 2] > o[i - 2] and b0 > 0.4 * rng[i - 2] and b1 < 0.35 * b0 and c[i] < o[i] and c[i] < (o[i - 2] + c[i - 2]) * 0.5:
                evening_star[i] = True
            # Three White Soldiers / Three Black Crows
            if (
                c[i - 2] > o[i - 2] and c[i - 1] > o[i - 1] and c[i] > o[i]
                and c[i] > c[i - 1] > c[i - 2]
                and b2 > 0.4 * a and b1 > 0.4 * a
            ):
                three_white_soldiers[i] = True
            if (
                c[i - 2] < o[i - 2] and c[i - 1] < o[i - 1] and c[i] < o[i]
                and c[i] < c[i - 1] < c[i - 2]
                and b2 > 0.4 * a and b1 > 0.4 * a
            ):
                three_black_crows[i] = True

        # Causal Chart Patterns
        if len(swing_highs_hist) >= 2 and len(swing_lows_hist) >= 1:
            (hx1, hy1), (hx2, hy2) = swing_highs_hist[-2], swing_highs_hist[-1]
            # Double Top: two swing highs within 0.6*ATR + breakdown below intermediate swing low
            if abs(hy1 - hy2) <= 0.6 * a:
                mid_lows = [ly for (lx, ly) in swing_lows_hist if hx1 < lx < hx2]
                if mid_lows:
                    neck = min(mid_lows)
                    if c[i] < neck and c[i - 1] >= neck:
                        double_top[i] = True
                        _add_event("double_top", "bear", i, c[i], level=neck)

        if len(swing_lows_hist) >= 2 and len(swing_highs_hist) >= 1:
            (lx1, ly1), (lx2, ly2) = swing_lows_hist[-2], swing_lows_hist[-1]
            if abs(ly1 - ly2) <= 0.6 * a:
                mid_highs = [hy for (hx, hy) in swing_highs_hist if lx1 < hx < lx2]
                if mid_highs:
                    neck = max(mid_highs)
                    if c[i] > neck and c[i - 1] <= neck:
                        double_bottom[i] = True
                        _add_event("double_bottom", "bull", i, c[i], level=neck)

        # Head & Shoulders / Inverse H&S
        if len(swing_highs_hist) >= 3 and len(swing_lows_hist) >= 2:
            (_, ls), (hx_h, head), (_, rs) = swing_highs_hist[-3], swing_highs_hist[-2], swing_highs_hist[-1]
            if head > ls + 0.3 * a and head > rs + 0.3 * a and abs(ls - rs) <= 0.9 * a:
                neck = min(swing_lows_hist[-1][1], swing_lows_hist[-2][1])
                if c[i] < neck and c[i - 1] >= neck:
                    head_and_shoulders[i] = True
                    _add_event("head_and_shoulders", "bear", i, c[i], level=neck)

        if len(swing_lows_hist) >= 3 and len(swing_highs_hist) >= 2:
            (_, ls), (_, head), (_, rs) = swing_lows_hist[-3], swing_lows_hist[-2], swing_lows_hist[-1]
            if head < ls - 0.3 * a and head < rs - 0.3 * a and abs(ls - rs) <= 0.9 * a:
                neck = max(swing_highs_hist[-1][1], swing_highs_hist[-2][1])
                if c[i] > neck and c[i - 1] <= neck:
                    inverse_head_and_shoulders[i] = True
                    _add_event("inverse_head_and_shoulders", "bull", i, c[i], level=neck)

        # Symmetrical Triangle & Rising/Falling Wedge
        if len(swing_highs_hist) >= 2 and len(swing_lows_hist) >= 2:
            (_, h1), (_, h2) = swing_highs_hist[-2], swing_highs_hist[-1]
            (_, l1), (_, l2) = swing_lows_hist[-2], swing_lows_hist[-1]
            if h2 < h1 and l2 > l1:  # converging triangle
                if c[i] > h2 and c[i - 1] <= h2:
                    triangle_breakout_up[i] = True
                elif c[i] < l2 and c[i - 1] >= l2:
                    triangle_breakout_down[i] = True
            elif h2 > h1 and l2 > l1 and (l2 - l1) > (h2 - h1):  # rising wedge
                if c[i] < l2 and c[i - 1] >= l2:
                    rising_wedge_breakdown[i] = True
            elif h2 < h1 and l2 < l1 and (h1 - h2) > (l1 - l2):  # falling wedge
                if c[i] > h2 and c[i - 1] <= h2:
                    falling_wedge_breakout[i] = True

        # Bull / Bear Flag Breakout (impulse pole followed by tight pause & break)
        if i >= 15:
            pole_ret = c[i - 6] - c[i - 15]
            cons_hi = float(np.max(h[i - 6 : i]))
            cons_lo = float(np.min(l[i - 6 : i]))
            if pole_ret > 2.5 * a and (cons_hi - cons_lo) < 1.8 * a and c[i] > cons_hi:
                bull_flag_breakout[i] = True
            if pole_ret < -2.5 * a and (cons_hi - cons_lo) < 1.8 * a and c[i] < cons_lo:
                bear_flag_breakout[i] = True

    arrays: dict[str, np.ndarray] = {
        "swing_high_confirmed": swing_high_confirmed,
        "swing_low_confirmed": swing_low_confirmed,
        "last_swing_high": last_swing_high,
        "last_swing_low": last_swing_low,
        "higher_high": hh,
        "higher_low": hl,
        "lower_high": lh,
        "lower_low": ll,
        "resistance": resistance,
        "support": support,
        "at_sr_support": at_sr_support,
        "at_sr_resistance": at_sr_resistance,
        "sr_flip_bull": sr_flip_bull,
        "sr_flip_bear": sr_flip_bear,
        "near_round_number": near_round_number,
        "pivot_pp": pivot_pp,
        "pivot_r1": pivot_r1,
        "pivot_s1": pivot_s1,
        "pivot_r2": pivot_r2,
        "pivot_s2": pivot_s2,
        "vwap": vwap_arr,
        "vwap_upper_1": vwap_upper_1,
        "vwap_lower_1": vwap_lower_1,
        "vwap_upper_2": vwap_upper_2,
        "vwap_lower_2": vwap_lower_2,
        "anchored_vwap": anchored_vwap_arr,
        "above_anchored_vwap": above_anchored_vwap,
        "vwap_band_bounce_bull": vwap_band_bounce_bull,
        "vwap_band_bounce_bear": vwap_band_bounce_bear,
        "vp_poc": vp_poc,
        "vp_vah": vp_vah,
        "vp_val": vp_val,
        "at_vp_poc": at_vp_poc,
        "at_vp_vah": at_vp_vah,
        "at_vp_val": at_vp_val,
        "htf_resistance": htf_resistance,
        "htf_support": htf_support,
        "breakout_up": breakout_up,
        "breakdown_down": breakdown_down,
        "failed_breakout_up": failed_breakout_up,
        "failed_breakdown_down": failed_breakdown_down,
        "breakout_retest_bull": breakout_retest_bull,
        "breakout_retest_bear": breakout_retest_bear,
        "trendline_slope_bull": trendline_slope_bull,
        "trendline_slope_bear": trendline_slope_bear,
        "trendline_break_retest_bull": trendline_break_retest_bull,
        "trendline_break_retest_bear": trendline_break_retest_bear,
        "inside_bar": inside_bar,
        "bullish_engulfing": bullish_engulfing,
        "bearish_engulfing": bearish_engulfing,
        "bullish_pin_bar": bullish_pin_bar,
        "bearish_pin_bar": bearish_pin_bar,
        "range_expansion": range_expansion,
        "doji": doji,
        "morning_star": morning_star,
        "evening_star": evening_star,
        "three_white_soldiers": three_white_soldiers,
        "three_black_crows": three_black_crows,
        "outside_bar_bull": outside_bar_bull,
        "outside_bar_bear": outside_bar_bear,
        "double_top": double_top,
        "double_bottom": double_bottom,
        "head_and_shoulders": head_and_shoulders,
        "inverse_head_and_shoulders": inverse_head_and_shoulders,
        "triangle_breakout_up": triangle_breakout_up,
        "triangle_breakout_down": triangle_breakout_down,
        "bull_flag_breakout": bull_flag_breakout,
        "bear_flag_breakout": bear_flag_breakout,
        "rising_wedge_breakdown": rising_wedge_breakdown,
        "falling_wedge_breakout": falling_wedge_breakout,
    }
    levels = {
        "sr_zones": [z.to_dict() for z in sr_zones],
        "vp_poc": float(vp_poc[-1]) if n and np.isfinite(vp_poc[-1]) else None,
        "vp_vah": float(vp_vah[-1]) if n and np.isfinite(vp_vah[-1]) else None,
        "vp_val": float(vp_val[-1]) if n and np.isfinite(vp_val[-1]) else None,
        "pivot_pp": float(pivot_pp[-1]) if n and np.isfinite(pivot_pp[-1]) else None,
    }
    return DetectionResult(arrays=arrays, zones=all_zones, events=all_events, levels=levels)


def compute_price_action_features(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_left: int = 3,
    swing_right: int = 3,
    sr_lookback: int = 50,
    atr_period: int = 14,
    open_time_ms: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    cluster_tol_atr: float = 0.6,
) -> dict[str, np.ndarray]:
    """Boolean & numeric array wrapper derived strictly from `detect_price_action(...)`."""
    return detect_price_action(
        open_=open_,
        high=high,
        low=low,
        close=close,
        open_time_ms=open_time_ms,
        volume=volume,
        swing_left=swing_left,
        swing_right=swing_right,
        sr_lookback=sr_lookback,
        atr_period=atr_period,
        cluster_tol_atr=cluster_tol_atr,
    ).arrays
