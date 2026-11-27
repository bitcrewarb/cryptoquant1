"""
ICT (Inner Circle Trader) structured causal engine (zero look-ahead).

Implements:
- Configurable timezone & half-hour precision Kill Zones (Asian 20:00-00:00, London 02:00-05:00,
  NY AM 08:30-11:00, NY PM 13:30-16:00 NY).
- Silver Bullet windows (03:00-04:00, 10:00-11:00, 14:00-15:00 NY) + FVG after sweep.
- True Day Open (00:00 NY), Midnight Open, 08:30 NY Open, NWOG, NDOG, Weekly Range.
- 20/40/60-day IPDA lookback ranges & raids.
- OTE ([0.62, 0.79]) computed from the actual causal impulse leg rather than static dealing range bounds.
- ICT Macro windows (XX:50-XX:10 & institutional execution macros).
- Draw-on-Liquidity (DOL) nearest unswept opposing pool selection.
- Real timestamp-aligned SMT Divergence (primary vs reference symbol such as BTC/ETH).
- Composite Daily Bias combining weekly/daily structure, premium/discount, and liquidity draw.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from crypto_quant_lab.core.structures import DetectionResult, EventObject, ZoneObject
from crypto_quant_lab.indicators.volatility import atr
from crypto_quant_lab.smc.engine import detect_smc


TimeSpec = int | float | tuple[int, int]


def _to_minutes(spec: TimeSpec) -> int:
    if isinstance(spec, tuple):
        return int(spec[0]) * 60 + int(spec[1])
    if isinstance(spec, float):
        hr = int(spec)
        mn = int(round((spec - hr) * 60))
        return hr * 60 + mn
    return int(spec) * 60


def _in_minute_window(minute_of_day: int, start_spec: TimeSpec, end_spec: TimeSpec) -> bool:
    s = _to_minutes(start_spec) % 1440
    e = _to_minutes(end_spec) % 1440
    if s < e:
        return s <= minute_of_day < e
    if s > e:
        return minute_of_day >= s or minute_of_day < e
    return True


@dataclass(frozen=True)
class ICTSessionConfig:
    tz_name: str = "America/New_York"
    asian_hours: tuple[TimeSpec, TimeSpec] = ((20, 0), (0, 0))
    london_kz_hours: tuple[TimeSpec, TimeSpec] = ((2, 0), (5, 0))
    ny_am_kz_hours: tuple[TimeSpec, TimeSpec] = ((8, 30), (11, 0))
    ny_pm_kz_hours: tuple[TimeSpec, TimeSpec] = ((13, 30), (16, 0))
    macro_windows: tuple[tuple[TimeSpec, TimeSpec], ...] = (
        ((2, 33), (3, 0)),
        ((4, 3), (4, 30)),
        ((8, 50), (9, 10)),
        ((9, 50), (10, 10)),
        ((10, 50), (11, 10)),
        ((11, 50), (12, 10)),
        ((13, 10), (13, 40)),
        ((15, 15), (15, 45)),
    )
    silver_bullet_windows: tuple[tuple[TimeSpec, TimeSpec], ...] = (
        ((3, 0), (4, 0)),
        ((10, 0), (11, 0)),
        ((14, 0), (15, 0)),
    )


def align_reference_ohlcv(
    primary_open_time_ms: np.ndarray,
    ref_open_time_ms: np.ndarray,
    ref_high: np.ndarray,
    ref_low: np.ndarray,
    ref_close: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Strictly causal backward timestamp alignment (`ref_time <= primary_time`)
    so SMT divergence between symbols (e.g. BTC vs ETH) never leaks future bars or misaligns gaps.
    """
    p_t = np.asarray(primary_open_time_ms, dtype=np.int64)
    r_t = np.asarray(ref_open_time_ms, dtype=np.int64)
    r_h = np.asarray(ref_high, dtype=np.float64)
    r_l = np.asarray(ref_low, dtype=np.float64)
    r_c = np.asarray(ref_close, dtype=np.float64) if ref_close is not None else None

    order = np.argsort(r_t)
    r_t = r_t[order]
    r_h = r_h[order]
    r_l = r_l[order]
    if r_c is not None:
        r_c = r_c[order]

    n = len(p_t)
    out_h = np.full(n, np.nan, dtype=np.float64)
    out_l = np.full(n, np.nan, dtype=np.float64)
    out_c = np.full(n, np.nan, dtype=np.float64) if r_c is not None else None

    if len(r_t) == 0:
        return out_h, out_l, out_c

    idx = np.searchsorted(r_t, p_t, side="right") - 1
    valid = idx >= 0
    out_h[valid] = r_h[idx[valid]]
    out_l[valid] = r_l[idx[valid]]
    if out_c is not None and r_c is not None:
        out_c[valid] = r_c[idx[valid]]
    return out_h, out_l, out_c


def detect_ict(
    open_time_ms: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    reference_high: np.ndarray | None = None,
    reference_low: np.ndarray | None = None,
    reference_open_time_ms: np.ndarray | None = None,
    session_cfg: ICTSessionConfig | None = None,
    min_fvg_atr_mult: float = 0.2,
    max_zone_age_bars: int = 100,
    trigger_on_retest: bool = False,
    swing_len: int = 5,
) -> DetectionResult:
    cfg = session_cfg or ICTSessionConfig()
    tz = ZoneInfo(cfg.tz_name)
    times = np.asarray(open_time_ms, dtype=np.int64)
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    n = len(c)

    if reference_open_time_ms is not None and reference_high is not None and reference_low is not None:
        ref_h, ref_l, _ = align_reference_ohlcv(times, reference_open_time_ms, reference_high, reference_low)
    else:
        ref_h = np.asarray(reference_high, dtype=np.float64) if reference_high is not None and len(reference_high) == n else None
        ref_l = np.asarray(reference_low, dtype=np.float64) if reference_low is not None and len(reference_low) == n else None

    atr_vals = atr(h, l, c, period=14)
    smc_res = detect_smc(
        o,
        h,
        l,
        c,
        open_time_ms=times,
        ext_swing=swing_len,
        min_fvg_atr_mult=min_fvg_atr_mult,
        max_zone_age_bars=max_zone_age_bars,
        trigger_on_retest=trigger_on_retest,
    )
    smc = smc_res.arrays

    london_kz = np.zeros(n, dtype=bool)
    ny_am_kz = np.zeros(n, dtype=bool)
    ny_pm_kz = np.zeros(n, dtype=bool)
    asian_session = np.zeros(n, dtype=bool)
    in_macro_window = np.zeros(n, dtype=bool)
    silver_bullet_window = np.zeros(n, dtype=bool)
    silver_bullet_bull = np.zeros(n, dtype=bool)
    silver_bullet_bear = np.zeros(n, dtype=bool)

    pdh = np.full(n, np.nan, dtype=np.float64)
    pdl = np.full(n, np.nan, dtype=np.float64)
    pwh = np.full(n, np.nan, dtype=np.float64)
    pwl = np.full(n, np.nan, dtype=np.float64)
    weekly_high_arr = np.full(n, np.nan, dtype=np.float64)
    weekly_low_arr = np.full(n, np.nan, dtype=np.float64)
    weekly_eq_arr = np.full(n, np.nan, dtype=np.float64)
    true_day_open_arr = np.full(n, np.nan, dtype=np.float64)
    midnight_open_arr = np.full(n, np.nan, dtype=np.float64)
    ny_0830_open_arr = np.full(n, np.nan, dtype=np.float64)

    ipda_20_high = np.full(n, np.nan, dtype=np.float64)
    ipda_20_low = np.full(n, np.nan, dtype=np.float64)
    ipda_40_high = np.full(n, np.nan, dtype=np.float64)
    ipda_40_low = np.full(n, np.nan, dtype=np.float64)
    ipda_60_high = np.full(n, np.nan, dtype=np.float64)
    ipda_60_low = np.full(n, np.nan, dtype=np.float64)

    dol_bull_target = np.full(n, np.nan, dtype=np.float64)
    dol_bear_target = np.full(n, np.nan, dtype=np.float64)

    daily_bias = np.zeros(n, dtype=np.int8)
    weekly_bias = np.zeros(n, dtype=np.int8)
    daily_bias_score = np.zeros(n, dtype=np.float64)

    ote_long = np.zeros(n, dtype=bool)
    ote_short = np.zeros(n, dtype=bool)
    liquidity_raid_bull = np.zeros(n, dtype=bool)
    liquidity_raid_bear = np.zeros(n, dtype=bool)
    judas_swing_bull = np.zeros(n, dtype=bool)
    judas_swing_bear = np.zeros(n, dtype=bool)
    po3_expansion_bull = np.zeros(n, dtype=bool)
    po3_expansion_bear = np.zeros(n, dtype=bool)
    opening_range_break_up = np.zeros(n, dtype=bool)
    opening_range_break_down = np.zeros(n, dtype=bool)
    consequent_encroachment_touch = np.zeros(n, dtype=bool)
    smt_divergence_bull = np.zeros(n, dtype=bool)
    smt_divergence_bear = np.zeros(n, dtype=bool)
    in_nwog = np.zeros(n, dtype=bool)
    in_ndog = np.zeros(n, dtype=bool)

    all_zones: list[ZoneObject] = list(smc_res.zones)
    all_events: list[EventObject] = list(smc_res.events)
    z_seq = 0
    ev_seq = 0

    def _add_zone(ztype: str, direction: str, lo_v: float, hi_v: float, bar_i: int, src: str, expiry_bars: int = max_zone_age_bars, meta: dict[str, Any] | None = None) -> ZoneObject:
        nonlocal z_seq
        z_seq += 1
        ts = int(times[bar_i])
        z = ZoneObject(
            id=f"ict_{ztype}_{direction}_{z_seq}",
            type=ztype,
            direction=direction,
            top=max(float(lo_v), float(hi_v)),
            bottom=min(float(lo_v), float(hi_v)),
            created_bar=bar_i,
            created_time=ts,
            confirmed_time=ts,
            status="fresh",
            expiry=bar_i + expiry_bars,
            source_event=src,
            strength=1.2,
            meta=dict(meta or {}),
        )
        all_zones.append(z)
        return z

    def _add_event(etype: str, direction: str, bar_i: int, px: float, level: float | None = None, linked: list[str] | None = None) -> EventObject:
        nonlocal ev_seq
        ev_seq += 1
        ev = EventObject(
            id=f"ict_ev_{etype}_{ev_seq}",
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

    cur_day = None
    day_high, day_low, day_open, day_close = -np.inf, np.inf, np.nan, np.nan
    prev_day_high, prev_day_low, prev_day_open, prev_day_close = np.nan, np.nan, np.nan, np.nan
    completed_days_h: deque[float] = deque(maxlen=60)
    completed_days_l: deque[float] = deque(maxlen=60)

    cur_week = None
    wk_high, wk_low, wk_open, wk_close = -np.inf, np.inf, np.nan, np.nan
    prev_wk_high, prev_wk_low, prev_wk_open, prev_wk_close = np.nan, np.nan, np.nan, np.nan

    in_asian_prev = False
    forming_asian_high, forming_asian_low = np.nan, np.nan
    completed_asian_high, completed_asian_low = np.nan, np.nan
    judas_bull_fired_today = False
    judas_bear_fired_today = False
    swept_asian_low_bar = -1
    swept_asian_high_bar = -1
    po3_bull_fired_today = False
    po3_bear_fired_today = False

    ny_open_range_high, ny_open_range_low = np.nan, np.nan
    cur_day_open_price = np.nan
    cur_midnight_open = np.nan
    cur_0830_open = np.nan

    active_ce_zones: list[ZoneObject] = []
    active_gaps: list[ZoneObject] = []

    # Causal impulse-leg trackers for true [0.62, 0.79] OTE
    bull_leg_lo, bull_leg_hi = np.nan, np.nan
    bear_leg_hi, bear_leg_lo = np.nan, np.nan
    recent_ext_sl, recent_ext_sh = np.nan, np.nan
    last_sweep_bull_bar = -100
    last_sweep_bear_bar = -100
    recent_day_raid_dir = 0

    for i in range(n):
        ts = int(times[i])
        dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).astimezone(tz)
        d_key = dt.date()
        iso_yr, iso_wk, _ = dt.isocalendar()
        w_key = (iso_yr, iso_wk)
        hr = dt.hour
        minute_of_day = hr * 60 + dt.minute
        a = atr_vals[i] if np.isfinite(atr_vals[i]) and atr_vals[i] > 0 else max(h[i] - l[i], 1e-9)

        if cur_day != d_key:
            if cur_day is not None:
                prev_day_high, prev_day_low = day_high, day_low
                prev_day_open, prev_day_close = day_open, day_close
                completed_days_h.append(float(day_high))
                completed_days_l.append(float(day_low))
                # NDOG: prior day close -> midnight open
                if abs(o[i] - prev_day_close) >= 0.05 * a:
                    gz = _add_zone(
                        "ndog",
                        "bull" if o[i] > prev_day_close else "bear",
                        min(o[i], prev_day_close),
                        max(o[i], prev_day_close),
                        i,
                        "NDOG",
                        expiry_bars=48,
                    )
                    active_gaps.append(gz)
                    active_gaps = active_gaps[-8:]
            cur_day = d_key
            day_high, day_low = h[i], l[i]
            day_open, day_close = o[i], c[i]
            cur_day_open_price = o[i]
            cur_midnight_open = o[i]
            cur_0830_open = np.nan
            ny_open_range_high, ny_open_range_low = np.nan, np.nan
            judas_bull_fired_today = False
            judas_bear_fired_today = False
            swept_asian_low_bar = -1
            swept_asian_high_bar = -1
            po3_bull_fired_today = False
            po3_bear_fired_today = False
            recent_day_raid_dir = 0
        else:
            day_high = max(day_high, h[i])
            day_low = min(day_low, l[i])
            day_close = c[i]

        if cur_week != w_key:
            if cur_week is not None:
                prev_wk_high, prev_wk_low = wk_high, wk_low
                prev_wk_open, prev_wk_close = wk_open, wk_close
                # NWOG: prior week close -> new week open
                if abs(o[i] - prev_wk_close) >= 0.05 * a:
                    gz = _add_zone(
                        "nwog",
                        "bull" if o[i] > prev_wk_close else "bear",
                        min(o[i], prev_wk_close),
                        max(o[i], prev_wk_close),
                        i,
                        "NWOG",
                        expiry_bars=120,
                    )
                    active_gaps.append(gz)
                    active_gaps = active_gaps[-8:]
            cur_week = w_key
            wk_high, wk_low = h[i], l[i]
            wk_open, wk_close = o[i], c[i]
        else:
            wk_high = max(wk_high, h[i])
            wk_low = min(wk_low, l[i])
            wk_close = c[i]

        if not np.isfinite(cur_0830_open) and minute_of_day >= 8 * 60 + 30:
            cur_0830_open = o[i]

        pdh[i] = prev_day_high
        pdl[i] = prev_day_low
        pwh[i] = prev_wk_high
        pwl[i] = prev_wk_low
        weekly_high_arr[i] = wk_high
        weekly_low_arr[i] = wk_low
        weekly_eq_arr[i] = (wk_high + wk_low) * 0.5
        true_day_open_arr[i] = cur_day_open_price
        midnight_open_arr[i] = cur_midnight_open
        ny_0830_open_arr[i] = cur_0830_open

        # 20 / 40 / 60-day IPDA lookbacks from completed days
        dh_list = list(completed_days_h)
        dl_list = list(completed_days_l)
        if dh_list:
            ipda_20_high[i] = max(dh_list[-20:])
            ipda_20_low[i] = min(dl_list[-20:])
            ipda_40_high[i] = max(dh_list[-40:])
            ipda_40_low[i] = min(dl_list[-40:])
            ipda_60_high[i] = max(dh_list[-60:])
            ipda_60_low[i] = min(dl_list[-60:])

        if np.isfinite(prev_wk_open) and np.isfinite(prev_wk_close):
            if prev_wk_close > prev_wk_open:
                weekly_bias[i] = 1
            elif prev_wk_close < prev_wk_open:
                weekly_bias[i] = -1

        # Active NWOG/NDOG touch tracking
        surv_gaps: list[ZoneObject] = []
        for gz in active_gaps:
            if i > (gz.expiry or (gz.created_bar + max_zone_age_bars)):
                gz.status = "expired"
                continue
            if i > gz.created_bar and l[i] <= gz.top and h[i] >= gz.bottom:
                if gz.type == "nwog":
                    in_nwog[i] = True
                else:
                    in_ndog[i] = True
                gz.touches += 1
            surv_gaps.append(gz)
        active_gaps = surv_gaps

        # Session & half-hour precision Kill Zone flags + Asian range carry-forward
        in_asian_now = _in_minute_window(minute_of_day, *cfg.asian_hours)
        if in_asian_now:
            asian_session[i] = True
            if not in_asian_prev:
                completed_asian_high, completed_asian_low = np.nan, np.nan
                forming_asian_high, forming_asian_low = h[i], l[i]
            else:
                forming_asian_high = h[i] if not np.isfinite(forming_asian_high) else max(forming_asian_high, h[i])
                forming_asian_low = l[i] if not np.isfinite(forming_asian_low) else min(forming_asian_low, l[i])
        else:
            if in_asian_prev and np.isfinite(forming_asian_high) and np.isfinite(forming_asian_low):
                completed_asian_high = forming_asian_high
                completed_asian_low = forming_asian_low
                forming_asian_high, forming_asian_low = np.nan, np.nan
        in_asian_prev = in_asian_now

        if _in_minute_window(minute_of_day, *cfg.london_kz_hours):
            london_kz[i] = True
        if _in_minute_window(minute_of_day, *cfg.ny_am_kz_hours):
            ny_am_kz[i] = True
        if _in_minute_window(minute_of_day, *cfg.ny_pm_kz_hours):
            ny_pm_kz[i] = True

        for ms_t, me_t in cfg.macro_windows:
            if _in_minute_window(minute_of_day, ms_t, me_t):
                in_macro_window[i] = True
                break
        for sb_s, sb_e in cfg.silver_bullet_windows:
            if _in_minute_window(minute_of_day, sb_s, sb_e):
                silver_bullet_window[i] = True
                break

        if hr == 9:
            ny_open_range_high = h[i] if not np.isfinite(ny_open_range_high) else max(ny_open_range_high, h[i])
            ny_open_range_low = l[i] if not np.isfinite(ny_open_range_low) else min(ny_open_range_low, l[i])
        elif hr > 9 and np.isfinite(ny_open_range_high) and np.isfinite(ny_open_range_low):
            if c[i] > ny_open_range_high:
                opening_range_break_up[i] = True
            elif c[i] < ny_open_range_low:
                opening_range_break_down[i] = True

        # Liquidity Raid on PDH/PDL
        if np.isfinite(prev_day_low) and l[i] < prev_day_low and c[i] > prev_day_low:
            liquidity_raid_bull[i] = True
            last_sweep_bull_bar = i
            recent_day_raid_dir = 1
            _add_event("liquidity_raid_bull", "bull", i, l[i], level=prev_day_low)
        if np.isfinite(prev_day_high) and h[i] > prev_day_high and c[i] < prev_day_high:
            liquidity_raid_bear[i] = True
            last_sweep_bear_bar = i
            recent_day_raid_dir = -1
            _add_event("liquidity_raid_bear", "bear", i, h[i], level=prev_day_high)

        if smc["ssl_sweep"][i]:
            last_sweep_bull_bar = i
        if smc["bsl_sweep"][i]:
            last_sweep_bear_bar = i

        # Composite Daily Bias: Weekly structure + Prior-Day P/D positioning + Liquidity draw + Structure BOS
        bias_pts = 0.0
        if weekly_bias[i] != 0:
            bias_pts += 0.35 * float(weekly_bias[i])
        if np.isfinite(prev_day_high) and np.isfinite(prev_day_low) and prev_day_high > prev_day_low and np.isfinite(prev_day_close):
            pd_mid = (prev_day_high + prev_day_low) * 0.5
            # Close near upper half with upward structure vs discount recovery
            if prev_day_close > prev_day_open:
                bias_pts += 0.25
            elif prev_day_close < prev_day_open:
                bias_pts -= 0.25
            bias_pts += 0.20 if c[i] < pd_mid else -0.20
        if recent_day_raid_dir != 0:
            bias_pts += 0.40 * float(recent_day_raid_dir)
        elif smc["dealing_range_pos"][i] < 0.45:
            bias_pts += 0.15
        elif smc["dealing_range_pos"][i] > 0.55:
            bias_pts -= 0.15

        daily_bias_score[i] = round(bias_pts, 4)
        if bias_pts > 0.1:
            daily_bias[i] = 1
        elif bias_pts < -0.1:
            daily_bias[i] = -1

        # Judas Swing
        if (london_kz[i] or ny_am_kz[i]) and np.isfinite(completed_asian_low) and np.isfinite(completed_asian_high):
            if l[i] < completed_asian_low:
                swept_asian_low_bar = i
            if h[i] > completed_asian_high:
                swept_asian_high_bar = i

            bull_disp = bool(smc["displacement_bull"][i]) or ((c[i] > o[i]) and (c[i] - o[i]) >= 0.5 * a)
            bear_disp = bool(smc["displacement_bear"][i]) or ((c[i] < o[i]) and (o[i] - c[i]) >= 0.5 * a)

            if (
                not judas_bull_fired_today
                and swept_asian_low_bar >= 0
                and (i - swept_asian_low_bar) <= 4
                and c[i] > completed_asian_low
                and (c[i] > cur_day_open_price or bull_disp)
            ):
                judas_swing_bull[i] = True
                judas_bull_fired_today = True
                _add_event("judas_swing_bull", "bull", i, c[i], level=completed_asian_low)

            if (
                not judas_bear_fired_today
                and swept_asian_high_bar >= 0
                and (i - swept_asian_high_bar) <= 4
                and c[i] < completed_asian_high
                and (c[i] < cur_day_open_price or bear_disp)
            ):
                judas_swing_bear[i] = True
                judas_bear_fired_today = True
                _add_event("judas_swing_bear", "bear", i, c[i], level=completed_asian_high)

        # Power of Three expansion
        if not in_asian_now and np.isfinite(completed_asian_high) and np.isfinite(completed_asian_low):
            if not po3_bull_fired_today and smc["displacement_bull"][i] and c[i] > completed_asian_high:
                po3_expansion_bull[i] = True
                po3_bull_fired_today = True
                _add_event("po3_expansion_bull", "bull", i, c[i], level=completed_asian_high)
            if not po3_bear_fired_today and smc["displacement_bear"][i] and c[i] < completed_asian_low:
                po3_expansion_bear[i] = True
                po3_bear_fired_today = True
                _add_event("po3_expansion_bear", "bear", i, c[i], level=completed_asian_low)

        # Silver Bullet setup: FVG formed inside Silver Bullet window within 8 bars of a liquidity sweep
        if silver_bullet_window[i]:
            if smc["fvg_bull"][i] and (i - last_sweep_bull_bar) <= 8:
                silver_bullet_bull[i] = True
                _add_event("silver_bullet_bull", "bull", i, c[i])
            if smc["fvg_bear"][i] and (i - last_sweep_bear_bar) <= 8:
                silver_bullet_bear[i] = True
                _add_event("silver_bullet_bear", "bear", i, c[i])

        # Track confirmed causal impulse legs for true OTE [0.62, 0.79]
        je = i - swing_len
        if je >= swing_len:
            wh_e = h[je - swing_len : je + swing_len + 1]
            wl_e = l[je - swing_len : je + swing_len + 1]
            if h[je] == np.max(wh_e) and np.sum(wh_e == h[je]) == 1:
                recent_ext_sh = float(h[je])
                if np.isfinite(recent_ext_sl) and recent_ext_sh > recent_ext_sl:
                    bull_leg_lo, bull_leg_hi = recent_ext_sl, recent_ext_sh
            if l[je] == np.min(wl_e) and np.sum(wl_e == l[je]) == 1:
                recent_ext_sl = float(l[je])
                if np.isfinite(recent_ext_sh) and recent_ext_sh > recent_ext_sl:
                    bear_leg_hi, bear_leg_lo = recent_ext_sh, recent_ext_sl

        # Fallback to dealing range if no full impulse leg yet
        if np.isfinite(bull_leg_hi) and np.isfinite(bull_leg_lo) and bull_leg_hi > bull_leg_lo:
            retrace_bull = (bull_leg_hi - l[i]) / (bull_leg_hi - bull_leg_lo)
            if 0.62 <= retrace_bull <= 0.79 and c[i] >= bull_leg_lo and daily_bias[i] >= 0:
                ote_long[i] = True
        else:
            dr = smc["dealing_range_pos"][i]
            if 0.214 <= dr <= 0.382 and daily_bias[i] >= 0:
                ote_long[i] = True

        if np.isfinite(bear_leg_hi) and np.isfinite(bear_leg_lo) and bear_leg_hi > bear_leg_lo:
            retrace_bear = (h[i] - bear_leg_lo) / (bear_leg_hi - bear_leg_lo)
            if 0.62 <= retrace_bear <= 0.79 and c[i] <= bear_leg_hi and daily_bias[i] <= 0:
                ote_short[i] = True
        else:
            dr = smc["dealing_range_pos"][i]
            if 0.618 <= dr <= 0.786 and daily_bias[i] <= 0:
                ote_short[i] = True

        # Draw-on-Liquidity (DOL) nearest unswept opposing liquidity level above / below close
        upside_cands = [lvl for lvl in (prev_day_high, prev_wk_high, completed_asian_high, ipda_20_high[i]) if np.isfinite(lvl) and lvl > c[i]]
        downside_cands = [lvl for lvl in (prev_day_low, prev_wk_low, completed_asian_low, ipda_20_low[i]) if np.isfinite(lvl) and lvl < c[i]]
        if upside_cands:
            dol_bull_target[i] = min(upside_cands)
        if downside_cands:
            dol_bear_target[i] = max(downside_cands)

        # FVG Consequent Encroachment (50% midpoint) lifecycle
        surviving_ce: list[ZoneObject] = []
        for z in active_ce_zones:
            if (i - z.created_bar) > max_zone_age_bars or z.status == "mitigated":
                continue
            if z.direction == "bull" and c[i] < z.bottom:
                z.status = "invalidated"
                continue
            if z.direction == "bear" and c[i] > z.top:
                z.status = "invalidated"
                continue
            ce_lvl = z.midpoint
            if i > z.created_bar and l[i] <= ce_lvl <= h[i]:
                if z.status == "fresh" or (trigger_on_retest and z.status == "tested"):
                    consequent_encroachment_touch[i] = True
                z.status = "mitigated"
                z.mitigated_time = ts
            else:
                surviving_ce.append(z)
        active_ce_zones = surviving_ce

        if i >= 2 and smc["fvg_bull"][i]:
            active_ce_zones.append(_add_zone("fvg_ce", "bull", float(h[i - 2]), float(l[i]), i, "FVG_CE_BULL"))
            active_ce_zones = active_ce_zones[-8:]
        if i >= 2 and smc["fvg_bear"][i]:
            active_ce_zones.append(_add_zone("fvg_ce", "bear", float(h[i]), float(l[i - 2]), i, "FVG_CE_BEAR"))
            active_ce_zones = active_ce_zones[-8:]

        # SMT Divergence against timestamp-aligned reference asset
        if ref_h is not None and ref_l is not None and i >= 20:
            win_rl = ref_l[i - 20 : i]
            win_rh = ref_h[i - 20 : i]
            if np.isfinite(ref_l[i]) and np.any(np.isfinite(win_rl)):
                prev_l = np.min(l[i - 20 : i])
                prev_rl = float(np.nanmin(win_rl))
                if (l[i] < prev_l and ref_l[i] >= prev_rl) or (l[i] >= prev_l and ref_l[i] < prev_rl):
                    smt_divergence_bull[i] = True
            if np.isfinite(ref_h[i]) and np.any(np.isfinite(win_rh)):
                prev_h = np.max(h[i - 20 : i])
                prev_rh = float(np.nanmax(win_rh))
                if (h[i] > prev_h and ref_h[i] <= prev_rh) or (h[i] <= prev_h and ref_h[i] > prev_rh):
                    smt_divergence_bear[i] = True

    arrays: dict[str, np.ndarray] = {
        "fvg_bull": smc["fvg_bull"],
        "fvg_bear": smc["fvg_bear"],
        "london_kill_zone": london_kz,
        "ny_am_kill_zone": ny_am_kz,
        "ny_pm_kill_zone": ny_pm_kz,
        "asian_session": asian_session,
        "in_macro_window": in_macro_window,
        "silver_bullet_window": silver_bullet_window,
        "silver_bullet_bull": silver_bullet_bull,
        "silver_bullet_bear": silver_bullet_bear,
        "pdh": pdh,
        "pdl": pdl,
        "pwh": pwh,
        "pwl": pwl,
        "weekly_high": weekly_high_arr,
        "weekly_low": weekly_low_arr,
        "weekly_eq": weekly_eq_arr,
        "true_day_open": true_day_open_arr,
        "midnight_open": midnight_open_arr,
        "ny_0830_open": ny_0830_open_arr,
        "ipda_20_high": ipda_20_high,
        "ipda_20_low": ipda_20_low,
        "ipda_40_high": ipda_40_high,
        "ipda_40_low": ipda_40_low,
        "ipda_60_high": ipda_60_high,
        "ipda_60_low": ipda_60_low,
        "dol_bull_target": dol_bull_target,
        "dol_bear_target": dol_bear_target,
        "daily_bias": daily_bias,
        "weekly_bias": weekly_bias,
        "daily_bias_score": daily_bias_score,
        "ote_long": ote_long,
        "ote_short": ote_short,
        "liquidity_raid_bull": liquidity_raid_bull,
        "liquidity_raid_bear": liquidity_raid_bear,
        "judas_swing_bull": judas_swing_bull,
        "judas_swing_bear": judas_swing_bear,
        "po3_expansion_bull": po3_expansion_bull,
        "po3_expansion_bear": po3_expansion_bear,
        "opening_range_break_up": opening_range_break_up,
        "opening_range_break_down": opening_range_break_down,
        "consequent_encroachment_touch": consequent_encroachment_touch,
        "smt_divergence_bull": smt_divergence_bull,
        "smt_divergence_bear": smt_divergence_bear,
        "in_nwog": in_nwog,
        "in_ndog": in_ndog,
    }
    levels = {
        "pdh": float(pdh[-1]) if n and np.isfinite(pdh[-1]) else None,
        "pdl": float(pdl[-1]) if n and np.isfinite(pdl[-1]) else None,
        "pwh": float(pwh[-1]) if n and np.isfinite(pwh[-1]) else None,
        "pwl": float(pwl[-1]) if n and np.isfinite(pwl[-1]) else None,
        "true_day_open": float(true_day_open_arr[-1]) if n and np.isfinite(true_day_open_arr[-1]) else None,
        "ny_0830_open": float(ny_0830_open_arr[-1]) if n and np.isfinite(ny_0830_open_arr[-1]) else None,
        "dol_bull_target": float(dol_bull_target[-1]) if n and np.isfinite(dol_bull_target[-1]) else None,
        "dol_bear_target": float(dol_bear_target[-1]) if n and np.isfinite(dol_bear_target[-1]) else None,
    }
    return DetectionResult(arrays=arrays, zones=all_zones, events=all_events, levels=levels)


def compute_ict_features(
    open_time_ms: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    reference_high: np.ndarray | None = None,
    reference_low: np.ndarray | None = None,
    reference_open_time_ms: np.ndarray | None = None,
    session_cfg: ICTSessionConfig | None = None,
    min_fvg_atr_mult: float = 0.2,
    max_zone_age_bars: int = 100,
    trigger_on_retest: bool = False,
    swing_len: int = 5,
) -> dict[str, np.ndarray]:
    """Boolean & numeric array wrapper derived strictly from `detect_ict(...)`."""
    return detect_ict(
        open_time_ms=open_time_ms,
        open_=open_,
        high=high,
        low=low,
        close=close,
        reference_high=reference_high,
        reference_low=reference_low,
        reference_open_time_ms=reference_open_time_ms,
        session_cfg=session_cfg,
        min_fvg_atr_mult=min_fvg_atr_mult,
        max_zone_age_bars=max_zone_age_bars,
        trigger_on_retest=trigger_on_retest,
        swing_len=swing_len,
    ).arrays
