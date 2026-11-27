"""
Smart Money Concepts (SMC) structured causal engine (zero look-ahead).

Returns structured ZoneObject and EventObject instances via `detect_smc(...)` as the
single source of truth for strategies, signal engine, and charts, while deriving
boolean/float screener arrays in `compute_smc_features(...)`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from crypto_quant_lab.core.structures import DetectionResult, EventObject, ZoneObject
from crypto_quant_lab.indicators.volatility import atr


def _safe_times(open_time_ms: np.ndarray | None, n: int) -> np.ndarray:
    if open_time_ms is not None and len(open_time_ms) == n:
        return np.asarray(open_time_ms, dtype=np.int64)
    return np.arange(n, dtype=np.int64) * 3_600_000


def detect_smc(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_time_ms: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    int_swing: int = 2,
    ext_swing: int = 5,
    eq_tol_atr: float = 0.25,
    displacement_atr_mult: float = 1.5,
    min_fvg_atr_mult: float = 0.2,
    max_zone_age_bars: int = 100,
    trigger_on_retest: bool = False,
    ob_refined_mode: str = "full",   # full | body | mean_threshold
    ob_require_fvg: bool = False,
) -> DetectionResult:
    """Strictly causal SMC detector producing structured zones, events, levels, and screener arrays."""
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    n = len(c)
    times = _safe_times(open_time_ms, n)
    vol = np.asarray(volume, dtype=np.float64) if volume is not None and len(volume) == n else np.ones(n, dtype=np.float64)

    int_swing = max(1, int(int_swing))
    ext_swing = max(int_swing + 1, int(ext_swing))

    atr_vals = atr(h, l, c, period=14)
    body = np.abs(c - o)

    # Screener arrays
    bos_bull = np.zeros(n, dtype=bool)
    bos_bear = np.zeros(n, dtype=bool)
    choch_bull = np.zeros(n, dtype=bool)
    choch_bear = np.zeros(n, dtype=bool)
    mss_bull = np.zeros(n, dtype=bool)
    mss_bear = np.zeros(n, dtype=bool)
    internal_bos_bull = np.zeros(n, dtype=bool)
    internal_bos_bear = np.zeros(n, dtype=bool)
    internal_choch_bull = np.zeros(n, dtype=bool)
    internal_choch_bear = np.zeros(n, dtype=bool)
    inducement_bull = np.zeros(n, dtype=bool)
    inducement_bear = np.zeros(n, dtype=bool)

    strong_high_flag = np.zeros(n, dtype=bool)
    weak_high_flag = np.zeros(n, dtype=bool)
    strong_low_flag = np.zeros(n, dtype=bool)
    weak_low_flag = np.zeros(n, dtype=bool)

    eqh = np.zeros(n, dtype=bool)
    eql = np.zeros(n, dtype=bool)
    bsl_sweep = np.zeros(n, dtype=bool)
    ssl_sweep = np.zeros(n, dtype=bool)

    fvg_bull = np.zeros(n, dtype=bool)
    fvg_bear = np.zeros(n, dtype=bool)
    ifvg_bull = np.zeros(n, dtype=bool)
    ifvg_bear = np.zeros(n, dtype=bool)
    bpr_bull = np.zeros(n, dtype=bool)
    bpr_bear = np.zeros(n, dtype=bool)
    volume_imbalance_bull = np.zeros(n, dtype=bool)
    volume_imbalance_bear = np.zeros(n, dtype=bool)
    displacement_bull = np.zeros(n, dtype=bool)
    displacement_bear = np.zeros(n, dtype=bool)

    in_bull_ob = np.zeros(n, dtype=bool)
    in_bear_ob = np.zeros(n, dtype=bool)
    in_bull_fvg = np.zeros(n, dtype=bool)
    in_bear_fvg = np.zeros(n, dtype=bool)
    in_bull_breaker = np.zeros(n, dtype=bool)
    in_bear_breaker = np.zeros(n, dtype=bool)
    in_mitigation_zone = np.zeros(n, dtype=bool)
    in_bpr = np.zeros(n, dtype=bool)

    premium_zone = np.zeros(n, dtype=bool)
    discount_zone = np.zeros(n, dtype=bool)
    near_equilibrium = np.zeros(n, dtype=bool)
    dealing_range_pos = np.full(n, 0.5, dtype=np.float64)
    equilibrium_arr = np.full(n, np.nan, dtype=np.float64)

    all_zones: list[ZoneObject] = []
    all_events: list[EventObject] = []
    zone_seq = 0
    event_seq = 0

    def _new_zone(
        ztype: str,
        direction: str,
        bottom: float,
        top: float,
        bar_i: int,
        source_event: str,
        strength: float = 1.0,
        expiry_bars: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ZoneObject:
        nonlocal zone_seq
        zone_seq += 1
        lo_v = min(float(bottom), float(top))
        hi_v = max(float(bottom), float(top))
        ts = int(times[bar_i])
        z = ZoneObject(
            id=f"smc_{ztype}_{direction}_{zone_seq}",
            type=ztype,
            direction=direction,
            top=hi_v,
            bottom=lo_v,
            created_bar=bar_i,
            created_time=ts,
            confirmed_time=ts,
            status="fresh",
            expiry=(bar_i + (expiry_bars if expiry_bars is not None else max_zone_age_bars)),
            source_event=source_event,
            strength=float(strength),
            meta=dict(meta or {}),
        )
        all_zones.append(z)
        return z

    def _new_event(
        etype: str,
        direction: str,
        bar_i: int,
        price: float,
        level: float | None = None,
        linked_zones: list[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> EventObject:
        nonlocal event_seq
        event_seq += 1
        ev = EventObject(
            id=f"smc_ev_{etype}_{event_seq}",
            type=etype,
            direction=direction,
            time=int(times[bar_i]),
            bar_index=bar_i,
            price=float(price),
            level=float(level) if level is not None and np.isfinite(level) else None,
            linked_zone_ids=list(linked_zones or []),
            meta=dict(meta or {}),
        )
        all_events.append(ev)
        return ev

    def _advance_zone_touch(z: ZoneObject, bar_i: int) -> bool:
        if bar_i <= z.created_bar or z.status in ("mitigated", "invalidated", "expired"):
            return False
        if z.expiry is not None and bar_i > z.expiry:
            z.status = "expired"
            return False
        if not (l[bar_i] <= z.top and h[bar_i] >= z.bottom):
            return False

        mid = z.midpoint
        height = max(z.top - z.bottom, 1e-12)
        if z.direction == "bull":
            pen = max(0.0, min(1.0, (z.top - l[bar_i]) / height))
        else:
            pen = max(0.0, min(1.0, (h[bar_i] - z.bottom) / height))
        z.fill_pct = max(z.fill_pct, pen)

        should_fire = z.status == "fresh" or (trigger_on_retest and z.status == "tested")
        z.touches += 1
        if (z.direction == "bull" and l[bar_i] <= mid) or (z.direction == "bear" and h[bar_i] >= mid) or z.touches >= 2:
            z.status = "mitigated"
            z.mitigated_time = int(times[bar_i])
        else:
            z.status = "tested"
        return should_fire

    # Structure trackers
    ext_sh, ext_sl = np.nan, np.nan
    ext_sh_bar, ext_sl_bar = -1, -1
    prev_ext_sh, prev_ext_sl = np.nan, np.nan
    strong_high_lvl, weak_high_lvl = np.nan, np.nan
    strong_low_lvl, weak_low_lvl = np.nan, np.nan
    int_sh, int_sl = np.nan, np.nan
    ext_bias = 0
    int_bias = 0

    active_bull_fvgs: list[ZoneObject] = []
    active_bear_fvgs: list[ZoneObject] = []
    active_bull_obs: list[ZoneObject] = []
    active_bear_obs: list[ZoneObject] = []
    active_bull_breakers: list[ZoneObject] = []
    active_bear_breakers: list[ZoneObject] = []
    active_bprs: list[ZoneObject] = []
    active_liq_pools: list[ZoneObject] = []

    # Calendar period H/L liquidity pools (PDH/PDL/PWH/PWL/PMH/PML)
    cur_day = None
    day_hi, day_lo = -np.inf, np.inf
    cur_week = None
    wk_hi, wk_lo = -np.inf, np.inf
    cur_month = None
    mo_hi, mo_lo = -np.inf, np.inf

    for i in range(n):
        ts = int(times[i])
        dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        d_key = dt.date()
        iso = dt.isocalendar()
        w_key = (iso[0], iso[1])
        m_key = (dt.year, dt.month)
        a = atr_vals[i] if np.isfinite(atr_vals[i]) and atr_vals[i] > 0 else max(h[i] - l[i], 1e-9)

        if cur_day != d_key:
            if cur_day is not None:
                active_liq_pools.append(_new_zone("liquidity_pool", "bear", day_hi - 0.05 * a, day_hi + 0.05 * a, i, "PDH", strength=1.5, expiry_bars=max_zone_age_bars * 2, meta={"pool": "PDH"}))
                active_liq_pools.append(_new_zone("liquidity_pool", "bull", day_lo - 0.05 * a, day_lo + 0.05 * a, i, "PDL", strength=1.5, expiry_bars=max_zone_age_bars * 2, meta={"pool": "PDL"}))
            cur_day = d_key
            day_hi, day_lo = h[i], l[i]
        else:
            day_hi = max(day_hi, h[i])
            day_lo = min(day_lo, l[i])

        if cur_week != w_key:
            if cur_week is not None:
                active_liq_pools.append(_new_zone("liquidity_pool", "bear", wk_hi - 0.05 * a, wk_hi + 0.05 * a, i, "PWH", strength=2.0, expiry_bars=max_zone_age_bars * 4, meta={"pool": "PWH"}))
                active_liq_pools.append(_new_zone("liquidity_pool", "bull", wk_lo - 0.05 * a, wk_lo + 0.05 * a, i, "PWL", strength=2.0, expiry_bars=max_zone_age_bars * 4, meta={"pool": "PWL"}))
            cur_week = w_key
            wk_hi, wk_lo = h[i], l[i]
        else:
            wk_hi = max(wk_hi, h[i])
            wk_lo = min(wk_lo, l[i])

        if cur_month != m_key:
            if cur_month is not None:
                active_liq_pools.append(_new_zone("liquidity_pool", "bear", mo_hi - 0.05 * a, mo_hi + 0.05 * a, i, "PMH", strength=2.5, expiry_bars=max_zone_age_bars * 8, meta={"pool": "PMH"}))
                active_liq_pools.append(_new_zone("liquidity_pool", "bull", mo_lo - 0.05 * a, mo_lo + 0.05 * a, i, "PML", strength=2.5, expiry_bars=max_zone_age_bars * 8, meta={"pool": "PML"}))
            cur_month = m_key
            mo_hi, mo_lo = h[i], l[i]
        else:
            mo_hi = max(mo_hi, h[i])
            mo_lo = min(mo_lo, l[i])

        if len(active_liq_pools) > 24:
            active_liq_pools = [p for p in active_liq_pools if p.status == "fresh"][-24:]

        if i < max(ext_swing * 2, 2):
            continue

        # 1. Confirm internal swing at j = i - int_swing
        ji = i - int_swing
        if ji >= int_swing:
            wh = h[ji - int_swing : ji + int_swing + 1]
            wl = l[ji - int_swing : ji + int_swing + 1]
            if h[ji] == np.max(wh) and np.sum(wh == h[ji]) == 1:
                int_sh = float(h[ji])
            if l[ji] == np.min(wl) and np.sum(wl == l[ji]) == 1:
                int_sl = float(l[ji])

        # 2. Confirm external swing at je = i - ext_swing
        je = i - ext_swing
        if je >= ext_swing:
            wh_e = h[je - ext_swing : je + ext_swing + 1]
            wl_e = l[je - ext_swing : je + ext_swing + 1]
            if h[je] == np.max(wh_e) and np.sum(wh_e == h[je]) == 1:
                prev_ext_sh = ext_sh
                ext_sh = float(h[je])
                ext_sh_bar = je
                if ext_bias == -1:
                    weak_high_lvl = ext_sh
                    weak_high_flag[i] = True
                if np.isfinite(prev_ext_sh) and abs(ext_sh - prev_ext_sh) <= eq_tol_atr * a:
                    eqh[i] = True
                    z_eqh = _new_zone(
                        "liquidity_pool",
                        "bear",
                        min(ext_sh, prev_ext_sh),
                        max(ext_sh, prev_ext_sh),
                        i,
                        "EQH",
                        strength=1.5,
                        meta={"pool": "EQH"},
                    )
                    active_liq_pools.append(z_eqh)
                    _new_event("eqh", "bear", i, ext_sh, level=prev_ext_sh, linked_zones=[z_eqh.id])
            if l[je] == np.min(wl_e) and np.sum(wl_e == l[je]) == 1:
                prev_ext_sl = ext_sl
                ext_sl = float(l[je])
                ext_sl_bar = je
                if ext_bias == 1:
                    weak_low_lvl = ext_sl
                    weak_low_flag[i] = True
                if np.isfinite(prev_ext_sl) and abs(ext_sl - prev_ext_sl) <= eq_tol_atr * a:
                    eql[i] = True
                    z_eql = _new_zone(
                        "liquidity_pool",
                        "bull",
                        min(ext_sl, prev_ext_sl),
                        max(ext_sl, prev_ext_sl),
                        i,
                        "EQL",
                        strength=1.5,
                        meta={"pool": "EQL"},
                    )
                    active_liq_pools.append(z_eql)
                    _new_event("eql", "bull", i, ext_sl, level=prev_ext_sl, linked_zones=[z_eql.id])

        # 3. Displacement
        if body[i] > displacement_atr_mult * a:
            if c[i] > o[i]:
                displacement_bull[i] = True
            elif c[i] < o[i]:
                displacement_bear[i] = True

        # 4. Internal BOS / CHoCH
        if np.isfinite(int_sh) and c[i] > int_sh and c[i - 1] <= int_sh:
            if int_bias == -1:
                internal_choch_bull[i] = True
            else:
                internal_bos_bull[i] = True
            int_bias = 1
        if np.isfinite(int_sl) and c[i] < int_sl and c[i - 1] >= int_sl:
            if int_bias == 1:
                internal_choch_bear[i] = True
            else:
                internal_bos_bear[i] = True
            int_bias = -1

        # 5. Advance lifecycle on active OBs, Breakers, BPRs, and FVGs BEFORE bar-i creation
        surviving_bull_obs: list[ZoneObject] = []
        for ob in active_bull_obs:
            if i > (ob.expiry or (ob.created_bar + max_zone_age_bars)):
                ob.status = "expired"
                continue
            if c[i] < ob.bottom:
                ob.status = "invalidated"
                ob.mitigated_time = ts
                brk = _new_zone("breaker", "bear", ob.bottom, ob.top, i, "OB_FAILED_BEAR", strength=ob.strength)
                active_bear_breakers.append(brk)
                active_bear_breakers = active_bear_breakers[-8:]
                _new_event("breaker_created", "bear", i, c[i], level=ob.bottom, linked_zones=[brk.id, ob.id])
            else:
                if _advance_zone_touch(ob, i):
                    in_bull_ob[i] = True
                    in_mitigation_zone[i] = True
                if ob.status != "mitigated":
                    surviving_bull_obs.append(ob)
        active_bull_obs = surviving_bull_obs

        surviving_bear_obs: list[ZoneObject] = []
        for ob in active_bear_obs:
            if i > (ob.expiry or (ob.created_bar + max_zone_age_bars)):
                ob.status = "expired"
                continue
            if c[i] > ob.top:
                ob.status = "invalidated"
                ob.mitigated_time = ts
                brk = _new_zone("breaker", "bull", ob.bottom, ob.top, i, "OB_FAILED_BULL", strength=ob.strength)
                active_bull_breakers.append(brk)
                active_bull_breakers = active_bull_breakers[-8:]
                _new_event("breaker_created", "bull", i, c[i], level=ob.top, linked_zones=[brk.id, ob.id])
            else:
                if _advance_zone_touch(ob, i):
                    in_bear_ob[i] = True
                    in_mitigation_zone[i] = True
                if ob.status != "mitigated":
                    surviving_bear_obs.append(ob)
        active_bear_obs = surviving_bear_obs

        surviving_bull_brk: list[ZoneObject] = []
        for brk in active_bull_breakers:
            if i > (brk.expiry or (brk.created_bar + max_zone_age_bars)):
                brk.status = "expired"
                continue
            if c[i] < brk.bottom:
                brk.status = "invalidated"
                brk.mitigated_time = ts
                continue
            if _advance_zone_touch(brk, i):
                in_bull_breaker[i] = True
            if brk.status != "mitigated":
                surviving_bull_brk.append(brk)
        active_bull_breakers = surviving_bull_brk

        surviving_bear_brk: list[ZoneObject] = []
        for brk in active_bear_breakers:
            if i > (brk.expiry or (brk.created_bar + max_zone_age_bars)):
                brk.status = "expired"
                continue
            if c[i] > brk.top:
                brk.status = "invalidated"
                brk.mitigated_time = ts
                continue
            if _advance_zone_touch(brk, i):
                in_bear_breaker[i] = True
            if brk.status != "mitigated":
                surviving_bear_brk.append(brk)
        active_bear_breakers = surviving_bear_brk

        surviving_bpr: list[ZoneObject] = []
        for bpr in active_bprs:
            if i > (bpr.expiry or (bpr.created_bar + max_zone_age_bars)):
                bpr.status = "expired"
                continue
            if _advance_zone_touch(bpr, i):
                in_bpr[i] = True
            if bpr.status != "mitigated":
                surviving_bpr.append(bpr)
        active_bprs = surviving_bpr

        # 6. External BOS / CHoCH / MSS + Strong/Weak extremes + Order Block creation
        def _extract_ob_bounds(k_idx: int, direction: str) -> tuple[float, float]:
            if ob_refined_mode == "body":
                return min(float(o[k_idx]), float(c[k_idx])), max(float(o[k_idx]), float(c[k_idx]))
            if ob_refined_mode == "mean_threshold":
                mid = (float(l[k_idx]) + float(h[k_idx])) * 0.5
                return (mid, float(h[k_idx])) if direction == "bull" else (float(l[k_idx]), mid)
            return float(l[k_idx]), float(h[k_idx])

        has_recent_bull_fvg = i >= 2 and (
            (l[i] > h[i - 2] and (l[i] - h[i - 2]) >= min_fvg_atr_mult * a and c[i - 1] > o[i - 1])
            or (i >= 3 and fvg_bull[i - 1])
        )
        has_recent_bear_fvg = i >= 2 and (
            (h[i] < l[i - 2] and (l[i - 2] - h[i]) >= min_fvg_atr_mult * a and c[i - 1] < o[i - 1])
            or (i >= 3 and fvg_bear[i - 1])
        )

        if np.isfinite(ext_sh) and c[i] > ext_sh and c[i - 1] <= ext_sh:
            if ext_bias == -1:
                choch_bull[i] = True
                ev_type = "mss_bull" if displacement_bull[i] else "choch_bull"
                if displacement_bull[i]:
                    mss_bull[i] = True
            else:
                bos_bull[i] = True
                ev_type = "bos_bull"
            ext_bias = 1
            if np.isfinite(ext_sl):
                strong_low_lvl = ext_sl
                strong_low_flag[i] = True

            linked: list[str] = []
            if not ob_require_fvg or has_recent_bull_fvg:
                for k in range(i - 1, max(0, i - 6) - 1, -1):
                    if c[k] < o[k]:
                        lo_ob, hi_ob = _extract_ob_bounds(k, "bull")
                        strength = 1.5 if has_recent_bull_fvg else 1.0
                        ob_z = _new_zone("ob", "bull", lo_ob, hi_ob, i, ev_type, strength=strength, meta={"candle_bar": k, "has_fvg": has_recent_bull_fvg})
                        active_bull_obs.append(ob_z)
                        active_bull_obs = active_bull_obs[-8:]
                        linked.append(ob_z.id)
                        break
            _new_event(ev_type, "bull", i, c[i], level=ext_sh, linked_zones=linked)

        if np.isfinite(ext_sl) and c[i] < ext_sl and c[i - 1] >= ext_sl:
            if ext_bias == 1:
                choch_bear[i] = True
                ev_type = "mss_bear" if displacement_bear[i] else "choch_bear"
                if displacement_bear[i]:
                    mss_bear[i] = True
            else:
                bos_bear[i] = True
                ev_type = "bos_bear"
            ext_bias = -1
            if np.isfinite(ext_sh):
                strong_high_lvl = ext_sh
                strong_high_flag[i] = True

            linked = []
            if not ob_require_fvg or has_recent_bear_fvg:
                for k in range(i - 1, max(0, i - 6) - 1, -1):
                    if c[k] > o[k]:
                        lo_ob, hi_ob = _extract_ob_bounds(k, "bear")
                        strength = 1.5 if has_recent_bear_fvg else 1.0
                        ob_z = _new_zone("ob", "bear", lo_ob, hi_ob, i, ev_type, strength=strength, meta={"candle_bar": k, "has_fvg": has_recent_bear_fvg})
                        active_bear_obs.append(ob_z)
                        active_bear_obs = active_bear_obs[-8:]
                        linked.append(ob_z.id)
                        break
            _new_event(ev_type, "bear", i, c[i], level=ext_sl, linked_zones=linked)

        # 7. Liquidity Sweeps (external swing or active liquidity pools)
        swept_any_bsl = False
        swept_any_ssl = False
        if np.isfinite(ext_sh) and h[i] > ext_sh and c[i] < ext_sh:
            bsl_sweep[i] = True
            swept_any_bsl = True
            _new_event("bsl_sweep", "bear", i, h[i], level=ext_sh)
        if np.isfinite(ext_sl) and l[i] < ext_sl and c[i] > ext_sl:
            ssl_sweep[i] = True
            swept_any_ssl = True
            _new_event("ssl_sweep", "bull", i, l[i], level=ext_sl)

        for pool in active_liq_pools:
            if pool.status != "fresh" or i <= pool.created_bar:
                continue
            if pool.direction == "bear" and h[i] > pool.top and c[i] < pool.top:
                pool.status = "mitigated"
                pool.mitigated_time = ts
                bsl_sweep[i] = True
                if not swept_any_bsl:
                    _new_event("bsl_sweep", "bear", i, h[i], level=pool.top, linked_zones=[pool.id], meta=pool.meta)
                    swept_any_bsl = True
            elif pool.direction == "bull" and l[i] < pool.bottom and c[i] > pool.bottom:
                pool.status = "mitigated"
                pool.mitigated_time = ts
                ssl_sweep[i] = True
                if not swept_any_ssl:
                    _new_event("ssl_sweep", "bull", i, l[i], level=pool.bottom, linked_zones=[pool.id], meta=pool.meta)
                    swept_any_ssl = True

        # Inducement: internal swing swept while unmitigated external OB sits deeper
        if np.isfinite(int_sl) and l[i] < int_sl and c[i] > int_sl:
            if any(ob.status == "fresh" and ob.top < int_sl for ob in active_bull_obs):
                inducement_bull[i] = True
        if np.isfinite(int_sh) and h[i] > int_sh and c[i] < int_sh:
            if any(ob.status == "fresh" and ob.bottom > int_sh for ob in active_bear_obs):
                inducement_bear[i] = True

        # 8. Inverse FVG check & FVG touch tracking on previously active FVGs
        new_bear_fvgs: list[ZoneObject] = []
        for g in active_bear_fvgs:
            if i > (g.expiry or (g.created_bar + max_zone_age_bars)):
                g.status = "expired"
                continue
            if c[i] > g.top:
                g.status = "invalidated"
                g.mitigated_time = ts
                ifvg_bull[i] = True
                iz = _new_zone("ifvg", "bull", g.bottom, g.top, i, "IFVG_BULL", strength=g.strength)
                _new_event("ifvg_bull", "bull", i, c[i], level=g.top, linked_zones=[iz.id, g.id])
            else:
                if _advance_zone_touch(g, i):
                    in_bear_fvg[i] = True
                if g.status != "mitigated":
                    new_bear_fvgs.append(g)
        active_bear_fvgs = new_bear_fvgs

        new_bull_fvgs: list[ZoneObject] = []
        for g in active_bull_fvgs:
            if i > (g.expiry or (g.created_bar + max_zone_age_bars)):
                g.status = "expired"
                continue
            if c[i] < g.bottom:
                g.status = "invalidated"
                g.mitigated_time = ts
                ifvg_bear[i] = True
                iz = _new_zone("ifvg", "bear", g.bottom, g.top, i, "IFVG_BEAR", strength=g.strength)
                _new_event("ifvg_bear", "bear", i, c[i], level=g.bottom, linked_zones=[iz.id, g.id])
            else:
                if _advance_zone_touch(g, i):
                    in_bull_fvg[i] = True
                if g.status != "mitigated":
                    new_bull_fvgs.append(g)
        active_bull_fvgs = new_bull_fvgs

        # 9. Fair Value Gaps + Balanced Price Range (overlap between opposing FVGs)
        min_gap = min_fvg_atr_mult * a
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= min_gap and c[i - 1] > o[i - 1]:
            fvg_bull[i] = True
            gap_ratio = (l[i] - h[i - 2]) / max(a, 1e-9)
            fz = _new_zone("fvg", "bull", float(h[i - 2]), float(l[i]), i, "FVG_BULL", strength=min(3.0, gap_ratio))
            active_bull_fvgs.append(fz)
            active_bull_fvgs = active_bull_fvgs[-10:]
            for bg in active_bear_fvgs:
                ov_lo = max(fz.bottom, bg.bottom)
                ov_hi = min(fz.top, bg.top)
                if ov_hi > ov_lo:
                    bpr_bull[i] = True
                    bz = _new_zone("bpr", "bull", ov_lo, ov_hi, i, "BPR_BULL", strength=1.5)
                    active_bprs.append(bz)
                    active_bprs = active_bprs[-6:]
                    break

        if h[i] < l[i - 2] and (l[i - 2] - h[i]) >= min_gap and c[i - 1] < o[i - 1]:
            fvg_bear[i] = True
            gap_ratio = (l[i - 2] - h[i]) / max(a, 1e-9)
            fz = _new_zone("fvg", "bear", float(h[i]), float(l[i - 2]), i, "FVG_BEAR", strength=min(3.0, gap_ratio))
            active_bear_fvgs.append(fz)
            active_bear_fvgs = active_bear_fvgs[-10:]
            for bg in active_bull_fvgs:
                ov_lo = max(fz.bottom, bg.bottom)
                ov_hi = min(fz.top, bg.top)
                if ov_hi > ov_lo:
                    bpr_bear[i] = True
                    bz = _new_zone("bpr", "bear", ov_lo, ov_hi, i, "BPR_BEAR", strength=1.5)
                    active_bprs.append(bz)
                    active_bprs = active_bprs[-6:]
                    break

        # 10. Volume Imbalance (gap between consecutive candle bodies with real open/close separation)
        prev_body_hi = max(o[i - 1], c[i - 1])
        prev_body_lo = min(o[i - 1], c[i - 1])
        cur_body_hi = max(o[i], c[i])
        cur_body_lo = min(o[i], c[i])
        if cur_body_lo > prev_body_hi and (cur_body_lo - prev_body_hi) >= 0.15 * a and c[i] > o[i]:
            volume_imbalance_bull[i] = True
            _new_zone("volume_imbalance", "bull", prev_body_hi, cur_body_lo, i, "VOL_IMB_BULL", strength=float(vol[i] > np.mean(vol[max(0, i - 10) : i + 1])))
        elif cur_body_hi < prev_body_lo and (prev_body_lo - cur_body_hi) >= 0.15 * a and c[i] < o[i]:
            volume_imbalance_bear[i] = True
            _new_zone("volume_imbalance", "bear", cur_body_hi, prev_body_lo, i, "VOL_IMB_BEAR", strength=float(vol[i] > np.mean(vol[max(0, i - 10) : i + 1])))

        # 11. Dealing Range, Equilibrium & Premium / Discount Zones
        if np.isfinite(ext_sh) and np.isfinite(ext_sl) and ext_sh > ext_sl:
            pos = (c[i] - ext_sl) / (ext_sh - ext_sl)
            eq_px = (ext_sh + ext_sl) * 0.5
            dealing_range_pos[i] = pos
            equilibrium_arr[i] = eq_px
            premium_zone[i] = pos >= 0.5
            discount_zone[i] = pos < 0.5
            near_equilibrium[i] = abs(pos - 0.5) <= 0.08

    arrays: dict[str, np.ndarray] = {
        "bos_bull": bos_bull,
        "bos_bear": bos_bear,
        "choch_bull": choch_bull,
        "choch_bear": choch_bear,
        "mss_bull": mss_bull,
        "mss_bear": mss_bear,
        "internal_bos_bull": internal_bos_bull,
        "internal_bos_bear": internal_bos_bear,
        "internal_choch_bull": internal_choch_bull,
        "internal_choch_bear": internal_choch_bear,
        "inducement_bull": inducement_bull,
        "inducement_bear": inducement_bear,
        "strong_high": strong_high_flag,
        "weak_high": weak_high_flag,
        "strong_low": strong_low_flag,
        "weak_low": weak_low_flag,
        "eqh": eqh,
        "eql": eql,
        "bsl_sweep": bsl_sweep,
        "ssl_sweep": ssl_sweep,
        "fvg_bull": fvg_bull,
        "fvg_bear": fvg_bear,
        "ifvg_bull": ifvg_bull,
        "ifvg_bear": ifvg_bear,
        "bpr_bull": bpr_bull,
        "bpr_bear": bpr_bear,
        "volume_imbalance_bull": volume_imbalance_bull,
        "volume_imbalance_bear": volume_imbalance_bear,
        "displacement_bull": displacement_bull,
        "displacement_bear": displacement_bear,
        "in_bull_ob": in_bull_ob,
        "in_bear_ob": in_bear_ob,
        "in_bull_fvg": in_bull_fvg,
        "in_bear_fvg": in_bear_fvg,
        "in_bull_breaker": in_bull_breaker,
        "in_bear_breaker": in_bear_breaker,
        "in_mitigation_zone": in_mitigation_zone,
        "in_bpr": in_bpr,
        "premium_zone": premium_zone,
        "discount_zone": discount_zone,
        "near_equilibrium": near_equilibrium,
        "dealing_range_pos": dealing_range_pos,
        "equilibrium": equilibrium_arr,
    }
    levels = {
        "ext_swing_high": float(ext_sh) if np.isfinite(ext_sh) else None,
        "ext_swing_low": float(ext_sl) if np.isfinite(ext_sl) else None,
        "strong_high": float(strong_high_lvl) if np.isfinite(strong_high_lvl) else None,
        "weak_high": float(weak_high_lvl) if np.isfinite(weak_high_lvl) else None,
        "strong_low": float(strong_low_lvl) if np.isfinite(strong_low_lvl) else None,
        "weak_low": float(weak_low_lvl) if np.isfinite(weak_low_lvl) else None,
        "ext_bias": int(ext_bias),
        "unswept_liquidity_pools": [p.to_dict() for p in active_liq_pools if p.status == "fresh"],
    }
    return DetectionResult(arrays=arrays, zones=all_zones, events=all_events, levels=levels)


def compute_smc_features(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    int_swing: int = 2,
    ext_swing: int = 5,
    eq_tol_atr: float = 0.25,
    displacement_atr_mult: float = 1.5,
    min_fvg_atr_mult: float = 0.2,
    max_zone_age_bars: int = 100,
    trigger_on_retest: bool = False,
    ob_refined_mode: str = "full",
    ob_require_fvg: bool = False,
    open_time_ms: np.ndarray | None = None,
    volume: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Boolean & numeric array wrapper derived strictly from `detect_smc(...)`."""
    return detect_smc(
        open_=open_,
        high=high,
        low=low,
        close=close,
        open_time_ms=open_time_ms,
        volume=volume,
        int_swing=int_swing,
        ext_swing=ext_swing,
        eq_tol_atr=eq_tol_atr,
        displacement_atr_mult=displacement_atr_mult,
        min_fvg_atr_mult=min_fvg_atr_mult,
        max_zone_age_bars=max_zone_age_bars,
        trigger_on_retest=trigger_on_retest,
        ob_refined_mode=ob_refined_mode,
        ob_require_fvg=ob_require_fvg,
    ).arrays
