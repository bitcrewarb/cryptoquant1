"""Causal vectorized trend indicators (zero look-ahead)."""

from __future__ import annotations

import numpy as np


def _to_f64(arr: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def sma(values: np.ndarray | list[float], period: int) -> np.ndarray:
    x = _to_f64(values)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0 or n < period:
        return out
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out[period - 1 :] = (cs[period:] - cs[:-period]) / period
    return out


def ema(values: np.ndarray | list[float], period: int) -> np.ndarray:
    x = _to_f64(values)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0 or n < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = float(np.mean(x[:period]))
    for i in range(period, n):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def wma(values: np.ndarray | list[float], period: int) -> np.ndarray:
    x = _to_f64(values)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0 or n < period:
        return out
    weights = np.arange(1, period + 1, dtype=np.float64)
    denom = weights.sum()
    for i in range(period - 1, n):
        out[i] = float(np.dot(x[i - period + 1 : i + 1], weights) / denom)
    return out


def hma(values: np.ndarray | list[float], period: int) -> np.ndarray:
    x = _to_f64(values)
    n = len(x)
    if period < 2 or n < period:
        return np.full(n, np.nan, dtype=np.float64)
    half = max(1, period // 2)
    sqrt_p = max(1, int(round(np.sqrt(period))))
    wma_half = wma(x, half)
    wma_full = wma(x, period)
    diff = 2.0 * wma_half - wma_full
    # Apply WMA only starting where diff is finite
    valid_idx = np.where(np.isfinite(diff))[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if len(valid_idx) >= sqrt_p:
        first_valid = int(valid_idx[0])
        sub_hma = wma(diff[first_valid:], sqrt_p)
        out[first_valid:] = sub_hma
    return out


def vwap(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    volume: np.ndarray | list[float],
    open_time_ms: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """UTC-day resetting cumulative VWAP (causal)."""
    h, l, c, v = _to_f64(high), _to_f64(low), _to_f64(close), _to_f64(volume)
    tp = (h + l + c) / 3.0
    n = len(tp)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    if open_time_ms is None:
        cum_pv = np.cumsum(tp * v)
        cum_v = np.cumsum(v)
        return np.where(cum_v > 0, cum_pv / cum_v, tp)

    times = np.asarray(open_time_ms, dtype=np.int64)
    days = times // 86_400_000
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = days[0]
    for i in range(n):
        if days[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = days[i]
        cum_pv += tp[i] * v[i]
        cum_v += v[i]
        out[i] = (cum_pv / cum_v) if cum_v > 0 else tp[i]
    return out


def anchored_vwap(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    volume: np.ndarray | list[float],
    anchor_index: int = 0,
) -> np.ndarray:
    h, l, c, v = _to_f64(high), _to_f64(low), _to_f64(close), _to_f64(volume)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if anchor_index < 0 or anchor_index >= n:
        return out
    tp = (h[anchor_index:] + l[anchor_index:] + c[anchor_index:]) / 3.0
    vol_slice = v[anchor_index:]
    cum_pv = np.cumsum(tp * vol_slice)
    cum_v = np.cumsum(vol_slice)
    out[anchor_index:] = np.where(cum_v > 0, cum_pv / cum_v, tp)
    return out


def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    out[period - 1] = float(np.sum(values[:period]))
    for i in range(period, n):
        out[i] = out[i - 1] - (out[i - 1] / period) + values[i]
    return out


def adx(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wilder's ADX, +DI, and -DI."""
    h, l, c = _to_f64(high), _to_f64(low), _to_f64(close)
    n = len(c)
    adx_arr = np.full(n, np.nan, dtype=np.float64)
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    if n < period * 2 + 1:
        return adx_arr, plus_di, minus_di

    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up_move = h[i] - h[i - 1]
        down_move = l[i - 1] - l[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    tr_s = _wilder_smooth(tr[1:], period)
    pdm_s = _wilder_smooth(plus_dm[1:], period)
    mdm_s = _wilder_smooth(minus_dm[1:], period)

    dx = np.full(len(tr_s), np.nan, dtype=np.float64)
    for i in range(period - 1, len(tr_s)):
        if tr_s[i] > 0:
            pdi = 100.0 * pdm_s[i] / tr_s[i]
            mdi = 100.0 * mdm_s[i] / tr_s[i]
            plus_di[i + 1] = pdi
            minus_di[i + 1] = mdi
            denom = pdi + mdi
            dx[i] = 100.0 * abs(pdi - mdi) / denom if denom > 0 else 0.0

    # Average DX over period
    start_dx = period - 1
    if start_dx + period <= len(dx):
        adx_val = float(np.nanmean(dx[start_dx : start_dx + period]))
        adx_arr[start_dx + period] = adx_val
        for i in range(start_dx + period, len(dx)):
            adx_val = ((adx_val * (period - 1)) + dx[i]) / period
            adx_arr[i + 1] = adx_val

    return adx_arr, plus_di, minus_di


def supertrend(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal Supertrend line and direction (+1 bullish, -1 bearish)."""
    from crypto_quant_lab.indicators.volatility import atr as calc_atr

    h, l, c = _to_f64(high), _to_f64(low), _to_f64(close)
    n = len(c)
    st = np.full(n, np.nan, dtype=np.float64)
    direction = np.zeros(n, dtype=np.int8)
    atr_vals = calc_atr(h, l, c, period=period)

    hl2 = (h + l) / 2.0
    upper_basic = hl2 + multiplier * atr_vals
    lower_basic = hl2 - multiplier * atr_vals
    upper_band = np.copy(upper_basic)
    lower_band = np.copy(lower_basic)

    first_valid = int(np.where(np.isfinite(atr_vals))[0][0]) if np.any(np.isfinite(atr_vals)) else n
    if first_valid >= n:
        return st, direction

    direction[first_valid] = 1
    st[first_valid] = lower_band[first_valid]

    for i in range(first_valid + 1, n):
        if upper_basic[i] < upper_band[i - 1] or c[i - 1] > upper_band[i - 1]:
            upper_band[i] = upper_basic[i]
        else:
            upper_band[i] = upper_band[i - 1]

        if lower_basic[i] > lower_band[i - 1] or c[i - 1] < lower_band[i - 1]:
            lower_band[i] = lower_basic[i]
        else:
            lower_band[i] = lower_band[i - 1]

        if direction[i - 1] == 1:
            if c[i] < lower_band[i]:
                direction[i] = -1
                st[i] = upper_band[i]
            else:
                direction[i] = 1
                st[i] = lower_band[i]
        else:
            if c[i] > upper_band[i]:
                direction[i] = 1
                st[i] = lower_band[i]
            else:
                direction[i] = -1
                st[i] = upper_band[i]

    return st, direction


def donchian_channels(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, l = _to_f64(high), _to_f64(low)
    n = len(h)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        upper[i] = float(np.max(h[i - period + 1 : i + 1]))
        lower[i] = float(np.min(l[i - period + 1 : i + 1]))
    middle = (upper + lower) / 2.0
    return upper, middle, lower


def ichimoku(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> dict[str, np.ndarray]:
    """Causal Ichimoku Cloud components (Senkou spans shifted forward cleanly from past data)."""
    h, l, c = _to_f64(high), _to_f64(low), _to_f64(close)
    n = len(c)

    def _midpoint(p: int) -> np.ndarray:
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(p - 1, n):
            out[i] = (float(np.max(h[i - p + 1 : i + 1])) + float(np.min(l[i - p + 1 : i + 1]))) / 2.0
        return out

    tenkan_sen = _midpoint(tenkan)
    kijun_sen = _midpoint(kijun)
    raw_sa = (tenkan_sen + kijun_sen) / 2.0
    raw_sb = _midpoint(senkou_b)

    # Shift forward by kijun bars: at index t, cloud uses values computed at t - kijun (strictly causal)
    span_a = np.full(n, np.nan, dtype=np.float64)
    span_b = np.full(n, np.nan, dtype=np.float64)
    if n > kijun:
        span_a[kijun:] = raw_sa[:-kijun]
        span_b[kijun:] = raw_sb[:-kijun]

    # Causal Chikou check: close[t] vs close[t - kijun]
    close_lagged = np.full(n, np.nan, dtype=np.float64)
    if n > kijun:
        close_lagged[kijun:] = c[:-kijun]

    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": span_a,
        "senkou_span_b": span_b,
        "close_lagged_kijun": close_lagged,
    }


def linear_regression_slope(values: np.ndarray | list[float], period: int = 20) -> np.ndarray:
    x = _to_f64(values)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if period < 2 or n < period:
        return out
    t = np.arange(period, dtype=np.float64)
    t_mean = t.mean()
    t_var = float(np.sum((t - t_mean) ** 2))
    for i in range(period - 1, n):
        y = x[i - period + 1 : i + 1]
        y_mean = float(np.mean(y))
        cov = float(np.sum((t - t_mean) * (y - y_mean)))
        out[i] = cov / t_var
    return out
