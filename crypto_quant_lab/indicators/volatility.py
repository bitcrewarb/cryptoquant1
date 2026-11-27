"""Causal vectorized volatility indicators and regime helpers."""

from __future__ import annotations

import numpy as np

from crypto_quant_lab.indicators.trend import donchian_channels, ema, sma


def _to_f64(arr: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def atr(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    period: int = 14,
) -> np.ndarray:
    h, l, c = _to_f64(high), _to_f64(low), _to_f64(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0 or n < period:
        return out

    tr = np.zeros(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    out[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        out[i] = ((out[i - 1] * (period - 1)) + tr[i]) / period
    return out


def historical_volatility(
    close: np.ndarray | list[float],
    period: int = 20,
    annualization_periods: float = 365.0 * 24.0,
) -> np.ndarray:
    c = _to_f64(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return out
    log_ret = np.full(n, np.nan, dtype=np.float64)
    log_ret[1:] = np.log(np.where(c[:-1] > 0, c[1:] / c[:-1], 1.0))
    scale = np.sqrt(annualization_periods)
    for i in range(period, n):
        out[i] = float(np.std(log_ret[i - period + 1 : i + 1], ddof=1)) * scale
    return out


def bollinger_bands(
    close: np.ndarray | list[float],
    period: int = 20,
    num_std: float = 2.0,
) -> dict[str, np.ndarray]:
    c = _to_f64(close)
    n = len(c)
    mid = sma(c, period)
    std = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        std[i] = float(np.std(c[i - period + 1 : i + 1], ddof=0))
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = np.where(mid > 0, (upper - lower) / mid, np.nan)
    denom = upper - lower
    percent_b = np.where(denom > 1e-12, (c - lower) / denom, 0.5)
    return {
        "upper": upper,
        "middle": mid,
        "lower": lower,
        "bandwidth": bandwidth,
        "percent_b": percent_b,
    }


def keltner_channels(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = _to_f64(close)
    mid = ema(c, ema_period)
    a = atr(high, low, c, atr_period)
    return mid + multiplier * a, mid, mid - multiplier * a


def donchian_width(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    period: int = 20,
) -> np.ndarray:
    upper, mid, lower = donchian_channels(high, low, period)
    return np.where(mid > 0, (upper - lower) / mid, np.nan)


def parkinson_volatility(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    period: int = 20,
    annualization_periods: float = 365.0 * 24.0,
) -> np.ndarray:
    h, l = _to_f64(high), _to_f64(low)
    n = len(h)
    out = np.full(n, np.nan, dtype=np.float64)
    hl = np.log(np.where((l > 0) & (h >= l), h / l, 1.0)) ** 2
    coef = 1.0 / (4.0 * np.log(2.0))
    for i in range(period - 1, n):
        var = coef * float(np.mean(hl[i - period + 1 : i + 1]))
        out[i] = float(np.sqrt(max(var, 0.0) * annualization_periods))
    return out


def garman_klass_volatility(
    open_: np.ndarray | list[float],
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    period: int = 20,
    annualization_periods: float = 365.0 * 24.0,
) -> np.ndarray:
    o, h, l, c = _to_f64(open_), _to_f64(high), _to_f64(low), _to_f64(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    hl = np.log(np.where((l > 0) & (h >= l), h / l, 1.0)) ** 2
    co = np.log(np.where((o > 0) & (c > 0), c / o, 1.0)) ** 2
    gk = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    for i in range(period - 1, n):
        var = float(np.mean(gk[i - period + 1 : i + 1]))
        out[i] = float(np.sqrt(max(var, 0.0) * annualization_periods))
    return out


def volatility_percentile(vol_series: np.ndarray | list[float], lookback: int = 100) -> np.ndarray:
    v = _to_f64(vol_series)
    n = len(v)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(lookback - 1, n):
        window = v[i - lookback + 1 : i + 1]
        finite = window[np.isfinite(window)]
        if len(finite) > 0 and np.isfinite(v[i]):
            out[i] = float(np.sum(finite <= v[i]) / len(finite) * 100.0)
    return out


def volatility_regime_labels(vol_pct: np.ndarray, slope_period: int = 5) -> list[str]:
    n = len(vol_pct)
    labels: list[str] = []
    for i in range(n):
        vp = vol_pct[i]
        if not np.isfinite(vp):
            labels.append("transitional")
            continue
        prev = vol_pct[i - slope_period] if i >= slope_period and np.isfinite(vol_pct[i - slope_period]) else vp
        delta = vp - prev
        if vp >= 75.0:
            labels.append("volatility_expansion" if delta > 10.0 else "high_volatility")
        elif vp <= 25.0:
            labels.append("volatility_compression" if delta < -10.0 else "low_volatility")
        elif delta >= 15.0:
            labels.append("volatility_expansion")
        elif delta <= -15.0:
            labels.append("volatility_compression")
        else:
            labels.append("normal_volatility")
    return labels
