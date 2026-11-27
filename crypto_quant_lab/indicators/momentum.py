"""Causal vectorized momentum indicators."""

from __future__ import annotations

import numpy as np

from crypto_quant_lab.indicators.trend import ema, sma


def _to_f64(arr: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def rsi(close: np.ndarray | list[float], period: int = 14) -> np.ndarray:
    c = _to_f64(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0 or n <= period:
        return out

    deltas = np.diff(c)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    out[period] = 100.0 if avg_loss == 0 else (100.0 - 100.0 / (1.0 + avg_gain / avg_loss))

    for i in range(period, len(deltas)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
        if avg_loss == 0.0:
            out[i + 1] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return out


def stochastic_rsi(
    close: np.ndarray | list[float],
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    r = rsi(close, period=rsi_period)
    n = len(r)
    raw_k = np.full(n, np.nan, dtype=np.float64)
    for i in range(rsi_period + stoch_period - 1, n):
        window = r[i - stoch_period + 1 : i + 1]
        if np.any(np.isnan(window)):
            continue
        r_min = float(np.min(window))
        r_max = float(np.max(window))
        raw_k[i] = 100.0 * (r[i] - r_min) / (r_max - r_min) if (r_max - r_min) > 1e-12 else 50.0

    # Smooth valid slice
    k = np.full(n, np.nan, dtype=np.float64)
    d = np.full(n, np.nan, dtype=np.float64)
    valid = np.where(np.isfinite(raw_k))[0]
    if len(valid) >= k_smooth:
        first = int(valid[0])
        k[first:] = sma(raw_k[first:], k_smooth)
        valid_k = np.where(np.isfinite(k))[0]
        if len(valid_k) >= d_smooth:
            first_k = int(valid_k[0])
            d[first_k:] = sma(k[first_k:], d_smooth)
    return k, d


def macd(
    close: np.ndarray | list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = _to_f64(close)
    ema_fast = ema(c, fast)
    ema_slow = ema(c, slow)
    macd_line = ema_fast - ema_slow
    n = len(c)
    sig_line = np.full(n, np.nan, dtype=np.float64)
    valid = np.where(np.isfinite(macd_line))[0]
    if len(valid) >= signal:
        first = int(valid[0])
        sig_line[first:] = ema(macd_line[first:], signal)
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


def cci(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    period: int = 20,
) -> np.ndarray:
    h, l, c = _to_f64(high), _to_f64(low), _to_f64(close)
    tp = (h + l + c) / 3.0
    n = len(tp)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        window = tp[i - period + 1 : i + 1]
        mean = float(np.mean(window))
        mad = float(np.mean(np.abs(window - mean)))
        out[i] = (tp[i] - mean) / (0.015 * mad) if mad > 1e-12 else 0.0
    return out


def roc(close: np.ndarray | list[float], period: int = 12) -> np.ndarray:
    c = _to_f64(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0 or n <= period:
        return out
    prev = c[:-period]
    out[period:] = np.where(prev > 0, (c[period:] - prev) / prev * 100.0, np.nan)
    return out


def momentum(close: np.ndarray | list[float], period: int = 10) -> np.ndarray:
    c = _to_f64(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0 or n <= period:
        return out
    out[period:] = c[period:] - c[:-period]
    return out


def williams_r(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    period: int = 14,
) -> np.ndarray:
    h, l, c = _to_f64(high), _to_f64(low), _to_f64(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        hh = float(np.max(h[i - period + 1 : i + 1]))
        ll = float(np.min(l[i - period + 1 : i + 1]))
        out[i] = -100.0 * (hh - c[i]) / (hh - ll) if (hh - ll) > 1e-12 else -50.0
    return out


def ultimate_oscillator(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    p1: int = 7,
    p2: int = 14,
    p3: int = 28,
) -> np.ndarray:
    h, l, c = _to_f64(high), _to_f64(low), _to_f64(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= p3:
        return out
    bp = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        true_low = min(l[i], c[i - 1])
        true_high = max(h[i], c[i - 1])
        bp[i] = c[i] - true_low
        tr[i] = true_high - true_low

    for i in range(p3, n):
        avg1 = np.sum(bp[i - p1 + 1 : i + 1]) / max(np.sum(tr[i - p1 + 1 : i + 1]), 1e-12)
        avg2 = np.sum(bp[i - p2 + 1 : i + 1]) / max(np.sum(tr[i - p2 + 1 : i + 1]), 1e-12)
        avg3 = np.sum(bp[i - p3 + 1 : i + 1]) / max(np.sum(tr[i - p3 + 1 : i + 1]), 1e-12)
        out[i] = 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0
    return out


def trix(close: np.ndarray | list[float], period: int = 15) -> np.ndarray:
    c = _to_f64(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    e1 = ema(c, period)
    v1 = np.where(np.isfinite(e1))[0]
    if len(v1) < period:
        return out
    e2 = np.full(n, np.nan, dtype=np.float64)
    e2[v1[0] :] = ema(e1[v1[0] :], period)
    v2 = np.where(np.isfinite(e2))[0]
    if len(v2) < period:
        return out
    e3 = np.full(n, np.nan, dtype=np.float64)
    e3[v2[0] :] = ema(e2[v2[0] :], period)
    for i in range(1, n):
        if np.isfinite(e3[i]) and np.isfinite(e3[i - 1]) and e3[i - 1] != 0:
            out[i] = (e3[i] - e3[i - 1]) / e3[i - 1] * 100.0
    return out
