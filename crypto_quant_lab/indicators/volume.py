"""Causal volume and order-flow proxy indicators."""

from __future__ import annotations

import numpy as np

from crypto_quant_lab.indicators.trend import sma


def _to_f64(arr: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def obv(close: np.ndarray | list[float], volume: np.ndarray | list[float]) -> np.ndarray:
    c, v = _to_f64(close), _to_f64(volume)
    n = len(c)
    out = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if c[i] > c[i - 1]:
            out[i] = out[i - 1] + v[i]
        elif c[i] < c[i - 1]:
            out[i] = out[i - 1] - v[i]
        else:
            out[i] = out[i - 1]
    return out


def volume_sma(volume: np.ndarray | list[float], period: int = 20) -> np.ndarray:
    return sma(volume, period)


def volume_zscore(volume: np.ndarray | list[float], period: int = 20) -> np.ndarray:
    v = _to_f64(volume)
    n = len(v)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        w = v[i - period + 1 : i + 1]
        mu = float(np.mean(w))
        sd = float(np.std(w, ddof=0))
        out[i] = (v[i] - mu) / sd if sd > 1e-12 else 0.0
    return out


def relative_volume(volume: np.ndarray | list[float], period: int = 20) -> np.ndarray:
    v = _to_f64(volume)
    vsma = sma(v, period)
    return np.where(vsma > 0, v / vsma, np.nan)


def cumulative_volume_delta(
    volume: np.ndarray | list[float],
    taker_buy_volume: np.ndarray | list[float] | None = None,
    close: np.ndarray | list[float] | None = None,
    open_: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Compute CVD from real taker buy/sell volume when available, else candle direction approximation."""
    v = _to_f64(volume)
    if taker_buy_volume is not None:
        tbv = _to_f64(taker_buy_volume)
        if np.any(tbv > 0):
            sell_vol = np.maximum(v - tbv, 0.0)
            delta = tbv - sell_vol
            return np.cumsum(delta)
    if close is not None and open_ is not None:
        c, o = _to_f64(close), _to_f64(open_)
        sign = np.where(c >= o, 1.0, -1.0)
        return np.cumsum(v * sign)
    return np.zeros(len(v), dtype=np.float64)


def buying_selling_pressure(
    open_: np.ndarray | list[float],
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    volume: np.ndarray | list[float],
    taker_buy_volume: np.ndarray | list[float] | None = None,
) -> dict[str, np.ndarray]:
    o, h, l, c, v = _to_f64(open_), _to_f64(high), _to_f64(low), _to_f64(close), _to_f64(volume)
    if taker_buy_volume is not None and np.any(_to_f64(taker_buy_volume) > 0):
        buy_v = _to_f64(taker_buy_volume)
        sell_v = np.maximum(v - buy_v, 0.0)
    else:
        rng = np.maximum(h - l, 1e-12)
        buy_ratio = np.clip((c - l) / rng, 0.0, 1.0)
        buy_v = v * buy_ratio
        sell_v = v - buy_v
    imbalance = np.where(v > 0, (buy_v - sell_v) / v, 0.0)
    return {
        "buying_volume": buy_v,
        "selling_volume": sell_v,
        "volume_imbalance": imbalance,
    }


def volume_profile(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    volume: np.ndarray | list[float],
    bins: int = 24,
) -> dict[str, float | list[dict[str, float]]]:
    """Compute Volume Profile (POC, Value Area High/Low, histogram) on a historical window."""
    h, l, c, v = _to_f64(high), _to_f64(low), _to_f64(close), _to_f64(volume)
    if len(c) == 0 or np.sum(v) <= 0:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "bins": []}

    p_min = float(np.min(l))
    p_max = float(np.max(h))
    if p_max <= p_min:
        return {"poc": p_min, "vah": p_min, "val": p_min, "bins": []}

    edges = np.linspace(p_min, p_max, bins + 1)
    hist = np.zeros(bins, dtype=np.float64)
    tp = (h + l + c) / 3.0
    bin_idx = np.clip(np.digitize(tp, edges) - 1, 0, bins - 1)
    for idx, vol in zip(bin_idx, v):
        hist[idx] += vol

    poc_idx = int(np.argmax(hist))
    poc_price = float((edges[poc_idx] + edges[poc_idx + 1]) / 2.0)

    total_vol = float(np.sum(hist))
    target_vol = 0.70 * total_vol
    lo_idx = poc_idx
    hi_idx = poc_idx
    acc_vol = float(hist[poc_idx])
    while acc_vol < target_vol and (lo_idx > 0 or hi_idx < bins - 1):
        left_v = hist[lo_idx - 1] if lo_idx > 0 else -1.0
        right_v = hist[hi_idx + 1] if hi_idx < bins - 1 else -1.0
        if right_v >= left_v and hi_idx < bins - 1:
            hi_idx += 1
            acc_vol += float(hist[hi_idx])
        elif lo_idx > 0:
            lo_idx -= 1
            acc_vol += float(hist[lo_idx])
        else:
            break

    bin_rows = [
        {"price_low": float(edges[i]), "price_high": float(edges[i + 1]), "volume": float(hist[i])}
        for i in range(bins)
    ]
    return {
        "poc": poc_price,
        "vah": float(edges[hi_idx + 1]),
        "val": float(edges[lo_idx]),
        "bins": bin_rows,
    }
