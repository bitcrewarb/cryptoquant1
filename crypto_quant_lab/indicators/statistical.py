"""Statistical, econometric, and time-series research models."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, stats
from statsmodels.tsa.stattools import acf, adfuller, coint, kpss, pacf


def _to_f64(arr: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def rolling_mean_std_zscore(values: np.ndarray | list[float], period: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _to_f64(values)
    n = len(x)
    mean = np.full(n, np.nan, dtype=np.float64)
    std = np.full(n, np.nan, dtype=np.float64)
    z = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        w = x[i - period + 1 : i + 1]
        mu = float(np.mean(w))
        sd = float(np.std(w, ddof=1))
        mean[i] = mu
        std[i] = sd
        z[i] = (x[i] - mu) / sd if sd > 1e-12 else 0.0
    return mean, std, z


def pearson_spearman_correlation(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> dict[str, float]:
    ax, ay = _to_f64(x), _to_f64(y)
    mask = np.isfinite(ax) & np.isfinite(ay)
    ax, ay = ax[mask], ay[mask]
    if len(ax) < 3:
        return {"pearson_r": 0.0, "pearson_p": 1.0, "spearman_rho": 0.0, "spearman_p": 1.0}
    pr, pp = stats.pearsonr(ax, ay)
    sr, sp = stats.spearmanr(ax, ay)
    return {
        "pearson_r": float(pr) if np.isfinite(pr) else 0.0,
        "pearson_p": float(pp) if np.isfinite(pp) else 1.0,
        "spearman_rho": float(sr) if np.isfinite(sr) else 0.0,
        "spearman_p": float(sp) if np.isfinite(sp) else 1.0,
    }


def autocorrelation_analysis(values: np.ndarray | list[float], nlags: int = 20) -> dict[str, list[float]]:
    x = _to_f64(values)
    x = x[np.isfinite(x)]
    max_lag = min(nlags, max(1, len(x) // 3 - 1))
    if len(x) < 10:
        return {"acf": [], "pacf": []}
    acf_vals = acf(x, nlags=max_lag, fft=True)
    pacf_vals = pacf(x, nlags=min(max_lag, len(x) // 2 - 1), method="ywm")
    return {"acf": [float(v) for v in acf_vals], "pacf": [float(v) for v in pacf_vals]}


def stationarity_tests(values: np.ndarray | list[float]) -> dict[str, Any]:
    x = _to_f64(values)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return {"adf_stat": 0.0, "adf_pvalue": 1.0, "is_stationary_adf": False, "kpss_stat": 0.0, "kpss_pvalue": 0.0}
    adf_res = adfuller(x, autolag="AIC")
    try:
        kpss_res = kpss(x, regression="c", nlags="auto")
        kpss_stat, kpss_p = float(kpss_res[0]), float(kpss_res[1])
    except Exception:
        kpss_stat, kpss_p = 0.0, 0.1
    return {
        "adf_stat": float(adf_res[0]),
        "adf_pvalue": float(adf_res[1]),
        "is_stationary_adf": bool(adf_res[1] < 0.05),
        "kpss_stat": kpss_stat,
        "kpss_pvalue": kpss_p,
        "is_stationary_kpss": bool(kpss_p >= 0.05),
    }


def engle_granger_cointegration(series_a: np.ndarray | list[float], series_b: np.ndarray | list[float]) -> dict[str, Any]:
    a, b = _to_f64(series_a), _to_f64(series_b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 30:
        return {"t_stat": 0.0, "pvalue": 1.0, "is_cointegrated": False, "hedge_ratio": 1.0}
    t_stat, pvalue, _ = coint(a, b)
    slope, intercept, r_val, _, _ = stats.linregress(b, a)
    spread = a - (slope * b + intercept)
    return {
        "t_stat": float(t_stat),
        "pvalue": float(pvalue),
        "is_cointegrated": bool(pvalue < 0.05),
        "hedge_ratio": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_val**2),
        "spread_std": float(np.std(spread, ddof=1)),
    }


def linear_regression_residuals(y: np.ndarray | list[float], x: np.ndarray | list[float] | None = None) -> dict[str, Any]:
    ay = _to_f64(y)
    ax = np.arange(len(ay), dtype=np.float64) if x is None else _to_f64(x)
    mask = np.isfinite(ay) & np.isfinite(ax)
    ay, ax = ay[mask], ax[mask]
    if len(ay) < 5:
        return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0, "residuals": []}
    res = stats.linregress(ax, ay)
    fitted = res.slope * ax + res.intercept
    residuals = ay - fitted
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r_squared": float(res.rvalue**2),
        "pvalue": float(res.pvalue),
        "residuals": [float(r) for r in residuals],
        "residual_std": float(np.std(residuals, ddof=1)),
    }


def kalman_filter_1d(
    values: np.ndarray | list[float],
    process_variance: float = 1e-4,
    measurement_variance: float = 1e-2,
) -> np.ndarray:
    """Strictly causal 1D Kalman state filter for price/spread denoising."""
    z = _to_f64(values)
    n = len(z)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    x_hat = z[0] if np.isfinite(z[0]) else 0.0
    p = 1.0
    for i in range(n):
        if not np.isfinite(z[i]):
            out[i] = x_hat
            continue
        p_pred = p + process_variance
        k = p_pred / (p_pred + measurement_variance)
        x_hat = x_hat + k * (z[i] - x_hat)
        p = (1.0 - k) * p_pred
        out[i] = x_hat
    return out


def hurst_exponent(close: np.ndarray | list[float], max_lag: int = 40) -> float:
    """Estimate Hurst exponent via R/S rescaled range on log prices."""
    c = _to_f64(close)
    c = c[np.isfinite(c) & (c > 0)]
    if len(c) < 60:
        return 0.5
    lags = range(2, min(max_lag, len(c) // 3))
    tau = [float(np.sqrt(np.std(np.subtract(c[lag:], c[:-lag])))) for lag in lags]
    valid = [(l, t) for l, t in zip(lags, tau) if t > 0]
    if len(valid) < 4:
        return 0.5
    x = np.log([v[0] for v in valid])
    y = np.log([v[1] for v in valid])
    poly = np.polyfit(x, y, 1)
    return float(np.clip(poly[0] * 2.0, 0.0, 1.0))


def shannon_entropy(returns: np.ndarray | list[float], bins: int = 16) -> float:
    r = _to_f64(returns)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return 0.0
    counts, _ = np.histogram(r, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def volatility_clustering_score(returns: np.ndarray | list[float], lag: int = 1) -> float:
    """Autocorrelation of squared returns measuring ARCH/volatility clustering."""
    r = _to_f64(returns)
    r = r[np.isfinite(r)]
    if len(r) < lag + 10:
        return 0.0
    sq = r**2
    x = sq[:-lag]
    y = sq[lag:]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def garch_11_volatility(
    returns: np.ndarray | list[float],
    annualization_periods: float = 365.0 * 24.0,
) -> dict[str, Any]:
    """Fit causal GARCH(1,1) conditional volatility via Quasi-MLE."""
    r = _to_f64(returns)
    r = np.where(np.isfinite(r), r, 0.0)
    n = len(r)
    if n < 50:
        return {
            "omega": 1e-6,
            "alpha": 0.05,
            "beta": 0.90,
            "conditional_vol": np.full(n, np.nan),
        }

    var_sample = float(np.var(r))
    if var_sample <= 1e-16:
        return {"omega": 0.0, "alpha": 0.0, "beta": 0.0, "conditional_vol": np.zeros(n)}

    def _neg_ll(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
            return 1e12
        sigma2 = np.empty(n, dtype=np.float64)
        sigma2[0] = var_sample
        for t in range(1, n):
            sigma2[t] = omega + alpha * (r[t - 1] ** 2) + beta * sigma2[t - 1]
        sigma2 = np.maximum(sigma2, 1e-12)
        return float(0.5 * np.sum(np.log(sigma2) + (r**2) / sigma2))

    x0 = np.array([var_sample * 0.05, 0.08, 0.87])
    res = optimize.minimize(
        _neg_ll,
        x0=x0,
        bounds=[(1e-12, var_sample * 2.0), (1e-4, 0.5), (0.1, 0.99)],
        method="L-BFGS-B",
    )
    omega, alpha, beta = res.x if res.success else x0
    if alpha + beta >= 0.999:
        scale = 0.98 / (alpha + beta)
        alpha *= scale
        beta *= scale

    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = var_sample
    for t in range(1, n):
        sigma2[t] = omega + alpha * (r[t - 1] ** 2) + beta * sigma2[t - 1]
    cond_vol = np.sqrt(np.maximum(sigma2, 0.0) * annualization_periods)
    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "persistence": float(alpha + beta),
        "conditional_vol": cond_vol,
    }
