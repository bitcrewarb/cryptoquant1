"""Cached feature computation and AST evaluation engine for high-throughput strategy screening."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.backtesting.event_driven.engine import run_event_driven_backtest
from crypto_quant_lab.backtesting.vectorized.engine import BacktestOutput, run_vectorized_backtest
from crypto_quant_lab.core.structures import DetectionResult, ZoneObject
from crypto_quant_lab.data.exchanges.base import TIMEFRAME_MS
from crypto_quant_lab.ict.engine import detect_ict
from crypto_quant_lab.indicators.momentum import cci, macd, roc, rsi, stochastic_rsi, williams_r
from crypto_quant_lab.indicators.trend import (
    adx,
    donchian_channels,
    ema,
    hma,
    ichimoku,
    linear_regression_slope,
    sma,
    supertrend,
    vwap,
)
from crypto_quant_lab.indicators.volatility import (
    atr,
    bollinger_bands,
    keltner_channels,
    volatility_percentile,
)
from crypto_quant_lab.indicators.volume import cumulative_volume_delta, relative_volume, volume_zscore
from crypto_quant_lab.indicators.statistical import kalman_filter_1d, rolling_mean_std_zscore
from crypto_quant_lab.price_action.engine import detect_price_action
from crypto_quant_lab.regimes.detector import detect_market_regimes
from crypto_quant_lab.smc.engine import detect_smc
from crypto_quant_lab.strategies.grammar import (
    ConditionNode,
    LeafCondition,
    LogicalExpr,
    StrategySpec,
    TemporalExpr,
    validate_strategy_spec,
)


class IndicatorCache:
    """Lazy-cached feature matrix per market dataset slice (strictly causal computations)."""

    def __init__(
        self,
        df: pl.DataFrame,
        timeframe: str = "1h",
        reference_df: pl.DataFrame | None = None,
    ) -> None:
        self.timeframe = timeframe
        self.n = df.height
        self.reference_df = reference_df
        self.open_time_ms = df["open_time_ms"].to_numpy()
        self.open = df["open"].to_numpy()
        self.high = df["high"].to_numpy()
        self.low = df["low"].to_numpy()
        self.close = df["close"].to_numpy()
        self.volume = df["volume"].to_numpy()
        self.taker_buy_vol = (
            df["taker_buy_volume"].to_numpy()
            if "taker_buy_volume" in df.columns
            else np.zeros(self.n, dtype=np.float64)
        )
        self._cache: dict[tuple[Any, ...], Any] = {}

    def get_ema(self, period: int) -> np.ndarray:
        key = ("ema", period)
        if key not in self._cache:
            self._cache[key] = ema(self.close, period)
        return self._cache[key]

    def get_sma(self, period: int) -> np.ndarray:
        key = ("sma", period)
        if key not in self._cache:
            self._cache[key] = sma(self.close, period)
        return self._cache[key]

    def get_hma(self, period: int) -> np.ndarray:
        key = ("hma", period)
        if key not in self._cache:
            self._cache[key] = hma(self.close, period)
        return self._cache[key]

    def get_vwap(self) -> np.ndarray:
        key = ("vwap",)
        if key not in self._cache:
            self._cache[key] = vwap(self.high, self.low, self.close, self.volume, self.open_time_ms)
        return self._cache[key]

    def get_adx(self, period: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = ("adx", period)
        if key not in self._cache:
            self._cache[key] = adx(self.high, self.low, self.close, period)
        return self._cache[key]

    def get_supertrend(self, period: int = 10, mult: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
        key = ("supertrend", period, mult)
        if key not in self._cache:
            self._cache[key] = supertrend(self.high, self.low, self.close, period, mult)
        return self._cache[key]

    def get_donchian(self, period: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = ("donchian", period)
        if key not in self._cache:
            self._cache[key] = donchian_channels(self.high, self.low, period)
        return self._cache[key]

    def get_ichimoku(self) -> dict[str, np.ndarray]:
        key = ("ichimoku",)
        if key not in self._cache:
            self._cache[key] = ichimoku(self.high, self.low, self.close)
        return self._cache[key]

    def get_linreg_slope(self, period: int = 20) -> np.ndarray:
        key = ("linreg_slope", period)
        if key not in self._cache:
            self._cache[key] = linear_regression_slope(self.close, period)
        return self._cache[key]

    def get_rsi(self, period: int = 14) -> np.ndarray:
        key = ("rsi", period)
        if key not in self._cache:
            self._cache[key] = rsi(self.close, period)
        return self._cache[key]

    def get_stoch_rsi(self, period: int = 14) -> tuple[np.ndarray, np.ndarray]:
        key = ("stoch_rsi", period)
        if key not in self._cache:
            self._cache[key] = stochastic_rsi(self.close, period, period)
        return self._cache[key]

    def get_macd(self, fast: int = 12, slow: int = 26, sig: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = ("macd", fast, slow, sig)
        if key not in self._cache:
            self._cache[key] = macd(self.close, fast, slow, sig)
        return self._cache[key]

    def get_cci(self, period: int = 20) -> np.ndarray:
        key = ("cci", period)
        if key not in self._cache:
            self._cache[key] = cci(self.high, self.low, self.close, period)
        return self._cache[key]

    def get_roc(self, period: int = 12) -> np.ndarray:
        key = ("roc", period)
        if key not in self._cache:
            self._cache[key] = roc(self.close, period)
        return self._cache[key]

    def get_williams_r(self, period: int = 14) -> np.ndarray:
        key = ("willr", period)
        if key not in self._cache:
            self._cache[key] = williams_r(self.high, self.low, self.close, period)
        return self._cache[key]

    def get_atr(self, period: int = 14) -> np.ndarray:
        key = ("atr", period)
        if key not in self._cache:
            self._cache[key] = atr(self.high, self.low, self.close, period)
        return self._cache[key]

    def get_bollinger(self, period: int = 20, std: float = 2.0) -> dict[str, np.ndarray]:
        key = ("bb", period, std)
        if key not in self._cache:
            self._cache[key] = bollinger_bands(self.close, period, std)
        return self._cache[key]

    def get_keltner(self, period: int = 20, mult: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = ("keltner", period, mult)
        if key not in self._cache:
            self._cache[key] = keltner_channels(self.high, self.low, self.close, period, max(7, period // 2), mult)
        return self._cache[key]

    def get_vol_pct(self, atr_period: int = 14) -> np.ndarray:
        key = ("vol_pct", atr_period)
        if key not in self._cache:
            a = self.get_atr(atr_period)
            norm = np.where(self.close > 0, a / self.close, np.nan)
            self._cache[key] = volatility_percentile(norm, lookback=100)
        return self._cache[key]

    def get_rvol(self, period: int = 20) -> np.ndarray:
        key = ("rvol", period)
        if key not in self._cache:
            self._cache[key] = relative_volume(self.volume, period)
        return self._cache[key]

    def get_vol_z(self, period: int = 20) -> np.ndarray:
        key = ("vol_z", period)
        if key not in self._cache:
            self._cache[key] = volume_zscore(self.volume, period)
        return self._cache[key]

    def get_cvd(self) -> np.ndarray:
        key = ("cvd",)
        if key not in self._cache:
            self._cache[key] = cumulative_volume_delta(self.volume, self.taker_buy_vol, self.close, self.open)
        return self._cache[key]

    def get_zscore(self, period: int = 20) -> np.ndarray:
        key = ("zscore", period)
        if key not in self._cache:
            _, _, z = rolling_mean_std_zscore(self.close, period)
            self._cache[key] = z
        return self._cache[key]

    def get_kalman_z(self, period: int = 20) -> np.ndarray:
        key = ("kalman_z", period)
        if key not in self._cache:
            kf = kalman_filter_1d(self.close)
            spread = self.close - kf
            _, _, z = rolling_mean_std_zscore(spread, period)
            self._cache[key] = z
        return self._cache[key]

    def get_pa_result(self, **params: Any) -> DetectionResult:
        allowed = {
            "swing_left",
            "swing_right",
            "sr_lookback",
            "atr_period",
            "cluster_tol_atr",
            "vp_lookback",
            "max_sr_zones",
        }
        clean = {k: params[k] for k in sorted(params) if k in allowed}
        key = ("pa_res", tuple(clean.items()))
        if key not in self._cache:
            self._cache[key] = detect_price_action(
                self.open,
                self.high,
                self.low,
                self.close,
                open_time_ms=self.open_time_ms,
                volume=self.volume,
                **clean,
            )
        return self._cache[key]

    def get_pa(self, **params: Any) -> dict[str, np.ndarray]:
        return self.get_pa_result(**params).arrays

    def get_smc_result(self, **params: Any) -> DetectionResult:
        allowed = {
            "int_swing",
            "ext_swing",
            "eq_tol_atr",
            "displacement_atr_mult",
            "min_fvg_atr_mult",
            "max_zone_age_bars",
            "trigger_on_retest",
            "ob_refined_mode",
            "ob_require_fvg",
        }
        clean = {k: params[k] for k in sorted(params) if k in allowed}
        key = ("smc_res", tuple(clean.items()))
        if key not in self._cache:
            self._cache[key] = detect_smc(
                self.open,
                self.high,
                self.low,
                self.close,
                open_time_ms=self.open_time_ms,
                volume=self.volume,
                **clean,
            )
        return self._cache[key]

    def get_smc(self, **params: Any) -> dict[str, np.ndarray]:
        return self.get_smc_result(**params).arrays

    def get_ict_result(self, **params: Any) -> DetectionResult:
        allowed = {
            "min_fvg_atr_mult",
            "max_zone_age_bars",
            "trigger_on_retest",
            "swing_len",
        }
        clean = {k: params[k] for k in sorted(params) if k in allowed}
        key = ("ict_res", tuple(clean.items()), self.reference_df is not None)
        if key not in self._cache:
            ref_h = self.reference_df["high"].to_numpy() if self.reference_df is not None else None
            ref_l = self.reference_df["low"].to_numpy() if self.reference_df is not None else None
            ref_t = self.reference_df["open_time_ms"].to_numpy() if self.reference_df is not None and "open_time_ms" in self.reference_df.columns else None
            self._cache[key] = detect_ict(
                self.open_time_ms,
                self.open,
                self.high,
                self.low,
                self.close,
                reference_high=ref_h,
                reference_low=ref_l,
                reference_open_time_ms=ref_t,
                **clean,
            )
        return self._cache[key]

    def get_ict(self, **params: Any) -> dict[str, np.ndarray]:
        return self.get_ict_result(**params).arrays

    def get_regimes(self) -> dict[str, Any]:
        key = ("regimes",)
        if key not in self._cache:
            self._cache[key] = detect_market_regimes(self.high, self.low, self.close, use_hmm=False)
        return self._cache[key]

    def get_htf_trend(self, factor: int = 4, ema_period: int = 20) -> np.ndarray:
        """
        Strictly causal timestamp-aligned Higher-Timeframe trend direction
        using ONLY completed HTF UTC candles (prefix-invariant, gap-safe).
        """
        key = ("htf_trend", factor, ema_period)
        if key not in self._cache:
            n = self.n
            out = np.zeros(n, dtype=np.int8)
            base_tf_ms = TIMEFRAME_MS.get(self.timeframe, 3_600_000)
            htf_ms = max(1, int(factor)) * base_tf_ms

            htf_closes: list[float] = []
            htf_ema_prev = np.nan
            alpha = 2.0 / (ema_period + 1.0)
            cur_dir = 0

            cur_bucket: int | None = None
            finalized_bucket: int | None = None
            prev_close = np.nan

            def _push_completed_htf(hc: float) -> None:
                nonlocal htf_ema_prev, cur_dir
                htf_closes.append(hc)
                if len(htf_closes) == ema_period:
                    htf_ema_prev = float(np.mean(htf_closes))
                elif len(htf_closes) > ema_period:
                    htf_ema_prev = alpha * hc + (1.0 - alpha) * htf_ema_prev
                if np.isfinite(htf_ema_prev):
                    cur_dir = 1 if hc > htf_ema_prev else -1

            for i in range(n):
                ts = int(self.open_time_ms[i])
                bucket = ts // htf_ms

                # If a gap skipped the closing bar of cur_bucket, finalize it now using prev_close
                if cur_bucket is not None and bucket != cur_bucket and finalized_bucket != cur_bucket:
                    if np.isfinite(prev_close):
                        _push_completed_htf(prev_close)
                    finalized_bucket = cur_bucket

                cur_bucket = bucket
                # If this bar sits at the closing boundary of the HTF UTC candle, finalize immediately
                if (ts + base_tf_ms) % htf_ms == 0:
                    _push_completed_htf(float(self.close[i]))
                    finalized_bucket = bucket

                prev_close = float(self.close[i])
                out[i] = cur_dir

            self._cache[key] = out
        return self._cache[key]


def _cross_above(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    out = np.zeros(n, dtype=bool)
    if n < 2:
        return out
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return np.nan_to_num(out, nan=False)


def _cross_below(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    out = np.zeros(n, dtype=bool)
    if n < 2:
        return out
    out[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return np.nan_to_num(out, nan=False)


def _eval_zone_state_leaf(leaf: LeafCondition, ctx: IndicatorCache) -> np.ndarray:
    """Strictly causal state-aware evaluation over ZoneObject lifecycles (zero look-ahead)."""
    n = ctx.n
    out = np.zeros(n, dtype=bool)
    p = leaf.params
    src = str(p.get("source", "smc"))
    ztype = p.get("zone_type")
    direction = p.get("direction")
    max_age = int(p.get("max_age", 100))
    fresh_only = bool(p.get("fresh_only", leaf.signal == "touching_fresh_zone"))

    if src == "ict":
        res = ctx.get_ict_result()
    elif src in ("pa", "price_action"):
        res = ctx.get_pa_result()
    else:
        res = ctx.get_smc_result()

    candidates: list[ZoneObject] = [
        z for z in res.zones
        if (ztype is None or z.type == ztype) and (direction is None or z.direction == direction)
    ]
    if not candidates:
        return out

    # Index zones by created_bar for O(N * active_zones) strictly causal scan
    by_created: dict[int, list[ZoneObject]] = {}
    for z in candidates:
        by_created.setdefault(z.created_bar, []).append(z)

    # Active tuples: [ZoneObject, touches_so_far]
    active: list[list[Any]] = []
    h, l, c = ctx.high, ctx.low, ctx.close

    for i in range(n):
        surv: list[list[Any]] = []
        for item in active:
            z: ZoneObject = item[0]
            touches: int = item[1]
            if z.expiry is not None and i > z.expiry:
                continue
            if (i - z.created_bar) >= max_age:
                continue
            # Causal invalidation at prior close or current close
            if z.direction == "bull" and i > z.created_bar + 1 and c[i - 1] < z.bottom:
                continue
            if z.direction == "bear" and i > z.created_bar + 1 and c[i - 1] > z.top:
                continue

            if leaf.signal == "zone_age_lt":
                out[i] = True
                surv.append(item)
                continue

            is_touch = bool(l[i] <= z.top and h[i] >= z.bottom)
            if is_touch:
                if (not fresh_only) or touches == 0:
                    out[i] = True
                item[1] = touches + 1
            if not (fresh_only and item[1] > 0):
                surv.append(item)
            elif not fresh_only:
                surv.append(item)
        active = surv[-24:]
        if i in by_created:
            for z_new in by_created[i]:
                active.append([z_new, 0])
    return out


def evaluate_leaf(leaf: LeafCondition, ctx: IndicatorCache) -> np.ndarray:
    sig = leaf.signal
    p = leaf.params
    c = ctx.close
    n = ctx.n

    # Trend
    if sig == "ema_gt_ema":
        return np.nan_to_num(ctx.get_ema(int(p["fast"])) > ctx.get_ema(int(p["slow"])), nan=False)
    if sig == "ema_lt_ema":
        return np.nan_to_num(ctx.get_ema(int(p["fast"])) < ctx.get_ema(int(p["slow"])), nan=False)
    if sig == "ema_cross_above":
        return _cross_above(ctx.get_ema(int(p["fast"])), ctx.get_ema(int(p["slow"])))
    if sig == "ema_cross_below":
        return _cross_below(ctx.get_ema(int(p["fast"])), ctx.get_ema(int(p["slow"])))
    if sig == "close_gt_ema":
        return np.nan_to_num(c > ctx.get_ema(int(p["period"])), nan=False)
    if sig == "close_lt_ema":
        return np.nan_to_num(c < ctx.get_ema(int(p["period"])), nan=False)
    if sig == "close_gt_sma":
        return np.nan_to_num(c > ctx.get_sma(int(p["period"])), nan=False)
    if sig == "close_lt_sma":
        return np.nan_to_num(c < ctx.get_sma(int(p["period"])), nan=False)
    if sig == "hma_slope_up":
        h = ctx.get_hma(int(p["period"]))
        out = np.zeros(n, dtype=bool)
        out[1:] = np.isfinite(h[1:]) & np.isfinite(h[:-1]) & (h[1:] > h[:-1])
        return out
    if sig == "hma_slope_down":
        h = ctx.get_hma(int(p["period"]))
        out = np.zeros(n, dtype=bool)
        out[1:] = np.isfinite(h[1:]) & np.isfinite(h[:-1]) & (h[1:] < h[:-1])
        return out
    if sig == "close_gt_vwap":
        return np.nan_to_num(c > ctx.get_vwap(), nan=False)
    if sig == "close_lt_vwap":
        return np.nan_to_num(c < ctx.get_vwap(), nan=False)
    if sig == "adx_above":
        a, _, _ = ctx.get_adx(int(p.get("period", 14)))
        return np.nan_to_num(a > float(p.get("threshold", 25)), nan=False)
    if sig == "supertrend_bull":
        _, d = ctx.get_supertrend(int(p.get("period", 10)), float(p.get("mult", 3.0)))
        return d == 1
    if sig == "supertrend_bear":
        _, d = ctx.get_supertrend(int(p.get("period", 10)), float(p.get("mult", 3.0)))
        return d == -1
    if sig == "donchian_break_up":
        upper, _, _ = ctx.get_donchian(int(p.get("period", 20)))
        out = np.zeros(n, dtype=bool)
        out[1:] = np.isfinite(upper[:-1]) & (c[1:] > upper[:-1])
        return out
    if sig == "donchian_break_down":
        _, _, lower = ctx.get_donchian(int(p.get("period", 20)))
        out = np.zeros(n, dtype=bool)
        out[1:] = np.isfinite(lower[:-1]) & (c[1:] < lower[:-1])
        return out
    if sig == "ichimoku_bull":
        ich = ctx.get_ichimoku()
        return np.nan_to_num((c > ich["senkou_span_a"]) & (c > ich["senkou_span_b"]) & (ich["tenkan_sen"] > ich["kijun_sen"]), nan=False)
    if sig == "ichimoku_bear":
        ich = ctx.get_ichimoku()
        return np.nan_to_num((c < ich["senkou_span_a"]) & (c < ich["senkou_span_b"]) & (ich["tenkan_sen"] < ich["kijun_sen"]), nan=False)
    if sig == "linreg_slope_pos":
        return np.nan_to_num(ctx.get_linreg_slope(int(p.get("period", 20))) > 0, nan=False)
    if sig == "linreg_slope_neg":
        return np.nan_to_num(ctx.get_linreg_slope(int(p.get("period", 20))) < 0, nan=False)

    # Momentum
    if sig == "rsi_below":
        return np.nan_to_num(ctx.get_rsi(int(p.get("period", 14))) < float(p.get("threshold", 30)), nan=False)
    if sig == "rsi_above":
        return np.nan_to_num(ctx.get_rsi(int(p.get("period", 14))) > float(p.get("threshold", 70)), nan=False)
    if sig == "stoch_rsi_bull":
        k, d = ctx.get_stoch_rsi(int(p.get("period", 14)))
        return np.nan_to_num((k > d) & (k < 40.0), nan=False)
    if sig == "stoch_rsi_bear":
        k, d = ctx.get_stoch_rsi(int(p.get("period", 14)))
        return np.nan_to_num((k < d) & (k > 60.0), nan=False)
    if sig == "macd_bull":
        m, s, _ = ctx.get_macd()
        return np.nan_to_num(m > s, nan=False)
    if sig == "macd_bear":
        m, s, _ = ctx.get_macd()
        return np.nan_to_num(m < s, nan=False)
    if sig == "cci_below":
        return np.nan_to_num(ctx.get_cci(int(p.get("period", 20))) < float(p.get("threshold", -100)), nan=False)
    if sig == "cci_above":
        return np.nan_to_num(ctx.get_cci(int(p.get("period", 20))) > float(p.get("threshold", 100)), nan=False)
    if sig == "roc_above":
        return np.nan_to_num(ctx.get_roc(int(p.get("period", 12))) > float(p.get("threshold", 0.5)), nan=False)
    if sig == "roc_below":
        return np.nan_to_num(ctx.get_roc(int(p.get("period", 12))) < float(p.get("threshold", -0.5)), nan=False)
    if sig == "willr_below":
        return np.nan_to_num(ctx.get_williams_r(int(p.get("period", 14))) < float(p.get("threshold", -80)), nan=False)
    if sig == "willr_above":
        return np.nan_to_num(ctx.get_williams_r(int(p.get("period", 14))) > float(p.get("threshold", -20)), nan=False)

    # Volatility
    if sig == "bb_below_lower":
        bb = ctx.get_bollinger(int(p.get("period", 20)), float(p.get("std", 2.0)))
        return np.nan_to_num(c < bb["lower"], nan=False)
    if sig == "bb_above_upper":
        bb = ctx.get_bollinger(int(p.get("period", 20)), float(p.get("std", 2.0)))
        return np.nan_to_num(c > bb["upper"], nan=False)
    if sig == "bb_squeeze":
        bb = ctx.get_bollinger(int(p.get("period", 20)), float(p.get("std", 2.0)))
        bw = bb["bandwidth"]
        out = np.zeros(n, dtype=bool)
        for i in range(n):
            if not np.isfinite(bw[i]):
                continue
            w = bw[max(0, i - 99) : i + 1]
            fin = w[np.isfinite(w)]
            if len(fin) >= 5:
                out[i] = bool(bw[i] < 0.8 * float(np.median(fin)))
        return out
    if sig == "keltner_break_up":
        u, _, _ = ctx.get_keltner(int(p.get("period", 20)), float(p.get("mult", 2.0)))
        return np.nan_to_num(c > u, nan=False)
    if sig == "keltner_break_down":
        _, _, l = ctx.get_keltner(int(p.get("period", 20)), float(p.get("mult", 2.0)))
        return np.nan_to_num(c < l, nan=False)
    if sig == "vol_pct_above":
        return np.nan_to_num(ctx.get_vol_pct(int(p.get("atr_period", 14))) > float(p.get("threshold", 50)), nan=False)
    if sig == "vol_pct_below":
        return np.nan_to_num(ctx.get_vol_pct(int(p.get("atr_period", 14))) < float(p.get("threshold", 60)), nan=False)

    # Volume
    if sig == "rvol_above":
        return np.nan_to_num(ctx.get_rvol(int(p.get("period", 20))) > float(p.get("threshold", 1.2)), nan=False)
    if sig == "vol_z_above":
        return np.nan_to_num(ctx.get_vol_z(int(p.get("period", 20))) > float(p.get("threshold", 1.0)), nan=False)
    if sig == "cvd_rising":
        cvd = ctx.get_cvd()
        lb = int(p.get("lookback", 5))
        out = np.zeros(n, dtype=bool)
        if n > lb:
            out[lb:] = cvd[lb:] > cvd[:-lb]
        return out
    if sig == "cvd_falling":
        cvd = ctx.get_cvd()
        lb = int(p.get("lookback", 5))
        out = np.zeros(n, dtype=bool)
        if n > lb:
            out[lb:] = cvd[lb:] < cvd[:-lb]
        return out

    # Statistical
    if sig == "zscore_below":
        return np.nan_to_num(ctx.get_zscore(int(p.get("period", 20))) < float(p.get("threshold", -1.5)), nan=False)
    if sig == "zscore_above":
        return np.nan_to_num(ctx.get_zscore(int(p.get("period", 20))) > float(p.get("threshold", 1.5)), nan=False)
    if sig == "kalman_z_below":
        return np.nan_to_num(ctx.get_kalman_z(int(p.get("period", 20))) < float(p.get("threshold", -1.5)), nan=False)
    if sig == "kalman_z_above":
        return np.nan_to_num(ctx.get_kalman_z(int(p.get("period", 20))) > float(p.get("threshold", 1.5)), nan=False)

    # Price Action
    if leaf.category == "price_action":
        pa = ctx.get_pa(**p)
        if sig in pa:
            return np.asarray(pa[sig], dtype=bool)

    # SMC
    if leaf.category == "smc":
        smc = ctx.get_smc(**p)
        if sig in smc:
            return np.asarray(smc[sig], dtype=bool)

    # ICT
    if leaf.category == "ict":
        ict = ctx.get_ict(**p)
        if sig == "daily_bias_bull":
            return ict["daily_bias"] == 1
        if sig == "daily_bias_bear":
            return ict["daily_bias"] == -1
        if sig == "weekly_bias_bull":
            return ict["weekly_bias"] == 1
        if sig == "weekly_bias_bear":
            return ict["weekly_bias"] == -1
        if sig in ict:
            return np.asarray(ict[sig], dtype=bool)

    # Regime
    if sig == "regime_in":
        allowed = set(p.get("regimes", []))
        regs = ctx.get_regimes()["primary_regime"]
        return np.array([r in allowed for r in regs], dtype=bool)

    # Multi-Timeframe
    if sig == "htf_bull":
        return ctx.get_htf_trend(int(p.get("factor", 4)), int(p.get("ema_period", 20))) == 1
    if sig == "htf_bear":
        return ctx.get_htf_trend(int(p.get("factor", 4)), int(p.get("ema_period", 20))) == -1

    # State-aware zone lifecycle queries
    if leaf.category == "zone_state":
        return _eval_zone_state_leaf(leaf, ctx)

    return np.zeros(n, dtype=bool)


def evaluate_condition(node: ConditionNode | None, ctx: IndicatorCache) -> np.ndarray:
    if node is None:
        return np.zeros(ctx.n, dtype=bool)
    if isinstance(node, LeafCondition):
        return evaluate_leaf(node, ctx)
    if isinstance(node, LogicalExpr):
        if node.op == "NOT":
            return ~evaluate_condition(node.children[0], ctx)
        if node.op == "AND":
            acc = np.ones(ctx.n, dtype=bool)
            for ch in node.children:
                acc &= evaluate_condition(ch, ctx)
            return acc
        if node.op == "OR":
            acc = np.zeros(ctx.n, dtype=bool)
            for ch in node.children:
                acc |= evaluate_condition(ch, ctx)
            return acc
        return np.zeros(ctx.n, dtype=bool)
    if isinstance(node, TemporalExpr):
        n = ctx.n
        w = max(1, int(node.window_bars))
        if node.op == "WITHIN":
            ch = evaluate_condition(node.children[0], ctx)
            out = np.zeros(n, dtype=bool)
            csum = np.cumsum(ch.astype(np.int32))
            for i in range(n):
                prev_sum = csum[i - w] if i >= w else 0
                out[i] = (csum[i] - prev_sum) > 0
            return out
        if node.op == "SEQUENCE":
            step_arrays = [evaluate_condition(ch, ctx) for ch in node.children]
            m = len(step_arrays)
            out = np.zeros(n, dtype=bool)
            if m == 0:
                return out
            last_bar = [-10_000_000] * m
            for i in range(n):
                # Reverse stage update guarantees strict causal ordering b_0 < b_1 < ... < b_{m-1} == i
                for s in range(m - 1, 0, -1):
                    if step_arrays[s][i] and 1 <= (i - last_bar[s - 1]) <= w:
                        last_bar[s] = i
                        if s == m - 1:
                            out[i] = True
                if step_arrays[0][i]:
                    last_bar[0] = i
            return out
    return np.zeros(ctx.n, dtype=bool)


def _compute_structural_execution_arrays(
    spec: StrategySpec,
    ctx: IndicatorCache,
    atr_arr: np.ndarray,
) -> dict[str, np.ndarray | None]:
    """Compute strictly causal per-bar limit prices, structural SL/TP levels, and session exits."""
    n = ctx.n
    needs_structural = (
        spec.entry_mode != "market"
        or spec.sl_mode != "atr"
        or spec.tp_mode != "atr"
        or spec.session_stop
    )
    if not needs_structural:
        return {
            "long_limit_prices": None,
            "short_limit_prices": None,
            "long_sl_prices": None,
            "short_sl_prices": None,
            "long_tp_prices": None,
            "short_tp_prices": None,
            "session_exit": None,
        }

    smc_res = ctx.get_smc_result()
    ict_res = ctx.get_ict_result()
    pa_res = ctx.get_pa_result()
    smc = smc_res.arrays
    ict = ict_res.arrays

    all_zones = smc_res.zones + [z for z in pa_res.zones if z.type == "sr"]
    bull_by_bar: dict[int, list[ZoneObject]] = {}
    bear_by_bar: dict[int, list[ZoneObject]] = {}
    for z in all_zones:
        if z.direction == "bull":
            bull_by_bar.setdefault(z.created_bar, []).append(z)
        elif z.direction == "bear":
            bear_by_bar.setdefault(z.created_bar, []).append(z)

    l_lim = np.full(n, np.nan, dtype=np.float64)
    s_lim = np.full(n, np.nan, dtype=np.float64)
    l_sl = np.full(n, np.nan, dtype=np.float64)
    s_sl = np.full(n, np.nan, dtype=np.float64)
    l_tp = np.full(n, np.nan, dtype=np.float64)
    s_tp = np.full(n, np.nan, dtype=np.float64)
    sess_exit = np.zeros(n, dtype=bool) if spec.session_stop else None

    active_bull: list[ZoneObject] = []
    active_bear: list[ZoneObject] = []

    h, l, c = ctx.high, ctx.low, ctx.close
    eq_arr = smc["equilibrium"]
    dr_arr = smc["dealing_range_pos"]

    for i in range(n):
        a = atr_arr[i] if np.isfinite(atr_arr[i]) and atr_arr[i] > 0 else c[i] * 0.015
        active_bull = [z for z in active_bull if (z.expiry is None or i <= z.expiry) and c[i] >= z.bottom][-8:]
        active_bear = [z for z in active_bear if (z.expiry is None or i <= z.expiry) and c[i] <= z.top][-8:]

        # Entry limits anchored below/above current close strictly causally
        if spec.entry_mode in ("zone_edge", "zone_50"):
            if active_bull:
                z = active_bull[-1]
                px = z.top if spec.entry_mode == "zone_edge" else z.midpoint
                if 0 < px <= c[i]:
                    l_lim[i] = px
            if active_bear:
                z = active_bear[-1]
                px = z.bottom if spec.entry_mode == "zone_edge" else z.midpoint
                if px >= c[i]:
                    s_lim[i] = px
        elif spec.entry_mode in ("ote_62", "ote_70") and np.isfinite(eq_arr[i]):
            ratio = 0.62 if spec.entry_mode == "ote_62" else 0.70
            # Reconstruct current dealing range high/low causally from eq & dr_pos when spread > 0
            pos = dr_arr[i]
            if abs(pos - 0.5) > 1e-4:
                half_rng = abs(c[i] - eq_arr[i]) / abs(pos - 0.5)
                if half_rng > 0:
                    dr_hi = eq_arr[i] + half_rng
                    dr_lo = eq_arr[i] - half_rng
                    ote_l = dr_hi - ratio * (dr_hi - dr_lo)
                    ote_s = dr_lo + ratio * (dr_hi - dr_lo)
                    if 0 < ote_l <= c[i]:
                        l_lim[i] = ote_l
                    if ote_s >= c[i]:
                        s_lim[i] = ote_s

        # Structural SL
        if spec.sl_mode == "sweep_wick":
            lo_w = float(np.min(l[max(0, i - 4) : i + 1]))
            hi_w = float(np.max(h[max(0, i - 4) : i + 1]))
            l_sl[i] = lo_w - spec.sl_atr_buffer * a
            s_sl[i] = hi_w + spec.sl_atr_buffer * a
        elif spec.sl_mode == "zone_invalidation":
            if active_bull:
                l_sl[i] = active_bull[-1].bottom - spec.sl_atr_buffer * a
            if active_bear:
                s_sl[i] = active_bear[-1].top + spec.sl_atr_buffer * a

        # Structural TP
        anchor_l = l_lim[i] if np.isfinite(l_lim[i]) else c[i]
        anchor_s = s_lim[i] if np.isfinite(s_lim[i]) else c[i]
        risk_l = max(anchor_l - l_sl[i], 0.5 * a) if np.isfinite(l_sl[i]) and l_sl[i] < anchor_l else max(spec.sl_atr_mult * a, anchor_l * 0.002)
        risk_s = max(s_sl[i] - anchor_s, 0.5 * a) if np.isfinite(s_sl[i]) and s_sl[i] > anchor_s else max(spec.sl_atr_mult * a, anchor_s * 0.002)

        if spec.tp_mode == "opposing_liquidity":
            dol_b = ict["dol_bull_target"][i]
            dol_s = ict["dol_bear_target"][i]
            if np.isfinite(dol_b) and dol_b > anchor_l + 0.8 * risk_l:
                l_tp[i] = float(dol_b)
            else:
                l_tp[i] = anchor_l + spec.tp_rr_mult * risk_l
            if np.isfinite(dol_s) and dol_s < anchor_s - 0.8 * risk_s:
                s_tp[i] = float(dol_s)
            else:
                s_tp[i] = anchor_s - spec.tp_rr_mult * risk_s
        elif spec.tp_mode == "rr_multiple":
            l_tp[i] = anchor_l + spec.tp_rr_mult * risk_l
            s_tp[i] = anchor_s - spec.tp_rr_mult * risk_s

        # Session exit right when a NY / London kill zone ends
        if sess_exit is not None and i >= 1:
            in_kz_prev = bool(ict["london_kill_zone"][i - 1] or ict["ny_am_kill_zone"][i - 1] or ict["ny_pm_kill_zone"][i - 1])
            in_kz_now = bool(ict["london_kill_zone"][i] or ict["ny_am_kill_zone"][i] or ict["ny_pm_kill_zone"][i])
            if in_kz_prev and not in_kz_now:
                sess_exit[i] = True

        if i in bull_by_bar:
            active_bull.extend(bull_by_bar[i])
        if i in bear_by_bar:
            active_bear.extend(bear_by_bar[i])

    return {
        "long_limit_prices": l_lim if spec.entry_mode != "market" else None,
        "short_limit_prices": s_lim if spec.entry_mode != "market" else None,
        "long_sl_prices": l_sl if spec.sl_mode != "atr" else None,
        "short_sl_prices": s_sl if spec.sl_mode != "atr" else None,
        "long_tp_prices": l_tp if spec.tp_mode != "atr" else None,
        "short_tp_prices": s_tp if spec.tp_mode != "atr" else None,
        "session_exit": sess_exit,
    }


def evaluate_strategy(
    spec: StrategySpec,
    ctx: IndicatorCache,
    engine_type: str = "vectorized",
    initial_capital: float = 100_000.0,
    costs: ExecutionCostConfig | None = None,
    funding_rates_map: dict[int, float] | None = None,
) -> BacktestOutput:
    """Evaluate AST signals on IndicatorCache and execute vectorized or event-driven backtest."""
    cost_cfg = costs or ExecutionCostConfig()
    validate_strategy_spec(spec, market_type=cost_cfg.market_type)

    allow_long = spec.direction in ("long_only", "long_short")
    allow_short = spec.direction in ("short_only", "long_short")

    long_entry = evaluate_condition(spec.long_entry, ctx) if allow_long else np.zeros(ctx.n, dtype=bool)
    long_exit = evaluate_condition(spec.long_exit, ctx) if allow_long else np.zeros(ctx.n, dtype=bool)
    short_entry = evaluate_condition(spec.short_entry, ctx) if allow_short else np.zeros(ctx.n, dtype=bool)
    short_exit = evaluate_condition(spec.short_exit, ctx) if allow_short else np.zeros(ctx.n, dtype=bool)

    atr_arr = ctx.get_atr(spec.atr_period)
    regimes = ctx.get_regimes()["primary_regime"]

    if engine_type == "event_driven":
        struct_arrays = _compute_structural_execution_arrays(spec, ctx, atr_arr)
        return run_event_driven_backtest(
            open_time_ms=ctx.open_time_ms,
            open_=ctx.open,
            high=ctx.high,
            low=ctx.low,
            close=ctx.close,
            volume=ctx.volume,
            long_entry=long_entry,
            long_exit=long_exit,
            short_entry=short_entry,
            short_exit=short_exit,
            atr_arr=atr_arr,
            regimes=regimes,
            timeframe=ctx.timeframe,
            initial_capital=initial_capital,
            risk_per_trade_pct=spec.risk_per_trade_pct,
            sl_atr_mult=spec.sl_atr_mult,
            tp_atr_mult=spec.tp_atr_mult,
            trailing_stop_atr=spec.trailing_stop_atr,
            max_holding_bars=spec.max_holding_bars,
            costs=cost_cfg,
            allow_long=allow_long,
            allow_short=allow_short,
            funding_rates_map=funding_rates_map,
            entry_ttl_bars=spec.entry_ttl_bars,
            partial_tp_ratio=spec.partial_tp_ratio,
            partial_tp_rr=spec.partial_tp_rr,
            move_to_be_after_tp1=spec.move_to_be_after_tp1,
            **struct_arrays,
        )

    return run_vectorized_backtest(
        open_time_ms=ctx.open_time_ms,
        open_=ctx.open,
        high=ctx.high,
        low=ctx.low,
        close=ctx.close,
        volume=ctx.volume,
        long_entry=long_entry,
        long_exit=long_exit,
        short_entry=short_entry,
        short_exit=short_exit,
        atr_arr=atr_arr,
        regimes=regimes,
        timeframe=ctx.timeframe,
        initial_capital=initial_capital,
        risk_per_trade_pct=spec.risk_per_trade_pct,
        sl_atr_mult=spec.sl_atr_mult,
        tp_atr_mult=spec.tp_atr_mult,
        max_holding_bars=spec.max_holding_bars,
        costs=cost_cfg,
        allow_long=allow_long,
        allow_short=allow_short,
        funding_rates_map=funding_rates_map,
    )


def screen_and_verify_strategies(
    specs: list[StrategySpec],
    ctx: IndicatorCache,
    top_k: int = 10,
    initial_capital: float = 100_000.0,
    costs: ExecutionCostConfig | None = None,
    funding_rates_map: dict[int, float] | None = None,
) -> dict[str, Any]:
    """
    Two-stage evaluation invariant:
    1) Vectorized engine rapidly pre-filters candidates and counts EVERY attempted trial.
    2) Top-K surviving candidates are re-run on the full event-driven simulator with structural limits/exits.
    """
    cost_cfg = costs or ExecutionCostConfig()
    screened: list[tuple[float, StrategySpec, BacktestOutput]] = []
    attempted_trials = 0
    for sp in specs:
        attempted_trials += 1
        try:
            vec_out = evaluate_strategy(
                sp, ctx, engine_type="vectorized", initial_capital=initial_capital, costs=cost_cfg, funding_rates_map=funding_rates_map
            )
            m = vec_out.metrics
            score = float(m.get("sharpe_ratio", -99.0)) if int(m.get("total_trades", 0)) >= 3 else -999.0
            screened.append((score, sp, vec_out))
        except Exception:
            continue

    screened.sort(key=lambda x: x[0], reverse=True)
    verified: list[tuple[StrategySpec, BacktestOutput]] = []
    for _, sp, _ in screened[: max(1, top_k)]:
        ev_out = evaluate_strategy(
            sp, ctx, engine_type="event_driven", initial_capital=initial_capital, costs=cost_cfg, funding_rates_map=funding_rates_map
        )
        verified.append((sp, ev_out))

    return {
        "attempted_trials": attempted_trials,
        "screened_count": len(screened),
        "verified": verified,
    }
