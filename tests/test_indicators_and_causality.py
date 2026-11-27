"""Unit, property-based (Hypothesis), and strict look-ahead bias invariance tests for quantitative modules."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from crypto_quant_lab.ict.engine import compute_ict_features
from crypto_quant_lab.indicators.momentum import cci, macd, momentum, roc, rsi, stochastic_rsi, trix, ultimate_oscillator, williams_r
from crypto_quant_lab.indicators.statistical import (
    autocorrelation_analysis,
    engle_granger_cointegration,
    garch_11_volatility,
    hurst_exponent,
    kalman_filter_1d,
    linear_regression_residuals,
    pearson_spearman_correlation,
    rolling_mean_std_zscore,
    shannon_entropy,
    stationarity_tests,
    volatility_clustering_score,
)
from crypto_quant_lab.indicators.trend import (
    adx,
    anchored_vwap,
    donchian_channels,
    ema,
    hma,
    ichimoku,
    linear_regression_slope,
    sma,
    supertrend,
    vwap,
    wma,
)
from crypto_quant_lab.indicators.volatility import (
    atr,
    bollinger_bands,
    donchian_width,
    garman_klass_volatility,
    historical_volatility,
    keltner_channels,
    parkinson_volatility,
    volatility_percentile,
)
from crypto_quant_lab.indicators.volume import buying_selling_pressure, cumulative_volume_delta, obv, relative_volume, volume_profile, volume_zscore
from crypto_quant_lab.price_action.engine import compute_price_action_features
from crypto_quant_lab.regimes.detector import ALL_REGIMES, detect_market_regimes
from crypto_quant_lab.smc.engine import compute_smc_features
from crypto_quant_lab.strategies.evaluator import IndicatorCache, evaluate_condition
from crypto_quant_lab.strategies.grammar import LeafCondition
from tests.conftest import make_synthetic_ohlcv


def test_all_indicators_compute_valid_shapes_and_ranges() -> None:
    df = make_synthetic_ohlcv(n=300, seed=7)
    t = df["open_time_ms"].to_numpy()
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    v = df["volume"].to_numpy()
    tbv = df["taker_buy_volume"].to_numpy()

    # Trend
    assert len(sma(c, 20)) == len(c)
    assert len(ema(c, 20)) == len(c)
    assert len(wma(c, 20)) == len(c)
    assert len(hma(c, 16)) == len(c)
    assert np.all(np.isfinite(vwap(h, l, c, v, t)))
    assert np.isfinite(anchored_vwap(h, l, c, v, anchor_index=50)[-1])
    adx_v, pdi, mdi = adx(h, l, c, 14)
    assert np.isfinite(adx_v[-1]) and 0.0 <= adx_v[-1] <= 100.0
    st_line, st_dir = supertrend(h, l, c, 10, 3.0)
    assert set(np.unique(st_dir[st_dir != 0])).issubset({-1, 1})
    du, dm, dl = donchian_channels(h, l, 20)
    assert np.all(du[20:] >= dl[20:])
    ich = ichimoku(h, l, c)
    assert "senkou_span_a" in ich and len(ich["tenkan_sen"]) == len(c)
    assert np.isfinite(linear_regression_slope(c, 20)[-1])

    # Momentum
    r = rsi(c, 14)
    assert np.all((r[15:] >= 0.0) & (r[15:] <= 100.0))
    sk, sd = stochastic_rsi(c, 14, 14)
    assert np.isfinite(sk[-1]) and np.isfinite(sd[-1])
    m_line, s_line, hist = macd(c)
    assert np.isfinite(hist[-1])
    assert np.isfinite(cci(h, l, c, 20)[-1])
    assert np.isfinite(roc(c, 12)[-1])
    assert np.isfinite(momentum(c, 10)[-1])
    wr = williams_r(h, l, c, 14)
    assert -100.0 <= wr[-1] <= 0.0
    uo = ultimate_oscillator(h, l, c)
    assert 0.0 <= uo[-1] <= 100.0
    assert np.isfinite(trix(c, 15)[-1])

    # Volatility
    a = atr(h, l, c, 14)
    assert np.all(a[14:] > 0)
    assert np.isfinite(historical_volatility(c, 20)[-1])
    bb = bollinger_bands(c, 20, 2.0)
    assert np.all(bb["upper"][20:] >= bb["lower"][20:])
    ku, km, kl = keltner_channels(h, l, c)
    assert np.all(ku[20:] >= kl[20:])
    assert np.isfinite(donchian_width(h, l, c, 20)[-1])
    assert np.isfinite(parkinson_volatility(h, l, 20)[-1])
    assert np.isfinite(garman_klass_volatility(o, h, l, c, 20)[-1])
    vp = volatility_percentile(a, 50)
    assert 0.0 <= vp[-1] <= 100.0

    # Volume
    assert len(obv(c, v)) == len(c)
    assert np.isfinite(volume_zscore(v, 20)[-1])
    assert np.isfinite(relative_volume(v, 20)[-1])
    assert len(cumulative_volume_delta(v, tbv, c, o)) == len(c)
    bsp = buying_selling_pressure(o, h, l, c, v, tbv)
    assert np.allclose(bsp["buying_volume"] + bsp["selling_volume"], v, rtol=1e-5)
    vprof = volume_profile(h, l, c, v)
    assert vprof["val"] <= vprof["poc"] <= vprof["vah"]

    # Statistical & Econometric
    _, _, zs = rolling_mean_std_zscore(c, 20)
    assert np.isfinite(zs[-1])
    log_ret = np.diff(np.log(c))
    corr = pearson_spearman_correlation(log_ret[:-1], log_ret[1:])
    assert -1.0 <= corr["pearson_r"] <= 1.0
    acf_res = autocorrelation_analysis(log_ret, nlags=10)
    assert len(acf_res["acf"]) > 1
    stat_res = stationarity_tests(log_ret)
    assert "adf_pvalue" in stat_res and stat_res["is_stationary_adf"] is True
    coint_res = engle_granger_cointegration(c, c * 1.5 + np.random.default_rng(1).normal(0, 10, len(c)))
    assert coint_res["is_cointegrated"] is True
    lr_res = linear_regression_residuals(c)
    assert len(lr_res["residuals"]) == len(c)
    kf = kalman_filter_1d(c)
    assert np.all(np.isfinite(kf))
    he = hurst_exponent(c)
    assert 0.0 <= he <= 1.0
    assert shannon_entropy(log_ret) > 0.0
    assert -1.0 <= volatility_clustering_score(log_ret) <= 1.0
    garch = garch_11_volatility(log_ret)
    assert 0.0 <= garch["persistence"] < 1.0


def test_price_action_smc_ict_and_regimes() -> None:
    df = make_synthetic_ohlcv(n=350, seed=19, vol=0.02)
    t = df["open_time_ms"].to_numpy()
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()

    pa = compute_price_action_features(o, h, l, c)
    assert np.sum(pa["swing_high_confirmed"]) > 0
    assert np.sum(pa["swing_low_confirmed"]) > 0

    smc = compute_smc_features(o, h, l, c)
    assert set(smc.keys()) >= {"bos_bull", "bos_bear", "choch_bull", "fvg_bull", "in_bull_ob", "bsl_sweep", "ssl_sweep"}
    assert np.all((smc["dealing_range_pos"] >= -5.0) & (smc["dealing_range_pos"] <= 5.0))

    ict = compute_ict_features(t, o, h, l, c, reference_high=h * 1.01, reference_low=l * 0.99)
    assert np.sum(ict["london_kill_zone"]) > 0
    assert np.sum(ict["ny_am_kill_zone"]) > 0
    assert np.any(np.isfinite(ict["pdh"]))

    reg = detect_market_regimes(h, l, c, use_hmm=False)
    assert len(reg["primary_regime"]) == len(c)
    assert set(reg["primary_regime"]).issubset(set(ALL_REGIMES))


def test_zero_lookahead_bias_prefix_invariance() -> None:
    """
    CRITICAL ANTI-LOOKAHEAD TEST:
    Truncating future candles after bar `k` or wildly altering future candles `t > k`
    MUST NEVER change any indicator, Price Action, SMC, ICT, Regime, or AST condition signal at index `<= k`.
    """
    df_full = make_synthetic_ohlcv(n=260, seed=11)
    k = 180
    df_prefix = df_full.slice(0, k)

    # Create a mutated future version where bars >= k jump 10x in price & volatility
    mut_pdf = df_full.to_pandas()
    mut_pdf.loc[k:, "open"] *= 10.0
    mut_pdf.loc[k:, "high"] *= 12.0
    mut_pdf.loc[k:, "low"] *= 8.0
    mut_pdf.loc[k:, "close"] *= 10.0
    mut_pdf.loc[k:, "volume"] *= 50.0
    import polars as pl
    df_shocked_future = pl.from_pandas(mut_pdf)

    ctx_pref = IndicatorCache(df_prefix, timeframe="1h")
    ctx_full = IndicatorCache(df_full, timeframe="1h")
    ctx_shock = IndicatorCache(df_shocked_future, timeframe="1h")

    # 1. Trend / Momentum / Volatility / Volume / Stat causal indicators
    for fn_name, args in [
        ("get_ema", (20,)),
        ("get_sma", (50,)),
        ("get_hma", (20,)),
        ("get_vwap", ()),
        ("get_rsi", (14,)),
        ("get_atr", (14,)),
        ("get_rvol", (20,)),
        ("get_cvd", ()),
        ("get_zscore", (20,)),
        ("get_kalman_z", (20,)),
        ("get_htf_trend", (4, 20)),
    ]:
        a_pref = getattr(ctx_pref, fn_name)(*args)
        a_shock = getattr(ctx_shock, fn_name)(*args)[:k]
        np.testing.assert_allclose(a_pref, a_shock, equal_nan=True, err_msg=f"Look-ahead leak in {fn_name}")

    # 2. Price Action, SMC, ICT, Regimes
    pa_pref = ctx_pref.get_pa()
    pa_shock = ctx_shock.get_pa()
    for key in ("swing_high_confirmed", "breakout_up", "resistance", "support"):
        np.testing.assert_allclose(pa_pref[key], pa_shock[key][:k], equal_nan=True, err_msg=f"PA look-ahead leak in {key}")

    smc_pref = ctx_pref.get_smc()
    smc_shock = ctx_shock.get_smc()
    for key in ("bos_bull", "choch_bull", "fvg_bull", "bsl_sweep", "in_bull_ob", "dealing_range_pos"):
        np.testing.assert_allclose(smc_pref[key], smc_shock[key][:k], equal_nan=True, err_msg=f"SMC look-ahead leak in {key}")

    ict_pref = ctx_pref.get_ict()
    ict_shock = ctx_shock.get_ict()
    for key in ("pdh", "pdl", "ote_long", "liquidity_raid_bull", "judas_swing_bull"):
        np.testing.assert_allclose(ict_pref[key], ict_shock[key][:k], equal_nan=True, err_msg=f"ICT look-ahead leak in {key}")

    reg_pref = ctx_pref.get_regimes()["primary_regime"]
    reg_shock = ctx_shock.get_regimes()["primary_regime"][:k]
    assert reg_pref == reg_shock

    # 3. Strategy leaf conditions including bb_squeeze
    for leaf in [
        LeafCondition("volatility", "bb_squeeze", {"period": 20, "std": 2.0}),
        LeafCondition("trend", "ema_cross_above", {"fast": 10, "slow": 30}),
        LeafCondition("smc", "bos_bull", {}),
        LeafCondition("ict", "ote_long", {}),
    ]:
        sig_pref = evaluate_condition(leaf, ctx_pref)
        sig_full = evaluate_condition(leaf, ctx_full)[:k]
        sig_shock = evaluate_condition(leaf, ctx_shock)[:k]
        np.testing.assert_array_equal(sig_pref, sig_full)
        np.testing.assert_array_equal(sig_pref, sig_shock)


@given(
    prices=st.lists(st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False), min_size=40, max_size=120),
    period=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=25)
def test_hypothesis_sma_ema_rsi_invariants(prices: list[float], period: int) -> None:
    arr = np.asarray(prices, dtype=np.float64)
    s = sma(arr, period)
    e = ema(arr, period)
    r = rsi(arr, period)

    assert len(s) == len(arr)
    finite_s = s[np.isfinite(s)]
    assert np.all(finite_s >= np.min(arr) - 1e-6)
    assert np.all(finite_s <= np.max(arr) + 1e-6)

    finite_e = e[np.isfinite(e)]
    assert np.all(finite_e >= np.min(arr) - 1e-6)
    assert np.all(finite_e <= np.max(arr) + 1e-6)

    finite_r = r[np.isfinite(r)]
    assert np.all((finite_r >= -1e-9) & (finite_r <= 100.0 + 1e-9))
