"""Automated combinatorial strategy discovery engine and 11 explicit parameterized strategy templates."""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator

from crypto_quant_lab.core.exceptions import StrategyGrammarError
from crypto_quant_lab.strategies.grammar import (
    DIRECTIONS,
    STRATEGY_FAMILIES,
    ConditionNode,
    LeafCondition,
    LogicalExpr,
    StrategySpec,
    TemporalExpr,
    validate_strategy_spec,
)

EMA_PERIODS = [10, 20, 30, 50, 100, 150, 200]
RSI_PERIODS = [7, 9, 14, 21, 28]
RSI_LOW_THRESHOLDS = [25, 30, 35, 40, 45]
RSI_HIGH_THRESHOLDS = [55, 60, 65, 70, 75]
ATR_PERIODS = [7, 10, 14, 20, 28]
SL_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
TP_MULTS = [1.5, 2.0, 3.0, 4.0, 5.0]
TIME_EXITS = [None, 24, 48, 96]


def generate_explicit_template_strategies() -> list[StrategySpec]:
    """Return the 11 explicit, readable, parameterized Phase 3 strategy templates."""
    templates: list[StrategySpec] = []

    # 1. ICT 2022 Model: Liquidity Sweep -> MSS -> FVG retest
    templates.append(
        StrategySpec(
            name="TPL_ICT_2022_Model",
            family="liquidity_sweep",
            direction="long_only",
            long_entry=TemporalExpr("SEQUENCE", (LeafCondition("smc", "ssl_sweep", {}), LeafCondition("smc", "mss_bull", {}), LeafCondition("smc", "fvg_bull", {})), window_bars=6),
            long_exit=LeafCondition("momentum", "rsi_above", {"period": 14, "threshold": 75}),
            short_entry=None,
            short_exit=None,
            sl_atr_mult=1.5,
            tp_atr_mult=3.5,
            entry_mode="zone_edge",
            sl_mode="sweep_wick",
            tp_mode="opposing_liquidity",
            partial_tp_ratio=0.5,
            partial_tp_rr=2.0,
            move_to_be_after_tp1=True,
            template_id="ict_2022",
        )
    )

    # 2. BOS Continuation to OB/FVG in discount
    templates.append(
        StrategySpec(
            name="TPL_BOS_OB_Continuation",
            family="pullback_trading",
            direction="long_only",
            long_entry=LogicalExpr("AND", (LeafCondition("smc", "bos_bull", {}), LeafCondition("smc", "discount_zone", {}), LeafCondition("smc", "in_bull_ob", {}))),
            long_exit=LeafCondition("trend", "ema_cross_below", {"fast": 10, "slow": 30}),
            short_entry=None,
            short_exit=None,
            sl_atr_mult=2.0,
            tp_atr_mult=3.0,
            entry_mode="zone_50",
            sl_mode="zone_invalidation",
            tp_mode="rr_multiple",
            template_id="bos_continuation",
        )
    )

    # 3. Silver Bullet (FVG entry inside kill-zone window after sweep)
    templates.append(
        StrategySpec(
            name="TPL_Silver_Bullet",
            family="liquidity_sweep",
            direction="long_only",
            long_entry=LogicalExpr("AND", (LeafCondition("ict", "silver_bullet_bull", {}), LeafCondition("ict", "ny_am_kill_zone", {}))),
            long_exit=None,
            short_entry=None,
            short_exit=None,
            sl_atr_mult=1.2,
            tp_atr_mult=3.0,
            entry_mode="zone_50",
            sl_mode="sweep_wick",
            tp_mode="opposing_liquidity",
            session_stop=True,
            template_id="silver_bullet",
        )
    )

    # 4. Judas Swing / PO3 Session Reversal
    templates.append(
        StrategySpec(
            name="TPL_Judas_Session_Reversal",
            family="liquidity_sweep",
            direction="long_only",
            long_entry=LogicalExpr("AND", (LeafCondition("ict", "judas_swing_bull", {}), LeafCondition("ict", "po3_expansion_bull", {}))),
            long_exit=None,
            short_entry=None,
            short_exit=None,
            sl_atr_mult=1.5,
            tp_atr_mult=4.0,
            entry_mode="zone_edge",
            sl_mode="sweep_wick",
            tp_mode="opposing_liquidity",
            session_stop=True,
            template_id="judas_po3",
        )
    )

    # 5. Previous Day / Week High/Low Raid and Reversal
    templates.append(
        StrategySpec(
            name="TPL_PDH_PDL_Raid_Reversal",
            family="liquidity_sweep",
            direction="long_only",
            long_entry=LogicalExpr("AND", (LeafCondition("ict", "liquidity_raid_bull", {}), LeafCondition("smc", "choch_bull", {}))),
            long_exit=LeafCondition("momentum", "rsi_above", {"period": 14, "threshold": 70}),
            short_entry=None,
            short_exit=None,
            sl_atr_mult=2.0,
            tp_atr_mult=3.5,
            entry_mode="market",
            sl_mode="sweep_wick",
            tp_mode="opposing_liquidity",
            template_id="raid_reversal",
        )
    )

    # 6. Breaker Block retest; IFVG flip entry
    templates.append(
        StrategySpec(
            name="TPL_Breaker_IFVG_Flip",
            family="order_block",
            direction="long_only",
            long_entry=LogicalExpr("OR", (LeafCondition("smc", "in_bull_breaker", {}), LeafCondition("smc", "ifvg_bull", {}))),
            long_exit=LeafCondition("trend", "ema_cross_below", {"fast": 10, "slow": 30}),
            short_entry=None,
            short_exit=None,
            sl_atr_mult=1.8,
            tp_atr_mult=3.0,
            entry_mode="zone_edge",
            sl_mode="zone_invalidation",
            tp_mode="rr_multiple",
            template_id="breaker_ifvg",
        )
    )

    # 7. S/R Flip Retest & Failed Breakout (Fakeout)
    templates.append(
        StrategySpec(
            name="TPL_SR_Flip_Fakeout",
            family="range_trading",
            direction="long_only",
            long_entry=LogicalExpr("OR", (LeafCondition("price_action", "sr_flip_bull", {}), LeafCondition("price_action", "failed_breakdown_down", {}))),
            long_exit=LeafCondition("price_action", "at_sr_resistance", {}),
            short_entry=None,
            short_exit=None,
            sl_atr_mult=1.5,
            tp_atr_mult=2.5,
            entry_mode="zone_50",
            sl_mode="zone_invalidation",
            tp_mode="opposing_liquidity",
            template_id="sr_flip",
        )
    )

    # 8. Multi-Timeframe: HTF Bias + LTF Entry Trigger via WITHIN
    templates.append(
        StrategySpec(
            name="TPL_MTF_Bias_LTF_Trigger",
            family="multi_timeframe",
            direction="long_only",
            long_entry=LogicalExpr("AND", (LeafCondition("mtf", "htf_bull", {"factor": 4, "ema_period": 20}), TemporalExpr("WITHIN", (LeafCondition("smc", "bos_bull", {}),), window_bars=3))),
            long_exit=LeafCondition("mtf", "htf_bear", {"factor": 4, "ema_period": 20}),
            short_entry=None,
            short_exit=None,
            sl_atr_mult=2.0,
            tp_atr_mult=4.0,
            entry_mode="market",
            sl_mode="atr",
            tp_mode="rr_multiple",
            template_id="mtf_bias",
        )
    )

    # 9. Pure Price Action: Engulfing / Pin Bar at S/R Zone with Trend Filter
    templates.append(
        StrategySpec(
            name="TPL_PA_PinBar_SR_Confluence",
            family="pullback_trading",
            direction="long_only",
            long_entry=LogicalExpr("AND", (LeafCondition("price_action", "bullish_pin_bar", {}), LeafCondition("price_action", "at_sr_support", {}), LeafCondition("trend", "close_gt_ema", {"period": 50}))),
            long_exit=LeafCondition("momentum", "rsi_above", {"period": 14, "threshold": 75}),
            short_entry=None,
            short_exit=None,
            sl_atr_mult=1.5,
            tp_atr_mult=3.0,
            entry_mode="market",
            sl_mode="sweep_wick",
            tp_mode="rr_multiple",
            template_id="pa_pinbar",
        )
    )

    # 10. Statistical Mean Reversion with Kalman Z-Score & BB Squeeze
    templates.append(
        StrategySpec(
            name="TPL_Stat_Kalman_Mean_Reversion",
            family="mean_reversion",
            direction="long_only",
            long_entry=LogicalExpr("AND", (LeafCondition("statistical", "kalman_z_below", {"period": 20, "threshold": -1.8}), LeafCondition("volatility", "bb_squeeze", {"period": 20, "std": 2.0}))),
            long_exit=LeafCondition("statistical", "kalman_z_above", {"period": 20, "threshold": 0.0}),
            short_entry=None,
            short_exit=None,
            sl_atr_mult=2.0,
            tp_atr_mult=2.0,
            entry_mode="market",
            sl_mode="atr",
            tp_mode="rr_multiple",
            template_id="kalman_mr",
        )
    )

    # 11. Hybrid Quant Confluence
    templates.append(
        StrategySpec(
            name="TPL_Hybrid_Quant_Confluence",
            family="hybrid_quant",
            direction="long_only",
            long_entry=LogicalExpr("AND", (LeafCondition("trend", "close_gt_vwap", {}), LeafCondition("volume", "rvol_above", {"period": 20, "threshold": 1.3}), LeafCondition("momentum", "macd_bull", {}))),
            long_exit=LeafCondition("trend", "close_lt_vwap", {}),
            short_entry=None,
            short_exit=None,
            sl_atr_mult=2.0,
            tp_atr_mult=3.5,
            entry_mode="market",
            sl_mode="atr",
            tp_mode="rr_multiple",
            template_id="hybrid_confluence",
        )
    )

    return templates


def _bull_trend_pool() -> list[LeafCondition]:
    leaves: list[LeafCondition] = []
    for f, s in [(10, 30), (10, 50), (20, 50), (20, 100), (30, 100), (50, 150), (50, 200)]:
        leaves.append(LeafCondition("trend", "ema_gt_ema", {"fast": f, "slow": s}))
        leaves.append(LeafCondition("trend", "ema_cross_above", {"fast": f, "slow": s}))
    for p in [20, 50, 100, 200]:
        leaves.append(LeafCondition("trend", "close_gt_ema", {"period": p}))
        leaves.append(LeafCondition("trend", "close_gt_sma", {"period": p}))
    for p in [14, 20, 30]:
        leaves.append(LeafCondition("trend", "hma_slope_up", {"period": p}))
        leaves.append(LeafCondition("trend", "linreg_slope_pos", {"period": p}))
    leaves.append(LeafCondition("trend", "close_gt_vwap", {}))
    leaves.append(LeafCondition("trend", "supertrend_bull", {"period": 10, "mult": 3.0}))
    leaves.append(LeafCondition("trend", "supertrend_bull", {"period": 14, "mult": 2.5}))
    leaves.append(LeafCondition("trend", "ichimoku_bull", {}))
    for thresh in [18, 22, 25, 30]:
        leaves.append(LeafCondition("trend", "adx_above", {"period": 14, "threshold": thresh}))
    return leaves


def _bear_trend_pool() -> list[LeafCondition]:
    leaves: list[LeafCondition] = []
    for f, s in [(10, 30), (10, 50), (20, 50), (20, 100), (30, 100), (50, 150), (50, 200)]:
        leaves.append(LeafCondition("trend", "ema_lt_ema", {"fast": f, "slow": s}))
        leaves.append(LeafCondition("trend", "ema_cross_below", {"fast": f, "slow": s}))
    for p in [20, 50, 100, 200]:
        leaves.append(LeafCondition("trend", "close_lt_ema", {"period": p}))
        leaves.append(LeafCondition("trend", "close_lt_sma", {"period": p}))
    for p in [14, 20, 30]:
        leaves.append(LeafCondition("trend", "hma_slope_down", {"period": p}))
        leaves.append(LeafCondition("trend", "linreg_slope_neg", {"period": p}))
    leaves.append(LeafCondition("trend", "close_lt_vwap", {}))
    leaves.append(LeafCondition("trend", "supertrend_bear", {"period": 10, "mult": 3.0}))
    leaves.append(LeafCondition("trend", "ichimoku_bear", {}))
    return leaves


def _bull_momentum_pool() -> list[LeafCondition]:
    leaves: list[LeafCondition] = [LeafCondition("momentum", "macd_bull", {})]
    for p in RSI_PERIODS:
        for th in RSI_LOW_THRESHOLDS:
            if p >= 21 and th <= 25:
                continue
            leaves.append(LeafCondition("momentum", "rsi_below", {"period": p, "threshold": th}))
        for th in [50, 55, 60]:
            leaves.append(LeafCondition("momentum", "rsi_above", {"period": p, "threshold": th}))
        leaves.append(LeafCondition("momentum", "stoch_rsi_bull", {"period": p}))
    for p in [14, 20]:
        leaves.append(LeafCondition("momentum", "cci_below", {"period": p, "threshold": -100}))
        leaves.append(LeafCondition("momentum", "willr_below", {"period": p, "threshold": -80}))
    for th in [0.2, 0.5, 1.0]:
        leaves.append(LeafCondition("momentum", "roc_above", {"period": 12, "threshold": th}))
    return leaves


def _bear_momentum_pool() -> list[LeafCondition]:
    leaves: list[LeafCondition] = [LeafCondition("momentum", "macd_bear", {})]
    for p in RSI_PERIODS:
        for th in RSI_HIGH_THRESHOLDS:
            if p >= 21 and th >= 75:
                continue
            leaves.append(LeafCondition("momentum", "rsi_above", {"period": p, "threshold": th}))
        for th in [40, 45, 50]:
            leaves.append(LeafCondition("momentum", "rsi_below", {"period": p, "threshold": th}))
        leaves.append(LeafCondition("momentum", "stoch_rsi_bear", {"period": p}))
    for p in [14, 20]:
        leaves.append(LeafCondition("momentum", "cci_above", {"period": p, "threshold": 100}))
        leaves.append(LeafCondition("momentum", "willr_above", {"period": p, "threshold": -20}))
    for th in [-0.2, -0.5, -1.0]:
        leaves.append(LeafCondition("momentum", "roc_below", {"period": 12, "threshold": th}))
    return leaves


def _volatility_pool(side: str) -> list[LeafCondition]:
    leaves: list[LeafCondition] = []
    for p in [14, 20, 30]:
        for std in [1.5, 2.0, 2.5]:
            if side == "long":
                leaves.append(LeafCondition("volatility", "bb_below_lower", {"period": p, "std": std}))
            else:
                leaves.append(LeafCondition("volatility", "bb_above_upper", {"period": p, "std": std}))
        leaves.append(LeafCondition("volatility", "bb_squeeze", {"period": p, "std": 2.0}))
        if side == "long":
            leaves.append(LeafCondition("volatility", "keltner_break_up", {"period": p, "mult": 2.0}))
        else:
            leaves.append(LeafCondition("volatility", "keltner_break_down", {"period": p, "mult": 2.0}))
    for th in [25, 40, 60]:
        leaves.append(LeafCondition("volatility", "vol_pct_above", {"atr_period": 14, "threshold": th}))
    for th in [50, 75]:
        leaves.append(LeafCondition("volatility", "vol_pct_below", {"atr_period": 14, "threshold": th}))
    return leaves


def _volume_pool(side: str) -> list[LeafCondition]:
    leaves: list[LeafCondition] = []
    for p in [14, 20]:
        for th in [1.1, 1.3, 1.5, 2.0]:
            leaves.append(LeafCondition("volume", "rvol_above", {"period": p, "threshold": th}))
        for th in [0.5, 1.0, 1.5]:
            leaves.append(LeafCondition("volume", "vol_z_above", {"period": p, "threshold": th}))
    for lb in [3, 5, 10]:
        leaves.append(LeafCondition("volume", "cvd_rising" if side == "long" else "cvd_falling", {"lookback": lb}))
    return leaves


def _stat_pool(side: str) -> list[LeafCondition]:
    leaves: list[LeafCondition] = []
    for p in [20, 30, 50]:
        for th in ([1.0, 1.5, 2.0] if side == "short" else [-1.0, -1.5, -2.0]):
            sig = "zscore_below" if side == "long" else "zscore_above"
            ksig = "kalman_z_below" if side == "long" else "kalman_z_above"
            leaves.append(LeafCondition("statistical", sig, {"period": p, "threshold": th}))
            leaves.append(LeafCondition("statistical", ksig, {"period": p, "threshold": th}))
    return leaves


def _pa_smc_ict_pool(side: str) -> list[LeafCondition]:
    if side == "long":
        return [
            LeafCondition("price_action", "breakout_up", {}),
            LeafCondition("price_action", "failed_breakdown_down", {}),
            LeafCondition("price_action", "bullish_engulfing", {}),
            LeafCondition("price_action", "bullish_pin_bar", {}),
            LeafCondition("price_action", "higher_low", {}),
            LeafCondition("smc", "bos_bull", {}),
            LeafCondition("smc", "choch_bull", {}),
            LeafCondition("smc", "mss_bull", {}),
            LeafCondition("smc", "ssl_sweep", {}),
            LeafCondition("smc", "fvg_bull", {}),
            LeafCondition("smc", "ifvg_bull", {}),
            LeafCondition("smc", "in_bull_ob", {}),
            LeafCondition("smc", "in_bull_breaker", {}),
            LeafCondition("smc", "discount_zone", {}),
            LeafCondition("ict", "ote_long", {}),
            LeafCondition("ict", "liquidity_raid_bull", {}),
            LeafCondition("ict", "judas_swing_bull", {}),
            LeafCondition("ict", "po3_expansion_bull", {}),
            LeafCondition("zone_state", "touching_fresh_zone", {"source": "smc", "zone_type": "ob", "direction": "bull"}),
            LeafCondition("zone_state", "touching_fresh_zone", {"source": "smc", "zone_type": "fvg", "direction": "bull"}),
        ]
    return [
        LeafCondition("price_action", "breakdown_down", {}),
        LeafCondition("price_action", "failed_breakout_up", {}),
        LeafCondition("price_action", "bearish_engulfing", {}),
        LeafCondition("price_action", "bearish_pin_bar", {}),
        LeafCondition("price_action", "lower_high", {}),
        LeafCondition("smc", "bos_bear", {}),
        LeafCondition("smc", "choch_bear", {}),
        LeafCondition("smc", "mss_bear", {}),
        LeafCondition("smc", "bsl_sweep", {}),
        LeafCondition("smc", "fvg_bear", {}),
        LeafCondition("smc", "ifvg_bear", {}),
        LeafCondition("smc", "in_bear_ob", {}),
        LeafCondition("smc", "in_bear_breaker", {}),
        LeafCondition("smc", "premium_zone", {}),
        LeafCondition("ict", "ote_short", {}),
        LeafCondition("ict", "liquidity_raid_bear", {}),
        LeafCondition("ict", "judas_swing_bear", {}),
        LeafCondition("ict", "po3_expansion_bear", {}),
        LeafCondition("zone_state", "touching_fresh_zone", {"source": "smc", "zone_type": "ob", "direction": "bear"}),
        LeafCondition("zone_state", "touching_fresh_zone", {"source": "smc", "zone_type": "fvg", "direction": "bear"}),
    ]


def _regime_pool(side: str) -> list[LeafCondition]:
    if side == "long":
        return [
            LeafCondition("regime", "regime_in", {"regimes": ["strong_bull_trend", "weak_bull_trend"]}),
            LeafCondition("regime", "regime_in", {"regimes": ["strong_bull_trend", "weak_bull_trend", "volatility_expansion"]}),
            LeafCondition("regime", "regime_in", {"regimes": ["sideways_market", "low_volatility", "volatility_compression"]}),
        ]
    return [
        LeafCondition("regime", "regime_in", {"regimes": ["strong_bear_trend", "weak_bear_trend"]}),
        LeafCondition("regime", "regime_in", {"regimes": ["strong_bear_trend", "weak_bear_trend", "volatility_expansion"]}),
        LeafCondition("regime", "regime_in", {"regimes": ["sideways_market", "low_volatility", "volatility_compression"]}),
    ]


def _mtf_pool(side: str) -> list[LeafCondition]:
    sig = "htf_bull" if side == "long" else "htf_bear"
    return [
        LeafCondition("mtf", sig, {"factor": 4, "ema_period": 20}),
        LeafCondition("mtf", sig, {"factor": 4, "ema_period": 50}),
        LeafCondition("mtf", sig, {"factor": 6, "ema_period": 20}),
    ]


def _combine_and(nodes: list[LeafCondition]) -> ConditionNode:
    if len(nodes) == 1:
        return nodes[0]
    return LogicalExpr("AND", tuple(nodes))


def build_family_condition_pools(family: str, side: str) -> tuple[list[LeafCondition], list[LeafCondition], list[LeafCondition]]:
    """Return (primary_triggers, secondary_filters, optional_confirmations) tailored to each of the 14 families."""
    trend = _bull_trend_pool() if side == "long" else _bear_trend_pool()
    mom = _bull_momentum_pool() if side == "long" else _bear_momentum_pool()
    vol = _volatility_pool(side)
    volm = _volume_pool(side)
    stat = _stat_pool(side)
    ps_ict = _pa_smc_ict_pool(side)
    reg = _regime_pool(side)
    mtf = _mtf_pool(side)

    if family == "trend_following":
        return trend, mom + mtf, volm + reg
    if family == "mean_reversion":
        return stat + [x for x in mom if "below" in x.signal or "above" in x.signal], vol, volm + reg
    if family == "momentum_breakout":
        return [LeafCondition("trend", "donchian_break_up" if side == "long" else "donchian_break_down", {"period": p}) for p in [14, 20, 30, 50]] + mom, volm, trend + reg
    if family == "volatility_breakout":
        return vol, volm + mom, trend + reg
    if family in ("statistical_arbitrage", "pairs_trading"):
        return stat, mom + vol, volm + reg
    if family == "pullback_trading":
        return mom + ps_ict, trend + mtf, volm
    if family == "range_trading":
        return [x for x in vol if "bb_" in x.signal] + stat, reg, mom + volm
    if family == "liquidity_sweep":
        return [x for x in ps_ict if "sweep" in x.signal or "raid" in x.signal or "judas" in x.signal], mom + volm, trend + reg
    if family == "bos_choch":
        return [x for x in ps_ict if "bos" in x.signal or "choch" in x.signal or "mss" in x.signal], trend + volm, mtf + reg
    if family == "fvg_strategy":
        return [x for x in ps_ict if "fvg" in x.signal or "ote" in x.signal], trend + mom, volm + reg
    if family == "order_block":
        return [x for x in ps_ict if "ob" in x.signal or "breaker" in x.signal or "zone" in x.signal], trend + mom, volm
    if family == "multi_timeframe":
        return mtf, trend + mom, volm + ps_ict
    return trend + ps_ict, mom + stat + vol, volm + mtf + reg


def stream_candidate_strategies(
    count: int = 20_000,
    families: list[str] | None = None,
    directions: list[str] | None = None,
    algorithm: str = "random_search",
    seed: int = 42,
    allowed_categories: set[str] | None = None,
) -> Iterator[StrategySpec]:
    """
    Memory-efficient lazy generator yielding `count` unique, validated StrategySpec candidates.
    Does NOT hold all candidate AST objects in RAM (stores only 32-byte/64-char hash digests for uniqueness).
    """
    active_families = [f for f in (families or list(STRATEGY_FAMILIES)) if f in STRATEGY_FAMILIES]
    if not active_families:
        active_families = list(STRATEGY_FAMILIES)
    active_dirs = [d for d in (directions or list(DIRECTIONS)) if d in DIRECTIONS]
    if not active_dirs:
        active_dirs = list(DIRECTIONS)

    rng = random.Random(seed)
    seen_hashes: set[str] = set()
    yielded = 0
    attempts = 0
    max_attempts = max(count * 12, 100_000)

    # First yield the 11 explicit Phase 3 templates if requested or during general stream
    explicit_tpls = generate_explicit_template_strategies()
    for tpl in explicit_tpls:
        if yielded >= count:
            return
        if tpl.family in active_families and tpl.direction in active_dirs:
            try:
                validate_strategy_spec(tpl)
            except StrategyGrammarError:
                continue
            h = tpl.ast_hash()
            if h not in seen_hashes:
                seen_hashes.add(h)
                yielded += 1
                yield tpl

    def _filter_cat(pool: list[LeafCondition]) -> list[LeafCondition]:
        if not allowed_categories:
            return pool
        filtered = [x for x in pool if x.category in allowed_categories]
        return filtered if filtered else pool

    if algorithm == "grid_search":
        for fam in active_families:
            lp1, lp2, lp3 = [_filter_cat(p) for p in build_family_condition_pools(fam, "long")]
            sp1, sp2, sp3 = [_filter_cat(p) for p in build_family_condition_pools(fam, "short")]
            for direction in active_dirs:
                for l1, l2, l3, sl, tp, atr_p in itertools.product(
                    lp1[:12], lp2[:10], [None] + lp3[:6], SL_MULTS[:4], TP_MULTS[:4], ATR_PERIODS[:3]
                ):
                    if yielded >= count:
                        return
                    s1 = sp1[attempts % len(sp1)]
                    s2 = sp2[(attempts // 3) % len(sp2)]
                    l_nodes = [l1, l2] + ([l3] if l3 else [])
                    s_nodes = [s1, s2]
                    spec = StrategySpec(
                        name=f"{fam}_{direction}_{yielded + 1:05d}",
                        family=fam,
                        direction=direction,
                        long_entry=_combine_and(l_nodes) if direction != "short_only" else None,
                        long_exit=LeafCondition("trend", "ema_cross_below", {"fast": 10, "slow": 30}) if direction != "short_only" else None,
                        short_entry=_combine_and(s_nodes) if direction != "long_only" else None,
                        short_exit=LeafCondition("trend", "ema_cross_above", {"fast": 10, "slow": 30}) if direction != "long_only" else None,
                        sl_atr_mult=sl,
                        tp_atr_mult=tp,
                        atr_period=atr_p,
                    )
                    attempts += 1
                    try:
                        validate_strategy_spec(spec)
                    except StrategyGrammarError:
                        continue
                    h = spec.ast_hash()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        yielded += 1
                        yield spec
        return

    pools_cache = {
        (fam, side): [_filter_cat(p) for p in build_family_condition_pools(fam, side)]
        for fam in active_families
        for side in ("long", "short")
    }
    long_exits = [None, LeafCondition("momentum", "rsi_above", {"period": 14, "threshold": 75}), LeafCondition("trend", "ema_cross_below", {"fast": 10, "slow": 30})]
    short_exits = [None, LeafCondition("momentum", "rsi_below", {"period": 14, "threshold": 25}), LeafCondition("trend", "ema_cross_above", {"fast": 10, "slow": 30})]

    while yielded < count and attempts < max_attempts:
        attempts += 1
        fam = rng.choice(active_families)
        direction = rng.choice(active_dirs)
        lp1, lp2, lp3 = pools_cache[(fam, "long")]
        sp1, sp2, sp3 = pools_cache[(fam, "short")]

        n_long_terms = rng.choice([2, 3, 3, 4])
        l_terms = [rng.choice(lp1), rng.choice(lp2)]
        if n_long_terms >= 3:
            l_terms.append(rng.choice(lp3))
        if n_long_terms == 4:
            l_terms.append(rng.choice(lp1 + lp2))

        n_short_terms = rng.choice([2, 3])
        s_terms = [rng.choice(sp1), rng.choice(sp2)]
        if n_short_terms == 3:
            s_terms.append(rng.choice(sp3))

        if len(l_terms) >= 3 and rng.random() < 0.2:
            long_entry: ConditionNode = LogicalExpr("AND", (l_terms[0], LogicalExpr("OR", (l_terms[1], l_terms[2]))))
        elif rng.random() < 0.15:
            long_entry = TemporalExpr("SEQUENCE", (l_terms[0], l_terms[1]), window_bars=5)
        else:
            long_entry = _combine_and(l_terms)

        short_entry: ConditionNode = _combine_and(s_terms)

        sl = rng.choice(SL_MULTS)
        tp = rng.choice(TP_MULTS)
        atr_p = rng.choice(ATR_PERIODS)
        t_exit = rng.choice(TIME_EXITS)

        spec = StrategySpec(
            name=f"{fam}_{direction}_{yielded + 1:05d}",
            family=fam,
            direction=direction,
            long_entry=long_entry if direction != "short_only" else None,
            long_exit=rng.choice(long_exits) if direction != "short_only" else None,
            short_entry=short_entry if direction != "long_only" else None,
            short_exit=rng.choice(short_exits) if direction != "long_only" else None,
            sl_atr_mult=sl,
            tp_atr_mult=tp,
            atr_period=atr_p,
            max_holding_bars=t_exit,
        )
        try:
            validate_strategy_spec(spec)
        except StrategyGrammarError:
            continue

        h = spec.ast_hash()
        if h not in seen_hashes:
            seen_hashes.add(h)
            yielded += 1
            yield spec
