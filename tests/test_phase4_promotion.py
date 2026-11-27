"""Phase 4 tests: multi-market accuracy validation, bootstrap expectancy CI, Wilson score win-rate CI, correlation clustering, and Strategy Card generation."""

from __future__ import annotations

import numpy as np
import pytest

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.strategies.evaluator import IndicatorCache, StrategySpec
from crypto_quant_lab.strategies.grammar import LeafCondition
from crypto_quant_lab.validation.promotion import (
    bootstrap_expectancy_ci,
    cluster_strategies_by_correlation,
    generate_strategy_card,
    multi_market_accuracy_validation,
    wilson_score_interval,
)
from tests.conftest import make_synthetic_ohlcv


def test_wilson_score_and_bootstrap_ci() -> None:
    lo, hi = wilson_score_interval(successes=55, total=100, confidence=0.95)
    assert 0.45 <= lo <= 0.55
    assert 0.55 <= hi <= 0.65

    rets = [0.01, 0.02, -0.005, 0.015, 0.03, -0.01, 0.008, 0.025]
    b_lo, b_hi = bootstrap_expectancy_ci(rets, n_bootstrap=200, confidence=0.95)
    assert b_lo < b_hi


def test_correlation_clustering_deduplication() -> None:
    rng = np.random.default_rng(42)
    base = rng.normal(0, 0.01, 100)
    col1 = base + rng.normal(0, 0.001, 100)
    col2 = base + rng.normal(0, 0.001, 100)  # highly correlated with col1 (> 0.95)
    col3 = rng.normal(0, 0.01, 100)          # uncorrelated
    mat = np.column_stack([col1, col2, col3])

    clusters = cluster_strategies_by_correlation(mat, threshold=0.8)
    assert len(clusters) == 2  # [0, 1] grouped together, [2] separate


def test_strategy_card_generator() -> None:
    spec = StrategySpec(
        name="TestCardStrategy",
        family="trend_following",
        direction="long_only",
        long_entry=LeafCondition("trend", "close_gt_ema", {"period": 20}),
        long_exit=None,
        short_entry=None,
        short_exit=None,
    )
    oos = {
        "total_trades": 120,
        "sharpe_ratio": 1.65,
        "profit_factor": 1.85,
        "max_drawdown_pct": 14.2,
        "win_rate": 0.58,
    }
    val = {
        "passed_gates": True,
        "deflated_sharpe": 0.75,
        "pbo_score": 0.25,
        "rejection_reasons": [],
    }
    card = generate_strategy_card(spec, oos, val, (0.005, 0.025), (0.50, 0.66))
    assert "Grade A" in card["grade"]
    assert card["diagnostics"]["passed_gates"] is True


def test_multi_market_accuracy_validation() -> None:
    df1 = make_synthetic_ohlcv(n=150, seed=10)
    df2 = make_synthetic_ohlcv(n=150, seed=20)
    df3 = make_synthetic_ohlcv(n=150, seed=30)
    datasets = {
        "BTCUSDT": IndicatorCache(df1, timeframe="1h"),
        "ETHUSDT": IndicatorCache(df2, timeframe="1h"),
        "SOLUSDT": IndicatorCache(df3, timeframe="1h"),
    }
    spec = StrategySpec(
        name="MultiMarketTest",
        family="trend_following",
        direction="long_only",
        long_entry=LeafCondition("trend", "close_gt_ema", {"period": 20}),
        long_exit=None,
        short_entry=None,
        short_exit=None,
    )
    res = multi_market_accuracy_validation(spec, datasets, costs=ExecutionCostConfig(market_type="futures"))
    assert len(res["multi_market_results"]) == 3
    assert "mean_cross_market_sharpe" in res
