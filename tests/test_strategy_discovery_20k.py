"""Strategy grammar, AST deduplication, 20,000+ unique candidate generation, Genetic, and Optuna tests."""

from __future__ import annotations

import time
import tracemalloc

import pytest

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.core.exceptions import StrategyGrammarError
from crypto_quant_lab.strategies.evaluator import IndicatorCache, evaluate_strategy
from crypto_quant_lab.strategies.generator import stream_candidate_strategies
from crypto_quant_lab.strategies.genetic import run_genetic_discovery
from crypto_quant_lab.strategies.grammar import LeafCondition, LogicalExpr, StrategySpec, validate_strategy_spec
from crypto_quant_lab.strategies.optimizer import run_optuna_optimization
from tests.conftest import make_synthetic_ohlcv


def test_grammar_canonicalization_and_contradiction_rejection() -> None:
    a = LeafCondition("momentum", "rsi_below", {"period": 14, "threshold": 30})
    b = LeafCondition("trend", "ema_gt_ema", {"fast": 10, "slow": 50})

    spec1 = StrategySpec(
        name="s1",
        family="hybrid_quant",
        direction="long_only",
        long_entry=LogicalExpr("AND", (a, b)),
        long_exit=None,
        short_entry=None,
        short_exit=None,
    )
    spec2 = StrategySpec(
        name="s2",
        family="hybrid_quant",
        direction="long_only",
        long_entry=LogicalExpr("AND", (b, a)),  # Swapped commutative order
        long_exit=None,
        short_entry=None,
        short_exit=None,
    )
    assert spec1.ast_hash() == spec2.ast_hash()
    validate_strategy_spec(spec1)

    # Contradictory RSI bounds (rsi > 70 AND rsi < 30)
    contradictory = StrategySpec(
        name="bad",
        family="mean_reversion",
        direction="long_only",
        long_entry=LogicalExpr(
            "AND",
            (
                LeafCondition("momentum", "rsi_above", {"period": 14, "threshold": 70}),
                LeafCondition("momentum", "rsi_below", {"period": 14, "threshold": 30}),
            ),
        ),
        long_exit=None,
        short_entry=None,
        short_exit=None,
    )
    with pytest.raises(StrategyGrammarError):
        validate_strategy_spec(contradictory)

    # Disjoint EMA fast >= slow
    bad_ema = StrategySpec(
        name="bad_ema",
        family="trend_following",
        direction="long_only",
        long_entry=LeafCondition("trend", "ema_gt_ema", {"fast": 50, "slow": 20}),
        long_exit=None,
        short_entry=None,
        short_exit=None,
    )
    with pytest.raises(StrategyGrammarError):
        validate_strategy_spec(bad_ema)


def test_generate_20000_unique_valid_strategies_memory_and_speed() -> None:
    """Verify the engine can lazily generate >= 20,000 strictly unique valid strategies cleanly."""
    tracemalloc.start()
    t0 = time.perf_counter()

    hashes: set[str] = set()
    families_seen: set[str] = set()
    count = 0
    for spec in stream_candidate_strategies(count=20_000, algorithm="random_search", seed=42):
        h = spec.ast_hash()
        assert h not in hashes
        hashes.add(h)
        families_seen.add(spec.family)
        count += 1

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert count == 20_000
    assert len(hashes) == 20_000
    assert len(families_seen) == 14
    assert peak_bytes < 150 * 1024 * 1024  # Bounded RAM well under 150 MB
    assert elapsed < 60.0


def test_grid_genetic_and_optuna_discovery_modes() -> None:
    df = make_synthetic_ohlcv(n=250, seed=13)
    ctx = IndicatorCache(df, timeframe="1h")

    # Grid search
    grid_specs = list(stream_candidate_strategies(count=50, algorithm="grid_search", seed=42))
    assert len(grid_specs) == 50
    for sp in grid_specs[:10]:
        out = evaluate_strategy(sp, ctx, engine_type="vectorized", costs=ExecutionCostConfig(market_type="futures"))
        assert "sharpe_ratio" in out.metrics

    # Genetic / Evolutionary search
    ga_res = run_genetic_discovery(ctx, population_size=12, generations=3, families=["trend_following", "mean_reversion"], seed=42)
    assert len(ga_res) >= 12

    # Optuna Bayesian TPE
    opt_res = run_optuna_optimization(ctx, n_trials=12, families=["trend_following", "momentum_breakout"], seed=42)
    assert len(opt_res) >= 5
