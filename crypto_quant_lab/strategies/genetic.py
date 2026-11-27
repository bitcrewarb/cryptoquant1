"""Genetic Algorithm & Evolutionary Strategy Generator for strategy discovery."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.core.exceptions import StrategyGrammarError
from crypto_quant_lab.strategies.evaluator import IndicatorCache, evaluate_strategy
from crypto_quant_lab.strategies.generator import (
    ATR_PERIODS,
    SL_MULTS,
    TP_MULTS,
    stream_candidate_strategies,
)
from crypto_quant_lab.strategies.grammar import LogicalExpr, StrategySpec, validate_strategy_spec


def _mutate_strategy(spec: StrategySpec, rng: random.Random, idx: int) -> StrategySpec:
    sl = rng.choice(SL_MULTS) if rng.random() < 0.5 else spec.sl_atr_mult
    tp = rng.choice(TP_MULTS) if rng.random() < 0.5 else spec.tp_atr_mult
    atr_p = rng.choice(ATR_PERIODS) if rng.random() < 0.35 else spec.atr_period
    t_exit = rng.choice([None, 24, 48, 96]) if rng.random() < 0.3 else spec.max_holding_bars
    return StrategySpec(
        name=f"{spec.family}_evo_{idx:05d}",
        family=spec.family,
        direction=spec.direction,
        long_entry=spec.long_entry,
        long_exit=spec.long_exit,
        short_entry=spec.short_entry,
        short_exit=spec.short_exit,
        sl_atr_mult=sl,
        tp_atr_mult=tp,
        atr_period=atr_p,
        max_holding_bars=t_exit,
        risk_per_trade_pct=spec.risk_per_trade_pct,
    )


def _crossover_strategies(parent_a: StrategySpec, parent_b: StrategySpec, rng: random.Random, idx: int) -> StrategySpec:
    # AST subtree & parameter crossover
    long_entry = parent_a.long_entry
    if parent_a.long_entry and parent_b.long_entry and isinstance(parent_a.long_entry, LogicalExpr) and isinstance(parent_b.long_entry, LogicalExpr):
        c_a = parent_a.long_entry.children
        c_b = parent_b.long_entry.children
        if c_a and c_b:
            long_entry = LogicalExpr("AND", (c_a[0], c_b[-1]))

    direction = parent_a.direction if rng.random() < 0.5 else parent_b.direction
    if direction == "long_short" and (long_entry is None or parent_b.short_entry is None):
        direction = "long_only" if long_entry is not None else "short_only"

    return StrategySpec(
        name=f"{parent_a.family}_ga_{idx:05d}",
        family=parent_a.family,
        direction=direction,
        long_entry=long_entry if direction != "short_only" else None,
        long_exit=parent_a.long_exit if direction != "short_only" else None,
        short_entry=parent_b.short_entry if direction != "long_only" else None,
        short_exit=parent_b.short_exit if direction != "long_only" else None,
        sl_atr_mult=parent_a.sl_atr_mult if rng.random() < 0.5 else parent_b.sl_atr_mult,
        tp_atr_mult=parent_b.tp_atr_mult if rng.random() < 0.5 else parent_a.tp_atr_mult,
        atr_period=parent_a.atr_period,
        max_holding_bars=parent_b.max_holding_bars,
    )


def run_genetic_discovery(
    ctx: IndicatorCache,
    population_size: int = 30,
    generations: int = 5,
    families: list[str] | None = None,
    directions: list[str] | None = None,
    seed: int = 42,
    on_evaluated: Callable[[StrategySpec, dict[str, Any]], None] | None = None,
    costs: ExecutionCostConfig | None = None,
    funding_rates_map: dict[int, float] | None = None,
) -> list[tuple[StrategySpec, dict[str, Any]]]:
    """Run Genetic Algorithm / Evolutionary Strategy search on training slice."""
    cost_cfg = costs or ExecutionCostConfig(market_type="futures" if (directions is None or any(d != "long_only" for d in directions)) else "spot")
    effective_dirs = ["long_only"] if cost_cfg.market_type == "spot" else directions

    rng = random.Random(seed)
    seen_hashes: set[str] = set()
    population: list[StrategySpec] = list(
        stream_candidate_strategies(count=population_size, families=families, directions=effective_dirs, seed=seed)
    )
    for s in population:
        seen_hashes.add(s.ast_hash())

    scored: list[tuple[float, StrategySpec, dict[str, Any]]] = []
    all_results: list[tuple[StrategySpec, dict[str, Any]]] = []

    def _fitness(metrics: dict[str, Any]) -> float:
        trades = metrics.get("total_trades", 0)
        if trades < 5:
            return -999.0
        sharpe = float(metrics.get("sharpe_ratio", 0.0))
        dd = float(metrics.get("max_drawdown_pct", 50.0))
        pf = min(float(metrics.get("profit_factor", 0.0)), 4.0)
        return sharpe + 0.3 * pf - 0.03 * dd

    counter = 0
    for gen in range(generations):
        scored = []
        for spec in population:
            out = evaluate_strategy(spec, ctx, engine_type="vectorized", costs=cost_cfg, funding_rates_map=funding_rates_map)
            fit = _fitness(out.metrics)
            scored.append((fit, spec, out.metrics))
            all_results.append((spec, out.metrics))
            if on_evaluated:
                on_evaluated(spec, out.metrics)

        scored.sort(key=lambda x: x[0], reverse=True)
        if gen == generations - 1:
            break

        # Elitism: preserve top 25%
        elite_n = max(2, population_size // 4)
        next_pop: list[StrategySpec] = [item[1] for item in scored[:elite_n]]
        parents = [item[1] for item in scored[: max(4, population_size // 2)]]

        tries = 0
        while len(next_pop) < population_size and tries < population_size * 10:
            tries += 1
            counter += 1
            if rng.random() < 0.65 and len(parents) >= 2:
                pa, pb = rng.sample(parents, 2)
                child = _crossover_strategies(pa, pb, rng, counter)
                if rng.random() < 0.4:
                    child = _mutate_strategy(child, rng, counter)
            else:
                child = _mutate_strategy(rng.choice(parents), rng, counter)

            try:
                validate_strategy_spec(child)
            except StrategyGrammarError:
                continue
            h = child.ast_hash()
            if h not in seen_hashes:
                seen_hashes.add(h)
                next_pop.append(child)

        # Refill if diversity exhausted
        if len(next_pop) < population_size:
            extra = stream_candidate_strategies(
                count=population_size - len(next_pop),
                families=families,
                directions=directions,
                seed=seed + gen + 100,
            )
            for s in extra:
                if s.ast_hash() not in seen_hashes:
                    seen_hashes.add(s.ast_hash())
                    next_pop.append(s)

        population = next_pop

    all_results.sort(key=lambda x: _fitness(x[1]), reverse=True)
    return all_results
