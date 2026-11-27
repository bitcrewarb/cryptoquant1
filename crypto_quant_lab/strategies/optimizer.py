"""Optuna Bayesian Optimization engine for quantitative strategy discovery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optuna

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.core.exceptions import StrategyGrammarError
from crypto_quant_lab.strategies.evaluator import IndicatorCache, evaluate_strategy
from crypto_quant_lab.strategies.generator import build_family_condition_pools
from crypto_quant_lab.strategies.grammar import LogicalExpr, StrategySpec, validate_strategy_spec

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_optimization(
    ctx: IndicatorCache,
    n_trials: int = 50,
    families: list[str] | None = None,
    directions: list[str] | None = None,
    seed: int = 42,
    on_evaluated: Callable[[StrategySpec, dict[str, Any]], None] | None = None,
    costs: ExecutionCostConfig | None = None,
    funding_rates_map: dict[int, float] | None = None,
) -> list[tuple[StrategySpec, dict[str, Any]]]:
    """Run Bayesian TPE optimization via Optuna over rules & risk parameters."""
    active_families = families or ["trend_following", "mean_reversion", "momentum_breakout", "bos_choch"]
    inferred_mkt = "futures" if (directions is None or any(d != "long_only" for d in directions)) else "spot"
    cost_cfg = costs or ExecutionCostConfig(market_type=inferred_mkt)
    active_dirs = ["long_only"] if cost_cfg.market_type == "spot" else (directions or ["long_only", "long_short"])
    seen_hashes: set[str] = set()
    results: list[tuple[StrategySpec, dict[str, Any]]] = []

    pools = {
        (fam, side): build_family_condition_pools(fam, side)
        for fam in active_families
        for side in ("long", "short")
    }

    def objective(trial: optuna.Trial) -> float:
        fam = trial.suggest_categorical("family", active_families)
        direction = trial.suggest_categorical("direction", active_dirs)
        lp1, lp2, lp3 = pools[(fam, "long")]
        sp1, sp2, _ = pools[(fam, "short")]

        i_l1 = trial.suggest_int(f"{fam}_l1", 0, len(lp1) - 1)
        i_l2 = trial.suggest_int(f"{fam}_l2", 0, len(lp2) - 1)
        i_l3 = trial.suggest_int(f"{fam}_l3", 0, len(lp3) - 1)
        i_s1 = trial.suggest_int(f"{fam}_s1", 0, len(sp1) - 1)
        i_s2 = trial.suggest_int(f"{fam}_s2", 0, len(sp2) - 1)

        sl_mult = trial.suggest_float("sl_atr_mult", 1.0, 3.5, step=0.25)
        tp_mult = trial.suggest_float("tp_atr_mult", 1.5, 6.0, step=0.5)
        atr_p = trial.suggest_categorical("atr_period", [7, 10, 14, 20, 28])

        spec = StrategySpec(
            name=f"{fam}_optuna_{trial.number:04d}",
            family=fam,
            direction=direction,
            long_entry=LogicalExpr("AND", (lp1[i_l1], lp2[i_l2], lp3[i_l3])) if direction != "short_only" else None,
            long_exit=None,
            short_entry=LogicalExpr("AND", (sp1[i_s1], sp2[i_s2])) if direction != "long_only" else None,
            short_exit=None,
            sl_atr_mult=sl_mult,
            tp_atr_mult=tp_mult,
            atr_period=atr_p,
        )
        try:
            validate_strategy_spec(spec, market_type=cost_cfg.market_type)
        except StrategyGrammarError:
            return -500.0

        out = evaluate_strategy(spec, ctx, engine_type="vectorized", costs=cost_cfg, funding_rates_map=funding_rates_map)
        h = spec.ast_hash()
        if h not in seen_hashes:
            seen_hashes.add(h)
            results.append((spec, out.metrics))
            if on_evaluated:
                on_evaluated(spec, out.metrics)

        if out.metrics["total_trades"] < 5:
            return -100.0
        return float(out.metrics["sharpe_ratio"]) - 0.02 * float(out.metrics["max_drawdown_pct"])

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return results
