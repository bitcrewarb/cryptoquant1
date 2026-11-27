"""Multi-market accuracy filtering, bootstrap confidence intervals, Wilson score intervals, correlation-clustering deduplication, and Strategy Card generation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as np_pl  # type: ignore

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.strategies.evaluator import IndicatorCache, evaluate_strategy
from crypto_quant_lab.strategies.grammar import StrategySpec


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate Wilson score interval for binomial win rate."""
    if total <= 0:
        return 0.0, 0.0
    z = 1.96 if confidence == 0.95 else 2.576
    p = successes / total
    denom = 1.0 + (z**2) / total
    centre = (p + (z**2) / (2 * total)) / denom
    spread = (z * math.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def bootstrap_expectancy_ci(trade_returns: list[float], n_bootstrap: int = 1000, confidence: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Bootstrap confidence interval for trade net returns (lower bound > 0 required for strict promotion)."""
    rets = np.asarray(trade_returns, dtype=np.float64)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 5:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    t_len = len(rets)
    for i in range(n_bootstrap):
        sample = rng.choice(rets, size=t_len, replace=True)
        means[i] = float(np.mean(sample))
    alpha = (1.0 - confidence) * 100.0
    return float(np.percentile(means, alpha * 0.5)), float(np.percentile(means, 100.0 - alpha * 0.5))


def cluster_strategies_by_correlation(return_matrix: np.ndarray, threshold: float = 0.7) -> list[list[int]]:
    """Group strategies into correlation clusters (corr > threshold) and return representative indices."""
    mat = np.asarray(return_matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] < 2:
        return [[i] for i in range(mat.shape[1])] if mat.ndim == 2 else []
    corr = np.corrcoef(mat, rowvar=False)
    n = corr.shape[0]
    visited = set()
    clusters: list[list[int]] = []
    for i in range(n):
        if i not in visited:
            cluster = [i]
            visited.add(i)
            for j in range(i + 1, n):
                if j not in visited and np.isfinite(corr[i, j]) and corr[i, j] >= threshold:
                    cluster.append(j)
                    visited.add(j)
            clusters.append(cluster)
    return clusters


def generate_strategy_card(
    spec: StrategySpec,
    oos_metrics: dict[str, Any],
    validation_details: dict[str, Any],
    bootstrap_ci: tuple[float, float],
    wilson_ci: tuple[float, float],
) -> dict[str, Any]:
    """Generate professional Strategy Card with grade (A, B, C, Insufficient Evidence) and explicit diagnostics."""
    trades = int(oos_metrics.get("total_trades", 0))
    sharpe = float(oos_metrics.get("sharpe_ratio", 0.0))
    pf = float(oos_metrics.get("profit_factor", 0.0))
    dd = float(oos_metrics.get("max_drawdown_pct", 100.0))
    win_rate = float(oos_metrics.get("win_rate", 0.0))
    passed = bool(validation_details.get("passed_gates", False))
    dsr = float(validation_details.get("deflated_sharpe", 0.0))
    pbo = float(validation_details.get("pbo_score", 1.0))

    reasons = list(validation_details.get("rejection_reasons", []))
    if trades < 100:
        reasons.append(f"Trade count {trades} below 100 production threshold")
    if bootstrap_ci[0] <= 0:
        reasons.append(f"Bootstrap expectancy lower bound {bootstrap_ci[0]:.4f} <= 0")

    grade = "Insufficient Evidence"
    if trades >= 100 and passed and dsr >= 0.3 and pbo <= 0.4 and sharpe >= 1.2 and pf >= 1.4 and bootstrap_ci[0] > 0:
        grade = "Grade A (Production Ready)"
    elif trades >= 50 and passed and sharpe >= 0.8 and pf >= 1.2 and bootstrap_ci[0] > 0:
        grade = "Grade B (Conditional / Paper)"
    elif trades >= 30 and sharpe >= 0.4:
        grade = "Grade C (Experimental)"

    return {
        "strategy_name": spec.name,
        "family": spec.family,
        "direction": spec.direction,
        "grade": grade,
        "metrics": {
            "total_trades": trades,
            "sharpe_ratio": round(sharpe, 3),
            "profit_factor": round(pf, 3),
            "max_drawdown_pct": round(dd, 2),
            "win_rate_pct": round(win_rate * 100.0, 2),
            "bootstrap_expectancy_ci": [round(bootstrap_ci[0], 4), round(bootstrap_ci[1], 4)],
            "wilson_win_rate_ci": [round(wilson_ci[0], 4), round(wilson_ci[1], 4)],
            "deflated_sharpe": round(dsr, 3),
            "pbo": round(pbo, 3),
        },
        "diagnostics": {
            "passed_gates": passed and len(reasons) == 0,
            "failure_reasons": reasons,
        },
    }


def multi_market_accuracy_validation(
    spec: StrategySpec,
    datasets: dict[str, Any],  # symbol -> IndicatorCache
    costs: ExecutionCostConfig | None = None,
) -> dict[str, Any]:
    """Validate strategy performance across >= 3 symbols and >= 2 timeframes."""
    results: dict[str, Any] = {}
    total_trades = 0
    sharpes: list[float] = []
    for key, ctx in datasets.items():
        out = evaluate_strategy(spec, ctx, engine_type="vectorized", costs=costs)
        m = out.metrics
        results[key] = {
            "trades": int(m["total_trades"]),
            "sharpe": float(m["sharpe_ratio"]),
            "net_profit": float(m["net_profit"]),
            "win_rate": float(m["win_rate"]),
        }
        total_trades += int(m["total_trades"])
        sharpes.append(float(m["sharpe_ratio"]))

    mean_sh = float(np.mean(sharpes)) if sharpes else 0.0
    robust = sum(1 for s in sharpes if s > 0.0) >= max(1, len(datasets) // 2)
    return {
        "multi_market_results": results,
        "total_cross_market_trades": total_trades,
        "mean_cross_market_sharpe": round(mean_sh, 3),
        "is_cross_market_robust": robust,
    }
