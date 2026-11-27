"""Configurable Strategy Ranking Engine with Hard Rejection Gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RankingWeights:
    oos_sharpe: float = 0.25
    profit_factor: float = 0.15
    drawdown: float = 0.20
    walk_forward: float = 0.20
    monte_carlo: float = 0.10
    param_stability: float = 0.10


@dataclass
class HardRejectionGates:
    min_trades: int = 10
    max_drawdown_pct: float = 35.0
    min_oos_sharpe: float = 0.20
    max_pbo: float = 0.55
    min_dsr: float = 0.20
    min_stress_survival: float = 0.40


def evaluate_and_rank_strategy(
    oos_metrics: dict[str, Any],
    wf_result: dict[str, Any],
    mc_result: dict[str, Any],
    stability_result: dict[str, Any],
    stress_result: dict[str, Any],
    dsr_result: dict[str, Any],
    weights: RankingWeights | None = None,
    gates: HardRejectionGates | None = None,
) -> dict[str, Any]:
    """Apply hard rejection rules first, then compute weighted multi-dimensional composite score."""
    w = weights or RankingWeights()
    g = gates or HardRejectionGates()

    reasons: list[str] = []
    trades = int(oos_metrics.get("total_trades", 0))
    oos_sharpe = float(oos_metrics.get("sharpe_ratio", 0.0))
    oos_dd = float(oos_metrics.get("max_drawdown_pct", 100.0))
    pf = float(oos_metrics.get("profit_factor", 0.0))
    pbo = float(stability_result.get("cscv_pbo", {}).get("pbo", 0.5))
    dsr = float(dsr_result.get("dsr", 0.0))
    stress_rate = float(stress_result.get("survival_rate", 0.0))

    if trades < g.min_trades:
        reasons.append(f"Insufficient OOS trades ({trades} < {g.min_trades})")
    if oos_dd > g.max_drawdown_pct:
        reasons.append(f"Excessive OOS drawdown ({oos_dd:.1f}% > {g.max_drawdown_pct:.1f}%)")
    if oos_sharpe < g.min_oos_sharpe:
        reasons.append(f"OOS Sharpe below gate ({oos_sharpe:.2f} < {g.min_oos_sharpe:.2f})")
    if pbo > g.max_pbo:
        reasons.append(f"High overfitting risk PBO ({pbo:.2f} > {g.max_pbo:.2f})")
    if dsr < g.min_dsr:
        reasons.append(f"Deflated Sharpe too low ({dsr:.2f} < {g.min_dsr:.2f})")
    if stress_rate < g.min_stress_survival:
        reasons.append(f"Failed stress tests ({stress_rate:.2f} < {g.min_stress_survival:.2f})")

    passed = len(reasons) == 0

    # Normalize each dimension into [0, 100]
    s_sharpe = max(0.0, min(100.0, oos_sharpe / 3.0 * 100.0))
    s_pf = max(0.0, min(100.0, (pf - 0.8) / 2.2 * 100.0))
    s_dd = max(0.0, min(100.0, (1.0 - oos_dd / 50.0) * 100.0))
    s_wf = max(0.0, min(100.0, float(wf_result.get("consistency_pct", 0.0))))
    mc_med = float(mc_result.get("median_return_pct", 0.0))
    mc_ruin = float(mc_result.get("ruin_probability", 1.0))
    s_mc = max(0.0, min(100.0, (1.0 - mc_ruin) * 60.0 + max(0.0, min(40.0, mc_med))))
    s_stab = max(0.0, min(100.0, float(stability_result.get("stability_score", 0.0)) * 100.0))

    total_w = max(w.oos_sharpe + w.profit_factor + w.drawdown + w.walk_forward + w.monte_carlo + w.param_stability, 1e-9)
    composite = (
        w.oos_sharpe * s_sharpe
        + w.profit_factor * s_pf
        + w.drawdown * s_dd
        + w.walk_forward * s_wf
        + w.monte_carlo * s_mc
        + w.param_stability * s_stab
    ) / total_w

    return {
        "passed_gates": passed,
        "rejection_reasons": reasons,
        "composite_score": round(composite, 2),
        "dimension_scores": {
            "oos_sharpe": round(s_sharpe, 1),
            "profit_factor": round(s_pf, 1),
            "drawdown": round(s_dd, 1),
            "walk_forward": round(s_wf, 1),
            "monte_carlo": round(s_mc, 1),
            "param_stability": round(s_stab, 1),
        },
    }
