"""Monte Carlo Simulation Engine (Trade Sequence Permutation, Bootstrap, and Block Bootstrap)."""

from __future__ import annotations

from typing import Any

import numpy as np

from crypto_quant_lab.backtesting.metrics import SimulatedTrade


def run_monte_carlo_simulation(
    trades: list[SimulatedTrade],
    initial_capital: float = 100_000.0,
    simulations: int = 1000,
    method: str = "trade_permutation",  # trade_permutation | bootstrap | block_bootstrap
    ruin_threshold_pct: float = 50.0,
    block_size: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run 1,000+ Monte Carlo simulations to quantify return dispersion, drawdown risk, and ruin probability."""
    if not trades:
        return {
            "method": method,
            "simulations": simulations,
            "median_return_pct": 0.0,
            "p05_return_pct": 0.0,
            "p25_return_pct": 0.0,
            "p75_return_pct": 0.0,
            "p95_return_pct": 0.0,
            "worst_return_pct": 0.0,
            "median_max_dd_pct": 0.0,
            "p95_max_dd_pct": 0.0,
            "worst_max_dd_pct": 0.0,
            "ruin_probability": 1.0,
            "sample_paths": [],
        }

    pnls = np.array([t.net_pnl for t in trades], dtype=np.float64)
    n_trades = len(pnls)
    rng = np.random.default_rng(seed)

    final_returns = np.empty(simulations, dtype=np.float64)
    max_dds = np.empty(simulations, dtype=np.float64)
    ruined_count = 0
    ruin_level = initial_capital * (1.0 - ruin_threshold_pct / 100.0)
    sample_paths: list[list[float]] = []

    for s in range(simulations):
        if method == "trade_permutation":
            seq = rng.permutation(pnls)
        elif method == "block_bootstrap":
            blk = max(1, min(block_size, n_trades))
            indices: list[int] = []
            while len(indices) < n_trades:
                start = int(rng.integers(0, n_trades))
                for k in range(blk):
                    indices.append((start + k) % n_trades)
                    if len(indices) >= n_trades:
                        break
            seq = pnls[indices]
        else:  # bootstrap
            idx = rng.integers(0, n_trades, size=n_trades)
            seq = pnls[idx]

        eq = np.empty(n_trades + 1, dtype=np.float64)
        eq[0] = initial_capital
        np.cumsum(seq, out=eq[1:])
        eq[1:] += initial_capital
        eq = np.maximum(eq, 0.0)

        if np.min(eq) <= ruin_level:
            ruined_count += 1

        peak = np.maximum.accumulate(eq)
        dd = np.where(peak > 0, (peak - eq) / peak * 100.0, 0.0)
        max_dds[s] = float(np.max(dd))
        final_returns[s] = (eq[-1] - initial_capital) / initial_capital * 100.0

        if s < 25:
            step = max(1, len(eq) // 80)
            sample_paths.append([round(float(x), 2) for x in eq[::step]])

    return {
        "method": method,
        "simulations": simulations,
        "median_return_pct": round(float(np.median(final_returns)), 2),
        "p05_return_pct": round(float(np.percentile(final_returns, 5)), 2),
        "p25_return_pct": round(float(np.percentile(final_returns, 25)), 2),
        "p75_return_pct": round(float(np.percentile(final_returns, 75)), 2),
        "p95_return_pct": round(float(np.percentile(final_returns, 95)), 2),
        "worst_return_pct": round(float(np.min(final_returns)), 2),
        "median_max_dd_pct": round(float(np.median(max_dds)), 2),
        "p95_max_dd_pct": round(float(np.percentile(max_dds, 95)), 2),
        "worst_max_dd_pct": round(float(np.max(max_dds)), 2),
        "ruin_probability": round(ruined_count / simulations, 4),
        "sample_paths": sample_paths,
    }
