"""Walk-Forward Analysis (Rolling, Expanding, and Anchored modes)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.strategies.evaluator import IndicatorCache, evaluate_strategy
from crypto_quant_lab.strategies.grammar import StrategySpec


def run_walk_forward_analysis(
    df: pl.DataFrame,
    spec: StrategySpec,
    timeframe: str = "1h",
    mode: str = "rolling",  # rolling | expanding | anchored
    num_windows: int = 5,
    train_frac: float = 0.65,
    costs: ExecutionCostConfig | None = None,
) -> dict[str, Any]:
    """
    Execute multi-window Walk-Forward evaluation across rolling, expanding, or anchored windows.
    """
    ordered = df.sort("open_time_ms")
    n = ordered.height
    if n < max(120, num_windows * 25):
        raise ValueError(f"Insufficient candles ({n}) for {num_windows}-window walk-forward analysis")

    # Determine window slicing
    total_test_bars = int(n * (1.0 - train_frac))
    test_step = max(15, total_test_bars // num_windows)
    initial_train_end = n - test_step * num_windows
    if initial_train_end < 40:
        initial_train_end = max(40, n // 2)
        test_step = max(10, (n - initial_train_end) // num_windows)

    windows: list[dict[str, Any]] = []
    is_sharpes: list[float] = []
    oos_sharpes: list[float] = []
    oos_returns: list[float] = []

    for w in range(num_windows):
        oos_start = initial_train_end + w * test_step
        oos_end = min(n, oos_start + test_step)
        if oos_end - oos_start < 10:
            break

        if mode == "rolling":
            train_start = max(0, oos_start - initial_train_end)
        else:  # expanding or anchored
            train_start = 0

        train_slice = ordered.slice(train_start, oos_start - train_start)
        oos_slice = ordered.slice(oos_start, oos_end - oos_start)

        is_ctx = IndicatorCache(train_slice, timeframe=timeframe)
        oos_ctx = IndicatorCache(oos_slice, timeframe=timeframe)

        is_res = evaluate_strategy(spec, is_ctx, engine_type="vectorized", costs=costs)
        oos_res = evaluate_strategy(spec, oos_ctx, engine_type="vectorized", costs=costs)

        is_sh = float(is_res.metrics["sharpe_ratio"])
        oos_sh = float(oos_res.metrics["sharpe_ratio"])
        oos_ret = float(oos_res.metrics["return_on_capital_pct"])

        is_sharpes.append(is_sh)
        oos_sharpes.append(oos_sh)
        oos_returns.append(oos_ret)

        windows.append(
            {
                "window": w + 1,
                "train_bars": train_slice.height,
                "oos_bars": oos_slice.height,
                "is_sharpe": round(is_sh, 4),
                "is_return_pct": round(float(is_res.metrics["return_on_capital_pct"]), 4),
                "oos_sharpe": round(oos_sh, 4),
                "oos_return_pct": round(oos_ret, 4),
                "oos_max_dd_pct": round(float(oos_res.metrics["max_drawdown_pct"]), 4),
                "oos_trades": int(oos_res.metrics["total_trades"]),
            }
        )

    if not windows:
        return {
            "mode": mode,
            "num_windows": 0,
            "consistency_pct": 0.0,
            "avg_oos_sharpe": 0.0,
            "wf_efficiency_ratio": 0.0,
            "windows": [],
        }

    profitable_windows = sum(1 for r in oos_returns if r > 0)
    consistency_pct = profitable_windows / len(windows) * 100.0
    avg_is = float(np.mean(is_sharpes))
    avg_oos = float(np.mean(oos_sharpes))
    wf_eff = (avg_oos / avg_is) if abs(avg_is) > 1e-6 else (1.0 if avg_oos > 0 else 0.0)

    return {
        "mode": mode,
        "num_windows": len(windows),
        "consistency_pct": round(consistency_pct, 2),
        "avg_is_sharpe": round(avg_is, 4),
        "avg_oos_sharpe": round(avg_oos, 4),
        "wf_efficiency_ratio": round(float(np.clip(wf_eff, -5.0, 5.0)), 4),
        "windows": windows,
    }
