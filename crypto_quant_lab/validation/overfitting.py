"""Overfitting Detection: CSCV PBO, Deflated Sharpe Ratio, FDR, and Parameter Stability."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from scipy import stats

from crypto_quant_lab.backtesting.costs import ExecutionCostConfig
from crypto_quant_lab.strategies.evaluator import IndicatorCache, evaluate_strategy
from crypto_quant_lab.strategies.grammar import StrategySpec


def compute_deflated_sharpe_ratio(
    observed_sharpe: float,
    bar_returns: np.ndarray,
    n_trials: int = 1,
    bars_per_year: float = 365.0 * 24.0,
) -> dict[str, float]:
    """
    Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR).
    Accounts for non-normal skewness/kurtosis and multiple testing across `n_trials`.
    """
    r = np.asarray(bar_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    t = len(r)
    if t < 10:
        return {"dsr": 0.0, "sharpe_benchmark": 0.0, "pvalue": 1.0}

    skew = float(stats.skew(r)) if np.std(r) > 1e-12 else 0.0
    kurt = float(stats.kurtosis(r, fisher=False)) if np.std(r) > 1e-12 else 3.0

    # Expected maximum Sharpe ratio under null across N independent trials (in non-annualized or annualized units)
    # Convert observed annualized SR to per-bar SR for exact Bailey-LdP formula
    sr_bar = observed_sharpe / np.sqrt(bars_per_year)
    n_eff = max(1, n_trials)
    if n_eff == 1:
        sr_zero = 0.0
    else:
        euler_gamma = 0.5772156649
        z1 = stats.norm.ppf(1.0 - 1.0 / n_eff)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_eff * np.e))
        # Estimate cross-trial SR std conservatively from observed SR variance scaling
        sr_std = 1.0 / np.sqrt(max(t - 1, 1))
        sr_zero = sr_std * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)

    denom_inner = max(1.0 - skew * sr_bar + ((kurt - 1.0) / 4.0) * (sr_bar**2), 1e-6)
    z_stat = ((sr_bar - sr_zero) * np.sqrt(max(t - 1, 1))) / np.sqrt(denom_inner)
    dsr_prob = float(stats.norm.cdf(z_stat))

    return {
        "dsr": round(dsr_prob, 4),
        "sharpe_benchmark": round(float(sr_zero * np.sqrt(bars_per_year)), 4),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
    }


def compute_cscv_pbo(return_matrix: np.ndarray, n_partitions: int = 8) -> dict[str, Any]:
    """
    Combinatorially Symmetric Cross-Validation (CSCV) Probability of Backtest Overfitting (PBO).
    `return_matrix` has shape (T, M) where T is time steps and M >= 2 is strategy configurations.
    """
    mat = np.asarray(return_matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] < 2 or mat.shape[0] < n_partitions * 4:
        return {"pbo": 0.5, "combinations": 0, "median_logit": 0.0}

    if n_partitions % 2 != 0:
        n_partitions -= 1
    n_partitions = max(4, min(n_partitions, 10))

    t_len, m_strats = mat.shape
    chunk_size = t_len // n_partitions
    chunks = [mat[k * chunk_size : (k + 1) * chunk_size, :] for k in range(n_partitions)]

    half = n_partitions // 2
    logits: list[float] = []
    overfit_count = 0

    for train_idx in itertools.combinations(range(n_partitions), half):
        test_idx = [i for i in range(n_partitions) if i not in train_idx]
        is_mat = np.vstack([chunks[i] for i in train_idx])
        oos_mat = np.vstack([chunks[i] for i in test_idx])

        is_std = np.std(is_mat, axis=0, ddof=1)
        is_sr = np.full(m_strats, -1e9, dtype=np.float64)
        np.divide(np.mean(is_mat, axis=0), is_std, out=is_sr, where=is_std > 1e-12)
        best_col = int(np.argmax(is_sr))

        oos_std = np.std(oos_mat, axis=0, ddof=1)
        oos_sr = np.full(m_strats, -1e9, dtype=np.float64)
        np.divide(np.mean(oos_mat, axis=0), oos_std, out=oos_sr, where=oos_std > 1e-12)

        # Rank of IS-best strategy in OOS (1 = worst, M = best)
        ranks = stats.rankdata(oos_sr, method="average")
        rel_rank = float(ranks[best_col]) / (m_strats + 1.0)
        rel_rank = float(np.clip(rel_rank, 1e-4, 1.0 - 1e-4))
        logit = float(np.log(rel_rank / (1.0 - rel_rank)))
        logits.append(logit)
        if logit <= 0.0:
            overfit_count += 1

    pbo = overfit_count / len(logits) if logits else 0.5
    return {
        "pbo": round(pbo, 4),
        "combinations": len(logits),
        "median_logit": round(float(np.median(logits)) if logits else 0.0, 4),
    }


def benjamini_hochberg_fdr(p_values: list[float]) -> list[float]:
    """Compute Benjamini-Hochberg False Discovery Rate adjusted q-values."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    q_vals = [1.0] * m
    min_q = 1.0
    for rank_rev in range(m, 0, -1):
        orig_idx, p = indexed[rank_rev - 1]
        q = min(min_q, (p * m) / rank_rev)
        min_q = q
        q_vals[orig_idx] = round(float(np.clip(q, 0.0, 1.0)), 6)
    return q_vals


def parameter_stability_analysis(
    ctx: IndicatorCache,
    spec: StrategySpec,
    costs: ExecutionCostConfig | None = None,
) -> dict[str, Any]:
    """Evaluate neighboring parameter perturbations (SL/TP/ATR) for robustness plateau vs sharp peak."""
    base_out = evaluate_strategy(spec, ctx, engine_type="vectorized", costs=costs)
    base_sharpe = float(base_out.metrics["sharpe_ratio"])

    sl_deltas = [-0.5, 0.0, 0.5]
    tp_deltas = [-0.5, 0.0, 0.5]
    neighbor_sharpes: list[float] = []
    neighbor_returns: list[float] = []
    return_matrix_cols: list[np.ndarray] = []

    for d_sl in sl_deltas:
        for d_tp in tp_deltas:
            sl_val = max(0.5, spec.sl_atr_mult + d_sl)
            tp_val = max(0.75, spec.tp_atr_mult + d_tp)
            pert = StrategySpec(
                name=spec.name,
                family=spec.family,
                direction=spec.direction,
                long_entry=spec.long_entry,
                long_exit=spec.long_exit,
                short_entry=spec.short_entry,
                short_exit=spec.short_exit,
                sl_atr_mult=sl_val,
                tp_atr_mult=tp_val,
                atr_period=spec.atr_period,
                max_holding_bars=spec.max_holding_bars,
                risk_per_trade_pct=spec.risk_per_trade_pct,
            )
            out = evaluate_strategy(pert, ctx, engine_type="vectorized", costs=costs)
            neighbor_sharpes.append(float(out.metrics["sharpe_ratio"]))
            neighbor_returns.append(float(out.metrics["return_on_capital_pct"]))
            return_matrix_cols.append(out.bar_returns)

    positive_frac = sum(1 for s in neighbor_sharpes if s > 0) / len(neighbor_sharpes)
    mean_sh = float(np.mean(neighbor_sharpes))
    std_sh = float(np.std(neighbor_sharpes))

    # Stability score in [0, 1]: rewards consistent positive neighbors & low dispersion relative to base
    if base_sharpe <= 0:
        stability_score = max(0.0, positive_frac * 0.5)
    else:
        ratio = np.clip(mean_sh / max(abs(base_sharpe), 1e-6), 0.0, 1.2)
        disp_penalty = np.clip(std_sh / max(abs(mean_sh) + 0.5, 0.5), 0.0, 1.0)
        stability_score = float(np.clip(0.5 * positive_frac + 0.5 * ratio * (1.0 - 0.5 * disp_penalty), 0.0, 1.0))

    min_len = min(len(c) for c in return_matrix_cols) if return_matrix_cols else 0
    ret_mat = np.column_stack([c[:min_len] for c in return_matrix_cols]) if min_len > 10 else np.empty((0, 0))
    cscv = compute_cscv_pbo(ret_mat) if ret_mat.size > 0 else {"pbo": 0.5, "combinations": 0, "median_logit": 0.0}

    return {
        "stability_score": round(stability_score, 4),
        "positive_neighbor_pct": round(positive_frac * 100.0, 2),
        "neighbor_sharpe_mean": round(mean_sh, 4),
        "neighbor_sharpe_std": round(std_sh, 4),
        "neighbor_sharpes": [round(s, 3) for s in neighbor_sharpes],
        "neighbor_returns": [round(r, 2) for r in neighbor_returns],
        "cscv_pbo": cscv,
    }
