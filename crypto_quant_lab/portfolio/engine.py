"""Portfolio Construction Engine with Correlation, Clustering, Risk Parity, Min Variance, and Drawdown Controls."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def construct_portfolio(
    strategy_ids: list[int],
    strategy_names: list[str],
    return_series_list: list[np.ndarray],
    allocation_method: str = "risk_parity",  # equal_weight | volatility_weighted | risk_parity | min_variance | sharpe_weighted | max_dd_constrained
    initial_capital: float = 100_000.0,
    max_single_weight: float = 0.40,
    max_cluster_weight: float = 0.60,
    target_vol_annual: float = 0.15,
) -> dict[str, Any]:
    """
    Evaluate strategy independence via return & drawdown correlations + hierarchical clustering,
    then optimize weights and compute portfolio equity curve & risk contributions.
    """
    m = len(strategy_ids)
    if m == 0:
        raise ValueError("At least one strategy is required for portfolio construction")

    min_len = min(len(r) for r in return_series_list)
    if min_len < 5:
        raise ValueError("Strategy return series too short to construct portfolio")

    ret_mat = np.column_stack([np.asarray(r[-min_len:], dtype=np.float64) for r in return_series_list])
    ret_mat = np.nan_to_num(ret_mat, nan=0.0, posinf=0.0, neginf=0.0)

    # Return correlation matrix
    if m == 1:
        corr_mat = np.array([[1.0]])
        dd_corr_mat = np.array([[1.0]])
        clusters = [1]
    else:
        corr_mat = np.corrcoef(ret_mat, rowvar=False)
        corr_mat = np.nan_to_num(corr_mat, nan=0.0)
        np.fill_diagonal(corr_mat, 1.0)

        # Drawdown series per strategy
        dd_cols = []
        for j in range(m):
            eq_j = np.cumprod(1.0 + ret_mat[:, j])
            pk_j = np.maximum.accumulate(eq_j)
            dd_j = (pk_j - eq_j) / np.maximum(pk_j, 1e-9)
            dd_cols.append(dd_j)
        dd_mat = np.column_stack(dd_cols)
        dd_corr_mat = np.corrcoef(dd_mat, rowvar=False)
        dd_corr_mat = np.nan_to_num(dd_corr_mat, nan=0.0)
        np.fill_diagonal(dd_corr_mat, 1.0)

        # Hierarchical clustering on distance sqrt(0.5 * (1 - corr))
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr_mat), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="ward")
        n_clust = min(m, max(2, m // 2))
        clusters = [int(c) for c in fcluster(Z, t=n_clust, criterion="maxclust")]

    cov = np.cov(ret_mat, rowvar=False) if m > 1 else np.array([[np.var(ret_mat[:, 0])]])
    cov = np.nan_to_num(cov, nan=0.0) + np.eye(m) * 1e-10
    vols = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    means = np.mean(ret_mat, axis=0)

    # Optimize weights
    if m == 1:
        weights = np.array([1.0])
    elif allocation_method == "equal_weight":
        weights = np.full(m, 1.0 / m)
    elif allocation_method == "volatility_weighted":
        inv_v = 1.0 / vols
        weights = inv_v / np.sum(inv_v)
    elif allocation_method == "sharpe_weighted":
        sr = np.maximum(means / vols, 0.01)
        weights = sr / np.sum(sr)
    elif allocation_method == "min_variance":
        bounds = [(0.0, max_single_weight) for _ in range(m)]
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = optimize.minimize(lambda w: float(w @ cov @ w), np.full(m, 1.0 / m), bounds=bounds, constraints=cons)
        weights = res.x if res.success else np.full(m, 1.0 / m)
    elif allocation_method == "max_dd_constrained":
        def _obj_dd(w: np.ndarray) -> float:
            p_ret = ret_mat @ w
            eq = np.cumprod(1.0 + p_ret)
            pk = np.maximum.accumulate(eq)
            max_dd = float(np.max((pk - eq) / np.maximum(pk, 1e-9)))
            sharpe_proxy = float(np.mean(p_ret) / max(np.std(p_ret), 1e-9))
            return max_dd * 10.0 - sharpe_proxy

        bounds = [(0.0, max_single_weight) for _ in range(m)]
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = optimize.minimize(_obj_dd, np.full(m, 1.0 / m), bounds=bounds, constraints=cons, method="SLSQP")
        weights = res.x if res.success else np.full(m, 1.0 / m)
    else:  # risk_parity
        def _rp_obj(w: np.ndarray) -> float:
            port_var = float(w @ cov @ w)
            mrc = cov @ w
            rc = w * mrc / max(np.sqrt(port_var), 1e-12)
            target_rc = np.sqrt(port_var) / m
            return float(np.sum((rc - target_rc) ** 2))

        bounds = [(1e-4, max_single_weight) for _ in range(m)]
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = optimize.minimize(_rp_obj, np.full(m, 1.0 / m), bounds=bounds, constraints=cons)
        weights = res.x if res.success else (1.0 / vols) / np.sum(1.0 / vols)

    # Cap concentration per cluster
    if m > 1:
        for cid in set(clusters):
            c_idx = [i for i, c in enumerate(clusters) if c == cid]
            c_sum = float(np.sum(weights[c_idx]))
            if c_sum > max_cluster_weight and c_sum > 0:
                weights[c_idx] *= max_cluster_weight / c_sum
        weights = np.maximum(weights, 0.0)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.full(m, 1.0 / m)

    # Risk contributions
    port_var = float(weights @ cov @ weights)
    port_std = np.sqrt(max(port_var, 1e-12))
    mrc = cov @ weights
    rc_pct = (weights * mrc / port_var * 100.0) if port_var > 1e-12 else np.full(m, 100.0 / m)

    # Volatility scaling factor (capped at 1.0x unlevered unless explicitly scaled)
    ann_port_vol = port_std * np.sqrt(365.0 * 24.0)
    vol_scale = min(1.0, target_vol_annual / max(ann_port_vol, 1e-6))

    port_rets = (ret_mat @ weights) * vol_scale
    eq = np.empty(len(port_rets) + 1, dtype=np.float64)
    eq[0] = initial_capital
    eq[1:] = initial_capital * np.cumprod(1.0 + port_rets)

    peak = np.maximum.accumulate(eq)
    dd_series = (peak - eq) / np.maximum(peak, 1e-9) * 100.0
    max_dd = float(np.max(dd_series))
    total_ret = (eq[-1] - initial_capital) / initial_capital * 100.0
    p_std = float(np.std(port_rets, ddof=1))
    sharpe = (float(np.mean(port_rets)) / p_std * np.sqrt(365.0 * 24.0)) if p_std > 1e-12 else 0.0

    step = max(1, len(eq) // 300)
    eq_points = [round(float(v), 2) for v in eq[::step]]

    return {
        "allocation_method": allocation_method,
        "vol_scale": round(float(vol_scale), 4),
        "allocations": [
            {
                "experiment_id": strategy_ids[i],
                "name": strategy_names[i],
                "weight": round(float(weights[i]), 4),
                "weight_pct": round(float(weights[i] * 100.0), 2),
                "risk_contribution_pct": round(float(rc_pct[i]), 2),
                "cluster_id": clusters[i],
            }
            for i in range(m)
        ],
        "return_correlation_matrix": [[round(float(x), 3) for x in row] for row in corr_mat],
        "drawdown_correlation_matrix": [[round(float(x), 3) for x in row] for row in dd_corr_mat],
        "metrics": {
            "total_return_pct": round(total_ret, 2),
            "sharpe_ratio": round(float(sharpe), 4),
            "max_drawdown_pct": round(max_dd, 2),
            "annualized_vol_pct": round(float(ann_port_vol * vol_scale * 100.0), 2),
            "final_equity": round(float(eq[-1]), 2),
        },
        "equity_curve": eq_points,
    }
