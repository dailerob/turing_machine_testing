"""
Shared evaluation primitives for the HMM forecasting comparison.

Given a fitted model, test prefixes, and ground-truth HMM, compute the
mean squared error (MSE) between the model's predicted next-symbol
distribution and the HMM's exact posterior predictive distribution, at
a range of forecasting horizons.

MSE is summed over alphabet per prefix, then averaged across prefixes.
"""

from __future__ import annotations

import numpy as np


def mse_at_horizons(model, hmm, test_prefixes, horizons):
    """Return dict {h: mean_squared_error} for the given model/hmm.

    model : must implement .horizon_emission(prefix, h) -> shape (nA,)
    hmm   : RandomHMM
    test_prefixes : list of 1-d int obs arrays
    horizons : iterable of positive ints
    """
    result = {h: [] for h in horizons}
    for prefix in test_prefixes:
        alpha = hmm.filter(prefix)
        for h in horizons:
            true_dist = hmm.horizon_emission(alpha, h)
            pred_dist = model.horizon_emission(prefix, h)
            result[h].append(float(np.mean((pred_dist - true_dist) ** 2)))
    return {h: float(np.mean(v)) for h, v in result.items()}


def uniform_baseline_mse(hmm, test_prefixes, horizons):
    """MSE of always predicting the uniform distribution."""
    nA = hmm.nA
    uniform = np.full(nA, 1.0 / nA)
    result = {h: [] for h in horizons}
    for prefix in test_prefixes:
        alpha = hmm.filter(prefix)
        for h in horizons:
            true_dist = hmm.horizon_emission(alpha, h)
            result[h].append(float(np.mean((uniform - true_dist) ** 2)))
    return {h: float(np.mean(v)) for h, v in result.items()}


def stationary_baseline_mse(hmm, test_prefixes, horizons):
    """MSE of predicting the marginal emission distribution under the
    stationary distribution of T."""
    # Stationary distribution via power iteration (robust for small n).
    T = hmm.T
    pi_stat = np.full(T.shape[0], 1.0 / T.shape[0])
    for _ in range(500):
        nxt = pi_stat @ T
        if np.linalg.norm(nxt - pi_stat) < 1e-12:
            pi_stat = nxt
            break
        pi_stat = nxt
    pi_stat = np.maximum(pi_stat, 0)
    s = pi_stat.sum()
    pi_stat = pi_stat / s if s > 0 else np.full(T.shape[0], 1.0 / T.shape[0])
    marginal = pi_stat @ hmm.E
    result = {h: [] for h in horizons}
    for prefix in test_prefixes:
        alpha = hmm.filter(prefix)
        for h in horizons:
            true_dist = hmm.horizon_emission(alpha, h)
            result[h].append(float(np.mean((marginal - true_dist) ** 2)))
    return {h: float(np.mean(v)) for h, v in result.items()}
