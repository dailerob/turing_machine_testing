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


def perplexity_at_horizons(model, hmm, test_prefixes, horizons,
                           eps=1e-12):
    """Soft cross-entropy and perplexity, mirroring PAutomaC scoring
    on predictive distributions instead of full sequences.

    For each (prefix, h) compute
        CE_h = -Σ_a true_dist[a] · log2 model_dist[a]
    averaged across prefixes per horizon.  The minimum is the entropy
    of the true posterior (achieved when model = truth).

    Returns dict {h: {cross_entropy_bits, entropy_floor_bits,
                       perplexity, entropy_floor_perplexity,
                       excess_perplexity}}.

    `excess_perplexity` = perplexity / entropy_floor_perplexity,
    lower bound 1.0; this is the closest analog to PAutomaC's
    "gap to entropy floor" reported elsewhere.
    """
    ce_per_h = {h: [] for h in horizons}
    floor_per_h = {h: [] for h in horizons}
    for prefix in test_prefixes:
        alpha = hmm.filter(prefix)
        for h in horizons:
            true_dist = hmm.horizon_emission(alpha, h)
            pred_dist = model.horizon_emission(prefix, h)
            pred_safe = np.maximum(pred_dist, eps)
            true_safe = np.maximum(true_dist, eps)
            ce = -float(np.sum(true_dist * np.log2(pred_safe)))
            floor = -float(np.sum(true_dist * np.log2(true_safe)))
            ce_per_h[h].append(ce)
            floor_per_h[h].append(floor)
    out = {}
    for h in horizons:
        ce = float(np.mean(ce_per_h[h]))
        floor = float(np.mean(floor_per_h[h]))
        out[h] = {
            'cross_entropy_bits': ce,
            'entropy_floor_bits': floor,
            'perplexity': 2.0 ** ce,
            'entropy_floor_perplexity': 2.0 ** floor,
            'excess_perplexity': 2.0 ** (ce - floor),
        }
    return out


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
