"""PAutomaC official scoring metric and reporting helpers."""

from __future__ import annotations
import numpy as np

EPS = 1e-300


def normalise(probs):
    s = float(np.sum(probs))
    return probs / s if s > 0 else np.full_like(probs, 1.0 / max(len(probs), 1))


def pautomac_score(model_log_probs, true_probs):
    """Official PAutomaC score.

    Args:
        model_log_probs : 1-D array of log probabilities (natural log) of
            each test sequence under the model.  Need NOT be normalised
            -- we normalise within this function via softmax in log space.
        true_probs : 1-D array of true probabilities of each test
            sequence under the target machine.

    Returns:
        score, entropy_floor, gap, lift, where:
            score          = 2^(- sum_t pT_norm(t) * log2 pM_norm(t))
            entropy_floor  = 2^H(pT_norm)  (best achievable)
            gap            = score - entropy_floor (perplexity units)
            lift           = (uniform_score - score)
                              / (uniform_score - entropy_floor)
            uniform_score  = N (test set size)
    """
    n = len(true_probs)
    pT = normalise(np.asarray(true_probs, dtype=np.float64))

    # Log-softmax of model log-probs to get pM_norm (numerically stable).
    log_M = np.asarray(model_log_probs, dtype=np.float64)
    m = np.max(log_M)
    log_norm = m + np.log(np.sum(np.exp(log_M - m)))
    log_pM_norm = log_M - log_norm           # natural log
    log2_pM_norm = log_pM_norm / np.log(2)   # convert to log2

    cross_entropy = -float(np.sum(pT * log2_pM_norm))
    score = 2.0 ** cross_entropy

    pT_safe = np.clip(pT, EPS, None)
    H = -float(np.sum(pT * np.log2(pT_safe)))
    entropy_floor = 2.0 ** H

    uniform_score = float(n)
    gap = score - entropy_floor
    denom = uniform_score - entropy_floor
    lift = ((uniform_score - score) / denom) if denom > 1e-9 else float('nan')
    return {
        'score': score, 'entropy_floor': entropy_floor,
        'gap': gap, 'lift': lift,
        'cross_entropy_bits': cross_entropy,
        'entropy_bits': H,
        'uniform_score': uniform_score,
    }


def report(model_name, model_log_probs, true_probs):
    r = pautomac_score(model_log_probs, true_probs)
    return {
        'model': model_name,
        'score': r['score'],
        'entropy_floor': r['entropy_floor'],
        'gap': r['gap'],
        'lift': r['lift'],
    }
