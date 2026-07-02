"""k-gram GDC variant: same chain-of-positions, same transition dynamics
(α/θ/diffuse), but the filter step matches against the last k chars of the
prefix rather than a single character.

Motivation: in plain GDC, conditioning on a length-k context is implicit —
each character is filtered separately and the transition between filters
loses (1-α) mass to diffusion per step, so context older than ~3-5 steps
is effectively discarded. With k-gram filtering, all k context characters
are matched in one shot, and only chain positions whose preceding k-gram
exactly matches survive. The chain-advance still propagates that posterior
forward to predict the next character.

Backoff: if the test prefix's k-gram has no match in training, we fall
back to (k-1)-gram, then (k-2)-gram, ..., 1-gram, then uniform.

Usage:
  bps = score_bps_kgram_gdc(train, test, k=3, alpha=0.85, theta=0.0,
                             alphabet_size=27)
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

PROB_FLOOR = 1e-12


def _build_kgram_indices(train, max_k):
    """For each k in 1..max_k, build a dict: k-tuple → np.array of state
    indices. State index i means 'just emitted train[i]', with the k
    preceding chars (inclusive of i) being train[i-k+1..i+1]."""
    N = len(train)
    indices = {}
    for k in range(1, max_k + 1):
        d = {}
        for i in range(k - 1, N):
            key = tuple(int(c) for c in train[i - k + 1: i + 1])
            d.setdefault(key, []).append(i)
        # Convert lists to numpy arrays
        d = {kk: np.asarray(vv, dtype=np.int64) for kk, vv in d.items()}
        indices[k] = d
    return indices


def _transition_self_loop(dist, alpha, theta, N):
    """GDC self-loop transition: theta to self, alpha to next, rest diffuses
    uniformly over non-current/non-next states. Pure numpy, no dependency
    on the GDC class."""
    if N <= 1:
        return dist
    diffuse = (1 - alpha - theta) / max(N - 2, 1)
    shifted = np.zeros(N)
    shifted[1:] = dist[:-1]
    total = dist.sum()
    self_part = theta * dist
    advance_part = alpha * shifted
    diffuse_part = diffuse * (total - dist - shifted)
    return self_part + advance_part + diffuse_part


def score_bps_kgram_gdc(train, test, k, alpha, theta, alphabet_size,
                         alpha_prior=0.001, backoff=True):
    """Run the k-gram GDC and return BPS over the test sequence.

    Parameters
    ----------
    train : np.ndarray of shape (N,)
        Training character sequence.
    test : np.ndarray of shape (T,)
        Test character sequence.
    k : int
        Filter context length. k=1 reduces to plain GDC.
    alpha, theta : float
        GDC transition parameters (must satisfy alpha + theta <= 1).
    alphabet_size : int
        Number of distinct emissions.
    alpha_prior : float
        Laplace floor on predictive distribution.
    backoff : bool
        If True, fall back to k-1, k-2, ..., 1-gram filter when the
        k-gram doesn't match. Otherwise OOV → uniform.
    """
    N = len(train)
    train_int = train.astype(np.int64)
    indices = _build_kgram_indices(train_int, k)

    dist = np.full(N, 1.0 / N)
    log2_total = 0.0
    T = len(test)
    floor = alpha_prior / alphabet_size if alpha_prior > 0 else 0.0

    for t in range(T):
        # Transition step
        if t > 0:
            dist = _transition_self_loop(dist, alpha, theta, N)
        # Predictive over emissions (state's emission = train[state index])
        probs = np.bincount(train_int, weights=dist,
                             minlength=alphabet_size).astype(np.float64)
        if alpha_prior > 0:
            probs = probs + floor
        s = probs.sum()
        if s > 0:
            probs /= s
        else:
            probs = np.full(alphabet_size, 1.0 / alphabet_size)
        c_obs = int(test[t])
        log2_total += -np.log2(max(float(probs[c_obs]), PROB_FLOOR))

        # Filter step: try to match the longest available context
        # Find longest kk in {min(k, t+1), ..., 1} for which the
        # corresponding kk-gram has at least one match in training.
        max_kk = min(k, t + 1)
        matched_states = None
        for kk in range(max_kk, 0, -1):
            key = tuple(int(c) for c in test[t - kk + 1: t + 1])
            arr = indices[kk].get(key)
            if arr is not None and arr.size:
                matched_states = arr
                break
            if not backoff:
                break
        if matched_states is not None:
            new_dist = np.zeros(N)
            new_dist[matched_states] = dist[matched_states]
            tot = new_dist.sum()
            if tot > 0:
                dist = new_dist / tot
            else:
                dist = np.full(N, 1.0 / N)
        else:
            # No match at any level → uniform
            dist = np.full(N, 1.0 / N)

    return log2_total / T


# --------------------------------------------------------------------
# Sweep on blake-poems
# --------------------------------------------------------------------
def main():
    from data_loader import load
    train, test, info = load('blake-poems', verbose=False)
    n_val = max(1, int(len(train) * 0.10))
    fit_seq, val_seq = train[:-n_val], train[-n_val:]
    print(f"fit={len(fit_seq):,}  val={len(val_seq):,}  test={len(test):,}\n")

    # Sweep grid
    K_GRID = [1, 2, 3, 4, 5, 6, 8]
    ALPHA_GRID = [0.5, 0.7, 0.85, 0.9, 0.95]
    THETA_GRID = [0.0, 0.005, 0.05]

    print(f"{'k':>3} {'alpha':>6} {'theta':>6} {'val_bps':>8} {'time':>5}")
    print('-' * 35)
    results = []
    for k in K_GRID:
        for alpha in ALPHA_GRID:
            for theta in THETA_GRID:
                if alpha + theta > 1.0:
                    continue
                t0 = time.time()
                bps = score_bps_kgram_gdc(
                    fit_seq, val_seq, k=k, alpha=alpha, theta=theta,
                    alphabet_size=27, alpha_prior=0.001)
                dt = time.time() - t0
                results.append((bps, k, alpha, theta, dt))
                print(f"{k:>3} {alpha:>6.2f} {theta:>6.3f} {bps:>8.3f} {dt:>5.1f}",
                      flush=True)

    results.sort()
    print(f"\nTop 10 by val BPS:")
    for bps, k, a, t, dt in results[:10]:
        print(f"  k={k} a={a} t={t}: val={bps:.3f}")

    # Eval test on top-3
    print(f"\nTest BPS for top-3 by val (retrained on full train):")
    seen = set()
    top_unique = []
    for r in results:
        key = (r[1], r[2], r[3])
        if key not in seen:
            seen.add(key); top_unique.append(r)
        if len(top_unique) == 3:
            break
    for bps_val, k, alpha, theta, _ in top_unique:
        t0 = time.time()
        bps_test = score_bps_kgram_gdc(
            train, test, k=k, alpha=alpha, theta=theta,
            alphabet_size=27, alpha_prior=0.001)
        print(f"  k={k} a={alpha} t={theta}: val={bps_val:.3f} test={bps_test:.3f} "
              f"[{time.time()-t0:.1f}s]")

    # Also baseline: k=1 (plain GDC) at the same alpha
    print(f"\nReference (k=1, plain GDC) at best alpha:")
    best_k1 = min((r for r in results if r[1] == 1), default=None)
    if best_k1:
        bps_val, k, alpha, theta, _ = best_k1
        bps_test = score_bps_kgram_gdc(
            train, test, k=1, alpha=alpha, theta=theta,
            alphabet_size=27, alpha_prior=0.001)
        print(f"  k=1 a={alpha} t={theta}: val={bps_val:.3f} test={bps_test:.3f}")


if __name__ == '__main__':
    main()
