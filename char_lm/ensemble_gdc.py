"""Ensemble of GDC chains at different α (and optionally different k).

Each chain processes the test sequence independently and emits a per-step
predictive distribution P_k(c | history). We combine them as
  P_mix(c | history) = sum_k w_k * P_k(c | history)
with weights w on the simplex, fit by EM on val data, then evaluated on test.

This gives a mixture-of-Markov-chains story: low-α chains capture diffuse
("global frequency") behavior, high-α chains capture sticky-context
behavior. The mixture trades off between them adaptively at each step.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from kgram_gdc import (_build_kgram_indices, _transition_self_loop,  # noqa: E402
                        score_bps_kgram_gdc, PROB_FLOOR)


def collect_predictives(train, test, k, alpha, theta, alphabet_size,
                         alpha_prior=0.001):
    """Run a k-gram GDC and return (T, V) per-step predictive
    distributions plus the BPS for the chain alone."""
    N = len(train)
    train_int = train.astype(np.int64)
    indices = _build_kgram_indices(train_int, k)
    V = alphabet_size
    floor = alpha_prior / V if alpha_prior > 0 else 0.0

    dist = np.full(N, 1.0 / N)
    T = len(test)
    predictives = np.zeros((T, V), dtype=np.float64)
    log2_total = 0.0

    for t in range(T):
        if t > 0:
            dist = _transition_self_loop(dist, alpha, theta, N)
        probs = np.bincount(train_int, weights=dist,
                             minlength=V).astype(np.float64)
        if alpha_prior > 0:
            probs = probs + floor
        s = probs.sum()
        if s > 0:
            probs /= s
        else:
            probs = np.full(V, 1.0 / V)
        predictives[t] = probs
        c_obs = int(test[t])
        log2_total += -np.log2(max(float(probs[c_obs]), PROB_FLOOR))
        # Filter step: try k-gram, backoff
        max_kk = min(k, t + 1)
        matched_states = None
        for kk in range(max_kk, 0, -1):
            key = tuple(int(c) for c in test[t - kk + 1: t + 1])
            arr = indices[kk].get(key)
            if arr is not None and arr.size:
                matched_states = arr
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
            dist = np.full(N, 1.0 / N)
    return predictives, log2_total / T


def fit_mixture_weights_em(predictives_list, c_obs, n_iter=500, tol=1e-9):
    """EM for mixture weights on the simplex.

    predictives_list : list of K (T, V) arrays
    c_obs : (T,) int array of observed indices
    Returns weights (K,) summing to 1 that minimize -mean(log2(sum_k w_k p_k)).
    """
    K = len(predictives_list)
    T = len(c_obs)
    # Per-step per-chain likelihood for the actual obs: L[k, t]
    L = np.stack([p[np.arange(T), c_obs] for p in predictives_list])  # (K, T)
    L = np.maximum(L, PROB_FLOOR)
    w = np.full(K, 1.0 / K)
    for it in range(n_iter):
        # E: responsibilities r[k, t] = w_k L[k, t] / sum_k' w_k' L[k', t]
        num = w[:, None] * L
        denom = num.sum(axis=0, keepdims=True)
        r = num / np.maximum(denom, PROB_FLOOR)
        # M: w_k = mean_t r[k, t]
        new_w = r.mean(axis=1)
        if np.max(np.abs(new_w - w)) < tol:
            w = new_w
            break
        w = new_w
    return w


def mixture_bps(predictives_list, c_obs, weights):
    T = len(c_obs)
    L = np.stack([p[np.arange(T), c_obs] for p in predictives_list])  # (K, T)
    L = np.maximum(L, PROB_FLOOR)
    p_mix = (weights[:, None] * L).sum(axis=0)
    return float(-np.log2(np.maximum(p_mix, PROB_FLOOR)).mean())


def main():
    from data_loader import load
    train, test, info = load('blake-poems', verbose=False)
    n_val = max(1, int(len(train) * 0.10))
    fit_seq, val_seq = train[:-n_val], train[-n_val:]
    print(f"fit={len(fit_seq):,}  val={len(val_seq):,}  test={len(test):,}\n")

    # Components: a spread of (k, α, θ)
    components = [
        ('k=1 a=0.50', 1, 0.50, 0.0),
        ('k=1 a=0.70', 1, 0.70, 0.0),
        ('k=1 a=0.85', 1, 0.85, 0.0),
        ('k=1 a=0.95', 1, 0.95, 0.0),
        ('k=2 a=0.70', 2, 0.70, 0.0),
        ('k=2 a=0.85', 2, 0.85, 0.0),
        ('k=2 a=0.95', 2, 0.95, 0.0),
    ]

    print("Fitting individual components on fit-set; scoring val and test:")
    val_preds = []
    test_preds = []
    val_bps_solo = []
    test_bps_solo = []
    for name, k, a, t in components:
        t0 = time.time()
        v_p, v_bps = collect_predictives(fit_seq, val_seq, k=k,
                                           alpha=a, theta=t, alphabet_size=27)
        # Test scoring uses full train (paper protocol)
        te_p, te_bps = collect_predictives(train, test, k=k,
                                             alpha=a, theta=t, alphabet_size=27)
        val_preds.append(v_p); test_preds.append(te_p)
        val_bps_solo.append(v_bps); test_bps_solo.append(te_bps)
        print(f"  {name:14s}: val_bps={v_bps:.3f} test_bps={te_bps:.3f} "
              f"[{time.time()-t0:.1f}s]", flush=True)

    # EM on val
    print("\nFitting mixture weights on val (EM)...")
    weights = fit_mixture_weights_em(val_preds, val_seq.astype(np.int64))
    print(f"  weights: {dict(zip([c[0] for c in components], np.round(weights, 4)))}")

    # Mixture val + test BPS
    val_mix = mixture_bps(val_preds, val_seq.astype(np.int64), weights)
    test_mix = mixture_bps(test_preds, test.astype(np.int64), weights)
    print(f"\nMixture (EM-weighted) val BPS: {val_mix:.3f}")
    print(f"Mixture (EM-weighted) test BPS: {test_mix:.3f}")

    # Reference: uniform mixture
    uniform_w = np.full(len(components), 1.0 / len(components))
    val_unif = mixture_bps(val_preds, val_seq.astype(np.int64), uniform_w)
    test_unif = mixture_bps(test_preds, test.astype(np.int64), uniform_w)
    print(f"\nMixture (uniform) val BPS: {val_unif:.3f}")
    print(f"Mixture (uniform) test BPS: {test_unif:.3f}")

    # Reference: best single component (val-picked)
    best_idx = int(np.argmin(val_bps_solo))
    print(f"\nBest single component (val-picked): {components[best_idx][0]}")
    print(f"  val={val_bps_solo[best_idx]:.3f} test={test_bps_solo[best_idx]:.3f}")

    print("\n" + "=" * 60)
    print("Summary on blake-poems test BPS:")
    print(f"  Best single component: {test_bps_solo[best_idx]:.3f} "
          f"({components[best_idx][0]})")
    print(f"  Mixture (uniform):     {test_unif:.3f}")
    print(f"  Mixture (EM-weighted): {test_mix:.3f}")
    print(f"  Paper CHMM:            1.60")
    print(f"  Paper SeqM:            1.71")


if __name__ == '__main__':
    main()
