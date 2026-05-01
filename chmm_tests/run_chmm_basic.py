"""Basic CHMM sanity test on synthetic random-HMM sequences.

Generates training + held-out sequences from a known RandomHMM,
trains a CHMM with several clone counts K, and reports:

    bps_chmm   — negative log2-likelihood per symbol of CHMM on eval
    bps_bayes  — same, computed with the true HMM (Bayes ceiling)
    bps_uniform— uniform-emission baseline (= log2 nA)

Sequences are concatenated for CHMM training (the upstream library
is built around a single long sequence); a single dummy action
channel is used since we have no action conditioning.

Run:
    python chmm_tests/run_chmm_basic.py
"""

from __future__ import annotations

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "hmm_comparison"))
sys.path.insert(0, os.path.join(HERE, "naturecomm_cscg"))

from random_hmm import random_dense_hmm  # noqa: E402
from chmm_actions import CHMM  # noqa: E402


def true_hmm_bps(hmm, eval_obs):
    """Average negative log2-likelihood per symbol under the true HMM,
    averaged across the held-out sequences.
    """
    total_log2, total_n = 0.0, 0
    for obs in eval_obs:
        # forward filter, accumulating log p(o_t | o_{<t})
        a = hmm.pi * hmm.E[:, obs[0]]
        s = a.sum()
        total_log2 += np.log2(max(s, 1e-300))
        a = a / s if s > 0 else np.full(hmm.nS, 1.0 / hmm.nS)
        for o in obs[1:]:
            a = (a @ hmm.T) * hmm.E[:, o]
            s = a.sum()
            total_log2 += np.log2(max(s, 1e-300))
            a = a / s if s > 0 else np.full(hmm.nS, 1.0 / hmm.nS)
        total_n += len(obs)
    return -total_log2 / total_n


def chmm_eval_bps(model, eval_obs):
    """Average bps of trained CHMM on each held-out sequence (boundary
    states reset between sequences via the model's Pi_x prior).
    """
    total_bps, total_n = 0.0, 0
    for obs in eval_obs:
        x = obs.astype(np.int64)
        a = np.zeros_like(x)
        # bps returns per-timestep negative log2-likelihood (length len(x))
        bps_arr = np.asarray(model.bps(x, a))
        total_bps += float(bps_arr.sum())
        total_n += len(x)
    return total_bps / total_n


def main():
    rng = np.random.default_rng(7)
    nS, nA = 6, 4
    hmm = random_dense_hmm(nS, nA, rng)

    n_train_seq, n_eval_seq, T_len = 200, 60, 50
    train_obs = hmm.sample_many(n_train_seq, T_len, rng)
    eval_obs = hmm.sample_many(n_eval_seq, T_len, rng)

    # Concatenate for CHMM (single long sequence + dummy action channel)
    x_train = np.concatenate(train_obs).astype(np.int64)
    a_train = np.zeros_like(x_train)
    print(f"True HMM nS={nS} nA={nA}, train tokens={len(x_train)}, "
          f"eval tokens={n_eval_seq * T_len}")

    bps_uniform = float(np.log2(nA))
    bps_bayes = true_hmm_bps(hmm, eval_obs)
    print(f"\nbaselines:  uniform={bps_uniform:.4f}  bayes(true HMM)={bps_bayes:.4f}\n")

    K_grid = [1, 2, 4, 8, 16]
    results = []
    for K in K_grid:
        n_clones = np.full(nA, K, dtype=np.int64)
        print(f"--- CHMM K={K} (n_states = K*nA = {K*nA}) ---")
        model = CHMM(n_clones=n_clones, x=x_train, a=a_train,
                     pseudocount=1e-3, seed=0)
        convergence = model.learn_em_T(x_train, a_train,
                                       n_iter=80, term_early=True)
        bps_train = float(np.asarray(convergence[-1]).mean())
        bps_eval = chmm_eval_bps(model, eval_obs)
        gap_to_bayes = bps_eval - bps_bayes
        results.append((K, bps_train, bps_eval, gap_to_bayes))
        print(f"K={K:2d}   bps_train={bps_train:.4f}  "
              f"bps_eval={bps_eval:.4f}  eval_gap_vs_bayes={gap_to_bayes:+.4f}")

    print("\n=== summary ===")
    print(f"{'K':>3} {'bps_train':>10} {'bps_eval':>10} {'gap_bayes':>10}")
    for K, btr, bev, gap in results:
        print(f"{K:>3d} {btr:>10.4f} {bev:>10.4f} {gap:>+10.4f}")
    print(f"\nbayes ceiling: {bps_bayes:.4f}    uniform: {bps_uniform:.4f}")


if __name__ == "__main__":
    main()
