"""Diagnostic: where does GDC go wrong on sparse_large?

Reproduces the canonical sparse_large failure cell (TL=200, N=400,
test seed 0) with the val-picked GDC config (alpha=0.8, theta=0.001,
beta=0.1) and probes:

1. The GDC's posterior over training-corpus positions on each test
   prefix — is it concentrated or diffuse?
2. The implicit posterior over the TRUE underlying HMM hidden state
   (computed by mapping each training position back to the hidden
   state that generated it, then summing the GDC's position posterior
   per hidden state). This is the apples-to-apples comparison vs the
   true HMM filter posterior.
3. The KL-divergence between the GDC's implicit hidden-state
   posterior and the true HMM filter posterior, at the time of the
   next-symbol prediction.
4. The KL of the GDC's predicted next-symbol distribution vs the
   true posterior over the next symbol.
5. CHMM as a reference: what does the EM-fit CHMM put its hidden-
   state posterior on for the same prefixes?

Output: a structured CSV with one row per test prefix, plus a
short markdown summary printed to stdout.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from random_hmm import random_sparse_topology_hmm
from model_wrappers import GDCForecaster
from chmm_alergia_wrappers import CHMMForecaster

# ---- canonical failure cell ---------------------------------------
REGIME = 'sparse_large'
nS, nA = 30, 8
fanout = 2
E_concentration = 0.1
TRAIN_LEN = 200
N_TRAIN = 400
N_TEST_PREFIXES = 50      # diagnostic subset (full benchmark uses 100)
TEST_PREFIX_LEN = 20
SEED = 0                   # one of the test seeds

# val-picked GDC config for this cell
GDC_ALPHA, GDC_THETA, GDC_BETA = 0.8, 0.001, 0.1
GDC_TRANSITION = 'self_loop'

# CHMM reference (val-picked from CHMM_KS = {4, 16, 32}; for this cell
# CHMM at K=32 is the typical pick — use K=32 as reference)
CHMM_K = 32


def kl(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1.0/len(p))
    q = q / q.sum() if q.sum() > 0 else np.full_like(q, 1.0/len(q))
    return float(np.sum(p * (np.log2(p + eps) - np.log2(q + eps))))


def sample_with_states(hmm, length, rng):
    """Like hmm.sample but returns (states, obs)."""
    return hmm.sample(length, rng)


def hmm_filter_posterior(hmm, prefix_obs):
    """Exact posterior P(state_t | obs_{1..t}) for the TRUE HMM."""
    pi, T, E = hmm.pi, hmm.T, hmm.E
    alpha = pi * E[:, prefix_obs[0]]
    alpha = alpha / alpha.sum()
    for o in prefix_obs[1:]:
        alpha = alpha @ T
        alpha = alpha * E[:, o]
        alpha = alpha / alpha.sum()
    return alpha


def hmm_next_obs_distribution(hmm, state_post, h=1):
    """P(obs_{t+h} | state_post) for the TRUE HMM."""
    Th = np.linalg.matrix_power(hmm.T, h)
    return state_post @ Th @ hmm.E


def aggregate_position_to_state(position_post, true_states_per_position):
    """Sum GDC's posterior over chain positions into per-hidden-state mass."""
    out = np.zeros(int(true_states_per_position.max()) + 1)
    np.add.at(out, true_states_per_position, position_post)
    return out


def main():
    print(f"=== Diagnose sparse_large failure cell (TL={TRAIN_LEN}, "
          f"N={N_TRAIN}, seed={SEED}) ===\n")

    # Reproduce the cell-RNG seeding from seq_len_sweep.py exactly
    seed_offset = 2  # 'sparse' regime -> +2
    rng = np.random.default_rng(60000 + SEED * 137 + nS * 7 + nA * 11
                                + seed_offset)

    hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                     E_concentration=E_concentration)

    # Sample full_train (max N) — same RNG sequence as the sweep
    full_train_with_states = []
    for _ in range(N_TRAIN):
        s, o = hmm.sample(TRAIN_LEN, rng)
        full_train_with_states.append((s, o))
    train_obs = [o for _, o in full_train_with_states]
    # True hidden-state assignment of every training position
    true_states_per_position = np.concatenate(
        [s for s, _ in full_train_with_states])

    # Test prefixes
    test_pf_with_states = []
    for _ in range(N_TEST_PREFIXES):
        s, o = hmm.sample(TEST_PREFIX_LEN, rng)
        test_pf_with_states.append((s, o))

    print(f"  HMM: nS={nS}, nA={nA}, sparse fanout={fanout}, "
          f"E_concentration={E_concentration}")
    print(f"  Training: N={N_TRAIN} sequences × T={TRAIN_LEN} = "
          f"{N_TRAIN*TRAIN_LEN} tokens (= GDC chain length)")
    print(f"  Test: {N_TEST_PREFIXES} prefixes of length {TEST_PREFIX_LEN}\n",
          flush=True)

    # ----- Train GDC ------------------------------------------------
    t0 = time.time()
    gdc = GDCForecaster(nA, alpha=GDC_ALPHA, theta=GDC_THETA, gamma=0.0,
                        beta=GDC_BETA, transition_type=GDC_TRANSITION,
                        initial_dist='sequence_starts')
    gdc.fit(train_obs)
    print(f"  Trained GDC in {time.time()-t0:.1f}s. "
          f"Chain length = {gdc.gdc.n_states}", flush=True)

    # ----- Train CHMM (reference) ----------------------------------
    t0 = time.time()
    chmm = CHMMForecaster(nA, K=CHMM_K, n_em_iters=50, seed=SEED)
    chmm.fit(train_obs)
    print(f"  Trained CHMM K={CHMM_K} in {time.time()-t0:.1f}s. "
          f"n_total = {chmm.n_total}\n", flush=True)

    # ----- Per-prefix diagnostic ------------------------------------
    rows = []
    for i, (true_states, obs) in enumerate(test_pf_with_states):
        true_state_t = int(true_states[-1])
        true_post = hmm_filter_posterior(hmm, obs)
        true_next = hmm_next_obs_distribution(hmm, true_post, h=1)

        # GDC: forward pass to get position posterior; transition once
        # to get position posterior for next step
        position_post_now = gdc.gdc.forward_pass(obs.reshape(-1, 1))
        position_post_next = gdc.gdc.forecast(position_post_now, n_steps=1)
        # Aggregate to hidden-state posterior using known true states
        gdc_state_post = aggregate_position_to_state(
            position_post_now, true_states_per_position)
        gdc_state_post = (gdc_state_post / gdc_state_post.sum()
                          if gdc_state_post.sum() > 0 else gdc_state_post)
        # GDC's next-symbol distribution (= what the metric scores)
        gdc_next = gdc.horizon_emission(obs, h=1)

        # CHMM: same idea
        chmm_next = chmm.horizon_emission(obs, h=1)

        # Diagnostic stats
        n_pos = len(position_post_now)
        position_post_sorted = np.sort(position_post_now)[::-1]
        ess = 1.0 / np.sum(position_post_now**2)  # effective sample size
        top1 = position_post_sorted[0]
        top10 = position_post_sorted[:10].sum()
        top100 = position_post_sorted[:100].sum()
        # KL divergence: implicit GDC state posterior vs true filter
        kl_state_gdc = kl(true_post, gdc_state_post)
        # KL divergence: predicted next-symbol vs true next-symbol
        kl_next_gdc = kl(true_next, gdc_next)
        kl_next_chmm = kl(true_next, chmm_next)
        # Argmax matches
        gdc_state_argmax_correct = int(np.argmax(gdc_state_post)
                                       == np.argmax(true_post))
        gdc_next_argmax_correct = int(np.argmax(gdc_next)
                                      == np.argmax(true_next))
        chmm_next_argmax_correct = int(np.argmax(chmm_next)
                                       == np.argmax(true_next))

        rows.append(dict(
            prefix_idx=i,
            true_state_t=true_state_t,
            true_state_argmax=int(np.argmax(true_post)),
            ess=ess, top1=top1, top10=top10, top100=top100,
            kl_state_gdc=kl_state_gdc,
            kl_next_gdc=kl_next_gdc,
            kl_next_chmm=kl_next_chmm,
            gdc_state_argmax_correct=gdc_state_argmax_correct,
            gdc_next_argmax_correct=gdc_next_argmax_correct,
            chmm_next_argmax_correct=chmm_next_argmax_correct,
            true_post_entropy=float(-np.sum(true_post
                                            * np.log2(true_post + 1e-12))),
            gdc_state_post_entropy=float(-np.sum(
                gdc_state_post * np.log2(gdc_state_post + 1e-12))),
        ))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(HERE, 'diagnose_sparse_large.csv')
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}\n")

    # ----- Summary --------------------------------------------------
    print("## Posterior concentration (over GDC's 80,000 chain positions)\n")
    print(f"  ESS (effective sample size):  median={df.ess.median():.1f}  "
          f"mean={df.ess.mean():.1f}  min={df.ess.min():.1f}  "
          f"max={df.ess.max():.1f}")
    print(f"  top-1 position mass:          median={df.top1.median():.3f}  "
          f"mean={df.top1.mean():.3f}")
    print(f"  top-10 mass:                  median={df.top10.median():.3f}  "
          f"mean={df.top10.mean():.3f}")
    print(f"  top-100 mass:                 median={df.top100.median():.3f}  "
          f"mean={df.top100.mean():.3f}\n")

    print("## Hidden-state posterior alignment (GDC vs true HMM filter)\n")
    print(f"  KL(true || GDC implicit-state-post)  "
          f"median={df.kl_state_gdc.median():.3f}  "
          f"mean={df.kl_state_gdc.mean():.3f}")
    print(f"  GDC argmax-state matches true argmax: "
          f"{df.gdc_state_argmax_correct.mean():.1%} ({df.gdc_state_argmax_correct.sum()}/{len(df)})")
    print(f"  True filter entropy (bits):           "
          f"median={df.true_post_entropy.median():.2f}")
    print(f"  GDC implicit-state entropy (bits):    "
          f"median={df.gdc_state_post_entropy.median():.2f}\n")

    print("## Next-symbol prediction (test metric — h=1)\n")
    print(f"  KL(true_next || GDC_next):    "
          f"median={df.kl_next_gdc.median():.4f}  "
          f"mean={df.kl_next_gdc.mean():.4f}")
    print(f"  KL(true_next || CHMM_next):   "
          f"median={df.kl_next_chmm.median():.4f}  "
          f"mean={df.kl_next_chmm.mean():.4f}")
    print(f"  GDC next-symbol argmax correct:  "
          f"{df.gdc_next_argmax_correct.mean():.1%} "
          f"({df.gdc_next_argmax_correct.sum()}/{len(df)})")
    print(f"  CHMM next-symbol argmax correct: "
          f"{df.chmm_next_argmax_correct.mean():.1%} "
          f"({df.chmm_next_argmax_correct.sum()}/{len(df)})")


if __name__ == "__main__":
    main()
