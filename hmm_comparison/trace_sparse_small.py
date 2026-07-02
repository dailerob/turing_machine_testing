"""Step-by-step trace on a single sparse_small HMM (TL=100, N=400).

Mirrors trace_high_alpha.py but for sparse_small (smaller HMM,
smaller alphabet, smaller chain). The sparse_small failure cell
has GDC = 1.088 vs CHMM = 1.016, gap 0.072 — smaller than
sparse_large's gap. This trace asks whether the same mechanism
(under-concentrated posterior at low α; over-concentrated /
sample-starved at high α) appears in this milder regime.

Procedure:
  1. Build sparse_small HMM at seed 0; sample 400 train seqs of
     length 100, 100 test prefixes of length 20.
  2. Train low-α (0.8) and high-α (0.99) GDCs (absorb + uniform).
  3. For each test prefix, score both at h=1 and find prefixes
     where low-α is right and high-α is wrong on next-symbol argmax.
  4. Trace the first 2-3 such prefixes step-by-step.
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from random_hmm import random_sparse_topology_hmm
from model_wrappers import GDCForecaster

# --- sparse_small parameters ---
nS, nA = 10, 4
fanout = 2
E_concentration = 0.1
TRAIN_LEN = 100
N_TRAIN = 400
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
SEED = 0

LOW_ALPHA = 0.8
HIGH_ALPHA = 0.99
THETA = 0.001
BETA = 0.1


def hmm_filter_history(hmm, prefix):
    pi, T, E = hmm.pi, hmm.T, hmm.E
    a = pi * E[:, prefix[0]]; a /= a.sum()
    out = [a.copy()]
    for o in prefix[1:]:
        a = a @ T; a = a * E[:, o]; a /= a.sum()
        out.append(a.copy())
    return out


def hmm_next_obs(hmm, post, h=1):
    Th = np.linalg.matrix_power(hmm.T, h)
    return post @ Th @ hmm.E


def aggregate_state_post(position_post, true_states_per_position, nS):
    sp = np.zeros(nS)
    np.add.at(sp, true_states_per_position, position_post)
    s = sp.sum()
    return sp / s if s > 0 else np.full(nS, 1.0/nS)


def predict_next_obs_gdc(gdc, position_post, h=1):
    fc = gdc.gdc.forecast(position_post, n_steps=h)
    out = np.zeros(gdc.nA)
    np.add.at(out, gdc._symbol_of_state, fc)
    return out / out.sum() if out.sum() > 0 else np.full(gdc.nA, 1.0/gdc.nA)


def setup():
    seed_offset = 2  # sparse
    rng = np.random.default_rng(60000 + SEED * 137 + nS * 7 + nA * 11
                                + seed_offset)
    hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                     E_concentration=E_concentration)
    full = []
    for _ in range(N_TRAIN):
        full.append(hmm.sample(TRAIN_LEN, rng))
    train = [o for _, o in full]
    true_states = np.concatenate([s for s, _ in full])
    test_pf = []
    for _ in range(N_TEST_PREFIXES):
        s, o = hmm.sample(TEST_PREFIX_LEN, rng)
        test_pf.append((s, o))
    return hmm, train, true_states, test_pf


def find_failure_prefixes(hmm, gdc_low, gdc_high, test_pf):
    candidates = []
    for i, (true_s, obs) in enumerate(test_pf):
        true_post = hmm_filter_history(hmm, obs)[-1]
        true_next = hmm_next_obs(hmm, true_post, h=1)
        true_argmax = int(np.argmax(true_next))
        low_next = gdc_low.horizon_emission(obs, h=1)
        high_next = gdc_high.horizon_emission(obs, h=1)
        low_correct = (int(np.argmax(low_next)) == true_argmax)
        high_correct = (int(np.argmax(high_next)) == true_argmax)
        if low_correct and not high_correct:
            candidates.append(i)
    return candidates


def trace_one(idx, hmm, gdc_low, gdc_high, true_states, prefix_obs,
              prefix_true_states):
    obs = prefix_obs
    print(f"\n=== Trace prefix {idx} ===")
    print(f"  Observed: {obs.tolist()}")
    print(f"  True hidden-state path: {prefix_true_states.tolist()}")

    true_history = hmm_filter_history(hmm, obs)
    low_history = gdc_low.gdc.forward_pass(obs.reshape(-1, 1),
                                            return_history=True)[1]
    high_history = gdc_high.gdc.forward_pass(obs.reshape(-1, 1),
                                              return_history=True)[1]

    print(f"\n  Step | obs | true argS | true H | "
          f"low: argS m(true) argY ess | "
          f"high: argS m(true) argY ess")
    print("  -----+-----+-----------+--------+"
          "----------------------------+----------------------------")
    for t in range(len(obs)):
        true_post = true_history[t]
        low_state = aggregate_state_post(low_history[t], true_states, hmm.nS)
        high_state = aggregate_state_post(high_history[t], true_states, hmm.nS)
        low_next = predict_next_obs_gdc(gdc_low, low_history[t], h=1)
        high_next = predict_next_obs_gdc(gdc_high, high_history[t], h=1)
        low_ess = 1.0 / np.sum(low_history[t]**2)
        high_ess = 1.0 / np.sum(high_history[t]**2)
        true_arg = int(np.argmax(true_post))
        true_H = float(-np.sum(true_post * np.log2(true_post + 1e-12)))
        print(f"  {t:>4d} | {int(obs[t]):>3d} | "
              f"{true_arg:>9d} | {true_H:>5.2f}  | "
              f"argS={int(np.argmax(low_state)):>2d} "
              f"m={low_state[true_arg]:>4.2f} "
              f"argY={int(np.argmax(low_next)):>1d} "
              f"ess={low_ess:>6.0f} | "
              f"argS={int(np.argmax(high_state)):>2d} "
              f"m={high_state[true_arg]:>4.2f} "
              f"argY={int(np.argmax(high_next)):>1d} "
              f"ess={high_ess:>6.0f}")

    # Final summary
    true_post = true_history[-1]
    true_next = hmm_next_obs(hmm, true_post, h=1)
    low_next = predict_next_obs_gdc(gdc_low, low_history[-1], h=1)
    high_next = predict_next_obs_gdc(gdc_high, high_history[-1], h=1)
    low_state = aggregate_state_post(low_history[-1], true_states, hmm.nS)
    high_state = aggregate_state_post(high_history[-1], true_states, hmm.nS)
    low_H = float(-np.sum(low_state * np.log2(low_state + 1e-12)))
    high_H = float(-np.sum(high_state * np.log2(high_state + 1e-12)))
    true_H = float(-np.sum(true_post * np.log2(true_post + 1e-12)))

    # State 'true_argmax' emission row
    true_arg_state = int(np.argmax(true_post))
    em_row = hmm.E[true_arg_state]

    print(f"\n  --- Final state posteriors ---")
    print(f"  True HMM argmax-state {true_arg_state} (mass={true_post.max():.3f}, "
          f"H={true_H:.2f} bits)  emission row={em_row.round(3).tolist()}")
    print(f"  Low-α  argmax-state {int(np.argmax(low_state))}  "
          f"top-3 mass={[f'{v:.3f}' for v in sorted(low_state, reverse=True)[:3]]}  "
          f"H={low_H:.2f}")
    print(f"  High-α argmax-state {int(np.argmax(high_state))}  "
          f"top-3 mass={[f'{v:.3f}' for v in sorted(high_state, reverse=True)[:3]]}  "
          f"H={high_H:.2f}")

    # Position-space concentration
    low_top1 = float(np.sort(low_history[-1])[::-1][0])
    high_top1 = float(np.sort(high_history[-1])[::-1][0])
    low_ess = float(1.0 / np.sum(low_history[-1]**2))
    high_ess = float(1.0 / np.sum(high_history[-1]**2))
    print(f"\n  --- Position-space concentration ---")
    print(f"  Low-α  ESS={low_ess:.0f} of {len(low_history[-1])} positions  "
          f"top-1 mass={low_top1:.3f}")
    print(f"  High-α ESS={high_ess:.0f} of {len(high_history[-1])} positions  "
          f"top-1 mass={high_top1:.3f}")

    print(f"\n  --- Next-symbol predictions ---")
    print(f"  True next:    {true_next.round(3).tolist()}  "
          f"argmax={int(np.argmax(true_next))}")
    print(f"  Low-α next:   {low_next.round(3).tolist()}  "
          f"argmax={int(np.argmax(low_next))}  "
          f"{'CORRECT' if int(np.argmax(low_next))==int(np.argmax(true_next)) else 'WRONG'}")
    print(f"  High-α next:  {high_next.round(3).tolist()}  "
          f"argmax={int(np.argmax(high_next))}  "
          f"{'CORRECT' if int(np.argmax(high_next))==int(np.argmax(true_next)) else 'WRONG'}")


def main():
    print(f"=== Loading sparse_small HMM (seed {SEED}) ===")
    hmm, train, true_states, test_pf = setup()
    print(f"  HMM: nS={nS}, nA={nA}, fanout={fanout}, "
          f"E_concentration={E_concentration}")
    print(f"  Training: {N_TRAIN} seqs × {TRAIN_LEN} = "
          f"{N_TRAIN*TRAIN_LEN} tokens (chain length)")
    print(f"  Test: {N_TEST_PREFIXES} prefixes × {TEST_PREFIX_LEN} tokens\n")

    print(f"=== Training GDCs (absorb + uniform) ===")
    gdc_low = GDCForecaster(nA, alpha=LOW_ALPHA, theta=THETA, gamma=0.0,
                             beta=BETA, transition_type='self_loop',
                             initial_dist='uniform',
                             terminal_behavior='absorb').fit(train)
    gdc_high = GDCForecaster(nA, alpha=HIGH_ALPHA, theta=THETA, gamma=0.0,
                              beta=BETA, transition_type='self_loop',
                              initial_dist='uniform',
                              terminal_behavior='absorb').fit(train)
    print(f"  Low α  = {LOW_ALPHA},  High α = {HIGH_ALPHA},  "
          f"diffuse = {1-LOW_ALPHA-THETA:.3f} vs {1-HIGH_ALPHA-THETA:.3f}\n")

    failures = find_failure_prefixes(hmm, gdc_low, gdc_high, test_pf)
    print(f"=== Failure prefixes (low correct, high wrong) ===")
    print(f"  {len(failures)} of {N_TEST_PREFIXES}\n")
    if not failures:
        print("  No failure prefixes; HMM is too easy. Trying both-wrong "
              "or both-right cases for context.")
        return

    for k, idx in enumerate(failures[:3]):
        true_states_seq, obs = test_pf[idx]
        trace_one(idx, hmm, gdc_low, gdc_high, true_states, obs,
                  true_states_seq)


if __name__ == "__main__":
    main()
