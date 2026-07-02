"""Step-by-step trace of GDC forward pass at α=0.8 vs α=0.99 on a
single sparse_large HMM.

Goal: GDC's state-argmax is correct 91% of the time at α=0.8 but
the next-symbol argmax is only 78%, suggesting the posterior is
under-concentrated. Pushing α to 0.99 should sharpen — but it
catastrophically degrades the metric (1.66 vs 1.29 mean excess pp).
This trace shows mechanically what's going wrong.

Procedure:
  1. Build the sparse_large HMM at seed 0; sample 400 training seqs
     of length 200 and 100 test prefixes of length 20.
  2. Train two GDCs: low-α (0.8) and high-α (0.99).
  3. For each test prefix, run both forward passes and record
     per-step position posteriors (with `return_history=True`).
  4. Find a prefix where:
       (a) low-α gets the next-symbol right
       (b) high-α gets it wrong
     This is the failure case to dissect.
  5. For that prefix, print step-by-step:
       - true HMM filter posterior (entropy + argmax-state)
       - low-α implicit state posterior + next-symbol distribution
       - high-α implicit state posterior + next-symbol distribution
     Track where they diverge and what the high-α posterior is doing.
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from random_hmm import random_sparse_topology_hmm
from model_wrappers import GDCForecaster

nS, nA = 30, 8
fanout = 2
E_concentration = 0.1
TRAIN_LEN = 200
N_TRAIN = 400
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
SEED = 0

LOW_ALPHA = 0.8
HIGH_ALPHA = 0.99
THETA = 0.001
BETA = 0.1


def hmm_filter_history(hmm, prefix):
    """Return list of true HMM filter posteriors at each step."""
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
    if s > 0: sp = sp / s
    return sp


def predict_next_obs_gdc(gdc, position_post, h=1):
    """One transition step + emit-marginalise from a GDC position post."""
    fc = gdc.gdc.forecast(position_post, n_steps=h)
    out = np.zeros(gdc.nA)
    np.add.at(out, gdc._symbol_of_state, fc)
    s = out.sum()
    return out / s if s > 0 else np.full(gdc.nA, 1.0/gdc.nA)


def setup():
    seed_offset = 2  # sparse
    rng = np.random.default_rng(60000 + SEED * 137 + nS * 7 + nA * 11
                                + seed_offset)
    hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                     E_concentration=E_concentration)
    full_train_states = []
    for _ in range(N_TRAIN):
        s, o = hmm.sample(TRAIN_LEN, rng)
        full_train_states.append((s, o))
    train = [o for _, o in full_train_states]
    true_states = np.concatenate([s for s, _ in full_train_states])
    test_pf = []
    for _ in range(N_TEST_PREFIXES):
        s, o = hmm.sample(TEST_PREFIX_LEN, rng)
        test_pf.append((s, o))
    return hmm, train, true_states, test_pf


def find_failure_prefix(hmm, gdc_low, gdc_high, test_pf):
    """Find a prefix where low-α is right and high-α is wrong on
    next-symbol argmax."""
    candidates = []
    for i, (true_s, obs) in enumerate(test_pf):
        true_post = hmm_filter_history(hmm, obs)[-1]
        true_next = hmm_next_obs(hmm, true_post, h=1)
        true_next_argmax = int(np.argmax(true_next))
        low_next = gdc_low.horizon_emission(obs, h=1)
        high_next = gdc_high.horizon_emission(obs, h=1)
        low_correct = (int(np.argmax(low_next)) == true_next_argmax)
        high_correct = (int(np.argmax(high_next)) == true_next_argmax)
        if low_correct and not high_correct:
            candidates.append((i, true_post, true_next, low_next,
                               high_next))
    return candidates


def trace_one_prefix(idx, hmm, gdc_low, gdc_high, true_states, prefix,
                     true_states_seq):
    """Print step-by-step trace for a single prefix."""
    obs = prefix
    print(f"\n=== Trace prefix {idx} ===")
    print(f"  Observed: {obs.tolist()}")
    print(f"  True hidden-state path: {true_states_seq.tolist()}")

    true_history = hmm_filter_history(hmm, obs)
    low_history = gdc_low.gdc.forward_pass(obs.reshape(-1, 1),
                                            return_history=True)[1]
    high_history = gdc_high.gdc.forward_pass(obs.reshape(-1, 1),
                                              return_history=True)[1]

    print(f"\n  Step | obs | true argmax | true H | "
          f"low: argmax(state) state-mass next-argmax(symbol) | "
          f"high: same")
    print("  -----+-----+-------------+--------+"
          "-----------------------------+----------------------------")
    for t in range(len(obs)):
        true_post = true_history[t]
        # Low-α aggregated state posterior
        low_state = aggregate_state_post(low_history[t], true_states,
                                         hmm.nS)
        high_state = aggregate_state_post(high_history[t], true_states,
                                          hmm.nS)
        # Next-symbol argmax (only meaningful at the last step but
        # compute throughout for diagnostic context)
        low_next = predict_next_obs_gdc(gdc_low, low_history[t], h=1)
        high_next = predict_next_obs_gdc(gdc_high, high_history[t], h=1)
        # ESS in position space
        low_ess = 1.0 / np.sum(low_history[t]**2)
        high_ess = 1.0 / np.sum(high_history[t]**2)

        true_arg = int(np.argmax(true_post))
        true_H = float(-np.sum(true_post * np.log2(true_post + 1e-12)))
        print(f"  {t:>4d} | {int(obs[t]):>3d} | "
              f"{true_arg:>11d} | {true_H:>5.2f}  | "
              f"argS={int(np.argmax(low_state)):>2d} "
              f"m={low_state[true_arg]:>4.2f} "
              f"argY={int(np.argmax(low_next)):>2d} "
              f"ess={low_ess:>6.0f} | "
              f"argS={int(np.argmax(high_state)):>2d} "
              f"m={high_state[true_arg]:>4.2f} "
              f"argY={int(np.argmax(high_next)):>2d} "
              f"ess={high_ess:>6.0f}")

    # Final summary
    true_post = true_history[-1]
    true_next = hmm_next_obs(hmm, true_post, h=1)
    low_next = predict_next_obs_gdc(gdc_low, low_history[-1], h=1)
    high_next = predict_next_obs_gdc(gdc_high, high_history[-1], h=1)

    print(f"\n  --- Final state posteriors ---")
    print(f"  True HMM filter argmax-state: "
          f"{int(np.argmax(true_post))}, "
          f"mass={true_post.max():.3f}, H={true_H:.2f} bits")
    low_state = aggregate_state_post(low_history[-1], true_states, hmm.nS)
    high_state = aggregate_state_post(high_history[-1], true_states,
                                      hmm.nS)
    low_H = float(-np.sum(low_state * np.log2(low_state + 1e-12)))
    high_H = float(-np.sum(high_state * np.log2(high_state + 1e-12)))
    print(f"  Low-α  argmax-state: {int(np.argmax(low_state))}, "
          f"top-3 mass: {sorted(low_state, reverse=True)[:3]}, "
          f"H={low_H:.2f}")
    print(f"  High-α argmax-state: {int(np.argmax(high_state))}, "
          f"top-3 mass: {sorted(high_state, reverse=True)[:3]}, "
          f"H={high_H:.2f}")

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
    print(f"=== Loading sparse_large HMM (seed {SEED}) ===")
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
    print(f"  Low α  = {LOW_ALPHA},  High α = {HIGH_ALPHA}")
    print(f"  diffuse fraction = {1-LOW_ALPHA-THETA:.3f} vs "
          f"{1-HIGH_ALPHA-THETA:.3f}\n")

    print(f"=== Finding failure prefix (low correct, high wrong) ===")
    failures = find_failure_prefix(hmm, gdc_low, gdc_high, test_pf)
    print(f"  Found {len(failures)} prefixes where low-α correct & "
          f"high-α wrong (out of {N_TEST_PREFIXES})")
    if not failures:
        print("  No such prefix; the sample HMM is too easy. Quitting.")
        return

    # Trace the first failure case
    idx, true_post, true_next, low_next, high_next = failures[0]
    true_states_seq, obs = test_pf[idx]
    trace_one_prefix(idx, hmm, gdc_low, gdc_high, true_states, obs,
                     true_states_seq)

    # Trace one or two more for breadth
    for k in range(1, min(3, len(failures))):
        idx2, _, _, _, _ = failures[k]
        true_states_seq2, obs2 = test_pf[idx2]
        trace_one_prefix(idx2, hmm, gdc_low, gdc_high, true_states,
                         obs2, true_states_seq2)


if __name__ == "__main__":
    main()
