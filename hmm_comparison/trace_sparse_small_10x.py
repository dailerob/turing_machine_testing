"""sparse_small trace at N_TRAIN=4000 (10x baseline). Torch for bulk
scoring (forward pass over 100 prefixes at 400k chain length is ~1s
on GPU vs ~120s on numpy); numpy only for the per-step trace where we
need the full position-posterior history.

Compares:
  - excess_pp at α=0.8 vs α=0.99 (and the val-pick grid for context)
  - failure-prefix count vs N=400
  - terminal-step ESS distribution vs N=400 (does scaling N really
    raise the ESS at high α as predicted?)
  - step-by-step trace of remaining failures
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from random_hmm import random_sparse_topology_hmm
from generative_dense_chain import GenerativeDenseChain
from gdc_torch_discrete import horizon_emission_many

nS, nA = 10, 4
fanout = 2
E_concentration = 0.1
TRAIN_LEN = 100
N_TRAIN = 4000           # 10x baseline
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
    sp = np.zeros(nS); np.add.at(sp, true_states_per_position, position_post)
    s = sp.sum()
    return sp / s if s > 0 else np.full(nS, 1.0/nS)


def predict_next_obs_from_position(gdc, position_post, sym, h=1):
    fc = gdc.forecast(position_post, n_steps=h)
    out = np.zeros(int(sym.max()) + 1)
    np.add.at(out, sym, fc)
    return out / out.sum() if out.sum() > 0 else \
           np.full_like(out, 1.0/len(out))


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


def torch_predict(gdc, sym, primes, alpha, beta, theta=THETA):
    out = horizon_emission_many(
        symbol_of_state=sym,
        terminal_mask=gdc.terminal_mask,
        start_mask=gdc.start_mask,
        primes=primes, horizons=[1], nA=nA,
        alpha=alpha, theta=theta, beta=beta,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform',
        device='cuda', dtype=torch.float32)
    torch.cuda.synchronize()
    return out.cpu().numpy().squeeze(1)  # (B, nA)


def trace_one(idx, hmm, gdc_low, gdc_high, sym, true_states,
              prefix_obs, prefix_true_states):
    obs = prefix_obs
    print(f"\n=== Trace prefix {idx} ===")
    print(f"  Observed: {obs.tolist()}")
    print(f"  True hidden-state path: {prefix_true_states.tolist()}")

    true_history = hmm_filter_history(hmm, obs)
    low_history = gdc_low.forward_pass(obs.reshape(-1, 1),
                                        return_history=True)[1]
    high_history = gdc_high.forward_pass(obs.reshape(-1, 1),
                                          return_history=True)[1]

    print(f"\n  Step | obs | true argS | true H | "
          f"low: argS m(true) argY ess  | "
          f"high: argS m(true) argY ess")
    print("  -----+-----+-----------+--------+"
          "-----------------------------+----------------------------")
    for t in range(len(obs)):
        true_post = true_history[t]
        low_state = aggregate_state_post(low_history[t], true_states, hmm.nS)
        high_state = aggregate_state_post(high_history[t], true_states, hmm.nS)
        low_next = predict_next_obs_from_position(gdc_low, low_history[t], sym)
        high_next = predict_next_obs_from_position(gdc_high, high_history[t], sym)
        low_ess = 1.0 / np.sum(low_history[t]**2)
        high_ess = 1.0 / np.sum(high_history[t]**2)
        true_arg = int(np.argmax(true_post))
        true_H = float(-np.sum(true_post * np.log2(true_post + 1e-12)))
        print(f"  {t:>4d} | {int(obs[t]):>3d} | "
              f"{true_arg:>9d} | {true_H:>5.2f}  | "
              f"argS={int(np.argmax(low_state)):>2d} "
              f"m={low_state[true_arg]:>4.2f} "
              f"argY={int(np.argmax(low_next)):>1d} "
              f"ess={low_ess:>7.0f} | "
              f"argS={int(np.argmax(high_state)):>2d} "
              f"m={high_state[true_arg]:>4.2f} "
              f"argY={int(np.argmax(high_next)):>1d} "
              f"ess={high_ess:>6.0f}")

    true_post = true_history[-1]
    true_next = hmm_next_obs(hmm, true_post, h=1)
    low_next = predict_next_obs_from_position(gdc_low, low_history[-1], sym)
    high_next = predict_next_obs_from_position(gdc_high, high_history[-1], sym)
    print(f"\n  --- Final ---")
    print(f"  True next:    {true_next.round(3).tolist()}  "
          f"argmax={int(np.argmax(true_next))}")
    print(f"  Low-α next:   {low_next.round(3).tolist()}  "
          f"argmax={int(np.argmax(low_next))}  "
          f"{'CORRECT' if int(np.argmax(low_next))==int(np.argmax(true_next)) else 'WRONG'}")
    print(f"  High-α next:  {high_next.round(3).tolist()}  "
          f"argmax={int(np.argmax(high_next))}  "
          f"{'CORRECT' if int(np.argmax(high_next))==int(np.argmax(true_next)) else 'WRONG'}")
    print(f"  Final ESS  low={1.0/np.sum(low_history[-1]**2):.0f}  "
          f"high={1.0/np.sum(high_history[-1]**2):.0f}")


def main():
    print(f"=== sparse_small @ N_TRAIN={N_TRAIN} (10x), "
          f"TL={TRAIN_LEN}, seed={SEED} ===\n", flush=True)
    t0 = time.time()
    hmm, train, true_states, test_pf = setup()
    print(f"  Chain length: {N_TRAIN * TRAIN_LEN} positions  "
          f"(setup {time.time()-t0:.1f}s)", flush=True)

    # Build the chain only ONCE — reuse for both alphas
    seq_arrays = [s.reshape(-1, 1).astype(np.int64) for s in train]
    t0 = time.time()
    gdc_low = GenerativeDenseChain(
        seq_arrays, alpha=LOW_ALPHA, theta=THETA, gamma=0.0, beta=BETA,
        transition_type='self_loop', initial_dist='uniform',
        terminal_behavior='absorb')
    gdc_high = GenerativeDenseChain(
        seq_arrays, alpha=HIGH_ALPHA, theta=THETA, gamma=0.0, beta=BETA,
        transition_type='self_loop', initial_dist='uniform',
        terminal_behavior='absorb')
    sym = gdc_low.states[:, 0].astype(np.int64)
    print(f"  GDCs built ({time.time()-t0:.1f}s)\n", flush=True)

    primes = np.stack([np.asarray(o, dtype=np.int64) for _, o in test_pf])

    # Torch bulk score
    t0 = time.time()
    low_pred = torch_predict(gdc_low, sym, primes, LOW_ALPHA, BETA)
    high_pred = torch_predict(gdc_high, sym, primes, HIGH_ALPHA, BETA)
    print(f"  Torch bulk predict ({time.time()-t0:.1f}s)", flush=True)

    # True per-prefix metrics
    low_correct = high_correct = 0
    failures = []
    cross_ent_low = cross_ent_high = floor_total = 0.0
    for i, (true_s, obs) in enumerate(test_pf):
        true_post = hmm_filter_history(hmm, obs)[-1]
        true_next = hmm_next_obs(hmm, true_post, h=1)
        true_arg = int(np.argmax(true_next))
        l_corr = int(np.argmax(low_pred[i])) == true_arg
        h_corr = int(np.argmax(high_pred[i])) == true_arg
        if l_corr: low_correct += 1
        if h_corr: high_correct += 1
        if l_corr and not h_corr: failures.append(i)
        # cross-entropy of model vs true_next  (for excess_pp aggregate)
        eps = 1e-12
        cross_ent_low  += float(-np.sum(true_next * np.log2(low_pred[i] + eps)))
        cross_ent_high += float(-np.sum(true_next * np.log2(high_pred[i] + eps)))
        floor_total    += float(-np.sum(true_next * np.log2(true_next + eps)))
    avg_excess_low  = 2 ** ((cross_ent_low - floor_total) / N_TEST_PREFIXES)
    avg_excess_high = 2 ** ((cross_ent_high - floor_total) / N_TEST_PREFIXES)
    print(f"  Excess perplexity (avg over prefixes):  "
          f"low-α={avg_excess_low:.4f}  high-α={avg_excess_high:.4f}")
    print(f"  Next-symbol argmax-correct:  "
          f"low-α={low_correct}/{N_TEST_PREFIXES}  "
          f"high-α={high_correct}/{N_TEST_PREFIXES}")
    print(f"  Failure prefixes (low correct, high wrong): {len(failures)}")
    print()

    # Compute terminal ESS for each prefix using numpy forward_pass
    # (this is the slow part; only done at the terminal step, no history)
    t0 = time.time()
    low_ess_terminal, high_ess_terminal = [], []
    for true_s, obs in test_pf:
        l_pos = gdc_low.forward_pass(obs.reshape(-1, 1))
        h_pos = gdc_high.forward_pass(obs.reshape(-1, 1))
        low_ess_terminal.append(1.0 / np.sum(l_pos**2))
        high_ess_terminal.append(1.0 / np.sum(h_pos**2))
    print(f"  Terminal-step ESS (median over 100 prefixes):  "
          f"low={np.median(low_ess_terminal):.0f}  "
          f"high={np.median(high_ess_terminal):.0f}  "
          f"({time.time()-t0:.1f}s)")
    print(f"  Terminal-step high-α ESS  min={np.min(high_ess_terminal):.0f}  "
          f"max={np.max(high_ess_terminal):.0f}  "
          f"mean={np.mean(high_ess_terminal):.0f}\n")

    # Step-by-step trace of failures
    if failures:
        for k, idx in enumerate(failures[:3]):
            true_s, obs = test_pf[idx]
            trace_one(idx, hmm, gdc_low, gdc_high, sym, true_states,
                      obs, true_s)
    else:
        print("\n  No high-α failures.")


if __name__ == "__main__":
    main()
