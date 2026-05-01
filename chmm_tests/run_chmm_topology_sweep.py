"""CHMM analogue of `paper_topology_and_samples.py` (EXP1).

For each of the same 9 topologies x 3 seeds, train a CHMM at several
clone counts K, and report:

    bps_chmm     bits-per-symbol on held-out eval sequences
    bayes_bps    same under the true HMM (Bayes ceiling)
    chmm_diag    weighted hidden-state alignment diagonal
                 (E_t [ alpha_chmm @ P_label[s_true_t] ])
    lift         (chmm_diag - stat) / (bayes_diag - stat)

Hidden-state alignment is computed exactly as in the GDC paper sweep:
* the chmm's smoothed posterior over clones at training timesteps,
  weighted by the (known) true hidden state at that timestep, gives a
  soft mapping P_lab[clone, hidden_state];
* on eval, run forward to get fwd messages over clones at each t,
  multiply by P_lab to get p_hidden[t, s], take the diagonal at the
  true s_t, and average.

The sweep is intentionally aligned with
`hmm_comparison/paper_topology_results.csv` (same N_train, T_len,
N_eval, EVAL_LEN, seeds) so the numbers compare directly.

Run:
    python chmm_tests/run_chmm_topology_sweep.py
"""

from __future__ import annotations

import os, sys, csv, time
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "hmm_comparison"))
sys.path.insert(0, os.path.join(HERE, "naturecomm_cscg"))

from paper_topology_and_samples import (  # noqa: E402
    TOPOLOGIES, make_hmm, stationary, sample_training_with_states,
    bayes_diag, N_TRAIN_SEQ_FIXED, TRAIN_LEN, N_EVAL_SEQ, EVAL_LEN, SEEDS,
)
from chmm_actions import CHMM, forward, backward  # noqa: E402

K_GRID = [1, 2, 4, 8]
N_EM_ITERS = 80


def chmm_label_map(model, x_train, a_train, h_train, nS):
    """Compute soft P_lab[global_clone, hidden_state] from training-time
    smoothed posterior weighted by the ground-truth hidden labels."""
    log2_lik, mess_fwd = forward(
        model.T.transpose(0, 2, 1), model.Pi_x, model.n_clones,
        x_train, a_train, store_messages=True,
    )
    mess_bwd = backward(model.T, model.n_clones, x_train, a_train)
    n_clones = model.n_clones
    state_loc = np.hstack(([0], n_clones)).cumsum()
    mess_loc = np.hstack(([0], n_clones[x_train])).cumsum()
    n_states = int(state_loc[-1])
    P_lab = np.zeros((n_states, nS), dtype=np.float64)
    weights = np.zeros(n_states, dtype=np.float64)
    for t in range(len(x_train)):
        xt = int(x_train[t])
        gs, ge = int(state_loc[xt]), int(state_loc[xt + 1])
        ms, me = int(mess_loc[t]), int(mess_loc[t + 1])
        gamma = mess_fwd[ms:me].astype(np.float64) * mess_bwd[ms:me].astype(np.float64)
        z = gamma.sum()
        if z <= 0:
            continue
        gamma /= z
        s_true = int(h_train[t])
        P_lab[gs:ge, s_true] += gamma
        weights[gs:ge] += gamma
    nz = weights > 1e-12
    P_lab[nz] /= weights[nz, None]
    # Rows with no weight: leave as zero (they'll never be sampled at
    # eval since they correspond to unused clones).
    return P_lab, state_loc


def chmm_diag_lift(model, eval_obs, eval_states, P_lab, state_loc):
    """Weighted diagonal of CHMM hidden-state posterior at the true s_t,
    averaged over eval timesteps."""
    total, n = 0.0, 0
    for obs, st in zip(eval_obs, eval_states):
        x = obs.astype(np.int64)
        a = np.zeros_like(x)
        log2_lik, mess_fwd = forward(
            model.T.transpose(0, 2, 1), model.Pi_x, model.n_clones,
            x, a, store_messages=True,
        )
        n_clones = model.n_clones
        mess_loc = np.hstack(([0], n_clones[x])).cumsum()
        for t in range(len(x)):
            xt = int(x[t])
            gs, ge = int(state_loc[xt]), int(state_loc[xt + 1])
            ms, me = int(mess_loc[t]), int(mess_loc[t + 1])
            alpha = mess_fwd[ms:me].astype(np.float64)
            p_hidden = alpha @ P_lab[gs:ge]
            total += float(p_hidden[int(st[t])])
            n += 1
    return total / max(n, 1)


def chmm_eval_bps(model, eval_obs):
    total_bps, total_n = 0.0, 0
    for obs in eval_obs:
        x = obs.astype(np.int64)
        a = np.zeros_like(x)
        bps_arr = np.asarray(model.bps(x, a))
        total_bps += float(bps_arr.sum())
        total_n += len(x)
    return total_bps / total_n


def true_hmm_bps(hmm, eval_obs):
    total, total_n = 0.0, 0
    for obs in eval_obs:
        a = hmm.pi * hmm.E[:, obs[0]]; s = a.sum()
        total += np.log2(max(s, 1e-300))
        a = a / s if s > 0 else np.full(hmm.nS, 1.0 / hmm.nS)
        for o in obs[1:]:
            a = (a @ hmm.T) * hmm.E[:, o]; s = a.sum()
            total += np.log2(max(s, 1e-300))
            a = a / s if s > 0 else np.full(hmm.nS, 1.0 / hmm.nS)
        total_n += len(obs)
    return -total / total_n


def main():
    t0 = time.time()
    rows = []
    for name in TOPOLOGIES:
        for seed in SEEDS:
            tic = time.time()
            hmm, nS, nA = make_hmm(name, seed)
            pi_stat = stationary(hmm.T)
            stat_self = float(np.sum(pi_stat ** 2))
            rng = np.random.default_rng(7777 + seed + hash(name) % 200)
            train_obs, h_train = sample_training_with_states(
                hmm, N_TRAIN_SEQ_FIXED, TRAIN_LEN, rng)
            eval_obs, eval_states = [], []
            for _ in range(N_EVAL_SEQ):
                s, o = hmm.sample(EVAL_LEN, rng)
                eval_obs.append(o); eval_states.append(s)
            bayes = bayes_diag(hmm, eval_obs, eval_states)
            bayes_bps = true_hmm_bps(hmm, eval_obs)
            denom = bayes - stat_self

            x_train = np.concatenate(train_obs).astype(np.int64)
            a_train = np.zeros_like(x_train)

            for K in K_GRID:
                n_clones = np.full(nA, K, dtype=np.int64)
                model = CHMM(n_clones=n_clones, x=x_train, a=a_train,
                             pseudocount=1e-3, seed=0)
                model.learn_em_T(x_train, a_train, n_iter=N_EM_ITERS,
                                 term_early=True)
                P_lab, state_loc = chmm_label_map(model, x_train, a_train,
                                                  h_train, nS)
                diag = chmm_diag_lift(model, eval_obs, eval_states,
                                      P_lab, state_loc)
                bps = chmm_eval_bps(model, eval_obs)
                lift = ((diag - stat_self) / denom
                        if abs(denom) > 1e-3 else float('nan'))
                rows.append({
                    'topology': name, 'seed': seed, 'nS': nS, 'nA': nA,
                    'K': K, 'n_states': int(K * nA),
                    'chmm_diag': diag, 'bayes': bayes,
                    'stationary': stat_self, 'lift': lift,
                    'gain': diag - stat_self, 'chmm_bps': bps,
                    'bayes_bps': bayes_bps,
                })
            best_lift = max((r['lift'] for r in rows
                             if r['topology'] == name and r['seed'] == seed
                             and not np.isnan(r['lift'])), default=float('nan'))
            best_bps = min(r['chmm_bps'] for r in rows
                           if r['topology'] == name and r['seed'] == seed)
            print(f'  {name:>16s} seed={seed} '
                  f'bayes={bayes:.3f} stat={stat_self:.3f} '
                  f'best_lift={best_lift:.3f} '
                  f'best_bps={best_bps:.3f} (bayes_bps={bayes_bps:.3f})  '
                  f'[{time.time()-tic:.1f}s]', flush=True)

    out_csv = os.path.join(HERE, 'chmm_topology_results.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('Wrote', out_csv)

    # Aggregate best K per topology by mean lift across seeds (fallback
    # to gain when lift is undefined for the topology).
    best_csv = os.path.join(HERE, 'chmm_topology_best.csv')
    by_grp = defaultdict(list)
    for r in rows:
        by_grp[(r['topology'], r['K'])].append(r)
    summary = []
    for name in TOPOLOGIES:
        any_lift_valid = any(
            not np.isnan(r['lift'])
            for K in K_GRID for r in by_grp[(name, K)]
        )
        best_K, best_score = K_GRID[0], -np.inf
        for K in K_GRID:
            rs = by_grp[(name, K)]
            if any_lift_valid:
                scores = [r['lift'] for r in rs if not np.isnan(r['lift'])]
                score = float(np.mean(scores)) if scores else -np.inf
            else:
                score = float(np.mean([r['gain'] for r in rs]))
            if score > best_score:
                best_score, best_K = score, K
        rs = by_grp[(name, best_K)]
        mean_lift = float(np.nanmean([r['lift'] for r in rs]))
        mean_gain = float(np.mean([r['gain'] for r in rs]))
        mean_bps = float(np.mean([r['chmm_bps'] for r in rs]))
        bayes_bps = float(np.mean([r['bayes_bps'] for r in rs]))
        summary.append((name, best_K, mean_lift, mean_gain, mean_bps, bayes_bps))
    with open(best_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['topology', 'best_K', 'mean_lift', 'mean_gain',
                    'chmm_bps', 'bayes_bps'])
        w.writerows(summary)
    print('Wrote', best_csv)
    print(f'\nTotal: {time.time()-t0:.1f}s')

    print('\n=== summary (best K per topology) ===')
    print(f'{"topology":>17s}  {"best_K":>6s}  {"lift":>6s}  {"gain":>6s}  '
          f'{"bps":>6s}  {"bayes_bps":>10s}')
    for name, K, lift, gain, bps, bbps in summary:
        print(f'{name:>17s}  {K:>6d}  {lift:>6.3f}  {gain:>+6.3f}  '
              f'{bps:>6.3f}  {bbps:>10.3f}')


if __name__ == '__main__':
    main()
