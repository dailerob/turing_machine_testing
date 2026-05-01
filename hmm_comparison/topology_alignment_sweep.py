"""
Cross-topology hidden-state alignment sweep.

For each HMM topology (and 2 seeds), we sweep GDC params (diffusion d,
emission-noise beta) and measure how well GDC's posterior routes mass to
the correct hidden state. Metric: alignment lift

    lift = (GDC mean diagonal - stationary self-overlap) /
           (Bayes mean diagonal - stationary self-overlap)

Bayes ceiling = HMM forward filter alpha_t. Lift = 1 means GDC matches
the Bayes-optimal estimator at this metric; 0 = no better than prior.

Outputs:
    topology_alignment_results.csv  one row per (topology, seed, d, beta)
    fig_topology_heatmaps.png       heatmap (d x beta) per topology, mean over seeds
    fig_topology_best.png           bar chart of best lift per topology
    topology_best_params.csv        best (d, beta) per topology
"""

from __future__ import annotations
import os, sys, csv, time
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from random_hmm import (random_dense_hmm, random_sparse_topology_hmm,
                        random_lowrank_hmm)
from model_wrappers import fit_gdc

N_TRAIN_SEQ = 250
TRAIN_LEN = 40
N_EVAL_SEQ = 120
EVAL_LEN = 40
SEEDS = [0, 1]

DIFFUSIONS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
BETAS = [0.05, 0.1, 0.2]
ALPHA_THETA_RATIO = (0.7, 0.3)  # of (1-d)


# Topologies
def make_hmm(name: str, seed: int):
    rng = np.random.default_rng(1234 + 17 * seed + hash(name) % 100)
    if name == 'small_dense':
        return random_dense_hmm(4, 3, rng), 4, 3
    if name == 'sparse_fanout2':
        return random_sparse_topology_hmm(6, 4, rng, fanout=2), 6, 4
    if name == 'low_rank2':
        return random_lowrank_hmm(6, 4, rank=2, rng=rng), 6, 4
    if name == 'det_emissions':
        return random_dense_hmm(4, 3, rng, E_concentration=0.1), 4, 3
    if name == 'uniform_emissions':
        return random_dense_hmm(4, 3, rng, E_concentration=10.0), 4, 3
    if name == 'large':
        return random_dense_hmm(8, 5, rng), 8, 5
    raise ValueError(name)


TOPOLOGIES = ['small_dense', 'sparse_fanout2', 'low_rank2',
              'det_emissions', 'uniform_emissions', 'large']


def stationary(T):
    n = T.shape[0]
    pi = np.full(n, 1.0 / n)
    for _ in range(500):
        pi = pi @ T
    pi = np.maximum(pi, 0)
    s = pi.sum()
    return pi / s if s > 0 else np.full(n, 1.0 / n)


def split(d):
    base = 1.0 - d
    a, t = ALPHA_THETA_RATIO
    return a * base, t * base


def sample_training_with_states(hmm, n_seq, length, rng):
    obs_seqs = []; h_train = []
    for _ in range(n_seq):
        states, obs = hmm.sample(length, rng)
        obs_seqs.append(obs); h_train.append(states)
    return obs_seqs, np.concatenate(h_train)


def bayes_diag(hmm, eval_obs, eval_states):
    """E[alpha_t[s_true_t]] -- weighted by actual class frequency.
    This is the proper Bayes ceiling because lift = (X - prior)/(Bayes-prior)
    needs both X and Bayes to be the *same* expectation (over timepoints
    drawn from the stationary state distribution at long sequence length)."""
    total = 0.0; n = 0
    for obs, st in zip(eval_obs, eval_states):
        a = hmm.pi * hmm.E[:, obs[0]]; a = a / max(a.sum(), 1e-12)
        H = [a.copy()]
        for ob in obs[1:]:
            a = (a @ hmm.T) * hmm.E[:, ob]; a = a / max(a.sum(), 1e-12)
            H.append(a.copy())
        H = np.vstack(H)            # (T, nS)
        for t in range(H.shape[0]):
            total += float(H[t, int(st[t])])
            n += 1
    return total / max(n, 1)


def gdc_diag(hmm, train_obs, h_train, eval_obs, eval_states, d, beta):
    """E[ p_hidden_t[s_true_t] ] -- weighted by actual class frequency."""
    a, t = split(d)
    gdc = fit_gdc(train_obs, alphabet_size=hmm.nA,
                  alpha=a, theta=t, gamma=0.0, beta=beta,
                  transition_type='self_loop', initial_dist='sequence_starts')
    nS = hmm.nS; n_gdc = gdc.gdc.n_states
    P_lab = np.zeros((n_gdc, nS))
    P_lab[np.arange(n_gdc), h_train] = 1.0
    total = 0.0; n = 0
    for obs, st in zip(eval_obs, eval_states):
        oc = np.asarray(obs, np.int64).reshape(-1, 1)
        _, hist = gdc.gdc.forward_pass(oc, return_history=True)
        p_hidden = hist @ P_lab
        for ti in range(p_hidden.shape[0]):
            total += float(p_hidden[ti, int(st[ti])])
            n += 1
    return total / max(n, 1)


def run_topology(name: str, seed: int):
    hmm, nS, nA = make_hmm(name, seed)
    pi_stat = stationary(hmm.T)
    stationary_self = float(np.sum(pi_stat ** 2))
    rng = np.random.default_rng(7777 + seed + hash(name) % 200)
    train_obs, h_train = sample_training_with_states(
        hmm, N_TRAIN_SEQ, TRAIN_LEN, rng)
    eval_obs, eval_states = [], []
    for _ in range(N_EVAL_SEQ):
        s, o = hmm.sample(EVAL_LEN, rng)
        eval_obs.append(o); eval_states.append(s)
    bayes = bayes_diag(hmm, eval_obs, eval_states)
    out = []
    for d in DIFFUSIONS:
        for beta in BETAS:
            gdc = gdc_diag(hmm, train_obs, h_train, eval_obs, eval_states,
                           d, beta)
            denom = bayes - stationary_self
            lift = ((gdc - stationary_self) / denom) if denom > 1e-9 else float('nan')
            out.append({'topology': name, 'seed': seed,
                        'nS': nS, 'nA': nA,
                        'd': d, 'beta': beta,
                        'gdc_diag': gdc, 'bayes_diag': bayes,
                        'stationary': stationary_self,
                        'lift': lift})
    return out


def main():
    rows = []
    t0 = time.time()
    for name in TOPOLOGIES:
        for seed in SEEDS:
            tic = time.time()
            r = run_topology(name, seed)
            rows.extend(r)
            best = max(r, key=lambda x: x['lift'])
            print(f'  {name:>20s} seed={seed} '
                  f'bayes={best["bayes_diag"]:.3f} stat={best["stationary"]:.3f} '
                  f'best_lift={best["lift"]:.3f} '
                  f'@(d={best["d"]:.2f},beta={best["beta"]:.2f})  '
                  f'[{time.time()-tic:.1f}s]', flush=True)
    print(f'Total: {time.time()-t0:.1f}s')

    # Save full sweep
    csv_path = os.path.join(_THIS_DIR, 'topology_alignment_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('Wrote', csv_path)

    # Per (topology, d, beta) mean over seeds.
    grid = defaultdict(list)
    for r in rows:
        grid[(r['topology'], r['d'], r['beta'])].append(r['lift'])
    mean_grid = {k: float(np.mean(v)) for k, v in grid.items()}

    # Also compute mean GDC diag per cell, so we can rank even when 'lift'
    # is nan because Bayes <= stationary.
    diag_grid = defaultdict(list)
    for r in rows:
        diag_grid[(r['topology'], r['d'], r['beta'])].append(r['gdc_diag'])
    mean_diag_grid = {k: float(np.mean(v)) for k, v in diag_grid.items()}

    # Best params per topology (mean over seeds). Rank by lift if valid;
    # fall back to raw gdc_diag if lift is undefined.
    best_rows = []
    for name in TOPOLOGIES:
        best_key = (DIFFUSIONS[0], BETAS[0])
        best_val = -np.inf
        used_fallback = False
        # try lift first
        any_valid = any(not np.isnan(mean_grid[(name, d, beta)])
                        for d in DIFFUSIONS for beta in BETAS)
        for d in DIFFUSIONS:
            for beta in BETAS:
                if any_valid:
                    v = mean_grid[(name, d, beta)]
                    if np.isnan(v):
                        continue
                else:
                    v = mean_diag_grid[(name, d, beta)]
                    used_fallback = True
                if v > best_val:
                    best_val = v; best_key = (d, beta)
        bayes_mean = float(np.mean([r['bayes_diag'] for r in rows
                                    if r['topology'] == name]))
        stat_mean = float(np.mean([r['stationary'] for r in rows
                                   if r['topology'] == name]))
        best_rows.append({'topology': name,
                          'best_d': best_key[0], 'best_beta': best_key[1],
                          'best_lift_mean_seeds': best_val,
                          'metric': 'gdc_diag' if used_fallback else 'lift',
                          'bayes_mean': bayes_mean,
                          'stationary_mean': stat_mean,
                          'best_gdc_diag_mean': mean_diag_grid[
                              (name, best_key[0], best_key[1])]})
        flag = ' (gdc_diag fallback; bayes <= stationary)' if used_fallback else ''
        print(f'  {name:>20s} BEST: d={best_key[0]:.2f} beta={best_key[1]:.2f} '
              f'val={best_val:.3f}{flag}  (bayes={bayes_mean:.3f}, '
              f'stat={stat_mean:.3f})')

    # Save best
    best_csv = os.path.join(_THIS_DIR, 'topology_best_params.csv')
    with open(best_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
        w.writeheader(); w.writerows(best_rows)
    print('Wrote', best_csv)

    # Heatmaps
    rows_heat = 2; cols_heat = 3
    fig, axes = plt.subplots(rows_heat, cols_heat,
                             figsize=(4 * cols_heat, 3.4 * rows_heat),
                             squeeze=False)
    for ax, name in zip(axes.flat, TOPOLOGIES):
        # Use lift if valid anywhere, else gdc_diag
        any_valid = any(not np.isnan(mean_grid[(name, d, beta)])
                        for d in DIFFUSIONS for beta in BETAS)
        if any_valid:
            H = np.array([[mean_grid[(name, d, beta)] for beta in BETAS]
                          for d in DIFFUSIONS])
            metric_label = 'lift'
        else:
            H = np.array([[mean_diag_grid[(name, d, beta)] for beta in BETAS]
                          for d in DIFFUSIONS])
            metric_label = 'gdc_diag (lift undefined)'
        im = ax.imshow(H, aspect='auto', cmap='viridis',
                       vmin=max(0.0, H.min() - 0.05),
                       vmax=min(1.05, H.max() + 0.05))
        ax.set_xticks(range(len(BETAS)))
        ax.set_xticklabels([f'{b:.2f}' for b in BETAS])
        ax.set_yticks(range(len(DIFFUSIONS)))
        ax.set_yticklabels([f'{d:.1f}' for d in DIFFUSIONS])
        ax.set_xlabel('beta (emission noise)')
        ax.set_ylabel('d (diffusion)')
        ax.set_title(f'{name}  [{metric_label}]')
        for i, d in enumerate(DIFFUSIONS):
            for j, beta in enumerate(BETAS):
                ax.text(j, i, f'{H[i,j]:.2f}',
                        ha='center', va='center',
                        color='white' if H[i, j] < 0.6 else 'black',
                        fontsize=9)
        plt.colorbar(im, ax=ax, label=metric_label, shrink=0.8)
    fig.suptitle('Hidden-state alignment lift over (d, beta), mean of 2 seeds.\n'
                 'lift = (GDC - stationary) / (Bayes - stationary).  '
                 '1.0 = Bayes-optimal.', fontsize=11)
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_topology_heatmaps.png')
    plt.savefig(out, dpi=130); plt.close()
    print('Wrote', out)

    # Best bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r['topology'] for r in best_rows]
    lifts = [r['best_lift_mean_seeds'] for r in best_rows]
    ax.bar(names, lifts, color='steelblue', edgecolor='black')
    for i, r in enumerate(best_rows):
        ax.text(i, lifts[i] + 0.01,
                f'd={r["best_d"]:.2f}\nbeta={r["best_beta"]:.2f}',
                ha='center', va='bottom', fontsize=8)
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.6,
               label='Bayes-optimal (1.0)')
    ax.axhline(0.0, color='grey', linestyle='-', alpha=0.6,
               label='no better than prior (0.0)')
    ax.set_ylabel('best alignment lift')
    ax.set_title('Best GDC param config per HMM topology')
    ax.legend(fontsize=9, loc='lower right')
    plt.xticks(rotation=20, ha='right')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_topology_best.png')
    plt.savefig(out, dpi=130); plt.close()
    print('Wrote', out)
    print('Done.')


if __name__ == '__main__':
    main()
