"""
Paper-grade topology + sample-efficiency sweep for GDC hidden-state
alignment.

Topologies (3 seeds each):
    1.  small_dense           Dirichlet(1) T,E.  (nS=4, nA=3)
    2.  sparse_fanout2        each row of T has 2 successors.
                              (nS=6, nA=4)
    3.  low_rank2             rank(T)=2.
                              (nS=6, nA=4)
    4.  det_emissions         E ~ Dirichlet(0.1).
                              (nS=4, nA=3)
    5.  uniform_emissions     E ~ Dirichlet(10).
                              (nS=4, nA=3)
    6.  large                 dense Dirichlet(1).
                              (nS=8, nA=5)
    7.  moore_ring            deterministic ring T, E cycles s_i -> i%nA.
                              (nS=8, nA=3)
    8.  mealy_det             two deterministic outgoing arcs per state,
                              each arc emits a fixed symbol.
                              (nS=6, nA=2)
    9.  path_chain            left-to-right chain, self_loop=0.7,
                              forward=0.3, Dirichlet(1) E.
                              (nS=6, nA=3)

Two experiments:

EXP1: Topology x (d, beta) heatmap with 3 seeds.
    For every (topology, seed, d, beta), train GDC on 250 sequences of
    length 40 and evaluate hidden-state alignment lift on 120x40
    sequences.  d in {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}; beta in
    {0.05, 0.1, 0.2}.  Best (d, beta) per topology reported.

EXP2: Sample efficiency.
    For 4 representative topologies (small_dense, sparse_fanout2,
    moore_ring, mealy_det), sweep N_train_seq in {25, 50, 100, 200, 400,
    800}, 3 seeds each, at GDC params taken from EXP1's best per topology
    (forced to (d=0.1, beta=0.05) if multi-modal).

Outputs:
    paper_topology_results.csv         (EXP1 rows)
    paper_topology_best.csv            (best per topology)
    paper_n_train_results.csv          (EXP2 rows)
    fig_paper_topology_heatmaps.png    (d x beta) heatmap per topology
    fig_paper_topology_best.png        bar of best lift per topology
    fig_paper_n_train.png              lift vs N_train
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

from random_hmm import (RandomHMM, random_dense_hmm,
                        random_sparse_topology_hmm, random_lowrank_hmm)
from model_wrappers import fit_gdc

# --- EXP1 config ---
N_TRAIN_SEQ_FIXED = 250
TRAIN_LEN = 40
N_EVAL_SEQ = 120
EVAL_LEN = 40
SEEDS = [0, 1, 2]
DIFFUSIONS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
BETAS = [0.05, 0.1, 0.2]
ALPHA_THETA_RATIO = (0.7, 0.3)

# --- EXP2 config ---
N_TRAIN_SWEEP = [25, 50, 100, 200, 400, 800]
N_TRAIN_TOPOLOGIES = ['small_dense', 'sparse_fanout2',
                      'moore_ring', 'mealy_det']

TOPOLOGIES = ['small_dense', 'sparse_fanout2', 'low_rank2',
              'det_emissions', 'uniform_emissions', 'large',
              'moore_ring', 'mealy_det', 'path_chain']


# --- Custom HMM builders for the new deterministic topologies ---
def moore_ring_hmm(nS=8, nA=3):
    """Deterministic ring: T[i, (i+1)%nS]=1.  E[i, i%nA]=1."""
    T = np.zeros((nS, nS))
    for i in range(nS):
        T[i, (i + 1) % nS] = 1.0
    E = np.zeros((nS, nA))
    for i in range(nS):
        E[i, i % nA] = 1.0
    pi = np.full(nS, 1.0 / nS)
    return RandomHMM(T, E, pi)


def mealy_det_hmm(nS=6, nA=2, rng=None):
    """Each state has 2 deterministic outgoing arcs; the arc taken when
    we are about to emit symbol a goes to a fixed next state.  Modelled
    as: T[i,j] = 1/nA for the chosen successor of arc a (a in 0..nA-1),
    and E_{(i,j)} concentrated on a.  We collapse to standard HMM by
    interpreting:  P(state=j_a, emit=a | state=i) = (1/nA) when (i, a) ->
    j_a is the wired arc.

    Concretely: pick rng-deterministic mapping arcs : (i, a) -> j;
    set T[i, j] = sum over a with arcs[i,a]=j of 1/nA;  E[i, a] = 1/nA
    times degree.

    The simpler equivalent HMM:
        Choose succ[i, a] in {0..nS-1}.
        T[i, j] = (1/nA) * |{a : succ[i, a] == j}|
        E[i, a] = sum_j P(state=j | i) * 1{ succ-arc-with-symbol = a }
                = (1/nA) for each a (since per-arc symbol is its arc).

    But the EMISSION at state i depends on the arc taken which depends on
    the next state -- so this isn't a vanilla HMM (it's a Mealy).
    Instead, re-encode states as (prev_state, prev_symbol) pairs so that
    the emission at the new state is determined.  Final equivalent HMM:
        new states  = nS * nA  (i, a) pairs
        T[(i, a), (succ[i, a], a')] = 1/nA  for all a'
        E[(i, a), a] = 1
    """
    if rng is None:
        rng = np.random.default_rng(0)
    succ = rng.integers(0, nS, size=(nS, nA))   # succ[i, a]: state to enter
    new_states = [(i, a) for i in range(nS) for a in range(nA)]
    M = len(new_states)
    idx = {sa: k for k, sa in enumerate(new_states)}
    T2 = np.zeros((M, M))
    E2 = np.zeros((M, nA))
    for (i, a), k in idx.items():
        E2[k, a] = 1.0
        for a_next in range(nA):
            j = int(succ[i, a_next])
            T2[k, idx[(j, a_next)]] += 1.0 / nA
    pi = np.full(M, 1.0 / M)
    return RandomHMM(T2, E2, pi)


def path_chain_hmm(nS=6, nA=3, rng=None):
    """Left-to-right chain with self-loop 0.7 and forward 0.3 (sink at
    end with self-loop 1.0).  Emissions Dirichlet(1)."""
    if rng is None:
        rng = np.random.default_rng(0)
    T = np.zeros((nS, nS))
    for i in range(nS - 1):
        T[i, i] = 0.7; T[i, i + 1] = 0.3
    T[nS - 1, nS - 1] = 1.0
    E = rng.dirichlet(np.full(nA, 1.0), size=nS)
    pi = np.zeros(nS); pi[0] = 1.0
    return RandomHMM(T, E, pi)


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
    if name == 'moore_ring':
        h = moore_ring_hmm(nS=8, nA=3)
        return h, h.nS, h.nA
    if name == 'mealy_det':
        h = mealy_det_hmm(nS=6, nA=2, rng=rng)
        return h, h.nS, h.nA
    if name == 'path_chain':
        h = path_chain_hmm(nS=6, nA=3, rng=rng)
        return h, h.nS, h.nA
    raise ValueError(name)


# --- Common helpers ---
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
    total = 0.0; n = 0
    for obs, st in zip(eval_obs, eval_states):
        a = hmm.pi * hmm.E[:, obs[0]]; a = a / max(a.sum(), 1e-12)
        H = [a.copy()]
        for ob in obs[1:]:
            a = (a @ hmm.T) * hmm.E[:, ob]; a = a / max(a.sum(), 1e-12)
            H.append(a.copy())
        H = np.vstack(H)
        for t in range(H.shape[0]):
            total += float(H[t, int(st[t])])
            n += 1
    return total / max(n, 1)


def gdc_diag(hmm, train_obs, h_train, eval_obs, eval_states, d, beta):
    a, t = split(d)
    gdc = fit_gdc(train_obs, alphabet_size=hmm.nA,
                  alpha=a, theta=t, gamma=0.0, beta=beta,
                  transition_type='self_loop',
                  initial_dist='sequence_starts')
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


# --- EXP1 ---
def run_exp1():
    rows = []
    t0 = time.time()
    for name in TOPOLOGIES:
        for seed in SEEDS:
            tic = time.time()
            hmm, nS, nA = make_hmm(name, seed)
            pi_stat = stationary(hmm.T)
            stat_self = float(np.sum(pi_stat ** 2))
            rng = np.random.default_rng(7777 + seed + hash(name) % 200)
            train_obs, h_train = sample_training_with_states(
                hmm, N_TRAIN_SEQ_FIXED, TRAIN_LEN, rng)
            eval_obs = []; eval_states = []
            for _ in range(N_EVAL_SEQ):
                s, o = hmm.sample(EVAL_LEN, rng)
                eval_obs.append(o); eval_states.append(s)
            bayes = bayes_diag(hmm, eval_obs, eval_states)
            best = (-np.inf, None)
            for d in DIFFUSIONS:
                for beta in BETAS:
                    g = gdc_diag(hmm, train_obs, h_train, eval_obs,
                                 eval_states, d, beta)
                    denom = bayes - stat_self
                    lift = ((g - stat_self) / denom) if abs(denom) > 1e-3 else float('nan')
                    rows.append({'topology': name, 'seed': seed, 'nS': nS,
                                 'nA': nA, 'd': d, 'beta': beta,
                                 'gdc_diag': g, 'bayes': bayes,
                                 'stationary': stat_self, 'lift': lift,
                                 'gain': g - stat_self})
                    if not np.isnan(lift) and lift > best[0]:
                        best = (lift, (d, beta))
            print(f'  EXP1 {name:>16s} seed={seed} '
                  f'bayes={bayes:.3f} stat={stat_self:.3f} '
                  f'best_lift={best[0]:.3f} @{best[1]}  '
                  f'[{time.time()-tic:.1f}s]', flush=True)
    print(f'EXP1 total: {time.time()-t0:.1f}s', flush=True)
    csv_path = os.path.join(_THIS_DIR, 'paper_topology_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('Wrote', csv_path)
    return rows


# --- EXP2 ---
def run_exp2(best_per_topology):
    rows = []
    t0 = time.time()
    for name in N_TRAIN_TOPOLOGIES:
        d_best, beta_best = best_per_topology.get(name, (0.1, 0.05))
        for n_train in N_TRAIN_SWEEP:
            for seed in SEEDS:
                tic = time.time()
                hmm, nS, nA = make_hmm(name, seed)
                pi_stat = stationary(hmm.T)
                stat_self = float(np.sum(pi_stat ** 2))
                rng = np.random.default_rng(33333 + seed + hash(name) % 200)
                train_obs, h_train = sample_training_with_states(
                    hmm, n_train, TRAIN_LEN, rng)
                eval_obs = []; eval_states = []
                for _ in range(N_EVAL_SEQ):
                    s, o = hmm.sample(EVAL_LEN, rng)
                    eval_obs.append(o); eval_states.append(s)
                bayes = bayes_diag(hmm, eval_obs, eval_states)
                g = gdc_diag(hmm, train_obs, h_train, eval_obs,
                             eval_states, d_best, beta_best)
                denom = bayes - stat_self
                lift = ((g - stat_self) / denom) if abs(denom) > 1e-3 else float('nan')
                rows.append({'topology': name, 'n_train': n_train,
                             'seed': seed, 'nS': nS, 'nA': nA,
                             'd': d_best, 'beta': beta_best,
                             'gdc_diag': g, 'bayes': bayes,
                             'stationary': stat_self, 'lift': lift,
                             'gain': g - stat_self})
                print(f'  EXP2 {name:>16s} N={n_train:>4d} seed={seed} '
                      f'lift={lift:.3f} gain={g - stat_self:.3f}  '
                      f'[{time.time()-tic:.1f}s]', flush=True)
    print(f'EXP2 total: {time.time()-t0:.1f}s', flush=True)
    csv_path = os.path.join(_THIS_DIR, 'paper_n_train_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('Wrote', csv_path)
    return rows


# --- Aggregation + plotting ---
def aggregate_best(rows):
    grid = defaultdict(list)
    diag_grid = defaultdict(list)
    for r in rows:
        grid[(r['topology'], r['d'], r['beta'])].append(r['lift'])
        diag_grid[(r['topology'], r['d'], r['beta'])].append(r['gdc_diag'])
    mean_lift = {k: float(np.nanmean(v)) for k, v in grid.items()}
    mean_diag = {k: float(np.mean(v)) for k, v in diag_grid.items()}
    best_per = {}
    for name in TOPOLOGIES:
        any_valid = any(not np.isnan(mean_lift[(name, d, b)])
                        for d in DIFFUSIONS for b in BETAS)
        best_key = (DIFFUSIONS[0], BETAS[0])
        best_val = -np.inf
        for d in DIFFUSIONS:
            for b in BETAS:
                v = mean_lift[(name, d, b)] if any_valid \
                    else mean_diag[(name, d, b)]
                if not np.isnan(v) and v > best_val:
                    best_val = v; best_key = (d, b)
        best_per[name] = best_key
    return mean_lift, mean_diag, best_per


def plot_exp1(mean_lift, mean_diag, best_per, save_dir):
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), squeeze=False)
    for ax, name in zip(axes.flat, TOPOLOGIES):
        any_valid = any(not np.isnan(mean_lift[(name, d, b)])
                        for d in DIFFUSIONS for b in BETAS)
        if any_valid:
            H = np.array([[mean_lift[(name, d, b)] for b in BETAS]
                          for d in DIFFUSIONS])
            label = 'lift'
        else:
            H = np.array([[mean_diag[(name, d, b)] for b in BETAS]
                          for d in DIFFUSIONS])
            label = 'gdc_diag'
        im = ax.imshow(H, aspect='auto', cmap='viridis')
        for i, d in enumerate(DIFFUSIONS):
            for j, b in enumerate(BETAS):
                ax.text(j, i, f'{H[i, j]:.2f}', ha='center', va='center',
                        color='white' if H[i, j] < (H.min() + H.max())/2
                        else 'black', fontsize=9)
        ax.set_xticks(range(len(BETAS)))
        ax.set_xticklabels([f'{b:.2f}' for b in BETAS])
        ax.set_yticks(range(len(DIFFUSIONS)))
        ax.set_yticklabels([f'{d:.1f}' for d in DIFFUSIONS])
        ax.set_xlabel('beta')
        ax.set_ylabel('d')
        d_b = best_per[name]
        ax.set_title(f'{name}\nbest @ d={d_b[0]:.2f}, beta={d_b[1]:.2f}',
                     fontsize=10)
        plt.colorbar(im, ax=ax, label=label, shrink=0.7)
    fig.suptitle('Hidden-state alignment lift over (d, beta), mean of 3 seeds',
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(save_dir, 'fig_paper_topology_heatmaps.png')
    plt.savefig(out, dpi=130); plt.close()
    print('Wrote', out)

    # Best bar chart with seed dispersion
    fig, ax = plt.subplots(figsize=(10, 5))
    seeds_lifts = defaultdict(list)
    seeds_diags = defaultdict(list)
    for name in TOPOLOGIES:
        for seed in SEEDS:
            d_b, b_b = best_per[name]
            for r in rows_exp1:
                if (r['topology'] == name and r['seed'] == seed
                        and r['d'] == d_b and r['beta'] == b_b):
                    seeds_lifts[name].append(r['lift'])
                    seeds_diags[name].append(r['gdc_diag'])
                    break
    means = []; sems = []; labels = []; meta = []
    for name in TOPOLOGIES:
        v = np.array(seeds_lifts[name], dtype=float)
        v = v[~np.isnan(v)]
        if len(v) > 0:
            means.append(v.mean()); sems.append(v.std() / max(np.sqrt(len(v)), 1))
        else:
            means.append(0.0); sems.append(0.0)
        d_b, b_b = best_per[name]
        labels.append(name)
        meta.append(f'd={d_b:.2f}\nb={b_b:.2f}')
    means_clipped = [min(m, 1.2) for m in means]
    bars = ax.bar(labels, means_clipped, yerr=sems, color='steelblue',
                  edgecolor='black', capsize=4)
    for i, (m, mt) in enumerate(zip(means, meta)):
        text = f'{mt}\n({m:.2f})' if m > 1.2 else mt
        ax.text(i, min(m, 1.2) + 0.02, text, ha='center', va='bottom',
                fontsize=8)
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.6,
               label='Bayes-optimal (1.0)')
    ax.set_ylabel('best alignment lift (mean of 3 seeds)')
    ax.set_title('Best GDC param config per HMM topology')
    ax.set_ylim(0, 1.25)
    plt.xticks(rotation=20, ha='right')
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    out = os.path.join(save_dir, 'fig_paper_topology_best.png')
    plt.savefig(out, dpi=130); plt.close()
    print('Wrote', out)


def plot_exp2(rows, save_dir):
    by_topo = defaultdict(lambda: defaultdict(list))
    by_topo_gain = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_topo[r['topology']][r['n_train']].append(r['lift'])
        by_topo_gain[r['topology']][r['n_train']].append(r['gain'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(N_TRAIN_TOPOLOGIES)))
    for name, c in zip(N_TRAIN_TOPOLOGIES, cmap):
        ns = sorted(by_topo[name].keys())
        means = [np.nanmean(by_topo[name][n]) for n in ns]
        sems = [np.nanstd(by_topo[name][n]) / max(np.sqrt(len(by_topo[name][n])), 1)
                for n in ns]
        axes[0].errorbar(ns, means, yerr=sems, fmt='o-', color=c,
                         capsize=4, label=name)
        means_gain = [np.mean(by_topo_gain[name][n]) for n in ns]
        sems_gain = [np.std(by_topo_gain[name][n])
                     / max(np.sqrt(len(by_topo_gain[name][n])), 1)
                     for n in ns]
        axes[1].errorbar(ns, means_gain, yerr=sems_gain, fmt='s-',
                         color=c, capsize=4, label=name)
    axes[0].axhline(1.0, color='green', linestyle='--', alpha=0.5,
                    label='Bayes-optimal')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('# training sequences')
    axes[0].set_ylabel('alignment lift (mean of 3 seeds)')
    axes[0].set_title('Sample efficiency: lift vs N_train')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('# training sequences')
    axes[1].set_ylabel('absolute gain over stationary')
    axes[1].set_title('Sample efficiency: gain vs N_train (absolute)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(save_dir, 'fig_paper_n_train.png')
    plt.savefig(out, dpi=130); plt.close()
    print('Wrote', out)


# Need to expose rows_exp1 to the plotting helper.
rows_exp1 = []


def main():
    global rows_exp1
    print('=== EXP1: topology x (d, beta) ===', flush=True)
    rows_exp1 = run_exp1()
    mean_lift, mean_diag, best_per = aggregate_best(rows_exp1)

    # Save best per topology
    best_csv = os.path.join(_THIS_DIR, 'paper_topology_best.csv')
    with open(best_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['topology', 'best_d', 'best_beta', 'mean_lift_3_seeds'])
        for name in TOPOLOGIES:
            d_b, b_b = best_per[name]
            v = mean_lift.get((name, d_b, b_b))
            w.writerow([name, d_b, b_b, v])
    print('Wrote', best_csv)

    plot_exp1(mean_lift, mean_diag, best_per, _THIS_DIR)

    print('=== EXP2: sample efficiency ===', flush=True)
    rows_exp2 = run_exp2(best_per)
    plot_exp2(rows_exp2, _THIS_DIR)
    print('Done.')


if __name__ == '__main__':
    main()
