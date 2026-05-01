"""
Dimensionality of the GDC posterior state space.

Claim under test: during inference, GDC's posterior over its many surface
states lives on a low-dimensional manifold whose intrinsic dimensionality
reflects the underlying HMM's state count (or dynamical rank).

Procedure (per HMM):
    1. Fit GDC on training sequences from the HMM.
    2. On fresh evaluation sequences, run GDC forward_pass with
       return_history=True, producing a (T, n_gdc_states) matrix of
       posterior state distributions at each timepoint. Also keep the
       HMM's true hidden state at each timepoint.
    3. Row-stack across sequences -> M (N x n_gdc_states).
    4. Truncated SVD on M. Report:
         - top-k singular values (scree)
         - effective rank (count of sigma_i / sigma_0 > 1e-3)
         - participation ratio (sum sigma)^2 / sum sigma^2
    5. Project rows onto top-2 right-singular directions; scatter colored
       by true hidden state.
    6. As a ground-truth reference, run the same SVD on the HMM's exact
       posteriors alpha_t -- effective rank must be <= nS.

Sweeps:
    * Vary true HMM state count nS in {2,3,4,5,6,8} at nA=5.
    * Vary transition rank r in {2,4,6,8} at nS=8 (fixed) to test whether
      SVD recovers the dynamical rank rather than nS.

Outputs:
    dim_results.csv           one row per (setting, seed)
    fig_dim_scree.png         scree per nS (log sigma)
    fig_dim_scree_rank.png    scree per low-rank setting
    fig_dim_projection.png    2D top-2 projection for 3 example nS values,
                              colored by true hidden state.
"""

from __future__ import annotations
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from random_hmm import random_dense_hmm, random_lowrank_hmm
from model_wrappers import fit_gdc


# -------- config --------
N_TRAIN_SEQ   = 300
TRAIN_LEN     = 40
N_EVAL_SEQ    = 100
EVAL_LEN      = 40
NA            = 5
NS_VALUES     = [2, 3, 4, 5, 6, 8]
RANK_VALUES   = [2, 4, 6, 8]   # at fixed nS=8
SEEDS         = [0, 1, 2]
GDC_KWARGS    = dict(alpha=0.7, theta=0.2, gamma=0.0, beta=0.1,
                     transition_type='self_loop',
                     initial_dist='sequence_starts')


# -------- helpers --------
def collect_gdc_posterior_matrix(gdc, eval_seq_obs):
    """Stack per-timestep GDC posteriors across a list of sequences.

    Returns M (sum_T, n_gdc_states) in row order of sequences/time.
    """
    blocks = []
    for obs in eval_seq_obs:
        obs_col = np.asarray(obs, dtype=np.int64).reshape(-1, 1)
        _, history = gdc.gdc.forward_pass(obs_col, return_history=True)
        blocks.append(history)
    return np.vstack(blocks)


def collect_hmm_posterior_matrix(hmm, eval_seq_obs):
    """Stack per-timestep HMM alpha_t across sequences."""
    blocks = []
    for obs in eval_seq_obs:
        obs = np.asarray(obs, dtype=np.int64)
        # Incremental filter to record history
        a = hmm.pi * hmm.E[:, obs[0]]
        s = a.sum()
        a = a / s if s > 0 else np.full(hmm.nS, 1.0 / hmm.nS)
        hist = [a.copy()]
        for o in obs[1:]:
            a = (a @ hmm.T) * hmm.E[:, o]
            s = a.sum()
            a = a / s if s > 0 else np.full(hmm.nS, 1.0 / hmm.nS)
            hist.append(a.copy())
        blocks.append(np.vstack(hist))
    return np.vstack(blocks)


def svd_summary(M, k_top=20):
    """Return dict of scree stats."""
    # Economy SVD is enough because rows >> cols typically.
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    s0 = S[0] if S[0] > 0 else 1.0
    eff_rank = int(np.sum(S / s0 > 1e-3))
    part_ratio = float(S.sum() ** 2 / np.sum(S ** 2))
    return {
        'S': S,
        'U': U,
        'Vt': Vt,
        'eff_rank_1e3': eff_rank,
        'participation_ratio': part_ratio,
        'top_k': S[:k_top].tolist(),
    }


def run_hmm(hmm, seed, label):
    """Train GDC, collect posterior matrices and labels."""
    rng = np.random.default_rng(seed)
    train = hmm.sample_many(N_TRAIN_SEQ, TRAIN_LEN, rng)
    # Evaluation seqs: keep states.
    eval_obs, eval_states = [], []
    for _ in range(N_EVAL_SEQ):
        s, o = hmm.sample(EVAL_LEN, rng)
        eval_obs.append(o)
        eval_states.append(s)
    gdc = fit_gdc(train, alphabet_size=hmm.nA, **GDC_KWARGS)
    M_gdc = collect_gdc_posterior_matrix(gdc, eval_obs)
    M_hmm = collect_hmm_posterior_matrix(hmm, eval_obs)
    labels = np.concatenate(eval_states)
    return M_gdc, M_hmm, labels, gdc.gdc.n_states


def sweep_ns():
    rows, cached = [], {}
    for nS in NS_VALUES:
        for seed in SEEDS:
            rng = np.random.default_rng(11 * seed + 7 * nS)
            hmm = random_dense_hmm(nS, NA, rng)
            M_gdc, M_hmm, labels, n_gdc = run_hmm(hmm, seed + 10 * nS,
                                                   f'nS={nS}')
            s_gdc = svd_summary(M_gdc)
            s_hmm = svd_summary(M_hmm)
            rows.append({
                'experiment': 'ns_sweep',
                'nS': nS, 'rank': nS, 'seed': seed,
                'n_gdc_states': n_gdc,
                'gdc_eff_rank_1e3': s_gdc['eff_rank_1e3'],
                'gdc_participation_ratio': s_gdc['participation_ratio'],
                'hmm_eff_rank_1e3': s_hmm['eff_rank_1e3'],
                'hmm_participation_ratio': s_hmm['participation_ratio'],
                'gdc_top10': ','.join(f'{v:.4f}' for v in s_gdc['top_k'][:10]),
                'hmm_top10': ','.join(f'{v:.4f}' for v in s_hmm['top_k'][:10]),
            })
            if seed == 0:
                cached[('ns', nS)] = (M_gdc, labels, s_gdc)
            print(f'  [ns_sweep] nS={nS} seed={seed} '
                  f'n_gdc={n_gdc} '
                  f'gdc_eff_rank={s_gdc["eff_rank_1e3"]} '
                  f'hmm_eff_rank={s_hmm["eff_rank_1e3"]}',
                  flush=True)
    return rows, cached


def sweep_rank():
    rows, cached = [], {}
    nS = 8
    for r in RANK_VALUES:
        for seed in SEEDS:
            rng = np.random.default_rng(31 * seed + 101 * r)
            hmm = random_lowrank_hmm(nS, NA, r, rng)
            M_gdc, M_hmm, labels, n_gdc = run_hmm(hmm, seed + 500 + 10 * r,
                                                   f'rank={r}')
            s_gdc = svd_summary(M_gdc)
            s_hmm = svd_summary(M_hmm)
            rows.append({
                'experiment': 'rank_sweep',
                'nS': nS, 'rank': r, 'seed': seed,
                'n_gdc_states': n_gdc,
                'gdc_eff_rank_1e3': s_gdc['eff_rank_1e3'],
                'gdc_participation_ratio': s_gdc['participation_ratio'],
                'hmm_eff_rank_1e3': s_hmm['eff_rank_1e3'],
                'hmm_participation_ratio': s_hmm['participation_ratio'],
                'gdc_top10': ','.join(f'{v:.4f}' for v in s_gdc['top_k'][:10]),
                'hmm_top10': ','.join(f'{v:.4f}' for v in s_hmm['top_k'][:10]),
            })
            if seed == 0:
                cached[('rank', r)] = (M_gdc, labels, s_gdc)
            print(f'  [rank_sweep] rank={r} seed={seed} '
                  f'n_gdc={n_gdc} '
                  f'gdc_eff_rank={s_gdc["eff_rank_1e3"]} '
                  f'hmm_eff_rank={s_hmm["eff_rank_1e3"]}',
                  flush=True)
    return rows, cached


# -------- plotting --------
def plot_scree(cached_ns, cached_rank):
    plt.figure(figsize=(7, 4))
    for nS in NS_VALUES:
        M, labels, s = cached_ns[('ns', nS)]
        sv = s['S']
        sv = sv / sv[0]
        plt.plot(range(1, min(20, len(sv)) + 1),
                 sv[:20], 'o-', label=f'nS={nS}')
    plt.yscale('log')
    plt.xlabel('Singular-value index')
    plt.ylabel('sigma_i / sigma_0')
    plt.title('GDC posterior-history spectrum vs true nS (nA=5)')
    plt.axhline(1e-3, color='k', linestyle=':', alpha=0.4,
                label='1e-3 threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_dim_scree.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('Wrote', out)

    plt.figure(figsize=(7, 4))
    for r in RANK_VALUES:
        M, labels, s = cached_rank[('rank', r)]
        sv = s['S']
        sv = sv / sv[0]
        plt.plot(range(1, min(20, len(sv)) + 1),
                 sv[:20], 'o-', label=f'rank={r}')
    plt.yscale('log')
    plt.xlabel('Singular-value index')
    plt.ylabel('sigma_i / sigma_0')
    plt.title('GDC posterior-history spectrum vs HMM transition rank '
             '(nS=8, nA=5)')
    plt.axhline(1e-3, color='k', linestyle=':', alpha=0.4,
                label='1e-3 threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_dim_scree_rank.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('Wrote', out)


def plot_projection(cached_ns, ns_to_show=(3, 5, 8)):
    fig, axes = plt.subplots(1, len(ns_to_show), figsize=(4.5 * len(ns_to_show), 4))
    if len(ns_to_show) == 1:
        axes = [axes]
    for ax, nS in zip(axes, ns_to_show):
        M, labels, s = cached_ns[('ns', nS)]
        # Scores on top-2 right-singular directions: U[:, :2] * S[:2]
        scores = s['U'][:, :2] * s['S'][:2]
        # Subsample to keep plot readable.
        rng = np.random.default_rng(0)
        idx = rng.choice(scores.shape[0],
                         size=min(2000, scores.shape[0]),
                         replace=False)
        sc = ax.scatter(scores[idx, 0], scores[idx, 1],
                        c=labels[idx], cmap='tab10', s=6, alpha=0.7)
        ax.set_title(f'nS={nS}  (eff rank {s["eff_rank_1e3"]})')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='true HMM state')
    plt.suptitle('GDC posterior rows in top-2 singular directions, '
                 'colored by HMM hidden state')
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_dim_projection.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('Wrote', out)


def plot_projection_rank(cached_rank, ranks_to_show=(2, 4, 8)):
    fig, axes = plt.subplots(1, len(ranks_to_show),
                              figsize=(4.5 * len(ranks_to_show), 4))
    if len(ranks_to_show) == 1:
        axes = [axes]
    for ax, r in zip(axes, ranks_to_show):
        M, labels, s = cached_rank[('rank', r)]
        scores = s['U'][:, :2] * s['S'][:2]
        rng = np.random.default_rng(0)
        idx = rng.choice(scores.shape[0],
                         size=min(2000, scores.shape[0]),
                         replace=False)
        sc = ax.scatter(scores[idx, 0], scores[idx, 1],
                        c=labels[idx], cmap='tab10', s=6, alpha=0.7)
        ax.set_title(f'rank={r} (nS=8)  (eff rank {s["eff_rank_1e3"]})')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='true HMM state')
    plt.suptitle('Low-rank HMMs: GDC posterior top-2 projection')
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_dim_projection_rank.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('Wrote', out)


def write_csv(rows):
    path = os.path.join(_THIS_DIR, 'dim_results.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {len(rows)} rows to {path}')


if __name__ == '__main__':
    print('--- nS sweep ---', flush=True)
    rows_ns, cached_ns = sweep_ns()
    print('--- rank sweep ---', flush=True)
    rows_rank, cached_rank = sweep_rank()
    write_csv(rows_ns + rows_rank)
    plot_scree(cached_ns, cached_rank)
    plot_projection(cached_ns, ns_to_show=(3, 5, 8))
    plot_projection_rank(cached_rank, ranks_to_show=(2, 4, 8))
    print('Done.')
