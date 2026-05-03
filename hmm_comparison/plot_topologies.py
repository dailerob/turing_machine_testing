"""Visualise the four HMM topologies used in the size + determinism sweep.

For each of {dense_small, dense_large, det_small, det_large} draw a
representative HMM at a fixed seed:
  - transition matrix T as a heatmap
  - emission matrix E as a heatmap
  - stationary distribution + emission marginal as bar charts

Output: fig_hmm_topologies.png
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from random_hmm import random_dense_hmm, random_sparse_topology_hmm  # noqa: E402

# (name, nS, nA, kind, E_conc, fanout)
# kind in {'dense', 'sparse'}
REGIMES = [
    ('dense_small',  10, 4, 'dense',  1.0, None),
    ('dense_large',  30, 8, 'dense',  1.0, None),
    ('det_small',    10, 4, 'dense',  0.1, None),
    ('det_large',    30, 8, 'dense',  0.1, None),
    ('sparse_small', 10, 4, 'sparse', 0.1, 2),
    ('sparse_large', 30, 8, 'sparse', 0.1, 2),
]


def stationary(T, n_iter=500):
    n = T.shape[0]
    pi = np.full(n, 1.0 / n)
    for _ in range(n_iter):
        nxt = pi @ T
        if np.linalg.norm(nxt - pi) < 1e-12:
            pi = nxt; break
        pi = nxt
    pi = np.maximum(pi, 0)
    s = pi.sum()
    return pi / s if s > 0 else np.full(n, 1.0 / n)


def main():
    fig, axes = plt.subplots(len(REGIMES), 4,
                              figsize=(15, 3 * len(REGIMES)))

    for row, (name, nS, nA, kind, E_conc, fanout) in enumerate(REGIMES):
        seed_offset = (1 if 'det' in name else 0) + (2 if 'sparse' in name else 0)
        rng = np.random.default_rng(20000 + nS * 7 + nA * 11 + seed_offset)
        if kind == 'sparse':
            hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                             E_concentration=E_conc)
        else:
            hmm = random_dense_hmm(nS, nA, rng,
                                   T_concentration=1.0,
                                   E_concentration=E_conc)
        T = hmm.T; E = hmm.E
        pi_stat = stationary(T)
        marginal = pi_stat @ E

        # Col 0: T heatmap
        ax = axes[row, 0]
        im = ax.imshow(T, aspect='auto', cmap='viridis',
                       vmin=0, vmax=T.max())
        ax.set_title(f'{name}: T (nS={nS})', fontsize=10)
        ax.set_xlabel('next state j'); ax.set_ylabel('state i')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Col 1: E heatmap
        ax = axes[row, 1]
        im = ax.imshow(E, aspect='auto', cmap='viridis',
                       vmin=0, vmax=1.0)
        ax.set_title(f'E (nA={nA}, E_conc={E_conc})', fontsize=10)
        ax.set_xlabel('emission a'); ax.set_ylabel('state i')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Col 2: stationary distribution + emission marginal
        ax = axes[row, 2]
        ax.bar(range(nS), pi_stat, color='steelblue', edgecolor='black',
               linewidth=0.5)
        ax.set_title('stationary π over states', fontsize=10)
        ax.set_xlabel('state'); ax.set_ylabel('P')
        ax.set_ylim(0, max(pi_stat.max() * 1.2, 0.1))

        # Col 3: emission marginal (= π @ E)
        ax = axes[row, 3]
        ax.bar(range(nA), marginal, color='salmon', edgecolor='black',
               linewidth=0.5)
        ax.set_title('marginal emission distribution', fontsize=10)
        ax.set_xlabel('emission a'); ax.set_ylabel('P')
        ax.set_ylim(0, max(marginal.max() * 1.2, 0.1))

    fig.suptitle('HMM topologies used in the sample-efficiency sweep',
                 fontsize=13, y=1.0)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig_hmm_topologies.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
