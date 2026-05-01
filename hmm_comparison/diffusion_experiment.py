"""
Single-HMM dimensionality vs GDC diffusion rate.

Setup:
    Fix one small HMM (nS=4, nA=3). Train GDC at several diffusion rates,
    where diffusion d = 1 - alpha - theta is the residual probability that
    spreads uniformly over all GDC states each step.

    For each d, run GDC forward_pass on a held-out eval set, stack the
    posterior history into M, run SVD.

    Hypothesis:
        - d = 0: posterior near one-hot on training-prefix state ->
          effective rank governed by training-prefix identity (~thousands).
        - d -> 1: every transition step replaces the posterior with the
          uniform distribution -> posterior at time t depends only on the
          last observation -> effective rank ~ nA.

Outputs:
    fig_hmm_diagram.png         drawing of the chosen HMM
    fig_diffusion_scree.png     log spectra (sigma_i / sigma_0) vs d
    fig_diffusion_effrank.png   effective rank / participation ratio vs d
    diffusion_results.csv       one row per d
"""

from __future__ import annotations
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from random_hmm import random_dense_hmm
from model_wrappers import fit_gdc

NS = 4
NA = 3
HMM_SEED = 7
N_TRAIN_SEQ = 200
TRAIN_LEN = 40
N_EVAL_SEQ = 80
EVAL_LEN = 40
DIFFUSIONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
EXP_SEED = 0

# alpha + theta = 1 - d.  Keep alpha:theta = 7:3 like the rest of the experiments.
def split(d):
    base = 1.0 - d
    return 0.7 * base, 0.3 * base


# ---------- HMM diagram ----------
def draw_hmm(hmm, path):
    """Drawing layout:
        - 4 nodes on a circle.
        - One transition edge per ordered pair (i,j), i!=j, drawn as a Bezier
          curve. We draw forward (i->j) and reverse (j->i) on opposite sides
          of the chord so labels don't collide.
        - Self-loops as small arcs hanging outside the node.
        - Probabilities below 0.05 are hidden to reduce clutter.
        - Each state shows a small bar chart of its emission distribution.
    """
    nS, nA = hmm.nS, hmm.nA
    MIN_P = 0.05

    fig, ax = plt.subplots(figsize=(8.5, 8))
    angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, nS, endpoint=False)
    R = 2.6
    pos = np.column_stack([R*np.cos(angles), R*np.sin(angles)])
    R_node = 0.55

    sym_colors = plt.cm.Set2(np.linspace(0, 1, nA))

    def edge(start, end, p, rad):
        """Draw a curved arrow with a label at ~70% along the curve."""
        col = (0.20, 0.40, 0.70, min(0.35 + p, 0.95))
        arr = FancyArrowPatch(start, end,
                              connectionstyle=f'arc3,rad={rad}',
                              arrowstyle='-|>',
                              lw=0.6 + 5.5 * p,
                              color=col,
                              shrinkA=R_node*30, shrinkB=R_node*30,
                              mutation_scale=12 + 8 * p)
        ax.add_patch(arr)
        # Label position: pull off the chord by an amount proportional to rad.
        mid = 0.5 * (start + end)
        # perpendicular direction (rotate end-start by 90)
        v = end - start
        perp = np.array([-v[1], v[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-9)
        label_pos = mid + perp * (0.55 * abs(rad) * np.linalg.norm(v))
        ax.text(label_pos[0], label_pos[1], f'{p:.2f}',
                fontsize=9, color=(0.10, 0.30, 0.60),
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', fc='white',
                          ec='none', alpha=0.9))

    # Off-diagonal transitions
    for i in range(nS):
        for j in range(nS):
            if i == j:
                continue
            p = hmm.T[i, j]
            if p < MIN_P:
                continue
            # Curve direction: forward edges curve one way, reverse the other.
            rad = 0.22 if (j - i) % nS < nS / 2 else -0.22
            edge(pos[i], pos[j], p, rad)

    # Self-loops: draw outside the node along the radial direction
    for i in range(nS):
        p = hmm.T[i, i]
        if p < MIN_P:
            continue
        cx, cy = pos[i]
        # offset center of loop outwards
        out = np.array([np.cos(angles[i]), np.sin(angles[i])])
        loop_center = np.array([cx, cy]) + out * (R_node + 0.27)
        loop = plt.Circle(loop_center, 0.27, fill=False,
                          lw=0.6 + 5.5 * p,
                          color=(0.20, 0.40, 0.70, min(0.35 + p, 0.95)))
        ax.add_patch(loop)
        # arrow tip on inner side of loop
        tip = loop_center - out * 0.27
        head = FancyArrowPatch(tip + np.array([-out[1], out[0]]) * 0.05,
                               tip,
                               arrowstyle='-|>',
                               color=(0.20, 0.40, 0.70, 0.95), lw=0.5,
                               mutation_scale=14)
        ax.add_patch(head)
        label_pos = loop_center + out * 0.42
        ax.text(label_pos[0], label_pos[1], f'{p:.2f}',
                fontsize=9, color=(0.10, 0.30, 0.60),
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', fc='white',
                          ec='none', alpha=0.95))

    # Nodes
    for i in range(nS):
        x, y = pos[i]
        circ = Circle((x, y), R_node, facecolor='white',
                      edgecolor='black', lw=2.0, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y + 0.11, f's{i}', ha='center', va='center',
                fontsize=15, fontweight='bold', zorder=4)
        # Emission bar chart inside the node, below the label
        bar_w_total = 0.7
        bar_h_max = 0.20
        baseline_y = y - 0.18
        for a in range(nA):
            ex = x - bar_w_total/2 + (a + 0.5) * bar_w_total / nA
            h = hmm.E[i, a] * bar_h_max
            ax.add_patch(plt.Rectangle((ex - bar_w_total/(2*nA), baseline_y),
                                       bar_w_total/nA - 0.01, h,
                                       color=sym_colors[a],
                                       ec='black', lw=0.4, zorder=4))

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=sym_colors[a],
                             ec='black', lw=0.4) for a in range(nA)]
    ax.legend(handles, [f'symbol {a}' for a in range(nA)],
              loc='lower left', fontsize=9, frameon=True,
              title='emissions')

    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(
        f'Test HMM (nS={nS}, nA={nA})\n'
        f'edge width ∝ P(transition); bars in each node = emission distribution',
        fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print('Wrote', path)


def collect_M(gdc, eval_obs):
    blocks = []
    for o in eval_obs:
        oc = np.asarray(o, np.int64).reshape(-1, 1)
        _, hist = gdc.gdc.forward_pass(oc, return_history=True)
        blocks.append(hist)
    return np.vstack(blocks)


def run_one(hmm, d, train_seqs, eval_obs, eval_states):
    a, t = split(d)
    gdc = fit_gdc(train_seqs, alphabet_size=hmm.nA,
                  alpha=a, theta=t, gamma=0.0, beta=0.1,
                  transition_type='self_loop', initial_dist='sequence_starts')
    M = collect_M(gdc, eval_obs)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    s0 = S[0] if S[0] > 0 else 1.0
    eff_rank_1e3 = int(np.sum(S / s0 > 1e-3))
    eff_rank_1e6 = int(np.sum(S / s0 > 1e-6))
    pr = float(S.sum() ** 2 / np.sum(S ** 2))
    return {
        'd': d, 'alpha': a, 'theta': t, 'n_gdc_states': gdc.gdc.n_states,
        'eff_rank_1e3': eff_rank_1e3,
        'eff_rank_1e6': eff_rank_1e6,
        'participation_ratio': pr,
        'top20_norm': (S[:20] / s0).tolist(),
    }


def main():
    rng_hmm = np.random.default_rng(HMM_SEED)
    hmm = random_dense_hmm(NS, NA, rng_hmm)
    draw_hmm(hmm, os.path.join(_THIS_DIR, 'fig_hmm_diagram.png'))

    rng = np.random.default_rng(EXP_SEED)
    train_seqs = hmm.sample_many(N_TRAIN_SEQ, TRAIN_LEN, rng)
    eval_obs, eval_states = [], []
    for _ in range(N_EVAL_SEQ):
        s, o = hmm.sample(EVAL_LEN, rng)
        eval_obs.append(o); eval_states.append(s)

    rows = []
    for d in DIFFUSIONS:
        r = run_one(hmm, d, train_seqs, eval_obs, eval_states)
        print(f'  d={d:5.2f}  n_gdc={r["n_gdc_states"]}  '
              f'eff_rank(1e-3)={r["eff_rank_1e3"]}  '
              f'eff_rank(1e-6)={r["eff_rank_1e6"]}  '
              f'PR={r["participation_ratio"]:.2f}',
              flush=True)
        rows.append(r)

    # CSV
    csv_path = os.path.join(_THIS_DIR, 'diffusion_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['d', 'alpha', 'theta',
                                          'n_gdc_states',
                                          'eff_rank_1e3', 'eff_rank_1e6',
                                          'participation_ratio'])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in
                        ['d', 'alpha', 'theta', 'n_gdc_states',
                         'eff_rank_1e3', 'eff_rank_1e6',
                         'participation_ratio']})
    print('Wrote', csv_path)

    # Scree plot
    plt.figure(figsize=(7, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(DIFFUSIONS)))
    for r, c in zip(rows, cmap):
        sv = r['top20_norm']
        plt.plot(range(1, len(sv) + 1), sv, 'o-', color=c,
                 label=f'd={r["d"]:.2f}')
    plt.yscale('log')
    plt.axhline(1e-3, color='k', linestyle=':', alpha=0.4)
    plt.axvline(NA, color='r', linestyle='--', alpha=0.5,
                label=f'k = nA = {NA}')
    plt.xlabel('Singular-value index')
    plt.ylabel('sigma_i / sigma_0')
    plt.title(f'GDC posterior spectrum vs diffusion rate (nS={NS}, nA={NA})')
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_diffusion_scree.png')
    plt.savefig(out, dpi=120); plt.close(); print('Wrote', out)

    # Effective rank vs d
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ds = [r['d'] for r in rows]
    axes[0].semilogy(ds, [r['eff_rank_1e3'] for r in rows], 'o-',
                     label=r'eff rank ($\sigma/\sigma_0>10^{-3}$)')
    axes[0].semilogy(ds, [r['eff_rank_1e6'] for r in rows], 's-',
                     label=r'eff rank ($\sigma/\sigma_0>10^{-6}$)')
    axes[0].axhline(NA, color='r', linestyle='--', alpha=0.6,
                    label=f'nA = {NA}')
    axes[0].axhline(NS, color='g', linestyle=':', alpha=0.6,
                    label=f'nS = {NS}')
    axes[0].set_xlabel('diffusion rate d = 1 - alpha - theta')
    axes[0].set_ylabel('effective rank (log)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Threshold-based effective rank')

    axes[1].plot(ds, [r['participation_ratio'] for r in rows], 'D-',
                 color='purple')
    axes[1].axhline(NA, color='r', linestyle='--', alpha=0.6,
                    label=f'nA = {NA}')
    axes[1].axhline(NS, color='g', linestyle=':', alpha=0.6,
                    label=f'nS = {NS}')
    axes[1].set_xlabel('diffusion rate d')
    axes[1].set_ylabel('participation ratio')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Participation ratio')
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_diffusion_effrank.png')
    plt.savefig(out, dpi=120); plt.close(); print('Wrote', out)
    print('Done.')


if __name__ == '__main__':
    main()
