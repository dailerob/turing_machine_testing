"""
Aggregate GDC states by their L-symbol emission context, then SVD.

Setup mirrors diffusion_experiment.py (same HMM): nS=4, nA=3, seed 7.
GDC trained at default low-diffusion settings (alpha=0.7, theta=0.2, d=0.1).

For each L in {1, 2, 3, 4}:
    Build A_L (n_gdc, nA**L). Column groups index the L-tuple of the most
    recent symbols at the GDC state's training-prefix position. States too
    early in their sequence (position < L-1) get a 'short prefix' bucket.

    M_L = M @ A_L  has shape (N, nA**L). SVD it.

Predict: L=1 caps at rank nA = 3 (no way to break HMM-state ties below
emission resolution). L >= 2 should expose a knee at nS = 4 because a
two-symbol context is generally enough to disambiguate the hidden states.

Outputs:
    fig_aggregate_scree.png    spectra for each L plus HMM ground truth
    fig_aggregate_effrank.png  eff_rank / PR vs L
    aggregate_results.csv
"""

from __future__ import annotations
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
L_VALUES = [1, 2, 3, 4]
EXP_SEED = 0

# Default GDC params (low diffusion -- the bloated regime).
GDC_KWARGS = dict(alpha=0.7, theta=0.2, gamma=0.0, beta=0.1,
                  transition_type='self_loop',
                  initial_dist='sequence_starts')


def collect_M(gdc, eval_obs):
    blocks = []
    for o in eval_obs:
        oc = np.asarray(o, np.int64).reshape(-1, 1)
        _, hist = gdc.gdc.forward_pass(oc, return_history=True)
        blocks.append(hist)
    return np.vstack(blocks)


def collect_M_hmm(hmm, eval_obs):
    blocks = []
    for o in eval_obs:
        a = hmm.pi * hmm.E[:, o[0]]; a = a / max(a.sum(), 1e-12)
        h = [a.copy()]
        for ob in o[1:]:
            a = (a @ hmm.T) * hmm.E[:, ob]; a = a / max(a.sum(), 1e-12)
            h.append(a.copy())
        blocks.append(np.vstack(h))
    return np.vstack(blocks)


def build_context_groups(symbols_per_state, train_len, L, nA):
    """For each GDC state index j, return its group id = encoding of the
    last-L symbols (s_{j-L+1}, ..., s_j) in its training prefix.

    For positions where j%train_len < L-1 (not enough history), reserve a
    single sentinel bucket id = nA**L.
    """
    n = len(symbols_per_state)
    group_ids = np.full(n, nA**L, dtype=np.int64)   # default sentinel
    for j in range(n):
        pos_in_seq = j % train_len
        if pos_in_seq < L - 1:
            continue                          # short-prefix sentinel
        # encode (s_{j-L+1},...,s_j) in base nA
        code = 0
        for k in range(L):
            sym = int(symbols_per_state[j - (L - 1 - k)])
            code = code * nA + sym
        group_ids[j] = code
    return group_ids


def aggregate(M, group_ids, n_groups):
    """Sum columns of M by group id. Output shape (N, n_groups)."""
    N = M.shape[0]
    out = np.zeros((N, n_groups))
    for g in range(n_groups):
        cols = np.where(group_ids == g)[0]
        if len(cols):
            out[:, g] = M[:, cols].sum(axis=1)
    return out


def svd_summary(M):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    s0 = S[0] if S[0] > 0 else 1.0
    return {
        'S': S,
        'top_norm': (S / s0).tolist(),
        'eff_rank_1e3': int(np.sum(S / s0 > 1e-3)),
        'eff_rank_1e6': int(np.sum(S / s0 > 1e-6)),
        'pr': float(S.sum() ** 2 / np.sum(S ** 2)),
    }


def main():
    rng_hmm = np.random.default_rng(HMM_SEED)
    hmm = random_dense_hmm(NS, NA, rng_hmm)

    rng = np.random.default_rng(EXP_SEED)
    train_seqs = hmm.sample_many(N_TRAIN_SEQ, TRAIN_LEN, rng)
    eval_obs = []
    for _ in range(N_EVAL_SEQ):
        _, o = hmm.sample(EVAL_LEN, rng)
        eval_obs.append(o)

    gdc = fit_gdc(train_seqs, alphabet_size=NA, **GDC_KWARGS)
    M_full = collect_M(gdc, eval_obs)
    M_hmm = collect_M_hmm(hmm, eval_obs)
    print(f'GDC posterior matrix M_full: {M_full.shape}')
    print(f'HMM alpha matrix M_hmm: {M_hmm.shape}')

    # Reference SVDs
    s_full = svd_summary(M_full)
    s_hmm = svd_summary(M_hmm)
    print(f'  full GDC: eff_rank(1e-3)={s_full["eff_rank_1e3"]} '
          f'PR={s_full["pr"]:.2f}')
    print(f'  HMM alpha:    eff_rank(1e-3)={s_hmm["eff_rank_1e3"]} '
          f'PR={s_hmm["pr"]:.2f}')

    # Aggregations
    syms = gdc.gdc.states[:, 0].astype(np.int64)
    rows = []
    summaries = {}

    # Save SVD U*S of M_full once, for the unaggregated reference.
    Uf, Sf, _ = np.linalg.svd(M_full, full_matrices=False)
    Z_full = (Uf * Sf)[:, :20]
    # R^2 reconstruction of HMM alpha from top-k SVD scores of each M_L.
    # If the HMM-state info lives in a nS-dim subspace of M_L, R^2 should
    # saturate by k = nS = 4.
    def r2_curve(scores, target, k_max):
        """Stagewise R^2 of OLS regression target ~ scores[:, :k] for k=1..k_max."""
        N = scores.shape[0]
        # centered target
        T0 = target - target.mean(axis=0, keepdims=True)
        ss_tot = np.sum(T0 ** 2)
        out = []
        for k in range(1, min(k_max, scores.shape[1]) + 1):
            X = scores[:, :k]
            X = np.hstack([X, np.ones((N, 1))])
            # solve X @ B = target
            B, *_ = np.linalg.lstsq(X, target, rcond=None)
            pred = X @ B
            ss_res = np.sum((target - pred) ** 2)
            out.append(1.0 - ss_res / ss_tot)
        return out

    K_MAX = 12
    r2_results = {}
    for L in L_VALUES:
        n_groups = NA**L + 1   # +1 for short-prefix sentinel
        group_ids = build_context_groups(syms, TRAIN_LEN, L, NA)
        M_L = aggregate(M_full, group_ids, n_groups)
        s = svd_summary(M_L)
        # SVD scores Z = U*S (already inside svd_summary via S, but rebuild)
        U_L, S_L, _ = np.linalg.svd(M_L, full_matrices=False)
        Z_L = U_L * S_L
        r2 = r2_curve(Z_L, M_hmm, K_MAX)
        r2_results[L] = r2
        summaries[L] = s
        print(f'  L={L} n_groups={n_groups} eff_rank(1e-3)={s["eff_rank_1e3"]} '
              f'eff_rank(1e-6)={s["eff_rank_1e6"]} PR={s["pr"]:.2f}  '
              f'R2(k=1)={r2[0]:.3f} R2(k=nS)={r2[min(NS, len(r2))-1]:.3f} '
              f'R2(k=max)={r2[-1]:.3f}')
        rows.append({'L': L, 'n_groups': n_groups,
                     'eff_rank_1e3': s['eff_rank_1e3'],
                     'eff_rank_1e6': s['eff_rank_1e6'],
                     'pr': s['pr']})

    # Also do R^2 for the unaggregated full GDC posterior.
    r2_full = r2_curve(Z_full, M_hmm, K_MAX)
    print(f'  full GDC: R2(k=1)={r2_full[0]:.3f} '
          f'R2(k=nS)={r2_full[min(NS,len(r2_full))-1]:.3f} '
          f'R2(k=max)={r2_full[-1]:.3f}')
    rows.append({'L': 'full', 'n_groups': M_full.shape[1],
                 'eff_rank_1e3': s_full['eff_rank_1e3'],
                 'eff_rank_1e6': s_full['eff_rank_1e6'],
                 'pr': s_full['pr']})
    rows.append({'L': 'hmm', 'n_groups': NS,
                 'eff_rank_1e3': s_hmm['eff_rank_1e3'],
                 'eff_rank_1e6': s_hmm['eff_rank_1e6'],
                 'pr': s_hmm['pr']})

    # CSV
    csv_path = os.path.join(_THIS_DIR, 'aggregate_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['L', 'n_groups',
                                          'eff_rank_1e3', 'eff_rank_1e6', 'pr'])
        w.writeheader(); w.writerows(rows)
    print('Wrote', csv_path)

    # Scree
    plt.figure(figsize=(8, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(L_VALUES)))
    for L, c in zip(L_VALUES, cmap):
        sv = summaries[L]['top_norm'][:20]
        plt.plot(range(1, len(sv) + 1), sv, 'o-', color=c,
                 label=f'L={L}  (n_groups={NA**L+1})')
    sv = s_full['top_norm'][:20]
    plt.plot(range(1, len(sv) + 1), sv, 's:', color='grey',
             label='full GDC posterior (no aggregation)')
    sv = s_hmm['top_norm'][:NS]
    plt.plot(range(1, len(sv) + 1), sv, 'D--', color='red',
             label=f'HMM alpha (ground truth, rank ≤ {NS})')
    plt.axvline(NS, color='g', linestyle=':', alpha=0.6,
                label=f'nS = {NS}')
    plt.axvline(NA, color='purple', linestyle=':', alpha=0.4,
                label=f'nA = {NA}')
    plt.yscale('log')
    plt.xlabel('Singular-value index')
    plt.ylabel('σ_i / σ_0')
    plt.title('Aggregated GDC posterior spectra by emission-context length L')
    plt.legend(fontsize=8, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_aggregate_scree.png')
    plt.savefig(out, dpi=130); plt.close()
    print('Wrote', out)

    # Eff rank / PR vs L
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    Ls = L_VALUES
    axes[0].plot(Ls, [summaries[L]['eff_rank_1e3'] for L in Ls], 'o-',
                 label='eff_rank(σ/σ₀ > 10⁻³)')
    axes[0].plot(Ls, [summaries[L]['eff_rank_1e6'] for L in Ls], 's-',
                 label='eff_rank(σ/σ₀ > 10⁻⁶)')
    axes[0].axhline(NS, color='g', linestyle=':', label=f'nS = {NS}')
    axes[0].axhline(NA, color='purple', linestyle=':', label=f'nA = {NA}')
    axes[0].axhline(s_hmm['eff_rank_1e3'], color='red', linestyle='--',
                    label=f'HMM alpha eff_rank = {s_hmm["eff_rank_1e3"]}')
    axes[0].set_xlabel('emission-context length L')
    axes[0].set_ylabel('effective rank')
    axes[0].set_xticks(Ls)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title('Threshold-based effective rank')

    axes[1].plot(Ls, [summaries[L]['pr'] for L in Ls], 'D-', color='purple')
    axes[1].axhline(NS, color='g', linestyle=':', label=f'nS = {NS}')
    axes[1].axhline(NA, color='red', linestyle=':', label=f'nA = {NA}')
    axes[1].axhline(s_hmm['pr'], color='red', linestyle='--',
                    label=f'HMM alpha PR = {s_hmm["pr"]:.2f}')
    axes[1].set_xlabel('emission-context length L')
    axes[1].set_ylabel('participation ratio')
    axes[1].set_xticks(Ls)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_title('Participation ratio')

    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_aggregate_effrank.png')
    plt.savefig(out, dpi=130); plt.close()
    print('Wrote', out)

    # R^2 plot
    plt.figure(figsize=(7, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(L_VALUES)))
    for L, c in zip(L_VALUES, cmap):
        r2 = r2_results[L]
        ks = np.arange(1, len(r2) + 1)
        plt.plot(ks, r2, 'o-', color=c,
                 label=f'L={L}  (cols={NA**L+1})')
    plt.plot(np.arange(1, len(r2_full) + 1), r2_full, 's:', color='grey',
             label='full GDC (no aggregation)')
    plt.axvline(NS, color='g', linestyle='--', alpha=0.6, label=f'k = nS = {NS}')
    plt.axvline(NA, color='purple', linestyle=':', alpha=0.4, label=f'k = nA = {NA}')
    plt.axhline(1.0, color='r', linestyle=':', alpha=0.5,
                label='R^2 = 1 (perfect HMM α reconstruction)')
    plt.xlabel('k = # top SVD scores of M_L used as regressors')
    plt.ylabel(r'R$^2$ reconstructing HMM $\alpha_t$ from top-k SVD of M_L')
    plt.title('Does the top-k subspace of aggregated M_L recover HMM posterior?')
    plt.ylim(0, 1.05)
    plt.legend(fontsize=8, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_aggregate_r2.png')
    plt.savefig(out, dpi=130); plt.close()
    print('Wrote', out)
    print('Done.')


if __name__ == '__main__':
    main()
