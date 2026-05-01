"""
Follow-up to dimensionality_experiment.py.

The naive "count singular values above a threshold" didn't recover nS -- GDC's
posterior has a long, slowly-decaying tail because each training-prefix is a
distinct surface state. The sharper question is: HOW MANY of the top singular
directions carry signal about the true HMM hidden state?

Procedure (per HMM):
    1. Fit GDC, collect posterior-history matrix M (N x n_gdc_states) and
       true hidden-state labels y (N,).
    2. Compute economy SVD of M (U, S, Vt).
    3. Project rows onto top-k directions: Z_k = U[:, :k] * S[:k].
    4. For each k in 1..20, fit a multinomial logistic-regression classifier
       y ~ Z_k on a 50/50 train/test split; report held-out accuracy.
    5. Baseline: multinomial-logreg directly on the HMM posterior alpha_t
       (nS-dim) -- this is the Bayes-optimal state classifier given the
       observations, and upper-bounds what any linear probe can reach.

If the "HMM dimensionality is encoded in the top directions" claim holds,
accuracy(Z_k) should saturate near the Bayes level by k ~= nS.

Outputs:
    fig_probe_accuracy.png   accuracy(k) curves, one per nS (ns_sweep) and
                             one per transition-rank (rank_sweep), plus Bayes.
    probe_results.csv        k at which accuracy first reaches 0.95 * Bayes.
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

from random_hmm import random_dense_hmm, random_lowrank_hmm
from model_wrappers import fit_gdc

# Config kept tight so this finishes quickly.
N_TRAIN_SEQ   = 200
TRAIN_LEN     = 40
N_EVAL_SEQ    = 80
EVAL_LEN      = 40
NA            = 5
NS_VALUES     = [2, 3, 4, 6, 8]
RANK_VALUES   = [2, 4, 8]    # at fixed nS=8
K_MAX         = 20
GDC_KWARGS    = dict(alpha=0.7, theta=0.2, gamma=0.0, beta=0.1,
                     transition_type='self_loop',
                     initial_dist='sequence_starts')

# Simple numerical multinomial logistic regression via IRLS / sklearn-free.
# Use gradient descent with small L2 for stability, on standardized features.

def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_multinomial_logreg(X, y, n_classes, l2=1e-3, lr=0.1, iters=500):
    N, D = X.shape
    W = np.zeros((D, n_classes))
    Y = np.eye(n_classes)[y]
    for _ in range(iters):
        P = _softmax(X @ W)
        grad = X.T @ (P - Y) / N + l2 * W
        W -= lr * grad
    return W


def eval_logreg(X_tr, y_tr, X_te, y_te, n_classes):
    # z-score features (column-wise) based on train
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-9
    X_tr_s = (X_tr - mu) / sd
    X_te_s = (X_te - mu) / sd
    # add intercept
    X_tr_s = np.hstack([X_tr_s, np.ones((X_tr_s.shape[0], 1))])
    X_te_s = np.hstack([X_te_s, np.ones((X_te_s.shape[0], 1))])
    W = fit_multinomial_logreg(X_tr_s, y_tr, n_classes)
    pred = np.argmax(X_te_s @ W, axis=1)
    return float(np.mean(pred == y_te))


def build_dataset(hmm, rng_seed):
    rng = np.random.default_rng(rng_seed)
    train = hmm.sample_many(N_TRAIN_SEQ, TRAIN_LEN, rng)
    eval_obs, eval_states = [], []
    for _ in range(N_EVAL_SEQ):
        s, o = hmm.sample(EVAL_LEN, rng)
        eval_obs.append(o); eval_states.append(s)
    gdc = fit_gdc(train, alphabet_size=hmm.nA, **GDC_KWARGS)
    blocks_gdc, blocks_hmm = [], []
    for obs in eval_obs:
        obs_col = np.asarray(obs, np.int64).reshape(-1, 1)
        _, hist = gdc.gdc.forward_pass(obs_col, return_history=True)
        blocks_gdc.append(hist)
        # hmm alpha history
        a = hmm.pi * hmm.E[:, obs[0]]; a = a / max(a.sum(), 1e-12)
        h = [a.copy()]
        for o in obs[1:]:
            a = (a @ hmm.T) * hmm.E[:, o]; a = a / max(a.sum(), 1e-12)
            h.append(a.copy())
        blocks_hmm.append(np.vstack(h))
    M_gdc = np.vstack(blocks_gdc)
    M_hmm = np.vstack(blocks_hmm)
    y = np.concatenate(eval_states)
    return M_gdc, M_hmm, y


def probe_one(hmm, seed):
    M_gdc, M_hmm, y = build_dataset(hmm, seed)
    # split
    N = M_gdc.shape[0]
    rng = np.random.default_rng(seed + 999)
    perm = rng.permutation(N)
    split = N // 2
    tr, te = perm[:split], perm[split:]
    # SVD once.
    U, S, _ = np.linalg.svd(M_gdc, full_matrices=False)
    # projections via U * S.
    Z = U * S
    accs = []
    for k in range(1, K_MAX + 1):
        accs.append(eval_logreg(Z[tr, :k], y[tr], Z[te, :k], y[te], hmm.nS))
    bayes = eval_logreg(M_hmm[tr], y[tr], M_hmm[te], y[te], hmm.nS)
    # S normalized
    sv = (S / S[0])[:K_MAX].tolist()
    return accs, bayes, sv


def main():
    rows = []
    curves_ns = {}   # nS -> list[(accs, bayes, sv)]
    curves_rk = {}
    print('--- ns_sweep ---', flush=True)
    for nS in NS_VALUES:
        curves_ns[nS] = []
        for seed in range(3):
            rng = np.random.default_rng(11 * seed + 7 * nS)
            hmm = random_dense_hmm(nS, NA, rng)
            accs, bayes, sv = probe_one(hmm, seed + 10 * nS)
            curves_ns[nS].append((accs, bayes, sv))
            k95 = next((k for k, a in enumerate(accs, 1) if a >= 0.95 * bayes), None)
            print(f'  nS={nS} seed={seed} bayes={bayes:.3f} '
                  f'k95={k95} accs[:{min(10,K_MAX)}]=' +
                  ' '.join(f'{x:.2f}' for x in accs[:10]), flush=True)
            rows.append({'exp': 'ns_sweep', 'nS': nS, 'rank': nS,
                         'seed': seed, 'bayes': bayes,
                         'k95': k95 if k95 is not None else K_MAX,
                         'acc_at_1': accs[0], 'acc_at_2': accs[1],
                         'acc_at_nS': accs[min(nS, K_MAX)-1],
                         'acc_at_kmax': accs[-1]})
    print('--- rank_sweep ---', flush=True)
    for r in RANK_VALUES:
        curves_rk[r] = []
        for seed in range(3):
            rng = np.random.default_rng(31 * seed + 101 * r)
            hmm = random_lowrank_hmm(8, NA, r, rng)
            accs, bayes, sv = probe_one(hmm, seed + 500 + 10 * r)
            curves_rk[r].append((accs, bayes, sv))
            k95 = next((k for k, a in enumerate(accs, 1) if a >= 0.95 * bayes), None)
            print(f'  rank={r} seed={seed} bayes={bayes:.3f} '
                  f'k95={k95} accs[:10]=' +
                  ' '.join(f'{x:.2f}' for x in accs[:10]), flush=True)
            rows.append({'exp': 'rank_sweep', 'nS': 8, 'rank': r,
                         'seed': seed, 'bayes': bayes,
                         'k95': k95 if k95 is not None else K_MAX,
                         'acc_at_1': accs[0], 'acc_at_2': accs[1],
                         'acc_at_nS': accs[min(r, K_MAX)-1],
                         'acc_at_kmax': accs[-1]})

    # Write CSV
    path = os.path.join(_THIS_DIR, 'probe_results.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('Wrote', path)

    # Plot ns_sweep accuracy curves
    plt.figure(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(NS_VALUES)))
    for nS, c in zip(NS_VALUES, colors):
        arr = np.array([curves_ns[nS][i][0] for i in range(3)])
        bayes = np.mean([curves_ns[nS][i][1] for i in range(3)])
        mean = arr.mean(axis=0); std = arr.std(axis=0)
        ks = np.arange(1, K_MAX + 1)
        plt.plot(ks, mean, 'o-', color=c, label=f'nS={nS} (Bayes {bayes:.2f})')
        plt.fill_between(ks, mean - std, mean + std, color=c, alpha=0.15)
        plt.axhline(bayes, linestyle=':', color=c, alpha=0.5)
        plt.axvline(nS, linestyle='--', color=c, alpha=0.35)
    plt.xlabel('k = # top singular directions used as features')
    plt.ylabel('held-out accuracy predicting true HMM hidden state')
    plt.title('How many singular directions of GDC-posterior carry HMM-state info?\n'
              '(vertical dashed = k=nS; horizontal dotted = Bayes baseline)')
    plt.legend(fontsize=8, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_probe_accuracy.png')
    plt.savefig(out, dpi=120); plt.close(); print('Wrote', out)

    # Plot rank_sweep
    plt.figure(figsize=(7, 5))
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(RANK_VALUES)))
    for r, c in zip(RANK_VALUES, colors):
        arr = np.array([curves_rk[r][i][0] for i in range(3)])
        bayes = np.mean([curves_rk[r][i][1] for i in range(3)])
        mean = arr.mean(axis=0); std = arr.std(axis=0)
        ks = np.arange(1, K_MAX + 1)
        plt.plot(ks, mean, 'o-', color=c, label=f'rank={r} (Bayes {bayes:.2f})')
        plt.fill_between(ks, mean - std, mean + std, color=c, alpha=0.15)
        plt.axhline(bayes, linestyle=':', color=c, alpha=0.5)
        plt.axvline(r, linestyle='--', color=c, alpha=0.35)
    plt.xlabel('k = # top singular directions used as features')
    plt.ylabel('held-out accuracy predicting true HMM hidden state')
    plt.title('Low-rank HMMs (nS=8): k needed for state recovery ~ rank(T)?\n'
              '(vertical dashed = k=rank; horizontal dotted = Bayes)')
    plt.legend(fontsize=8, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_probe_accuracy_rank.png')
    plt.savefig(out, dpi=120); plt.close(); print('Wrote', out)

    print('Done.')


if __name__ == '__main__':
    main()
