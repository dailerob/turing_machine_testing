"""
Does GDC route posterior mass to the correct underlying HMM hidden state?

Same toy HMM as diffusion_experiment.py (nS=4, nA=3, seed 7).

For each diffusion rate d:
    1. Sample TRAINING sequences from the HMM. Record hidden states.
       Each GDC training state j inherits a label h_train[j] = the hidden
       state that emitted observation j during training.
    2. Fit GDC on the (observation-only) training sequences.
    3. Sample fresh EVALUATION sequences with hidden states. Run GDC
       forward_pass with return_history=True over each.
    4. For every (test timepoint t, hidden-state class c):
           p[t, c] = sum_{j : h_train[j] == c} M[t, j]
       This is the GDC's posterior probability that we are currently in
       hidden state c, integrated over all training states aligned with c.
    5. Confusion matrix:
           C[i, c] = mean over timepoints with s_test[t] = i  of  p[t, c]
       Diagonal = "did GDC put weight on training states sampled from the
       same HMM state I'm actually in?".
    6. Report:
           - mean diagonal vs uniform (1/nS) and stationary-marginal baselines
           - per-class accuracy
           - confusion-matrix heatmap (one per d)

Outputs:
    fig_hidden_alignment_confusion.png   grid of confusion matrices vs d
    fig_hidden_alignment_summary.png     mean diagonal & per-class diag vs d
    hidden_alignment_results.csv
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
EXP_SEED = 0
DIFFUSIONS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


def split(d):
    base = 1.0 - d
    return 0.7 * base, 0.3 * base


def stationary_distribution(T, iters=500, tol=1e-12):
    n = T.shape[0]
    pi = np.full(n, 1.0 / n)
    for _ in range(iters):
        nxt = pi @ T
        if np.linalg.norm(nxt - pi) < tol:
            pi = nxt
            break
        pi = nxt
    pi = np.maximum(pi, 0)
    s = pi.sum()
    return pi / s if s > 0 else np.full(n, 1.0 / n)


def sample_training_with_states(hmm, n_seq, length, rng):
    """Like hmm.sample_many but also returns concatenated hidden-state labels
    in the same flat order as GDC's self.states (vstack of obs, sequence by
    sequence)."""
    obs_seqs = []
    h_train = []
    for _ in range(n_seq):
        states, obs = hmm.sample(length, rng)
        obs_seqs.append(obs)
        h_train.append(states)
    return obs_seqs, np.concatenate(h_train)


def run_one(hmm, d, rng):
    a, t = split(d)
    train_obs, h_train = sample_training_with_states(
        hmm, N_TRAIN_SEQ, TRAIN_LEN, rng)
    eval_obs, eval_states = [], []
    for _ in range(N_EVAL_SEQ):
        s, o = hmm.sample(EVAL_LEN, rng)
        eval_obs.append(o); eval_states.append(s)
    gdc = fit_gdc(train_obs, alphabet_size=hmm.nA,
                  alpha=a, theta=t, gamma=0.0, beta=0.1,
                  transition_type='self_loop', initial_dist='sequence_starts')

    # Sanity: h_train should align row-for-row with gdc.gdc.states
    assert len(h_train) == gdc.gdc.n_states, (
        f"hidden-state label count {len(h_train)} != GDC n_states "
        f"{gdc.gdc.n_states}")

    # Build label projection P_lab (n_gdc x nS): one-hot of h_train[j].
    nS = hmm.nS
    n_gdc = gdc.gdc.n_states
    P_lab = np.zeros((n_gdc, nS))
    P_lab[np.arange(n_gdc), h_train] = 1.0

    # Inference
    rows = []
    blocks = []
    state_blocks = []
    for obs, st in zip(eval_obs, eval_states):
        oc = np.asarray(obs, np.int64).reshape(-1, 1)
        _, hist = gdc.gdc.forward_pass(oc, return_history=True)
        # hist (T x n_gdc).  Multiply by P_lab to get (T x nS) marginal-by-
        # hidden-state.
        p_hidden = hist @ P_lab          # (T, nS)
        blocks.append(p_hidden)
        state_blocks.append(st)
    p_hidden_all = np.vstack(blocks)              # (Ntot, nS)
    s_true_all   = np.concatenate(state_blocks)   # (Ntot,)

    # Confusion matrix C[i, c] = mean_{t: s_true=i} p_hidden_all[t, c]
    C = np.zeros((nS, nS))
    counts = np.zeros(nS)
    for i in range(nS):
        mask = s_true_all == i
        counts[i] = mask.sum()
        if mask.any():
            C[i] = p_hidden_all[mask].mean(axis=0)
    return C, counts, p_hidden_all, s_true_all


def main():
    rng_hmm = np.random.default_rng(HMM_SEED)
    hmm = random_dense_hmm(NS, NA, rng_hmm)
    pi_stat = stationary_distribution(hmm.T)
    print(f'HMM nS={hmm.nS} nA={hmm.nA}')
    print(f'Stationary distribution: {pi_stat}')
    print(f'Uniform baseline diagonal: {1/NS:.4f}')
    print(f'Stationary baseline diagonal (trace pi_stat): '
          f'{np.sum(pi_stat**2):.4f}')

    # Bayes-optimal upper bound: HMM forward filter applied to the same
    # eval observations. This is the maximum possible "weight on correct
    # hidden state" given the observations alone.
    rng_bayes = np.random.default_rng(EXP_SEED + 1000)
    # Re-sample eval seqs identical to those in run_one(). For consistency
    # we just reuse the rng pattern, recognising that NumPy generators are
    # deterministic.  We sample fresh:
    bayes_diag = np.zeros(NS)
    bayes_counts = np.zeros(NS)
    bayes_C = np.zeros((NS, NS))
    rng_b = np.random.default_rng(EXP_SEED + 2000)
    # use a fresh eval set just for the Bayes ceiling
    for _ in range(N_EVAL_SEQ):
        s, o = hmm.sample(EVAL_LEN, rng_b)
        # forward filter
        a = hmm.pi * hmm.E[:, o[0]]; a = a / max(a.sum(), 1e-12)
        hist = [a.copy()]
        for ob in o[1:]:
            a = (a @ hmm.T) * hmm.E[:, ob]; a = a / max(a.sum(), 1e-12)
            hist.append(a.copy())
        H = np.vstack(hist)               # (T, nS)  alpha
        for i in range(NS):
            mask = s == i
            if mask.any():
                bayes_C[i] += H[mask].sum(axis=0)
                bayes_counts[i] += mask.sum()
    bayes_C = bayes_C / np.maximum(bayes_counts[:, None], 1)
    bayes_diag = np.diag(bayes_C)
    bayes_mean = float(bayes_diag.mean())
    bayes_weighted = float((bayes_diag * bayes_counts).sum()
                           / bayes_counts.sum())
    print(f'  Bayes (HMM alpha):  per-class diag='
          f'{np.array2string(bayes_diag, precision=3)} '
          f'mean={bayes_mean:.3f}  weighted={bayes_weighted:.3f}')

    confs = {'bayes': (bayes_C, bayes_counts)}
    summary_rows = []
    for d in DIFFUSIONS:
        rng = np.random.default_rng(EXP_SEED + 1000)
        C, counts, p_all, s_all = run_one(hmm, d, rng)
        diag = np.diag(C)
        mean_diag = float(diag.mean())
        # Weighted mean diag (by class frequency)
        weighted = float((diag * counts).sum() / counts.sum())
        confs[d] = (C, counts)
        print(f'  d={d:5.2f}  per-class diag={np.array2string(diag, precision=3)} '
              f'mean={mean_diag:.3f}  weighted={weighted:.3f}')
        summary_rows.append({
            'd': d, 'mean_diag': mean_diag, 'weighted_diag': weighted,
            **{f'diag_s{i}': diag[i] for i in range(NS)},
        })

    # CSV
    csv_path = os.path.join(_THIS_DIR, 'hidden_alignment_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print('Wrote', csv_path)

    # Confusion matrices grid + Bayes panel on the right
    keys = list(DIFFUSIONS) + ['bayes']
    cols = len(keys)
    fig, axes = plt.subplots(1, cols, figsize=(2.6 * cols, 3.0),
                             squeeze=False)
    for ax, d in zip(axes[0], keys):
        C, _ = confs[d]
        im = ax.imshow(C, cmap='viridis', vmin=0, vmax=1)
        if d == 'bayes':
            ax.set_title(f'Bayes (HMM α)\nmean diag={np.diag(C).mean():.2f}',
                         fontsize=10)
        else:
            ax.set_title(f'd={d:.2f}\nmean diag={np.diag(C).mean():.2f}',
                         fontsize=10)
        ax.set_xticks(range(NS)); ax.set_yticks(range(NS))
        ax.set_xticklabels([f's{c}' for c in range(NS)], fontsize=8)
        ax.set_yticklabels([f's{c}' for c in range(NS)], fontsize=8)
        if d == DIFFUSIONS[0]:
            ax.set_ylabel('true HMM state at test time')
        ax.set_xlabel('GDC weight on hidden state')
        for i in range(NS):
            for j in range(NS):
                ax.text(j, i, f'{C[i,j]:.2f}', ha='center', va='center',
                        fontsize=7,
                        color='white' if C[i, j] < 0.5 else 'black')
    fig.suptitle('Confusion: mass GDC routes to each hidden-state class\n'
                 '(rows: true state at test time; columns: aggregated '
                 'GDC posterior over training states sampled from each state)',
                 fontsize=11)
    fig.colorbar(im, ax=axes[0], shrink=0.7, label='posterior weight')
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_hidden_alignment_confusion.png')
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print('Wrote', out)

    # Summary plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ds = [r['d'] for r in summary_rows]
    md = [r['mean_diag'] for r in summary_rows]
    wd = [r['weighted_diag'] for r in summary_rows]
    ax.plot(ds, md, 'o-', label='GDC mean per-class diag (unweighted)',
            color='steelblue')
    ax.plot(ds, wd, 's-', label='GDC mean per-class diag (weighted)',
            color='darkorange')
    ax.axhline(bayes_mean, color='green', linestyle='-', linewidth=2,
               label=f'Bayes (HMM α) ceiling: {bayes_mean:.3f}')
    ax.axhline(1.0 / NS, color='grey', linestyle='--',
               label=f'uniform: 1/nS = {1/NS:.3f}')
    ax.axhline(np.sum(pi_stat ** 2), color='red', linestyle=':',
               label=f'stationary self-overlap: {np.sum(pi_stat**2):.3f}')
    ax.set_xlabel('diffusion rate d')
    ax.set_ylabel('GDC posterior weight on correct hidden state class')
    ax.set_title('Does GDC route mass to the right hidden state? '
                 '(toy HMM, nS=4, nA=3)')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig_hidden_alignment_summary.png')
    plt.savefig(out, dpi=130); plt.close()
    print('Wrote', out)
    print('Done.')


if __name__ == '__main__':
    main()
