"""Independent α × β sweep at TL=25 (HMM forecasting benchmark).

Drops the bundled (α, θ, β) 5-config grid that the headline sweep used.
Sweeps α ∈ {0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.99} × β ∈ {0.0,
0.001, 0.01, 0.05, 0.10, 0.20, 0.30}, holding θ = 0.001 fixed.

GDC modes: absorb + uniform.

Uses the torch batched discrete kernel `horizon_emission_many` with all
100 prefixes per cell batched into one call per (α, β). True HMM filter
distributions are pre-computed per cell.

Per cell we val-pick the (α, β) with lowest mean val-seed excess_pp,
report test-seed mean.
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np
import torch
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from random_hmm import random_dense_hmm, random_sparse_topology_hmm
from generative_dense_chain import GenerativeDenseChain
from gdc_torch_discrete import horizon_emission_many

REGIMES = [
    ('dense_small',  10, 4, 'dense',  1.0, None),
    ('dense_large',  30, 8, 'dense',  1.0, None),
    ('det_small',    10, 4, 'dense',  0.1, None),
    ('det_large',    30, 8, 'dense',  0.1, None),
    ('sparse_small', 10, 4, 'sparse', 0.1, 2),
    ('sparse_large', 30, 8, 'sparse', 0.1, 2),
]
SEEDS = [0, 1, 2, 3, 4, 5]
VAL_SEEDS = {3, 4, 5}
TEST_SEEDS = {0, 1, 2}
N_TRAIN_VALUES = [25, 100, 400]
TRAIN_LEN = 25
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
H = 1  # horizon

ALPHAS = [0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.99]
BETAS  = [0.00, 0.001, 0.01, 0.05, 0.10, 0.20, 0.30]
THETA  = 0.001

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

OUT_RAW   = os.path.join(HERE, 'alpha_beta_sweep_tl25_results.csv')
OUT_PICKS = os.path.join(HERE, 'alpha_beta_sweep_tl25_picks.csv')


def true_filter_marginals(hmm, prefixes, h=1):
    """Return arrays of true HMM filter posteriors (B, nS) and
    h-step-ahead next-symbol distributions (B, nA) for each prefix.
    Also returns floor entropy (one bit-value per prefix)."""
    B = len(prefixes); nS, nA = hmm.nS, hmm.nA
    Th = np.linalg.matrix_power(hmm.T, h)
    state_post = np.zeros((B, nS))
    next_obs = np.zeros((B, nA))
    floor_bits = np.zeros(B)
    for i, p in enumerate(prefixes):
        a = hmm.filter(p)
        state_post[i] = a
        nd = a @ Th @ hmm.E
        next_obs[i] = nd
        floor_bits[i] = -float(np.sum(nd * np.log2(np.maximum(nd, 1e-12))))
    return state_post, next_obs, floor_bits


def excess_pp_from_preds(preds, true_next, floor_bits):
    """preds: (B, nA), true_next: (B, nA), floor_bits: (B,).
    Returns excess perplexity = 2^(mean_CE - mean_floor)."""
    pred_safe = np.maximum(preds, 1e-12)
    ce = -np.sum(true_next * np.log2(pred_safe), axis=1)  # (B,)
    return float(2 ** (ce.mean() - floor_bits.mean()))


def run_cell(regime_name, nS, nA, kind, E_conc, fanout, seed, N):
    seed_offset = (1 if 'det' in regime_name else 0) \
                  + (2 if 'sparse' in regime_name else 0)
    rng = np.random.default_rng(60000 + seed * 137 + nS * 7 + nA * 11
                                + seed_offset)
    if kind == 'sparse':
        hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                         E_concentration=E_conc)
    else:
        hmm = random_dense_hmm(nS, nA, rng,
                               T_concentration=1.0, E_concentration=E_conc)
    full_train = [hmm.sample(TRAIN_LEN, rng)[1]
                  for _ in range(max(N_TRAIN_VALUES))]
    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]

    train = full_train[:N]
    seq_arrays = [s.reshape(-1, 1).astype(np.int64) for s in train]
    gdc = GenerativeDenseChain(
        seq_arrays, alpha=ALPHAS[0], theta=THETA, gamma=0.0,
        beta=BETAS[0], transition_type='self_loop',
        initial_dist='uniform', terminal_behavior='absorb')
    sym = gdc.states[:, 0].astype(np.int64)

    # Pre-compute true HMM marginals for all prefixes in this cell
    _, true_next, floor_bits = true_filter_marginals(hmm, test_pf, h=H)

    primes = np.stack([np.asarray(p, dtype=np.int64) for p in test_pf])

    rows = []
    for alpha in ALPHAS:
        for beta in BETAS:
            out = horizon_emission_many(
                symbol_of_state=sym,
                terminal_mask=gdc.terminal_mask,
                start_mask=gdc.start_mask,
                primes=primes, horizons=[H], nA=nA,
                alpha=alpha, theta=THETA, beta=beta,
                transition_type='self_loop',
                terminal_behavior='absorb',
                initial_dist='uniform',
                device=DEVICE, dtype=DTYPE)
            preds = out.cpu().numpy().reshape(N_TEST_PREFIXES, nA)
            ex_pp = excess_pp_from_preds(preds, true_next, floor_bits)
            rows.append(dict(regime=regime_name, nS=nS, nA=nA,
                             seed=seed, N_train=N,
                             alpha=alpha, theta=THETA, beta=beta,
                             excess_pp=ex_pp))
    return rows


def main():
    n_cells = sum(1 for _ in REGIMES) * len(SEEDS) * len(N_TRAIN_VALUES)
    n_configs = len(ALPHAS) * len(BETAS)
    print(f"=== α × β sweep at TL={TRAIN_LEN} ===")
    print(f"  α grid: {ALPHAS}")
    print(f"  β grid: {BETAS}")
    print(f"  θ fixed at {THETA}")
    print(f"  GDC mode: absorb + uniform (torch fp32, batched 100 prefixes/call)")
    print(f"  Cells: {n_cells}, configs/cell: {n_configs}")
    print(f"  Total batched calls: {n_cells * n_configs}\n", flush=True)

    all_rows = []
    t0 = time.time()
    done = 0
    for (name, nS, nA, kind, conc, fanout) in REGIMES:
        for seed in SEEDS:
            for N in N_TRAIN_VALUES:
                cell_rows = run_cell(name, nS, nA, kind, conc, fanout,
                                      seed, N)
                all_rows.extend(cell_rows)
                done += 1
                if done % 6 == 0 or done == n_cells:
                    print(f"  {done}/{n_cells} cells  "
                          f"[{time.time()-t0:.0f}s]", flush=True)

    fields = ['regime', 'nS', 'nA', 'seed', 'N_train',
              'alpha', 'theta', 'beta', 'excess_pp']
    with open(OUT_RAW, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_RAW}  ({len(all_rows)} rows)\n")

    df = pd.DataFrame(all_rows)
    pick_rows = []
    for regime in [r[0] for r in REGIMES]:
        for N in N_TRAIN_VALUES:
            cell = df[(df.regime == regime) & (df.N_train == N)]
            cval = cell[cell.seed.isin(VAL_SEEDS)]
            ctest = cell[cell.seed.isin(TEST_SEEDS)]
            val_means = (cval.groupby(['alpha','beta'])['excess_pp']
                              .mean())
            (best_alpha, best_beta) = val_means.idxmin()
            best_val = float(val_means.min())
            test_pp = float(ctest[(ctest.alpha == best_alpha)
                                   & (ctest.beta == best_beta)]
                            .excess_pp.mean())
            pick_rows.append(dict(regime=regime, N=N,
                                  alpha=best_alpha, beta=best_beta,
                                  val_pp=best_val, test_pp=test_pp))
    picks = pd.DataFrame(pick_rows)
    picks.to_csv(OUT_PICKS, index=False)
    print(f"Wrote {OUT_PICKS}\n")

    print("## Val-picked (α, β) per cell (TL=25)\n")
    print("| Regime | N | α | β | val_pp | test_pp |")
    print("|---|---:|---:|---:|---:|---:|")
    for _, r in picks.iterrows():
        print(f"| {r.regime} | {int(r.N)} | {r.alpha:.2f} | "
              f"{r.beta:.3f} | {r.val_pp:.4f} | {r.test_pp:.4f} |")

    orig_path = os.path.join(HERE, 'seq_len_table.csv')
    if os.path.exists(orig_path):
        orig = pd.read_csv(orig_path)
        orig = orig[(orig.train_len == TRAIN_LEN)
                     & (orig.model_class == 'gdc')]
        print("\n## Bundled-grid vs free α × β (TL=25)\n")
        print("| Regime | N | bundled test_pp | free test_pp | Δ% of gap |")
        print("|---|---:|---:|---:|---:|")
        for _, r in picks.iterrows():
            o = orig[(orig.regime == r.regime) & (orig.N == r.N)]
            if o.empty:
                continue
            ot = float(o.test_pp.iloc[0])
            delta_gap = (r.test_pp - ot) / max(ot - 1, 1e-6) * 100
            arrow = "↓" if r.test_pp < ot else ("↑" if r.test_pp > ot
                                                 else "=")
            print(f"| {r.regime} | {int(r.N)} | {ot:.4f} | "
                  f"{r.test_pp:.4f} | {arrow}{abs(delta_gap):.1f}% |")


if __name__ == "__main__":
    main()
