"""α × β sweep at TL=25 on the four new regimes (bimodal_small,
cyclic_K8, binary_deep, reset_chain). Mirrors alpha_beta_sweep_tl25.py
but uses the new HMM constructors.

Sweeps α ∈ {0.10, 0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99} ×
β ∈ {0.0, 0.001, 0.01, 0.05, 0.10, 0.20, 0.30}, θ = 0.001.
Mode: absorb + uniform; torch batched fp32.

Aggregates against existing baseline numbers from
new_regimes_25_results.csv to show the GDC-vs-others comparison
under a wider config grid.
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np
import torch
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from random_hmm import (random_bimodal_hmm, random_cyclic_hmm,
                       random_binary_deep_hmm,
                       random_reset_chain_hmm)
from generative_dense_chain import GenerativeDenseChain
from gdc_torch_discrete import horizon_emission_many

REGIMES = [
    ('bimodal_small',  10, 4, 'bimodal'),
    ('cyclic_K8',       8, 8, 'cyclic'),
    ('binary_deep',    30, 2, 'binary_deep'),
    ('reset_chain',    20, 4, 'reset_chain'),
]
SEEDS = [0, 1, 2, 3, 4, 5]
VAL_SEEDS = {3, 4, 5}
TEST_SEEDS = {0, 1, 2}
N_TRAIN_VALUES = [25, 100, 400]
TRAIN_LEN = 25
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
H = 1

ALPHAS = [0.10, 0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
BETAS  = [0.00, 0.001, 0.01, 0.05, 0.10, 0.20, 0.30]
THETA  = 0.001

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

OUT_RAW = os.path.join(HERE, 'alpha_beta_new_regimes_results.csv')
OUT_PICKS = os.path.join(HERE, 'alpha_beta_new_regimes_picks.csv')


def make_hmm(kind, nS, nA, rng):
    if kind == 'bimodal':
        return random_bimodal_hmm(nS, nA, rng,
                                  sticky_prob=0.95, E_concentration=0.1)
    if kind == 'cyclic':
        return random_cyclic_hmm(nS, nA, rng,
                                 advance_prob=0.95, E_concentration=0.1)
    if kind == 'binary_deep':
        return random_binary_deep_hmm(nS, rng,
                                      fanout=2, E_concentration=0.1)
    if kind == 'reset_chain':
        return random_reset_chain_hmm(nS, nA, rng,
                                      advance_prob=0.90, reset_prob=0.05,
                                      E_concentration=0.1)
    raise ValueError(f"unknown kind: {kind}")


def true_filter_marginals(hmm, prefixes, h=1):
    B = len(prefixes); nA = hmm.nA
    Th = np.linalg.matrix_power(hmm.T, h)
    next_obs = np.zeros((B, nA))
    floor_bits = np.zeros(B)
    for i, p in enumerate(prefixes):
        a = hmm.filter(p)
        nd = a @ Th @ hmm.E
        next_obs[i] = nd
        floor_bits[i] = -float(np.sum(nd * np.log2(np.maximum(nd, 1e-12))))
    return next_obs, floor_bits


def excess_pp_from_preds(preds, true_next, floor_bits):
    pred_safe = np.maximum(preds, 1e-12)
    ce = -np.sum(true_next * np.log2(pred_safe), axis=1)
    return float(2 ** (ce.mean() - floor_bits.mean()))


def run_cell(regime_name, nS, nA, kind, seed, N):
    kind_tag = {'bimodal': 0, 'cyclic': 10, 'binary_deep': 20,
                'reset_chain': 30}[kind]
    rng = np.random.default_rng(70000 + kind_tag + seed * 137 + nS * 7
                                + nA * 11)
    hmm = make_hmm(kind, nS, nA, rng)
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

    true_next, floor_bits = true_filter_marginals(hmm, test_pf, h=H)
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
    n_cells = len(REGIMES) * len(SEEDS) * len(N_TRAIN_VALUES)
    n_configs = len(ALPHAS) * len(BETAS)
    print(f"=== α × β sweep on new regimes, TL={TRAIN_LEN} ===")
    print(f"  Regimes: {[r[0] for r in REGIMES]}")
    print(f"  α grid: {ALPHAS}")
    print(f"  β grid: {BETAS}")
    print(f"  Cells: {n_cells} × configs/cell: {n_configs} = "
          f"{n_cells * n_configs} batched calls\n", flush=True)

    all_rows = []
    t0 = time.time()
    done = 0
    for (name, nS, nA, kind) in REGIMES:
        for seed in SEEDS:
            for N in N_TRAIN_VALUES:
                cell_rows = run_cell(name, nS, nA, kind, seed, N)
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
    print(f"\nWrote {OUT_RAW} ({len(all_rows)} rows)\n")

    # Val-pick per (regime, N), test on test seeds
    df = pd.DataFrame(all_rows)
    pick_rows = []
    for regime, *_ in REGIMES:
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

    # Print picks
    print("## Free GDC α × β picks per cell\n")
    print("| Regime | N | α | β | val_pp | test_pp |")
    print("|---|---:|---:|---:|---:|---:|")
    for _, r in picks.iterrows():
        print(f"| {r.regime} | {int(r.N)} | {r.alpha:.2f} | "
              f"{r.beta:.3f} | {r.val_pp:.4f} | {r.test_pp:.4f} |")

    # Compare against existing methods from new_regimes_25_results.csv
    base_path = os.path.join(HERE, 'new_regimes_25_results.csv')
    if not os.path.exists(base_path):
        print(f"\n  (No baseline file at {base_path}; skipping comparison)")
        return

    base = pd.read_csv(base_path)
    base = base[(base.train_len == TRAIN_LEN) & (base.horizon == 1)]

    METHODS = ['gdc_free', 'chmm', 'alergia', 'parrot', 'hpylm', 'ppm', 'freq']
    PRETTY = {'gdc_free': 'GDC-free', 'chmm': 'CHMM', 'alergia': 'ALERGIA',
              'parrot': 'Parrot', 'hpylm': 'HPYLM', 'ppm': 'PPM-D',
              'freq': 'Freq'}
    print(f"\n## Combined: GDC-free vs original baselines (TL={TRAIN_LEN}, h=1)\n")
    print("**Bold = best in row (excluding Freq).**\n")
    for regime, *_ in REGIMES:
        print(f"### {regime}\n")
        print("| N | " + " | ".join(PRETTY[m] for m in METHODS) + " |")
        print("|---:|" + "---:|" * len(METHODS))
        for N in N_TRAIN_VALUES:
            cell = base[(base.regime == regime) & (base.N_train == N)]
            test_pp = {}
            for m in ['chmm', 'alergia', 'parrot', 'hpylm', 'ppm', 'freq']:
                csub = cell[cell.model_class == m]
                cval = csub[csub.seed.isin(VAL_SEEDS)]
                ctest = csub[csub.seed.isin(TEST_SEEDS)]
                if csub.empty:
                    test_pp[m] = float('nan'); continue
                if m in ('alergia', 'freq'):
                    test_pp[m] = float(ctest.excess_perplexity.mean())
                else:
                    val_means = (cval.groupby('model')
                                     ['excess_perplexity'].mean())
                    if val_means.empty:
                        test_pp[m] = float('nan'); continue
                    pick = val_means.idxmin()
                    test_pp[m] = float(ctest[ctest.model == pick]
                                       .excess_perplexity.mean())
            # GDC-free from picks df
            pr = picks[(picks.regime == regime) & (picks.N == N)]
            test_pp['gdc_free'] = float(pr.test_pp.iloc[0]) \
                                  if not pr.empty else float('nan')

            non_freq = [test_pp[m] for m in METHODS
                        if m != 'freq' and not np.isnan(test_pp[m])]
            best = min(non_freq) if non_freq else float('nan')
            cells = []
            for m in METHODS:
                v = test_pp[m]
                if np.isnan(v):
                    cells.append('—')
                elif m == 'freq':
                    cells.append(f"_{v:.4f}_")
                elif abs(v - best) < 1e-4:
                    cells.append(f"**{v:.4f}**")
                else:
                    cells.append(f"{v:.4f}")
            print(f"| {N} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
