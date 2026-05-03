"""GDC high-beta refinement on sparse regimes.

Earlier zero-theta sweep showed sparse_large MSE monotonically
decreasing across β ∈ {0.0, 0.05, 0.1, 0.2} — the upper bound.
This script extends β much higher to find the true ceiling.

    alpha       in {0.70, 0.80, 0.85, 0.90}
    theta       = 0.0
    beta        in {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
    transition  : self_loop
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
from itertools import product
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REGIMES = [
    ('sparse_small', 10, 4, 0.1, 2),
    ('sparse_large', 30, 8, 0.1, 2),
]
SEEDS = [0, 1, 2]
N_TRAIN_VALUES = [100, 400]
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 5]

ALPHAS = [0.70, 0.80, 0.85, 0.90]
BETAS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
THETA = 0.0

CHMM_KS = [32]
ALERGIA_EPS = 0.05

OUT_CSV = os.path.join(HERE, 'high_beta_sparse_results.csv')


def run_cell(args):
    regime_name, nS, nA, E_conc, fanout, seed = args
    sys.path.insert(0, HERE)
    from random_hmm import random_sparse_topology_hmm
    from model_wrappers import fit_gdc
    from chmm_alergia_wrappers import fit_chmm, fit_alergia
    from evaluation import mse_at_horizons, perplexity_at_horizons

    rng = np.random.default_rng(80000 + seed * 137 + nS * 7 + nA * 11 + 2)
    hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                     E_concentration=E_conc)
    full_train = [hmm.sample(TRAIN_LEN, rng)[1]
                  for _ in range(max(N_TRAIN_VALUES))]
    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]

    rows = []

    def record(name, model, N):
        mse = mse_at_horizons(model, hmm, test_pf, HORIZONS)
        ppl = perplexity_at_horizons(model, hmm, test_pf, HORIZONS)
        for h in HORIZONS:
            r = ppl[h]
            rows.append(dict(regime=regime_name, nS=nS, nA=nA,
                             seed=seed, N_train=N, model=name, horizon=h,
                             mse=mse[h],
                             excess_perplexity=r['excess_perplexity']))

    for N in N_TRAIN_VALUES:
        train = full_train[:N]
        for a, b in product(ALPHAS, BETAS):
            try:
                m = fit_gdc(train, nA, alpha=a, theta=THETA, gamma=0.0, beta=b,
                            transition_type='self_loop',
                            initial_dist='sequence_starts')
                record(f'gdc-a{a}-t0.0-b{b}', m, N)
            except Exception as e:
                sys.stderr.write(f"[gdc fail a={a} b={b} {regime_name} N={N}] {e}\n")
        for K in CHMM_KS:
            try:
                m = fit_chmm(train, nA, K=K, n_em_iters=50)
                record(f'chmm-K{K}', m, N)
            except Exception as e:
                sys.stderr.write(f"[chmm K={K} fail {regime_name} N={N}] {e}\n")
        try:
            m = fit_alergia(train, nA, eps=ALERGIA_EPS)
            record('alergia-eps0.05', m, N)
        except Exception as e:
            sys.stderr.write(f"[alergia fail {regime_name} N={N}] {e}\n")
    return rows


def main():
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    tasks = [(name, nS, nA, conc, fanout, seed)
             for (name, nS, nA, conc, fanout) in REGIMES
             for seed in SEEDS]
    print(f"High-beta sparse sweep: {len(ALPHAS)}x{len(BETAS)} GDC + 2 ref, "
          f"{len(tasks)} cells, {n_workers} workers", flush=True)

    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for cell_rows in pool.imap_unordered(run_cell, tasks, chunksize=1):
            all_rows.extend(cell_rows)
            done += 1
            print(f"  {done}/{len(tasks)} done [{time.time()-t0:.0f}s]",
                  flush=True)

    fields = ['regime', 'nS', 'nA', 'seed', 'N_train', 'model', 'horizon',
              'mse', 'excess_perplexity']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}  [{time.time()-t0:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
