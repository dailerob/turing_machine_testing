"""Sparse-regime sweep with longer training sequences.

Earlier sparse experiments used TRAIN_LEN=50.  At fanout=2 with
nS=30, a length-50 sequence visits at most ~50 of the ~60 reachable
transitions; the highly concentrated stationary distribution means
many short sequences may not even leave a small subset of states.

This script tests whether longer sequences close GDC's gap to CHMM
on sparse data.

Sweep:
    TRAIN_LEN in {50, 200, 500}
    N_train  in {100, 400}
    regimes  : sparse_small (nS=10, nA=4), sparse_large (nS=30, nA=8)
    seeds    : 3

Models:
    GDC: 5 configs spanning the dense and sparse optima
    CHMM K in {4, 16, 32}, with 30 EM iters (capped to keep K=32 tractable
    at T=500)
    ALERGIA eps=0.05
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REGIMES = [
    ('sparse_small', 10, 4, 0.1, 2),
    ('sparse_large', 30, 8, 0.1, 2),
]
SEEDS = [0, 1, 2]
N_TRAIN_VALUES = [100, 400]
TRAIN_LENS = [50, 200, 500]
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 5]

GDC_CONFIGS = [
    (0.10, 0.001, 0.20),  # dense-large optimum
    (0.50, 0.005, 0.20),  # midline
    (0.70, 0.010, 0.20),  # earlier sparse-best
    (0.80, 0.001, 0.10),  # high-alpha sparse-best (long-N=400 winner)
    (0.85, 0.001, 0.10),  # nearby
]
CHMM_KS = [4, 16, 32]
CHMM_EM_ITERS = 30
ALERGIA_EPS = 0.05

OUT_CSV = os.path.join(HERE, 'long_seqs_sparse_results.csv')


def run_cell(args):
    regime_name, nS, nA, E_conc, fanout, seed, T_len = args
    sys.path.insert(0, HERE)
    from random_hmm import random_sparse_topology_hmm
    from model_wrappers import fit_gdc
    from chmm_alergia_wrappers import fit_chmm, fit_alergia
    from evaluation import mse_at_horizons

    rng = np.random.default_rng(50000 + seed * 137 + nS * 7 + nA * 11
                                + T_len * 3)
    hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                     E_concentration=E_conc)
    full_train = [hmm.sample(T_len, rng)[1]
                  for _ in range(max(N_TRAIN_VALUES))]
    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]

    rows = []
    for N in N_TRAIN_VALUES:
        train = full_train[:N]
        for (a, t, b) in GDC_CONFIGS:
            m = fit_gdc(train, nA, alpha=a, theta=t, gamma=0.0, beta=b,
                        transition_type='self_loop',
                        initial_dist='sequence_starts')
            for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
                rows.append(dict(regime=regime_name, T_len=T_len,
                                 nS=nS, nA=nA, seed=seed, N_train=N,
                                 model=f'gdc-a{a}-t{t}-b{b}',
                                 horizon=h, mse=mse))
        for K in CHMM_KS:
            try:
                m = fit_chmm(train, nA, K=K, n_em_iters=CHMM_EM_ITERS)
                for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
                    rows.append(dict(regime=regime_name, T_len=T_len,
                                     nS=nS, nA=nA, seed=seed, N_train=N,
                                     model=f'chmm-K{K}',
                                     horizon=h, mse=mse))
            except Exception as e:
                sys.stderr.write(
                    f"[chmm K={K} fail {regime_name} T={T_len} N={N}] {e}\n")
        try:
            m = fit_alergia(train, nA, eps=ALERGIA_EPS)
            for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
                rows.append(dict(regime=regime_name, T_len=T_len,
                                 nS=nS, nA=nA, seed=seed, N_train=N,
                                 model='alergia-eps0.05',
                                 horizon=h, mse=mse))
        except Exception as e:
            sys.stderr.write(
                f"[alergia fail {regime_name} T={T_len} N={N}] {e}\n")
    return rows


def main():
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    tasks = [(name, nS, nA, conc, fanout, seed, T_len)
             for (name, nS, nA, conc, fanout) in REGIMES
             for seed in SEEDS
             for T_len in TRAIN_LENS]
    print(f"Long-sequence sparse sweep: {len(tasks)} cells, "
          f"{n_workers} workers", flush=True)
    print(f"  GDC configs: {len(GDC_CONFIGS)}, CHMM Ks: {CHMM_KS}, "
          f"EM iters cap = {CHMM_EM_ITERS}", flush=True)

    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for cell_rows in pool.imap_unordered(run_cell, tasks, chunksize=1):
            all_rows.extend(cell_rows)
            done += 1
            print(f"  {done}/{len(tasks)} cells done  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    fields = ['regime', 'T_len', 'nS', 'nA', 'seed', 'N_train', 'model',
              'horizon', 'mse']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}  [{time.time()-t0:.1f}s, "
          f"{len(all_rows)} rows]", flush=True)


if __name__ == "__main__":
    main()
