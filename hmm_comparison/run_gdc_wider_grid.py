"""Wider GDC hyperparameter grid, including high alpha (0.99) and low
beta (0.01), on a representative set of regimes.

Grid:
    alpha in {0.5, 0.7, 0.9, 0.95, 0.99}
    theta in {0.001, 0.01, 0.05, 0.1}
    beta  in {0.01, 0.05, 0.1, 0.2, 0.3}
    transition: self_loop only
    constraint: alpha + theta <= 1

Regimes (4):
    dense_small  : nS=10, nA=4, E_concentration=1.0
    dense_large  : nS=30, nA=8, E_concentration=1.0
    det_small    : nS=10, nA=4, E_concentration=0.1
    det_large    : nS=30, nA=8, E_concentration=0.1

N_train values: {25, 200}  (low- and high-data)
Seeds: 3

For each (regime, N_train) cell, sweep all valid GDC configs and
also evaluate ALERGIA and CHMM K in {4, 16, 32} as reference.

Output: gdc_wider_grid_results.csv
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
from itertools import product
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ALPHAS = [0.5, 0.7, 0.9, 0.95, 0.99]
THETAS = [0.001, 0.01, 0.05, 0.1]
BETAS = [0.01, 0.05, 0.1, 0.2, 0.3]

REGIMES = [
    ('dense_small', 10, 4, 1.0),
    ('dense_large', 30, 8, 1.0),
    ('det_small',   10, 4, 0.1),
    ('det_large',   30, 8, 0.1),
]
SEEDS = [0, 1, 2]
N_TRAIN_VALUES = [25, 200]
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 5]

CHMM_KS = [4, 16, 32]
ALERGIA_EPS = 0.05

OUT_CSV = os.path.join(HERE, 'gdc_wider_grid_results.csv')


def gdc_configs():
    out = []
    for a, t, b in product(ALPHAS, THETAS, BETAS):
        if a + t <= 1.0:
            out.append((a, t, b))
    return out


def run_cell(args):
    """Worker: one (regime, seed) cell.  Sweeps GDC grid for both N
    values, plus ALERGIA + CHMM at multiple K."""
    regime_name, nS, nA, E_conc, seed = args
    sys.path.insert(0, HERE)
    from random_hmm import random_dense_hmm
    from model_wrappers import fit_gdc
    from chmm_alergia_wrappers import fit_chmm, fit_alergia
    from evaluation import mse_at_horizons

    rng = np.random.default_rng(
        30000 + seed * 137 + nS * 7 + nA * 11
        + (1 if 'det' in regime_name else 0))
    hmm = random_dense_hmm(nS, nA, rng,
                           T_concentration=1.0, E_concentration=E_conc)
    full_train = [hmm.sample(TRAIN_LEN, rng)[1]
                  for _ in range(max(N_TRAIN_VALUES))]
    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]

    rows = []
    for N in N_TRAIN_VALUES:
        train = full_train[:N]
        for (a, t, b) in gdc_configs():
            m = fit_gdc(train, nA, alpha=a, theta=t, gamma=0.0, beta=b,
                        transition_type='self_loop',
                        initial_dist='sequence_starts')
            for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
                rows.append(dict(regime=regime_name, nS=nS, nA=nA,
                                 seed=seed, N_train=N,
                                 model=f'gdc-a{a}-t{t}-b{b}',
                                 alpha=a, theta=t, beta=b,
                                 horizon=h, mse=mse))
        for K in CHMM_KS:
            try:
                m = fit_chmm(train, nA, K=K, n_em_iters=50)
                for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
                    rows.append(dict(regime=regime_name, nS=nS, nA=nA,
                                     seed=seed, N_train=N,
                                     model=f'chmm-K{K}',
                                     alpha=None, theta=None, beta=None,
                                     horizon=h, mse=mse))
            except Exception as e:
                sys.stderr.write(f"[chmm K={K} fail {regime_name} N={N}] {e}\n")
        try:
            m = fit_alergia(train, nA, eps=ALERGIA_EPS)
            for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
                rows.append(dict(regime=regime_name, nS=nS, nA=nA,
                                 seed=seed, N_train=N,
                                 model='alergia-eps0.05',
                                 alpha=None, theta=None, beta=None,
                                 horizon=h, mse=mse))
        except Exception as e:
            sys.stderr.write(f"[alergia fail {regime_name} N={N}] {e}\n")
    return rows


def main():
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    tasks = [(name, nS, nA, conc, seed)
             for (name, nS, nA, conc) in REGIMES
             for seed in SEEDS]
    n_gdc = len(gdc_configs())
    print(f"Wider GDC grid: {n_gdc} GDC configs + 4 baselines, "
          f"{len(tasks)} cells x {len(N_TRAIN_VALUES)} N values, "
          f"{n_workers} workers", flush=True)

    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for cell_rows in pool.imap_unordered(run_cell, tasks, chunksize=1):
            all_rows.extend(cell_rows)
            done += 1
            print(f"  {done}/{len(tasks)} cells done  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    fields = ['regime', 'nS', 'nA', 'seed', 'N_train', 'model',
              'alpha', 'theta', 'beta', 'horizon', 'mse']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}  [{time.time()-t0:.1f}s, "
          f"{len(all_rows)} rows]", flush=True)


if __name__ == "__main__":
    main()
