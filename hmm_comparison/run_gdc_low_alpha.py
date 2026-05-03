"""GDC sweep pushing alpha down and beta up — find the true optimum.

Earlier sweeps had α=0.5 winning on every regime, hitting the grid's
lower bound. β=0.3 also won on many regimes at the upper bound. This
sweep extends both boundaries:

    alpha in {0.1, 0.2, 0.3, 0.4, 0.5, 0.7}
    theta in {0.0, 0.001, 0.005, 0.01, 0.05}
    beta  in {0.2, 0.3, 0.4, 0.5}
    transition: self_loop, alpha + theta <= 1

Same 4 regimes x 2 N values x 3 seeds.
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
from itertools import product
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
THETAS = [0.0, 0.001, 0.005, 0.01, 0.05]
BETAS = [0.2, 0.3, 0.4, 0.5]

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

OUT_CSV = os.path.join(HERE, 'gdc_low_alpha_results.csv')


def gdc_configs():
    return [(a, t, b) for a, t, b in product(ALPHAS, THETAS, BETAS)
            if a + t <= 1.0]


def run_cell(args):
    regime_name, nS, nA, E_conc, seed = args
    sys.path.insert(0, HERE)
    from random_hmm import random_dense_hmm
    from model_wrappers import fit_gdc
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
                                 alpha=a, theta=t, beta=b,
                                 horizon=h, mse=mse))
    return rows


def main():
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    tasks = [(name, nS, nA, conc, seed)
             for (name, nS, nA, conc) in REGIMES
             for seed in SEEDS]
    n_gdc = len(gdc_configs())
    print(f"Low-alpha GDC grid: {n_gdc} configs, {len(tasks)} cells, "
          f"{len(N_TRAIN_VALUES)} N values, {n_workers} workers", flush=True)

    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for cell_rows in pool.imap_unordered(run_cell, tasks, chunksize=1):
            all_rows.extend(cell_rows)
            done += 1
            print(f"  {done}/{len(tasks)} cells done  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    fields = ['regime', 'nS', 'nA', 'seed', 'N_train',
              'alpha', 'theta', 'beta', 'horizon', 'mse']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}  [{time.time()-t0:.1f}s, "
          f"{len(all_rows)} rows]", flush=True)


if __name__ == "__main__":
    main()
