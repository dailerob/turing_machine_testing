"""Add a sparse-transition + sparse-emission regime to the comparison.

Regimes (6 total):
    dense_small   nS=10  nA=4  T=Dirichlet(1.0)  E=Dirichlet(1.0)
    dense_large   nS=30  nA=8  T=Dirichlet(1.0)  E=Dirichlet(1.0)
    det_small     nS=10  nA=4  T=Dirichlet(1.0)  E=Dirichlet(0.1)
    det_large     nS=30  nA=8  T=Dirichlet(1.0)  E=Dirichlet(0.1)
    sparse_small  nS=10  nA=4  T=fanout-2        E=Dirichlet(0.1)
    sparse_large  nS=30  nA=8  T=fanout-2        E=Dirichlet(0.1)

For each regime test a small but reasonable GDC config grid (so we
do not pre-commit to the dense-regime optimum we found earlier) plus
ALERGIA + CHMM K in {4, 8, 16, 32}.

N_train sweep: {25, 50, 100, 200, 400}, 3 seeds.
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
from itertools import product
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# (name, nS, nA, kind, E_conc, fanout)
REGIMES = [
    ('dense_small',  10, 4, 'dense',  1.0, None),
    ('dense_large',  30, 8, 'dense',  1.0, None),
    ('det_small',    10, 4, 'dense',  0.1, None),
    ('det_large',    30, 8, 'dense',  0.1, None),
    ('sparse_small', 10, 4, 'sparse', 0.1, 2),
    ('sparse_large', 30, 8, 'sparse', 0.1, 2),
]
SEEDS = [0, 1, 2]
N_TRAIN_VALUES = [25, 50, 100, 200, 400]
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 5]

# Sparse regimes have non-uniform transitions, so we want both
# sharper (high alpha) and softer configs in the grid.
GDC_CONFIGS = [
    (0.1, 0.001, 0.20),   # dense-large optimum
    (0.3, 0.005, 0.30),   # dense-small low-N optimum
    (0.5, 0.005, 0.20),   # midline
    (0.7, 0.010, 0.20),   # det-small high-N optimum
    (0.5, 0.05,  0.20),   # original 'tuned'
]
CHMM_KS = [4, 8, 16, 32]
ALERGIA_EPS = 0.05

OUT_CSV = os.path.join(HERE, 'sparse_sweep_results.csv')


def run_cell(args):
    regime_name, nS, nA, kind, E_conc, fanout, seed = args
    sys.path.insert(0, HERE)
    from random_hmm import random_dense_hmm, random_sparse_topology_hmm
    from model_wrappers import fit_gdc
    from chmm_alergia_wrappers import fit_chmm, fit_alergia
    from evaluation import mse_at_horizons

    seed_offset = (1 if 'det' in regime_name else 0) \
                  + (2 if 'sparse' in regime_name else 0)
    rng = np.random.default_rng(40000 + seed * 137 + nS * 7 + nA * 11
                                + seed_offset)
    if kind == 'sparse':
        hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                         E_concentration=E_conc)
    else:
        hmm = random_dense_hmm(nS, nA, rng,
                               T_concentration=1.0,
                               E_concentration=E_conc)
    full_train = [hmm.sample(TRAIN_LEN, rng)[1]
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
                rows.append(dict(regime=regime_name, nS=nS, nA=nA,
                                 seed=seed, N_train=N,
                                 model=f'gdc-a{a}-t{t}-b{b}',
                                 horizon=h, mse=mse))
        for K in CHMM_KS:
            try:
                m = fit_chmm(train, nA, K=K, n_em_iters=50)
                for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
                    rows.append(dict(regime=regime_name, nS=nS, nA=nA,
                                     seed=seed, N_train=N,
                                     model=f'chmm-K{K}',
                                     horizon=h, mse=mse))
            except Exception as e:
                sys.stderr.write(f"[chmm K={K} fail {regime_name} N={N}] {e}\n")
        try:
            m = fit_alergia(train, nA, eps=ALERGIA_EPS)
            for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
                rows.append(dict(regime=regime_name, nS=nS, nA=nA,
                                 seed=seed, N_train=N,
                                 model='alergia-eps0.05',
                                 horizon=h, mse=mse))
        except Exception as e:
            sys.stderr.write(f"[alergia fail {regime_name} N={N}] {e}\n")
    return rows


def main():
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    tasks = [(name, nS, nA, kind, conc, fanout, seed)
             for (name, nS, nA, kind, conc, fanout) in REGIMES
             for seed in SEEDS]
    print(f"Sparse-regime sweep: {len(REGIMES)} regimes x 3 seeds = "
          f"{len(tasks)} cells, {n_workers} workers", flush=True)

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
              'horizon', 'mse']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}  [{time.time()-t0:.1f}s, "
          f"{len(all_rows)} rows]", flush=True)


if __name__ == "__main__":
    main()
