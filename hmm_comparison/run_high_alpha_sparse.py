"""High-alpha GDC refinement on the sparse regimes.

Earlier sparse_large sweep showed alpha=0.7 winning at the upper bound
of the tested grid, with monotonic improvement as alpha grew. Push
the alpha grid higher to find GDC's true ceiling on sparse data, and
also test self_loop_two_step transitions (never tested on this metric).

    alpha       in {0.5, 0.7, 0.8, 0.85, 0.9, 0.95}
    theta       in {0.001, 0.01, 0.05}
    beta        in {0.1, 0.2, 0.3}
    transition  in {self_loop, self_loop_two_step}
    constraint  : alpha + theta <= 1

Regimes: sparse_small, sparse_large
N_train: {25, 100, 400}
Seeds: 3
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
from itertools import product
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ALPHAS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]
THETAS = [0.001, 0.01, 0.05]
BETAS = [0.1, 0.2, 0.3]
TRANSITIONS = ['self_loop', 'self_loop_two_step']

REGIMES = [
    ('sparse_small', 10, 4, 0.1, 2),
    ('sparse_large', 30, 8, 0.1, 2),
]
SEEDS = [0, 1, 2]
N_TRAIN_VALUES = [25, 100, 400]
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 5]

CHMM_KS = [4, 16, 32]
ALERGIA_EPS = 0.05

OUT_CSV = os.path.join(HERE, 'high_alpha_sparse_results.csv')


def gdc_configs():
    return [(a, t, b, tt) for a, t, b, tt in product(
        ALPHAS, THETAS, BETAS, TRANSITIONS) if a + t <= 1.0]


def run_cell(args):
    regime_name, nS, nA, E_conc, fanout, seed = args
    sys.path.insert(0, HERE)
    from random_hmm import random_sparse_topology_hmm
    from model_wrappers import fit_gdc
    from chmm_alergia_wrappers import fit_chmm, fit_alergia
    from evaluation import mse_at_horizons

    rng = np.random.default_rng(40000 + seed * 137 + nS * 7 + nA * 11 + 2)
    hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                     E_concentration=E_conc)
    full_train = [hmm.sample(TRAIN_LEN, rng)[1]
                  for _ in range(max(N_TRAIN_VALUES))]
    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]

    rows = []
    for N in N_TRAIN_VALUES:
        train = full_train[:N]
        for (a, t, b, tt) in gdc_configs():
            m = fit_gdc(train, nA, alpha=a, theta=t, gamma=0.0, beta=b,
                        transition_type=tt,
                        initial_dist='sequence_starts')
            for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
                rows.append(dict(regime=regime_name, nS=nS, nA=nA,
                                 seed=seed, N_train=N,
                                 alpha=a, theta=t, beta=b, trans=tt,
                                 model=f'gdc-a{a}-t{t}-b{b}-{tt}',
                                 horizon=h, mse=mse))
        # Reference baselines
        for K in CHMM_KS:
            try:
                m = fit_chmm(train, nA, K=K, n_em_iters=50)
                for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
                    rows.append(dict(regime=regime_name, nS=nS, nA=nA,
                                     seed=seed, N_train=N,
                                     alpha=None, theta=None, beta=None,
                                     trans=None,
                                     model=f'chmm-K{K}',
                                     horizon=h, mse=mse))
            except Exception as e:
                sys.stderr.write(f"[chmm K={K} fail {regime_name} N={N}] {e}\n")
        try:
            m = fit_alergia(train, nA, eps=ALERGIA_EPS)
            for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
                rows.append(dict(regime=regime_name, nS=nS, nA=nA,
                                 seed=seed, N_train=N,
                                 alpha=None, theta=None, beta=None,
                                 trans=None,
                                 model='alergia-eps0.05',
                                 horizon=h, mse=mse))
        except Exception as e:
            sys.stderr.write(f"[alergia fail {regime_name} N={N}] {e}\n")
    return rows


def main():
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    tasks = [(name, nS, nA, conc, fanout, seed)
             for (name, nS, nA, conc, fanout) in REGIMES
             for seed in SEEDS]
    n_gdc = len(gdc_configs())
    print(f"High-alpha sparse sweep: {n_gdc} GDC configs, "
          f"{len(tasks)} cells, {len(N_TRAIN_VALUES)} N values, "
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
              'alpha', 'theta', 'beta', 'trans', 'horizon', 'mse']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}  [{time.time()-t0:.1f}s, "
          f"{len(all_rows)} rows]", flush=True)


if __name__ == "__main__":
    main()
