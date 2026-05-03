"""Large + more-deterministic HMM sample-efficiency sweep.

Two axes of interest:

  1. larger nS (10, 20, 30, 50) paired with proportionally larger nA
  2. emission determinism: dense (E_concentration=1.0) vs det_em
     (E_concentration=0.1; near-deterministic emissions)

Models tested at every (cell, N_train) combo:
  - gdc-tuned : alpha=0.5, theta=0.05, beta=0.2 (HMM-forecast best)
  - chmm-K   : K in {4, 8, 16, 32}
  - alergia  : eps=0.05

Multiprocessed across (size_idx, regime, seed) cells.
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# (nS, nA) pairs scaled up
SIZE_GRID = [(10, 4), (20, 6), (30, 8), (50, 10)]
REGIMES = [
    ('dense', 1.0),    # default Dirichlet emissions
    ('det_em', 0.1),   # near-deterministic emissions
]
SEEDS = [0, 1, 2]
N_TRAIN_VALUES = [25, 50, 100, 200, 400]
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 5, 20]

CHMM_KS = [4, 8, 16, 32]
GDC_TUNED = dict(alpha=0.5, theta=0.05, gamma=0.0, beta=0.2,
                 transition_type='self_loop',
                 initial_dist='sequence_starts')
ALERGIA_EPS = 0.05

OUT_CSV = os.path.join(HERE, 'large_det_sweep_results.csv')


def run_cell(args):
    """Worker for one (nS, nA, regime_name, E_concentration, seed) cell."""
    nS, nA, regime_name, E_conc, seed = args
    sys.path.insert(0, HERE)
    from random_hmm import random_dense_hmm
    from model_wrappers import fit_gdc
    from chmm_alergia_wrappers import fit_chmm, fit_alergia
    from evaluation import mse_at_horizons

    rng = np.random.default_rng(
        20000 + seed * 113 + nS * 7 + nA * 11 + (1 if regime_name == 'det_em' else 0))
    hmm = random_dense_hmm(nS, nA, rng,
                           T_concentration=1.0, E_concentration=E_conc)
    full_train = [hmm.sample(TRAIN_LEN, rng)[1]
                  for _ in range(max(N_TRAIN_VALUES))]
    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]

    rows = []
    for N in N_TRAIN_VALUES:
        train = full_train[:N]
        # GDC tuned
        m = fit_gdc(train, nA, **GDC_TUNED)
        for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
            rows.append(dict(nS=nS, nA=nA, regime=regime_name,
                             seed=seed, N_train=N,
                             model='gdc-tuned', horizon=h, mse=mse))
        # CHMM at multiple K
        for K in CHMM_KS:
            try:
                m = fit_chmm(train, nA, K=K, n_em_iters=50)
                for h, mse in mse_at_horizons(m, hmm, test_pf,
                                              HORIZONS).items():
                    rows.append(dict(nS=nS, nA=nA, regime=regime_name,
                                     seed=seed, N_train=N,
                                     model=f'chmm-K{K}',
                                     horizon=h, mse=mse))
            except Exception as e:
                sys.stderr.write(
                    f"[chmm K={K} fail nS={nS} nA={nA} N={N} {regime_name}] {e}\n")
        # ALERGIA
        try:
            m = fit_alergia(train, nA, eps=ALERGIA_EPS)
            for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
                rows.append(dict(nS=nS, nA=nA, regime=regime_name,
                                 seed=seed, N_train=N,
                                 model='alergia-eps0.05',
                                 horizon=h, mse=mse))
        except Exception as e:
            sys.stderr.write(
                f"[alergia fail nS={nS} nA={nA} N={N} {regime_name}] {e}\n")
    return rows


def main():
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    tasks = [(nS, nA, name, conc, seed)
             for (nS, nA) in SIZE_GRID
             for (name, conc) in REGIMES
             for seed in SEEDS]
    print(f"Large-deterministic sweep: {len(tasks)} cells, "
          f"{len(N_TRAIN_VALUES)} N values, {1+len(CHMM_KS)+1} models, "
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

    fields = ['nS', 'nA', 'regime', 'seed', 'N_train', 'model',
              'horizon', 'mse']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}  [{time.time()-t0:.1f}s, "
          f"{len(all_rows)} rows]", flush=True)


if __name__ == "__main__":
    main()
