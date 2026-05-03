"""Sample-efficiency sweep on HMM forecasting (multiprocessing).

For each (nS, nA) on the 9x9 grid, 3 seeds, and N_train in
{10, 25, 50, 100, 200}, fit four models and report MSE at horizons
{1, 5, 20}:

    gdc-baseline  : alpha=0.7, theta=0.2, beta=0.1
    gdc-tuned     : alpha=0.5, theta=0.05, beta=0.2  (HMM-forecast best)
    chmm-K4       : K=4 clones per emission, EM 50 iters
    alergia       : eps=0.05

All 200 training sequences are generated once per (nS, nA, seed) cell,
then the model is trained on the first N_train of them. Test prefixes
are also generated once and shared across all (N_train, model) runs in
that cell.

Cells are distributed across worker processes via Pool.imap_unordered.

Output:
    sample_efficiency_results.csv   (one row per (cell, N_train, model, horizon))
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# These are imported lazily inside worker tasks so the module can be
# pickled across processes on Windows.
GRID = [(nS, nA) for nS in range(2, 11) for nA in range(2, 11)]
SEEDS = [0, 1, 2]
N_TRAIN_VALUES = [10, 25, 50, 100, 200]
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 5, 20]

GDC_BASELINE = dict(alpha=0.7, theta=0.2, gamma=0.0, beta=0.1,
                    transition_type='self_loop',
                    initial_dist='sequence_starts')
GDC_TUNED = dict(alpha=0.5, theta=0.05, gamma=0.0, beta=0.2,
                 transition_type='self_loop',
                 initial_dist='sequence_starts')

OUT_CSV = os.path.join(HERE, 'sample_efficiency_results.csv')


def run_cell(args):
    """Worker for one (nS, nA, seed) cell.  Returns a list of result dicts.
    """
    nS, nA, seed = args
    sys.path.insert(0, HERE)
    from random_hmm import random_dense_hmm
    from model_wrappers import fit_gdc
    from chmm_alergia_wrappers import fit_chmm, fit_alergia
    from evaluation import mse_at_horizons

    rng = np.random.default_rng(11000 + seed * 97 + nS * 13 + nA)
    hmm = random_dense_hmm(nS, nA, rng)
    full_train = [hmm.sample(TRAIN_LEN, rng)[1]
                  for _ in range(max(N_TRAIN_VALUES))]
    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]

    rows = []
    for N in N_TRAIN_VALUES:
        train = full_train[:N]
        # GDC baseline
        m = fit_gdc(train, nA, **GDC_BASELINE)
        for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
            rows.append(dict(nS=nS, nA=nA, seed=seed, N_train=N,
                             model='gdc-baseline', horizon=h, mse=mse))
        # GDC tuned
        m = fit_gdc(train, nA, **GDC_TUNED)
        for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
            rows.append(dict(nS=nS, nA=nA, seed=seed, N_train=N,
                             model='gdc-tuned', horizon=h, mse=mse))
        # CHMM K=4
        try:
            m = fit_chmm(train, nA, K=4, n_em_iters=50)
            for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
                rows.append(dict(nS=nS, nA=nA, seed=seed, N_train=N,
                                 model='chmm-K4', horizon=h, mse=mse))
        except Exception as e:
            sys.stderr.write(f"[chmm fail nS={nS} nA={nA} N={N}] {e}\n")
        # ALERGIA eps=0.05
        try:
            m = fit_alergia(train, nA, eps=0.05)
            for h, mse in mse_at_horizons(m, hmm, test_pf, HORIZONS).items():
                rows.append(dict(nS=nS, nA=nA, seed=seed, N_train=N,
                                 model='alergia-eps0.05', horizon=h, mse=mse))
        except Exception as e:
            sys.stderr.write(f"[alergia fail nS={nS} nA={nA} N={N}] {e}\n")
    return rows


def main():
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"Sample-efficiency sweep across {len(GRID)*len(SEEDS)} cells "
          f"using {n_workers} worker processes", flush=True)
    tasks = [(nS, nA, seed) for (nS, nA) in GRID for seed in SEEDS]

    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for cell_rows in pool.imap_unordered(run_cell, tasks, chunksize=1):
            all_rows.extend(cell_rows)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} cells done  "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    fields = ['nS', 'nA', 'seed', 'N_train', 'model', 'horizon', 'mse']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}  [{time.time()-t0:.1f}s, "
          f"{len(all_rows)} rows]", flush=True)


if __name__ == "__main__":
    main()
