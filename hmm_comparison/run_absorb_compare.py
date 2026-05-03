"""Re-run the sample-efficiency sweep with both terminal_behavior options
to measure the impact of absorb mode on HMM forecasting MSE.

Same setup as run_sample_efficiency.py — 9x9 (nS, nA) grid, 3 seeds,
N_train ∈ {10, 25, 50, 100, 200}.  For each cell, run:
  gdc-baseline-diffuse  : alpha=0.7, theta=0.2, beta=0.1, diffuse
  gdc-baseline-absorb   : same config, absorb
  gdc-tuned-diffuse     : alpha=0.5, theta=0.05, beta=0.2, diffuse
  gdc-tuned-absorb      : same config, absorb

Reports mean MSE per (model, N, horizon).
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GRID = [(nS, nA) for nS in range(2, 11) for nA in range(2, 11)]
SEEDS = [0, 1, 2]
N_TRAIN_VALUES = [10, 25, 50, 100, 200]
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 5, 20]

CONFIGS = [
    dict(name='gdc-baseline', alpha=0.7,  theta=0.2,   beta=0.1),
    dict(name='gdc-tuned',    alpha=0.5,  theta=0.05,  beta=0.2),
]

OUT_CSV = os.path.join(HERE, 'absorb_compare_results.csv')


def run_cell(args):
    nS, nA, seed = args
    sys.path.insert(0, HERE)
    from random_hmm import random_dense_hmm
    from model_wrappers import fit_gdc
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
        for cfg in CONFIGS:
            for tb in ['diffuse', 'absorb']:
                kwargs = {k: v for k, v in cfg.items() if k != 'name'}
                m = fit_gdc(train, nA, **kwargs,
                            transition_type='self_loop',
                            initial_dist='sequence_starts',
                            terminal_behavior=tb)
                res = mse_at_horizons(m, hmm, test_pf, HORIZONS)
                model_name = f"{cfg['name']}-{tb}"
                for h in HORIZONS:
                    rows.append(dict(nS=nS, nA=nA, seed=seed, N_train=N,
                                     model=model_name, horizon=h,
                                     mse=res[h]))
    return rows


def main():
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"Absorb-vs-diffuse sweep: {len(GRID) * len(SEEDS)} cells, "
          f"{n_workers} workers", flush=True)
    tasks = [(nS, nA, s) for (nS, nA) in GRID for s in SEEDS]
    all_rows = []
    t0 = time.time(); done = 0
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_cell, tasks, chunksize=1):
            all_rows.extend(r); done += 1
            if done % 30 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} [{time.time()-t0:.0f}s]",
                      flush=True)
    fields = ['nS', 'nA', 'seed', 'N_train', 'model', 'horizon', 'mse']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote {OUT_CSV}", flush=True)

    from collections import defaultdict
    print("\n=== Mean MSE per (model, N, horizon) ===")
    by = defaultdict(list)
    for r in all_rows:
        by[(r['model'], r['N_train'], r['horizon'])].append(r['mse'])
    for h in HORIZONS:
        print(f"\nhorizon h={h}")
        print(f"  {'model':>30s}  " + "  ".join(f'N={n:>3d}' for n in N_TRAIN_VALUES))
        for cfg in CONFIGS:
            for tb in ['diffuse', 'absorb']:
                m = f"{cfg['name']}-{tb}"
                row = "  ".join(f"{np.mean(by[(m, n, h)]):.5f}" for n in N_TRAIN_VALUES)
                print(f"  {m:>30s}  {row}")

    # Per-cell winner counts at h=1 (diffuse vs absorb per config)
    print("\n=== Win counts diffuse vs absorb at h=1 ===")
    for cfg in CONFIGS:
        for N in N_TRAIN_VALUES:
            d = defaultdict(list); a = defaultdict(list)
            for r in all_rows:
                if r['horizon'] != 1 or r['N_train'] != N: continue
                key = (r['nS'], r['nA'], r['seed'])
                if r['model'] == f"{cfg['name']}-diffuse":
                    d[key].append(r['mse'])
                elif r['model'] == f"{cfg['name']}-absorb":
                    a[key].append(r['mse'])
            d_wins = a_wins = ties = 0
            for key in d.keys() | a.keys():
                dv = np.mean(d[key]) if d[key] else float('inf')
                av = np.mean(a[key]) if a[key] else float('inf')
                if abs(dv - av) < 1e-9: ties += 1
                elif dv < av: d_wins += 1
                else: a_wins += 1
            total = d_wins + a_wins + ties
            print(f"  {cfg['name']} N={N:>3d}: diffuse {d_wins}/{total}, "
                  f"absorb {a_wins}/{total}, tied {ties}/{total}")


if __name__ == "__main__":
    main()
