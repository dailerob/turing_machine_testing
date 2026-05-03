"""Absorb vs diffuse comparison across all 6 HMM regimes.

Mirrors the structure of run_large_det_sweep.py + run_sparse_sweep.py,
but tests both terminal_behavior options for GDC.

Regimes:
  dense_small   nS=10  nA=4  T=Dirichlet(1.0)  E=Dirichlet(1.0)
  dense_large   nS=30  nA=8  T=Dirichlet(1.0)  E=Dirichlet(1.0)
  det_small     nS=10  nA=4  T=Dirichlet(1.0)  E=Dirichlet(0.1)
  det_large     nS=30  nA=8  T=Dirichlet(1.0)  E=Dirichlet(0.1)
  sparse_small  nS=10  nA=4  T=fanout-2        E=Dirichlet(0.1)
  sparse_large  nS=30  nA=8  T=fanout-2        E=Dirichlet(0.1)

GDC configs (per-regime tuned values from earlier sweeps):
  dense_small  : alpha=0.5, theta=0.05, beta=0.2
  dense_large  : alpha=0.1, theta=0.001, beta=0.2
  det_small    : alpha=0.7, theta=0.01, beta=0.2
  det_large    : alpha=0.3, theta=0.005, beta=0.3
  sparse_small : alpha=0.8, theta=0.0, beta=0.05  (per high_alpha sweep)
  sparse_large : alpha=0.8, theta=0.0, beta=0.2   (per high_alpha sweep)
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REGIMES = [
    ('dense_small',  10, 4, 'dense',  1.0, None,
     dict(alpha=0.5, theta=0.05, beta=0.2)),
    ('dense_large',  30, 8, 'dense',  1.0, None,
     dict(alpha=0.1, theta=0.001, beta=0.2)),
    ('det_small',    10, 4, 'dense',  0.1, None,
     dict(alpha=0.7, theta=0.01, beta=0.2)),
    ('det_large',    30, 8, 'dense',  0.1, None,
     dict(alpha=0.3, theta=0.005, beta=0.3)),
    ('sparse_small', 10, 4, 'sparse', 0.1, 2,
     dict(alpha=0.8, theta=0.0, beta=0.05)),
    ('sparse_large', 30, 8, 'sparse', 0.1, 2,
     dict(alpha=0.8, theta=0.0, beta=0.2)),
]
SEEDS = [0, 1, 2]
N_TRAIN_VALUES = [25, 100, 400]
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 5, 20]

OUT_CSV = os.path.join(HERE, 'absorb_compare_regimes_results.csv')


def run_cell(args):
    name, nS, nA, kind, E_conc, fanout, gdc_cfg, seed = args
    sys.path.insert(0, HERE)
    from random_hmm import random_dense_hmm, random_sparse_topology_hmm
    from model_wrappers import fit_gdc
    from evaluation import mse_at_horizons, perplexity_at_horizons

    seed_offset = (1 if 'det' in name else 0) \
                  + (2 if 'sparse' in name else 0)
    rng = np.random.default_rng(80000 + seed * 137 + nS * 7 + nA * 11
                                + seed_offset)
    if kind == 'sparse':
        hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                         E_concentration=E_conc)
    else:
        hmm = random_dense_hmm(nS, nA, rng,
                               T_concentration=1.0, E_concentration=E_conc)
    full_train = [hmm.sample(TRAIN_LEN, rng)[1]
                  for _ in range(max(N_TRAIN_VALUES))]
    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]

    rows = []
    for N in N_TRAIN_VALUES:
        train = full_train[:N]
        for tb in ['diffuse', 'absorb']:
            m = fit_gdc(train, nA, **gdc_cfg,
                        transition_type='self_loop',
                        initial_dist='sequence_starts',
                        terminal_behavior=tb)
            mse_res = mse_at_horizons(m, hmm, test_pf, HORIZONS)
            ppl_res = perplexity_at_horizons(m, hmm, test_pf, HORIZONS)
            for h in HORIZONS:
                rows.append(dict(regime=name, nS=nS, nA=nA, seed=seed,
                                 N_train=N, mode=tb, horizon=h,
                                 mse=mse_res[h],
                                 excess_perplexity=ppl_res[h]['excess_perplexity'],
                                 perplexity=ppl_res[h]['perplexity'],
                                 entropy_floor_perplexity=ppl_res[h]['entropy_floor_perplexity'],
                                 **gdc_cfg))
    return rows


def main():
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    tasks = [(name, nS, nA, kind, conc, fanout, cfg, seed)
             for (name, nS, nA, kind, conc, fanout, cfg) in REGIMES
             for seed in SEEDS]
    print(f"Absorb-compare on {len(REGIMES)} regimes x 3 seeds = "
          f"{len(tasks)} cells, {n_workers} workers", flush=True)
    all_rows = []
    t0 = time.time(); done = 0
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_cell, tasks, chunksize=1):
            all_rows.extend(r); done += 1
            print(f"  {done}/{len(tasks)} [{time.time()-t0:.0f}s]", flush=True)

    fields = ['regime', 'nS', 'nA', 'seed', 'N_train', 'mode',
              'horizon', 'mse', 'excess_perplexity', 'perplexity',
              'entropy_floor_perplexity', 'alpha', 'theta', 'beta']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote {OUT_CSV}", flush=True)

    from collections import defaultdict
    for metric_name, metric_key, fmt in [('MSE', 'mse', '{:.5f}'),
                                          ('excess perplexity',
                                           'excess_perplexity', '{:.4f}')]:
        print(f"\n=== Mean {metric_name} per (regime, N, mode) at h=1 ===")
        print(f"{'regime':>14s}  {'N':>4s}  {'diffuse':>10s}  {'absorb':>10s}  "
              f"{'absorb/diffuse':>16s}  {'winner':>10s}")
        by = defaultdict(list)
        for r in all_rows:
            if r['horizon'] != 1: continue
            by[(r['regime'], r['N_train'], r['mode'])].append(r[metric_key])
        for regime in [r[0] for r in REGIMES]:
            for N in N_TRAIN_VALUES:
                d = float(np.mean(by[(regime, N, 'diffuse')]))
                a = float(np.mean(by[(regime, N, 'absorb')]))
                ratio = a / d if d > 0 else float('nan')
                winner = 'diffuse' if d < a else ('absorb' if a < d else 'tied')
                d_str = fmt.format(d); a_str = fmt.format(a)
                print(f"{regime:>14s}  {N:>4d}  {d_str:>10s}  {a_str:>10s}  "
                      f"{ratio:>16.3f}  {winner:>10s}")

    # Per-cell head-to-head winner counts
    print("\n=== Per-cell winners (h=1, 3 seeds per (regime, N)) ===")
    print(f"{'regime':>14s}  {'N':>4s}  {'diffuse':>9s}  {'absorb':>9s}  {'tied':>5s}")
    by_cell = defaultdict(dict)
    for r in all_rows:
        if r['horizon'] != 1: continue
        by_cell[(r['regime'], r['N_train'], r['seed'])][r['mode']] = r['mse']
    for regime in [r[0] for r in REGIMES]:
        for N in N_TRAIN_VALUES:
            d_wins = a_wins = ties = 0
            for seed in SEEDS:
                cell = by_cell[(regime, N, seed)]
                if not cell: continue
                d, a = cell['diffuse'], cell['absorb']
                if abs(d - a) < 1e-12: ties += 1
                elif d < a: d_wins += 1
                else: a_wins += 1
            print(f"{regime:>14s}  {N:>4d}  {d_wins:>9d}  {a_wins:>9d}  {ties:>5d}")


if __name__ == "__main__":
    main()
