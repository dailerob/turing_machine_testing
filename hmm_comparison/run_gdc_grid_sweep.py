"""GDC hyperparameter sweep on HMM forecasting.

Phase 1: small (nS, nA) subset sweep across an 18-config grid.
    alpha   in {0.5, 0.7, 0.9}
    theta   in {0.05, 0.2}
    beta    in {0.05, 0.1, 0.2}
    transition: 'self_loop' only

Phase 2: pick top-3 configs by mean MSE-at-h=1 from phase 1, run on
the full 9x9x3 grid.

Writes:
    gdc_grid_phase1.csv   (small-grid, all 18 configs)
    gdc_grid_phase2.csv   (full grid, best 3 configs)
"""
from __future__ import annotations
import os, sys, csv, time
from collections import defaultdict
from itertools import product
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from random_hmm import random_dense_hmm  # noqa: E402
from model_wrappers import fit_gdc  # noqa: E402
from evaluation import mse_at_horizons  # noqa: E402

ALPHAS = [0.5, 0.7, 0.9]
THETAS = [0.05, 0.2]
BETAS = [0.05, 0.1, 0.2]
TRANSITION = 'self_loop'

PHASE1_GRID = [(nS, nA) for nS in (3, 5, 7) for nA in (3, 5, 7)]
PHASE2_GRID = [(nS, nA) for nS in range(2, 11) for nA in range(2, 11)]
SEEDS = [0, 1, 2]
N_TRAIN_SEQ = 200
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 2, 5, 10, 20]


def make_data(nS, nA, seed):
    rng = np.random.default_rng(11000 + seed * 97 + nS * 13 + nA)
    hmm = random_dense_hmm(nS, nA, rng)
    train = [hmm.sample(TRAIN_LEN, rng)[1] for _ in range(N_TRAIN_SEQ)]
    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]
    return hmm, train, test_pf


def run_phase(grid, configs, label, out_csv):
    rows = []
    t0 = time.time()
    for (nS, nA) in grid:
        for seed in SEEDS:
            hmm, train, test_pf = make_data(nS, nA, seed)
            for cfg in configs:
                gdc = fit_gdc(train, nA,
                              alpha=cfg['alpha'], theta=cfg['theta'],
                              gamma=0.0, beta=cfg['beta'],
                              transition_type=TRANSITION,
                              initial_dist='sequence_starts')
                res = mse_at_horizons(gdc, hmm, test_pf, HORIZONS)
                for h in HORIZONS:
                    rows.append(dict(nS=nS, nA=nA, seed=seed,
                                     alpha=cfg['alpha'], theta=cfg['theta'],
                                     beta=cfg['beta'],
                                     horizon=h, mse=res[h]))
            print(f"  [{label}] nS={nS} nA={nA} seed={seed}  "
                  f"[total={time.time()-t0:.0f}s]", flush=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out_csv}  [{time.time()-t0:.1f}s]", flush=True)
    return rows


def main():
    print("=== Phase 1: 9 configs × 9 cells × 3 seeds ===", flush=True)
    configs = [{'alpha': a, 'theta': t, 'beta': b}
               for a, t, b in product(ALPHAS, THETAS, BETAS)
               if a + t <= 1.0]
    print(f"  {len(configs)} configs", flush=True)
    rows1 = run_phase(PHASE1_GRID, configs, 'phase1',
                      os.path.join(HERE, 'gdc_grid_phase1.csv'))

    # Aggregate phase 1 by (alpha, theta, beta), gmean MSE at h=1
    by_cfg = defaultdict(list)
    for r in rows1:
        if r['horizon'] == 1:
            by_cfg[(r['alpha'], r['theta'], r['beta'])].append(r['mse'])
    cfg_scores = []
    for cfg_key, mses in by_cfg.items():
        mses_arr = np.maximum(mses, 1e-12)
        gmean = float(np.exp(np.log(mses_arr).mean()))
        cfg_scores.append((gmean, cfg_key))
    cfg_scores.sort()
    print("\n=== Phase 1 ranking (gmean MSE at h=1) ===")
    for gm, cfg in cfg_scores:
        print(f"  alpha={cfg[0]:.2f} theta={cfg[1]:.3f} beta={cfg[2]:.2f}  "
              f"gmean(h1)={gm:.5f}")

    # Top 3
    top3 = [cfg for _, cfg in cfg_scores[:3]]
    print(f"\n=== Phase 2: top-3 configs on full 9x9x3 grid ===", flush=True)
    print(f"  configs: {top3}", flush=True)
    configs2 = [{'alpha': a, 'theta': t, 'beta': b} for (a, t, b) in top3]
    rows2 = run_phase(PHASE2_GRID, configs2, 'phase2',
                      os.path.join(HERE, 'gdc_grid_phase2.csv'))


if __name__ == "__main__":
    main()
