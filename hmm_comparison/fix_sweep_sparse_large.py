"""Sweep candidate fixes on the sparse_large failure cell.

The diagnostic showed GDC's posterior is far too diffuse (entropy 0.83
bits vs true 0.01). The val-picked config has alpha=0.8, theta=0.001,
beta=0.1 — so 1 - alpha - theta ≈ 0.20 of mass diffuses uniformly per
step, smearing across all 80,000 chain positions. The val-tune grid
caps alpha at 0.80 and beta at 0.20; this sweep tests whether values
outside that grid close the gap to CHMM.

Sweeps:
  alpha ∈ {0.80, 0.90, 0.95, 0.99, 0.999}
  beta  ∈ {0.0, 0.05, 0.1, 0.2}
  theta = 0.001 fixed (val-picked value)

Reports excess perplexity for each combo, with CHMM K=32 as reference.
"""
from __future__ import annotations
import os, sys, time, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from random_hmm import random_sparse_topology_hmm
from model_wrappers import GDCForecaster
from chmm_alergia_wrappers import CHMMForecaster
from evaluation import perplexity_at_horizons

# Same cell + sampling protocol as the seq-len sweep
nS, nA = 30, 8
fanout = 2
E_concentration = 0.1
TRAIN_LEN = 200
N_TRAIN = 400
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20

ALPHAS = [0.80, 0.90, 0.95, 0.99, 0.999]
BETAS  = [0.0, 0.05, 0.10, 0.20]
THETA  = 0.001

OUT_CSV = os.path.join(HERE, 'fix_sweep_sparse_large.csv')


def run_seed(seed: int):
    seed_offset = 2  # sparse
    rng = np.random.default_rng(60000 + seed * 137 + nS * 7 + nA * 11
                                + seed_offset)
    hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                     E_concentration=E_concentration)
    train = [hmm.sample(TRAIN_LEN, rng)[1] for _ in range(N_TRAIN)]
    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]

    rows = []
    # GDC sweep
    for alpha in ALPHAS:
        for beta in BETAS:
            t0 = time.time()
            gdc = GDCForecaster(nA, alpha=alpha, theta=THETA, gamma=0.0,
                                beta=beta, transition_type='self_loop',
                                initial_dist='sequence_starts')
            gdc.fit(train)
            ppl = perplexity_at_horizons(gdc, hmm, test_pf, [1])
            r = ppl[1]
            rows.append(dict(seed=seed, model='gdc', alpha=alpha, theta=THETA,
                             beta=beta,
                             excess_pp=r['excess_perplexity'],
                             elapsed_s=time.time()-t0))

    # CHMM K=32 reference
    t0 = time.time()
    chmm = CHMMForecaster(nA, K=32, n_em_iters=50, seed=seed)
    chmm.fit(train)
    ppl = perplexity_at_horizons(chmm, hmm, test_pf, [1])
    r = ppl[1]
    rows.append(dict(seed=seed, model='chmm', alpha=None, theta=None,
                     beta=None, excess_pp=r['excess_perplexity'],
                     elapsed_s=time.time()-t0))
    return rows


def main():
    all_rows = []
    for seed in [0, 1, 2]:  # test seeds only, mirrors final reporting
        print(f"=== seed {seed} ===", flush=True)
        rs = run_seed(seed)
        all_rows.extend(rs)
        for r in rs:
            print(f"  {r['model']:>4s}  "
                  f"alpha={r['alpha'] if r['alpha'] is not None else '   ':<6}  "
                  f"beta={r['beta'] if r['beta'] is not None else '  ':<5}  "
                  f"excess_pp={r['excess_pp']:.4f}  "
                  f"({r['elapsed_s']:.1f}s)", flush=True)

    fields = ['seed', 'model', 'alpha', 'theta', 'beta', 'excess_pp',
              'elapsed_s']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}")

    # Summary: best GDC alpha/beta combo per seed, mean across seeds
    import pandas as pd
    df = pd.DataFrame(all_rows)
    gdc = df[df.model == 'gdc']
    chmm = df[df.model == 'chmm']
    print("\n## Mean excess_pp across seeds (lower = better)\n")
    pivot = (gdc.groupby(['alpha','beta'])['excess_pp'].mean()
                .unstack().round(4))
    print("Rows = alpha; cols = beta; CHMM K=32 mean = "
          f"{chmm.excess_pp.mean():.4f}")
    print(pivot)


if __name__ == "__main__":
    main()
