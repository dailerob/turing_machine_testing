"""(nS, nA) sweep with CHMM and ALERGIA — same setup as
run_main_sweep.py but for the new models.

Uses identical settings (200 train sequences of length 50, 100 test
prefixes of length 20, horizons 1/2/5/10/20, 3 seeds per config) so
results compare directly to `main_sweep_results.csv`.

Writes to `chmm_alergia_sweep_results.csv`.
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from random_hmm import random_dense_hmm  # noqa: E402
from chmm_alergia_wrappers import fit_chmm, fit_alergia  # noqa: E402
from evaluation import (  # noqa: E402
    mse_at_horizons, uniform_baseline_mse, stationary_baseline_mse)

STATE_COUNTS = range(2, 11)
ALPHABET_SIZES = range(2, 11)
SEEDS = [0, 1, 2]
N_TRAIN_SEQ = 200
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 2, 5, 10, 20]

OUT_CSV = os.path.join(HERE, "chmm_alergia_sweep_results.csv")


def log(*a):
    print(*a, flush=True)


def main():
    rows = []
    t_global = time.time()
    for nS in STATE_COUNTS:
        for nA in ALPHABET_SIZES:
            for seed in SEEDS:
                tic = time.time()
                rng = np.random.default_rng(11000 + seed * 97 + nS * 13 + nA)
                hmm = random_dense_hmm(nS, nA, rng)
                train_seqs = [hmm.sample(TRAIN_LEN, rng)[1]
                              for _ in range(N_TRAIN_SEQ)]
                test_prefixes = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
                                 for _ in range(N_TEST_PREFIXES)]

                base_uniform = uniform_baseline_mse(hmm, test_prefixes, HORIZONS)
                base_stat = stationary_baseline_mse(hmm, test_prefixes, HORIZONS)
                for h in HORIZONS:
                    rows.append(dict(nS=nS, nA=nA, seed=seed, model='uniform',
                                     horizon=h, mse=base_uniform[h]))
                    rows.append(dict(nS=nS, nA=nA, seed=seed, model='stationary',
                                     horizon=h, mse=base_stat[h]))

                # CHMM K=4 (chosen as the algorithm-benchmark sweet spot)
                t0 = time.time()
                try:
                    chmm = fit_chmm(train_seqs, nA, K=4, n_em_iters=50)
                    chmm_t = time.time() - t0
                    res = mse_at_horizons(chmm, hmm, test_prefixes, HORIZONS)
                    for h in HORIZONS:
                        rows.append(dict(nS=nS, nA=nA, seed=seed,
                                         model='chmm-K4', horizon=h,
                                         mse=res[h], fit_s=chmm_t))
                except Exception as e:
                    log(f"  chmm fail nS={nS} nA={nA} seed={seed}: {e}")

                # ALERGIA eps=0.05
                t0 = time.time()
                try:
                    aler = fit_alergia(train_seqs, nA, eps=0.05)
                    aler_t = time.time() - t0
                    res = mse_at_horizons(aler, hmm, test_prefixes, HORIZONS)
                    for h in HORIZONS:
                        rows.append(dict(nS=nS, nA=nA, seed=seed,
                                         model='alergia-eps0.05', horizon=h,
                                         mse=res[h], fit_s=aler_t))
                except Exception as e:
                    log(f"  alergia fail nS={nS} nA={nA} seed={seed}: {e}")

                log(f"  nS={nS} nA={nA} seed={seed}  "
                    f"[{time.time()-tic:.1f}s, total={time.time()-t_global:.0f}s]")

    # Use a stable column order (some baseline rows lack 'fit_s')
    fields = ['nS', 'nA', 'seed', 'model', 'horizon', 'mse', 'fit_s']
    for r in rows:
        r.setdefault('fit_s', None)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    log(f"\nWrote {OUT_CSV}")
    log(f"Total: {time.time()-t_global:.1f}s")


if __name__ == "__main__":
    main()
