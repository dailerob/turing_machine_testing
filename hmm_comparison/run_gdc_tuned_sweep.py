"""Single representative GDC config check on the HMM-forecasting grid.

Same setup as run_main_sweep.py and run_chmm_alergia_sweep.py (200 train
seqs of length 50, 100 test prefixes of length 20, horizons 1/2/5/10/20,
3 seeds, 9x9 (nS, nA) grid).

The baseline GDC config in run_main_sweep was:
    alpha=0.7, theta=0.2, gamma=0.0, beta=0.1, self_loop

This script tests the "PAutomaC-style sharper" config that the
algorithmic and PAutomaC sweeps found to be a strong default:
    alpha=0.95, theta=0.005, gamma=0.0, beta=0.0, self_loop

Writes to gdc_tuned_sweep_results.csv.
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from random_hmm import random_dense_hmm  # noqa: E402
from model_wrappers import fit_gdc  # noqa: E402
from evaluation import mse_at_horizons  # noqa: E402

STATE_COUNTS = range(2, 11)
ALPHABET_SIZES = range(2, 11)
SEEDS = [0, 1, 2]
N_TRAIN_SEQ = 200
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 2, 5, 10, 20]

GDC_TUNED = dict(alpha=0.95, theta=0.005, gamma=0.0, beta=0.0,
                 transition_type='self_loop',
                 initial_dist='sequence_starts')

OUT_CSV = os.path.join(HERE, "gdc_tuned_sweep_results.csv")


def main():
    rows = []
    t0 = time.time()
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
                gdc = fit_gdc(train_seqs, nA, **GDC_TUNED)
                res = mse_at_horizons(gdc, hmm, test_prefixes, HORIZONS)
                for h in HORIZONS:
                    rows.append(dict(nS=nS, nA=nA, seed=seed,
                                     model='gdc-tuned',
                                     horizon=h, mse=res[h]))
                print(f"  nS={nS} nA={nA} seed={seed}  "
                      f"[{time.time()-tic:.2f}s, total={time.time()-t0:.0f}s]",
                      flush=True)

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['nS', 'nA', 'seed', 'model',
                                          'horizon', 'mse'])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {OUT_CSV}", flush=True)
    print(f"Total: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
