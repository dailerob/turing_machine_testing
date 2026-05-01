"""
Main (nS, nA) sweep: for every state count 2..10 and alphabet size 2..10,
generate a dense random HMM, train OOM and GDC, and measure MSE of their
horizon-h next-symbol forecasts against the HMM's exact posterior
predictive distribution.

Writes one CSV row per (nS, nA, seed, model, horizon).
"""

from __future__ import annotations

import os
import sys
import csv
import time
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from random_hmm import random_dense_hmm
from model_wrappers import fit_oom, fit_gdc
from evaluation import mse_at_horizons, uniform_baseline_mse, stationary_baseline_mse


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STATE_COUNTS    = range(2, 11)     # nS in 2..10
ALPHABET_SIZES  = range(2, 11)     # nA in 2..10
SEEDS           = [0, 1, 2]        # 3 HMM seeds per config
N_TRAIN_SEQ     = 200
TRAIN_LEN       = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS        = [1, 2, 5, 10, 20]

# Model hyperparameters
OOM_L           = 3                # max basis length
GDC_KWARGS      = dict(
    alpha=0.7, theta=0.2, gamma=0.0, beta=0.1,
    transition_type='self_loop', initial_dist='sequence_starts',
)

OUT_CSV = os.path.join(_THIS_DIR, "main_sweep_results.csv")


def log(*a):
    print(*a, flush=True)


def run_one(nS: int, nA: int, seed: int):
    rng = np.random.default_rng(seed * 1000 + nS * 11 + nA)
    hmm = random_dense_hmm(nS, nA, rng)

    train_seqs = hmm.sample_many(N_TRAIN_SEQ, TRAIN_LEN, rng)
    test_prefixes = hmm.sample_many(N_TEST_PREFIXES, TEST_PREFIX_LEN, rng)

    t0 = time.time()
    oom_clip = fit_oom(train_seqs, alphabet_size=nA,
                       max_basis_length=OOM_L, prob_mode='clip')
    oom_soft = fit_oom(train_seqs, alphabet_size=nA,
                       max_basis_length=OOM_L, prob_mode='softmax')
    t_oom_fit = time.time() - t0

    t0 = time.time()
    gdc = fit_gdc(train_seqs, alphabet_size=nA, **GDC_KWARGS)
    t_gdc_fit = time.time() - t0

    oomc_mse = mse_at_horizons(oom_clip, hmm, test_prefixes, HORIZONS)
    ooms_mse = mse_at_horizons(oom_soft, hmm, test_prefixes, HORIZONS)
    gdc_mse  = mse_at_horizons(gdc,      hmm, test_prefixes, HORIZONS)
    uni_mse  = uniform_baseline_mse(hmm, test_prefixes, HORIZONS)
    sta_mse  = stationary_baseline_mse(hmm, test_prefixes, HORIZONS)

    return {
        'oom_fit_s':  t_oom_fit,
        'gdc_fit_s':  t_gdc_fit,
        'oomc_mse':   oomc_mse,
        'ooms_mse':   ooms_mse,
        'gdc_mse':    gdc_mse,
        'uni_mse':    uni_mse,
        'sta_mse':    sta_mse,
        'oom_rank':   oom_soft.oom._rank_used,
        'gdc_hidden': gdc.gdc.n_states,
    }


def main():
    rows = []
    total = len(list(STATE_COUNTS)) * len(list(ALPHABET_SIZES)) * len(SEEDS)
    i = 0
    log(f"Running main sweep: {total} configs")
    log(f"  train: {N_TRAIN_SEQ} seqs x {TRAIN_LEN} steps; "
        f"test: {N_TEST_PREFIXES} prefixes x {TEST_PREFIX_LEN} steps")
    log(f"  horizons = {HORIZONS}")
    for nS in STATE_COUNTS:
        for nA in ALPHABET_SIZES:
            for seed in SEEDS:
                i += 1
                t0 = time.time()
                try:
                    r = run_one(nS, nA, seed)
                    dt = time.time() - t0
                    log(f"  [{i:3d}/{total}] nS={nS} nA={nA} seed={seed} "
                        f"oomS(h1)={r['ooms_mse'][1]:.4f} "
                        f"oomC(h1)={r['oomc_mse'][1]:.4f} "
                        f"gdc(h1)={r['gdc_mse'][1]:.4f} "
                        f"uni(h1)={r['uni_mse'][1]:.4f} "
                        f"({dt:.1f}s)")
                except Exception as e:
                    log(f"  [{i:3d}/{total}] nS={nS} nA={nA} seed={seed} "
                        f"FAILED: {type(e).__name__}: {e}")
                    continue
                for h in HORIZONS:
                    rows.append({
                        'nS': nS, 'nA': nA, 'seed': seed, 'horizon': h,
                        'oom_clip_mse':    r['oomc_mse'][h],
                        'oom_softmax_mse': r['ooms_mse'][h],
                        'gdc_mse':         r['gdc_mse'][h],
                        'uni_mse':         r['uni_mse'][h],
                        'sta_mse':         r['sta_mse'][h],
                        'oom_rank':        r['oom_rank'],
                        'gdc_hidden':      r['gdc_hidden'],
                        'oom_fit_s':       r['oom_fit_s'],
                        'gdc_fit_s':       r['gdc_fit_s'],
                    })

    # Write CSV
    if rows:
        with open(OUT_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log(f"Wrote {len(rows)} rows to {OUT_CSV}")

    # Quick summary by horizon, averaged across all configs
    log("\n=== Summary (mean MSE across all (nS, nA, seed)) ===")
    log(f"{'h':<4} {'OOM-clip':<12} {'OOM-soft':<12} {'GDC':<12} "
        f"{'Uniform':<12} {'Stationary':<12}")
    for h in HORIZONS:
        oomc = np.mean([r['oom_clip_mse']    for r in rows if r['horizon'] == h])
        ooms = np.mean([r['oom_softmax_mse'] for r in rows if r['horizon'] == h])
        gdc_m = np.mean([r['gdc_mse']        for r in rows if r['horizon'] == h])
        uni_m = np.mean([r['uni_mse']        for r in rows if r['horizon'] == h])
        sta_m = np.mean([r['sta_mse']        for r in rows if r['horizon'] == h])
        log(f"{h:<4} {oomc:<12.5f} {ooms:<12.5f} {gdc_m:<12.5f} "
            f"{uni_m:<12.5f} {sta_m:<12.5f}")


if __name__ == "__main__":
    main()
