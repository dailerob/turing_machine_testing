"""
Focused hypothesis tests.

H1  Low-rank HMMs favor OOM.
    Fix nS=10, nA=6. Sweep transition-matrix rank r in {1..10}. Prediction:
    OOM-MSE falls when r is small (OOM rank truncation aligns with truth);
    GDC doesn't benefit.

H2  Near-deterministic emissions favor GDC.
    Fix nS=6, nA=6. Sweep emission-Dirichlet concentration c in
    {0.05, 0.3, 1.0, 3.0, 10.0}. Prediction: when c<<1, each hidden state
    emits essentially one symbol, so each prefix ~= its hidden state chain
    and GDC (prefix memoriser) dominates.

H3  Horizon scaling.
    Already covered by the main sweep; we re-plot here for the two fixed
    settings above and mark the relative degradation vs horizon.

H4  Sparse vs dense topology.
    Fix nS=8, nA=4. Compare dense Dirichlet transitions vs sparse fanout=2
    transitions. Prediction: sparse favors GDC's prefix memorisation;
    dense favors OOM.

Writes one CSV per hypothesis.
"""

from __future__ import annotations

import os
import sys
import csv
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from random_hmm import (
    random_dense_hmm, random_sparse_topology_hmm, random_lowrank_hmm,
)
from model_wrappers import fit_oom, fit_gdc
from evaluation import mse_at_horizons, uniform_baseline_mse


N_TRAIN_SEQ     = 200
TRAIN_LEN       = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS        = [1, 2, 5, 10, 20]
SEEDS           = [0, 1, 2, 3, 4]
OOM_L           = 3
GDC_KWARGS      = dict(
    alpha=0.7, theta=0.2, gamma=0.0, beta=0.1,
    transition_type='self_loop', initial_dist='sequence_starts',
)


def log(*a):
    print(*a, flush=True)


def train_and_eval(hmm, rng):
    train_seqs = hmm.sample_many(N_TRAIN_SEQ, TRAIN_LEN, rng)
    test_prefixes = hmm.sample_many(N_TEST_PREFIXES, TEST_PREFIX_LEN, rng)
    oom = fit_oom(train_seqs, alphabet_size=hmm.nA,
                  max_basis_length=OOM_L, prob_mode='softmax')
    gdc = fit_gdc(train_seqs, alphabet_size=hmm.nA, **GDC_KWARGS)
    return {
        'oom':     mse_at_horizons(oom, hmm, test_prefixes, HORIZONS),
        'gdc':     mse_at_horizons(gdc, hmm, test_prefixes, HORIZONS),
        'uniform': uniform_baseline_mse(hmm, test_prefixes, HORIZONS),
    }


# ---------------------------------------------------------------------------
# H1 — rank sweep
# ---------------------------------------------------------------------------
def run_h1(nS: int = 10, nA: int = 6):
    ranks = list(range(1, nS + 1))
    rows = []
    log(f"\n=== H1: low-rank HMMs favor OOM (nS={nS}, nA={nA}) ===")
    log(f"{'rank':<6} {'seed':<6} {'oom(h=1)':<12} {'gdc(h=1)':<12} "
        f"{'oom(h=5)':<12} {'gdc(h=5)':<12}")
    for r in ranks:
        for s in SEEDS:
            rng = np.random.default_rng(1000 * s + 13 * r)
            hmm = random_lowrank_hmm(nS, nA, r, rng)
            res = train_and_eval(hmm, rng)
            log(f"{r:<6} {s:<6} {res['oom'][1]:<12.5f} {res['gdc'][1]:<12.5f} "
                f"{res['oom'][5]:<12.5f} {res['gdc'][5]:<12.5f}")
            for h in HORIZONS:
                rows.append({
                    'rank': r, 'seed': s, 'horizon': h,
                    'oom_mse': res['oom'][h],
                    'gdc_mse': res['gdc'][h],
                    'uni_mse': res['uniform'][h],
                })
    _write_csv(rows, 'h1_rank_results.csv')


# ---------------------------------------------------------------------------
# H2 — emission concentration sweep
# ---------------------------------------------------------------------------
def run_h2(nS: int = 6, nA: int = 6):
    concentrations = [0.05, 0.3, 1.0, 3.0, 10.0]
    rows = []
    log(f"\n=== H2: near-deterministic emissions favor GDC "
        f"(nS={nS}, nA={nA}) ===")
    log(f"{'c':<8} {'seed':<6} {'oom(h=1)':<12} {'gdc(h=1)':<12} "
        f"{'oom(h=5)':<12} {'gdc(h=5)':<12}")
    for c in concentrations:
        for s in SEEDS:
            rng = np.random.default_rng(2000 * s + int(c * 100))
            hmm = random_dense_hmm(nS, nA, rng,
                                   T_concentration=1.0, E_concentration=c)
            res = train_and_eval(hmm, rng)
            log(f"{c:<8} {s:<6} {res['oom'][1]:<12.5f} {res['gdc'][1]:<12.5f} "
                f"{res['oom'][5]:<12.5f} {res['gdc'][5]:<12.5f}")
            for h in HORIZONS:
                rows.append({
                    'E_concentration': c, 'seed': s, 'horizon': h,
                    'oom_mse': res['oom'][h],
                    'gdc_mse': res['gdc'][h],
                    'uni_mse': res['uniform'][h],
                })
    _write_csv(rows, 'h2_emission_concentration_results.csv')


# ---------------------------------------------------------------------------
# H4 — sparse vs dense
# ---------------------------------------------------------------------------
def run_h4(nS: int = 8, nA: int = 4):
    rows = []
    log(f"\n=== H4: sparse vs dense topology (nS={nS}, nA={nA}) ===")
    log(f"{'topology':<10} {'seed':<6} {'oom(h=1)':<12} {'gdc(h=1)':<12} "
        f"{'oom(h=5)':<12} {'gdc(h=5)':<12}")
    for topo in ('dense', 'sparse2'):
        for s in SEEDS:
            rng = np.random.default_rng(3000 * s + hash(topo) % 1000)
            if topo == 'dense':
                hmm = random_dense_hmm(nS, nA, rng)
            else:
                hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=2)
            res = train_and_eval(hmm, rng)
            log(f"{topo:<10} {s:<6} {res['oom'][1]:<12.5f} "
                f"{res['gdc'][1]:<12.5f} {res['oom'][5]:<12.5f} "
                f"{res['gdc'][5]:<12.5f}")
            for h in HORIZONS:
                rows.append({
                    'topology': topo, 'seed': s, 'horizon': h,
                    'oom_mse': res['oom'][h],
                    'gdc_mse': res['gdc'][h],
                    'uni_mse': res['uniform'][h],
                })
    _write_csv(rows, 'h4_topology_results.csv')


def _write_csv(rows, name):
    path = os.path.join(_THIS_DIR, name)
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    run_h1()
    run_h2()
    run_h4()
