"""Sequence-length scaling sweep for the HMM forecasting benchmark.

Runs all six methods (GDC, CHMM, ALERGIA, Parrot, HPYLM, PPM-D) at a
single TRAIN_LEN value (length of each training sequence). Mirrors the
canonical setup of `run_perplexity_sweep.py` / `parrot_eval.py` /
`hpylm_eval.py` / `ppm_eval.py`, with TRAIN_LEN parameterised.

  - 6 regimes × 3 N_train × 6 seeds (val 3,4,5; test 0,1,2)
  - 100 test prefixes per (regime, N, seed), each length 20
  - Horizons {1, 5, 20} (Table 7 reports h=1)
  - Metric: excess perplexity = 2^(CE - entropy_floor); 1.000 = floor

Usage:  python hmm_comparison/seq_len_sweep.py <TRAIN_LEN>
        produces hmm_comparison/seq_len_<TRAIN_LEN>_results.csv

The HMM RNG seed depends only on (regime, seed) — NOT on TRAIN_LEN —
so the same HMM is used across TRAIN_LEN values for a given (regime,
seed). Training/test sequences are sampled from that same HMM at each
TRAIN_LEN. This lets us isolate the sequence-length effect from HMM
variability.
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from discrete_parrot import DiscreteParrotPool
from discrete_hpylm import HPYLMPool
from discrete_ppm import PPMPool

REGIMES = [
    ('dense_small',  10, 4, 'dense',  1.0, None),
    ('dense_large',  30, 8, 'dense',  1.0, None),
    ('det_small',    10, 4, 'dense',  0.1, None),
    ('det_large',    30, 8, 'dense',  0.1, None),
    ('sparse_small', 10, 4, 'sparse', 0.1, 2),
    ('sparse_large', 30, 8, 'sparse', 0.1, 2),
]
SEEDS = [0, 1, 2, 3, 4, 5]
N_TRAIN_VALUES = [25, 100, 400]
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 5, 20]

GDC_CONFIGS = [
    (0.10, 0.001, 0.20),
    (0.30, 0.005, 0.30),
    (0.50, 0.005, 0.20),
    (0.70, 0.010, 0.20),
    (0.80, 0.001, 0.10),
]
CHMM_KS = [4, 16, 32]
ALERGIA_EPS = 0.05

PARROT_LS = [1, 2, 3, 4]
PARROT_KS = [1, 5, 25, 100, 400]
PARROT_ALPHAS = [0.1, 1.0]

HPYLM_DEPTHS = [2, 3, 4, 6]
HPYLM_DISCOUNTS = [0.25, 0.5, 0.75]
HPYLM_CONCS = [0.5, 1.0, 5.0]
HPYLM_ALPHA_PRIOR = 0.01

PPM_DEPTHS = [2, 3, 4, 6]
PPM_DISCOUNTS = [0.25, 0.5, 0.75]
PPM_ALPHA_PRIOR = 0.01


class _PoolForecaster:
    """Adapter: wrap any pool exposing predict_distribution(prefix, h, **kw)
    in the horizon_emission(prefix, h) interface that evaluation expects."""
    def __init__(self, pool, **predict_kw):
        self.pool = pool
        self.predict_kw = predict_kw
        self.nA = getattr(pool, 'A', None) or getattr(pool, 'alphabet_size')

    def horizon_emission(self, prefix_obs, h):
        return self.pool.predict_distribution(np.asarray(prefix_obs), h=h,
                                              **self.predict_kw)


def run_cell(args):
    train_len, regime_name, nS, nA, kind, E_conc, fanout, seed = args
    sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
    from random_hmm import random_dense_hmm, random_sparse_topology_hmm
    from evaluation import perplexity_at_horizons
    from model_wrappers import fit_gdc
    from chmm_alergia_wrappers import fit_chmm, fit_alergia

    # HMM seed is independent of TRAIN_LEN: same HMM across TRAIN_LEN runs
    seed_offset = (1 if 'det' in regime_name else 0) \
                  + (2 if 'sparse' in regime_name else 0)
    rng = np.random.default_rng(60000 + seed * 137 + nS * 7 + nA * 11
                                + seed_offset)
    if kind == 'sparse':
        hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                         E_concentration=E_conc)
    else:
        hmm = random_dense_hmm(nS, nA, rng,
                               T_concentration=1.0, E_concentration=E_conc)
    full_train = [hmm.sample(train_len, rng)[1]
                  for _ in range(max(N_TRAIN_VALUES))]
    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]

    rows = []
    base = dict(train_len=train_len, regime=regime_name, nS=nS, nA=nA,
                seed=seed)

    def record(N, model_class, model_name, model):
        ppl = perplexity_at_horizons(model, hmm, test_pf, HORIZONS)
        for h in HORIZONS:
            r = ppl[h]
            rows.append(dict(base, N_train=N, model_class=model_class,
                             model=model_name, horizon=h,
                             cross_entropy_bits=r['cross_entropy_bits'],
                             entropy_floor_bits=r['entropy_floor_bits'],
                             perplexity=r['perplexity'],
                             entropy_floor_perplexity=r['entropy_floor_perplexity'],
                             excess_perplexity=r['excess_perplexity']))

    for N in N_TRAIN_VALUES:
        train = full_train[:N]

        # GDC
        for (a, t, b) in GDC_CONFIGS:
            m = fit_gdc(train, nA, alpha=a, theta=t, gamma=0.0, beta=b,
                        transition_type='self_loop',
                        initial_dist='sequence_starts')
            record(N, 'gdc', f'gdc-a{a}-t{t}-b{b}', m)

        # CHMM
        for K in CHMM_KS:
            try:
                m = fit_chmm(train, nA, K=K, n_em_iters=50)
                record(N, 'chmm', f'chmm-K{K}', m)
            except Exception as e:
                sys.stderr.write(
                    f"[chmm K={K} fail TL={train_len} {regime_name} N={N}] {e}\n")

        # ALERGIA (single config — no val tuning)
        try:
            m = fit_alergia(train, nA, eps=ALERGIA_EPS)
            record(N, 'alergia', f'alergia-eps{ALERGIA_EPS}', m)
        except Exception as e:
            sys.stderr.write(
                f"[alergia fail TL={train_len} {regime_name} N={N}] {e}\n")

        # Parrot
        parrot_pools = {L: DiscreteParrotPool(train, alphabet_size=nA, L=L)
                        for L in PARROT_LS}
        for L in PARROT_LS:
            for K in PARROT_KS:
                for ap in PARROT_ALPHAS:
                    pool = parrot_pools[L]
                    fc = _PoolForecaster(pool, K=K, alpha_prior=ap)
                    record(N, 'parrot', f'parrot-L{L}-K{K}-a{ap}', fc)

        # HPYLM
        for D in HPYLM_DEPTHS:
            for d in HPYLM_DISCOUNTS:
                for c in HPYLM_CONCS:
                    pool = HPYLMPool(train, alphabet_size=nA, max_depth=D,
                                     discount=d, concentration=c, seed=seed)
                    fc = _PoolForecaster(pool, alpha_prior=HPYLM_ALPHA_PRIOR)
                    record(N, 'hpylm', f'hpylm-D{D}-d{d}-a{c}', fc)

        # PPM-D
        for D in PPM_DEPTHS:
            for d in PPM_DISCOUNTS:
                pool = PPMPool(train, alphabet_size=nA, max_depth=D,
                               discount=d)
                fc = _PoolForecaster(pool, alpha_prior=PPM_ALPHA_PRIOR)
                record(N, 'ppm', f'ppm-D{D}-d{d}', fc)

    return rows


def main():
    if len(sys.argv) != 2:
        print("usage: python seq_len_sweep.py <TRAIN_LEN>")
        sys.exit(1)
    train_len = int(sys.argv[1])

    out_csv = os.path.join(HERE, f'seq_len_{train_len}_results.csv')
    n_workers = max(1, min(20, (os.cpu_count() or 4) - 1))
    tasks = [(train_len, name, nS, nA, kind, conc, fanout, seed)
             for (name, nS, nA, kind, conc, fanout) in REGIMES
             for seed in SEEDS]
    print(f"Seq-len sweep: TRAIN_LEN={train_len}  "
          f"{len(tasks)} cells × all 6 methods × {len(N_TRAIN_VALUES)} "
          f"N_train × {len(HORIZONS)} horizons, {n_workers} workers",
          flush=True)

    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for cell_rows in pool.imap_unordered(run_cell, tasks, chunksize=1):
            all_rows.extend(cell_rows); done += 1
            print(f"  {done}/{len(tasks)} cells done  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    fields = ['train_len', 'regime', 'nS', 'nA', 'seed', 'N_train',
              'model_class', 'model', 'horizon',
              'cross_entropy_bits', 'entropy_floor_bits',
              'perplexity', 'entropy_floor_perplexity', 'excess_perplexity']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {out_csv}  [{time.time()-t0:.1f}s, {len(all_rows)} rows]",
          flush=True)


if __name__ == "__main__":
    main()
