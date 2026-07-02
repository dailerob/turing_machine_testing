"""HPYLM (fixed-depth Sequence Memoizer) on the HMM forecasting benchmark.

Mirrors `parrot_eval.py` exactly so HPYLM numbers slot into the same
leakage-free Table-7 comparison.

Variant grid (val-tuned per cell):
  - max_depth ∈ {2, 3, 4, 6}
  - discount  ∈ {0.25, 0.5, 0.75}
  - concentration ∈ {0.5, 1.0, 5.0}
  - alpha_prior ∈ {0.0, 0.01, 0.1}    (Laplace smoothing on the
    output distribution; needed for cross-entropy scoring when the
    HPYLM still leaves bins at 0 due to discrete deltas at low depth.)

That's 4 × 3 × 3 × 3 = 108 configs per cell. We use a smaller grid
(36 configs) by tying alpha_prior to a single value (0.01), since
the smoothing barely shifts the result.
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from discrete_hpylm import HPYLMPool

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
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 5, 20]

DEPTHS = [2, 3, 4, 6]
DISCOUNTS = [0.25, 0.5, 0.75]
CONCENTRATIONS = [0.5, 1.0, 5.0]
ALPHA_PRIOR = 0.01

OUT_CSV = os.path.join(HERE, 'hpylm_results.csv')


def variants():
    return [(D, d, a) for D in DEPTHS for d in DISCOUNTS for a in CONCENTRATIONS]


class HPYLMForecaster:
    """Wraps an HPYLMPool to expose the horizon_emission(prefix, h) interface
    that hmm_comparison.evaluation expects."""
    def __init__(self, pool: HPYLMPool, alpha_prior: float = ALPHA_PRIOR):
        self.pool = pool
        self.alpha_prior = alpha_prior
        self.nA = pool.A

    def horizon_emission(self, prefix_obs, h: int) -> np.ndarray:
        return self.pool.predict_distribution(np.asarray(prefix_obs),
                                              h=h,
                                              alpha_prior=self.alpha_prior)


def run_cell(args):
    regime_name, nS, nA, kind, E_conc, fanout, seed = args
    sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
    from random_hmm import random_dense_hmm, random_sparse_topology_hmm
    from evaluation import perplexity_at_horizons

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
    full_train = [hmm.sample(TRAIN_LEN, rng)[1]
                  for _ in range(max(N_TRAIN_VALUES))]
    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]

    rows = []
    for N in N_TRAIN_VALUES:
        train = full_train[:N]
        for D, disc, conc in variants():
            pool = HPYLMPool(train, alphabet_size=nA, max_depth=D,
                             discount=disc, concentration=conc, seed=seed)
            forecaster = HPYLMForecaster(pool, alpha_prior=ALPHA_PRIOR)
            ppl = perplexity_at_horizons(forecaster, hmm, test_pf, HORIZONS)
            for h in HORIZONS:
                r = ppl[h]
                rows.append(dict(regime=regime_name, nS=nS, nA=nA,
                                 seed=seed, N_train=N,
                                 model=f'hpylm-D{D}-d{disc}-a{conc}',
                                 max_depth=D, discount=disc, concentration=conc,
                                 alpha_prior=ALPHA_PRIOR, horizon=h,
                                 cross_entropy_bits=r['cross_entropy_bits'],
                                 entropy_floor_bits=r['entropy_floor_bits'],
                                 perplexity=r['perplexity'],
                                 entropy_floor_perplexity=r['entropy_floor_perplexity'],
                                 excess_perplexity=r['excess_perplexity']))
    return rows


def main():
    n_workers = max(1, min(20, (os.cpu_count() or 4) - 1))
    tasks = [(name, nS, nA, kind, conc, fanout, seed)
             for (name, nS, nA, kind, conc, fanout) in REGIMES
             for seed in SEEDS]
    n_variants = len(variants())
    print(f"HPYLM-HMM sweep: {len(tasks)} cells × {n_variants} variants × "
          f"{len(N_TRAIN_VALUES)} N_train × {len(HORIZONS)} horizons, "
          f"{n_workers} workers", flush=True)

    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for cell_rows in pool.imap_unordered(run_cell, tasks, chunksize=1):
            all_rows.extend(cell_rows); done += 1
            print(f"  {done}/{len(tasks)} cells done  [{time.time()-t0:.0f}s]",
                  flush=True)

    fields = ['regime', 'nS', 'nA', 'seed', 'N_train', 'model',
              'max_depth', 'discount', 'concentration', 'alpha_prior',
              'horizon', 'cross_entropy_bits', 'entropy_floor_bits',
              'perplexity', 'entropy_floor_perplexity', 'excess_perplexity']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}  [{time.time()-t0:.1f}s, {len(all_rows)} rows]",
          flush=True)


if __name__ == "__main__":
    main()
