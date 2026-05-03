"""HMM-forecasting sweep computing BOTH MSE and perplexity (cross-entropy).

Same regimes as the size+det+sparse sweeps; reports per (model, regime,
N, horizon):
    mse                    — squared-error against true posterior
    cross_entropy_bits     — soft cross-entropy of model distribution
                             against true posterior
    entropy_floor_bits     — entropy of true posterior (the floor)
    excess_perplexity      — 2^(CE - floor); analog to PAutomaC gap

Goal: see whether MSE and perplexity disagree about the GDC-vs-CHMM-
vs-ALERGIA ranking on different regimes.
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REGIMES = [
    ('dense_small',  10, 4, 'dense',  1.0, None),
    ('dense_large',  30, 8, 'dense',  1.0, None),
    ('det_small',    10, 4, 'dense',  0.1, None),
    ('det_large',    30, 8, 'dense',  0.1, None),
    ('sparse_small', 10, 4, 'sparse', 0.1, 2),
    ('sparse_large', 30, 8, 'sparse', 0.1, 2),
]
SEEDS = [0, 1, 2]
N_TRAIN_VALUES = [25, 100, 400]
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 5, 20]

# A focused GDC config grid covering the key configs each regime prefers.
GDC_CONFIGS = [
    (0.10, 0.001, 0.20),  # dense-large optimum
    (0.30, 0.005, 0.30),  # dense-small low-N optimum
    (0.50, 0.005, 0.20),  # midline / dense-small high-N
    (0.70, 0.010, 0.20),  # det-small high-N optimum
    (0.80, 0.001, 0.10),  # sparse high-alpha optimum
]
CHMM_KS = [4, 16, 32]
ALERGIA_EPS = 0.05

OUT_CSV = os.path.join(HERE, 'perplexity_sweep_results.csv')


def run_cell(args):
    regime_name, nS, nA, kind, E_conc, fanout, seed = args
    sys.path.insert(0, HERE)
    from random_hmm import random_dense_hmm, random_sparse_topology_hmm
    from model_wrappers import fit_gdc
    from chmm_alergia_wrappers import fit_chmm, fit_alergia
    from evaluation import mse_at_horizons, perplexity_at_horizons

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
    def record(model_name, model):
        mse = mse_at_horizons(model, hmm, test_pf, HORIZONS)
        ppl = perplexity_at_horizons(model, hmm, test_pf, HORIZONS)
        for h in HORIZONS:
            r = ppl[h]
            rows.append(dict(regime=regime_name, nS=nS, nA=nA,
                             seed=seed, N_train=N, model=model_name,
                             horizon=h, mse=mse[h],
                             cross_entropy_bits=r['cross_entropy_bits'],
                             entropy_floor_bits=r['entropy_floor_bits'],
                             perplexity=r['perplexity'],
                             entropy_floor_perplexity=r['entropy_floor_perplexity'],
                             excess_perplexity=r['excess_perplexity']))

    for N in N_TRAIN_VALUES:
        train = full_train[:N]
        for (a, t, b) in GDC_CONFIGS:
            m = fit_gdc(train, nA, alpha=a, theta=t, gamma=0.0, beta=b,
                        transition_type='self_loop',
                        initial_dist='sequence_starts')
            record(f'gdc-a{a}-t{t}-b{b}', m)
        for K in CHMM_KS:
            try:
                m = fit_chmm(train, nA, K=K, n_em_iters=50)
                record(f'chmm-K{K}', m)
            except Exception as e:
                sys.stderr.write(f"[chmm K={K} fail {regime_name} N={N}] {e}\n")
        try:
            m = fit_alergia(train, nA, eps=ALERGIA_EPS)
            record('alergia-eps0.05', m)
        except Exception as e:
            sys.stderr.write(f"[alergia fail {regime_name} N={N}] {e}\n")
    return rows


def main():
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    tasks = [(name, nS, nA, kind, conc, fanout, seed)
             for (name, nS, nA, kind, conc, fanout) in REGIMES
             for seed in SEEDS]
    print(f"Perplexity sweep: {len(tasks)} cells, {n_workers} workers",
          flush=True)

    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for cell_rows in pool.imap_unordered(run_cell, tasks, chunksize=1):
            all_rows.extend(cell_rows)
            done += 1
            print(f"  {done}/{len(tasks)} cells done  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    fields = ['regime', 'nS', 'nA', 'seed', 'N_train', 'model', 'horizon',
              'mse', 'cross_entropy_bits', 'entropy_floor_bits',
              'perplexity', 'entropy_floor_perplexity', 'excess_perplexity']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}  [{time.time()-t0:.1f}s, "
          f"{len(all_rows)} rows]", flush=True)


if __name__ == "__main__":
    main()
