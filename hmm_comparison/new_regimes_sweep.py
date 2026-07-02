"""HMM forecasting sweep on four new regimes designed to defeat the
training-frequency baseline (bimodal_small, cyclic_K8, binary_deep,
reset_chain). Mirrors `seq_len_sweep.py` exactly otherwise.

  - 4 regimes × 3 N_train ∈ {25, 100, 400} × 6 seeds (val 3,4,5; test 0,1,2)
  - Variable TRAIN_LEN (defaults to 25; argv override accepted)
  - 100 test prefixes per cell, length 20
  - Horizon h=1
  - Methods: GDC (5 fixed configs), CHMM (val K), ALERGIA, Parrot
    (40 configs), HPYLM (36), PPM-D (12)
  - Plus the Freq (training-unigram) baseline as a reference column

Usage: python new_regimes_sweep.py [TRAIN_LEN]
       outputs new_regimes_<TL>_results.csv (per (config, seed) raw rows)
       and prints per-regime perplexity tables with the Freq column.
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from discrete_parrot import DiscreteParrotPool
from discrete_hpylm import HPYLMPool
from discrete_ppm import PPMPool


# (name, nS, nA, kind)
REGIMES = [
    ('bimodal_small',  10, 4, 'bimodal'),
    ('cyclic_K8',       8, 8, 'cyclic'),
    ('binary_deep',    30, 2, 'binary_deep'),
    ('reset_chain',    20, 4, 'reset_chain'),
]
SEEDS = [0, 1, 2, 3, 4, 5]
VAL_SEEDS = {3, 4, 5}
TEST_SEEDS = {0, 1, 2}
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
    def __init__(self, pool, **predict_kw):
        self.pool = pool
        self.predict_kw = predict_kw
        self.nA = getattr(pool, 'A', None) or getattr(pool, 'alphabet_size')
    def horizon_emission(self, prefix_obs, h):
        return self.pool.predict_distribution(np.asarray(prefix_obs), h=h,
                                              **self.predict_kw)


def make_hmm(kind, nS, nA, rng):
    sys.path.insert(0, HERE)
    from random_hmm import (random_bimodal_hmm, random_cyclic_hmm,
                             random_binary_deep_hmm,
                             random_reset_chain_hmm)
    if kind == 'bimodal':
        return random_bimodal_hmm(nS, nA, rng,
                                  sticky_prob=0.95, E_concentration=0.1)
    if kind == 'cyclic':
        return random_cyclic_hmm(nS, nA, rng,
                                 advance_prob=0.95, E_concentration=0.1)
    if kind == 'binary_deep':
        return random_binary_deep_hmm(nS, rng,
                                      fanout=2, E_concentration=0.1)
    if kind == 'reset_chain':
        return random_reset_chain_hmm(nS, nA, rng,
                                      advance_prob=0.90, reset_prob=0.05,
                                      E_concentration=0.1)
    raise ValueError(f"unknown kind: {kind}")


def freq_excess_pp(hmm, train, test_pf, h=1, alpha_smooth=1e-6):
    nA = hmm.nA
    counts = np.zeros(nA)
    for seq in train:
        for v in np.asarray(seq, dtype=np.int64):
            counts[v] += 1
    freq = (counts + alpha_smooth) / (counts.sum() + nA * alpha_smooth)
    Th = np.linalg.matrix_power(hmm.T, h)
    ces = []; floors = []
    for prefix in test_pf:
        a = hmm.filter(prefix)
        true_next = a @ Th @ hmm.E
        ce = -float(np.sum(true_next * np.log2(np.maximum(freq, 1e-12))))
        floor = -float(np.sum(true_next * np.log2(np.maximum(true_next, 1e-12))))
        ces.append(ce); floors.append(floor)
    return float(2 ** (np.mean(ces) - np.mean(floors)))


def run_cell(args):
    train_len, regime_name, nS, nA, kind, seed = args
    sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
    from evaluation import perplexity_at_horizons
    from model_wrappers import fit_gdc
    from chmm_alergia_wrappers import fit_chmm, fit_alergia

    # New seed-offset to keep deterministic seeds distinct from the
    # original seq_len_sweep regimes (offset 100 + per-kind tag).
    kind_tag = {'bimodal': 0, 'cyclic': 10, 'binary_deep': 20,
                'reset_chain': 30}[kind]
    rng = np.random.default_rng(70000 + kind_tag + seed * 137 + nS * 7
                                + nA * 11)
    hmm = make_hmm(kind, nS, nA, rng)
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

        for (a, t, b) in GDC_CONFIGS:
            m = fit_gdc(train, nA, alpha=a, theta=t, gamma=0.0, beta=b,
                        transition_type='self_loop',
                        initial_dist='sequence_starts')
            record(N, 'gdc', f'gdc-a{a}-t{t}-b{b}', m)

        for K in CHMM_KS:
            try:
                m = fit_chmm(train, nA, K=K, n_em_iters=50)
                record(N, 'chmm', f'chmm-K{K}', m)
            except Exception as e:
                sys.stderr.write(
                    f"[chmm K={K} fail TL={train_len} {regime_name} N={N}] {e}\n")

        try:
            m = fit_alergia(train, nA, eps=ALERGIA_EPS)
            record(N, 'alergia', f'alergia-eps{ALERGIA_EPS}', m)
        except Exception as e:
            sys.stderr.write(
                f"[alergia fail TL={train_len} {regime_name} N={N}] {e}\n")

        parrot_pools = {L: DiscreteParrotPool(train, alphabet_size=nA, L=L)
                        for L in PARROT_LS}
        for L in PARROT_LS:
            for K in PARROT_KS:
                for ap in PARROT_ALPHAS:
                    pool = parrot_pools[L]
                    fc = _PoolForecaster(pool, K=K, alpha_prior=ap)
                    record(N, 'parrot', f'parrot-L{L}-K{K}-a{ap}', fc)

        for D in HPYLM_DEPTHS:
            for d in HPYLM_DISCOUNTS:
                for c in HPYLM_CONCS:
                    pool = HPYLMPool(train, alphabet_size=nA, max_depth=D,
                                     discount=d, concentration=c, seed=seed)
                    fc = _PoolForecaster(pool, alpha_prior=HPYLM_ALPHA_PRIOR)
                    record(N, 'hpylm', f'hpylm-D{D}-d{d}-a{c}', fc)

        for D in PPM_DEPTHS:
            for d in PPM_DISCOUNTS:
                pool = PPMPool(train, alphabet_size=nA, max_depth=D,
                               discount=d)
                fc = _PoolForecaster(pool, alpha_prior=PPM_ALPHA_PRIOR)
                record(N, 'ppm', f'ppm-D{D}-d{d}', fc)

        # Freq baseline at h=1 only (other horizons would just duplicate)
        f_pp = freq_excess_pp(hmm, train, test_pf, h=1)
        rows.append(dict(base, N_train=N, model_class='freq',
                         model='freq', horizon=1,
                         cross_entropy_bits=float('nan'),
                         entropy_floor_bits=float('nan'),
                         perplexity=float('nan'),
                         entropy_floor_perplexity=float('nan'),
                         excess_perplexity=f_pp))

    return rows


def main():
    train_len = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    out_csv = os.path.join(HERE,
                           f'new_regimes_{train_len}_results.csv')

    n_workers = max(1, min(20, (os.cpu_count() or 4) - 1))
    tasks = [(train_len, name, nS, nA, kind, seed)
             for (name, nS, nA, kind) in REGIMES
             for seed in SEEDS]
    print(f"=== new regimes sweep TL={train_len} ===")
    print(f"  Regimes: {[r[0] for r in REGIMES]}")
    print(f"  Cells: {len(tasks)},  workers: {n_workers}\n", flush=True)

    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for cell_rows in pool.imap_unordered(run_cell, tasks, chunksize=1):
            all_rows.extend(cell_rows); done += 1
            print(f"  {done}/{len(tasks)} cells  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    fields = ['train_len', 'regime', 'nS', 'nA', 'seed', 'N_train',
              'model_class', 'model', 'horizon',
              'cross_entropy_bits', 'entropy_floor_bits',
              'perplexity', 'entropy_floor_perplexity', 'excess_perplexity']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {out_csv} ({len(all_rows)} rows, "
          f"{time.time()-t0:.1f}s)\n", flush=True)

    # Aggregation: val-pick per (regime, N, model_class) on val seeds
    df = pd.DataFrame(all_rows)
    df = df[df.horizon == 1]
    METHODS = ['gdc', 'chmm', 'alergia', 'parrot', 'hpylm', 'ppm', 'freq']
    PRETTY  = {'gdc':'GDC','chmm':'CHMM','alergia':'ALERGIA',
               'parrot':'Parrot','hpylm':'HPYLM','ppm':'PPM-D',
               'freq':'Freq'}
    print(f"## Excess perplexity at TL={train_len}, h=1, "
          "test seeds {0,1,2} after val-picking on {3,4,5}.\n")
    print("**Bold = best in row** (excluding Freq).\n")

    for regime, *_ in REGIMES:
        print(f"### {regime}\n")
        print("| N | " + " | ".join(PRETTY[m] for m in METHODS) + " |")
        print("|---:|" + "---:|" * len(METHODS))
        for N in N_TRAIN_VALUES:
            cell = df[(df.regime == regime) & (df.N_train == N)]
            test_pp = {}
            for m in METHODS:
                csub = cell[cell.model_class == m]
                cval = csub[csub.seed.isin(VAL_SEEDS)]
                ctest = csub[csub.seed.isin(TEST_SEEDS)]
                if csub.empty:
                    test_pp[m] = float('nan'); continue
                if m in ('alergia', 'freq'):
                    test_pp[m] = float(ctest.excess_perplexity.mean())
                else:
                    val_means = (cval.groupby('model')
                                     ['excess_perplexity'].mean())
                    if val_means.empty:
                        test_pp[m] = float('nan'); continue
                    pick = val_means.idxmin()
                    test_pp[m] = float(ctest[ctest.model == pick]
                                       .excess_perplexity.mean())
            non_freq = [test_pp[m] for m in METHODS
                        if m != 'freq' and not np.isnan(test_pp[m])]
            best = min(non_freq) if non_freq else float('nan')
            cells = []
            for m in METHODS:
                v = test_pp[m]
                if np.isnan(v):
                    cells.append('—')
                elif m == 'freq':
                    cells.append(f"_{v:.4f}_")
                elif abs(v - best) < 1e-4:
                    cells.append(f"**{v:.4f}**")
                else:
                    cells.append(f"{v:.4f}")
            print(f"| {N} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
