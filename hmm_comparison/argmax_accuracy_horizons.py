"""Argmax-accuracy at h ∈ {1, 5, 20} on the four new regimes (TL=25).

Mirrors argmax_accuracy_new_regimes.py but evaluates each method at
multiple forecast horizons. For each h, the realized target is the
actual observation at position prefix_len + h - 1 in the test trajectory.

Note on h>1 semantics:
  - CHMM, GDC torch, ALERGIA: genuine h-step transition iteration
    (option A — what the user asked for).
  - Parrot: direct corpus lookup at offset h-1 from neighbour positions
    (effectively option A in spirit).
  - HPYLM, PPM-D: greedy argmax rollout (option B) — these models'
    predict_distribution(prefix, h>1) extends prefix by argmax and
    repredicts. Marked with † in the writeup.

Two-pass design: GDC torch sweep in main process, CPU methods in
mp.Pool fork.
"""
from __future__ import annotations
import os, sys, time, multiprocessing as mp
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from random_hmm import (random_bimodal_hmm, random_cyclic_hmm,
                       random_binary_deep_hmm,
                       random_reset_chain_hmm)
from generative_dense_chain import GenerativeDenseChain
from gdc_torch_discrete import horizon_emission_many

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
TRAIN_LEN = 25
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 5, 20]
MAX_H = max(HORIZONS)

GDC_ALPHAS = [0.10, 0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
GDC_BETAS  = [0.00, 0.001, 0.01, 0.05, 0.10, 0.20, 0.30]
GDC_THETA  = 0.001

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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

OUT_RAW = os.path.join(HERE, 'argmax_accuracy_horizons.csv')


def make_hmm(kind, nS, nA, rng):
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


def setup_cell_data(regime_name, nS, nA, kind, seed, N):
    kind_tag = {'bimodal': 0, 'cyclic': 10, 'binary_deep': 20,
                'reset_chain': 30}[kind]
    rng = np.random.default_rng(70000 + kind_tag + seed * 137 + nS * 7
                                + nA * 11)
    hmm = make_hmm(kind, nS, nA, rng)
    full_train = [hmm.sample(TRAIN_LEN, rng)[1]
                  for _ in range(max(N_TRAIN_VALUES))]
    # Sample TEST_PREFIX_LEN + MAX_H observations per test prefix
    test_pf_full = [hmm.sample(TEST_PREFIX_LEN + MAX_H, rng)[1]
                    for _ in range(N_TEST_PREFIXES)]
    test_pf = [t[:TEST_PREFIX_LEN] for t in test_pf_full]
    realized_per_h = {}
    for h in HORIZONS:
        realized_per_h[h] = np.array(
            [int(t[TEST_PREFIX_LEN + h - 1]) for t in test_pf_full])
    # True next-symbol distributions per h (for cross-entropy & oracle)
    true_next_per_h = {}
    floor_bits_per_h = {}
    for h in HORIZONS:
        Th = np.linalg.matrix_power(hmm.T, h)
        true_next = np.zeros((N_TEST_PREFIXES, nA))
        floor_bits = np.zeros(N_TEST_PREFIXES)
        for i, p in enumerate(test_pf):
            a = hmm.filter(p)
            nd = a @ Th @ hmm.E
            true_next[i] = nd
            floor_bits[i] = -float(np.sum(nd
                                          * np.log2(np.maximum(nd, 1e-12))))
        true_next_per_h[h] = true_next
        floor_bits_per_h[h] = floor_bits
    return (hmm, full_train[:N], test_pf, realized_per_h,
            true_next_per_h, floor_bits_per_h)


def metrics(preds, true_next, floor_bits, realized):
    pred_safe = np.maximum(preds, 1e-12)
    ce = -np.sum(true_next * np.log2(pred_safe), axis=1)
    ex_pp = float(2 ** (ce.mean() - floor_bits.mean()))
    pred_argmax = np.argmax(preds, axis=1)
    acc = float(np.mean(pred_argmax == realized))
    return ex_pp, acc


def gdc_sweep_cell(args):
    regime_name, nS, nA, kind, seed, N = args
    (hmm, train, test_pf, realized_per_h,
     true_next_per_h, floor_bits_per_h) = setup_cell_data(
        regime_name, nS, nA, kind, seed, N)
    seq_arrays = [s.reshape(-1, 1).astype(np.int64) for s in train]
    gdc = GenerativeDenseChain(
        seq_arrays, alpha=GDC_ALPHAS[0], theta=GDC_THETA, gamma=0.0,
        beta=GDC_BETAS[0], transition_type='self_loop',
        initial_dist='uniform', terminal_behavior='absorb')
    sym = gdc.states[:, 0].astype(np.int64)
    primes = np.stack([np.asarray(p, dtype=np.int64) for p in test_pf])
    rows = []
    base = dict(regime=regime_name, nS=nS, nA=nA, seed=seed, N_train=N)
    for alpha in GDC_ALPHAS:
        for beta in GDC_BETAS:
            out = horizon_emission_many(
                symbol_of_state=sym, terminal_mask=gdc.terminal_mask,
                start_mask=gdc.start_mask,
                primes=primes, horizons=HORIZONS, nA=nA,
                alpha=alpha, theta=GDC_THETA, beta=beta,
                transition_type='self_loop',
                terminal_behavior='absorb', initial_dist='uniform',
                device=DEVICE, dtype=DTYPE)
            preds_per_h = out.cpu().numpy()  # (B, H, nA)
            for hi, h in enumerate(HORIZONS):
                preds = preds_per_h[:, hi, :]
                ex_pp, acc = metrics(preds, true_next_per_h[h],
                                     floor_bits_per_h[h],
                                     realized_per_h[h])
                rows.append(dict(base, model_class='gdc',
                                 model=f'gdc-a{alpha}-b{beta}',
                                 horizon=h, excess_pp=ex_pp,
                                 acc_realized=acc))
    return rows


def cpu_sweep_cell(args):
    regime_name, nS, nA, kind, seed, N = args
    sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
    from chmm_alergia_wrappers import fit_chmm, fit_alergia
    from discrete_parrot import DiscreteParrotPool
    from discrete_hpylm import HPYLMPool
    from discrete_ppm import PPMPool

    (hmm, train, test_pf, realized_per_h,
     true_next_per_h, floor_bits_per_h) = setup_cell_data(
        regime_name, nS, nA, kind, seed, N)
    rows = []
    base = dict(regime=regime_name, nS=nS, nA=nA, seed=seed, N_train=N)

    def add_for_horizons(model_class, model_name, model_obj):
        for h in HORIZONS:
            preds = np.stack([model_obj.horizon_emission(p, h=h)
                              if hasattr(model_obj, 'horizon_emission')
                              else model_obj.predict_distribution(np.asarray(p),
                                                                   h=h)
                              for p in test_pf])
            ex_pp, acc = metrics(preds, true_next_per_h[h],
                                 floor_bits_per_h[h],
                                 realized_per_h[h])
            rows.append(dict(base, model_class=model_class,
                             model=model_name, horizon=h,
                             excess_pp=ex_pp, acc_realized=acc))

    def add_pool_for_horizons(model_class, model_name, pool, **predict_kw):
        for h in HORIZONS:
            preds = np.stack([pool.predict_distribution(np.asarray(p), h=h,
                                                         **predict_kw)
                              for p in test_pf])
            ex_pp, acc = metrics(preds, true_next_per_h[h],
                                 floor_bits_per_h[h],
                                 realized_per_h[h])
            rows.append(dict(base, model_class=model_class,
                             model=model_name, horizon=h,
                             excess_pp=ex_pp, acc_realized=acc))

    for K in CHMM_KS:
        try:
            m = fit_chmm(train, nA, K=K, n_em_iters=50)
            add_for_horizons('chmm', f'chmm-K{K}', m)
        except Exception:
            pass
    try:
        m = fit_alergia(train, nA, eps=ALERGIA_EPS)
        add_for_horizons('alergia', f'alergia-eps{ALERGIA_EPS}', m)
    except Exception:
        pass
    parrot_pools = {L: DiscreteParrotPool(train, alphabet_size=nA, L=L)
                    for L in PARROT_LS}
    for L in PARROT_LS:
        for K in PARROT_KS:
            for ap in PARROT_ALPHAS:
                add_pool_for_horizons(
                    'parrot', f'parrot-L{L}-K{K}-a{ap}',
                    parrot_pools[L], K=K, alpha_prior=ap)
    for D in HPYLM_DEPTHS:
        for d in HPYLM_DISCOUNTS:
            for c in HPYLM_CONCS:
                pool = HPYLMPool(train, alphabet_size=nA, max_depth=D,
                                 discount=d, concentration=c, seed=seed)
                add_pool_for_horizons('hpylm', f'hpylm-D{D}-d{d}-a{c}',
                                       pool,
                                       alpha_prior=HPYLM_ALPHA_PRIOR)
    for D in PPM_DEPTHS:
        for d in PPM_DISCOUNTS:
            pool = PPMPool(train, alphabet_size=nA, max_depth=D,
                           discount=d)
            add_pool_for_horizons('ppm', f'ppm-D{D}-d{d}', pool,
                                   alpha_prior=PPM_ALPHA_PRIOR)

    # Freq baseline (same prediction at all horizons since data-only)
    counts = np.zeros(nA)
    for s in train:
        for v in np.asarray(s, dtype=np.int64):
            counts[v] += 1
    freq = (counts + 1e-6) / (counts.sum() + nA * 1e-6)
    preds_freq = np.tile(freq, (N_TEST_PREFIXES, 1))
    for h in HORIZONS:
        ex_pp, acc = metrics(preds_freq, true_next_per_h[h],
                             floor_bits_per_h[h], realized_per_h[h])
        rows.append(dict(base, model_class='freq', model='freq',
                         horizon=h, excess_pp=ex_pp, acc_realized=acc))
    # Oracle (true distribution per h)
    for h in HORIZONS:
        ex_pp, acc = metrics(true_next_per_h[h], true_next_per_h[h],
                             floor_bits_per_h[h], realized_per_h[h])
        rows.append(dict(base, model_class='oracle', model='oracle',
                         horizon=h, excess_pp=ex_pp, acc_realized=acc))
    # Uniform
    preds_uniform = np.full((N_TEST_PREFIXES, nA), 1.0 / nA)
    for h in HORIZONS:
        ex_pp, acc = metrics(preds_uniform, true_next_per_h[h],
                             floor_bits_per_h[h], realized_per_h[h])
        rows.append(dict(base, model_class='uniform', model='uniform',
                         horizon=h, excess_pp=ex_pp, acc_realized=acc))

    return rows


def main():
    tasks = [(name, nS, nA, kind, seed, N)
             for (name, nS, nA, kind) in REGIMES
             for seed in SEEDS for N in N_TRAIN_VALUES]
    n_cells = len(tasks)
    print(f"=== argmax accuracy at h ∈ {HORIZONS}, TL={TRAIN_LEN} ===")
    print(f"  Cells: {n_cells}\n", flush=True)

    t0 = time.time()
    gdc_rows = []
    for i, args in enumerate(tasks):
        gdc_rows.extend(gdc_sweep_cell(args))
        if (i + 1) % 12 == 0 or (i + 1) == n_cells:
            print(f"  GDC pass: {i+1}/{n_cells} cells "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    print()

    n_workers = max(1, min(8, (os.cpu_count() or 4) - 1))
    print(f"  Starting CPU pass with {n_workers} workers...", flush=True)
    t1 = time.time()
    cpu_rows = []
    with mp.Pool(processes=n_workers) as pool:
        done = 0
        for cell_rows in pool.imap_unordered(cpu_sweep_cell, tasks,
                                              chunksize=1):
            cpu_rows.extend(cell_rows); done += 1
            if done % 6 == 0 or done == n_cells:
                print(f"  CPU pass: {done}/{n_cells} cells "
                      f"[{time.time()-t1:.0f}s]", flush=True)

    df = pd.DataFrame(gdc_rows + cpu_rows)
    df.to_csv(OUT_RAW, index=False)
    print(f"\nWrote {OUT_RAW} ({len(df)} rows, "
          f"{time.time()-t0:.0f}s total)\n")

    METHODS = ['gdc', 'chmm', 'alergia', 'parrot', 'hpylm', 'ppm',
               'freq', 'oracle', 'uniform']
    PRETTY  = {'gdc':'GDC','chmm':'CHMM','alergia':'ALERGIA',
               'parrot':'Parrot','hpylm':'HPYLM†','ppm':'PPM-D†',
               'freq':'Freq','oracle':'Oracle','uniform':'Unif'}

    for h in HORIZONS:
        print(f"\n## Argmax accuracy at h={h} (val-pick on h={h} excess_pp)\n")
        if h > 1:
            print("† HPYLM and PPM-D use greedy rollout for h>1 (see top of file).\n")
        for regime, *_ in REGIMES:
            print(f"### {regime}\n")
            print("| N | " + " | ".join(PRETTY[m] for m in METHODS) + " |")
            print("|---:|" + "---:|" * len(METHODS))
            for N in N_TRAIN_VALUES:
                cell = df[(df.regime == regime) & (df.N_train == N)
                          & (df.horizon == h)]
                test_acc = {}
                for m in METHODS:
                    csub = cell[cell.model_class == m]
                    cval = csub[csub.seed.isin(VAL_SEEDS)]
                    ctest = csub[csub.seed.isin(TEST_SEEDS)]
                    if csub.empty:
                        test_acc[m] = float('nan'); continue
                    if m in ('alergia', 'freq', 'oracle', 'uniform'):
                        test_acc[m] = float(ctest.acc_realized.mean())
                    else:
                        val_means = (cval.groupby('model')['excess_pp']
                                         .mean())
                        if val_means.empty:
                            test_acc[m] = float('nan'); continue
                        pick = val_means.idxmin()
                        test_acc[m] = float(ctest[ctest.model == pick]
                                            .acc_realized.mean())
                non_special = [test_acc[m] for m in METHODS
                               if m not in ('oracle', 'uniform', 'freq')
                               and not np.isnan(test_acc[m])]
                best = max(non_special) if non_special else float('nan')
                cells = []
                for m in METHODS:
                    v = test_acc[m]
                    if np.isnan(v):
                        cells.append('—')
                    elif m in ('oracle', 'uniform', 'freq'):
                        cells.append(f"_{v:.3f}_")
                    elif abs(v - best) < 1e-3:
                        cells.append(f"**{v:.3f}**")
                    else:
                        cells.append(f"{v:.3f}")
                print(f"| {N} | " + " | ".join(cells) + " |")
            print()


if __name__ == "__main__":
    main()
