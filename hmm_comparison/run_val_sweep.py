"""Validation sweep for leakage-free hyperparameter selection.

Mirrors run_perplexity_sweep.py exactly except:
  - seeds = [3, 4, 5]  (disjoint from the existing test seeds [0, 1, 2])
  - GDC uses the torch kernel (gdc_torch_discrete) for speed
  - writes to perplexity_val_results.csv

Pair with build_leakage_free_table.py: pick best per (regime, N, model_class)
on val excess PP at h=1, then look up the corresponding test excess PP from
the existing perplexity_sweep_results.csv.
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np
import torch

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
SEEDS = [3, 4, 5]                     # disjoint from test seeds [0,1,2]
N_TRAIN_VALUES = [25, 100, 400]
TRAIN_LEN = 50
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

OUT_CSV = os.path.join(HERE, 'perplexity_val_results.csv')


def gdc_predictions_torch(seqs, primes, horizons, nA, alpha, theta, beta,
                           device='cuda', dtype=torch.float32):
    """Build GDC state arrays and run the torch batched kernel."""
    from gdc_torch_discrete import horizon_emission_many
    sym_list = []
    terminal_mask = []
    start_mask = []
    for s in seqs:
        L_s = len(s)
        sym_list.extend(int(x) for x in s)
        for i in range(L_s):
            terminal_mask.append(i == L_s - 1)
            start_mask.append(i == 0)
    sym = np.asarray(sym_list, dtype=np.int64)
    terminal_mask = np.asarray(terminal_mask, dtype=bool)
    start_mask = np.asarray(start_mask, dtype=bool)
    primes_arr = np.stack([np.asarray(p, dtype=np.int64) for p in primes])
    out = horizon_emission_many(
        symbol_of_state=sym,
        terminal_mask=terminal_mask,
        start_mask=start_mask,
        primes=primes_arr,
        horizons=horizons,
        nA=nA,
        alpha=alpha, theta=theta, beta=beta,
        transition_type='self_loop',
        terminal_behavior='diffuse',
        initial_dist='sequence_starts',
        device=device, dtype=dtype,
    )
    return out.cpu().numpy()    # (B, n_horizons, nA)


def metrics_from_predictions(preds, hmm, test_prefixes, horizons, eps=1e-12):
    """preds: (B, n_horizons, nA). Compute excess PP / MSE per horizon."""
    rows = {}
    for j, h in enumerate(horizons):
        ce_list = []; floor_list = []; mse_list = []
        for i, prefix in enumerate(test_prefixes):
            alpha_state = hmm.filter(prefix)
            true_dist = hmm.horizon_emission(alpha_state, h)
            pred = preds[i, j, :]
            pred_safe = np.maximum(pred, eps)
            true_safe = np.maximum(true_dist, eps)
            ce = -float(np.sum(true_dist * np.log2(pred_safe)))
            floor = -float(np.sum(true_dist * np.log2(true_safe)))
            ce_list.append(ce); floor_list.append(floor)
            mse_list.append(float(np.mean((pred - true_dist) ** 2)))
        ce = float(np.mean(ce_list))
        floor = float(np.mean(floor_list))
        rows[h] = dict(
            mse=float(np.mean(mse_list)),
            cross_entropy_bits=ce, entropy_floor_bits=floor,
            perplexity=2.0**ce, entropy_floor_perplexity=2.0**floor,
            excess_perplexity=2.0**(ce - floor),
        )
    return rows


def run_cell(regime_name, nS, nA, kind, E_conc, fanout, seed):
    from random_hmm import random_dense_hmm, random_sparse_topology_hmm
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
    for N in N_TRAIN_VALUES:
        train = full_train[:N]
        # GDC: torch batched
        for (a, t, b) in GDC_CONFIGS:
            preds = gdc_predictions_torch(train, test_pf, HORIZONS, nA,
                                            a, t, b)
            metrics = metrics_from_predictions(preds, hmm, test_pf, HORIZONS)
            for h in HORIZONS:
                m = metrics[h]
                rows.append(dict(regime=regime_name, nS=nS, nA=nA, seed=seed,
                                 N_train=N, model=f'gdc-a{a}-t{t}-b{b}',
                                 horizon=h, **m))
        # CHMM: numpy EM
        for K in CHMM_KS:
            try:
                m_chmm = fit_chmm(train, nA, K=K, n_em_iters=50)
                mse = mse_at_horizons(m_chmm, hmm, test_pf, HORIZONS)
                ppl = perplexity_at_horizons(m_chmm, hmm, test_pf, HORIZONS)
                for h in HORIZONS:
                    r = ppl[h]
                    rows.append(dict(regime=regime_name, nS=nS, nA=nA,
                                     seed=seed, N_train=N, model=f'chmm-K{K}',
                                     horizon=h, mse=mse[h], **r))
            except Exception as e:
                sys.stderr.write(f"[chmm K={K} {regime_name} N={N} s{seed}] {e}\n")
        try:
            m_a = fit_alergia(train, nA, eps=ALERGIA_EPS)
            mse = mse_at_horizons(m_a, hmm, test_pf, HORIZONS)
            ppl = perplexity_at_horizons(m_a, hmm, test_pf, HORIZONS)
            for h in HORIZONS:
                r = ppl[h]
                rows.append(dict(regime=regime_name, nS=nS, nA=nA, seed=seed,
                                 N_train=N, model='alergia-eps0.05',
                                 horizon=h, mse=mse[h], **r))
        except Exception as e:
            sys.stderr.write(f"[alergia {regime_name} N={N} s{seed}] {e}\n")
    return rows


def main():
    print(f"Val sweep: seeds={SEEDS}, GDC via torch on "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    fields = ['regime', 'nS', 'nA', 'seed', 'N_train', 'model', 'horizon',
              'mse', 'cross_entropy_bits', 'entropy_floor_bits',
              'perplexity', 'entropy_floor_perplexity', 'excess_perplexity']
    f = open(OUT_CSV, 'w', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); f.flush()

    t_start = time.time()
    cells = [(name, nS, nA, kind, conc, fanout, seed)
             for (name, nS, nA, kind, conc, fanout) in REGIMES
             for seed in SEEDS]
    for i, args in enumerate(cells):
        t0 = time.time()
        rows = run_cell(*args)
        for r in rows:
            w.writerow(r)
        f.flush()
        print(f"  [{i+1}/{len(cells)}] {args[0]} seed={args[6]}  "
              f"{time.time()-t0:.0f}s  total={time.time()-t_start:.0f}s",
              flush=True)
    f.close()
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
