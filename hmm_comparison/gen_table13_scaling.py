"""Canonical generator for paper Table 13 (tab:hmm_scaling).

RECONSTRUCTED 2026-06 (companion to gen_table7_forecasting.py; see
paper/PROTOCOL_STANDARDIZATION.md S7). Data-scaling on the SAME four
regimes as Table 7 (cyclic, reset_chain, bimodal, sparse), at horizon
h=1 only, sweeping the number of training sequences N in {1,3,5,10,25}.

By reusing gen_table7_forecasting.gen_data with the SAME seeds, the
N=25 column reproduces Table 7's h=1 column exactly (same experiment).

Leakage-free: per (regime, N, method) the config is selected on a
disjoint set of validation HMM seeds and reported on the test HMM seeds
(identical protocol to Table 7). Same per-method grids as Table 7.

Two passes (CPU baselines via mp.Pool; GPU GDC in main). Writes
table13_scaling_results.csv and prints the per-regime N-scaling tables.

Usage:
  python gen_table13_scaling.py            # full run
  python gen_table13_scaling.py --smoke    # tiny sanity check
"""
from __future__ import annotations
import os, sys, csv, time, argparse, multiprocessing as mp
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'algorithmic_benchmarks'))

import gen_table7_forecasting as T7
from gen_table7_forecasting import (REGIMES, gen_data, _gdc_state_arrays,
                                     _PoolForecaster, _KN3Forecaster,
                                     freq_excess_pp, METHODS, PRETTY)

N_VALUES = [1, 3, 5, 10, 25]
HORIZON = 1
OUT_CSV = os.path.join(HERE, 'table13_scaling_results.csv')


def _apply_smoke():
    T7._apply_smoke()
    global N_VALUES
    N_VALUES = [1, 5, 25]


# ---------------------------------------------------------------------------
# Pass A: CPU baselines (all N for one cell)
# ---------------------------------------------------------------------------
def run_baselines_cell(args):
    regime_name, nS, nA, kind, seed = args
    sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'algorithmic_benchmarks'))
    from evaluation import perplexity_at_horizons
    from chmm_alergia_wrappers import fit_chmm, fit_alergia
    from discrete_parrot import DiscreteParrotPool
    from discrete_hpylm import HPYLMPool
    from discrete_ppm import PPMPool
    from kn3_eval import KN3Model

    hmm, full_train, test_pf = gen_data(kind, nS, nA, seed)
    rows = []

    def record(N, model_class, model_name, model):
        ppl = perplexity_at_horizons(model, hmm, test_pf, [HORIZON])
        rows.append(dict(regime=regime_name, seed=seed, N_train=N,
                         model_class=model_class, model=model_name,
                         horizon=HORIZON,
                         excess_perplexity=ppl[HORIZON]['excess_perplexity']))

    for N in N_VALUES:
        train = full_train[:N]
        for K in T7.CHMM_KS:
            try:
                record(N, 'chmm', f'chmm-K{K}',
                       fit_chmm(train, nA, K=K, n_em_iters=50))
            except Exception as e:
                sys.stderr.write(f"[chmm K={K} {regime_name} N{N} s{seed}] {e}\n")
        for eps in T7.ALERGIA_EPS:
            try:
                record(N, 'alergia', f'alergia-eps{eps}',
                       fit_alergia(train, nA, eps=eps))
            except Exception as e:
                sys.stderr.write(f"[alergia {regime_name} N{N} s{seed}] {e}\n")
        for L in T7.PARROT_LS:
            pool = DiscreteParrotPool(train, alphabet_size=nA, L=L)
            for K in T7.PARROT_KS:
                for ap in T7.PARROT_ALPHAS:
                    record(N, 'parrot', f'parrot-L{L}-K{K}-a{ap}',
                           _PoolForecaster(pool, K=K, alpha_prior=ap))
        for D in T7.HPYLM_DEPTHS:
            for d in T7.HPYLM_DISCOUNTS:
                for c in T7.HPYLM_CONCS:
                    pool = HPYLMPool(train, alphabet_size=nA, max_depth=D,
                                     discount=d, concentration=c, seed=seed)
                    record(N, 'hpylm', f'hpylm-D{D}-d{d}-c{c}',
                           _PoolForecaster(pool, alpha_prior=T7.HPYLM_ALPHA_PRIOR))
        for D in T7.PPM_DEPTHS:
            for d in T7.PPM_DISCOUNTS:
                pool = PPMPool(train, alphabet_size=nA, max_depth=D, discount=d)
                record(N, 'ppm', f'ppm-D{D}-d{d}',
                       _PoolForecaster(pool, alpha_prior=T7.PPM_ALPHA_PRIOR))
        for d in T7.KN3_DISCOUNTS:
            m = KN3Model(V=nA, discount=d)
            m.fit([np.asarray(s, dtype=np.int64) for s in train])
            record(N, 'kn3', f'kn3-d{d}', _KN3Forecaster(m))
        rows.append(dict(regime=regime_name, seed=seed, N_train=N,
                         model_class='freq', model='freq', horizon=HORIZON,
                         excess_perplexity=freq_excess_pp(hmm, train, test_pf,
                                                          h=HORIZON)))
    return rows


# ---------------------------------------------------------------------------
# Pass B: GDC on GPU
# ---------------------------------------------------------------------------
def run_gdc_all(device, dtype):
    from gdc_torch_discrete import horizon_emission_many
    at_pairs = [(a, t) for a in T7.GDC_ALPHAS for t in T7.GDC_THETAS
                if a + t <= 1.0 + 1e-9]
    fc_variants = [('single', None, None), ('dual', 1.0, 0.0)]
    rows = []
    cells = [(name, nS, nA, kind, seed)
             for (name, nS, nA, kind) in REGIMES
             for seed in (T7.TEST_SEEDS + T7.VAL_SEEDS)]
    n_cfg = len(at_pairs) * len(T7.GDC_BETAS) * len(fc_variants)
    print(f"  GDC: {len(cells)} cells x {len(N_VALUES)} N x {n_cfg} cfg",
          flush=True)
    t0 = time.time()
    for ci, (regime_name, nS, nA, kind, seed) in enumerate(cells):
        hmm, full_train, test_pf = gen_data(kind, nS, nA, seed)
        primes = np.stack([np.asarray(p, dtype=np.int64) for p in test_pf])
        alphas = [hmm.filter(p) for p in test_pf]
        true_d = np.stack([hmm.horizon_emission(a, HORIZON) for a in alphas])
        floor_bits = -np.sum(true_d * np.log2(np.maximum(true_d, 1e-12)),
                             axis=1).mean()
        for N in N_VALUES:
            sym, term, start = _gdc_state_arrays(full_train[:N])
            for (a, t) in at_pairs:
                for b in T7.GDC_BETAS:
                    for (fc_name, a_fc, t_fc) in fc_variants:
                        preds = horizon_emission_many(
                            symbol_of_state=sym, terminal_mask=term,
                            start_mask=start, primes=primes, horizons=[HORIZON],
                            nA=nA, alpha=a, theta=t, beta=b,
                            transition_type='self_loop',
                            terminal_behavior='diffuse',
                            initial_dist='sequence_starts',
                            beta_scaling='asymptotic',
                            alpha_forecast=a_fc, theta_forecast=t_fc,
                            device=device, dtype=dtype).cpu().numpy()
                        pr = np.maximum(preds[:, 0, :], 1e-12)
                        ce = -np.sum(true_d * np.log2(pr), axis=1).mean()
                        rows.append(dict(
                            regime=regime_name, seed=seed, N_train=N,
                            model_class='gdc',
                            model=f'gdc-a{a}-t{t}-b{b}-{fc_name}',
                            horizon=HORIZON,
                            excess_perplexity=float(2.0 ** (ce - floor_bits))))
        if (ci + 1) % 10 == 0 or ci == len(cells) - 1:
            print(f"    [{ci+1}/{len(cells)}] cells  [{time.time()-t0:.0f}s]",
                  flush=True)
    return rows


def aggregate_and_print(df):
    val_set, test_set = set(T7.VAL_SEEDS), set(T7.TEST_SEEDS)
    for regime, *_ in REGIMES:
        print(f"### {regime}\n")
        print("| Method | " + " | ".join(f"N={N}" for N in N_VALUES) + " |")
        print("|---|" + "---:|" * len(N_VALUES))
        rsub = df[df.regime == regime]
        for m in METHODS:
            cells = []
            for N in N_VALUES:
                sub = rsub[(rsub.model_class == m) & (rsub.N_train == N)]
                if sub.empty:
                    cells.append("---"); continue
                if m == 'freq':
                    cells.append(f"{sub[sub.seed.isin(test_set)].excess_perplexity.mean():.3f}")
                    continue
                vmeans = sub[sub.seed.isin(val_set)].groupby('model')['excess_perplexity'].mean()
                if vmeans.empty:
                    cells.append("---"); continue
                pick = vmeans.idxmin()
                test = sub[(sub.seed.isin(test_set)) & (sub.model == pick)]
                cells.append(f"{test.excess_perplexity.mean():.3f}")
            print(f"| {PRETTY[m]} | " + " | ".join(cells) + " |")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        _apply_smoke(); print("=== SMOKE MODE ===")

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    print(f"device={device}  N_VALUES={N_VALUES}  "
          f"val={len(T7.VAL_SEEDS)} test={len(T7.TEST_SEEDS)}", flush=True)

    cells = [(name, nS, nA, kind, seed)
             for (name, nS, nA, kind) in REGIMES
             for seed in (T7.TEST_SEEDS + T7.VAL_SEEDS)]

    n_workers = args.workers or max(1, min(20, (os.cpu_count() or 4) - 1))
    print(f"\n[Pass A] CPU baselines: {len(cells)} cells x {len(N_VALUES)} N, "
          f"{n_workers} workers", flush=True)
    tA = time.time(); base_rows = []
    with mp.Pool(processes=n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_baselines_cell, cells,
                                                  chunksize=1)):
            base_rows.extend(r)
            if (i + 1) % 10 == 0 or i == len(cells) - 1:
                print(f"    [{i+1}/{len(cells)}]  [{time.time()-tA:.0f}s]",
                      flush=True)
    print(f"[Pass A] done ({time.time()-tA:.0f}s, {len(base_rows)} rows)")

    print(f"\n[Pass B] GDC on {device}", flush=True)
    tB = time.time(); gdc_rows = run_gdc_all(device, dtype)
    print(f"[Pass B] done ({time.time()-tB:.0f}s, {len(gdc_rows)} rows)")

    all_rows = base_rows + gdc_rows
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['regime', 'seed', 'N_train',
                                          'model_class', 'model', 'horizon',
                                          'excess_perplexity'])
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV} ({len(all_rows)} rows)\n")
    aggregate_and_print(pd.DataFrame(all_rows))


if __name__ == "__main__":
    main()
