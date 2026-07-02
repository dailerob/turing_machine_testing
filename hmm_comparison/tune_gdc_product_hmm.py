"""GDC hyperparameter sweep on the 6-component product HMM.

Wider grid than `compare_product_hmm.py` to find a good GDC config for
the alphabet=64 setup. Reports h=1..5 excess perplexity, ranks configs
per horizon, plus a small set of baseline anchors (Parrot, CHMM, Freq).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from product_hmm import (build_product_hmm, sample_product,
                          make_state_preferred_components)
from gdc_torch_discrete import horizon_emission_many
from chmm_alergia_wrappers import fit_chmm
from discrete_parrot import DiscreteParrotPool

# Setup matches compare_product_hmm.py
N_COMPONENTS = 6
NS_PER_COMP = 2
NA_PER_COMP = 2
FANOUT = 2
MIN_PREF_PROB = 0.7
MIN_SELF_PROB = 0.9
N_TRAIN_SEQ = 40
N_TEST_SEQ = 20
SEQ_LEN = 20
HORIZONS = [1, 2, 3, 4, 5]
SEEDS = [0, 1, 2]
EPS = 1e-12
LAPLACE_FLOOR = 1.0 / 800.0
PROD_NA = NA_PER_COMP ** N_COMPONENTS

# Wide GDC grid
ALPHAS = [0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.99]
BETAS = [0.0, 0.005, 0.025, 0.05, 0.1, 0.2]


def build_chain_metadata(train_seqs):
    state_syms, term, start = [], [], []
    for seq in train_seqs:
        L = len(seq)
        if L == 0: continue
        state_syms.append(np.asarray(seq, dtype=np.int64))
        tm = np.zeros(L, dtype=bool); tm[-1] = True
        sm = np.zeros(L, dtype=bool); sm[0] = True
        term.append(tm); start.append(sm)
    return (np.concatenate(state_syms), np.concatenate(term),
            np.concatenate(start))


def excess_pp_pred(pred_dist, hmm, prefix, h):
    n = len(pred_dist)
    pred_safe = (1.0 - LAPLACE_FLOOR) * pred_dist + LAPLACE_FLOOR / n
    af = hmm.filter(prefix)
    true_d = hmm.horizon_emission(af, h)
    ce = -float(np.sum(true_d * np.log2(np.maximum(pred_safe, EPS))))
    floor = -float(np.sum(true_d * np.log2(np.maximum(true_d, EPS))))
    return ce, floor


def eval_gdc(cfg, hmm, train, test):
    sym, term, start = build_chain_metadata(train)
    primes = np.stack([np.asarray(p, dtype=np.int64) for p in test])
    pred = horizon_emission_many(
        symbol_of_state=sym, terminal_mask=term, start_mask=start,
        primes=primes, horizons=HORIZONS, nA=PROD_NA, **cfg,
        transition_type='self_loop', terminal_behavior='diffuse',
        initial_dist='sequence_starts',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=torch.float64).cpu().numpy()
    out = {}
    for hi, h in enumerate(HORIZONS):
        ces, floors = [], []
        for b, prefix in enumerate(test):
            ce, fl = excess_pp_pred(pred[b, hi], hmm, prefix, h)
            ces.append(ce); floors.append(fl)
        out[h] = 2.0 ** (np.mean(ces) - np.mean(floors))
    return out


def make_gdc_grid():
    configs = []
    for a in ALPHAS:
        # theta options depend on alpha (constraint α + θ ≤ 1)
        thetas = []
        if a + 0.005 <= 1: thetas.append(0.005)
        if a + 0.05 <= 1: thetas.append(0.05)
        if a + 0.1 <= 1: thetas.append(0.1)
        if a + 0.2 <= 1: thetas.append(0.2)
        for t in thetas:
            for b in BETAS:
                for scal in ['asymptotic', 'none' if b == 0.0 else 'asymptotic']:
                    if scal == 'none' and b != 0.0:
                        continue
                    name = f"a={a} θ={t} β={b} {scal}"
                    configs.append((name, dict(alpha=a, theta=t, beta=b,
                                                  beta_scaling=scal)))
    # dedup
    seen = set(); uniq = []
    for name, cfg in configs:
        key = (cfg['alpha'], cfg['theta'], cfg['beta'], cfg['beta_scaling'])
        if key in seen: continue
        seen.add(key); uniq.append((name, cfg))
    return uniq


def main():
    gdc_grid = make_gdc_grid()
    print(f"GDC grid: {len(gdc_grid)} configs")

    # Pre-generate seed data once
    seed_data = {}
    for seed in SEEDS:
        rng = np.random.default_rng(7000 + seed * 41)
        components = make_state_preferred_components(
            N_COMPONENTS, NS_PER_COMP, NA_PER_COMP, FANOUT, rng,
            min_pref_prob=MIN_PREF_PROB,
            min_self_prob=MIN_SELF_PROB)
        hmm = build_product_hmm(components)
        train = []
        for _ in range(N_TRAIN_SEQ):
            _, obs = sample_product(components, SEQ_LEN, rng)
            train.append(obs)
        test = []
        for _ in range(N_TEST_SEQ):
            _, obs = sample_product(components, SEQ_LEN, rng)
            test.append(obs)
        seed_data[seed] = (hmm, train, test)

    # Run GDC grid
    print(f"Sweeping GDC ({len(gdc_grid)} configs × {len(SEEDS)} seeds)...")
    t0 = time.time()
    gdc_results = {}
    for name, cfg in gdc_grid:
        per_seed = {h: [] for h in HORIZONS}
        for seed in SEEDS:
            hmm, train, test = seed_data[seed]
            res = eval_gdc(cfg, hmm, train, test)
            for h in HORIZONS:
                per_seed[h].append(res[h])
        gdc_results[name] = {h: float(np.mean(per_seed[h])) for h in HORIZONS}
    print(f"  done in {time.time()-t0:.1f}s")

    # Anchors
    print("Running anchors...")
    anchors = {}
    for name, fn, cfg in [
        ('Parrot L=4 K=100', lambda: None, ('parrot', 4, 100)),
        ('Parrot L=2 K=25', lambda: None, ('parrot', 2, 25)),
        ('Parrot L=4 K=25', lambda: None, ('parrot', 4, 25)),
        ('CHMM K=32', lambda: None, ('chmm', 32)),
        ('Freq', lambda: None, ('freq',)),
    ]:
        per_seed = {h: [] for h in HORIZONS}
        for seed in SEEDS:
            hmm, train, test = seed_data[seed]
            if cfg[0] == 'parrot':
                _, L, K = cfg
                pool = DiscreteParrotPool(
                    [np.asarray(s, dtype=np.int64) for s in train],
                    alphabet_size=PROD_NA, L=L)
                for h in HORIZONS:
                    ces, floors = [], []
                    for prefix in test:
                        prefix = np.asarray(prefix, dtype=np.int64)
                        pred = pool.predict_distribution(
                            prefix, h=h, K=K, alpha_prior=1.0)
                        ce, fl = excess_pp_pred(pred, hmm, prefix, h)
                        ces.append(ce); floors.append(fl)
                    per_seed[h].append(2.0 ** (np.mean(ces) - np.mean(floors)))
            elif cfg[0] == 'chmm':
                _, K = cfg
                model = fit_chmm(train, PROD_NA, K=K, n_em_iters=50)
                for h in HORIZONS:
                    ces, floors = [], []
                    for prefix in test:
                        pred = model.horizon_emission(prefix, h)
                        ce, fl = excess_pp_pred(pred, hmm, prefix, h)
                        ces.append(ce); floors.append(fl)
                    per_seed[h].append(2.0 ** (np.mean(ces) - np.mean(floors)))
            elif cfg[0] == 'freq':
                tr_chars = np.concatenate(
                    [np.asarray(s, dtype=np.int64) for s in train])
                counts = np.bincount(tr_chars,
                                      minlength=PROD_NA).astype(np.float64)
                pred = (counts + 1.0) / (counts.sum() + PROD_NA)
                for h in HORIZONS:
                    ces, floors = [], []
                    for prefix in test:
                        prefix = np.asarray(prefix, dtype=np.int64)
                        ce, fl = excess_pp_pred(pred, hmm, prefix, h)
                        ces.append(ce); floors.append(fl)
                    per_seed[h].append(2.0 ** (np.mean(ces) - np.mean(floors)))
        anchors[name] = {h: float(np.mean(per_seed[h])) for h in HORIZONS}

    # Print top GDC configs per horizon
    print(f"\n{N_COMPONENTS}-component product HMM (alphabet={PROD_NA}); "
          f"train: {N_TRAIN_SEQ} × {SEQ_LEN}; test: {N_TEST_SEQ} × {SEQ_LEN}\n")
    for h in HORIZONS:
        print(f"\n## Top GDC configs at h={h}")
        ranked = sorted(gdc_results.items(), key=lambda x: x[1][h])
        print(f"  {'config':<32} h={h}")
        for name, res in ranked[:10]:
            mark = ''
            for an, ar in anchors.items():
                if res[h] < ar[h]:
                    mark = '✓'
                    break
            print(f"  {name:<32} {res[h]:.4f}")
        print(f"  --- anchors ---")
        for an, ar in anchors.items():
            print(f"  {an:<32} {ar[h]:.4f}")

    # Per-horizon best-of-GDC vs anchors table
    print(f"\n## Best-of-GDC vs anchors\n")
    print(f"{'method':<32} " + " ".join(f"{f'h={h}':>9}" for h in HORIZONS))
    print('-' * 80)
    # Best GDC overall (by sum across horizons)
    overall_best = min(gdc_results.items(),
                        key=lambda x: sum(x[1][h] for h in HORIZONS))
    print(f"{'GDC best overall ('+overall_best[0]+')':<32} "
          + " ".join(f"{overall_best[1][h]:>9.4f}" for h in HORIZONS))
    # Best per horizon
    for h in HORIZONS:
        best = min(gdc_results.items(), key=lambda x: x[1][h])
        print(f"  GDC best at h={h:<3} ({best[0][:18]}) "
              + " ".join(f"{best[1][hh]:>9.4f}" for hh in HORIZONS))
    print()
    for name, res in anchors.items():
        print(f"{name:<32} " + " ".join(f"{res[h]:>9.4f}" for h in HORIZONS))


if __name__ == '__main__':
    main()
