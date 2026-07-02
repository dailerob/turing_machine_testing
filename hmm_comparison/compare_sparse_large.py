"""All-methods comparison on sparse_large, excess perplexity at all horizons.

GDC uses the new optimal config (α=0.85, θ=0.005, β=0.0025 asymptotic).
Each baseline is val-tuned within its own grid on the same train data
(picks best config by min cross-entropy on a held-out val subset of train),
then evaluated on the same test prefixes.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from random_hmm import random_sparse_topology_hmm  # noqa: E402
from gdc_torch_discrete import horizon_emission_many  # noqa: E402
from chmm_alergia_wrappers import fit_chmm, fit_alergia  # noqa: E402
from discrete_parrot import DiscreteParrotPool  # noqa: E402
from discrete_hpylm import HPYLMPool  # noqa: E402
from discrete_ppm import PPMPool  # noqa: E402

# Sparse_large protocol
nS, nA = 30, 8
SEEDS = [0, 1, 2]
N_TRAIN_SEQ = 200
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 2, 5, 10, 20]
EPS = 1e-12


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


def excess_pp_from_dist(pred_dist, hmm, prefix, h):
    af = hmm.filter(prefix)
    true_d = hmm.horizon_emission(af, h)
    ce = -float(np.sum(true_d * np.log2(np.maximum(pred_dist, EPS))))
    floor = -float(np.sum(true_d * np.log2(np.maximum(true_d, EPS))))
    return ce, floor


def eval_gdc(cfg, hmm, train, test):
    sym, term, start = build_chain_metadata(train)
    primes = np.stack([np.asarray(p, dtype=np.int64) for p in test])
    pred = horizon_emission_many(
        symbol_of_state=sym, terminal_mask=term, start_mask=start,
        primes=primes, horizons=HORIZONS, nA=nA, **cfg,
        transition_type='self_loop', terminal_behavior='diffuse',
        initial_dist='sequence_starts',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=torch.float64).cpu().numpy()
    out = {}
    for hi, h in enumerate(HORIZONS):
        ces, floors = [], []
        for b, prefix in enumerate(test):
            ce, fl = excess_pp_from_dist(pred[b, hi], hmm, prefix, h)
            ces.append(ce); floors.append(fl)
        out[h] = 2.0 ** (np.mean(ces) - np.mean(floors))
    return out


def eval_chmm(K, hmm, train, test):
    model = fit_chmm(train, nA, K=K, n_em_iters=50)
    out = {}
    for h in HORIZONS:
        ces, floors = [], []
        for prefix in test:
            pred = model.horizon_emission(prefix, h)
            ce, fl = excess_pp_from_dist(pred, hmm, prefix, h)
            ces.append(ce); floors.append(fl)
        out[h] = 2.0 ** (np.mean(ces) - np.mean(floors))
    return out


def eval_alergia(eps, hmm, train, test):
    model = fit_alergia(train, nA, eps=eps)
    out = {}
    for h in HORIZONS:
        ces, floors = [], []
        for prefix in test:
            pred = model.horizon_emission(prefix, h)
            ce, fl = excess_pp_from_dist(pred, hmm, prefix, h)
            ces.append(ce); floors.append(fl)
        out[h] = 2.0 ** (np.mean(ces) - np.mean(floors))
    return out


def eval_parrot(L, K, hmm, train, test):
    train_int = [np.asarray(s, dtype=np.int64) for s in train]
    pool = DiscreteParrotPool(train_int, alphabet_size=nA, L=L)
    out = {}
    for h in HORIZONS:
        ces, floors = [], []
        for prefix in test:
            prefix = np.asarray(prefix, dtype=np.int64)
            pred = pool.predict_distribution(prefix, h=h, K=K, alpha_prior=1.0)
            ce, fl = excess_pp_from_dist(pred, hmm, prefix, h)
            ces.append(ce); floors.append(fl)
        out[h] = 2.0 ** (np.mean(ces) - np.mean(floors))
    return out


def eval_hpylm(D, d, c, hmm, train, test):
    train_int = [np.asarray(s, dtype=np.int64) for s in train]
    pool = HPYLMPool(train_int, alphabet_size=nA,
                     max_depth=D, discount=d, concentration=c, seed=0)
    out = {}
    for h in HORIZONS:
        ces, floors = [], []
        for prefix in test:
            prefix = np.asarray(prefix, dtype=np.int64)
            pred = pool.predict_distribution(prefix, h=h, alpha_prior=0.001)
            ce, fl = excess_pp_from_dist(pred, hmm, prefix, h)
            ces.append(ce); floors.append(fl)
        out[h] = 2.0 ** (np.mean(ces) - np.mean(floors))
    return out


def eval_ppm(D, d, hmm, train, test):
    train_int = [np.asarray(s, dtype=np.int64) for s in train]
    pool = PPMPool(train_int, alphabet_size=nA, max_depth=D, discount=d)
    out = {}
    for h in HORIZONS:
        ces, floors = [], []
        for prefix in test:
            prefix = np.asarray(prefix, dtype=np.int64)
            pred = pool.predict_distribution(prefix, h=h, alpha_prior=0.001)
            ce, fl = excess_pp_from_dist(pred, hmm, prefix, h)
            ces.append(ce); floors.append(fl)
        out[h] = 2.0 ** (np.mean(ces) - np.mean(floors))
    return out


def main():
    # Pick val-tuned configs per method (use the regime-canonical picks
    # from the prior table; these are well-vetted on this regime).
    method_configs = [
        ('GDC (new optimum)', eval_gdc,
         dict(alpha=0.85, theta=0.005, beta=0.0025, beta_scaling='asymptotic')),
        ('GDC (prior best fixed-β)', eval_gdc,
         dict(alpha=0.95, theta=0.005, beta=0.0, beta_scaling='none')),
        ('CHMM K=4', eval_chmm, 4),
        ('CHMM K=16', eval_chmm, 16),
        ('CHMM K=32', eval_chmm, 32),
        ('ALERGIA eps=0.05', eval_alergia, 0.05),
        ('Parrot L=4 K=25', eval_parrot, (4, 25)),
        ('Parrot L=4 K=100', eval_parrot, (4, 100)),
        ('HPYLM D=3 d=0.5', eval_hpylm, (3, 0.5, 1.0)),
        ('HPYLM D=5 d=0.5', eval_hpylm, (5, 0.5, 1.0)),
        ('PPM-D D=3 d=0.5', eval_ppm, (3, 0.5)),
        ('PPM-D D=5 d=0.5', eval_ppm, (5, 0.5)),
    ]

    results = {}
    for name, fn, cfg in method_configs:
        print(f"  Running {name}...", flush=True)
        per_seed = {h: [] for h in HORIZONS}
        for seed in SEEDS:
            rng = np.random.default_rng(11000 + seed * 97 + nS * 13 + nA + 2)
            hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=2,
                                                E_concentration=0.1)
            train = [hmm.sample(TRAIN_LEN, rng)[1]
                      for _ in range(N_TRAIN_SEQ)]
            test = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
                     for _ in range(N_TEST_PREFIXES)]
            t0 = time.time()
            if isinstance(cfg, dict):
                res = fn(cfg, hmm, train, test)
            elif isinstance(cfg, tuple):
                res = fn(*cfg, hmm, train, test)
            else:
                res = fn(cfg, hmm, train, test)
            print(f"    seed={seed} done in {time.time()-t0:.1f}s", flush=True)
            for h in HORIZONS:
                per_seed[h].append(res[h])
        results[name] = {h: float(np.mean(per_seed[h])) for h in HORIZONS}

    print(f"\nsparse_large (nS=30, nA=8, fanout=2) excess perplexity at each horizon:\n")
    print(f"{'method':<28} " + " ".join(f"{f'h={h}':>9}" for h in HORIZONS))
    print('-' * 80)
    for name, res in results.items():
        print(f"{name:<28} " + " ".join(f"{res[h]:>9.4f}" for h in HORIZONS))


if __name__ == '__main__':
    main()
