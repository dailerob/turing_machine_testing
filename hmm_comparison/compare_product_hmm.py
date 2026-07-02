"""Forecasting comparison on a product HMM made of N independent
binary-alphabet components.

Setup:
  - N=3 component HMMs, each with 2 hidden states and 2 emission symbols
    (binary), sparse topology (fanout=2, i.e. fully connected since
    fanout >= nS) with E_concentration=0.1 for some bias
  - Product alphabet = 2^3 = 8 symbols
  - Train: 20 sequences of length 20
  - Test: 20 sequences of length 20 (used as full-prefix forecast inputs)
  - Horizons: 1, 2, 5, 10, 20
  - Metric: excess perplexity (lower bound 1.0 = at the entropy floor)
  - Methods: GDC (best config from sparse_large), CHMM (K=4, 8, 16),
    Parrot (val-tuned L), HPYLM, PPM-D, KN-3
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from product_hmm import (build_product_hmm, sample_product,                # noqa: E402
                          make_state_preferred_components)
from gdc_torch_discrete import horizon_emission_many                      # noqa: E402
from chmm_alergia_wrappers import fit_chmm, fit_alergia                   # noqa: E402
from discrete_parrot import DiscreteParrotPool                            # noqa: E402
from discrete_hpylm import HPYLMPool                                       # noqa: E402
from discrete_ppm import PPMPool                                           # noqa: E402

# Add KN-3
sys.path.insert(0, os.path.join(ROOT, 'algorithmic_benchmarks'))
from kn3_eval import KN3Model                                              # noqa: E402

# Setup
N_COMPONENTS = 6
NS_PER_COMP = 2
NA_PER_COMP = 2
FANOUT = 2
MIN_PREF_PROB = 0.7  # state-preferred emissions: E[i, i%nA] >= this
MIN_SELF_PROB = 0.9  # self-transitions: T[i, i] >= this (slow mixing)
N_TRAIN_SEQ = 40
N_TEST_SEQ = 20
SEQ_LEN = 20
HORIZONS = [1, 2, 3, 4, 5]
SEEDS = [0, 1, 2]
EPS = 1e-12

PROD_NA = NA_PER_COMP ** N_COMPONENTS  # 8


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


LAPLACE_FLOOR = 1.0 / 400.0  # ~ 1/n_train_symbols; covers OOV products


def excess_pp_pred(pred_dist, hmm, prefix, h):
    # Apply a uniform Laplace floor so that methods which give exactly
    # zero probability to symbols missing from training (GDC delta-emission
    # output, KN-3 with no continuation count, ALERGIA on unseen labels)
    # don't dominate the comparison via 1/0 cross-entropy. This is the
    # standard "add-α smoothing" used by HPYLM/PPM-D internally; we apply
    # it post-hoc to every method uniformly.
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


def eval_chmm(K, hmm, train, test):
    model = fit_chmm(train, PROD_NA, K=K, n_em_iters=50)
    out = {}
    for h in HORIZONS:
        ces, floors = [], []
        for prefix in test:
            pred = model.horizon_emission(prefix, h)
            ce, fl = excess_pp_pred(pred, hmm, prefix, h)
            ces.append(ce); floors.append(fl)
        out[h] = 2.0 ** (np.mean(ces) - np.mean(floors))
    return out


def eval_alergia(eps, hmm, train, test):
    model = fit_alergia(train, PROD_NA, eps=eps)
    out = {}
    for h in HORIZONS:
        ces, floors = [], []
        for prefix in test:
            pred = model.horizon_emission(prefix, h)
            ce, fl = excess_pp_pred(pred, hmm, prefix, h)
            ces.append(ce); floors.append(fl)
        out[h] = 2.0 ** (np.mean(ces) - np.mean(floors))
    return out


def eval_parrot(L, K, hmm, train, test):
    train_int = [np.asarray(s, dtype=np.int64) for s in train]
    pool = DiscreteParrotPool(train_int, alphabet_size=PROD_NA, L=L)
    out = {}
    for h in HORIZONS:
        ces, floors = [], []
        for prefix in test:
            prefix = np.asarray(prefix, dtype=np.int64)
            pred = pool.predict_distribution(prefix, h=h, K=K, alpha_prior=1.0)
            ce, fl = excess_pp_pred(pred, hmm, prefix, h)
            ces.append(ce); floors.append(fl)
        out[h] = 2.0 ** (np.mean(ces) - np.mean(floors))
    return out


def eval_hpylm(D, d, c, hmm, train, test):
    train_int = [np.asarray(s, dtype=np.int64) for s in train]
    pool = HPYLMPool(train_int, alphabet_size=PROD_NA,
                     max_depth=D, discount=d, concentration=c, seed=0)
    out = {}
    for h in HORIZONS:
        ces, floors = [], []
        for prefix in test:
            prefix = np.asarray(prefix, dtype=np.int64)
            pred = pool.predict_distribution(prefix, h=h, alpha_prior=0.001)
            ce, fl = excess_pp_pred(pred, hmm, prefix, h)
            ces.append(ce); floors.append(fl)
        out[h] = 2.0 ** (np.mean(ces) - np.mean(floors))
    return out


def eval_ppm(D, d, hmm, train, test):
    train_int = [np.asarray(s, dtype=np.int64) for s in train]
    pool = PPMPool(train_int, alphabet_size=PROD_NA, max_depth=D, discount=d)
    out = {}
    for h in HORIZONS:
        ces, floors = [], []
        for prefix in test:
            prefix = np.asarray(prefix, dtype=np.int64)
            pred = pool.predict_distribution(prefix, h=h, alpha_prior=0.001)
            ce, fl = excess_pp_pred(pred, hmm, prefix, h)
            ces.append(ce); floors.append(fl)
        out[h] = 2.0 ** (np.mean(ces) - np.mean(floors))
    return out


def eval_freq(_unused, hmm, train, test):
    """Unigram baseline: predict the empirical training symbol frequency
    at every time step (no context). Laplace +1 smoothing to handle
    OOV product symbols."""
    train_chars = np.concatenate([np.asarray(s, dtype=np.int64)
                                    for s in train])
    counts = np.bincount(train_chars, minlength=PROD_NA).astype(np.float64)
    pred = (counts + 1.0) / (counts.sum() + PROD_NA)
    out = {}
    for h in HORIZONS:
        ces, floors = [], []
        for prefix in test:
            prefix = np.asarray(prefix, dtype=np.int64)
            ce, fl = excess_pp_pred(pred, hmm, prefix, h)
            ces.append(ce); floors.append(fl)
        out[h] = 2.0 ** (np.mean(ces) - np.mean(floors))
    return out


def eval_kn3(disc, hmm, train, test):
    train_int = [np.asarray(s, dtype=np.int64) for s in train]
    m = KN3Model(V=PROD_NA, discount=disc); m.fit(train_int)
    out = {}
    for h in HORIZONS:
        ces, floors = [], []
        for prefix in test:
            prefix = np.asarray(prefix, dtype=np.int64)
            pred = m.predict_distribution(prefix)  # KN3 only does h=1; reuse for all h
            ce, fl = excess_pp_pred(pred, hmm, prefix, h)
            ces.append(ce); floors.append(fl)
        out[h] = 2.0 ** (np.mean(ces) - np.mean(floors))
    return out


def main():
    method_configs = [
        ('Freq (unigram)',        eval_freq, None),
        ('GDC α=0.7 β=0.05 asym', eval_gdc,
         dict(alpha=0.7, theta=0.2, beta=0.05, beta_scaling='asymptotic')),
        ('CHMM K=32',             eval_chmm, 32),
        ('ALERGIA eps=0.05',      eval_alergia, 0.05),
        ('Parrot L=4 K=100',      eval_parrot, (4, 100)),
        ('HPYLM D=3 d=0.5',       eval_hpylm, (3, 0.5, 1.0)),
        ('PPM-D D=3 d=0.5',       eval_ppm, (3, 0.5)),
        ('KN-3 d=0.5',            eval_kn3, 0.5),
    ]

    results = {}
    for name, fn, cfg in method_configs:
        print(f"  Running {name}...", flush=True)
        per_seed = {h: [] for h in HORIZONS}
        for seed in SEEDS:
            rng = np.random.default_rng(7000 + seed * 41)
            components = make_state_preferred_components(
                N_COMPONENTS, NS_PER_COMP, NA_PER_COMP, FANOUT, rng,
                min_pref_prob=MIN_PREF_PROB,
                min_self_prob=MIN_SELF_PROB)
            hmm_prod = build_product_hmm(components)
            train = []
            for _ in range(N_TRAIN_SEQ):
                _, obs = sample_product(components, SEQ_LEN, rng)
                train.append(obs)
            test = []
            for _ in range(N_TEST_SEQ):
                _, obs = sample_product(components, SEQ_LEN, rng)
                test.append(obs)
            t0 = time.time()
            try:
                if isinstance(cfg, dict):
                    res = fn(cfg, hmm_prod, train, test)
                elif isinstance(cfg, tuple):
                    res = fn(*cfg, hmm_prod, train, test)
                else:
                    res = fn(cfg, hmm_prod, train, test)
                print(f"    seed={seed} done in {time.time()-t0:.1f}s",
                      flush=True)
                for h in HORIZONS:
                    per_seed[h].append(res[h])
            except Exception as e:
                print(f"    seed={seed} FAILED: {e}", flush=True)
        results[name] = {h: float(np.mean(per_seed[h])) if per_seed[h] else float('nan')
                          for h in HORIZONS}

    print(f"\n{N_COMPONENTS}-component product HMM (each: nS={NS_PER_COMP}, "
          f"nA={NA_PER_COMP}, fanout={FANOUT}, "
          f"min_pref_prob={MIN_PREF_PROB}, min_self_prob={MIN_SELF_PROB})")
    print(f"Product alphabet={PROD_NA}; train: {N_TRAIN_SEQ} seq × len "
          f"{SEQ_LEN}; test: {N_TEST_SEQ} seq × len {SEQ_LEN}\n")
    print(f"{'method':<24} " + " ".join(f"{f'h={h}':>9}" for h in HORIZONS))
    print('-' * 76)
    for name, res in results.items():
        print(f"{name:<24} " + " ".join(f"{res[h]:>9.4f}" for h in HORIZONS))


if __name__ == '__main__':
    main()
