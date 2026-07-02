"""Wider α × β grid for asymptotic L-scaling on sparse_large only.

The wide-grid run found α=0.95 β=0.0025-0.005 best for sparse_large
(1.518 vs fixed-a95-b0=1.529). Here we densely scan α to see if
intermediate values do even better.

Single regime, single horizon (h=1), fast.
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from random_hmm import random_sparse_topology_hmm  # noqa: E402
from gdc_torch_discrete import horizon_emission_many  # noqa: E402

REGIME = ('sparse_large', 30, 8, 'sparse', 0.1, 2)
SEEDS = [0, 1, 2]
N_TRAIN_SEQ = 200
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 2, 5, 10, 20]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64

ALPHAS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
BETAS  = [0.0, 0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]


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


EPS = 1e-12

def evaluate(cfg, hmm, train_seqs, test_prefixes, nA):
    sym, term, start = build_chain_metadata(train_seqs)
    primes = np.stack([np.asarray(p, dtype=np.int64) for p in test_prefixes])
    pred = horizon_emission_many(
        symbol_of_state=sym, terminal_mask=term, start_mask=start,
        primes=primes, horizons=HORIZONS, nA=nA,
        alpha=cfg['alpha'], theta=cfg['theta'], beta=cfg['beta'],
        beta_scaling=cfg['beta_scaling'],
        transition_type='self_loop', terminal_behavior='diffuse',
        initial_dist='sequence_starts', device=DEVICE, dtype=DTYPE)
    pred_np = pred.cpu().numpy()
    B = primes.shape[0]
    true_d = np.zeros((B, len(HORIZONS), nA))
    for b in range(B):
        af = hmm.filter(test_prefixes[b])
        for hi, h in enumerate(HORIZONS):
            true_d[b, hi] = hmm.horizon_emission(af, h)
    out = {}
    for hi, h in enumerate(HORIZONS):
        ce = -float(np.mean(np.sum(true_d[:, hi]
                                     * np.log2(np.maximum(pred_np[:, hi], EPS)),
                                     axis=1)))
        floor = -float(np.mean(np.sum(true_d[:, hi]
                                        * np.log2(np.maximum(true_d[:, hi], EPS)),
                                        axis=1)))
        out[h] = 2.0 ** (ce - floor)
    return out


def main():
    name, nS, nA, kind, conc, fanout = REGIME
    rows = []
    t0 = time.time()
    n_total = len(ALPHAS) * len(BETAS) * len(SEEDS)
    n_done = 0
    print(f"sparse_large: {len(ALPHAS)} α × {len(BETAS)} β × {len(SEEDS)} "
          f"seeds = {n_total} fits", flush=True)

    for seed in SEEDS:
        seed_offset = 2  # sparse seed offset
        rng = np.random.default_rng(11000 + seed * 97 + nS * 13 + nA + seed_offset)
        hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                            E_concentration=conc)
        train_seqs = [hmm.sample(TRAIN_LEN, rng)[1] for _ in range(N_TRAIN_SEQ)]
        test_pref = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
                      for _ in range(N_TEST_PREFIXES)]
        for alpha in ALPHAS:
            theta = 0.005 if alpha >= 0.9 else 0.2
            for beta in BETAS:
                if beta == 0.0:
                    cfg = dict(alpha=alpha, theta=theta, beta=0.0,
                                beta_scaling='none')
                    name_short = f"none-a{int(alpha*100)}-b0"
                else:
                    cfg = dict(alpha=alpha, theta=theta, beta=beta,
                                beta_scaling='asymptotic')
                    name_short = f"asym-a{int(alpha*100)}-b{beta:.4f}"
                res = evaluate(cfg, hmm, train_seqs, test_pref, nA)
                for h in HORIZONS:
                    rows.append(dict(alpha=alpha, theta=theta, beta=beta,
                                      beta_scaling=cfg['beta_scaling'],
                                      seed=seed, horizon=h,
                                      excess_pp=res[h]))
                n_done += 1
        print(f"  seed={seed} done [{n_done}/{n_total}, "
              f"total={time.time()-t0:.0f}s]", flush=True)

    out_csv = os.path.join(HERE, 'gdc_lscale_sparse_large_alpha_sweep.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out_csv}", flush=True)


if __name__ == '__main__':
    main()
