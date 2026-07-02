"""Dense α × θ × β sweep around the α=0.9 sweet spot, sparse_large only.

Prior result: α=0.9, θ=0.005, β=0.005 (asymptotic) → excess PP 1.4229.
Here we densely sweep α∈[0.85, 0.95] and θ∈[0.005, 0.15] subject to
the α+θ≤1 constraint, with β around the 0.001-0.01 band.
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

ALPHAS = [0.85, 0.88, 0.90, 0.92, 0.95]
THETAS = [0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15]
BETAS  = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025]
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
    valid_configs = [(a, t, b) for a in ALPHAS for t in THETAS for b in BETAS
                      if a + t <= 1.0]
    print(f"sparse_large: {len(valid_configs)} (α,θ,β) configs × "
          f"{len(SEEDS)} seeds = {len(valid_configs)*len(SEEDS)} fits",
          flush=True)

    for seed in SEEDS:
        seed_offset = 2
        rng = np.random.default_rng(11000 + seed * 97 + nS * 13 + nA + seed_offset)
        hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                            E_concentration=conc)
        train_seqs = [hmm.sample(TRAIN_LEN, rng)[1] for _ in range(N_TRAIN_SEQ)]
        test_pref = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
                      for _ in range(N_TEST_PREFIXES)]
        for alpha, theta, beta in valid_configs:
            scaling = 'asymptotic' if beta > 0 else 'none'
            cfg = dict(alpha=alpha, theta=theta, beta=beta,
                        beta_scaling=scaling)
            res = evaluate(cfg, hmm, train_seqs, test_pref, nA)
            for h in HORIZONS:
                rows.append(dict(alpha=alpha, theta=theta, beta=beta,
                                  beta_scaling=scaling, seed=seed,
                                  horizon=h, excess_pp=res[h]))
        print(f"  seed={seed} done [total={time.time()-t0:.0f}s]", flush=True)

    out_csv = os.path.join(HERE,
                            'gdc_lscale_sparse_large_atb_sweep.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out_csv}", flush=True)


if __name__ == '__main__':
    main()
