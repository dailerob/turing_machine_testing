"""GDC discrete L-scaling experiment on the 4 canonical HMM-forecasting
regimes. **Uses the GPU-batched torch kernel** (`gdc_torch_discrete`)
for ~50× speedup vs the numpy GDC.

Hypothesis: scaling β with prefix length L (the discrete analog of the
continuous σ·√L kernel-bandwidth trick) might improve GDC's predictive
distribution on stochastic finite-state regimes.

Three β schedules per (α, θ, β₀) config:
  - 'none'   : β_eff = β₀ (current behavior; baseline)
  - 'linear' : β_eff = min(β₀ · L, 1.0)        ← variance ∝ L
  - 'sqrt'   : β_eff = min(β₀ · √L, 1.0)       ← std-dev ∝ √L

Output: gdc_lscale_results.csv (rows = (config, regime, seed, horizon)).
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from random_hmm import (random_dense_hmm,                        # noqa: E402
                          random_sparse_topology_hmm)
from gdc_torch_discrete import horizon_emission_many             # noqa: E402

REGIMES = [
    ('dense_small',  10, 4, 'dense',  1.0, None),
    ('dense_large',  30, 8, 'dense',  1.0, None),
    ('sparse_small', 10, 4, 'sparse', 0.1, 2),
    ('sparse_large', 30, 8, 'sparse', 0.1, 2),
]
SEEDS = [0, 1, 2]
N_TRAIN_SEQ = 200
TRAIN_LEN = 50
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
HORIZONS = [1, 2, 5, 10, 20]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64


# β_eff at L=20 for the asymptotic schedule (1 - (1-β)^L) for reference:
#   β=0.0001 → 0.002       β=0.0025 → 0.0488      β=0.025 → 0.397
#   β=0.0005 → 0.00995     β=0.005  → 0.0954      β=0.05  → 0.642
#   β=0.001  → 0.0198      β=0.01   → 0.182       β=0.1   → 0.878
ALPHAS = [0.5, 0.7, 0.95]
BETAS_SMALL = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]

CONFIGS = []

# Fixed-β baselines from prior sweeps
CONFIGS += [
    dict(name='fixed-a95-b0',  alpha=0.95, theta=0.005, beta=0.0,
         beta_scaling='none'),
    dict(name='fixed-a7-b1',   alpha=0.7,  theta=0.2,   beta=0.1,
         beta_scaling='none'),
    dict(name='fixed-a7-b2',   alpha=0.7,  theta=0.2,   beta=0.2,
         beta_scaling='none'),
    dict(name='fixed-a5-b1',   alpha=0.5,  theta=0.2,   beta=0.1,
         beta_scaling='none'),
]

# Asymptotic, full β grid × α grid
def short_b(b):
    s = f"{b:.4f}".rstrip('0').rstrip('.')
    return s.replace('0.', 'b')

for a in ALPHAS:
    theta = 0.005 if a >= 0.9 else 0.2
    for b in BETAS_SMALL:
        CONFIGS.append(
            dict(name=f"asym-a{int(a*100)}-{short_b(b)}",
                 alpha=a, theta=theta, beta=b,
                 beta_scaling='asymptotic'))

# Linear / sqrt anchors (smaller subset since we already know they hit
# saturation at the higher β values)
CONFIGS += [
    dict(name='lin-a7-b005',  alpha=0.7,  theta=0.2,   beta=0.005,
         beta_scaling='linear'),
    dict(name='lin-a7-b025',  alpha=0.7,  theta=0.2,   beta=0.025,
         beta_scaling='linear'),
    dict(name='lin-a5-b005',  alpha=0.5,  theta=0.2,   beta=0.005,
         beta_scaling='linear'),
    dict(name='lin-a5-b025',  alpha=0.5,  theta=0.2,   beta=0.025,
         beta_scaling='linear'),
    dict(name='lin-a5-b05',   alpha=0.5,  theta=0.2,   beta=0.05,
         beta_scaling='linear'),
    dict(name='sqrt-a5-b025', alpha=0.5,  theta=0.2,   beta=0.025,
         beta_scaling='sqrt'),
    dict(name='sqrt-a5-b05',  alpha=0.5,  theta=0.2,   beta=0.05,
         beta_scaling='sqrt'),
    dict(name='sqrt-a5-b1',   alpha=0.5,  theta=0.2,   beta=0.1,
         beta_scaling='sqrt'),
]

OUT_CSV = os.path.join(HERE, 'gdc_lscale_results.csv')


def make_hmm(kind, nS, nA, conc, fanout, rng):
    if kind == 'dense':
        return random_dense_hmm(nS, nA, rng, E_concentration=conc)
    elif kind == 'sparse':
        return random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                            E_concentration=conc)
    raise ValueError(kind)


def build_chain_metadata(train_seqs):
    """Stack training sequences into a single chain and build the
    state-symbol/terminal/start arrays the torch kernel needs."""
    state_syms = []
    term_mask = []
    start_mask = []
    for seq in train_seqs:
        L = len(seq)
        if L == 0:
            continue
        state_syms.append(np.asarray(seq, dtype=np.int64))
        tm = np.zeros(L, dtype=bool); tm[-1] = True
        sm = np.zeros(L, dtype=bool); sm[0] = True
        term_mask.append(tm)
        start_mask.append(sm)
    return (np.concatenate(state_syms),
            np.concatenate(term_mask),
            np.concatenate(start_mask))


EPS_PROB = 1e-12


def evaluate_config_on_regime(cfg, hmm, train_seqs, test_prefixes, nA):
    """Build chain once; eval all horizons in one batched torch call.
    Returns dict {h: {mse, ce, floor, excess_pp}} aggregated across the
    100 test prefixes."""
    sym, term, start = build_chain_metadata(train_seqs)
    primes = np.stack([np.asarray(p, dtype=np.int64) for p in test_prefixes])
    pred = horizon_emission_many(
        symbol_of_state=sym,
        terminal_mask=term,
        start_mask=start,
        primes=primes,
        horizons=HORIZONS,
        nA=nA,
        alpha=cfg['alpha'],
        theta=cfg['theta'],
        beta=cfg['beta'],
        beta_scaling=cfg['beta_scaling'],
        transition_type='self_loop',
        terminal_behavior='diffuse',
        initial_dist='sequence_starts',
        device=DEVICE,
        dtype=DTYPE,
    )  # (B, n_horizons, nA) torch tensor
    pred_np = pred.cpu().numpy()
    # Compare to true HMM posterior predictives at each horizon
    out = {}
    B = primes.shape[0]
    # Pre-compute true distributions per (b, h)
    true_dists = np.zeros((B, len(HORIZONS), nA))
    for b in range(B):
        alpha_filt = hmm.filter(test_prefixes[b])
        for hi, h in enumerate(HORIZONS):
            true_dists[b, hi] = hmm.horizon_emission(alpha_filt, h)
    for hi, h in enumerate(HORIZONS):
        diff_sq = (pred_np[:, hi] - true_dists[:, hi]) ** 2  # (B, nA)
        mse = float(np.mean(diff_sq))
        # Cross-entropy: -Σ true · log2(pred), averaged over prefixes
        pred_safe = np.maximum(pred_np[:, hi], EPS_PROB)
        true_safe = np.maximum(true_dists[:, hi], EPS_PROB)
        ce = -float(np.mean(np.sum(true_dists[:, hi]
                                     * np.log2(pred_safe), axis=1)))
        floor = -float(np.mean(np.sum(true_dists[:, hi]
                                        * np.log2(true_safe), axis=1)))
        out[h] = dict(mse=mse, ce=ce, floor=floor,
                       excess_pp=2.0 ** (ce - floor),
                       pp=2.0 ** ce, pp_floor=2.0 ** floor)
    return out


def main():
    rows = []
    t0 = time.time()
    n_done = 0
    n_total = len(REGIMES) * len(SEEDS) * len(CONFIGS)
    print(f"Running {len(CONFIGS)} GDC configs × {len(REGIMES)} regimes × "
          f"{len(SEEDS)} seeds = {n_total} fits  (device={DEVICE})",
          flush=True)

    for (regime, nS, nA, kind, conc, fanout) in REGIMES:
        for seed in SEEDS:
            tic = time.time()
            seed_offset = (2 if 'sparse' in regime else 0)
            rng = np.random.default_rng(11000 + seed * 97
                                          + nS * 13 + nA + seed_offset)
            hmm = make_hmm(kind, nS, nA, conc, fanout, rng)
            train_seqs = [hmm.sample(TRAIN_LEN, rng)[1]
                          for _ in range(N_TRAIN_SEQ)]
            test_prefixes = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
                             for _ in range(N_TEST_PREFIXES)]
            for cfg in CONFIGS:
                res = evaluate_config_on_regime(cfg, hmm, train_seqs,
                                                  test_prefixes, nA)
                for h in HORIZONS:
                    rec = res[h]
                    rows.append(dict(regime=regime, nS=nS, nA=nA,
                                      kind=kind, seed=seed,
                                      model=cfg['name'],
                                      alpha=cfg['alpha'],
                                      theta=cfg['theta'],
                                      beta=cfg['beta'],
                                      beta_scaling=cfg['beta_scaling'],
                                      horizon=h,
                                      mse=rec['mse'],
                                      ce=rec['ce'],
                                      floor=rec['floor'],
                                      excess_pp=rec['excess_pp'],
                                      pp=rec['pp'],
                                      pp_floor=rec['pp_floor']))
                n_done += 1
            print(f"  {regime} seed={seed} done in {time.time()-tic:.1f}s "
                  f"[{n_done}/{n_total} fits, total={time.time()-t0:.0f}s]",
                  flush=True)

    fields = ['regime', 'nS', 'nA', 'kind', 'seed', 'model', 'alpha',
              'theta', 'beta', 'beta_scaling', 'horizon',
              'mse', 'ce', 'floor', 'excess_pp', 'pp', 'pp_floor']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {OUT_CSV}", flush=True)
    print(f"Total: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
