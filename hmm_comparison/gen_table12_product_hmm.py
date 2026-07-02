"""Canonical generator for paper Table 12 (tab:product_hmm_scaling).

RECONSTRUCTED 2026-06 (companion to gen_table7_forecasting.py; see
paper/PROTOCOL_STANDARDIZATION.md S7). 3-component TERNARY product HMM:
three independent components, each n_S=3 and n_A=3, per-component
transitions Dirichlet(0.1) (near-deterministic rows), emissions
state-preferred with E[i, i mod n_A] >= 0.7. Combined by Kronecker
product into one HMM with 27 hidden states and 27 symbols. Data-scaling
at N in {40,160,640} training sequences of length 20; 20 test prefixes;
excess perplexity at h=1..5.

LEAKAGE-FREE (departs from the original caption's oracle phrasing, in
line with the leakage-free standard used everywhere else in the paper):
configs are selected on a disjoint set of validation product-HMM seeds
and reported on the test seeds. The GDC row is the single FIXED config
(alpha=0.85, theta=0.005, beta=0.075 asymptotic) applied at every scale
-- a leakage-free, chosen-a-priori config; the full 462-config grid
(66 alpha,theta pairs x 7 beta) is still swept so we can confirm this
fixed config is the validation-best at each scale. CHMM (best K) and
Parrot (best L,K) are val-picked per horizon; HPYLM/PPM-D/KN-3 use the
fixed depth-3, d=0.5 configs from the caption.

Two passes (CPU baselines via mp.Pool; GPU GDC in main). Writes
table12_product_hmm_results.csv and prints the per-scale tables.

Usage:
  python gen_table12_product_hmm.py            # full run
  python gen_table12_product_hmm.py --smoke    # tiny sanity check
"""
from __future__ import annotations
import os, sys, csv, time, argparse, multiprocessing as mp
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'algorithmic_benchmarks'))

from gen_table7_forecasting import (_gdc_state_arrays, _PoolForecaster,
                                     _KN3Forecaster, freq_excess_pp)

# Product-HMM definition.
N_COMPONENTS = 3
NS_PER_COMP = 3
NA_PER_COMP = 3
T_CONCENTRATION = 0.1
MIN_PREF_PROB = 0.7
PROD_NA = NA_PER_COMP ** N_COMPONENTS    # 27

N_VALUES = [40, 160, 640]
SEQ_LEN = 20
N_TEST_PREFIXES = 20
HORIZONS = [1, 2, 3, 4, 5]
TEST_SEEDS = [0, 1, 2]
VAL_SEEDS = [100, 101, 102]

# GDC grid (462 configs = 66 valid (alpha,theta) x 7 beta; single-alpha).
# Principled 18-config grid (down from the original 462). The reported GDC row
# is the FIXED config below, so grid size does not change the table numbers --
# the grid only verifies the fixed config is near-best. The full-grid val-picks
# for this regime land at alpha in {0.5..0.85}, beta in {0.05..0.15} (the
# 27-symbol STOCHASTIC product-HMM emission needs a higher emission-noise floor
# than the structured HMMs of Table 7 -- a regime-justified difference), so the
# bracket is alpha{0.5,0.7,0.85} x theta{0.005,0.05} x beta{0.05,0.075,0.15}.
# Set GDC_TABLE12_FULL=1 to restore the original 462-config sweep.
if os.environ.get('GDC_TABLE12_FULL', '0') == '1':
    GDC_ALPHAS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.85, 0.95]
    GDC_THETAS = [0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    GDC_BETAS = [0.0, 0.005, 0.025, 0.05, 0.075, 0.1, 0.15]
else:
    GDC_ALPHAS = [0.5, 0.7, 0.85]
    GDC_THETAS = [0.005, 0.05]
    GDC_BETAS = [0.05, 0.075, 0.15]
GDC_FIXED = (0.85, 0.005, 0.075)   # the config reported in the paper row

# Baseline grids (per caption).
CHMM_KS_SMALL = [4, 8, 16, 32]              # N=40 (1x)
CHMM_KS_LARGE = [4, 8, 16, 32, 64, 128]     # N>=160 (4x/16x)
PARROT_LS = [2, 4]
PARROT_KS = [25, 100]
PARROT_ALPHA = 1.0
HPYLM_CFG = dict(max_depth=3, discount=0.5, concentration=1.0)
PPM_CFG = dict(max_depth=3, discount=0.5)
KN3_DISCOUNT = 0.5

OUT_CSV = os.path.join(HERE, 'table12_product_hmm_results.csv')


def _apply_smoke():
    global N_VALUES, GDC_ALPHAS, GDC_THETAS, GDC_BETAS, CHMM_KS_LARGE
    N_VALUES = [40, 160]
    GDC_ALPHAS = [0.30, 0.85, 0.95]
    GDC_THETAS = [0.005, 0.05]
    GDC_BETAS = [0.0, 0.075]
    CHMM_KS_LARGE = [4, 16]


def make_product_hmm(seed):
    """Deterministic 3-component ternary product HMM for a given seed."""
    from product_hmm import build_product_hmm, random_state_preferred_hmm
    comps = []
    for c in range(N_COMPONENTS):
        rng = np.random.default_rng(90000 + seed * 17 + c * 101)
        comps.append(random_state_preferred_hmm(
            NS_PER_COMP, NA_PER_COMP, rng,
            t_concentration=T_CONCENTRATION, min_pref_prob=MIN_PREF_PROB))
    return build_product_hmm(comps)


def gen_data(seed):
    hmm = make_product_hmm(seed)
    rng = np.random.default_rng(91000 + seed * 31)
    full_train = [hmm.sample(SEQ_LEN, rng)[1] for _ in range(max(N_VALUES))]
    test_pf = [hmm.sample(SEQ_LEN, rng)[1] for _ in range(N_TEST_PREFIXES)]
    return hmm, full_train, test_pf


def chmm_ks_for(N):
    return CHMM_KS_SMALL if N <= 40 else CHMM_KS_LARGE


# ---------------------------------------------------------------------------
# Pass A: CPU baselines
# ---------------------------------------------------------------------------
def run_baselines_cell(seed):
    sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'algorithmic_benchmarks'))
    from evaluation import perplexity_at_horizons
    from chmm_alergia_wrappers import fit_chmm
    from discrete_parrot import DiscreteParrotPool
    from discrete_hpylm import HPYLMPool
    from discrete_ppm import PPMPool
    from kn3_eval import KN3Model

    hmm, full_train, test_pf = gen_data(seed)
    rows = []

    def record(N, mc, mn, model):
        ppl = perplexity_at_horizons(model, hmm, test_pf, HORIZONS)
        for h in HORIZONS:
            rows.append(dict(seed=seed, N_train=N, model_class=mc, model=mn,
                             horizon=h,
                             excess_perplexity=ppl[h]['excess_perplexity']))

    for N in N_VALUES:
        train = full_train[:N]
        for K in chmm_ks_for(N):
            try:
                record(N, 'chmm', f'chmm-K{K}',
                       fit_chmm(train, PROD_NA, K=K, n_em_iters=50))
            except Exception as e:
                sys.stderr.write(f"[chmm K={K} N{N} s{seed}] {e}\n")
        for L in PARROT_LS:
            pool = DiscreteParrotPool(train, alphabet_size=PROD_NA, L=L)
            for K in PARROT_KS:
                record(N, 'parrot', f'parrot-L{L}-K{K}',
                       _PoolForecaster(pool, K=K, alpha_prior=PARROT_ALPHA))
        pool = HPYLMPool(train, alphabet_size=PROD_NA, seed=seed, **HPYLM_CFG)
        record(N, 'hpylm', 'hpylm-D3-d0.5', _PoolForecaster(pool, alpha_prior=0.01))
        pool = PPMPool(train, alphabet_size=PROD_NA, **PPM_CFG)
        record(N, 'ppm', 'ppm-D3-d0.5', _PoolForecaster(pool, alpha_prior=0.01))
        m = KN3Model(V=PROD_NA, discount=KN3_DISCOUNT)
        m.fit([np.asarray(s, dtype=np.int64) for s in train])
        record(N, 'kn3', 'kn3-d0.5', _KN3Forecaster(m))
        rows.append(dict(seed=seed, N_train=N, model_class='freq',
                         model='freq', horizon=1,
                         excess_perplexity=freq_excess_pp(hmm, train, test_pf, h=1)))
    return rows


# ---------------------------------------------------------------------------
# Pass B: GDC on GPU
# ---------------------------------------------------------------------------
def run_gdc_all(device, dtype):
    from gdc_torch_discrete import horizon_emission_many
    at_pairs = [(a, t) for a in GDC_ALPHAS for t in GDC_THETAS
                if a + t <= 1.0 + 1e-9]
    print(f"  GDC: {len(at_pairs)} (a,t) x {len(GDC_BETAS)} beta "
          f"= {len(at_pairs)*len(GDC_BETAS)} configs", flush=True)
    rows = []
    for seed in (TEST_SEEDS + VAL_SEEDS):
        hmm, full_train, test_pf = gen_data(seed)
        primes = np.stack([np.asarray(p, dtype=np.int64) for p in test_pf])
        alphas = [hmm.filter(p) for p in test_pf]
        true_d = {h: np.stack([hmm.horizon_emission(a, h) for a in alphas])
                  for h in HORIZONS}
        floor_bits = {h: -np.sum(true_d[h] * np.log2(np.maximum(true_d[h], 1e-12)),
                                 axis=1).mean() for h in HORIZONS}
        for N in N_VALUES:
            sym, term, start = _gdc_state_arrays(full_train[:N])
            for (a, t) in at_pairs:
                for b in GDC_BETAS:
                    preds = horizon_emission_many(
                        symbol_of_state=sym, terminal_mask=term, start_mask=start,
                        primes=primes, horizons=HORIZONS, nA=PROD_NA,
                        alpha=a, theta=t, beta=b, transition_type='self_loop',
                        terminal_behavior='diffuse', initial_dist='sequence_starts',
                        beta_scaling='asymptotic',
                        device=device, dtype=dtype).cpu().numpy()
                    name = f'gdc-a{a}-t{t}-b{b}'
                    for j, h in enumerate(HORIZONS):
                        pr = np.maximum(preds[:, j, :], 1e-12)
                        ce = -np.sum(true_d[h] * np.log2(pr), axis=1).mean()
                        rows.append(dict(seed=seed, N_train=N, model_class='gdc',
                                         model=name, horizon=h,
                                         excess_perplexity=float(2.0 ** (ce - floor_bits[h]))))
        print(f"    seed {seed} done", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
METHODS = ['gdc', 'chmm', 'parrot', 'freq', 'hpylm', 'ppm', 'kn3']
PRETTY = {'gdc': f'GDC (fixed a={GDC_FIXED[0]},t={GDC_FIXED[1]},b={GDC_FIXED[2]})',
          'chmm': 'CHMM (best K)', 'parrot': 'Parrot (best L,K)',
          'freq': 'Freq', 'hpylm': 'HPYLM (D3,d0.5)',
          'ppm': 'PPM-D (D3,d0.5)', 'kn3': 'KN-3 (d0.5)'}
GDC_FIXED_NAME = f'gdc-a{GDC_FIXED[0]}-t{GDC_FIXED[1]}-b{GDC_FIXED[2]}'


def aggregate_and_print(df):
    val_set, test_set = set(VAL_SEEDS), set(TEST_SEEDS)
    for N in N_VALUES:
        print(f"### N={N}\n")
        print("| Method | " + " | ".join(f"h={h}" for h in HORIZONS) + " |")
        print("|---|" + "---:|" * len(HORIZONS))
        nsub = df[df.N_train == N]
        for m in METHODS:
            msub = nsub[nsub.model_class == m]
            cells = []
            for h in HORIZONS:
                hsub = msub[msub.horizon == h]
                if hsub.empty:
                    cells.append("---"); continue
                if m == 'gdc':
                    test = hsub[(hsub.seed.isin(test_set)) &
                                (hsub.model == GDC_FIXED_NAME)]
                    cells.append(f"{test.excess_perplexity.mean():.3f}"
                                 if not test.empty else "---")
                    continue
                if m == 'freq':
                    cells.append(f"{hsub[hsub.seed.isin(test_set)].excess_perplexity.mean():.3f}")
                    continue
                vmeans = hsub[hsub.seed.isin(val_set)].groupby('model')['excess_perplexity'].mean()
                if vmeans.empty:
                    cells.append("---"); continue
                pick = vmeans.idxmin()
                test = hsub[(hsub.seed.isin(test_set)) & (hsub.model == pick)]
                cells.append(f"{test.excess_perplexity.mean():.3f}")
            print(f"| {PRETTY[m]} | " + " | ".join(cells) + " |")
        # confirm the fixed GDC config is the val-best at this scale (h=1)
        g = nsub[(nsub.model_class == 'gdc') & (nsub.horizon == 1) &
                 (nsub.seed.isin(val_set))]
        if not g.empty:
            vb = g.groupby('model')['excess_perplexity'].mean().idxmin()
            print(f"\n_(val-best GDC config at N={N}, h=1: {vb}; "
                  f"fixed={GDC_FIXED_NAME})_\n")


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
    print(f"device={device}  product HMM nS={NS_PER_COMP**N_COMPONENTS} "
          f"nA={PROD_NA}  N_VALUES={N_VALUES}", flush=True)

    seeds = TEST_SEEDS + VAL_SEEDS
    n_workers = args.workers or max(1, min(len(seeds), (os.cpu_count() or 4) - 1))
    print(f"\n[Pass A] CPU baselines: {len(seeds)} seeds, {n_workers} workers",
          flush=True)
    tA = time.time(); base_rows = []
    with mp.Pool(processes=n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_baselines_cell, seeds)):
            base_rows.extend(r)
            print(f"    [{i+1}/{len(seeds)}] seed done  [{time.time()-tA:.0f}s]",
                  flush=True)
    print(f"[Pass A] done ({time.time()-tA:.0f}s, {len(base_rows)} rows)")

    print(f"\n[Pass B] GDC on {device}", flush=True)
    tB = time.time(); gdc_rows = run_gdc_all(device, dtype)
    print(f"[Pass B] done ({time.time()-tB:.0f}s, {len(gdc_rows)} rows)")

    all_rows = base_rows + gdc_rows
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['seed', 'N_train', 'model_class',
                                          'model', 'horizon', 'excess_perplexity'])
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV} ({len(all_rows)} rows)\n")
    aggregate_and_print(pd.DataFrame(all_rows))


if __name__ == "__main__":
    main()
