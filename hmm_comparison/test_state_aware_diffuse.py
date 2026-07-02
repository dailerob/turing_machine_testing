"""Test the emission-aware-diffuse fix on sparse_large.

Compares:
  - Vanilla GDC at α=0.8, β=0.1 (current val pick under absorb+uniform)
  - Vanilla GDC swept across α ∈ {0.5, 0.8, 0.9, 0.95, 0.99}
  - EmissionAware GDC swept across the same α grid
  - CHMM K=32 reference
  - Oracle clustering (upper bound any fix could deliver)

If the fix works, EmissionAware at α=0.99 should approach the oracle
because the previously-fatal "few specific positions advance" failure
mode is replaced by "average over all positions sharing the dominant
emission group."

All runs use absorb + uniform.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from random_hmm import random_sparse_topology_hmm
from model_wrappers import GDCForecaster
from chmm_alergia_wrappers import CHMMForecaster
from evaluation import perplexity_at_horizons
from state_aware_gdc import make_emission_aware_gdc_forecaster

nS, nA = 30, 8
fanout = 2
E_concentration = 0.1
TRAIN_LEN = 200
N_TRAIN = 400
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20

ALPHAS = [0.50, 0.80, 0.90, 0.95, 0.99]
BETAS  = [0.05, 0.10, 0.20]
THETA  = 0.001


def setup(seed):
    seed_offset = 2  # sparse
    rng = np.random.default_rng(60000 + seed * 137 + nS * 7 + nA * 11
                                + seed_offset)
    hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                     E_concentration=E_concentration)
    full = []
    for _ in range(N_TRAIN):
        full.append(hmm.sample(TRAIN_LEN, rng))
    train = [o for _, o in full]
    true_states = np.concatenate([s for s, _ in full])
    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]
    return hmm, train, true_states, test_pf


class OracleClusteredGDC:
    """Aggregates GDC's position posterior to the true 30 hidden states
    and applies the true HMM dynamics for h-step prediction."""
    def __init__(self, gdc, true_states, hmm):
        self.gdc = gdc; self.true_states = true_states
        self.hmm = hmm; self.nA = hmm.nA
    def horizon_emission(self, prefix_obs, h):
        obs = np.asarray(prefix_obs, dtype=np.int64).reshape(-1, 1)
        pp = self.gdc.gdc.forward_pass(obs)
        sp = np.zeros(self.hmm.nS); np.add.at(sp, self.true_states, pp)
        sp = sp / sp.sum() if sp.sum() > 0 else \
             np.full(self.hmm.nS, 1.0/self.hmm.nS)
        Th = np.linalg.matrix_power(self.hmm.T, h)
        return sp @ Th @ self.hmm.E


def run_seed(seed):
    print(f"--- seed {seed} ---", flush=True)
    hmm, train, true_states, test_pf = setup(seed)
    rows = []

    # Vanilla GDC sweep (absorb + uniform)
    for alpha in ALPHAS:
        for beta in BETAS:
            t0 = time.time()
            gdc = GDCForecaster(nA, alpha=alpha, theta=THETA, gamma=0.0,
                                beta=beta, transition_type='self_loop',
                                initial_dist='uniform',
                                terminal_behavior='absorb').fit(train)
            ppl = perplexity_at_horizons(gdc, hmm, test_pf, [1])
            rows.append(dict(seed=seed, model='gdc_vanilla',
                             alpha=alpha, beta=beta,
                             excess_pp=ppl[1]['excess_perplexity'],
                             elapsed_s=time.time()-t0))

    # EmissionAware GDC sweep
    for alpha in ALPHAS:
        for beta in BETAS:
            t0 = time.time()
            gdc = make_emission_aware_gdc_forecaster(
                nA, alpha=alpha, theta=THETA, beta=beta,
                transition_type='self_loop', initial_dist='uniform',
                terminal_behavior='absorb').fit(train)
            ppl = perplexity_at_horizons(gdc, hmm, test_pf, [1])
            rows.append(dict(seed=seed, model='gdc_emission_aware',
                             alpha=alpha, beta=beta,
                             excess_pp=ppl[1]['excess_perplexity'],
                             elapsed_s=time.time()-t0))

    # Oracle (vanilla GDC at val pick, true-state aggregation)
    gdc_v = GDCForecaster(nA, alpha=0.8, theta=THETA, gamma=0.0, beta=0.1,
                          transition_type='self_loop',
                          initial_dist='uniform',
                          terminal_behavior='absorb').fit(train)
    oracle = OracleClusteredGDC(gdc_v, true_states, hmm)
    ppl = perplexity_at_horizons(oracle, hmm, test_pf, [1])
    rows.append(dict(seed=seed, model='gdc_oracle_cluster',
                     alpha=0.8, beta=0.1,
                     excess_pp=ppl[1]['excess_perplexity'],
                     elapsed_s=0.0))

    # CHMM reference
    t0 = time.time()
    chmm = CHMMForecaster(nA, K=32, n_em_iters=50, seed=seed).fit(train)
    ppl = perplexity_at_horizons(chmm, hmm, test_pf, [1])
    rows.append(dict(seed=seed, model='chmm_K32', alpha=None, beta=None,
                     excess_pp=ppl[1]['excess_perplexity'],
                     elapsed_s=time.time()-t0))
    return rows


def main():
    all_rows = []
    for seed in [0, 1, 2]:
        rs = run_seed(seed)
        all_rows.extend(rs)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(HERE, 'state_aware_diffuse_test.csv'),
              index=False)

    print("\n## Mean excess_pp across 3 seeds (CHMM K=32 = "
          f"{df[df.model=='chmm_K32'].excess_pp.mean():.4f}, "
          f"oracle = {df[df.model=='gdc_oracle_cluster'].excess_pp.mean():.4f})\n")

    for model in ['gdc_vanilla', 'gdc_emission_aware']:
        sub = df[df.model == model]
        pv = sub.groupby(['alpha','beta'])['excess_pp'].mean().unstack()
        print(f"### {model}")
        print(pv.round(4))
        print(f"  best: {pv.min().min():.4f} at "
              f"{pv.stack().idxmin()}\n")


if __name__ == "__main__":
    main()
