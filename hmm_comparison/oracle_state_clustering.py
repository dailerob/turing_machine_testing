"""Oracle test: how much would GDC improve if it could collapse its
80,000 chain positions to the 30 true hidden states before scoring?

Procedure:
  1. Train the val-picked GDC (alpha=0.8, theta=0.001, beta=0.1).
  2. For each test prefix, get GDC's position posterior at step t.
  3. Aggregate position posterior by the TRUE hidden state of each
     training position (CHEAT: this is oracle information).
  4. Predict next-symbol via the TRUE HMM's emission * T from this
     state posterior.
  5. Compare the resulting excess perplexity to vanilla GDC and CHMM.

This bounds what a "cluster GDC positions before predicting" fix
could possibly deliver. If the oracle is close to the entropy floor,
the right intervention is structural (better aggregation). If the
oracle is still far above the floor, the issue is the position
posterior itself — diffuse-mass goes to wrong positions, not just
under-aggregated.

Also reports a second oracle: collapse positions to the 30 hidden
states, then transition + emit using the TRUE HMM. This is what
GDC + perfect-state-clustering + perfect-dynamics would score.
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

nS, nA = 30, 8
fanout = 2
E_concentration = 0.1
TRAIN_LEN = 200
N_TRAIN = 400
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20


class OracleClusteredGDC:
    """Wraps a GDC: at predict time, aggregates position posterior by
    the TRUE hidden state of each training position, then transitions
    via the TRUE HMM and emits. Pure oracle — uses generative ground
    truth at predict time.
    """
    def __init__(self, gdc, true_states_per_position, hmm):
        self.gdc = gdc
        self.true_states_per_position = true_states_per_position
        self.hmm = hmm
        self.nA = hmm.nA

    def horizon_emission(self, prefix_obs, h: int) -> np.ndarray:
        obs = np.asarray(prefix_obs, dtype=np.int64).reshape(-1, 1)
        position_post = self.gdc.gdc.forward_pass(obs)
        # Aggregate to hidden-state posterior
        state_post = np.zeros(self.hmm.nS)
        np.add.at(state_post, self.true_states_per_position, position_post)
        s = state_post.sum()
        if s > 0:
            state_post = state_post / s
        else:
            state_post = np.full(self.hmm.nS, 1.0 / self.hmm.nS)
        # Apply h true-HMM transitions, then emit
        Th = np.linalg.matrix_power(self.hmm.T, h)
        return state_post @ Th @ self.hmm.E


def run_seed(seed: int):
    seed_offset = 2  # sparse
    rng = np.random.default_rng(60000 + seed * 137 + nS * 7 + nA * 11
                                + seed_offset)
    hmm = random_sparse_topology_hmm(nS, nA, rng, fanout=fanout,
                                     E_concentration=E_concentration)

    full_train_with_states = []
    for _ in range(N_TRAIN):
        s, o = hmm.sample(TRAIN_LEN, rng)
        full_train_with_states.append((s, o))
    train_obs = [o for _, o in full_train_with_states]
    true_states = np.concatenate([s for s, _ in full_train_with_states])

    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]

    rows = []

    # Vanilla GDC (val-picked config)
    gdc = GDCForecaster(nA, alpha=0.8, theta=0.001, gamma=0.0, beta=0.1,
                        transition_type='self_loop',
                        initial_dist='sequence_starts')
    gdc.fit(train_obs)
    ppl = perplexity_at_horizons(gdc, hmm, test_pf, [1])
    rows.append(dict(seed=seed, model='gdc_vanilla',
                     excess_pp=ppl[1]['excess_perplexity']))

    # Oracle GDC: aggregate positions to true hidden states, then
    # use the true HMM dynamics for the prediction step.
    oracle = OracleClusteredGDC(gdc, true_states, hmm)
    ppl = perplexity_at_horizons(oracle, hmm, test_pf, [1])
    rows.append(dict(seed=seed, model='gdc_oracle_cluster',
                     excess_pp=ppl[1]['excess_perplexity']))

    # CHMM K=32 reference
    chmm = CHMMForecaster(nA, K=32, n_em_iters=50, seed=seed)
    chmm.fit(train_obs)
    ppl = perplexity_at_horizons(chmm, hmm, test_pf, [1])
    rows.append(dict(seed=seed, model='chmm_K32',
                     excess_pp=ppl[1]['excess_perplexity']))

    # Pure-oracle baseline: use the TRUE HMM filter posterior for prediction.
    # This is the entropy floor — should give excess_pp ≈ 1.0.
    class TrueOracle:
        def __init__(self, hmm): self.hmm = hmm; self.nA = hmm.nA
        def horizon_emission(self, prefix_obs, h):
            from random_hmm import RandomHMM
            pi, T, E = self.hmm.pi, self.hmm.T, self.hmm.E
            alpha = pi * E[:, prefix_obs[0]]; alpha /= alpha.sum()
            for o in prefix_obs[1:]:
                alpha = alpha @ T; alpha = alpha * E[:, o]
                alpha /= alpha.sum()
            Th = np.linalg.matrix_power(T, h)
            return alpha @ Th @ E
    truth = TrueOracle(hmm)
    ppl = perplexity_at_horizons(truth, hmm, test_pf, [1])
    rows.append(dict(seed=seed, model='true_oracle',
                     excess_pp=ppl[1]['excess_perplexity']))

    # GDC's own posterior collapse (alpha=0.8 etc.) BUT keep the
    # collapsed-to-state posterior, then push through GDC's own
    # transition matrix instead of the HMM's. (Tests whether the
    # GDC's transition matrix dynamics are the culprit, vs the
    # position posterior.)

    return rows


def main():
    all_rows = []
    for seed in [0, 1, 2]:
        print(f"=== seed {seed} ===", flush=True)
        rs = run_seed(seed)
        all_rows.extend(rs)
        for r in rs:
            print(f"  {r['model']:<28s}  excess_pp = {r['excess_pp']:.4f}",
                  flush=True)

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(HERE, 'oracle_state_clustering.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}\n")

    summary = df.groupby('model')['excess_pp'].mean().round(4)
    print("Mean excess_pp across 3 test seeds:")
    print(summary)


if __name__ == "__main__":
    main()
