"""sparse_large failure analysis with terminal_behavior x initial_dist
on/off — the user flagged that the diagnostic/fix scripts inherited
the default (terminal_behavior='diffuse', initial_dist='sequence_starts')
when recent forecasting work uses ('absorb', 'uniform'). This script
reruns the full diagnostic for all four corners of that 2x2.

For each corner:
  1. Diagnostic at val-picked config (alpha=0.8, theta=0.001, beta=0.1)
  2. Alpha sweep at the corner's best beta
  3. Oracle state-clustering test
  4. CHMM K=32 reference (mode-independent)

All numbers averaged over test seeds {0, 1, 2}.
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

ALPHAS = [0.10, 0.30, 0.50, 0.80, 0.90, 0.95, 0.99]
BETAS  = [0.05, 0.10, 0.20]
THETA  = 0.001

# 2x2 mode corners
MODES = [
    ('diffuse', 'sequence_starts'),  # original (matches earlier numbers)
    ('absorb',  'sequence_starts'),
    ('diffuse', 'uniform'),
    ('absorb',  'uniform'),
]


def kl(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1.0/len(p))
    q = q / q.sum() if q.sum() > 0 else np.full_like(q, 1.0/len(q))
    return float(np.sum(p * (np.log2(p + eps) - np.log2(q + eps))))


def setup_seed(seed: int):
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
    true_states = np.concatenate(
        [s for s, _ in full_train_with_states])
    test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
               for _ in range(N_TEST_PREFIXES)]
    return hmm, train_obs, true_states, test_pf


def hmm_filter_post(hmm, prefix):
    pi, T, E = hmm.pi, hmm.T, hmm.E
    a = pi * E[:, prefix[0]]; a /= a.sum()
    for o in prefix[1:]:
        a = a @ T; a = a * E[:, o]; a /= a.sum()
    return a


def make_oracle(gdc, true_states, hmm):
    class O:
        def __init__(self):
            self.gdc = gdc; self.true_states = true_states
            self.hmm = hmm; self.nA = hmm.nA
        def horizon_emission(self, prefix_obs, h):
            obs = np.asarray(prefix_obs, dtype=np.int64).reshape(-1, 1)
            pp = self.gdc.gdc.forward_pass(obs)
            sp = np.zeros(self.hmm.nS)
            np.add.at(sp, self.true_states, pp)
            sp = sp / sp.sum() if sp.sum() > 0 else \
                 np.full(self.hmm.nS, 1.0/self.hmm.nS)
            Th = np.linalg.matrix_power(self.hmm.T, h)
            return sp @ Th @ self.hmm.E
    return O()


def diagnostic_for_mode(seed, hmm, train_obs, true_states, test_pf,
                        terminal, init):
    """One-config diagnostic at val-picked alpha=0.8 beta=0.1."""
    gdc = GDCForecaster(nA, alpha=0.8, theta=THETA, gamma=0.0, beta=0.1,
                        transition_type='self_loop',
                        initial_dist=init,
                        terminal_behavior=terminal)
    gdc.fit(train_obs)

    excess_pps = []
    ess_list, top1_list, top100_list = [], [], []
    state_entropy_list = []
    kl_state_list = []
    argmax_correct = 0

    for s, o in [(None, p) for p in test_pf]:
        true_post = hmm_filter_post(hmm, o)
        pp = gdc.gdc.forward_pass(o.reshape(-1, 1))
        sp = np.zeros(hmm.nS); np.add.at(sp, true_states, pp)
        if sp.sum() > 0: sp = sp / sp.sum()
        else: sp = np.full(hmm.nS, 1.0/hmm.nS)
        ess_list.append(1.0 / np.sum(pp**2))
        sorted_pp = np.sort(pp)[::-1]
        top1_list.append(sorted_pp[0])
        top100_list.append(sorted_pp[:100].sum())
        state_entropy_list.append(
            float(-np.sum(sp * np.log2(sp + 1e-12))))
        kl_state_list.append(kl(true_post, sp))
        if int(np.argmax(sp)) == int(np.argmax(true_post)):
            argmax_correct += 1

    ppl = perplexity_at_horizons(gdc, hmm, test_pf, [1])
    excess_pp = ppl[1]['excess_perplexity']

    # Oracle aggregate
    oracle = make_oracle(gdc, true_states, hmm)
    oppl = perplexity_at_horizons(oracle, hmm, test_pf, [1])
    oracle_excess_pp = oppl[1]['excess_perplexity']

    return dict(
        seed=seed, terminal=terminal, init=init,
        excess_pp=excess_pp,
        oracle_excess_pp=oracle_excess_pp,
        median_ess=float(np.median(ess_list)),
        median_top1=float(np.median(top1_list)),
        median_top100=float(np.median(top100_list)),
        median_state_entropy=float(np.median(state_entropy_list)),
        median_kl_state=float(np.median(kl_state_list)),
        argmax_correct=argmax_correct / len(test_pf),
    )


def alpha_sweep_for_mode(seed, hmm, train_obs, test_pf,
                         terminal, init):
    rows = []
    for alpha in ALPHAS:
        for beta in BETAS:
            gdc = GDCForecaster(nA, alpha=alpha, theta=THETA, gamma=0.0,
                                beta=beta, transition_type='self_loop',
                                initial_dist=init,
                                terminal_behavior=terminal)
            gdc.fit(train_obs)
            ppl = perplexity_at_horizons(gdc, hmm, test_pf, [1])
            rows.append(dict(seed=seed, terminal=terminal, init=init,
                             alpha=alpha, beta=beta,
                             excess_pp=ppl[1]['excess_perplexity']))
    return rows


def main():
    print("=== sparse_large failure x mode 2x2 ===\n", flush=True)

    diag_rows = []
    sweep_rows = []
    chmm_rows = []
    for seed in [0, 1, 2]:
        print(f"--- seed {seed} ---", flush=True)
        hmm, train_obs, true_states, test_pf = setup_seed(seed)

        # CHMM reference (mode-independent)
        chmm = CHMMForecaster(nA, K=32, n_em_iters=50, seed=seed)
        chmm.fit(train_obs)
        ppl = perplexity_at_horizons(chmm, hmm, test_pf, [1])
        chmm_rows.append(dict(seed=seed,
                              excess_pp=ppl[1]['excess_perplexity']))

        for terminal, init in MODES:
            t0 = time.time()
            d = diagnostic_for_mode(seed, hmm, train_obs, true_states,
                                    test_pf, terminal, init)
            diag_rows.append(d)
            print(f"  {terminal:>7s} | {init:>15s}  "
                  f"excess_pp={d['excess_pp']:.4f}  "
                  f"oracle={d['oracle_excess_pp']:.4f}  "
                  f"ess={d['median_ess']:.0f}  "
                  f"top1={d['median_top1']:.3f}  "
                  f"H_state={d['median_state_entropy']:.2f}  "
                  f"argmax={d['argmax_correct']:.0%}  "
                  f"({time.time()-t0:.1f}s)",
                  flush=True)
            sweep_rows.extend(alpha_sweep_for_mode(seed, hmm, train_obs,
                                                   test_pf, terminal, init))

    diag_df = pd.DataFrame(diag_rows)
    sweep_df = pd.DataFrame(sweep_rows)
    chmm_df = pd.DataFrame(chmm_rows)

    diag_df.to_csv(os.path.join(HERE, 'sparse_large_modes_diag.csv'),
                   index=False)
    sweep_df.to_csv(os.path.join(HERE, 'sparse_large_modes_sweep.csv'),
                    index=False)

    print(f"\n## Mean diagnostic across seeds (val-pick: a=0.8, "
          f"theta={THETA}, beta=0.1)  CHMM K=32 = "
          f"{chmm_df.excess_pp.mean():.4f}\n", flush=True)
    diag_summary = (diag_df.groupby(['terminal','init'])
                          .agg(excess_pp=('excess_pp','mean'),
                               oracle=('oracle_excess_pp','mean'),
                               ess=('median_ess','mean'),
                               H_state=('median_state_entropy','mean'),
                               argmax=('argmax_correct','mean'))
                          .round(4))
    print(diag_summary)

    print(f"\n## Best alpha/beta per mode (mean over seeds)\n", flush=True)
    sweep_mean = (sweep_df.groupby(['terminal','init','alpha','beta'])
                          ['excess_pp'].mean().reset_index())
    for terminal, init in MODES:
        sub = sweep_mean[(sweep_mean.terminal==terminal)
                         & (sweep_mean.init==init)]
        best = sub.loc[sub.excess_pp.idxmin()]
        print(f"  {terminal:>7s} | {init:>15s}  "
              f"best (a={best.alpha}, b={best.beta}) "
              f"excess_pp={best.excess_pp:.4f}")
    print()

    print("## Full alpha-x-beta heatmap per mode (cols=beta, rows=alpha)\n")
    for terminal, init in MODES:
        sub = sweep_mean[(sweep_mean.terminal==terminal)
                         & (sweep_mean.init==init)]
        pv = sub.pivot(index='alpha', columns='beta', values='excess_pp')
        print(f"### {terminal} / {init}")
        print(pv.round(4))
        print()


if __name__ == "__main__":
    main()
