"""Argmax-rollout (autoregressive) accuracy on the 4 new regimes.

For each method's val-picked config (val-picked on h=1 excess_pp from
the bundled cross-entropy sweeps), we run a 20-step argmax rollout
per test prefix:
    cur = prefix
    for step in 1..H:
        dist = model.horizon_emission(cur, h=1)
        next_argmax = argmax(dist)
        record (step, next_argmax)
        cur = cur + [next_argmax]

Then accuracy at step h = % of test prefixes where next_argmax at
step h matches realized observation at position prefix_len + h - 1.

This is "option B" — autoregressive next-token forecasting where the
model conditions only on its own argmax outputs, not on the realized
observations. Errors compound across the rollout.

For all methods, rollout uses h=1 calls only (so HPYLM/PPM-D's
internal greedy rollout-for-h>1 logic is bypassed and they're treated
the same as the others).
"""
from __future__ import annotations
import os, sys, time, multiprocessing as mp
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from random_hmm import (random_bimodal_hmm, random_cyclic_hmm,
                       random_binary_deep_hmm,
                       random_reset_chain_hmm)
from generative_dense_chain import GenerativeDenseChain
from gdc_torch_discrete import horizon_emission_many

REGIMES = [
    ('bimodal_small',  10, 4, 'bimodal'),
    ('cyclic_K8',       8, 8, 'cyclic'),
    ('binary_deep',    30, 2, 'binary_deep'),
    ('reset_chain',    20, 4, 'reset_chain'),
]
SEEDS = [0, 1, 2, 3, 4, 5]
VAL_SEEDS = {3, 4, 5}
TEST_SEEDS = {0, 1, 2}
N_TRAIN_VALUES = [25, 100, 400]
TRAIN_LEN = 25
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
ROLLOUT_LEN = 20      # rollout 20 future steps
HORIZONS_REPORT = [1, 5, 20]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

OUT_RAW = os.path.join(HERE, 'rollout_accuracy.csv')


def make_hmm(kind, nS, nA, rng):
    if kind == 'bimodal':
        return random_bimodal_hmm(nS, nA, rng,
                                  sticky_prob=0.95, E_concentration=0.1)
    if kind == 'cyclic':
        return random_cyclic_hmm(nS, nA, rng,
                                 advance_prob=0.95, E_concentration=0.1)
    if kind == 'binary_deep':
        return random_binary_deep_hmm(nS, rng,
                                      fanout=2, E_concentration=0.1)
    if kind == 'reset_chain':
        return random_reset_chain_hmm(nS, nA, rng,
                                      advance_prob=0.90, reset_prob=0.05,
                                      E_concentration=0.1)


def setup_cell_data(regime_name, nS, nA, kind, seed, N):
    kind_tag = {'bimodal': 0, 'cyclic': 10, 'binary_deep': 20,
                'reset_chain': 30}[kind]
    rng = np.random.default_rng(70000 + kind_tag + seed * 137 + nS * 7
                                + nA * 11)
    hmm = make_hmm(kind, nS, nA, rng)
    full_train = [hmm.sample(TRAIN_LEN, rng)[1]
                  for _ in range(max(N_TRAIN_VALUES))]
    test_pf_full = [hmm.sample(TEST_PREFIX_LEN + ROLLOUT_LEN, rng)[1]
                    for _ in range(N_TEST_PREFIXES)]
    test_pf = [t[:TEST_PREFIX_LEN] for t in test_pf_full]
    realized = np.stack([t[TEST_PREFIX_LEN:TEST_PREFIX_LEN + ROLLOUT_LEN]
                         for t in test_pf_full])  # (B, ROLLOUT_LEN)
    return hmm, full_train[:N], test_pf, realized


def load_val_picks():
    """Pick best (regime, N, method_class) config from
    argmax_accuracy_horizons.csv on val seeds at h=1 excess_pp."""
    src = os.path.join(HERE, 'argmax_accuracy_horizons.csv')
    df = pd.read_csv(src)
    df = df[df.horizon == 1]
    cval = df[df.seed.isin(VAL_SEEDS)]
    val_means = (cval.groupby(['regime', 'N_train', 'model_class', 'model'])
                       ['excess_pp'].mean().reset_index())
    picks = (val_means.sort_values('excess_pp')
                       .groupby(['regime', 'N_train', 'model_class'])
                       .first().reset_index())
    return picks  # columns: regime, N_train, model_class, model, excess_pp


def parse_gdc(model_str):
    """Parse 'gdc-a0.85-b0.001' → (alpha=0.85, beta=0.001)."""
    parts = model_str.split('-')
    a = float(parts[1][1:])
    b = float(parts[2][1:])
    return a, b


def rollout_gdc_torch(train, alpha, beta, theta, primes, nA, n_steps):
    """20-step argmax rollout using torch kernel. primes shape (B, L).
    Returns (B, n_steps) argmax sequences."""
    seq_arrays = [s.reshape(-1, 1).astype(np.int64) for s in train]
    gdc = GenerativeDenseChain(
        seq_arrays, alpha=alpha, theta=theta, gamma=0.0, beta=beta,
        transition_type='self_loop', initial_dist='uniform',
        terminal_behavior='absorb')
    sym = gdc.states[:, 0].astype(np.int64)
    cur = primes.copy()  # (B, L)
    out = np.zeros((primes.shape[0], n_steps), dtype=np.int64)
    for step in range(n_steps):
        preds = horizon_emission_many(
            symbol_of_state=sym, terminal_mask=gdc.terminal_mask,
            start_mask=gdc.start_mask,
            primes=cur, horizons=[1], nA=nA,
            alpha=alpha, theta=theta, beta=beta,
            transition_type='self_loop',
            terminal_behavior='absorb', initial_dist='uniform',
            device=DEVICE, dtype=DTYPE).cpu().numpy().reshape(
                primes.shape[0], nA)
        argmax = preds.argmax(axis=1)
        out[:, step] = argmax
        cur = np.concatenate([cur, argmax[:, None]], axis=1)
    return out


def rollout_cpu_method(model_predict_fn, prefix, n_steps):
    """Argmax rollout using a model's per-prefix predict_distribution
    (h=1 only). Returns array of length n_steps of argmax symbols."""
    cur = list(int(x) for x in prefix)
    out = np.zeros(n_steps, dtype=np.int64)
    for step in range(n_steps):
        dist = model_predict_fn(np.asarray(cur, dtype=np.int64))
        a = int(np.argmax(dist))
        out[step] = a
        cur.append(a)
    return out


def gdc_cell(args):
    regime_name, nS, nA, kind, seed, N, picks = args
    hmm, train, test_pf, realized = setup_cell_data(
        regime_name, nS, nA, kind, seed, N)
    primes = np.stack([np.asarray(p, dtype=np.int64) for p in test_pf])
    pick = picks[(picks.regime == regime_name)
                 & (picks.N_train == N)
                 & (picks.model_class == 'gdc')]
    if pick.empty:
        return []
    alpha, beta = parse_gdc(pick.model.iloc[0])
    theta = 0.001
    rollouts = rollout_gdc_torch(train, alpha, beta, theta,
                                  primes, nA, ROLLOUT_LEN)
    rows = []
    for step in range(ROLLOUT_LEN):
        acc = float(np.mean(rollouts[:, step] == realized[:, step]))
        rows.append(dict(regime=regime_name, nS=nS, nA=nA, seed=seed,
                         N_train=N, model_class='gdc',
                         model=pick.model.iloc[0],
                         step=step + 1, accuracy=acc))
    return rows


def cpu_cell(args):
    regime_name, nS, nA, kind, seed, N, picks = args
    sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
    from chmm_alergia_wrappers import fit_chmm, fit_alergia
    from discrete_parrot import DiscreteParrotPool
    from discrete_hpylm import HPYLMPool
    from discrete_ppm import PPMPool

    hmm, train, test_pf, realized = setup_cell_data(
        regime_name, nS, nA, kind, seed, N)
    rows = []
    base = dict(regime=regime_name, nS=nS, nA=nA, seed=seed, N_train=N)

    def add_rollouts(model_class, model_name, predict_fn):
        # Per-prefix rollout (model conditions on its own argmax outputs)
        all_rolls = np.zeros((N_TEST_PREFIXES, ROLLOUT_LEN), dtype=np.int64)
        for i, p in enumerate(test_pf):
            all_rolls[i] = rollout_cpu_method(predict_fn, p, ROLLOUT_LEN)
        for step in range(ROLLOUT_LEN):
            acc = float(np.mean(all_rolls[:, step] == realized[:, step]))
            rows.append(dict(base, model_class=model_class,
                             model=model_name,
                             step=step + 1, accuracy=acc))

    # CHMM
    pick = picks[(picks.regime == regime_name)
                 & (picks.N_train == N) & (picks.model_class == 'chmm')]
    if not pick.empty:
        K = int(pick.model.iloc[0].split('K')[1])
        try:
            m = fit_chmm(train, nA, K=K, n_em_iters=50)
            add_rollouts('chmm', pick.model.iloc[0],
                         lambda p: m.horizon_emission(p, h=1))
        except Exception:
            pass

    # ALERGIA
    pick = picks[(picks.regime == regime_name)
                 & (picks.N_train == N) & (picks.model_class == 'alergia')]
    if not pick.empty:
        try:
            m = fit_alergia(train, nA, eps=0.05)
            add_rollouts('alergia', pick.model.iloc[0],
                         lambda p: m.horizon_emission(p, h=1))
        except Exception:
            pass

    # Parrot
    pick = picks[(picks.regime == regime_name)
                 & (picks.N_train == N) & (picks.model_class == 'parrot')]
    if not pick.empty:
        # Parse parrot-L{L}-K{K}-a{ap}
        parts = pick.model.iloc[0].split('-')
        L = int(parts[1][1:]); K = int(parts[2][1:])
        ap = float(parts[3][1:])
        pool = DiscreteParrotPool(train, alphabet_size=nA, L=L)
        add_rollouts('parrot', pick.model.iloc[0],
                     lambda p: pool.predict_distribution(np.asarray(p),
                                                          h=1, K=K,
                                                          alpha_prior=ap))

    # HPYLM
    pick = picks[(picks.regime == regime_name)
                 & (picks.N_train == N) & (picks.model_class == 'hpylm')]
    if not pick.empty:
        # hpylm-D{D}-d{d}-a{c}
        parts = pick.model.iloc[0].split('-')
        D = int(parts[1][1:]); d = float(parts[2][1:]); c = float(parts[3][1:])
        pool = HPYLMPool(train, alphabet_size=nA, max_depth=D,
                         discount=d, concentration=c, seed=seed)
        add_rollouts('hpylm', pick.model.iloc[0],
                     lambda p: pool.predict_distribution(np.asarray(p),
                                                          h=1,
                                                          alpha_prior=0.01))

    # PPM-D
    pick = picks[(picks.regime == regime_name)
                 & (picks.N_train == N) & (picks.model_class == 'ppm')]
    if not pick.empty:
        parts = pick.model.iloc[0].split('-')
        D = int(parts[1][1:]); d = float(parts[2][1:])
        pool = PPMPool(train, alphabet_size=nA, max_depth=D, discount=d)
        add_rollouts('ppm', pick.model.iloc[0],
                     lambda p: pool.predict_distribution(np.asarray(p),
                                                          h=1,
                                                          alpha_prior=0.01))

    # Freq baseline (predicts argmax of training-unigram every step)
    counts = np.zeros(nA)
    for s in train:
        for v in np.asarray(s, dtype=np.int64):
            counts[v] += 1
    freq = (counts + 1e-6) / (counts.sum() + nA * 1e-6)
    freq_argmax = int(np.argmax(freq))
    for step in range(ROLLOUT_LEN):
        acc = float(np.mean(realized[:, step] == freq_argmax))
        rows.append(dict(base, model_class='freq', model='freq',
                         step=step + 1, accuracy=acc))

    # Oracle: at each step, use the true filtering+propagation distribution
    # to take argmax, condition on argmax, repeat
    nS_h = hmm.nS
    for i, p in enumerate(test_pf):
        cur = list(int(x) for x in p)
        post = hmm.filter(np.asarray(cur, dtype=np.int64))
        rolls = np.zeros(ROLLOUT_LEN, dtype=np.int64)
        for step in range(ROLLOUT_LEN):
            next_dist = post @ hmm.T @ hmm.E
            a = int(np.argmax(next_dist))
            rolls[step] = a
            # Update posterior: transition + emit-conditioned on a
            post = post @ hmm.T
            post = post * hmm.E[:, a]
            s_ = post.sum()
            if s_ > 0:
                post = post / s_
            cur.append(a)
        if i == 0:
            oracle_rolls = np.zeros((N_TEST_PREFIXES, ROLLOUT_LEN),
                                     dtype=np.int64)
        oracle_rolls[i] = rolls
    for step in range(ROLLOUT_LEN):
        acc = float(np.mean(oracle_rolls[:, step] == realized[:, step]))
        rows.append(dict(base, model_class='oracle', model='oracle',
                         step=step + 1, accuracy=acc))

    return rows


def main():
    picks = load_val_picks()
    print("Loaded val picks:")
    print(picks.to_string(index=False))
    print()

    tasks = [(name, nS, nA, kind, seed, N, picks)
             for (name, nS, nA, kind) in REGIMES
             for seed in SEEDS for N in N_TRAIN_VALUES]
    n_cells = len(tasks)
    print(f"=== rollout argmax sweep ({ROLLOUT_LEN}-step), TL={TRAIN_LEN} ===")
    print(f"  Cells: {n_cells}\n", flush=True)

    t0 = time.time()
    gdc_rows = []
    for i, args in enumerate(tasks):
        gdc_rows.extend(gdc_cell(args))
        if (i + 1) % 12 == 0 or (i + 1) == n_cells:
            print(f"  GDC pass: {i+1}/{n_cells} cells "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    print()

    n_workers = max(1, min(8, (os.cpu_count() or 4) - 1))
    print(f"  Starting CPU pass with {n_workers} workers...", flush=True)
    t1 = time.time()
    cpu_rows = []
    with mp.Pool(processes=n_workers) as pool:
        done = 0
        for cell_rows in pool.imap_unordered(cpu_cell, tasks, chunksize=1):
            cpu_rows.extend(cell_rows); done += 1
            if done % 6 == 0 or done == n_cells:
                print(f"  CPU pass: {done}/{n_cells} cells "
                      f"[{time.time()-t1:.0f}s]", flush=True)

    df = pd.DataFrame(gdc_rows + cpu_rows)
    df.to_csv(OUT_RAW, index=False)
    print(f"\nWrote {OUT_RAW} ({len(df)} rows, "
          f"{time.time()-t0:.0f}s total)\n")

    # Aggregation
    METHODS = ['gdc', 'chmm', 'alergia', 'parrot', 'hpylm', 'ppm',
               'freq', 'oracle']
    PRETTY = {'gdc':'GDC','chmm':'CHMM','alergia':'ALERGIA',
              'parrot':'Parrot','hpylm':'HPYLM','ppm':'PPM-D',
              'freq':'Freq','oracle':'Oracle'}

    test_df = df[df.seed.isin(TEST_SEEDS)]
    for h in HORIZONS_REPORT:
        print(f"\n## {ROLLOUT_LEN}-step rollout — argmax accuracy at "
              f"step {h} (autoregressive, all methods condition on "
              f"their own outputs)\n")
        for regime, *_ in REGIMES:
            print(f"### {regime}\n")
            print("| N | " + " | ".join(PRETTY[m] for m in METHODS) + " |")
            print("|---:|" + "---:|" * len(METHODS))
            for N in N_TRAIN_VALUES:
                cell = test_df[(test_df.regime == regime)
                               & (test_df.N_train == N)
                               & (test_df.step == h)]
                test_acc = {}
                for m in METHODS:
                    sub = cell[cell.model_class == m]
                    if sub.empty:
                        test_acc[m] = float('nan')
                    else:
                        test_acc[m] = float(sub.accuracy.mean())
                non_special = [test_acc[m] for m in METHODS
                               if m not in ('oracle', 'freq')
                               and not np.isnan(test_acc[m])]
                best = max(non_special) if non_special else float('nan')
                cells = []
                for m in METHODS:
                    v = test_acc[m]
                    if np.isnan(v):
                        cells.append('—')
                    elif m in ('oracle', 'freq'):
                        cells.append(f"_{v:.3f}_")
                    elif abs(v - best) < 1e-3:
                        cells.append(f"**{v:.3f}**")
                    else:
                        cells.append(f"{v:.3f}")
                print(f"| {N} | " + " | ".join(cells) + " |")
            print()


if __name__ == "__main__":
    main()
