"""Compute the training-frequency baseline at TL=25.

The freq predictor outputs a fixed distribution over symbols equal to
the unigram frequency in the training data. It uses the test prefix
only for the cross-entropy floor target, not for inference.

Output:
  - hmm_comparison/freq_results_tl25.csv: per (regime, seed, N) cell
  - tables on stdout: per-regime tables with the new Freq column,
    matching the format of the seq_len_table.csv at TL=25
"""
from __future__ import annotations
import os, sys, csv
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from random_hmm import random_dense_hmm, random_sparse_topology_hmm

REGIMES = [
    ('dense_small',  10, 4, 'dense',  1.0, None),
    ('dense_large',  30, 8, 'dense',  1.0, None),
    ('det_small',    10, 4, 'dense',  0.1, None),
    ('det_large',    30, 8, 'dense',  0.1, None),
    ('sparse_small', 10, 4, 'sparse', 0.1, 2),
    ('sparse_large', 30, 8, 'sparse', 0.1, 2),
]
SEEDS = [0, 1, 2, 3, 4, 5]
TEST_SEEDS = {0, 1, 2}
N_TRAIN_VALUES = [25, 100, 400]
TRAIN_LEN = 25
N_TEST_PREFIXES = 100
TEST_PREFIX_LEN = 20
H = 1

OUT = os.path.join(HERE, 'freq_results_tl25.csv')


def compute_freq_excess_pp(hmm, train, test_pf, alpha_smooth=1e-6):
    """Compute excess perplexity of the freq baseline on this cell.
    Freq predictor: P(next=s) = (count(s in train) + alpha) / (total + nA*alpha)."""
    nA = hmm.nA
    counts = np.zeros(nA)
    for seq in train:
        for v in np.asarray(seq, dtype=np.int64):
            counts[v] += 1
    freq = (counts + alpha_smooth) / (counts.sum() + nA * alpha_smooth)

    Th = np.linalg.matrix_power(hmm.T, H)
    ces = []
    floors = []
    for prefix in test_pf:
        a = hmm.filter(prefix)
        true_next = a @ Th @ hmm.E
        true_safe = np.maximum(true_next, 1e-12)
        freq_safe = np.maximum(freq, 1e-12)
        ce = -float(np.sum(true_next * np.log2(freq_safe)))
        floor = -float(np.sum(true_next * np.log2(true_safe)))
        ces.append(ce)
        floors.append(floor)
    return float(2 ** (np.mean(ces) - np.mean(floors)))


def main():
    rows = []
    for (name, nS, nA, kind, conc, fanout) in REGIMES:
        seed_offset = (1 if 'det' in name else 0) \
                      + (2 if 'sparse' in name else 0)
        for seed in SEEDS:
            rng = np.random.default_rng(60000 + seed * 137 + nS * 7
                                        + nA * 11 + seed_offset)
            if kind == 'sparse':
                hmm = random_sparse_topology_hmm(nS, nA, rng,
                                                 fanout=fanout,
                                                 E_concentration=conc)
            else:
                hmm = random_dense_hmm(nS, nA, rng,
                                       T_concentration=1.0,
                                       E_concentration=conc)
            full_train = [hmm.sample(TRAIN_LEN, rng)[1]
                          for _ in range(max(N_TRAIN_VALUES))]
            test_pf = [hmm.sample(TEST_PREFIX_LEN, rng)[1]
                       for _ in range(N_TEST_PREFIXES)]
            for N in N_TRAIN_VALUES:
                train = full_train[:N]
                ex_pp = compute_freq_excess_pp(hmm, train, test_pf)
                rows.append(dict(regime=name, nS=nS, nA=nA, seed=seed,
                                 N=N, excess_pp=ex_pp))

    fields = ['regime', 'nS', 'nA', 'seed', 'N', 'excess_pp']
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {OUT}\n")

    # Aggregate to per-(regime, N) test_pp on test seeds
    df = pd.DataFrame(rows)
    test = df[df.seed.isin(TEST_SEEDS)]
    freq_pp = (test.groupby(['regime','N'])['excess_pp'].mean()
                   .reset_index().rename(columns={'excess_pp':'freq_pp'}))

    # Reconstruct tables: existing methods from seq_len_table.csv +
    # new Freq column.
    seq = pd.read_csv(os.path.join(HERE, 'seq_len_table.csv'))
    seq = seq[seq.train_len == TRAIN_LEN]

    # Pivot to (regime, N) × method
    pivot = (seq.pivot_table(index=['regime','N'],
                              columns='model_class',
                              values='test_pp')
                 .reset_index())
    merged = pivot.merge(freq_pp, on=['regime','N'])

    # Column order
    METHODS = ['gdc', 'chmm', 'alergia', 'parrot', 'hpylm', 'ppm', 'freq']
    PRETTY  = {'gdc':'GDC','chmm':'CHMM','alergia':'ALERGIA',
               'parrot':'Parrot','hpylm':'HPYLM','ppm':'PPM-D',
               'freq':'Freq'}

    print("Excess perplexity at TRAIN_LEN=25 with training-frequency "
          "baseline added.\n**Bold = best in row** (excluding Freq, "
          "which is shown as a reference baseline).\n")

    REGIMES_ORDER = [r[0] for r in REGIMES]
    N_VALUES_ORDER = [25, 100, 400]
    for regime in REGIMES_ORDER:
        print(f"## {regime}\n")
        print("| N | " + " | ".join(PRETTY[m] for m in METHODS) + " |")
        print("|---:|" + "---:|" * len(METHODS))
        for N in N_VALUES_ORDER:
            row = merged[(merged.regime == regime) & (merged.N == N)]
            if row.empty:
                continue
            r = row.iloc[0]
            vals = {m: float(r[m if m != 'freq' else 'freq_pp'])
                    for m in METHODS if m != 'freq'}
            vals['freq'] = float(r['freq_pp'])
            best = min(v for k, v in vals.items() if k != 'freq')
            cells = []
            for m in METHODS:
                v = vals[m]
                if m == 'freq':
                    cells.append(f"_{v:.4f}_")
                elif abs(v - best) < 1e-4:
                    cells.append(f"**{v:.4f}**")
                else:
                    cells.append(f"{v:.4f}")
            print(f"| {N} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
