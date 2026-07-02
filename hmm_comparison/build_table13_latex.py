"""Emit the LaTeX body for paper Table 13 from table13_scaling_results.csv.

Leakage-free val-pick per (regime, N, method); bold = best per column.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import gen_table13_scaling as T13
from gen_table7_forecasting import REGIMES, METHODS, PRETTY

ROW_ORDER = ['freq', 'kn3', 'ppm', 'alergia', 'parrot', 'hpylm', 'gdc', 'chmm']
REGIME_HDR = {'cyclic': '(a) cyclic', 'reset_chain': r'(b) reset\_chain',
              'bimodal': '(c) bimodal',
              'sparse': '(d) sparse topology (fanout=2)'}


def aggregate(df):
    val_set, test_set = set(T13.T7.VAL_SEEDS), set(T13.T7.TEST_SEEDS)
    table = {}
    for regime, *_ in REGIMES:
        table[regime] = {}
        rsub = df[df.regime == regime]
        for m in METHODS:
            table[regime][m] = {}
            for N in T13.N_VALUES:
                sub = rsub[(rsub.model_class == m) & (rsub.N_train == N)]
                if sub.empty:
                    table[regime][m][N] = float('nan'); continue
                if m == 'freq':
                    table[regime][m][N] = float(
                        sub[sub.seed.isin(test_set)].excess_perplexity.mean())
                    continue
                vmeans = sub[sub.seed.isin(val_set)].groupby('model')['excess_perplexity'].mean()
                if vmeans.empty:
                    table[regime][m][N] = float('nan'); continue
                pick = vmeans.idxmin()
                table[regime][m][N] = float(
                    sub[(sub.seed.isin(test_set)) & (sub.model == pick)].excess_perplexity.mean())
    return table


def main():
    df = pd.read_csv(T13.OUT_CSV)
    table = aggregate(df)
    for regime, *_ in REGIMES:
        col_min = {}
        for N in T13.N_VALUES:
            vals = [table[regime][m][N] for m in METHODS
                    if not np.isnan(table[regime][m].get(N, float('nan')))]
            col_min[N] = min(vals) if vals else float('nan')
        print(r"\multicolumn{6}{l}{\emph{" + REGIME_HDR[regime] + r"}} \\")
        for m in ROW_ORDER:
            cells = []
            for N in T13.N_VALUES:
                v = table[regime][m].get(N, float('nan'))
                if np.isnan(v):
                    cells.append("---"); continue
                if v >= 100:
                    # degenerate blow-up (sharp method on a single-cluster
                    # sample): scientific notation + ddagger.
                    exp = int(np.floor(np.log10(v)))
                    mant = v / 10 ** exp
                    s = rf"$\sim{mant:.0f}{{\times}}10^{{{exp}}}$\,$^\ddagger$"
                else:
                    s = f"{v:.3f}"
                    if round(v, 3) == round(col_min[N], 3):
                        s = r"\textbf{" + s + "}"
                cells.append(s)
            print(f"{PRETTY[m]:<8s}& " + " & ".join(f"{c:<14s}" for c in cells) + r" \\")
        print(r"\midrule")


if __name__ == "__main__":
    main()
