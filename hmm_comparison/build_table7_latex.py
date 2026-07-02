"""Emit the LaTeX body for paper Table 7 from table7_forecasting_results.csv.

Reads the raw per-(cell,config,horizon) rows produced by
gen_table7_forecasting.py, does the leakage-free val-pick aggregation
(per (regime, horizon, method): pick the config with lowest mean excess
PP over the validation HMM seeds, report its mean over the test HMM
seeds), and prints the LaTeX rows. Bold = best per column; KN-3 cells
with excess PP > 1.5 get a dagger (the wrapper assigns ~0 probability to
symbols appearing only at h>1 on near-deterministic regimes).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from gen_table7_forecasting import (HORIZONS, VAL_SEEDS, TEST_SEEDS,
                                     METHODS, PRETTY, OUT_CSV, REGIMES)

DAGGER_THRESHOLD = 1.5


def aggregate(df):
    val_set, test_set = set(VAL_SEEDS), set(TEST_SEEDS)
    table = {}
    for regime, *_ in REGIMES:
        table[regime] = {}
        rsub = df[df.regime == regime]
        for m in METHODS:
            msub = rsub[rsub.model_class == m]
            table[regime][m] = {}
            for h in HORIZONS:
                hsub = msub[msub.horizon == h]
                if hsub.empty:
                    table[regime][m][h] = float('nan'); continue
                if m == 'freq':
                    table[regime][m][h] = (
                        float(hsub[hsub.seed.isin(test_set)].excess_perplexity.mean())
                        if h == 1 else float('nan'))
                    continue
                val = hsub[hsub.seed.isin(val_set)]
                test = hsub[hsub.seed.isin(test_set)]
                vmeans = val.groupby('model')['excess_perplexity'].mean()
                if vmeans.empty:
                    table[regime][m][h] = float('nan'); continue
                pick = vmeans.idxmin()
                table[regime][m][h] = float(
                    test[test.model == pick].excess_perplexity.mean())
    return table


REGIME_HDR = {
    'cyclic': r"(a) cyclic ($\mathrm{advance\_prob}{=}0.95$): deterministic ring, "
              r"state $i$ emits $i \bmod n_A$",
    'reset_chain': r"(b) reset\_chain ($\mathrm{advance\_prob}{=}0.90$, "
                   r"$\mathrm{reset\_prob}{=}0.05$): linear chain with periodic resets",
    'bimodal': r"(c) bimodal ($\mathrm{sticky\_prob}{=}0.95$): two state clusters "
               r"with disjoint emission supports",
    'sparse': r"(d) sparse topology (fanout=2, $E_{\mathrm{conc}}{=}0.1$): each "
              r"state's transition supported on 2 random successors",
}


def main():
    df = pd.read_csv(OUT_CSV)
    table = aggregate(df)
    for regime, *_ in REGIMES:
        # per-column (horizon) minimum across all methods present
        col_min = {}
        for h in HORIZONS:
            vals = [table[regime][m][h] for m in METHODS
                    if not np.isnan(table[regime][m].get(h, float('nan')))]
            col_min[h] = min(vals) if vals else float('nan')
        print(r"\multicolumn{6}{l}{\emph{" + REGIME_HDR[regime] + r"}} \\")
        # rows ordered worst->best by h=1 (Freq top, best at bottom)
        order = sorted(METHODS, key=lambda m: -(table[regime][m].get(1, -1)
                       if not np.isnan(table[regime][m].get(1, float('nan')))
                       else -1))
        for m in order:
            cells = []
            for h in HORIZONS:
                v = table[regime][m].get(h, float('nan'))
                if np.isnan(v):
                    cells.append("---"); continue
                s = f"{v:.3f}"
                if m == 'kn3' and v > DAGGER_THRESHOLD:
                    s += r"$^\dagger$"
                if round(v, 3) == round(col_min[h], 3):
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            print(f"{PRETTY[m]:<8s}& " + " & ".join(f"{c:<14s}" for c in cells) + r" \\")
        print(r"\midrule")


if __name__ == "__main__":
    main()
