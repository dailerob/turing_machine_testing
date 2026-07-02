"""Emit the LaTeX body for paper Table 12 from table12_product_hmm_results.csv.

GDC row = fixed config (alpha=0.85,theta=0.005,beta=0.075); CHMM (best K)
and Parrot (best L,K) val-picked per horizon on val seeds; HPYLM/PPM/KN-3
fixed depth-3 d=0.5. Bold = best per column; KN-3 cells > 1.5 daggered.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import gen_table12_product_hmm as T12
from gen_table12_product_hmm import (N_VALUES, HORIZONS, TEST_SEEDS, VAL_SEEDS,
                                     METHODS, GDC_FIXED, GDC_FIXED_NAME, OUT_CSV)

ROW_ORDER = ['gdc', 'chmm', 'parrot', 'freq', 'hpylm', 'ppm', 'kn3']
PRETTY_TEX = {
    'gdc': rf"GDC ($\alpha{{=}}{GDC_FIXED[0]}$, $\theta{{=}}{GDC_FIXED[1]}$, $\beta{{=}}{GDC_FIXED[2]}$)",
    'chmm': "CHMM (best $K$)", 'parrot': "Parrot (best $L,K$)",
    'freq': "Freq (unigram)", 'hpylm': r"HPYLM ($D{=}3$, $d{=}0.5$)",
    'ppm': r"PPM-D ($D{=}3$, $d{=}0.5$)", 'kn3': r"KN-3 ($d{=}0.5$)"}
SCALE_LABEL = {40: r"(a) $N{=}40$ sequences = 800 training chars (1$\times$)",
               160: r"(b) $N{=}160$ sequences = 3{,}200 training chars (4$\times$)",
               640: r"(c) $N{=}640$ sequences = 12{,}800 training chars (16$\times$)"}


def cell_value(df, N, m, h):
    val_set, test_set = set(VAL_SEEDS), set(TEST_SEEDS)
    sub = df[(df.N_train == N) & (df.model_class == m) & (df.horizon == h)]
    if sub.empty:
        return float('nan')
    if m == 'gdc':
        t = sub[(sub.seed.isin(test_set)) & (sub.model == GDC_FIXED_NAME)]
        return float(t.excess_perplexity.mean()) if not t.empty else float('nan')
    if m == 'freq':
        return float(sub[sub.seed.isin(test_set)].excess_perplexity.mean())
    vmeans = sub[sub.seed.isin(val_set)].groupby('model')['excess_perplexity'].mean()
    if vmeans.empty:
        return float('nan')
    pick = vmeans.idxmin()
    return float(sub[(sub.seed.isin(test_set)) & (sub.model == pick)].excess_perplexity.mean())


def main():
    df = pd.read_csv(OUT_CSV)
    for N in N_VALUES:
        vals = {m: {h: cell_value(df, N, m, h) for h in HORIZONS} for m in METHODS}
        col_min = {}
        for h in HORIZONS:
            present = [vals[m][h] for m in METHODS if not np.isnan(vals[m][h])]
            col_min[h] = min(present) if present else float('nan')
        print(r"\multicolumn{6}{l}{\emph{" + SCALE_LABEL[N] + r"}} \\")
        for m in ROW_ORDER:
            cells = []
            for h in HORIZONS:
                v = vals[m][h]
                if np.isnan(v):
                    cells.append("---"); continue
                s = f"{v:.3f}"
                if m == 'kn3' and v > 1.5:
                    s += r"$^\dagger$"
                if round(v, 3) == round(col_min[h], 3):
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            label = PRETTY_TEX[m]
            print(f"{label} & " + " & ".join(f"{c}" for c in cells) + r" \\")
        print(r"\midrule")


if __name__ == "__main__":
    main()
