"""Side-by-side comparison: GDC vs CHMM vs ALERGIA vs PARROT (val-tuned).

For PARROT: pick best (L, K, alpha_prior) on val seeds {3,4,5} from
parrot_results.csv, report on test seeds {0,1,2}.

For GDC, CHMM, ALERGIA: lift directly from leakage_free_table_v2.csv,
which mirrors paper/tables.tex Table 7.
"""
from __future__ import annotations
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARROT_CSV    = os.path.join(HERE, 'parrot_results.csv')
HORIZON = 1
VAL_SEEDS = {3, 4, 5}
TEST_SEEDS = {0, 1, 2}
REGIMES = ['dense_small', 'dense_large', 'det_small', 'det_large',
           'sparse_small', 'sparse_large']
NS = [25, 100, 400]

# GDC / CHMM / ALERGIA test excess-perplexity at h=1 from paper/tables.tex Table 7
PUBLISHED = {
    ('dense_small',  25):  (1.0038, 1.0362, 1.0054),
    ('dense_small',  100): (1.0017, 1.0031, 1.0013),
    ('dense_small',  400): (1.0014, 1.0008, 1.0008),
    ('dense_large',  25):  (1.0044, 1.7623, 1.0247),
    ('dense_large',  100): (1.0012, 1.1013, 1.0055),
    ('dense_large',  400): (1.0005, 1.0100, 1.0015),
    ('det_small',    25):  (1.0111, 1.0387, 1.0092),
    ('det_small',    100): (1.0107, 1.0047, 1.0044),
    ('det_small',    400): (1.0077, 1.0012, 1.0034),
    ('det_large',    25):  (1.0100, 1.7667, 1.0643),
    ('det_large',    100): (1.0057, 1.0913, 1.0057),
    ('det_large',    400): (1.0045, 1.0112, 1.0021),
    ('sparse_small', 25):  (1.1086, 1.0543, 1.1931),
    ('sparse_small', 100): (1.0862, 1.0306, 1.1159),
    ('sparse_small', 400): (1.0811, 1.0226, 1.1343),
    ('sparse_large', 25):  (1.4103, 1.5656, 1.6557),
    ('sparse_large', 100): (1.3646, 1.2087, 1.4634),
    ('sparse_large', 400): (1.3171, 1.0745, 1.4234),
}


def main():
    par = pd.read_csv(PARROT_CSV)
    par_h = par[par.horizon == HORIZON].copy()

    rows = []
    parrot_wins = gdc_wins = ties = 0
    for regime in REGIMES:
        for N in NS:
            sub = par_h[(par_h.regime == regime) & (par_h.N_train == N)]
            v = sub[sub.seed.isin(VAL_SEEDS)]
            t = sub[sub.seed.isin(TEST_SEEDS)]
            val_means = v.groupby('model')['excess_perplexity'].mean()
            best_model = val_means.idxmin()
            test_pp = float(t[t.model == best_model].excess_perplexity.mean())
            val_pp = float(val_means.min())
            gdc_pp, chmm_pp, alergia_pp = PUBLISHED[(regime, N)]
            row = dict(regime=regime, N=N,
                       gdc=gdc_pp, chmm=chmm_pp, alergia=alergia_pp,
                       parrot=test_pp,
                       parrot_pick=best_model,
                       parrot_val=val_pp)
            # Compare parrot vs GDC
            d = (test_pp - row['gdc']) / row['gdc'] * 100
            if d < -2: tag = 'PARROT'; parrot_wins += 1
            elif d > 2: tag = 'GDC'; gdc_wins += 1
            else: tag = 'tied'; ties += 1
            row['vs_gdc'] = tag
            rows.append(row)

    print(f"\n{'regime':>13s} {'N':>4s}  {'GDC':>7s}  {'CHMM':>7s}  "
          f"{'ALERGIA':>7s}  {'PARROT':>7s}  {'parrot_pick':>22s}  vs_GDC")
    print("-" * 95)
    prev = None
    for r in rows:
        if prev and r['regime'] != prev: print()
        prev = r['regime']
        print(f"{r['regime']:>13s} {r['N']:>4d}  "
              f"{r['gdc']:>7.4f}  {r['chmm']:>7.4f}  "
              f"{r['alergia']:>7.4f}  {r['parrot']:>7.4f}  "
              f"{r['parrot_pick']:>22s}  {r['vs_gdc']}")

    print()
    print(f"Parrot vs GDC across 18 (regime, N) cells: "
          f"parrot {parrot_wins}, GDC {gdc_wins}, tied {ties}")

    out = os.path.join(HERE, 'parrot_comparison.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
