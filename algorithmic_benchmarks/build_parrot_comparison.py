"""Side-by-side comparison: GDC vs CHMM vs ALERGIA vs PARROT (val-tuned).

Mirrors paper/tables.tex Tables 8 (standard) and 9 (no-read).

For PARROT: read parrot_benchmark_results.csv from parrot_eval.py.
For others: hardcoded from paper/tables.tex.
"""
from __future__ import annotations
import os
import csv
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARROT_CSV = os.path.join(HERE, 'parrot_benchmark_results.csv')

# Hardcoded from paper/tables.tex Tables 8 & 9 (best K per task chosen there).
PUBLISHED = {
    # variant, task: (gdc_errors, gdc_total, chmm, alergia)
    ('original', 'parity'):       (8,    506,    12,    12),
    ('original', 'increment'):    (0,    266,    0,     3),
    ('original', 'reverse'):      (149,  13646,  329,   601),
    ('original', 'binary_adder'): (59,   72217,  10,    5579),
    ('original', 'dyck1'):        (5059, 10920,  4497,  5557),
    ('noread',   'parity'):       (8,    506,    12,    12),
    ('noread',   'increment'):    (0,    266,    0,     0),
    ('noread',   'reverse'):      (0,    13646,  140,   6011),
    ('noread',   'binary_adder'): (0,    72217,  0,     1466),
}


def main():
    par = pd.read_csv(PARROT_CSV, keep_default_na=False)
    print(f"\n{'task':<14s}  {'variant':<9s}  {'L':>2s} {'K':>3s}  "
          f"{'GDC':>14s}  {'CHMM':>14s}  {'ALERGIA':>14s}  {'PARROT':>14s}  outcome")
    print("-" * 115)
    parrot_wins = gdc_wins = ties = 0
    for _, r in par.iterrows():
        task = r['task']; variant = r['variant']
        terr = int(r['tuple_errors'])
        npred = int(r['n_predictions'])
        # Dyck1 has variant 'n/a' in parrot output; treat it as 'original' for lookup.
        lookup_variant = 'original' if (variant == 'n/a' and task == 'dyck1') else variant
        if (lookup_variant, task) not in PUBLISHED:
            print(f"{task:<14s}  {variant:<9s}  parrot {terr}/{npred}  (no published comparison)")
            continue
        g, gt, c, a = PUBLISHED[(lookup_variant, task)]
        # Compare error rates
        gdc_rate = g / gt
        par_rate = terr / npred
        delta = (par_rate - gdc_rate) / max(gdc_rate, 1e-9) * 100 if gdc_rate > 0 else 0
        if gdc_rate == 0 and par_rate == 0:
            outcome = 'tied (both perfect)'
            ties += 1
        elif par_rate < gdc_rate:
            outcome = 'PARROT'; parrot_wins += 1
        elif par_rate > gdc_rate * 1.05:
            outcome = 'GDC'; gdc_wins += 1
        else:
            outcome = 'tied'; ties += 1
        print(f"{task:<14s}  {variant:<9s}  {int(r['L']):>2d} {int(r['K']):>3d}  "
              f"{g:>5d}/{gt:>6d}  {c:>5d}/{gt:>6d}  {a:>5d}/{gt:>6d}  "
              f"{terr:>5d}/{npred:>6d}  {outcome}")
    print()
    print(f"Parrot vs GDC: parrot {parrot_wins}, GDC {gdc_wins}, "
          f"tied {ties} (out of {parrot_wins + gdc_wins + ties})")


if __name__ == "__main__":
    main()
