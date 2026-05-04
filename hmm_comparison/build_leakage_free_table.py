"""Build a leakage-free comparison table.

Protocol:
  1. For each (regime, N) and each model class (gdc, chmm, alergia):
       - Pick the candidate config with lowest mean val excess PP at h=1
         on the val seeds {3, 4, 5} (from perplexity_val_results.csv).
  2. Look up the corresponding TEST excess PP for that config from
     perplexity_sweep_results.csv (test seeds {0, 1, 2}).

  Selection happens entirely on val; reporting happens on test; no overlap.
"""
from __future__ import annotations
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_CSV  = os.path.join(HERE, 'perplexity_val_results.csv')
TEST_CSV = os.path.join(HERE, 'perplexity_sweep_results.csv')

REGIMES = ['dense_small', 'dense_large', 'det_small', 'det_large',
           'sparse_small', 'sparse_large']
N_VALUES = [25, 100, 400]
HORIZON = 1   # selection metric

MODEL_CLASSES = {
    'gdc':     lambda m: m.startswith('gdc-'),
    'chmm':    lambda m: m.startswith('chmm-'),
    'alergia': lambda m: m.startswith('alergia-'),
}


def main():
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)
    val_h  = val[val.horizon == HORIZON].copy()
    test_h = test[test.horizon == HORIZON].copy()

    rows = []
    for regime in REGIMES:
        for N in N_VALUES:
            row = dict(regime=regime, N=N)
            for cls_name, pred in MODEL_CLASSES.items():
                # On val, average excess PP over val seeds for each candidate.
                vsub = val_h[(val_h.regime == regime) & (val_h.N_train == N)]
                vsub = vsub[vsub.model.apply(pred)]
                if vsub.empty:
                    row[f'{cls_name}_pick'] = '—'
                    row[f'{cls_name}_test_excess_pp'] = float('nan')
                    continue
                grouped = vsub.groupby('model')['excess_perplexity'].mean()
                best_model = grouped.idxmin()
                # On test: average excess PP at this picked model over test seeds.
                tsub = test_h[(test_h.regime == regime) &
                               (test_h.N_train == N) &
                               (test_h.model == best_model)]
                if tsub.empty:
                    test_pp = float('nan')
                else:
                    test_pp = float(tsub.excess_perplexity.mean())
                row[f'{cls_name}_pick'] = best_model
                row[f'{cls_name}_val_excess_pp'] = float(grouped.min())
                row[f'{cls_name}_test_excess_pp'] = test_pp
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(HERE, 'leakage_free_table.csv')
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}\n")

    # Pretty print
    print(f"{'regime':>14s}  {'N':>4s}  "
          f"{'GDC pick':>22s}  {'GDC test':>9s}  "
          f"{'CHMM pick':>10s}  {'CHMM test':>10s}  "
          f"{'ALERGIA pick':>16s}  {'ALERGIA test':>12s}")
    for r in rows:
        print(f"{r['regime']:>14s}  {r['N']:>4d}  "
              f"{r['gdc_pick']:>22s}  {r['gdc_test_excess_pp']:>9.4f}  "
              f"{r['chmm_pick']:>10s}  {r['chmm_test_excess_pp']:>10.4f}  "
              f"{r['alergia_pick']:>16s}  {r['alergia_test_excess_pp']:>12.4f}")


if __name__ == "__main__":
    main()
