"""Leakage-free table v2: GDC selected over expanded grid
   (5 alpha-theta-beta x 2 terminal_behavior x 2 initial_dist = 20 candidates).

For GDC: pick best on val seeds {3,4,5} from gdc_expanded_results.csv,
         report on test seeds {0,1,2} from same file.
For CHMM: pick best K on val from perplexity_val_results.csv,
          report on test from perplexity_sweep_results.csv.
For ALERGIA: fixed eps=0.05, report from perplexity_sweep_results.csv.
"""
from __future__ import annotations
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GDC_CSV  = os.path.join(HERE, 'gdc_expanded_results.csv')
VAL_CSV  = os.path.join(HERE, 'perplexity_val_results.csv')
TEST_CSV = os.path.join(HERE, 'perplexity_sweep_results.csv')

REGIMES = ['dense_small', 'dense_large', 'det_small', 'det_large',
           'sparse_small', 'sparse_large']
N_VALUES = [25, 100, 400]
HORIZON = 1
VAL_SEEDS  = {3, 4, 5}
TEST_SEEDS = {0, 1, 2}


def main():
    gdc = pd.read_csv(GDC_CSV)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)

    gdc_h  = gdc[gdc.horizon == HORIZON].copy()
    val_h  = val[val.horizon == HORIZON].copy()
    test_h = test[test.horizon == HORIZON].copy()

    rows = []
    for regime in REGIMES:
        for N in N_VALUES:
            row = dict(regime=regime, N=N)

            # --- GDC ---
            gsub = gdc_h[(gdc_h.regime == regime) & (gdc_h.N_train == N)]
            gval = gsub[gsub.seed.isin(VAL_SEEDS)]
            gtest = gsub[gsub.seed.isin(TEST_SEEDS)]
            val_means = gval.groupby('model')['excess_perplexity'].mean()
            best_model = val_means.idxmin()
            test_pp = float(gtest[gtest.model == best_model]
                              .excess_perplexity.mean())
            row['gdc_pick'] = best_model
            row['gdc_val_pp'] = float(val_means.min())
            row['gdc_test_pp'] = test_pp

            # --- CHMM ---
            csub_val = val_h[(val_h.regime == regime) & (val_h.N_train == N)]
            csub_val = csub_val[csub_val.model.str.startswith('chmm-')]
            cmeans = csub_val.groupby('model')['excess_perplexity'].mean()
            best_chmm = cmeans.idxmin()
            ctest = test_h[(test_h.regime == regime) &
                            (test_h.N_train == N) &
                            (test_h.model == best_chmm)]
            row['chmm_pick'] = best_chmm
            row['chmm_test_pp'] = float(ctest.excess_perplexity.mean())

            # --- ALERGIA (single config) ---
            asub = test_h[(test_h.regime == regime) &
                           (test_h.N_train == N) &
                           (test_h.model == 'alergia-eps0.05')]
            row['alergia_test_pp'] = float(asub.excess_perplexity.mean())

            rows.append(row)

    df = pd.DataFrame(rows)
    out = os.path.join(HERE, 'leakage_free_table_v2.csv')
    df.to_csv(out, index=False)
    print(f"Wrote {out}\n")

    print(f"{'regime':>14s} {'N':>4s}  "
          f"{'GDC pick':>62s}  {'GDC':>7s}  "
          f"{'CHMM':>9s}  {'CHMM':>7s}  {'ALERGIA':>8s}")
    for r in rows:
        print(f"{r['regime']:>14s} {r['N']:>4d}  "
              f"{r['gdc_pick']:>62s}  {r['gdc_test_pp']:>7.4f}  "
              f"{r['chmm_pick']:>9s}  {r['chmm_test_pp']:>7.4f}  "
              f"{r['alergia_test_pp']:>8.4f}")


if __name__ == "__main__":
    main()
