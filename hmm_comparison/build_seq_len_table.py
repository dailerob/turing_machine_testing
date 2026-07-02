"""Aggregate seq_len_<TL>_results.csv files into a leakage-free table.

Per (TRAIN_LEN, regime, N, model_class):
  - val-pick the best-mean-val-excess-perplexity model on VAL_SEEDS
  - report test mean on TEST_SEEDS at horizon 1

Methods covered: GDC (5 configs), CHMM (3 K), ALERGIA (1, no tune),
                 Parrot (40), HPYLM (36), PPM-D (12).

Output: hmm_comparison/seq_len_table.csv  (long-format)
        + a markdown summary printed to stdout suitable for pasting
          into the writeup.
"""
from __future__ import annotations
import os, glob, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REGIMES = ['dense_small', 'dense_large', 'det_small', 'det_large',
           'sparse_small', 'sparse_large']
N_VALUES = [25, 100, 400]
HORIZON = 1
VAL_SEEDS = {3, 4, 5}
TEST_SEEDS = {0, 1, 2}
MODEL_CLASSES = ['gdc', 'chmm', 'alergia', 'parrot', 'hpylm', 'ppm']
PRETTY_CLASS = {'gdc': 'GDC', 'chmm': 'CHMM', 'alergia': 'ALERGIA',
                'parrot': 'Parrot', 'hpylm': 'HPYLM', 'ppm': 'PPM-D'}


def load_all():
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, 'seq_len_*_results.csv'))):
        m = re.search(r'seq_len_(\d+)_results\.csv$', path)
        if not m:
            continue
        df = pd.read_csv(path)
        rows.append(df)
    if not rows:
        raise SystemExit('No seq_len_*_results.csv files found.')
    return pd.concat(rows, ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.horizon == HORIZON].copy()
    out = []
    for tl, sub_tl in df.groupby('train_len'):
        for regime in REGIMES:
            for N in N_VALUES:
                cell = sub_tl[(sub_tl.regime == regime)
                              & (sub_tl.N_train == N)]
                for cls in MODEL_CLASSES:
                    csub = cell[cell.model_class == cls]
                    if csub.empty:
                        continue
                    cval = csub[csub.seed.isin(VAL_SEEDS)]
                    ctest = csub[csub.seed.isin(TEST_SEEDS)]
                    if cls == 'alergia':
                        # Single config; no val-pick
                        models = csub.model.unique()
                        assert len(models) == 1, models
                        pick = models[0]
                    else:
                        val_means = (cval.groupby('model')['excess_perplexity']
                                         .mean())
                        if val_means.empty:
                            continue
                        pick = val_means.idxmin()
                    test_pp = float(ctest[ctest.model == pick]
                                    .excess_perplexity.mean())
                    val_pp = (float(cval[cval.model == pick]
                                    .excess_perplexity.mean())
                              if cls != 'alergia' else float('nan'))
                    out.append(dict(train_len=int(tl), regime=regime, N=N,
                                    model_class=cls, pick=pick,
                                    val_pp=val_pp, test_pp=test_pp))
    return pd.DataFrame(out)


def print_markdown_table(agg: pd.DataFrame):
    """One table per TRAIN_LEN, regimes × N × method."""
    train_lens = sorted(agg.train_len.unique())

    print("# HMM forecasting — sequence-length scaling\n")
    print("Excess perplexity (lower bound 1.000) at horizon $h{=}1$, "
          "test mean over seeds {0,1,2} after val-picking per (TRAIN_LEN, "
          "regime, N) on val seeds {3,4,5}. **Bold = best in row.**\n")

    for tl in train_lens:
        print(f"\n## TRAIN_LEN = {tl}\n")
        print("| Regime | N | "
              + " | ".join(PRETTY_CLASS[c] for c in MODEL_CLASSES)
              + " |")
        print("|---|---:|" + "---:|" * len(MODEL_CLASSES))
        sub = agg[agg.train_len == tl]
        for regime in REGIMES:
            for N in N_VALUES:
                row_cells = []
                row_data = sub[(sub.regime == regime) & (sub.N == N)]
                pp_by_class = {}
                for cls in MODEL_CLASSES:
                    rec = row_data[row_data.model_class == cls]
                    if rec.empty or np.isnan(rec.test_pp.iloc[0]):
                        pp_by_class[cls] = float('nan')
                    else:
                        pp_by_class[cls] = float(rec.test_pp.iloc[0])
                vals = [v for v in pp_by_class.values() if not np.isnan(v)]
                best = min(vals) if vals else float('nan')
                for cls in MODEL_CLASSES:
                    v = pp_by_class[cls]
                    if np.isnan(v):
                        row_cells.append('—')
                    elif abs(v - best) < 1e-4:
                        row_cells.append(f"**{v:.4f}**")
                    else:
                        row_cells.append(f"{v:.4f}")
                print(f"| {regime} | {N} | " + " | ".join(row_cells) + " |")

    # Cross-TRAIN_LEN summary: # of regime×N cells each method wins per TL
    print("\n## Win counts per TRAIN_LEN (out of 18 regime×N cells)\n")
    print("| Method | " + " | ".join(f"TL={tl}" for tl in train_lens) + " |")
    print("|---|" + "---:|" * len(train_lens))
    for cls in MODEL_CLASSES:
        wins = []
        for tl in train_lens:
            sub = agg[agg.train_len == tl]
            won = 0
            for regime in REGIMES:
                for N in N_VALUES:
                    row = sub[(sub.regime == regime) & (sub.N == N)]
                    if row.empty:
                        continue
                    cls_pp = float(row[row.model_class == cls].test_pp.iloc[0]
                                   if not row[row.model_class == cls].empty
                                   else float('nan'))
                    all_pp = [float(r.test_pp) for _, r in row.iterrows()
                              if not np.isnan(r.test_pp)]
                    if not all_pp or np.isnan(cls_pp):
                        continue
                    if abs(cls_pp - min(all_pp)) < 1e-4:
                        won += 1
            wins.append(won)
        print(f"| {PRETTY_CLASS[cls]} | "
              + " | ".join(str(w) for w in wins) + " |")


def main():
    df = load_all()
    agg = aggregate(df)
    out_csv = os.path.join(HERE, 'seq_len_table.csv')
    agg.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}\n", flush=True)
    print_markdown_table(agg)


if __name__ == "__main__":
    main()
