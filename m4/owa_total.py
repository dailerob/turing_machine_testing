"""Compute M4-style series-weighted total OWA across all 100,000 series.

M4's published "Total" across the dataset weights every series equally
(each contributes one sMAPE and one MASE), then aggregates:
  total_sMAPE = mean over all 100k series of per-series sMAPE
  total_MASE  = mean over all 100k series of per-series MASE
  total_OWA   = 0.5 * (total_sMAPE / total_naive2_sMAPE +
                       total_MASE  / total_naive2_MASE)

Naive 2 totals (computed exactly from m4/naive2.py):
  Yearly    23000 series,  16.342 / 3.974
  Quarterly 24000 series,  11.012 / 1.371
  Monthly   48000 series,  14.427 / 1.063
  Weekly      359 series,   9.161 / 2.777
  Daily      4227 series,   3.045 / 3.278
  Hourly      414 series,  18.383 / 2.395
"""
from __future__ import annotations
import csv
import os
import sys
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

FREQS = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
SERIES_COUNT = {"Yearly": 23000, "Quarterly": 24000, "Monthly": 48000,
                "Weekly": 359, "Daily": 4227, "Hourly": 414}
NAIVE2 = {
    "Yearly":    (16.342, 3.974),
    "Quarterly": (11.012, 1.371),
    "Monthly":   (14.427, 1.063),
    "Weekly":    ( 9.161, 2.777),
    "Daily":     ( 3.045, 3.278),
    "Hourly":    (18.383, 2.395),
}


def load(freq):
    """Returns dict sid -> {model: (val_smape, test_smape, val_mase, test_mase)}"""
    csv_path = os.path.join(HERE, freq.lower(), "clean_eval_results.csv")
    by_sid = defaultdict(dict)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            def _f(k):
                v = r.get(k, '')
                if v == '' or v.lower() == 'nan': return float('nan')
                try: return float(v)
                except ValueError: return float('nan')
            by_sid[r['sid']][r['model']] = (
                _f('val_smape'), _f('test_smape'),
                _f('val_mase'),  _f('test_mase'))
    return by_sid


def main():
    # Aggregate per-series sMAPE and MASE for: (1) per-series val-tuned,
    # (2) global pick by val-sMAPE, (2') global pick by val-MASE
    all_n2_sm, all_n2_ma = [], []
    all_ps_sm, all_ps_ma = [], []
    all_g_sm_sm, all_g_sm_ma = [], []   # picks by val-sMAPE
    all_g_ma_sm, all_g_ma_ma = [], []   # picks by val-MASE
    per_freq = []

    for freq in FREQS:
        by_sid = load(freq)
        sids = list(by_sid.keys())
        # Per-series val-tuned (by val-sMAPE), recording (test_smape, test_mase)
        ps_sm, ps_ma = [], []
        # For global picks: compute mean val sMAPE / val MASE per config
        models = set()
        for d in by_sid.values(): models.update(d.keys())
        global_val_sm = {m: [] for m in models}
        global_val_ma = {m: [] for m in models}
        for sid, d in by_sid.items():
            for m, (vs, ts, vm, tm) in d.items():
                if m == 'naive_last':
                    continue  # exclude naive_last fallback from picks
                if not np.isnan(vs): global_val_sm[m].append(vs)
                if not np.isnan(vm): global_val_ma[m].append(vm)
        global_means_sm = {m: np.mean(vs) if vs else np.inf
                           for m, vs in global_val_sm.items()}
        global_means_ma = {m: np.mean(vs) if vs else np.inf
                           for m, vs in global_val_ma.items()}
        gsm_pick = min(global_means_sm, key=global_means_sm.get)
        gma_pick = min(global_means_ma, key=global_means_ma.get)
        g_sm_sm, g_sm_ma = [], []
        g_ma_sm, g_ma_ma = [], []
        for sid, d in by_sid.items():
            # per-series tune (by val_smape; if nan, fall back to test of naive_last)
            elig = [(name, vs, ts, vm, tm) for name, (vs, ts, vm, tm) in d.items()
                    if not np.isnan(vs) and name != 'naive_last']
            if elig:
                _, _, ts, _, tm = min(elig, key=lambda x: x[1])
            else:
                _, ts, _, tm = d.get('naive_last', (np.nan, np.nan, np.nan, np.nan))
            if not np.isnan(ts): ps_sm.append(ts)
            if not np.isnan(tm): ps_ma.append(tm)
            # global by val-sMAPE pick
            if gsm_pick in d:
                _, ts, _, tm = d[gsm_pick]
                if not np.isnan(ts): g_sm_sm.append(ts)
                if not np.isnan(tm): g_sm_ma.append(tm)
            # global by val-MASE pick
            if gma_pick in d:
                _, ts, _, tm = d[gma_pick]
                if not np.isnan(ts): g_ma_sm.append(ts)
                if not np.isnan(tm): g_ma_ma.append(tm)

        ps_sm = np.array(ps_sm); ps_ma = np.array(ps_ma)
        g_sm_sm = np.array(g_sm_sm); g_sm_ma = np.array(g_sm_ma)
        g_ma_sm = np.array(g_ma_sm); g_ma_ma = np.array(g_ma_ma)
        n2_sm, n2_ma = NAIVE2[freq]

        owa_ps = 0.5*(ps_sm.mean()/n2_sm + ps_ma.mean()/n2_ma)
        owa_gsm = 0.5*(g_sm_sm.mean()/n2_sm + g_sm_ma.mean()/n2_ma)
        owa_gma = 0.5*(g_ma_sm.mean()/n2_sm + g_ma_ma.mean()/n2_ma)
        per_freq.append((freq, len(sids), n2_sm, n2_ma,
                         ps_sm.mean(), ps_ma.mean(), owa_ps,
                         g_sm_sm.mean(), g_sm_ma.mean(), owa_gsm, gsm_pick,
                         g_ma_sm.mean(), g_ma_ma.mean(), owa_gma, gma_pick))

        # Accumulate Naive 2 means weighted by series for the global total
        all_n2_sm.append(n2_sm * len(sids)); all_n2_ma.append(n2_ma * len(sids))
        all_ps_sm.extend(ps_sm.tolist()); all_ps_ma.extend(ps_ma.tolist())
        all_g_sm_sm.extend(g_sm_sm.tolist()); all_g_sm_ma.extend(g_sm_ma.tolist())
        all_g_ma_sm.extend(g_ma_sm.tolist()); all_g_ma_ma.extend(g_ma_ma.tolist())

    print(f"{'freq':>10s}  {'n':>6s}  "
          f"{'(1) sMAPE':>10s} {'MASE':>6s} {'OWA':>6s}  "
          f"{'(2sM) OWA':>9s}  {'(2MA) OWA':>9s}")
    for (freq, n, n2sm, n2ma, ps_s, ps_m, ps_o,
         gsm_s, gsm_m, gsm_o, gsm_pk,
         gma_s, gma_m, gma_o, gma_pk) in per_freq:
        print(f"{freq:>10s}  {n:>6d}  "
              f"{ps_s:>10.3f} {ps_m:>6.3f} {ps_o:>6.4f}  "
              f"{gsm_o:>9.4f}  {gma_o:>9.4f}")

    # Series-weighted totals (M4 official aggregation)
    total_n = sum(SERIES_COUNT[f] for f in FREQS)
    total_n2_sm = sum(all_n2_sm) / total_n
    total_n2_ma = sum(all_n2_ma) / total_n
    print(f"\n--- Series-weighted totals (n={total_n}) ---")
    print(f"  Naive 2 total sMAPE = {total_n2_sm:.3f}, MASE = {total_n2_ma:.4f}")
    for label, sm_arr, ma_arr in [
        ("(1) per-series val-tune", all_ps_sm, all_ps_ma),
        ("(2) global by val-sMAPE", all_g_sm_sm, all_g_sm_ma),
        ("(2') global by val-MASE", all_g_ma_sm, all_g_ma_ma),
    ]:
        sm = np.mean(sm_arr); ma = np.mean(ma_arr)
        owa = 0.5 * (sm/total_n2_sm + ma/total_n2_ma)
        print(f"  {label:>26s}: sMAPE={sm:.3f}  MASE={ma:.4f}  "
              f"OWA = {owa:.4f}")


if __name__ == "__main__":
    main()
