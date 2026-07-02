"""Series-weighted OWA totals for parrot, mirroring owa_total.py exactly.

Reads m4/<freq>/parrot_eval_results.csv produced by parrot_eval.py.
"""
from __future__ import annotations
import csv, os
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FREQS = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
SERIES_COUNT = {"Yearly": 23000, "Quarterly": 24000, "Monthly": 48000,
                "Weekly": 359, "Daily": 4227, "Hourly": 414}
NAIVE2 = {
    "Yearly":    (16.342, 3.974), "Quarterly": (11.012, 1.371),
    "Monthly":   (14.427, 1.063), "Weekly":    (9.161,  2.777),
    "Daily":     (3.045,  3.278), "Hourly":    (18.383, 2.395),
}


def load(freq):
    csv_path = os.path.join(HERE, freq.lower(), "parrot_eval_results.csv")
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
    all_n2_sm, all_n2_ma = [], []
    all_ps_sm, all_ps_ma = [], []
    all_g_sm_sm, all_g_sm_ma = [], []
    all_g_ma_sm, all_g_ma_ma = [], []
    per_freq = []
    for freq in FREQS:
        by_sid = load(freq)
        sids = list(by_sid.keys())
        ps_sm, ps_ma = [], []
        models = set()
        for d in by_sid.values(): models.update(d.keys())
        global_val_sm = {m: [] for m in models}
        global_val_ma = {m: [] for m in models}
        for sid, d in by_sid.items():
            for m, (vs, ts, vm, tm) in d.items():
                if m == 'naive_last': continue
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
            elig = [(name, vs, ts, vm, tm) for name, (vs, ts, vm, tm) in d.items()
                    if not np.isnan(vs) and name != 'naive_last']
            if elig:
                _, _, ts, _, tm = min(elig, key=lambda x: x[1])
            else:
                _, ts, _, tm = d.get('naive_last', (np.nan, np.nan, np.nan, np.nan))
            if not np.isnan(ts): ps_sm.append(ts)
            if not np.isnan(tm): ps_ma.append(tm)
            if gsm_pick in d:
                _, ts, _, tm = d[gsm_pick]
                if not np.isnan(ts): g_sm_sm.append(ts)
                if not np.isnan(tm): g_sm_ma.append(tm)
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
        per_freq.append((freq, len(sids), ps_sm.mean(), ps_ma.mean(), owa_ps,
                         owa_gsm, gsm_pick, owa_gma, gma_pick))
        all_n2_sm.append(n2_sm * len(sids)); all_n2_ma.append(n2_ma * len(sids))
        all_ps_sm.extend(ps_sm.tolist()); all_ps_ma.extend(ps_ma.tolist())
        all_g_sm_sm.extend(g_sm_sm.tolist()); all_g_sm_ma.extend(g_sm_ma.tolist())
        all_g_ma_sm.extend(g_ma_sm.tolist()); all_g_ma_ma.extend(g_ma_ma.tolist())

    print(f"{'freq':>10s}  {'n':>6s}  "
          f"{'(1) sMAPE':>10s} {'MASE':>6s} {'OWA':>6s}  "
          f"{'(2sM) OWA':>9s}  {'(2MA) OWA':>9s}  picks")
    for (freq, n, ps_s, ps_m, ps_o, gsm_o, gsm_pk, gma_o, gma_pk) in per_freq:
        print(f"{freq:>10s}  {n:>6d}  "
              f"{ps_s:>10.3f} {ps_m:>6.3f} {ps_o:>6.4f}  "
              f"{gsm_o:>9.4f}  {gma_o:>9.4f}  {gsm_pk}/{gma_pk}")

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
        print(f"  {label:>26s}: sMAPE={sm:.3f}  MASE={ma:.4f}  OWA={owa:.4f}")


if __name__ == "__main__":
    main()
