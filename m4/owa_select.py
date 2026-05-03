"""Compute GDC pick protocols using val OWA (joint sMAPE+MASE).

For each series:
  val_owa(config) = 0.5 * (val_smape(config)/val_n2_smape +
                           val_mase(config)/val_n2_mase)
where val_n2_* is the series's own Naive 2 sMAPE/MASE on the held-out
last-h validation slice.

Two protocols:
  (1') per-series: pick argmin_config val_owa per series, score on test
  (2'') global: aggregate mean val OWA per config across the whole
        dataset, pick the global winner, apply to every series, score on test.

Outputs both per-frequency and series-weighted-total OWA tables.
"""
from __future__ import annotations
import os, sys, csv
import numpy as np
import multiprocessing as mp
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_loader as dl
from naive2 import naive2_forecast, smape as n2_smape, mase as n2_mase, M4_PERIOD

FREQS = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
SERIES_COUNT = {"Yearly": 23000, "Quarterly": 24000, "Monthly": 48000,
                "Weekly": 359, "Daily": 4227, "Hourly": 414}
NAIVE2_TOTAL = {  # per-frequency totals from naive2.py / supplementary
    "Yearly": (16.342, 3.974), "Quarterly": (11.012, 1.371),
    "Monthly": (14.427, 1.063), "Weekly": (9.161, 2.777),
    "Daily": (3.045, 3.278),  "Hourly": (18.383, 2.395),
}


def compute_val_naive2_for_series(args):
    """For one series, run Naive 2 on train[:-h] and score on train[-h:]."""
    sid, train, h, m = args
    if len(train) < 2 * h + 4 or len(train) < 2:
        return sid, float('nan'), float('nan')
    tr = train[:-h]; val = train[-h:]
    f = naive2_forecast(tr, h, m)
    sm = n2_smape(val, f)
    ma = n2_mase(tr, val, f, m)
    return sid, sm, ma


def load_clean_eval_csv(freq):
    """Returns dict sid -> {model: (val_smape, test_smape, val_mase, test_mase)}."""
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


def process_freq(freq):
    print(f"=== {freq} ===", flush=True)
    h = dl.horizon(freq); m = M4_PERIOD[freq]
    train = dl.load_train(freq); test = dl.load_test(freq)
    ids = sorted(train.keys(), key=lambda s: int(s[1:]))

    # 1) compute val Naive2 (sMAPE/MASE) per series
    print(f"  Computing val Naive 2 for {len(ids)} series...", flush=True)
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    val_n2 = {}
    with mp.Pool(n_workers) as pool:
        tasks = [(sid, train[sid], h, m) for sid in ids]
        t0 = time.time(); done = 0
        every = max(1, len(tasks) // 20)
        for sid, sm, ma in pool.imap_unordered(
                compute_val_naive2_for_series, tasks, chunksize=64):
            val_n2[sid] = (sm, ma); done += 1
            if done % every == 0 or done == len(tasks):
                print(f"    {done}/{len(tasks)} [{time.time()-t0:.0f}s]",
                      flush=True)

    # 2) load existing GDC val/test sMAPE/MASE per (sid, model)
    by_sid = load_clean_eval_csv(freq)

    # 3) compute val OWA per (sid, model). Exclude naive_last from picks —
    # it's a fallback for short series, not a GDC method. (GDC functions
    # already fall back to naive internally when there's not enough data.)
    val_owa = defaultdict(dict)
    for sid in ids:
        n2_sm, n2_ma = val_n2.get(sid, (float('nan'), float('nan')))
        for model, (vs, ts, vm, tm) in by_sid[sid].items():
            if model == 'naive_last':
                continue  # excluded from selection
            if not (np.isnan(vs) or np.isnan(vm) or np.isnan(n2_sm) or np.isnan(n2_ma)):
                val_owa[sid][model] = 0.5 * (vs / max(n2_sm, 1e-9) +
                                              vm / max(n2_ma, 1e-9))

    # 4a) Per-series pick by val OWA
    test_sm_ps = []; test_ma_ps = []; pick_counts = defaultdict(int)
    for sid in ids:
        d = val_owa.get(sid, {})
        if not d:
            # fallback to naive_last test scores
            ts, tm = by_sid[sid].get('naive_last', (None, np.nan, None, np.nan))[1::2]
            best = 'naive_last (fallback)'
        else:
            best = min(d, key=d.get)
            _, ts, _, tm = by_sid[sid][best]
        if not np.isnan(ts): test_sm_ps.append(ts)
        if not np.isnan(tm): test_ma_ps.append(tm)
        pick_counts[best] += 1
    ps_sm = np.mean(test_sm_ps); ps_ma = np.mean(test_ma_ps)
    n2tot_sm, n2tot_ma = NAIVE2_TOTAL[freq]
    ps_owa = 0.5 * (ps_sm / n2tot_sm + ps_ma / n2tot_ma)

    # 4b) Global pick by **dataset-level** val OWA:
    # mean val OWA per config = 0.5 * (mean val sMAPE / mean val Naive 2 sMAPE
    #                                + mean val MASE / mean val Naive 2 MASE)
    val_n2_sms = [v[0] for v in val_n2.values() if not np.isnan(v[0])]
    val_n2_mas = [v[1] for v in val_n2.values() if not np.isnan(v[1])
                  and not np.isinf(v[1])]
    mean_val_n2_sm = np.mean(val_n2_sms); mean_val_n2_ma = np.mean(val_n2_mas)
    by_model_smapes = defaultdict(list); by_model_mases = defaultdict(list)
    for sid in ids:
        for model, (vs, ts, vm, tm) in by_sid[sid].items():
            if model == 'naive_last':
                continue  # excluded — not a GDC method
            if not np.isnan(vs): by_model_smapes[model].append(vs)
            if not np.isnan(vm) and not np.isinf(vm):
                by_model_mases[model].append(vm)
    means = {}
    for model in set(by_model_smapes) | set(by_model_mases):
        if by_model_smapes[model] and by_model_mases[model]:
            means[model] = 0.5 * (np.mean(by_model_smapes[model]) / mean_val_n2_sm
                                 + np.mean(by_model_mases[model]) / mean_val_n2_ma)
        else:
            means[model] = np.inf
    if not means:
        gpick, g_sm, g_ma, g_owa = 'NONE', np.nan, np.nan, np.nan
    else:
        gpick = min(means, key=means.get)
        # apply gpick to every series with valid test values
        gsms, gmas = [], []
        for sid in ids:
            triple = by_sid[sid].get(gpick)
            if triple:
                _, ts, _, tm = triple
                if not np.isnan(ts): gsms.append(ts)
                if not np.isnan(tm): gmas.append(tm)
        g_sm = np.mean(gsms); g_ma = np.mean(gmas)
        g_owa = 0.5 * (g_sm / n2tot_sm + g_ma / n2tot_ma)

    print(f"  (1') per-series val-OWA: sMAPE={ps_sm:.3f} MASE={ps_ma:.4f} OWA={ps_owa:.4f}", flush=True)
    print(f"  (2'') global val-OWA pick = {gpick}", flush=True)
    print(f"        test:  sMAPE={g_sm:.3f} MASE={g_ma:.4f} OWA={g_owa:.4f}", flush=True)
    print(f"  Picks (top 6): " + ", ".join(
        f"{n}:{c}" for n, c in sorted(pick_counts.items(), key=lambda x: -x[1])[:6]),
        flush=True)

    return dict(freq=freq, n=len(ids), n2tot_sm=n2tot_sm, n2tot_ma=n2tot_ma,
                ps_sm=ps_sm, ps_ma=ps_ma, ps_owa=ps_owa,
                g_pick=gpick, g_sm=g_sm, g_ma=g_ma, g_owa=g_owa,
                ps_test_sm=test_sm_ps, ps_test_ma=test_ma_ps,
                g_test_sm=gsms if 'gsms' in locals() else [],
                g_test_ma=gmas if 'gmas' in locals() else [])


def main():
    results = []
    for f in FREQS:
        results.append(process_freq(f))

    print("\n\n=== Per-frequency OWA (val-OWA picks) ===")
    h1 = "(1ps) sMAPE"; h2 = "(2gl) sMAPE"
    print(f"{'freq':>10s}  {'n':>6s}  "
          f"{h1:>11s}  {'MASE':>6s}  {'OWA':>6s}  "
          f"{h2:>11s}  {'MASE':>6s}  {'OWA':>6s}  {'pick':>30s}")
    for r in results:
        print(f"{r['freq']:>10s}  {r['n']:>6d}  "
              f"{r['ps_sm']:>10.3f}  {r['ps_ma']:>5.3f}  {r['ps_owa']:>5.3f}  "
              f"{r['g_sm']:>10.3f}  {r['g_ma']:>5.3f}  {r['g_owa']:>5.3f}  "
              f"{r['g_pick']:>30s}")

    # Series-weighted totals
    total_n = sum(SERIES_COUNT[f] for f in FREQS)
    total_n2_sm = sum(NAIVE2_TOTAL[f][0] * SERIES_COUNT[f] for f in FREQS) / total_n
    total_n2_ma = sum(NAIVE2_TOTAL[f][1] * SERIES_COUNT[f] for f in FREQS) / total_n

    all_ps_sm, all_ps_ma = [], []
    all_g_sm,  all_g_ma  = [], []
    for r in results:
        all_ps_sm.extend(r['ps_test_sm']); all_ps_ma.extend(r['ps_test_ma'])
        all_g_sm.extend(r['g_test_sm']);   all_g_ma.extend(r['g_test_ma'])

    print(f"\n--- Series-weighted totals (n={total_n}) ---")
    print(f"  Naive 2 totals: sMAPE={total_n2_sm:.3f}, MASE={total_n2_ma:.4f}")
    for label, sms, mas in [
        ("(1') per-series val-OWA", all_ps_sm, all_ps_ma),
        ("(2'') global val-OWA",    all_g_sm,  all_g_ma),
    ]:
        sm = np.mean(sms); ma = np.mean(mas)
        owa = 0.5 * (sm / total_n2_sm + ma / total_n2_ma)
        print(f"  {label:>26s}: sMAPE={sm:.3f}  MASE={ma:.4f}  OWA={owa:.4f}")


if __name__ == "__main__":
    main()
