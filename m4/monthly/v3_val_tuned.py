"""V3: per-series val-tuned ensemble for M4 monthly. h=18 may be long
enough to make val-tune work this time."""
from __future__ import annotations
import os
import sys
import csv
import time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M4_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(M4_ROOT)
sys.path.insert(0, HERE); sys.path.insert(0, M4_ROOT); sys.path.insert(0, ROOT)
import data_loader as dl
from v0_baselines import smape, naive_last, drift, naive_seasonal, HORIZON
from v1_nn_diff import nn_diff_forecast
from v2_gdc_diff import gdc_diff_forecast


CONFIGS = [
    ('naive_last',                'naive', dict()),
    ('drift',                     'drift', dict()),
    ('naive_seasonal12',          'seas',  dict()),
    ('nn_L6_s0.50',               'nn',    dict(window_len=6,  sigma_frac=0.50)),
    ('nn_L12_s0.50',              'nn',    dict(window_len=12, sigma_frac=0.50)),
    ('gdc_L12_s0.25_a0.95',       'gdc',   dict(window_len=12, sigma_frac=0.25, alpha=0.95, theta=0.0)),
    ('gdc_L18_s0.25_a0.95',       'gdc',   dict(window_len=18, sigma_frac=0.25, alpha=0.95, theta=0.0)),
    ('gdc_L6_s0.25_a0.95',        'gdc',   dict(window_len=6,  sigma_frac=0.25, alpha=0.95, theta=0.0)),
]


def predict(kind, cfg, train, h=HORIZON):
    if kind == 'naive': return naive_last(train, h)
    if kind == 'drift': return drift(train, h)
    if kind == 'seas':  return naive_seasonal(train, h)
    if kind == 'nn':    return nn_diff_forecast(train, h=h, **cfg)
    if kind == 'gdc':   return gdc_diff_forecast(train, h=h, **cfg)


def run_series(args):
    sid, train, test = args
    h = HORIZON
    if len(train) < 3 * h + 24:
        pred = naive_last(train, h)
        return [dict(sid=sid, model='val_tuned', best_cfg='naive_last',
                     val_smape=float('nan'),
                     test_smape=smape(test, pred),
                     naive_smape=smape(test, naive_last(train, h)))]
    tr = train[:-h]; val = train[-h:]
    val_scores = {}
    for name, kind, cfg in CONFIGS:
        try:
            p = predict(kind, cfg, tr, h=h)
            val_scores[name] = smape(val, p)
        except Exception:
            val_scores[name] = float('inf')
    best_name = min(val_scores, key=val_scores.get)
    bk, bc = next((kind, cfg) for n, kind, cfg in CONFIGS if n == best_name)
    pred = predict(bk, bc, train, h=h)
    rows = [dict(sid=sid, model='val_tuned', best_cfg=best_name,
                 val_smape=val_scores[best_name],
                 test_smape=smape(test, pred),
                 naive_smape=smape(test, naive_last(train, h)))]
    for name, kind, cfg in CONFIGS:
        p = predict(kind, cfg, train, h=h)
        rows.append(dict(sid=sid, model=name, best_cfg=name,
                         val_smape=val_scores[name],
                         test_smape=smape(test, p),
                         naive_smape=smape(test, naive_last(train, h))))
    return rows


def main():
    train = dl.load_train("Monthly"); test = dl.load_test("Monthly")
    ids = sorted(train.keys(), key=lambda s: int(s[1:]))
    tasks = [(sid, train[sid], test[sid]) for sid in ids]
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"Val-tuned on {len(tasks)} series, {len(CONFIGS)} configs, "
          f"{n_workers} workers", flush=True)
    all_rows = []
    t0 = time.time(); done = 0
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_series, tasks, chunksize=16):
            all_rows.extend(r); done += 1
            if done % 5000 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} [{time.time()-t0:.0f}s]",
                      flush=True)

    out_csv = os.path.join(HERE, "v3_val_tuned_results.csv")
    fields = list(all_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote {out_csv}")

    print(f"\n=== Per-model test sMAPE on M4 monthly ({len(ids)} series) ===")
    print(f"{'model':>30s}  {'mean':>7s}  {'median':>7s}")
    from collections import defaultdict
    by_m = defaultdict(list)
    for r in all_rows:
        by_m[r['model']].append(r['test_smape'])
    order = ['val_tuned'] + [n for n, _, _ in CONFIGS]
    for name in order:
        v = np.array(by_m[name])
        if len(v) == 0:
            continue
        print(f"{name:>30s}  {v.mean():>6.2f}%  {np.median(v):>6.2f}%")

    from collections import Counter
    pick_counts = Counter(r['best_cfg'] for r in all_rows
                          if r['model'] == 'val_tuned')
    print(f"\n=== val-tune picks ===")
    for name, c in pick_counts.most_common():
        print(f"  {name:>30s}: {c}")


if __name__ == "__main__":
    main()
