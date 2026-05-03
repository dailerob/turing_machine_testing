"""Combine main_sweep_results.csv (GDC, OOM) with
chmm_alergia_sweep_results.csv (CHMM, ALERGIA) and report aggregate
MSE per model per horizon, plus a per-(nS, nA) winner table."""
from __future__ import annotations
import os, csv
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, 'main_sweep_results.csv')
NEW = os.path.join(HERE, 'chmm_alergia_sweep_results.csv')


def load():
    rows = []   # (nS, nA, seed, horizon, model, mse)
    with open(MAIN) as f:
        for r in csv.DictReader(f):
            base = (int(r['nS']), int(r['nA']), int(r['seed']),
                    int(r['horizon']))
            for col, model in [('oom_clip_mse', 'oom-clip'),
                               ('oom_softmax_mse', 'oom-softmax'),
                               ('gdc_mse', 'gdc-baseline'),
                               ('uni_mse', 'uniform'),
                               ('sta_mse', 'stationary')]:
                rows.append(base + (model, float(r[col])))
    with open(NEW) as f:
        for r in csv.DictReader(f):
            rows.append((int(r['nS']), int(r['nA']), int(r['seed']),
                         int(r['horizon']), r['model'], float(r['mse'])))
    return rows


def main():
    rows = load()
    horizons = sorted({r[3] for r in rows})
    models = sorted({r[4] for r in rows})

    # Aggregate: mean MSE per (model, horizon) over the full grid
    by_mh = defaultdict(list)
    for nS, nA, sd, h, m, mse in rows:
        by_mh[(m, h)].append(mse)
    print(f"=== Mean MSE per (model, horizon) ===")
    print(f"{'model':>20s}  " + "  ".join(f'h={h:>2d}' for h in horizons))
    for m in models:
        means = [np.mean(by_mh.get((m, h), [np.nan])) for h in horizons]
        print(f"{m:>20s}  " + "  ".join(f'{x:>7.4f}' for x in means))

    # gmean ratio against uniform
    print(f"\n=== gmean(MSE / uniform_MSE) ===")
    print(f"{'model':>20s}  " + "  ".join(f'h={h:>2d}' for h in horizons))
    for m in models:
        gms = []
        for h in horizons:
            ratios = []
            by_seed = defaultdict(dict)
            for nS, nA, sd, hh, mm, mse in rows:
                if hh != h: continue
                by_seed[(nS, nA, sd)][mm] = mse
            for k, mm_dict in by_seed.items():
                if m in mm_dict and 'uniform' in mm_dict and mm_dict['uniform'] > 0:
                    ratios.append(mm_dict[m] / mm_dict['uniform'])
            if ratios:
                lr = np.log(np.maximum(ratios, 1e-12))
                gms.append(float(np.exp(lr.mean())))
            else:
                gms.append(float('nan'))
        print(f"{m:>20s}  " + "  ".join(f'{x:>7.4f}' for x in gms))

    # Wins (best model per (nS, nA, seed, horizon))
    win_count = defaultdict(int)
    by_key = defaultdict(dict)
    for nS, nA, sd, h, m, mse in rows:
        by_key[(nS, nA, sd, h)][m] = mse
    excluded = {'uniform', 'stationary'}
    for k, mm_dict in by_key.items():
        candidates = {m: v for m, v in mm_dict.items() if m not in excluded}
        if not candidates:
            continue
        best_m = min(candidates.items(), key=lambda kv: kv[1])[0]
        win_count[best_m] += 1
    total = sum(win_count.values())
    print(f"\n=== Win counts (smallest MSE among non-baseline models) ===")
    print(f"  total cells: {total}")
    for m in sorted(win_count, key=lambda k: -win_count[k]):
        print(f"  {m:>20s}  {win_count[m]:>4d}  "
              f"({100*win_count[m]/total:>4.1f}%)")

    # Win counts at h=1 only
    print(f"\n=== Win counts at h=1 only ===")
    win1 = defaultdict(int)
    for k, mm_dict in by_key.items():
        if k[3] != 1: continue
        candidates = {m: v for m, v in mm_dict.items() if m not in excluded}
        if not candidates: continue
        best_m = min(candidates.items(), key=lambda kv: kv[1])[0]
        win1[best_m] += 1
    total1 = sum(win1.values())
    print(f"  total cells (h=1): {total1}")
    for m in sorted(win1, key=lambda k: -win1[k]):
        print(f"  {m:>20s}  {win1[m]:>4d}  "
              f"({100*win1[m]/total1:>4.1f}%)")


if __name__ == "__main__":
    main()
