"""Summarise the full PAutomaC sweep CSV.

Reports per-problem winner, geometric-mean gap-to-floor across all 48
problems for each model, and head-to-head GDC vs CHMM win counts.
"""
from __future__ import annotations
import os, csv
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'results', 'full_sweep.csv')


def main():
    rows = []
    with open(CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(
                model=r['model'],
                problem=int(r['problem']),
                A=int(r['alphabet_size']),
                score=float(r['score']),
                floor=float(r['entropy_floor']),
                gap=float(r['gap']),
                lift=float(r['lift']),
                train_s=float(r['train_s']),
                eval_s=float(r['eval_s']),
            ))

    by_problem = defaultdict(dict)
    for r in rows:
        by_problem[r['problem']][r['model']] = r

    models = sorted({r['model'] for r in rows})
    print(f"Models: {models}")
    print(f"Problems: {sorted(by_problem.keys())}")

    # Per-model summary across all 48 problems
    print(f"\n=== Per-model gap statistics across 48 problems ===")
    print(f"{'model':>30s}  {'mean_gap':>10s}  {'median_gap':>11s}  "
          f"{'gmean_ratio':>11s}  {'n_wins':>7s}  {'n_perfect':>9s}  "
          f"{'mean_train_s':>13s}  {'mean_eval_s':>12s}")
    summary = {}
    for m in models:
        gaps, ratios, train_ts, eval_ts = [], [], [], []
        for p in sorted(by_problem):
            r = by_problem[p].get(m)
            if r is None: continue
            gaps.append(r['gap'])
            # ratio: gap / (uniform_gap), where uniform_gap = (uniform_score) - floor
            uni_score = by_problem[p]['uniform']['score']
            floor = r['floor']
            uniform_gap = uni_score - floor
            ratios.append(max(r['gap'], 1e-9) / max(uniform_gap, 1e-9))
            train_ts.append(r['train_s']); eval_ts.append(r['eval_s'])
        gaps = np.array(gaps); ratios = np.array(ratios)
        n_perfect = int(np.sum(gaps < 0.01))
        log_ratio = np.log(np.maximum(ratios, 1e-12))
        gmean_ratio = float(np.exp(log_ratio.mean()))
        # Wins = how many problems this model has the smallest gap on
        n_wins = 0
        for p in sorted(by_problem):
            best = min(by_problem[p].items(), key=lambda kv: kv[1]['gap'])
            if best[0] == m:
                n_wins += 1
        summary[m] = (float(gaps.mean()), float(np.median(gaps)),
                      gmean_ratio, n_wins, n_perfect,
                      float(np.mean(train_ts)), float(np.mean(eval_ts)))
        print(f"{m:>30s}  {gaps.mean():>10.3f}  {np.median(gaps):>11.3f}  "
              f"{gmean_ratio:>11.4f}  {n_wins:>7d}  {n_perfect:>9d}  "
              f"{np.mean(train_ts):>13.2f}  {np.mean(eval_ts):>12.2f}")

    # GDC-best vs CHMM-best head-to-head
    gdc_models = [m for m in models if 'gdc' in m]
    chmm_models = [m for m in models if 'chmm' in m]
    gdc_wins = chmm_wins = ties = 0
    gdc_better_by = []
    for p in sorted(by_problem):
        gdc_min = min(by_problem[p][m]['gap'] for m in gdc_models)
        chmm_min = min(by_problem[p][m]['gap'] for m in chmm_models)
        if gdc_min < chmm_min - 1e-6: gdc_wins += 1
        elif chmm_min < gdc_min - 1e-6: chmm_wins += 1
        else: ties += 1
        gdc_better_by.append(chmm_min - gdc_min)
    print(f"\n=== GDC-best vs CHMM-best head-to-head ===")
    print(f"GDC wins:  {gdc_wins} / 48")
    print(f"CHMM wins: {chmm_wins} / 48")
    print(f"ties:      {ties} / 48")
    print(f"mean (chmm_gap - gdc_gap):   {np.mean(gdc_better_by):+.3f}")
    print(f"median (chmm_gap - gdc_gap): {np.median(gdc_better_by):+.3f}")

    # Per-problem table (summary form)
    print(f"\n=== Per-problem gaps (lower is better) ===")
    cols = ['uniform', 'unigram', 'bigram', 'chmm-K2', 'chmm-K4',
            'chmm-K8', 'fastgdc-a0.95-t0.05-1step',
            'fastgdc-a0.5-t0.005-1step']
    short = {'uniform':'uni', 'unigram':'1gr', 'bigram':'2gr',
             'chmm-K2':'C-K2', 'chmm-K4':'C-K4', 'chmm-K8':'C-K8',
             'fastgdc-a0.95-t0.05-1step':'GDC-a.95',
             'fastgdc-a0.5-t0.005-1step':'GDC-a.50'}
    print(f"{'p':>3s}  {'A':>3s}  {'floor':>9s}  " +
          "  ".join(f"{short[c]:>10s}" for c in cols) +
          "  best")
    for p in sorted(by_problem):
        floor = by_problem[p][cols[0]]['floor']
        gaps = {m: by_problem[p][m]['gap'] for m in cols}
        best_m = min(gaps.items(), key=lambda kv: kv[1])[0]
        cells = "  ".join(f"{gaps[m]:>10.2f}" for m in cols)
        A = by_problem[p][cols[0]]['A']
        print(f"{p:>3d}  {A:>3d}  {floor:>9.2f}  {cells}  {short[best_m]}")


if __name__ == "__main__":
    main()
