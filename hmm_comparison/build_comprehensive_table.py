"""Aggregate per-regime, per-N, per-model MSE and excess-perplexity
numbers into a single comprehensive table for the HMM_EXPERIMENTS_SUMMARY.

Sources:
  perplexity_sweep_results.csv         — main 6-regime x 3-N x 9-model
                                          sweep with both metrics
  absorb_compare_regimes_results.csv   — GDC diffuse vs absorb at
                                          per-regime tuned configs

Output: prints a markdown-formatted table to stdout for direct
inclusion in HMM_EXPERIMENTS_SUMMARY.md.
"""
from __future__ import annotations
import os, csv
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PPL_CSV = os.path.join(HERE, 'perplexity_sweep_results.csv')
ABS_CSV = os.path.join(HERE, 'absorb_compare_regimes_results.csv')

REGIMES = ['dense_small', 'dense_large', 'det_small', 'det_large',
           'sparse_small', 'sparse_large']
N_VALUES = [25, 100, 400]
HORIZON = 1   # focus on h=1


def load_ppl():
    rows = []
    with open(PPL_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(dict(regime=r['regime'], seed=int(r['seed']),
                             N=int(r['N_train']), model=r['model'],
                             h=int(r['horizon']),
                             mse=float(r['mse']),
                             ex_pp=float(r['excess_perplexity']),
                             floor_pp=float(r['entropy_floor_perplexity'])))
    return rows


def load_absorb():
    rows = []
    with open(ABS_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(dict(regime=r['regime'], seed=int(r['seed']),
                             N=int(r['N_train']), mode=r['mode'],
                             h=int(r['horizon']),
                             mse=float(r['mse']),
                             ex_pp=float(r['excess_perplexity']),
                             floor_pp=float(r['entropy_floor_perplexity']),
                             alpha=float(r['alpha']), theta=float(r['theta']),
                             beta=float(r['beta'])))
    return rows


def model_class(m):
    if m.startswith('gdc-'): return 'GDC*'
    if m.startswith('chmm-'): return 'CHMM*'
    return m


def main():
    ppl = load_ppl()
    absorb = load_absorb()

    # Floor perplexity per (regime, N) -- nearly identical across seeds
    floor = {}
    for r in ppl:
        if r['h'] != HORIZON: continue
        key = (r['regime'], r['N'])
        floor.setdefault(key, []).append(r['floor_pp'])
    floor = {k: float(np.mean(v)) for k, v in floor.items()}

    print("# Comprehensive HMM forecasting results @ h=1")
    print()
    print("Best of each model class per (regime, N), at horizon h=1, "
          "averaged over 3 seeds.  All models are run on the same "
          "(nS, nA, seed)-matched HMMs as the perplexity sweep "
          "(see [run_perplexity_sweep.py](run_perplexity_sweep.py)).")
    print()

    for regime in REGIMES:
        print(f"## {regime}")
        print()
        print("| N | model | MSE | excess PP | abs PP |")
        print("|---|---|---:|---:|---:|")
        for N in N_VALUES:
            print(f"| **{N}** | _entropy floor_ | -- | 1.000 "
                  f"| {floor.get((regime, N), float('nan')):.3f} |")
            # Best of each model class from the perplexity sweep
            for cls in ['GDC*', 'CHMM*', 'alergia-eps0.05']:
                by_sub = defaultdict(lambda: defaultdict(list))
                for r in ppl:
                    if (r['regime'] == regime and r['N'] == N and r['h'] == HORIZON
                            and model_class(r['model']) == cls):
                        by_sub[r['model']]['mse'].append(r['mse'])
                        by_sub[r['model']]['ex_pp'].append(r['ex_pp'])
                if not by_sub:
                    continue
                # Best by MSE
                best_by_mse = min(by_sub.items(), key=lambda kv: np.mean(kv[1]['mse']))
                m_name, m_data = best_by_mse
                mse = np.mean(m_data['mse'])
                expp = np.mean(m_data['ex_pp'])
                abspp = expp * floor.get((regime, N), 1.0)
                # Strip noise from name
                pretty_name = m_name
                if pretty_name.startswith('gdc-'):
                    pretty_name = pretty_name.replace('gdc-', 'GDC ')
                elif pretty_name.startswith('chmm-'):
                    pretty_name = pretty_name.replace('chmm-K', 'CHMM K=')
                elif pretty_name == 'alergia-eps0.05':
                    pretty_name = 'ALERGIA eps=0.05'
                print(f"| {N} | {pretty_name} | {mse:.5f} | {expp:.4f} "
                      f"| {abspp:.3f} |")
            # GDC tuned + diffuse and absorb (from absorb sweep)
            for mode in ['diffuse', 'absorb']:
                rows = [r for r in absorb
                        if r['regime'] == regime and r['N'] == N
                        and r['h'] == HORIZON and r['mode'] == mode]
                if not rows:
                    continue
                mse = float(np.mean([r['mse'] for r in rows]))
                expp = float(np.mean([r['ex_pp'] for r in rows]))
                abspp = expp * floor.get((regime, N), 1.0)
                cfg = (rows[0]['alpha'], rows[0]['theta'], rows[0]['beta'])
                print(f"| {N} | GDC tuned ({mode}) "
                      f"alpha={cfg[0]} theta={cfg[1]} beta={cfg[2]} "
                      f"| {mse:.5f} | {expp:.4f} | {abspp:.3f} |")
        print()


if __name__ == "__main__":
    main()
