"""Focused PAutomaC sweep: GDC + Parrot via torch adapters only.

Mirrors run_eval.py's loop but uses TorchGDCModel and TorchParrotModel
(GPU-batched). All 48 problems × 2 configs each × 2 methods = 4
configs per problem.

Sequential across problems (the GPU is the bottleneck, not the
problem-level parallelism). Total expected runtime: ~3-5 minutes.
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_loader import load_problem, summary
from scoring import pautomac_score
from torch_adapters import TorchGDCModel, TorchParrotModel

OUT = os.path.join(HERE, 'results', 'pautomac_torch_sweep.csv')
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def get_torch_models():
    return [
        TorchGDCModel(alpha=0.95, theta=0.05, beta=0.0,
                      terminal_behavior='diffuse',
                      initial_dist='sequence_starts'),
        TorchGDCModel(alpha=0.50, theta=0.005, beta=0.0,
                      terminal_behavior='diffuse',
                      initial_dist='sequence_starts'),
        TorchParrotModel(L=2, K=5,  alpha_prior=1.0),
        TorchParrotModel(L=4, K=25, alpha_prior=0.1),
    ]


def main():
    print(f"=== PAutomaC torch-only sweep ===")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"  Models: {[m.name for m in get_torch_models()]}\n",
          flush=True)

    rows = []
    t_total = time.time()
    for pi in range(1, 49):
        try:
            problem = load_problem(pi)
        except FileNotFoundError:
            print(f"  problem {pi}: missing"); continue
        s = summary(problem)
        for model in get_torch_models():
            try:
                t0 = time.time()
                model.fit(problem['train'], problem['alphabet_size'])
                fit_t = time.time() - t0
                t0 = time.time()
                log_probs = model.score_test_set(problem['test'])
                eval_t = time.time() - t0
                r = pautomac_score(log_probs, problem['true_probs'])
                rows.append(dict(problem=pi,
                                  alphabet_size=s['alphabet_size'],
                                  model=model.name,
                                  **r,
                                  fit_s=fit_t, eval_s=eval_t))
            except Exception as e:
                rows.append(dict(problem=pi,
                                  alphabet_size=s['alphabet_size'],
                                  model=model.name,
                                  score=float('nan'),
                                  entropy_floor=float('nan'),
                                  gap=float('nan'),
                                  lift=float('nan'),
                                  fit_s=0.0, eval_s=0.0,
                                  error=str(e)))
        # print one summary row per problem
        cells = [r for r in rows if r['problem'] == pi]
        if cells:
            best = min(cells, key=lambda r: r.get('gap', float('inf'))
                       if r.get('gap') == r.get('gap') else float('inf'))
            print(f"  problem {pi:>2d} (A={s['alphabet_size']:>2d}, "
                  f"floor={cells[0]['entropy_floor']:>7.2f}): "
                  f"best={best['model']:<22s} gap={best['gap']:>7.3f}  "
                  f"[{time.time()-t_total:.0f}s]", flush=True)

    fields = ['problem', 'alphabet_size', 'model', 'score',
              'entropy_floor', 'gap', 'lift', 'fit_s', 'eval_s']
    if any('error' in r for r in rows):
        fields.append('error')
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {OUT} ({len(rows)} rows, "
          f"{time.time()-t_total:.0f}s total)")


if __name__ == "__main__":
    main()
