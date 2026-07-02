"""Parallel PAutomaC sweep across all 48 problems.

Mirrors run_eval.py's pipeline but parallelizes across problems with
mp.Pool (fork). Each worker loads a problem, instantiates the full
model list, fits and scores each model, and returns the per-problem
rows.

Usage:
    python pautomac/run_eval_parallel.py --workers 8 --problems all
"""
from __future__ import annotations
import os, sys, argparse, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_loader import load_problem, summary
from scoring import pautomac_score
from run_eval import get_models, parse_problems


def evaluate_problem(pi):
    """Run all models on a single PAutomaC problem. Returns list of
    per-model row dicts (or empty list if problem is missing)."""
    try:
        problem = load_problem(pi)
    except FileNotFoundError:
        return []
    s = summary(problem)
    rows = []
    for model in get_models():
        try:
            t0 = time.time()
            model.fit(problem['train'], problem['alphabet_size'])
            train_t = time.time() - t0
            t0 = time.time()
            if hasattr(model, 'score_test_set'):
                log_probs = model.score_test_set(problem['test'])
            else:
                log_probs = np.empty(len(problem['test']),
                                      dtype=np.float64)
                for i, seq in enumerate(problem['test']):
                    log_probs[i] = model.log_prob(seq)
            eval_t = time.time() - t0
            r = pautomac_score(log_probs, problem['true_probs'])
            rows.append(dict(problem=pi,
                             alphabet_size=s['alphabet_size'],
                             model=model.name,
                             **r,
                             train_s=train_t, eval_s=eval_t))
        except Exception as e:
            rows.append(dict(problem=pi,
                             alphabet_size=s['alphabet_size'],
                             model=model.name,
                             score=float('nan'),
                             entropy_floor=float('nan'),
                             gap=float('nan'),
                             lift=float('nan'),
                             train_s=0.0, eval_s=0.0,
                             error=str(e)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problems', default='all')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default=os.path.join(HERE, 'results',
                                                   'pautomac_full.csv'))
    args = ap.parse_args()

    problems = parse_problems(args.problems)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"=== Parallel PAutomaC sweep: {len(problems)} problems × "
          f"{len(get_models())} models, {args.workers} workers ===",
          flush=True)
    t0 = time.time()
    all_rows = []
    done = 0
    with mp.Pool(processes=args.workers) as pool:
        for rows in pool.imap_unordered(evaluate_problem, problems,
                                         chunksize=1):
            all_rows.extend(rows); done += 1
            if rows:
                pi = rows[0]['problem']
                print(f"  problem {pi:>2d} done  "
                      f"({done}/{len(problems)})  "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    fields = ['problem', 'alphabet_size', 'model', 'score',
              'entropy_floor', 'gap', 'lift', 'train_s', 'eval_s']
    if any('error' in r for r in all_rows):
        fields.append('error')
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {args.out} ({len(all_rows)} rows, "
          f"{time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
