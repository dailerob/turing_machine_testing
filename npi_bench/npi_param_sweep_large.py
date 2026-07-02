"""Hyperparameter sweep (alpha × theta) on the n_train=10000 (1-3 digit)
NPI addition setup.

Skips beta (the previous sweep showed it has no effect on the partial-match
argmax). Focuses on (alpha, theta) tuples that bracket the previous best
config (alpha=0.85, theta=0.05).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from npi_program import generate_trace, BLANK                                 # noqa: E402
from npi_eval_torch import forecast_bucket_torch                              # noqa: E402
from npi_gdc_torch import NpiGDCTorch                                         # noqa: E402


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}")

    # Training set: 10000 pairs, 1-3 digit operands.
    rng = np.random.default_rng(42)
    n_train = 10000
    train_pairs = []
    for _ in range(n_train):
        da = int(rng.integers(1, 4)); db = int(rng.integers(1, 4))
        train_pairs.append((int(rng.integers(10**(da-1), 10**da)),
                             int(rng.integers(10**(db-1), 10**db))))
    train_traces = [generate_trace(a, b) for (a, b) in train_pairs]
    total_rows = sum(t.shape[0] for t in train_traces)
    print(f"n_train={n_train}  total chain rows={total_rows}")

    # Eval buckets.
    import random as _r
    _r.seed(43)
    def py_digit_pair(n_digits, n):
        return [(_r.randint(10**(n_digits-1), 10**n_digits - 1),
                 _r.randint(10**(n_digits-1), 10**n_digits - 1))
                for _ in range(n)]
    eval_buckets = [
        ('len-1',  py_digit_pair(1, 25)),
        ('len-2',  py_digit_pair(2, 25)),
        ('len-3',  py_digit_pair(3, 25)),
        ('len-4',  py_digit_pair(4, 25)),
        ('len-5',  py_digit_pair(5, 25)),
        ('len-7',  py_digit_pair(7, 25)),
        ('len-10', py_digit_pair(10, 25)),
        ('len-15', py_digit_pair(15, 25)),
        ('len-20', py_digit_pair(20, 25)),
    ]
    bucket_names = [b[0] for b in eval_buckets]

    # Sweep grid. Constraint: alpha + theta <= 0.99 (keep at least 1% diffusion).
    alpha_grid = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    theta_grid = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]
    configs = [(a, t) for a in alpha_grid for t in theta_grid
               if a + t <= 0.99]
    print(f"Sweep: {len(configs)} configs")

    results = []
    t_total = time.time()
    for ci, (alpha, theta) in enumerate(configs):
        t0 = time.time()
        gdc = NpiGDCTorch(train_traces, alpha=alpha, theta=theta, beta=0.0,
                           device=device, dtype=torch.float64)
        row = dict(alpha=alpha, theta=theta)
        for bn, pairs in eval_buckets:
            res = forecast_bucket_torch(gdc, pairs, device,
                                          max_steps=600, train_obs_set=None)
            n_correct = sum(res['exact_match'])
            act = (res['action_correct'] / res['action_total']
                   if res['action_total'] else 0.0)
            row[f'{bn}_act'] = act
            row[f'{bn}_exact'] = n_correct
        results.append(row)
        elapsed = time.time() - t0
        # Compact per-row print
        bucket_str = "  ".join(
            f"{bn}={row[f'{bn}_act']*100:>5.1f}%"
            for bn in bucket_names
        )
        exact_str = sum(row[f'{bn}_exact'] for bn in bucket_names)
        print(f"  [{ci+1:>2d}/{len(configs)}]  "
              f"α={alpha:.2f} θ={theta:.3f} (diff={1-alpha-theta:.3f})   "
              f"{bucket_str}   exact={exact_str:>2d}  ({elapsed:.1f}s)")
        del gdc
    print(f"\nTotal sweep: {time.time()-t_total:.1f}s")

    # ---- Summaries ----
    for r in results:
        r['mean_act'] = np.mean([r[f'{bn}_act'] for bn in bucket_names])
        r['total_exact'] = sum(r[f'{bn}_exact'] for bn in bucket_names)
        # Also: mean of just OOD buckets (length >= 4)
        r['mean_ood'] = np.mean([r[f'{bn}_act'] for bn in
                                  ['len-4', 'len-5', 'len-7', 'len-10',
                                   'len-15', 'len-20']])

    results.sort(key=lambda r: -r['mean_act'])
    print(f"\n=== Top 10 by mean action accuracy (all 9 buckets) ===")
    print(f"{'α':>5s} {'θ':>5s} {'diff':>5s}  "
          f"{'mean_all':>8s}  {'mean_OOD':>8s}  {'exact':>5s}  "
          + "  ".join(f"{bn:>5s}" for bn in bucket_names))
    for r in results[:10]:
        diff = 1.0 - r['alpha'] - r['theta']
        print(f"{r['alpha']:>5.2f} {r['theta']:>5.3f} {diff:>5.3f}  "
              f"{100*r['mean_act']:>7.2f}%  {100*r['mean_ood']:>7.2f}%  "
              f"{r['total_exact']:>5d}  "
              + "  ".join(f"{100*r[f'{bn}_act']:>4.1f}%" for bn in bucket_names))

    print(f"\n=== Top 10 by mean OOD action accuracy (lengths 4-20) ===")
    res_ood = sorted(results, key=lambda r: -r['mean_ood'])
    print(f"{'α':>5s} {'θ':>5s} {'diff':>5s}  "
          f"{'mean_all':>8s}  {'mean_OOD':>8s}  {'exact':>5s}  "
          + "  ".join(f"{bn:>5s}" for bn in bucket_names))
    for r in res_ood[:10]:
        diff = 1.0 - r['alpha'] - r['theta']
        print(f"{r['alpha']:>5.2f} {r['theta']:>5.3f} {diff:>5.3f}  "
              f"{100*r['mean_act']:>7.2f}%  {100*r['mean_ood']:>7.2f}%  "
              f"{r['total_exact']:>5d}  "
              + "  ".join(f"{100*r[f'{bn}_act']:>4.1f}%" for bn in bucket_names))

    print(f"\n=== Top 5 by total exact-match (out of {9*25}=225) ===")
    res_ex = sorted(results, key=lambda r: -r['total_exact'])
    for r in res_ex[:5]:
        diff = 1.0 - r['alpha'] - r['theta']
        print(f"  α={r['alpha']:.2f} θ={r['theta']:.3f} diff={diff:.3f}  "
              f"exact={r['total_exact']:>3d}   "
              + "  ".join(f"{bn}={r[f'{bn}_exact']:>2d}"
                          for bn in bucket_names))


if __name__ == "__main__":
    main()
