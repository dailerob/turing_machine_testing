"""(alpha, theta, beta) parameter sweep for NPI addition with GDC partial_match.

Reuses the torch kernel; each (alpha, theta, beta) configuration takes ~1-2 s
total over all 7 length buckets.

Reports per-config:
  - per-bucket action accuracy (the per-step prediction-correctness rate)
  - per-bucket exact-match count (out of 25)
  - the best-by-action-acc length-3 bucket for the picked-cell summary
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from npi_program import generate_trace, BLANK                                 # noqa: E402
from npi_eval import _Simulator, _make_init_rows                              # noqa: E402
from npi_eval_torch import forecast_bucket_torch                              # noqa: E402
from npi_gdc_torch import NpiGDCTorch                                         # noqa: E402


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}\n")

    # Training set: same as smoke (n=1200, 1-3 digit operands, seed 42).
    rng = np.random.default_rng(42)
    n_train = 1200
    train_pairs = []
    for _ in range(n_train):
        da = int(rng.integers(1, 4)); db = int(rng.integers(1, 4))
        train_pairs.append((int(rng.integers(10**(da-1), 10**da)),
                             int(rng.integers(10**(db-1), 10**db))))
    train_traces = [generate_trace(a, b) for (a, b) in train_pairs]
    total_rows = sum(t.shape[0] for t in train_traces)
    print(f"n_train={n_train}  total chain rows={total_rows}")

    # Build training obs set for coverage stat (computed once).
    train_obs_set = set()
    for tr in train_traces:
        for row in tr:
            t = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            if t != (BLANK, BLANK, BLANK, BLANK):
                train_obs_set.add(t)

    # Test buckets (same as smoke; fix seed so cells are comparable).
    bucket_rng = np.random.default_rng(42 + 1)
    def n_digit_pair(lo, hi, n):
        return [(int(bucket_rng.integers(lo, hi)),
                 int(bucket_rng.integers(lo, hi))) for _ in range(n)]
    # Re-roll the same pairs as in npi_eval_torch.run_smoke_torch: this is
    # what the bucket_rng above lines up with given the seed.
    eval_buckets = [
        ('len-1',  n_digit_pair(0, 10, 25)),
        ('len-2',  n_digit_pair(10, 100, 25)),
        ('len-3',  n_digit_pair(100, 1000, 25)),
        ('len-4',  n_digit_pair(1000, 10000, 25)),
        ('len-5',  n_digit_pair(10000, 100000, 25)),
        ('len-7',  n_digit_pair(10**6, 10**7, 25)),
        ('len-10', n_digit_pair(10**9, 10**10, 25)),
    ]

    # Sweep grid. Only valid (alpha+theta<=1) tuples.
    alpha_theta_grid = [
        (0.85, 0.00), (0.85, 0.05), (0.85, 0.10),
        (0.90, 0.00), (0.90, 0.05), (0.90, 0.10),
        (0.95, 0.00), (0.95, 0.05),
        (0.99, 0.00), (0.99, 0.01),
        (1.00, 0.00),
    ]
    beta_grid = [0.0, 0.01, 0.05, 0.1]
    configs = [(a, t, b) for (a, t) in alpha_theta_grid for b in beta_grid]
    print(f"Sweep: {len(configs)} configs "
          f"({len(alpha_theta_grid)} α/θ × {len(beta_grid)} β)\n")

    # Run all configs and collect per-(config, bucket) action_acc / exact.
    bucket_names = [b[0] for b in eval_buckets]
    results = []   # list of dicts
    t_total = time.time()
    for ci, (alpha, theta, beta) in enumerate(configs):
        t0 = time.time()
        gdc = NpiGDCTorch(train_traces, alpha=alpha, theta=theta, beta=beta,
                           device=device, dtype=torch.float64)
        row = dict(alpha=alpha, theta=theta, beta=beta)
        for bn, pairs in eval_buckets:
            res = forecast_bucket_torch(gdc, pairs, device, max_steps=400,
                                          train_obs_set=None)
            n_correct = sum(res['exact_match'])
            act = (res['action_correct'] / res['action_total']
                   if res['action_total'] else 0.0)
            row[f'{bn}_act'] = act
            row[f'{bn}_exact'] = n_correct
        results.append(row)
        elapsed = time.time() - t0
        print(f"  [{ci+1:>2d}/{len(configs)}]  "
              f"α={alpha:.2f} θ={theta:.2f} β={beta:.2f}   "
              + "  ".join(
                  f"{bn}={row[f'{bn}_act']*100:>5.1f}%"
                  for bn in bucket_names
              )
              + f"   ({elapsed:.1f}s)")
    print(f"\nTotal sweep time: {time.time()-t_total:.1f}s")

    # ---- Summaries ----
    # (1) Best config by mean action_acc across all 7 buckets.
    for r in results:
        r['mean_act'] = np.mean([r[f'{bn}_act'] for bn in bucket_names])
        r['total_exact'] = sum(r[f'{bn}_exact'] for bn in bucket_names)
    results.sort(key=lambda r: -r['mean_act'])
    print(f"\n=== Top 10 configs by mean action accuracy across all 7 buckets ===")
    print(f"{'α':>5s} {'θ':>5s} {'β':>5s}  {'mean_act':>9s}  "
          f"{'exact_total':>11s}  " + "  ".join(f"{bn:>6s}" for bn in bucket_names))
    for r in results[:10]:
        print(f"{r['alpha']:>5.2f} {r['theta']:>5.2f} {r['beta']:>5.2f}  "
              f"{100*r['mean_act']:>8.2f}%  "
              f"{r['total_exact']:>11d}  "
              + "  ".join(f"{100*r[f'{bn}_act']:>5.1f}%" for bn in bucket_names))

    # (2) Best config by exact-match count.
    results2 = sorted(results, key=lambda r: -r['total_exact'])
    print(f"\n=== Top 10 configs by total exact-match count (out of {7*25}=175) ===")
    print(f"{'α':>5s} {'θ':>5s} {'β':>5s}  {'exact_tot':>10s}  {'mean_act':>9s}  "
          + "  ".join(f"{bn:>6s}" for bn in bucket_names))
    for r in results2[:10]:
        print(f"{r['alpha']:>5.2f} {r['theta']:>5.2f} {r['beta']:>5.2f}  "
              f"{r['total_exact']:>10d}  {100*r['mean_act']:>8.2f}%  "
              + "  ".join(f"{r[f'{bn}_exact']:>4d}/25" for bn in bucket_names))


if __name__ == "__main__":
    main()
