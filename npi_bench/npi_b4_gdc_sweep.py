"""GDC hyperparameter sweep at base 4, n_train=10000."""
from __future__ import annotations
import os, sys, time, random
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

# Monkey-patch npi_program → npi_program_b4 so npi_eval picks it up.
import npi_program_b4 as _b4
sys.modules['npi_program'] = _b4

import importlib
import npi_eval as _npi_eval
importlib.reload(_npi_eval)
import npi_eval_torch as _npi_eval_torch
importlib.reload(_npi_eval_torch)
import npi_gdc_torch as _npi_gdc_torch
importlib.reload(_npi_gdc_torch)

from npi_program_b4 import (generate_trace, BLANK, BASE,                       # noqa: E402
    to_base_str, AT_INIT, AT_INIT_A, AT_INIT_B,
    INIT_BEGIN, INIT_A_END, INIT_B_END, INIT_END)
from npi_gdc_torch import NpiGDCTorch                                         # noqa: E402

# Patch base-4 init rows + decode.
def _make_init_rows_b4(a, b):
    rows = []
    def emit(at, arg):
        rows.append((BLANK, BLANK, BLANK, BLANK, at, arg))
    emit(AT_INIT, INIT_BEGIN)
    for d in to_base_str(a):
        emit(AT_INIT_A, int(d))
    emit(AT_INIT, INIT_A_END)
    for d in to_base_str(b):
        emit(AT_INIT_B, int(d))
    emit(AT_INIT, INIT_B_END)
    emit(AT_INIT, INIT_END)
    return np.array(rows, dtype=np.int64)

_npi_eval._make_init_rows = _make_init_rows_b4
importlib.reload(_npi_eval_torch)

from npi_eval import _Simulator                                              # noqa: E402
from npi_eval_torch import forecast_bucket_torch                              # noqa: E402

def _decode_output_b4(self):
    cells = {i: v for i, v in enumerate(self.row4) if v is not None}
    if not cells:
        return 0
    mx = max(cells)
    s = ''.join(str(cells.get(i, 0)) for i in range(mx, -1, -1)).lstrip('0')
    return int(s, BASE) if s else 0
_Simulator.decode_output = _decode_output_b4


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}  BASE={BASE}")

    # Training: 10000 1-3 base-4 digit pairs.
    rng_np = np.random.default_rng(42)
    n_train = 10000
    train_pairs = []
    for _ in range(n_train):
        d = int(rng_np.integers(1, 4))
        train_pairs.append((int(rng_np.integers(BASE**(d-1), BASE**d)),
                             int(rng_np.integers(BASE**(d-1), BASE**d))))
    train_traces = [generate_trace(a, b) for (a, b) in train_pairs]
    total_rows = sum(t.shape[0] for t in train_traces)
    print(f"n_train={n_train}  chain rows={total_rows}\n")

    rng_py = random.Random(43)
    def digit_pair(n_digits, n):
        return [(rng_py.randint(BASE**(n_digits-1), BASE**n_digits - 1),
                 rng_py.randint(BASE**(n_digits-1), BASE**n_digits - 1))
                for _ in range(n)]
    eval_buckets = [
        ('1-dig (in)',  digit_pair(1, 25)),
        ('2-dig (in)',  digit_pair(2, 25)),
        ('3-dig (in)',  digit_pair(3, 25)),
        ('4-dig OOD',   digit_pair(4, 25)),
        ('5-dig OOD',   digit_pair(5, 25)),
        ('7-dig OOD',   digit_pair(7, 25)),
        ('10-dig OOD',  digit_pair(10, 25)),
    ]
    bucket_names = [b[0] for b in eval_buckets]

    # Sweep grid.
    alpha_grid = [0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    theta_grid = [0.0, 0.025, 0.05, 0.1]
    init_grid = ['uniform', 'sequence_starts']
    configs = []
    for a in alpha_grid:
        for t in theta_grid:
            if a + t > 0.99: continue
            for init in init_grid:
                configs.append((a, t, init))
    print(f"Sweep: {len(configs)} configs\n")

    results = []
    print(f"  {'α':>5s} {'θ':>5s} {'init':>16s}  " +
          '  '.join(f'{bn:>7s}' for bn in bucket_names) +
          f"  {'mean_act':>8s}  {'exact':>5s}")
    t_total = time.time()
    for ci, (alpha, theta, init) in enumerate(configs):
        t0 = time.time()
        gdc = NpiGDCTorch(train_traces, alpha=alpha, theta=theta, beta=0.0,
                           initial_dist=init,
                           device=device, dtype=torch.float64)
        row = dict(alpha=alpha, theta=theta, init=init)
        for bn, pairs in eval_buckets:
            res = forecast_bucket_torch(gdc, pairs, device, max_steps=600,
                                          train_obs_set=None)
            row[f'{bn}_act'] = (res['action_correct'] / res['action_total']
                                if res['action_total'] else 0.0)
            row[f'{bn}_exact'] = sum(res['exact_match'])
        row['mean_act'] = np.mean([row[f'{bn}_act'] for bn in bucket_names])
        row['total_exact'] = sum(row[f'{bn}_exact'] for bn in bucket_names)
        results.append(row)
        elapsed = time.time() - t0
        cells = '  '.join(f'{row[f"{bn}_exact"]:>2d}/{row[f"{bn}_act"]*100:>3.0f}%'
                          for bn in bucket_names)
        print(f"  {alpha:>5.2f} {theta:>5.3f} {init:>16s}  {cells}  "
              f"{row['mean_act']*100:>7.2f}%  {row['total_exact']:>5d}")
        del gdc

    print(f"\nTotal: {time.time()-t_total:.1f}s")

    # Summary
    results.sort(key=lambda r: -r['mean_act'])
    print(f"\n=== Top 10 by mean action accuracy across all 7 buckets ===")
    print(f"  {'α':>5s} {'θ':>5s} {'init':>16s}  {'mean_act':>8s}  {'exact':>5s}  " +
          '  '.join(f'{bn:>7s}' for bn in bucket_names))
    for r in results[:10]:
        cells = '  '.join(f'{r[f"{bn}_act"]*100:>5.1f}%'
                          for bn in bucket_names)
        print(f"  {r['alpha']:>5.2f} {r['theta']:>5.3f} {r['init']:>16s}  "
              f"{r['mean_act']*100:>7.2f}%  {r['total_exact']:>5d}  {cells}")

    print(f"\n=== Top 5 by total exact match ===")
    res_ex = sorted(results, key=lambda r: -r['total_exact'])
    for r in res_ex[:5]:
        cells = '  '.join(f'{r[f"{bn}_exact"]:>2d}'
                          for bn in bucket_names)
        print(f"  α={r['alpha']:.2f} θ={r['theta']:.3f} init={r['init']:>16s}  "
              f"exact={r['total_exact']:>2d}  per-bucket {cells}")


if __name__ == "__main__":
    main()
