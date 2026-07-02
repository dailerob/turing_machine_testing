"""GDC + LSTM on NPI-style BASE-4 addition.

Middle point between base 2 (where local context was too ambiguous) and
base 10 (where the addition table was too big). Base 4 has:
    addition table = 4 × 4 × 2 = 32 cells
    obs space = 5^4 = 625 tuples
    digit values = {0, 1, 2, 3, BLANK}
"""
from __future__ import annotations
import os, sys, time, random
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

# Monkey-patch npi_program to be npi_program_b4 so npi_eval picks it up.
import npi_program_b4 as _npi_b4
sys.modules['npi_program'] = _npi_b4

import importlib                                                              # noqa: E402
import npi_eval as _npi_eval
importlib.reload(_npi_eval)
import npi_eval_torch as _npi_eval_torch
importlib.reload(_npi_eval_torch)
import npi_gdc_torch as _npi_gdc_torch
importlib.reload(_npi_gdc_torch)
import npi_lstm as _npi_lstm
importlib.reload(_npi_lstm)

from npi_program_b4 import generate_trace, BLANK, BASE                        # noqa: E402
from npi_gdc_torch import NpiGDCTorch                                         # noqa: E402
from npi_eval_torch import forecast_bucket_torch                              # noqa: E402
from npi_lstm import FlatLSTM, train as lstm_train, lstm_forecast_one         # noqa: E402
from npi_eval import _Simulator, _make_init_rows                              # noqa: E402


# Patch simulator decode for base 4.
def _decode_output_b4(self):
    cells = {i: v for i, v in enumerate(self.row4) if v is not None}
    if not cells:
        return 0
    max_col = max(cells.keys())
    digits = [str(cells.get(i, 0)) for i in range(max_col, -1, -1)]
    s = ''.join(digits).lstrip('0')
    return int(s, BASE) if s else 0

_Simulator.decode_output = _decode_output_b4

# CRITICAL FIX: _make_init_rows uses str(a) (base-10 string) by default.
# For base 4 we need to use base-4 digit string.
import npi_eval as _npi_eval_mod
from npi_program_b4 import (to_base_str, AT_INIT, AT_INIT_A, AT_INIT_B,
                              INIT_BEGIN, INIT_A_END, INIT_B_END, INIT_END)

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

_npi_eval_mod._make_init_rows = _make_init_rows_b4
# Re-import so the patched _make_init_rows is picked up by downstream modules.
importlib.reload(_npi_eval_torch)
importlib.reload(_npi_lstm)
from npi_eval import _Simulator, _make_init_rows                              # noqa: E402, F811
from npi_eval_torch import forecast_bucket_torch                              # noqa: E402, F811
from npi_lstm import FlatLSTM, train as lstm_train, lstm_forecast_one         # noqa: E402, F811


def digit_pair(n_digits: int, n: int, rng: random.Random):
    """Sample (a, b) where both have exactly n_digits base-4 digits."""
    out = []
    lo = BASE ** (n_digits - 1) if n_digits > 0 else 0
    hi = BASE ** n_digits
    for _ in range(n):
        out.append((rng.randint(lo, hi - 1), rng.randint(lo, hi - 1)))
    return out


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}  BASE={BASE}")

    rng_np = np.random.default_rng(42)
    n_train = 10000
    train_pairs = []
    for _ in range(n_train):
        digs = int(rng_np.integers(1, 4))   # 1, 2, or 3 base-4 digits
        a = int(rng_np.integers(BASE**(digs-1), BASE**digs))
        b = int(rng_np.integers(BASE**(digs-1), BASE**digs))
        train_pairs.append((a, b))
    train_traces = [generate_trace(a, b) for (a, b) in train_pairs]
    total_rows = sum(t.shape[0] for t in train_traces)
    avg_trace = total_rows / n_train
    print(f"n_train={n_train}  total chain rows={total_rows}  "
          f"avg trace len={avg_trace:.1f}")

    train_obs_set = set()
    for tr in train_traces:
        for row in tr:
            t = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            if t != (BLANK, BLANK, BLANK, BLANK):
                train_obs_set.add(t)
    print(f"Distinct training 4-tape obs tuples: {len(train_obs_set)}")

    rng_py = random.Random(43)
    eval_buckets = [
        ('1-dig (in)',   digit_pair(1, 25, rng_py)),
        ('2-dig (in)',   digit_pair(2, 25, rng_py)),
        ('3-dig (in)',   digit_pair(3, 25, rng_py)),
        ('4-dig OOD',    digit_pair(4, 25, rng_py)),
        ('5-dig OOD',    digit_pair(5, 25, rng_py)),
        ('7-dig OOD',    digit_pair(7, 25, rng_py)),
        ('10-dig OOD',   digit_pair(10, 25, rng_py)),
    ]

    # GDC
    print(f"\n=== GDC (alpha=0.90, theta=0.0, uniform init) ===")
    gdc = NpiGDCTorch(train_traces, alpha=0.90, theta=0.0, beta=0.0,
                       initial_dist='uniform',
                       device=device, dtype=torch.float64)
    print(f"  Chain N = {gdc.N}")
    gdc_results = []
    t_total = time.time()
    for bn, pairs in eval_buckets:
        res = forecast_bucket_torch(gdc, pairs, device, max_steps=800,
                                      train_obs_set=train_obs_set)
        n_correct = sum(res['exact_match'])
        act = (100.0 * res['action_correct'] / res['action_total']
               if res['action_total'] else 0.0)
        cov = (100.0 * res['obs_in_train'] / res['obs_total']
               if res['obs_total'] else 0.0)
        gdc_results.append((bn, n_correct, act, cov, res['time_sec']))
        rate = 100.0 * n_correct / len(pairs)
        print(f"  [{bn:>14s}]  exact={n_correct:>2d}/25 ({rate:>5.1f}%)  "
              f"action={act:>5.1f}%  obs_in_train={cov:>5.1f}%  "
              f"({res['time_sec']:.1f}s)")
    print(f"  GDC total: {time.time()-t_total:.1f}s")

    # LSTM
    print(f"\n=== LSTM (2-layer × 256, 60 epochs) ===")
    model = FlatLSTM(emb_dim=32, hidden=256, n_layers=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params/1e6:.2f}M")
    lstm_train(model, train_traces, device, n_epochs=60, batch_size=64,
                lr=1e-3, log_every=15)

    lstm_results = []
    for bn, pairs in eval_buckets:
        t0 = time.time()
        n_correct = 0; action_correct = 0; action_total = 0
        for a, b in pairs:
            res = lstm_forecast_one(model, a, b, device, max_steps=800)
            ok = (res['predicted_output'] == a + b)
            n_correct += int(ok)
            gt = generate_trace(a, b)
            gt_post_init = gt[len(_make_init_rows(a, b)):]
            gt_actions = [(int(r[4]), int(r[5])) for r in gt_post_init]
            L = min(len(res['predicted_actions']), len(gt_actions))
            for i in range(L):
                if res['predicted_actions'][i] == gt_actions[i]:
                    action_correct += 1
            action_total += L
        rate = 100.0 * n_correct / len(pairs)
        act_rate = (100.0 * action_correct / action_total
                    if action_total else 0.0)
        lstm_results.append((bn, n_correct, act_rate, time.time() - t0))
        print(f"  [{bn:>14s}]  exact={n_correct:>2d}/25 ({rate:>5.1f}%)  "
              f"action={act_rate:>5.1f}%  ({time.time()-t0:.1f}s)")

    print(f"\n=== Summary: BASE-4 NPI addition, n_train={n_train}, 1-3 digit train ===")
    print(f"{'bucket':>16s}  {'GDC exact':>10s}  {'LSTM exact':>11s}  "
          f"{'GDC act':>8s}  {'LSTM act':>9s}  {'obs cov':>8s}")
    for i, (bn, gex, gact, gcov, _) in enumerate(gdc_results):
        _, lex, lact, _ = lstm_results[i]
        print(f"{bn:>16s}  {gex:>4d}/25      {lex:>4d}/25       "
              f"{gact:>6.1f}%  {lact:>7.1f}%   {gcov:>6.1f}%")


if __name__ == "__main__":
    main()
