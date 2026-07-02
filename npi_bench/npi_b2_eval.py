"""Compare GDC and LSTM on NPI-style BINARY addition traces.

Same protocol as `npi_eval_torch.py` / `npi_lstm.py` but using the base-2
trace generator. Hypothesis: with the addition table reduced from 200 to 8
cells, GDC's coverage problem goes away and it should reach much higher
action accuracy / more exact matches on length-OOD test.
"""
from __future__ import annotations
import os, sys, time, random
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

# Important: import the binary versions of the program/eval modules. The
# helpers in npi_eval.py reference `generate_trace`, `BLANK`, `AT_HALT`, etc.
# We patch the npi_eval / npi_eval_torch / npi_lstm modules' imports by
# replacing module-level symbols.

# We need the simulator and init-prefix builder to use the BINARY conventions.
# The simplest: monkey-patch `npi_program` in sys.modules to be `npi_program_b2`
# before importing npi_eval. That way `from npi_program import generate_trace`
# inside npi_eval picks up the binary version.
import npi_program_b2 as _npi_b2
sys.modules['npi_program'] = _npi_b2

import importlib                                                              # noqa: E402
import npi_eval as _npi_eval                                                  # noqa: E402
importlib.reload(_npi_eval)
import npi_eval_torch as _npi_eval_torch                                      # noqa: E402
importlib.reload(_npi_eval_torch)
import npi_gdc_torch as _npi_gdc_torch                                        # noqa: E402
importlib.reload(_npi_gdc_torch)
import npi_lstm as _npi_lstm                                                  # noqa: E402
importlib.reload(_npi_lstm)

from npi_program_b2 import generate_trace, BLANK                              # noqa: E402
from npi_gdc_torch import NpiGDCTorch                                         # noqa: E402
from npi_eval_torch import forecast_bucket_torch                              # noqa: E402
from npi_lstm import FlatLSTM, train as lstm_train, lstm_forecast_one         # noqa: E402
from npi_eval import _Simulator, _make_init_rows                              # noqa: E402


# The simulator's decode_output() reads row 4 as base-10 digits — patch
# it to decode as base 2 for this experiment.
def _decode_output_binary(self):
    cells = {i: v for i, v in enumerate(self.row4) if v is not None}
    if not cells:
        return 0
    max_col = max(cells.keys())
    bits = ''.join(str(cells.get(i, 0)) for i in range(max_col, -1, -1)).lstrip('0')
    return int(bits, 2) if bits else 0

_Simulator.decode_output = _decode_output_binary

# CRITICAL FIX: _make_init_rows in npi_eval uses str(a) which is base-10.
# For base 2 we need binary digit string (bin(a)[2:]).
import npi_eval as _npi_eval_mod
from npi_program_b2 import (AT_INIT, AT_INIT_A, AT_INIT_B,
                              INIT_BEGIN, INIT_A_END, INIT_B_END, INIT_END)

def _make_init_rows_b2(a, b):
    rows = []
    def emit(at, arg):
        rows.append((BLANK, BLANK, BLANK, BLANK, at, arg))
    emit(AT_INIT, INIT_BEGIN)
    for d in bin(a)[2:]:
        emit(AT_INIT_A, int(d))
    emit(AT_INIT, INIT_A_END)
    for d in bin(b)[2:]:
        emit(AT_INIT_B, int(d))
    emit(AT_INIT, INIT_B_END)
    emit(AT_INIT, INIT_END)
    return np.array(rows, dtype=np.int64)

_npi_eval_mod._make_init_rows = _make_init_rows_b2
importlib.reload(_npi_eval_torch)
importlib.reload(_npi_lstm)
from npi_eval import _Simulator, _make_init_rows                              # noqa: E402, F811
from npi_eval_torch import forecast_bucket_torch                              # noqa: E402, F811
from npi_lstm import FlatLSTM, train as lstm_train, lstm_forecast_one         # noqa: E402, F811


def bit_pair(n_bits: int, n: int, rng: random.Random):
    """Sample (a, b) pairs where both operands have exactly n_bits bits."""
    out = []
    lo = 1 << (n_bits - 1) if n_bits > 0 else 0
    hi = 1 << n_bits
    for _ in range(n):
        out.append((rng.randint(lo, hi - 1), rng.randint(lo, hi - 1)))
    return out


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}")

    # --- Training data: 1-3 bit operands, 200 random pairs ---
    rng_np = np.random.default_rng(42)
    n_train = 200
    train_pairs = []
    for _ in range(n_train):
        # operands have 1, 2, or 3 bits → values in 1..7
        bits = int(rng_np.integers(1, 4))
        a = int(rng_np.integers(1 << (bits - 1), 1 << bits))
        b = int(rng_np.integers(1 << (bits - 1), 1 << bits))
        train_pairs.append((a, b))
    train_traces = [generate_trace(a, b) for (a, b) in train_pairs]
    total_rows = sum(t.shape[0] for t in train_traces)
    avg_trace = total_rows / n_train
    print(f"n_train={n_train}  total chain rows={total_rows}  "
          f"avg trace len={avg_trace:.1f}")

    # Build training obs set for coverage stat.
    train_obs_set = set()
    for tr in train_traces:
        for row in tr:
            t = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            if t != (BLANK, BLANK, BLANK, BLANK):
                train_obs_set.add(t)
    print(f"Distinct training 4-tape obs tuples: {len(train_obs_set)}")

    # --- Eval buckets: vary bit count ---
    rng_py = random.Random(43)
    eval_buckets = [
        ('1-bit (in)',   bit_pair(1, 25, rng_py)),
        ('2-bit (in)',   bit_pair(2, 25, rng_py)),
        ('3-bit (in)',   bit_pair(3, 25, rng_py)),
        ('4-bit OOD',    bit_pair(4, 25, rng_py)),
        ('5-bit OOD',    bit_pair(5, 25, rng_py)),
        ('7-bit OOD',    bit_pair(7, 25, rng_py)),
        ('10-bit OOD',   bit_pair(10, 25, rng_py)),
        ('15-bit OOD',   bit_pair(15, 25, rng_py)),
        ('20-bit OOD',   bit_pair(20, 25, rng_py)),
    ]

    # =========================================================================
    # GDC (alpha=0.90, theta=0, uniform init — best from base-10 sweep)
    # =========================================================================
    print(f"\n=== GDC (alpha=0.90, theta=0.0, uniform init) ===")
    gdc = NpiGDCTorch(train_traces, alpha=0.90, theta=0.0, beta=0.0,
                       initial_dist='uniform',
                       device=device, dtype=torch.float64)
    print(f"  Chain N = {gdc.N}")
    t_total = time.time()
    gdc_results = []
    for bn, pairs in eval_buckets:
        res = forecast_bucket_torch(gdc, pairs, device,
                                      max_steps=800,
                                      train_obs_set=train_obs_set)
        n_correct = sum(res['exact_match'])
        rate = 100.0 * n_correct / len(pairs)
        act = (100.0 * res['action_correct'] / res['action_total']
               if res['action_total'] else 0.0)
        cov = (100.0 * res['obs_in_train'] / res['obs_total']
               if res['obs_total'] else 0.0)
        gdc_results.append((bn, n_correct, len(pairs), act, cov,
                             res['time_sec']))
        print(f"  [{bn:>14s}]  exact={n_correct:>2d}/25 ({rate:>5.1f}%)  "
              f"action={act:>5.1f}%  obs_in_train={cov:>5.1f}%  "
              f"({res['time_sec']:.1f}s)")
    print(f"  GDC total time: {time.time()-t_total:.1f}s")

    # =========================================================================
    # LSTM (2-layer × 256 hidden, same as base-10)
    # =========================================================================
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
        lstm_results.append((bn, n_correct, len(pairs), act_rate))
        print(f"  [{bn:>14s}]  exact={n_correct:>2d}/25 ({rate:>5.1f}%)  "
              f"action={act_rate:>5.1f}%   ({time.time()-t0:.1f}s)")

    print(f"\n=== Summary: binary NPI addition (n_train={n_train}, 1-3 bit train) ===")
    print(f"{'bucket':>16s}  {'GDC exact':>10s}  {'LSTM exact':>11s}  "
          f"{'GDC act':>8s}  {'LSTM act':>9s}  {'obs cov':>8s}")
    for i, (bn, *_) in enumerate(gdc_results):
        gex, gn, gact, gcov, _ = gdc_results[i][1], gdc_results[i][2], \
                                  gdc_results[i][3], gdc_results[i][4], \
                                  gdc_results[i][5]
        _, lex, ln, lact = lstm_results[i]
        print(f"{bn:>16s}  {gex:>4d}/{gn:<3d}    {lex:>4d}/{ln:<3d}      "
              f"{gact:>6.1f}%  {lact:>7.1f}%   {gcov:>6.1f}%")


if __name__ == "__main__":
    main()
