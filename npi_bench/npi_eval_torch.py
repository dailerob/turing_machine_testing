"""Torch-batched eval of NPI addition traces with GDC partial_match.

Mirrors npi_eval.run_smoke but processes a whole bucket of test pairs in
parallel through NpiGDCTorch. The chain lives on GPU; per-step forward
pass is a batched transition + emission update. Simulators are kept in
Python (the action set is too irregular to vectorize cleanly).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from npi_program import (generate_trace, BLANK,                              # noqa: E402
    AT_HALT, AT_INIT, AT_INIT_A, AT_INIT_B,
    INIT_BEGIN, INIT_A_END, INIT_B_END, INIT_END)
from npi_eval import _Simulator, _make_init_rows                              # noqa: E402
from npi_gdc_torch import NpiGDCTorch                                         # noqa: E402


def _pad_init_rows_to_batch(pairs, device):
    """Init blocks have different lengths (depends on digit count of a, b).
    To feed them as a (B, max_init_len, k) tensor we pad with "noop" rows
    that won't disturb the posterior much. Simpler: process each pair's
    init individually before the parallel forecast loop, since the init
    cost is tiny relative to the forecast cost."""
    return [_make_init_rows(a, b) for (a, b) in pairs]


def forecast_bucket_torch(gdc: NpiGDCTorch, pairs, device,
                           max_steps: int = 400,
                           train_obs_set: 'set | None' = None):
    """Forecast a whole bucket of test pairs in parallel.

    Returns dict with:
        exact_match : list of bool, len(pairs)
        action_correct, action_total : int
        obs_in_train, obs_total : int (counted only if train_obs_set given)
        time_sec : float
    """
    B = len(pairs)
    init_rows_list = _pad_init_rows_to_batch(pairs, device)
    init_lens = [len(r) for r in init_rows_list]
    max_init = max(init_lens)

    sims = [_Simulator(a, b, n_cols_extra=4 + max(len(str(a)), len(str(b))))
            for (a, b) in pairs]
    # Precompute ground-truth post-init actions for action_acc.
    gt_traces = [generate_trace(a, b) for (a, b) in pairs]
    gt_post_init = [tr[init_lens[i]:] for i, tr in enumerate(gt_traces)]
    gt_actions = [[(int(r[4]), int(r[5])) for r in gt]
                  for gt in gt_post_init]

    t0 = time.time()
    gdc.reset(batch_size=B)
    # Consume init prefix. To keep it simple we feed init rows sequentially;
    # different batches' init blocks may have different lengths so pad with
    # an INIT_END marker (already in vocab) at the end.
    for step_i in range(max_init):
        row_batch = np.zeros((B, gdc.k), dtype=np.int64)
        for b in range(B):
            if step_i < init_lens[b]:
                row_batch[b] = init_rows_list[b][step_i]
            else:
                # Pad: re-feed the last init row (INIT_END) — harmless replay.
                row_batch[b] = init_rows_list[b][-1]
        row_t = torch.as_tensor(row_batch, device=device)
        gdc.step_full(row_t)

    # Forecast loop.
    halted = [False] * B
    pred_actions_per_b = [[] for _ in range(B)]
    obs_in_train = 0; obs_total = 0
    for step in range(max_steps):
        # Build (B, k) obs tensor.
        obs_batch = np.full((B, gdc.k), -1, dtype=np.int64)
        for b in range(B):
            if halted[b]: continue
            obs = sims[b].current_obs()
            obs_batch[b, 0] = obs[0]; obs_batch[b, 1] = obs[1]
            obs_batch[b, 2] = obs[2]; obs_batch[b, 3] = obs[3]
            obs_batch[b, 4] = -1;     obs_batch[b, 5] = -1
            if train_obs_set is not None:
                obs_total += 1
                if obs in train_obs_set:
                    obs_in_train += 1
        if all(halted): break
        obs_t = torch.as_tensor(obs_batch, device=device)
        pat, parg = gdc.step_predict(obs_t, predict_cols=(4, 5))
        pat_np = pat.cpu().numpy(); parg_np = parg.cpu().numpy()
        for b in range(B):
            if halted[b]: continue
            pa = int(pat_np[b]); pg = int(parg_np[b])
            pred_actions_per_b[b].append((pa, pg))
            sims[b].apply(pa, pg)
            if pa == AT_HALT:
                halted[b] = True

    # Score.
    exact_match = []
    action_correct = 0; action_total = 0
    for b in range(B):
        a_b = pairs[b][0]; b_b = pairs[b][1]
        ok = (sims[b].decode_output() == a_b + b_b)
        exact_match.append(ok)
        L = min(len(pred_actions_per_b[b]), len(gt_actions[b]))
        for i in range(L):
            if pred_actions_per_b[b][i] == gt_actions[b][i]:
                action_correct += 1
        action_total += L

    return dict(
        exact_match=exact_match,
        action_correct=action_correct,
        action_total=action_total,
        obs_in_train=obs_in_train,
        obs_total=obs_total,
        time_sec=time.time() - t0,
    )


def run_smoke_torch():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}")
    rng = np.random.default_rng(42)
    n_train = 1200
    train_pairs = []
    for _ in range(n_train):
        da = int(rng.integers(1, 4)); db = int(rng.integers(1, 4))
        train_pairs.append((int(rng.integers(10**(da-1), 10**da)),
                             int(rng.integers(10**(db-1), 10**db))))
    print(f"Training pairs: {n_train}")
    train_traces = [generate_trace(a, b) for (a, b) in train_pairs]
    total_rows = sum(t.shape[0] for t in train_traces)
    print(f"Total chain rows: {total_rows}")

    t0 = time.time()
    gdc = NpiGDCTorch(train_traces, alpha=0.95, theta=0.05, beta=0.05,
                       device=device, dtype=torch.float64)
    print(f"GDC torch built: N={gdc.N}  ({time.time()-t0:.2f}s)")

    # Build training obs set for coverage stat.
    train_obs_set = set()
    for tr in train_traces:
        for row in tr:
            op1, op2, op3, op4 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            if not (op1 == BLANK and op2 == BLANK and op3 == BLANK and op4 == BLANK):
                train_obs_set.add((op1, op2, op3, op4))
    print(f"Distinct training 4-tape obs tuples: {len(train_obs_set)}")

    def n_digit_pair(lo, hi, n):
        return [(int(rng.integers(lo, hi)), int(rng.integers(lo, hi)))
                for _ in range(n)]
    eval_buckets = [
        ('train-len-1', n_digit_pair(0, 10, 25)),
        ('train-len-2', n_digit_pair(10, 100, 25)),
        ('train-len-3', n_digit_pair(100, 1000, 25)),
        ('len-4 OOD',   n_digit_pair(1000, 10000, 25)),
        ('len-5 OOD',   n_digit_pair(10000, 100000, 25)),
        ('len-7 OOD',   n_digit_pair(10**6, 10**7, 25)),
        ('len-10 OOD',  n_digit_pair(10**9, 10**10, 25)),
    ]
    for bucket_name, pairs in eval_buckets:
        res = forecast_bucket_torch(gdc, pairs, device,
                                     max_steps=400,
                                     train_obs_set=train_obs_set)
        n_correct = sum(res['exact_match'])
        rate = 100.0 * n_correct / len(pairs)
        act_rate = (100.0 * res['action_correct'] / res['action_total']
                    if res['action_total'] else 0.0)
        cov_rate = (100.0 * res['obs_in_train'] / res['obs_total']
                    if res['obs_total'] else 0.0)
        print(f"  [{bucket_name:>13s}]  exact={n_correct:>2d}/{len(pairs)} ({rate:>5.1f}%)  "
              f"action_acc={act_rate:>5.1f}%  obs_in_train={cov_rate:>5.1f}% "
              f"({res['obs_in_train']}/{res['obs_total']})  ({res['time_sec']:.1f}s)")


if __name__ == "__main__":
    run_smoke_torch()
