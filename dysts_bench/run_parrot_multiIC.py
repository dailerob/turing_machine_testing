"""Parrot baseline on dysts univariate with fixed-fit-len multi-IC val.

Same Parrot 16-variant grid as `run_parrot_gdc.py` (mode × L × k) but the
val signal is now averaged across 3 sliding val windows within the train
trajectory, each using a fit_len of 90 (held constant across windows so
the state-space size is identical across val replicates).

Test eval unchanged: fit on test[:150], score 30-step forecast on
test[150:179].

Output: results/parrot_dysts_multiIC3_fitlen90.csv
"""
from __future__ import annotations
import os, sys, json, csv, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'skolr_bench', 'forecast'))

from parrot_torch import forecast_many_parrot, forecast_many_parrot_diff

DATA_DIR   = os.path.join(HERE, 'data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.json')
TEST_PATH  = os.path.join(DATA_DIR, 'test.json')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE  = torch.float32

TRAIN_FIT_LEN  = 150
TEST_TRUTH_LEN = 29
PRED_LEN       = 30

# Multi-IC val: 3 windows at fixed fit_len=90
VAL_FIT_LEN = 90
VAL_STARTS  = [90, 120, 150]

PARROT_VARIANTS = []
for L in (2, 4, 8, 16):
    for k in (1, 5):
        PARROT_VARIANTS.append(('raw',  L, k))
        PARROT_VARIANTS.append(('diff', L, k))


def smape(t, p):
    t = np.asarray(t, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    denom = (np.abs(t) + np.abs(p)) / 2.0
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return float(100 * np.mean(np.abs(t - p) / denom))


def mse(t, p):
    return float(np.mean((np.asarray(t, dtype=np.float64) -
                          np.asarray(p, dtype=np.float64)) ** 2))


def mae(t, p):
    return float(np.mean(np.abs(np.asarray(t, dtype=np.float64) -
                                np.asarray(p, dtype=np.float64))))


def parrot_forecast(state_series, prime, T, mode, L, k):
    state_series = np.asarray(state_series, dtype=np.float64)
    if mode == 'raw':
        if len(prime) < L:
            return np.full(T, prime[-1])
        primes = np.asarray(prime[-L:], dtype=np.float64).reshape(1, L)
        fc = forecast_many_parrot(state_series, primes, T, k=k,
                                   device=DEVICE, dtype=DTYPE)
    else:
        if len(prime) < L + 1:
            return np.full(T, prime[-1])
        primes_d = np.asarray(prime[-(L + 1):], dtype=np.float64).reshape(1, L + 1)
        fc = forecast_many_parrot_diff(state_series, primes_d, T, k=k,
                                        device=DEVICE, dtype=DTYPE)
    return fc.detach().cpu().numpy().ravel().astype(np.float64)


def run_one_system(system, train_traj, test_traj):
    train = np.asarray(train_traj, dtype=np.float64)
    test  = np.asarray(test_traj,  dtype=np.float64)
    if len(train) < TRAIN_FIT_LEN + 1 or len(test) < TRAIN_FIT_LEN + TEST_TRUTH_LEN:
        return None

    # Build (fit, val_target) pairs for each val window (fixed fit_len).
    val_specs = []
    for vs in VAL_STARTS:
        fit_start = vs - VAL_FIT_LEN
        if fit_start < 0:
            continue
        fit_w = train[fit_start:vs]
        target = train[vs:vs + TEST_TRUTH_LEN]
        if len(fit_w) < VAL_FIT_LEN or len(target) < TEST_TRUTH_LEN:
            continue
        val_specs.append((fit_w, target))
    if not val_specs:
        return None

    # Average val sMAPE across windows per config.
    parrot_val = []
    for mode, L, k in PARROT_VARIANTS:
        sms = []
        try:
            for fit_w, target in val_specs:
                fc = parrot_forecast(fit_w, fit_w, T=PRED_LEN,
                                      mode=mode, L=L, k=k)
                sms.append(smape(target, fc[:TEST_TRUTH_LEN]))
            if sms:
                parrot_val.append((float(np.mean(sms)), mode, L, k))
        except Exception:
            continue
    parrot_val.sort(key=lambda x: x[0])
    if not parrot_val:
        return None
    p_val_sm, p_mode, p_L, p_k = parrot_val[0]

    # Test eval: original protocol (fit on test[:150]).
    test_fit = test[:TRAIN_FIT_LEN]
    test_truth = test[TRAIN_FIT_LEN:TRAIN_FIT_LEN + TEST_TRUTH_LEN]
    p_fc = parrot_forecast(test_fit, test_fit, T=PRED_LEN,
                            mode=p_mode, L=p_L, k=p_k)[:TEST_TRUTH_LEN]
    return dict(
        system=system,
        parrot_smape=smape(test_truth, p_fc),
        parrot_mse=mse(test_truth, p_fc),
        parrot_mae=mae(test_truth, p_fc),
        parrot_pick=f'{p_mode}_L{p_L}_k{p_k}',
        parrot_val_smape=p_val_sm,
    )


def main():
    print(f"=== Parrot on dysts univariate (multi-IC val) ===")
    print(f"Device: {DEVICE}  dtype={DTYPE}")
    print(f"Val starts: {VAL_STARTS}  fit_len={VAL_FIT_LEN}  "
          f"({len(VAL_STARTS)} windows averaged)")
    print(f"Parrot variants: {len(PARROT_VARIANTS)}")
    train_data = json.load(open(TRAIN_PATH))
    test_data  = json.load(open(TEST_PATH))
    systems = sorted(set(train_data) & set(test_data))
    print(f"Systems: {len(systems)}")

    rows = []
    t0 = time.time()
    for i, sys_name in enumerate(systems):
        try:
            train_v = (train_data[sys_name].get('values')
                       if isinstance(train_data[sys_name], dict)
                       else train_data[sys_name])
            test_v  = (test_data[sys_name].get('values')
                       if isinstance(test_data[sys_name], dict)
                       else test_data[sys_name])
            r = run_one_system(sys_name, train_v, test_v)
            if r is not None:
                rows.append(r)
        except Exception as e:
            print(f"  [skip] {sys_name}: {e}")
        if (i + 1) % 20 == 0 or i == len(systems) - 1:
            print(f"  {i+1}/{len(systems)} systems  [{time.time()-t0:.1f}s]")
    print(f"Total: {time.time() - t0:.1f}s, {len(rows)} systems scored")

    out = os.path.join(HERE, 'results', 'parrot_dysts_multiIC3_fitlen90.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = ['system', 'parrot_smape', 'parrot_mse', 'parrot_mae',
              'parrot_pick', 'parrot_val_smape']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}")

    sm = np.array([r['parrot_smape'] for r in rows
                    if np.isfinite(r['parrot_smape'])])
    print(f"\nParrot (multi-IC val): median={np.median(sm):.2f}  mean={sm.mean():.2f}  "
          f"n={len(sm)}")


if __name__ == "__main__":
    main()
