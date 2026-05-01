"""Sweep eta (soft emission noise) for GDC on the algorithmic
benchmarks at the per-task tuned hyperparameters from
`TUNED_GDC_RESULTS.md`.

For each (task, variant) we use the best (alpha, theta, transition)
discovered in the GDC sweep, then sweep eta in {0.0, 0.01, 0.05, 0.1,
0.2, 0.3}.  eta=0 reproduces the standard hard-emission GDC.
"""

from __future__ import annotations
import os, sys, time, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import parity_tm, increment_tm, reverse_tm  # noqa: E402
from _tm_common import apply_noread_to_runs  # noqa: E402
from binary_alphabet_adder import (  # noqa: E402
    simulate_random_binary_alphabet_adders, BINARY_ALPHABET_ADDER)
from generative_dense_chain import GenerativeDenseChain  # noqa: E402
from soft_gdc import SoftEmissionGDC  # noqa: E402

ETA_GRID = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]

# Tuned hyperparameters (from gdc_all_sweep.csv) per task / variant.
# Keys: (task, variant) -> (alpha, theta, transition_type)
TUNED = {
    ('parity', 'original'):     (0.50, 0.05, 'self_loop_two_step'),
    ('parity', 'noread'):       (0.50, 0.05, 'self_loop_two_step'),
    ('increment', 'original'):  (0.50, 0.005, 'self_loop'),
    ('increment', 'noread'):    (0.50, 0.005, 'self_loop'),
    ('reverse', 'original'):    (0.95, 0.05, 'self_loop'),
    ('reverse', 'noread'):      (0.95, 0.05, 'self_loop'),
    ('binary_adder', 'original'): (0.50, 0.005, 'self_loop'),
    ('binary_adder', 'noread'):   (0.90, 0.05, 'self_loop_two_step'),
}


def evaluate_soft(soft, test_runs, n_test):
    """Per-position accuracy with soft conditioning + soft prediction."""
    correct = np.zeros(3, dtype=np.int64)
    total = np.zeros(3, dtype=np.int64)
    tuple_errors, perfect = 0, 0
    gdc = soft.gdc
    for tape in test_runs:
        if len(tape) < 2:
            perfect += 1; continue
        valid = tape[:, 0] != -1
        tape_red = tape[valid][:, 1:4].astype(np.int64)
        if len(tape_red) < 2:
            perfect += 1; continue
        _, hist = gdc.forward_pass(tape_red, return_history=True)
        tape_err = 0
        for t in range(len(tape_red) - 1):
            forecast = gdc.forecast(hist[t], n_steps=1)
            actual_next = tape_red[t + 1]
            cond = np.array([float(actual_next[0]), np.nan, np.nan])
            pred = soft.predict(forecast, cond)
            mismatch = False
            for pos in range(3):
                if not np.isnan(pred[pos]):
                    total[pos] += 1
                    if int(pred[pos]) == int(actual_next[pos]):
                        correct[pos] += 1
                    else:
                        mismatch = True
            if mismatch:
                tape_err += 1; tuple_errors += 1
        if tape_err == 0:
            perfect += 1
    acc = correct / np.maximum(total, 1)
    return acc, total, tuple_errors, perfect


def run_tm(name, module, train_range, test_range,
           n_train, n_test, max_steps, log, variant):
    nr = (variant == 'noread')
    log(f"\n{'='*66}\nSoft-GDC on {name} ({variant})\n{'='*66}")
    alpha, theta, trans = TUNED[(name, variant)]
    log(f"  tuned config: alpha={alpha}, theta={theta}, trans={trans}")
    tr = module.simulate(n_train, train_range, max_steps=max_steps,
                         seed=42, noread=nr)
    te = module.simulate(n_test, test_range, max_steps=max_steps * 4,
                         seed=123, noread=nr)
    train_red = [t[t[:, 0] != -1][:, 1:4].astype(np.int64) for t in tr['runs']]
    train_red = [t for t in train_red if len(t) > 0]
    t0 = time.time()
    gdc = GenerativeDenseChain(
        train_red, alpha=alpha, theta=theta, gamma=0.0, beta=0.0,
        transition_type=trans, initial_dist='sequence_starts')
    train_t = time.time() - t0
    log(f"  GDC built: {gdc.n_states} states (train={train_t:.2f}s)")

    rows = []
    for eta in ETA_GRID:
        soft = SoftEmissionGDC(gdc, eta=eta)
        t0 = time.time()
        acc, total, terr, perf = evaluate_soft(soft, te['runs'], n_test)
        eval_t = time.time() - t0
        n_pred = int(total[0])
        log(f"  eta={eta:>4}: read={acc[0]:.4f} write={acc[1]:.4f} "
            f"dir={acc[2]:.4f} mean={acc.mean():.4f}  "
            f"errors={terr}/{n_pred} ({100*terr/max(n_pred,1):.3f}%)  "
            f"perfect={perf}/{n_test}  [eval={eval_t:.1f}s]")
        rows.append(dict(task=name, variant=variant, eta=eta,
                         alpha=alpha, theta=theta, transition=trans,
                         n_states=gdc.n_states,
                         eval_s=eval_t,
                         acc_read=acc[0], acc_write=acc[1], acc_dir=acc[2],
                         mean_acc=acc.mean(),
                         tuple_errors=terr, n_predictions=n_pred,
                         perfect_tapes=perf, n_test=n_test))
    return rows


def run_binary_adder(log, variant, n_train=200, n_test=10):
    name = 'binary_adder'
    log(f"\n{'='*66}\nSoft-GDC on {name} ({variant})\n{'='*66}")
    alpha, theta, trans = TUNED[(name, variant)]
    log(f"  tuned config: alpha={alpha}, theta={theta}, trans={trans}")
    tr = simulate_random_binary_alphabet_adders(
        n_runs=n_train, num_range=(0, 32), max_steps=200_000, seed=42)
    te = simulate_random_binary_alphabet_adders(
        n_runs=n_test, num_range=(0, 1000), max_steps=200_000, seed=123)
    if variant == 'noread':
        merged_se = dict(tr['symbol_encoding'])
        for k in te['symbol_encoding']:
            if k not in merged_se: merged_se[k] = len(merged_se)
        merged_st = dict(tr['state_encoding'])
        for k in te['state_encoding']:
            if k not in merged_st: merged_st[k] = len(merged_st)
        tr['runs'], _ = apply_noread_to_runs(
            tr['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)
        te['runs'], _ = apply_noread_to_runs(
            te['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)
    train_red = [t[t[:, 0] != -1][:, 1:4].astype(np.int64) for t in tr['runs']]
    train_red = [t for t in train_red if len(t) > 0]
    t0 = time.time()
    gdc = GenerativeDenseChain(
        train_red, alpha=alpha, theta=theta, gamma=0.0, beta=0.0,
        transition_type=trans, initial_dist='sequence_starts')
    train_t = time.time() - t0
    log(f"  GDC built: {gdc.n_states} states (train={train_t:.2f}s)")

    rows = []
    for eta in ETA_GRID:
        soft = SoftEmissionGDC(gdc, eta=eta)
        t0 = time.time()
        acc, total, terr, perf = evaluate_soft(soft, te['runs'], n_test)
        eval_t = time.time() - t0
        n_pred = int(total[0])
        log(f"  eta={eta:>4}: read={acc[0]:.4f} write={acc[1]:.4f} "
            f"dir={acc[2]:.4f} mean={acc.mean():.4f}  "
            f"errors={terr}/{n_pred} ({100*terr/max(n_pred,1):.3f}%)  "
            f"perfect={perf}/{n_test}  [eval={eval_t:.1f}s]")
        rows.append(dict(task=name, variant=variant, eta=eta,
                         alpha=alpha, theta=theta, transition=trans,
                         n_states=gdc.n_states, eval_s=eval_t,
                         acc_read=acc[0], acc_write=acc[1], acc_dir=acc[2],
                         mean_acc=acc.mean(), tuple_errors=terr,
                         n_predictions=n_pred,
                         perfect_tapes=perf, n_test=n_test))
    return rows


def main():
    log_lines = []
    def log(msg=""): print(msg, flush=True); log_lines.append(str(msg))

    log("=== Soft-emission GDC sweep on algorithmic benchmarks ===")
    rows = []
    for variant in ('original', 'noread'):
        rows += run_tm('parity', parity_tm, train_range=(3, 8),
                       test_range=(16, 32), n_train=300, n_test=20,
                       max_steps=200, log=log, variant=variant)
        rows += run_tm('increment', increment_tm, train_range=(1, 5),
                       test_range=(8, 12), n_train=300, n_test=20,
                       max_steps=200, log=log, variant=variant)
        rows += run_tm('reverse', reverse_tm, train_range=(3, 6),
                       test_range=(10, 16), n_train=300, n_test=20,
                       max_steps=10000, log=log, variant=variant)
        rows += run_binary_adder(log=log, variant=variant,
                                 n_train=200, n_test=10)

    out_csv = os.path.join(HERE, 'soft_gdc_results.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log(f"\nWrote {out_csv}")

    log("\n=== Soft-GDC SUMMARY (best eta per task-variant) ===")
    log(f"{'task':>14s}  {'variant':>8s}  {'eta':>5s}  "
        f"{'mean_acc':>9s}  {'errors':>14s}  {'perfect':>8s}")
    from collections import defaultdict
    by_tv = defaultdict(list)
    for r in rows: by_tv[(r['task'], r['variant'])].append(r)
    for (task, variant), rs in by_tv.items():
        # Best eta = min tuple_errors, ties broken by max perfect
        best = min(rs, key=lambda r: (r['tuple_errors'], -r['perfect_tapes']))
        terr_str = f"{best['tuple_errors']}/{best['n_predictions']}"
        log(f"{task:>14s}  {variant:>8s}  {best['eta']:>5}  "
            f"{best['mean_acc']:>9.4f}  {terr_str:>14s}  "
            f"{best['perfect_tapes']}/{best['n_test']}")

    log_path = os.path.join(HERE, 'soft_gdc.log')
    with open(log_path, 'w') as f: f.write('\n'.join(log_lines))


if __name__ == "__main__":
    main()
