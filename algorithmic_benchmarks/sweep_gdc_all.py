"""GDC hyperparameter sweep on parity, increment, reverse, dyck1.

Mirrors `sweep_gdc_adder.py` but generalised to all four algorithmic
tasks.  For each task we generate train/test once (per variant) then
sweep:

    alpha      : {0.50, 0.70, 0.90, 0.95, 0.99}
    theta      : {0.005, 0.05}
    transition : {'self_loop', 'self_loop_two_step'}

Skipping (alpha + theta > 1).

For TM tasks: both `original` and `noread` variants.
For dyck1: single (no-variant) sweep.

Per-task sizing matches the values in `run_benchmarks.py`.

Run:
    python algorithmic_benchmarks/sweep_gdc_all.py
"""

from __future__ import annotations
import os, sys, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import parity_tm, increment_tm, reverse_tm, dyck1  # noqa: E402
from generative_dense_chain import GenerativeDenseChain  # noqa: E402
from run_benchmarks import (  # noqa: E402
    gdc_eval_tm_reduced, gdc_eval_dyck)

ALPHAS = [0.50, 0.70, 0.90, 0.95, 0.99]
THETAS = [0.005, 0.05]
TRANSITIONS = ['self_loop', 'self_loop_two_step']

TM_TASKS = [
    ('parity',    parity_tm,    (3, 8),  (16, 32), 300, 20, 200),
    ('increment', increment_tm, (1, 5),  (8, 12),  300, 20, 200),
    ('reverse',   reverse_tm,   (3, 6),  (10, 16), 300, 20, 10000),
]


def sweep_tm_task(name, module, train_range, test_range, n_train, n_test,
                  max_steps, log):
    rows = []
    for variant in ('original', 'noread'):
        nr = (variant == 'noread')
        tr = module.simulate(n_train, train_range, max_steps=max_steps,
                             seed=42, noread=nr)
        te = module.simulate(n_test, test_range, max_steps=max_steps * 4,
                             seed=123, noread=nr)
        train_red = [t[t[:, 0] != -1][:, 1:4].astype(np.int64)
                     for t in tr['runs']]
        train_red = [t for t in train_red if len(t) > 0]
        log(f"\n--- {name} / {variant} (n_states will be the same; "
            f"sweeping smoothing) ---")
        for alpha in ALPHAS:
            for theta in THETAS:
                if alpha + theta > 1.0:
                    continue
                for trans in TRANSITIONS:
                    t0 = time.time()
                    gdc = GenerativeDenseChain(
                        train_red, alpha=alpha, theta=theta, gamma=0.0,
                        transition_type=trans,
                        initial_dist='sequence_starts')
                    train_t = time.time() - t0
                    t0 = time.time()
                    acc, total, terr, perf = gdc_eval_tm_reduced(
                        gdc, te['runs'])
                    eval_t = time.time() - t0
                    n_pred = int(total[0])
                    rate = 100 * terr / max(n_pred, 1)
                    log(f"  alpha={alpha:.2f} theta={theta:.3f} "
                        f"trans={trans:<19s}  states={gdc.n_states:>5d}  "
                        f"err={terr:>4d}/{n_pred} ({rate:5.2f}%)  "
                        f"perf={perf}/{n_test}")
                    rows.append(dict(
                        task=name, variant=variant, alpha=alpha,
                        theta=theta, transition=trans,
                        n_states=gdc.n_states, train_s=train_t,
                        eval_s=eval_t, acc_read=acc[0],
                        acc_write=acc[1], acc_dir=acc[2],
                        mean_acc=acc.mean(),
                        tuple_errors=terr, n_predictions=n_pred,
                        perfect_tapes=perf, n_test=n_test))
    return rows


def sweep_dyck(log):
    name = 'dyck1'
    rows = []
    tr = dyck1.simulate(1000, max_depth=4, length_min=4, length_max=200,
                        seed=42)
    te = dyck1.simulate(200, max_depth=8, length_min=4, length_max=400,
                        seed=123)
    train_seqs = [s.reshape(-1, 1).astype(np.int64) for s in tr['sequences']]
    log(f"\n--- {name} (sequence; no variant) ---")
    for alpha in ALPHAS:
        for theta in THETAS:
            if alpha + theta > 1.0:
                continue
            for trans in TRANSITIONS:
                t0 = time.time()
                gdc = GenerativeDenseChain(
                    train_seqs, alpha=alpha, theta=theta, gamma=0.0,
                    transition_type=trans,
                    initial_dist='sequence_starts')
                train_t = time.time() - t0
                t0 = time.time()
                acc, total, correct = gdc_eval_dyck(gdc, te['sequences'])
                eval_t = time.time() - t0
                log(f"  alpha={alpha:.2f} theta={theta:.3f} "
                    f"trans={trans:<19s}  states={gdc.n_states:>5d}  "
                    f"acc={acc:.4f} ({correct}/{total})")
                rows.append(dict(
                    task=name, variant='n/a', alpha=alpha, theta=theta,
                    transition=trans, n_states=gdc.n_states,
                    train_s=train_t, eval_s=eval_t,
                    acc_read=np.nan, acc_write=np.nan, acc_dir=np.nan,
                    mean_acc=acc, tuple_errors=total - correct,
                    n_predictions=total, perfect_tapes=-1, n_test=200))
    return rows


def main():
    log_lines = []
    def log(msg=""):
        print(msg, flush=True); log_lines.append(str(msg))

    log(f"=== GDC hyperparameter sweep across algorithmic benchmarks ===")
    log(f"alphas={ALPHAS}, thetas={THETAS}, transitions={TRANSITIONS}")

    rows = []
    for task_args in TM_TASKS:
        log(f"\n{'='*66}\nTASK: {task_args[0]}\n{'='*66}")
        rows += sweep_tm_task(*task_args, log=log)
    log(f"\n{'='*66}\nTASK: dyck1\n{'='*66}")
    rows += sweep_dyck(log=log)

    out_csv = os.path.join(HERE, 'gdc_all_sweep.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log(f"\nWrote {out_csv}")

    log(f"\n{'='*70}\nBEST GDC CONFIG PER (TASK, VARIANT)\n{'='*70}")
    log(f"{'task':>10s}  {'variant':>8s}  {'alpha':>5s}  {'theta':>5s}  "
        f"{'trans':>20s}  {'errors':>14s}  {'perfect':>7s}")
    seen = set()
    for r in rows:
        seen.add((r['task'], r['variant']))
    for task, variant in sorted(seen):
        sub = [r for r in rows if r['task'] == task
               and r['variant'] == variant]
        sub.sort(key=lambda r: (r['tuple_errors'], -r['mean_acc']))
        b = sub[0]
        terr_str = f"{b['tuple_errors']}/{b['n_predictions']}"
        perf_str = (f"{b['perfect_tapes']}/{b['n_test']}"
                    if b['perfect_tapes'] >= 0 else '   -   ')
        log(f"{task:>10s}  {variant:>8s}  {b['alpha']:>5.2f}  "
            f"{b['theta']:>5.3f}  {b['transition']:>20s}  "
            f"{terr_str:>14s}  {perf_str:>7s}")

    log_path = os.path.join(HERE, 'gdc_all_sweep.log')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"Wrote {log_path}", flush=True)


if __name__ == "__main__":
    main()
