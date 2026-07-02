"""Tuned GDC hyperparameter sweep on all TM tasks (incl. new ones).

Leakage-free protocol with stretched val:
  - Simulate n_train training runs from train_range (used to fit GDC).
  - Simulate n_val = ceil(n_train * 0.1) validation runs from a
    val_range that sits strictly between train_range and test_range
    (used to pick the best (alpha, theta) per task). Stretched val
    discriminates among configs on length-extrapolation, which the
    original train-distribution val could not (configs tied at 0).
  - Simulate n_test test runs from test_range (used only to score the
    chosen config — never inspected during selection).
  - Tie-break: when multiple configs achieve the same val error, pick
    highest alpha then lowest theta (most "exact-prefix" config).

Mirrors the convention of `sweep_gdc_adder.py` and TUNED_GDC_RESULTS.md
but uses the torch GDC kernel for speed. Limited to transition_type=
self_loop because that's what the torch kernel supports; the canonical
TUNED table also tries self_loop_two_step. So tasks where the canonical
optimum was 2-step (parity, binary_adder-noread, increment-tied) may
land slightly worse here; tasks where the canonical optimum was
self_loop (reverse, dyck1) should match exactly.

Sweep grid:
  alpha ∈ {0.50, 0.70, 0.90, 0.95, 0.99}
  theta ∈ {0.005, 0.05}
  transition fixed = self_loop
  beta fixed = 0.0

We skip configs with alpha + theta > 1.
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# HERE first to win against hmm_comparison/parrot_eval.py
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from torch_tm_adapters import TorchTMGDC
from run_benchmarks import (reduced_alphabet, encode_reduced,
                             encode_reduced_for_torch,
                             torch_gdc_eval_tm_reduced)
from _tm_task_config import TM_TASKS, TASK_ORDER, simulate_train_val_test, SUFFIX

ALPHAS = [0.50, 0.70, 0.90, 0.95, 0.99]
THETAS = [0.005, 0.05]


def configs():
    """(alpha, theta, alpha_fc) configs. Single-alpha (alpha_fc == alpha)
    is the original grid; dual-alpha adds a deterministic prediction step
    (alpha_fc = 1.0) on top of each alpha < 1 context config."""
    out = []
    for a in ALPHAS:
        for t in THETAS:
            if a + t > 1.0:
                continue
            out.append((a, t, a))              # single-alpha
            if a < 1.0:
                out.append((a, t, 1.0))        # dual-alpha (alpha_fc=1.0)
    return out


def pick_best(rows):
    """Pick best config by val_tuple_errors; tie-break by -alpha then theta
    then prefer single-alpha (alpha_fc == alpha) over dual on exact ties."""
    return sorted(rows,
                  key=lambda r: (r['val_tuple_errors'], -r['alpha'], r['theta'],
                                 0 if r['alpha_fc'] == r['alpha'] else 1))[0]


def sweep_task(name, log, variant='original'):
    cfg = TM_TASKS[name]
    n_test, n_val = cfg['n_test'], cfg['n_val']
    tr_runs, val_runs, te_runs = simulate_train_val_test(name, variant)
    tuple_to_id, id_to_tuple = reduced_alphabet(tr_runs)
    nA = len(id_to_tuple)
    train_red = [encode_reduced_for_torch(t, tuple_to_id) for t in tr_runs]
    train_red = [s for s in train_red if len(s) > 0]
    log(f"{name}/{variant}: nA={nA}, n_val={n_val}", end='')

    rows = []
    for alpha, theta, alpha_fc in configs():
        gdc = TorchTMGDC(alpha=alpha, theta=theta, beta=0.0,
                         transition_type='self_loop',
                         initial_dist='sequence_starts',
                         terminal_behavior='diffuse',
                         alpha_fc=(None if alpha_fc == alpha else alpha_fc),
                         theta_fc=(None if alpha_fc == alpha else 0.0))
        gdc.fit(train_red, alphabet_size=nA)
        v_acc, v_total, v_terr, v_perf = torch_gdc_eval_tm_reduced(
            gdc, val_runs, tuple_to_id, id_to_tuple)
        t_acc, t_total, t_terr, t_perf = torch_gdc_eval_tm_reduced(
            gdc, te_runs, tuple_to_id, id_to_tuple)
        rows.append(dict(task=name, variant=variant,
                          alpha=alpha, theta=theta, alpha_fc=alpha_fc,
                          val_tuple_errors=v_terr,
                          val_n_predictions=int(v_total[0]),
                          val_perfect_tapes=v_perf, n_val=n_val,
                          tuple_errors=t_terr,
                          n_predictions=int(t_total[0]),
                          perfect_tapes=t_perf, n_test=n_test,
                          mean_acc=float(t_acc.mean())))
    best = pick_best(rows)
    afc_tag = '' if best['alpha_fc'] == best['alpha'] else f", afc={best['alpha_fc']}"
    log(f"  picked (a={best['alpha']}, t={best['theta']}{afc_tag}): "
        f"val={best['val_tuple_errors']}/{best['val_n_predictions']}, "
        f"test={best['tuple_errors']}/{best['n_predictions']} errors, "
        f"{best['perfect_tapes']}/{n_test} perfect")
    for r in rows:
        r['picked'] = (r['alpha'] == best['alpha'] and r['theta'] == best['theta']
                       and r['alpha_fc'] == best['alpha_fc'])
    return rows


def main():
    out_csv = os.path.join(HERE, f'tuned_gdc_sweep{SUFFIX}.csv')
    log_lines = []
    def log(msg='', end='\n'):
        s = str(msg) + end
        print(s, end='', flush=True); log_lines.append(s.rstrip('\n'))

    rows_all = []
    log(f"=== Tuned GDC sweep (transition=self_loop, val-tuned, stretched val) ===")
    log(f"  {len(configs())} configs per task: a × t = "
        f"{ALPHAS} × {THETAS} (skip a+t>1)")
    log(f"  Val drawn from val_range strictly between train and test\n")
    t0 = time.time()
    for variant in ('original', 'noread'):
        for name in TASK_ORDER:
            rows_all.extend(sweep_task(name, log, variant))
    log(f"\nTotal sweep: {time.time()-t0:.0f}s")

    fields = ['task', 'variant', 'alpha', 'theta', 'alpha_fc', 'picked',
              'val_tuple_errors', 'val_n_predictions', 'val_perfect_tapes', 'n_val',
              'tuple_errors', 'n_predictions',
              'perfect_tapes', 'n_test', 'mean_acc']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows_all)
    log(f"Wrote {out_csv}")
    log_path = out_csv.replace('.csv', '.log')
    with open(log_path, 'w') as f:
        f.write(''.join(log_lines))
    print(f"Wrote {log_path}", flush=True)


if __name__ == "__main__":
    main()
