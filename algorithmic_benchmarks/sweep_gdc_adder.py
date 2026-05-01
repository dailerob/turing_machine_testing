"""GDC hyperparameter sweep on the binary-adder task, original vs no-read.

The fixed GDC config (alpha=0.99, theta=0.005, two-step) was chosen for
the original variant.  Under no-read the trace structure is different
(many emissions collapse), so the optimum may shift.  Sweep:

    alpha          : {0.50, 0.70, 0.90, 0.95, 0.99}
    theta          : {0.005, 0.05}
    transition     : {'self_loop', 'self_loop_two_step'}

CHMM-adder reference numbers from `run_benchmarks.py` v2:
    original  CHMM K=4  -> 10 errors / 72,217  (no perfect tapes)
    no-read   CHMM K=4  ->  0 errors / 72,217  (10/10 perfect)

Run:
    python algorithmic_benchmarks/sweep_gdc_adder.py
"""

from __future__ import annotations

import os, sys, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from _tm_common import apply_noread_to_runs  # noqa: E402
from generative_dense_chain import GenerativeDenseChain  # noqa: E402
from binary_alphabet_adder import (  # noqa: E402
    simulate_random_binary_alphabet_adders, BINARY_ALPHABET_ADDER)


def gdc_eval_tm_reduced(gdc, test_tapes):
    """Per-position eval, conditional on read.  Returns acc[3], total[3],
    tuple_errors, perfect_tapes."""
    correct = np.zeros(3, dtype=np.int64)
    total = np.zeros(3, dtype=np.int64)
    tuple_errors, perfect = 0, 0
    for tape in test_tapes:
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
            cond = np.array([actual_next[0], np.nan, np.nan])
            pred = gdc.greedy_sample(forecast, conditional=cond)
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


def main():
    N_TRAIN, N_TEST = 200, 10
    print(f"=== GDC hyperparameter sweep on binary adder "
          f"(N_TRAIN={N_TRAIN}, N_TEST={N_TEST}) ===", flush=True)

    print(f"\n[1/2] Generating data...", flush=True)
    tr = simulate_random_binary_alphabet_adders(
        n_runs=N_TRAIN, num_range=(0, 32), max_steps=200_000, seed=42)
    te = simulate_random_binary_alphabet_adders(
        n_runs=N_TEST, num_range=(0, 1000), max_steps=200_000, seed=123)
    print(f"  train n={N_TRAIN}, test n={N_TEST}", flush=True)

    # Build noread variant by re-using the same encodings
    merged_se = dict(tr['symbol_encoding'])
    for k in te['symbol_encoding']:
        if k not in merged_se:
            merged_se[k] = len(merged_se)
    merged_st = dict(tr['state_encoding'])
    for k in te['state_encoding']:
        if k not in merged_st:
            merged_st[k] = len(merged_st)
    tr_runs_nr, _ = apply_noread_to_runs(
        tr['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)
    te_runs_nr, _ = apply_noread_to_runs(
        te['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)

    only_variant = os.environ.get('GDC_VARIANT')
    all_variants = {
        'original': (tr['runs'], te['runs']),
        'noread':   (tr_runs_nr, te_runs_nr),
    }
    if only_variant:
        variants = {only_variant: all_variants[only_variant]}
    else:
        variants = all_variants

    ALPHAS = [0.50, 0.70, 0.90, 0.95, 0.99]
    THETAS = [0.005, 0.05]
    TRANSITIONS = ['self_loop', 'self_loop_two_step']

    rows = []
    print(f"\n[2/2] Sweeping {len(ALPHAS)} alphas x {len(THETAS)} thetas "
          f"x {len(TRANSITIONS)} transitions x 2 variants = "
          f"{len(ALPHAS)*len(THETAS)*len(TRANSITIONS)*2} runs", flush=True)

    for variant, (train_runs, test_runs) in variants.items():
        train_red = [t[t[:, 0] != -1][:, 1:4].astype(np.int64)
                     for t in train_runs]
        train_red = [t for t in train_red if len(t) > 0]
        n_pred_total = sum(max(len(t) - 1, 0) for t in test_runs
                           if (test_runs[0][:, 0] != -1).any())
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
                        gdc, test_runs)
                    eval_t = time.time() - t0
                    n_pred = int(total[0])
                    rate = 100 * terr / max(n_pred, 1)
                    print(f"  {variant:>9s}  alpha={alpha:.2f} "
                          f"theta={theta:.3f}  trans={trans:<19s}  "
                          f"states={gdc.n_states:>6d}  "
                          f"errors={terr:>5d}/{n_pred}  ({rate:5.2f}%)  "
                          f"perfect={perf}/{N_TEST}  "
                          f"[train={train_t:.1f}s eval={eval_t:.1f}s]",
                          flush=True)
                    rows.append(dict(
                        variant=variant, alpha=alpha, theta=theta,
                        transition=trans, n_states=gdc.n_states,
                        train_s=train_t, eval_s=eval_t,
                        acc_read=acc[0], acc_write=acc[1], acc_dir=acc[2],
                        mean_acc=acc.mean(),
                        tuple_errors=terr, n_predictions=n_pred,
                        perfect_tapes=perf, n_test=N_TEST))

    out_csv = os.path.join(HERE, 'gdc_adder_sweep.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out_csv}", flush=True)

    print(f"\n=== Best GDC config per variant ===", flush=True)
    for variant in variants:
        sub = [r for r in rows if r['variant'] == variant]
        sub.sort(key=lambda r: r['tuple_errors'])
        best = sub[0]
        print(f"  {variant:>9s}  best: alpha={best['alpha']:.2f} "
              f"theta={best['theta']:.3f} trans={best['transition']}  "
              f"errors={best['tuple_errors']}/{best['n_predictions']}  "
              f"perfect={best['perfect_tapes']}/{N_TEST}  "
              f"states={best['n_states']}", flush=True)


if __name__ == "__main__":
    main()
