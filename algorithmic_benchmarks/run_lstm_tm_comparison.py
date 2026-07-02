"""Compare TorchTMLSTM vs TorchTMGDC across several TM tasks at 4x training.

Tests three of GDC's "headline 0-error" tasks (reverse, binary_adder,
subtraction) + parity (where GDC is competitive but not dominant). All
at the noread variant, which is where GDC's chain-position-matching
advantage is most pronounced.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

# Force 4x training budget.
os.environ['TM_TRAIN_MULT'] = '4'

import torch                                                                  # noqa: E402

from _tm_task_config import simulate_train_val_test, TM_TASKS                 # noqa: E402
from torch_tm_adapters import TorchTMGDC                                       # noqa: E402
from torch_tm_lstm import TorchTMLSTM                                          # noqa: E402


def reduced_alphabet(runs):
    seen = set()
    for arr in runs:
        for row in arr:
            if int(row[0]) == -1: continue
            seen.add((int(row[1]), int(row[2]), int(row[3])))
    id_to_tuple = sorted(seen)
    return {t: i for i, t in enumerate(id_to_tuple)}, id_to_tuple


def encode_reduced_for_torch(arr, tuple_to_id):
    out = []
    for row in arr:
        if int(row[0]) == -1: continue
        key = (int(row[1]), int(row[2]), int(row[3]))
        if key in tuple_to_id:
            out.append(tuple_to_id[key])
    return np.asarray(out, dtype=np.int64)


def eval_model(model, test_tapes, tuple_to_id, id_to_tuple):
    by_read = {}
    for tid, tup in enumerate(id_to_tuple):
        by_read.setdefault(tup[0], []).append(tid)
    encoded = [encode_reduced_for_torch(t, tuple_to_id) for t in test_tapes]
    valid = [(i, x) for i, x in enumerate(encoded) if len(x) >= 2]
    perfect = sum(1 for x in encoded if len(x) < 2)
    if not valid:
        return 0, 0, perfect
    _, xs = zip(*valid)
    actuals_per_tape = [
        [id_to_tuple[int(x[t + 1])][0] for t in range(len(x) - 1)]
        for x in xs]
    all_preds = model.score_tapes_batched(list(xs), actuals_per_tape, by_read)
    tuple_errors = 0; total = 0
    for x, preds in zip(xs, all_preds):
        tape_err = 0
        for t in range(len(x) - 1):
            actual = id_to_tuple[int(x[t + 1])]
            pred = id_to_tuple[int(preds[t])]
            total += 1
            if pred != actual:
                tape_err += 1; tuple_errors += 1
        if tape_err == 0:
            perfect += 1
    return tuple_errors, total, perfect


def run_one_task(task_name: str, variant: str, device: str,
                  lstm_epochs: int = 60):
    print(f"\n{'='*72}\nTASK: {task_name} ({variant})\n{'='*72}")
    tr_runs, _, te_runs = simulate_train_val_test(task_name, variant)
    n_train, n_test = len(tr_runs), len(te_runs)
    train_lens = [t.shape[0] for t in tr_runs]
    test_lens = [t.shape[0] for t in te_runs]
    print(f"  n_train={n_train}, n_test={n_test}")
    print(f"  train trace lens: min={min(train_lens)}, max={max(train_lens)}, "
          f"mean={np.mean(train_lens):.0f}")
    print(f"  test trace lens:  min={min(test_lens)}, max={max(test_lens)}, "
          f"mean={np.mean(test_lens):.0f}  "
          f"(OOD ratio: {np.mean(test_lens)/np.mean(train_lens):.1f}x)")

    tuple_to_id, id_to_tuple = reduced_alphabet(tr_runs)
    nA = len(id_to_tuple)
    train_red = [encode_reduced_for_torch(t, tuple_to_id) for t in tr_runs]
    train_red = [s for s in train_red if len(s) > 0]
    total_train_tokens = sum(len(s) for s in train_red)
    print(f"  reduced alphabet size: {nA}   "
          f"total training tokens: {total_train_tokens}")

    # GDC — canonical config from run_benchmarks.py GDC_PARAMS:
    # alpha=0.99, theta=0.005, terminal='diffuse', initial='sequence_starts'.
    t0 = time.time()
    gdc = TorchTMGDC(alpha=0.99, theta=0.005, beta=0.0,
                      transition_type='self_loop',
                      initial_dist='sequence_starts',
                      terminal_behavior='diffuse',
                      device=device)
    gdc.fit(train_red, alphabet_size=nA)
    t_gdc_fit = time.time() - t0
    t0 = time.time()
    err_gdc, tot_gdc, perf_gdc = eval_model(gdc, te_runs, tuple_to_id, id_to_tuple)
    t_gdc_eval = time.time() - t0
    print(f"  GDC:  {err_gdc:>6d}/{tot_gdc:<6d} errors ({100*err_gdc/max(tot_gdc,1):.3f}%), "
          f"perfect={perf_gdc:>2d}/{n_test:<2d}  "
          f"(fit={t_gdc_fit:.1f}s, eval={t_gdc_eval:.1f}s)")

    # LSTM
    print(f"  Training LSTM ({lstm_epochs} epochs)...")
    lstm = TorchTMLSTM(hidden=256, n_layers=2, emb_dim=64,
                        lr=1e-3, n_epochs=lstm_epochs, batch_size=32,
                        device=device)
    lstm.fit(train_red, alphabet_size=nA, verbose=False)
    t0 = time.time()
    err_lstm, tot_lstm, perf_lstm = eval_model(lstm, te_runs, tuple_to_id, id_to_tuple)
    t_lstm_eval = time.time() - t0
    print(f"  LSTM: {err_lstm:>6d}/{tot_lstm:<6d} errors ({100*err_lstm/max(tot_lstm,1):.3f}%), "
          f"perfect={perf_lstm:>2d}/{n_test:<2d}  "
          f"(eval={t_lstm_eval:.1f}s)")

    return dict(task=task_name, variant=variant,
                err_gdc=err_gdc, total=tot_gdc, perfect_gdc=perf_gdc,
                err_lstm=err_lstm, perfect_lstm=perf_lstm, n_test=n_test,
                ood_ratio=np.mean(test_lens)/np.mean(train_lens))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}, TM_TRAIN_MULT={os.environ['TM_TRAIN_MULT']}")
    print("Note: matches the 4x training budget headlines in CLAUDE.md")

    task_order = ['parity', 'increment', 'reverse', 'binary_adder',
                   'shift_left', 'bit_count_mod3', 'anbn',
                   'palindrome', 'subtraction']
    variants = ['original', 'noread']

    results = []
    for tname in task_order:
        for variant in variants:
            r = run_one_task(tname, variant, device, lstm_epochs=60)
            results.append(r)

    print(f"\n{'='*72}")
    print(f"=== Summary: LSTM vs GDC on all 18 TM cells (4x training) ===")
    print(f"{'='*72}")
    print(f"{'task':>14s}  {'var':>4s}  {'OOD':>5s}  "
          f"{'GDC err':>12s}  {'LSTM err':>12s}  "
          f"{'GDC perf':>9s}  {'LSTM perf':>10s}")
    for r in results:
        print(f"{r['task']:>14s}  {r['variant'][:4]:>4s}  {r['ood_ratio']:>4.1f}x  "
              f"{r['err_gdc']:>5d}/{r['total']:<6d}  {r['err_lstm']:>5d}/{r['total']:<6d}  "
              f"{r['perfect_gdc']:>4d}/{r['n_test']:<3d}  {r['perfect_lstm']:>5d}/{r['n_test']:<3d}")

    # Save CSV for downstream table updates.
    import csv
    out_csv = os.path.join(HERE, 'lstm_vs_gdc_4x.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
