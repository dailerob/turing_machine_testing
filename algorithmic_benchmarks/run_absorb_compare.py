"""Algorithmic / Turing benchmarks: compare GDC diffuse vs absorb mode.

For each (task, variant), uses the per-task tuned config from
TUNED_GDC_RESULTS.md and runs both diffuse and absorb terminal_behavior.
Reports error counts side by side.

Tasks tested:
  parity (original + noread)        : alpha=0.50, theta=0.05, two_step
  increment (original + noread)     : alpha=0.50, theta=0.005, self_loop
  reverse (original)                : alpha=0.95, theta=0.05, self_loop
  reverse (noread)                  : alpha=0.95, theta=0.05, self_loop
  binary_adder (original)           : alpha=0.50, theta=0.005, self_loop
  binary_adder (noread)             : alpha=0.90, theta=0.05, two_step
  dyck1                             : alpha=0.95, theta=0.05, self_loop
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

# Per-task tuned configs (from TUNED_GDC_RESULTS.md)
TM_TASKS = [
    ('parity',     'parity_tm',    'original',  (3, 8),  (16, 32),  300, 20, 200,
     dict(alpha=0.50, theta=0.05,  gamma=0.0,
          transition_type='self_loop_two_step')),
    ('parity',     'parity_tm',    'noread',    (3, 8),  (16, 32),  300, 20, 200,
     dict(alpha=0.50, theta=0.05,  gamma=0.0,
          transition_type='self_loop_two_step')),
    ('increment',  'increment_tm', 'original',  (1, 5),  (8, 12),   300, 20, 200,
     dict(alpha=0.50, theta=0.005, gamma=0.0,
          transition_type='self_loop')),
    ('increment',  'increment_tm', 'noread',    (1, 5),  (8, 12),   300, 20, 200,
     dict(alpha=0.50, theta=0.005, gamma=0.0,
          transition_type='self_loop')),
    ('reverse',    'reverse_tm',   'original',  (3, 6),  (10, 16),  300, 20, 10000,
     dict(alpha=0.95, theta=0.05,  gamma=0.0,
          transition_type='self_loop')),
    ('reverse',    'reverse_tm',   'noread',    (3, 6),  (10, 16),  300, 20, 10000,
     dict(alpha=0.95, theta=0.05,  gamma=0.0,
          transition_type='self_loop')),
]

ADDER_TASKS = [
    ('binary_adder', 'original',
     dict(alpha=0.50, theta=0.005, gamma=0.0,
          transition_type='self_loop')),
    ('binary_adder', 'noread',
     dict(alpha=0.90, theta=0.05,  gamma=0.0,
          transition_type='self_loop_two_step')),
]

DYCK_CFG = dict(alpha=0.95, theta=0.05, gamma=0.0,
                transition_type='self_loop')

OUT_CSV = os.path.join(HERE, 'absorb_results.csv')


def reduced_alphabet(runs):
    seen = set()
    for arr in runs:
        for row in arr:
            if int(row[0]) == -1: continue
            seen.add((int(row[1]), int(row[2]), int(row[3])))
    id_to_tuple = sorted(seen)
    return {t: i for i, t in enumerate(id_to_tuple)}, id_to_tuple


def encode_reduced(arr, tuple_to_id):
    out = []
    for row in arr:
        if int(row[0]) == -1: continue
        key = (int(row[1]), int(row[2]), int(row[3]))
        if key in tuple_to_id:
            out.append(tuple_to_id[key])
    return np.asarray(out, dtype=np.int64)


def gdc_eval_tm_reduced(gdc, test_runs):
    correct = np.zeros(3, dtype=np.int64)
    total = np.zeros(3, dtype=np.int64)
    tuple_errors, perfect = 0, 0
    for arr in test_runs:
        if len(arr) < 2: perfect += 1; continue
        valid = arr[:, 0] != -1
        tape_red = arr[valid][:, 1:4].astype(np.int64)
        if len(tape_red) < 2: perfect += 1; continue
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
            if mismatch: tape_err += 1; tuple_errors += 1
        if tape_err == 0: perfect += 1
    acc = correct / np.maximum(total, 1)
    return acc, total, tuple_errors, perfect


def gdc_eval_dyck(gdc, test_seqs):
    nA = gdc.k
    correct, total = 0, 0
    for seq in test_seqs:
        if len(seq) < 2: continue
        col = seq.reshape(-1, 1).astype(np.int64)
        _, hist = gdc.forward_pass(col, return_history=True)
        for t in range(len(seq) - 1):
            forecast = gdc.forecast(hist[t], n_steps=1)
            pred = gdc.greedy_sample(forecast)
            actual = int(seq[t + 1])
            total += 1
            if int(pred[0]) == actual: correct += 1
    return correct / max(total, 1), total, correct


def run_tm_cell(args):
    task_name, mod_name, variant, train_range, test_range, n_train, n_test, max_steps, gdc_cfg = args
    sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
    import importlib
    from generative_dense_chain import GenerativeDenseChain
    mod = importlib.import_module(mod_name)
    nr = (variant == 'noread')
    tr = mod.simulate(n_train, train_range, max_steps=max_steps,
                      seed=42, noread=nr)
    te = mod.simulate(n_test, test_range, max_steps=max_steps * 4,
                      seed=123, noread=nr)
    train_red = [t[t[:, 0] != -1][:, 1:4].astype(np.int64)
                 for t in tr['runs']]
    train_red = [t for t in train_red if len(t) > 0]
    rows = []
    for tb in ['diffuse', 'absorb']:
        gdc = GenerativeDenseChain(train_red, beta=0.0,
                                    initial_dist='sequence_starts',
                                    terminal_behavior=tb,
                                    **gdc_cfg)
        acc, total, terr, perf = gdc_eval_tm_reduced(gdc, te['runs'])
        rows.append(dict(task=task_name, variant=variant, mode=tb,
                         model='gdc-tuned', mean_acc=float(acc.mean()),
                         errors=int(terr), n_predictions=int(total[0]),
                         perfect=int(perf), n_test=n_test,
                         n_states=int(gdc.n_states), **gdc_cfg))
    return rows


def run_adder_cell(args):
    name, variant, gdc_cfg = args
    sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
    from generative_dense_chain import GenerativeDenseChain
    from binary_alphabet_adder import (
        simulate_random_binary_alphabet_adders, BINARY_ALPHABET_ADDER)
    from _tm_common import apply_noread_to_runs
    n_train, n_test = 200, 10
    tr = simulate_random_binary_alphabet_adders(n_train, num_range=(0, 32),
        max_steps=200_000, seed=42)
    te = simulate_random_binary_alphabet_adders(n_test, num_range=(0, 1000),
        max_steps=200_000, seed=123)
    if variant == 'noread':
        merged_se = dict(tr['symbol_encoding'])
        for k in te['symbol_encoding']:
            if k not in merged_se: merged_se[k] = len(merged_se)
        merged_st = dict(tr['state_encoding'])
        for k in te['state_encoding']:
            if k not in merged_st: merged_st[k] = len(merged_st)
        tr_runs, _ = apply_noread_to_runs(tr['runs'], BINARY_ALPHABET_ADDER,
                                          merged_st, merged_se)
        te_runs, _ = apply_noread_to_runs(te['runs'], BINARY_ALPHABET_ADDER,
                                          merged_st, merged_se)
        tr['runs'] = tr_runs; te['runs'] = te_runs
    train_red = [t[t[:, 0] != -1][:, 1:4].astype(np.int64)
                 for t in tr['runs']]
    train_red = [t for t in train_red if len(t) > 0]
    rows = []
    for tb in ['diffuse', 'absorb']:
        gdc = GenerativeDenseChain(train_red, beta=0.0,
                                    initial_dist='sequence_starts',
                                    terminal_behavior=tb,
                                    **gdc_cfg)
        acc, total, terr, perf = gdc_eval_tm_reduced(gdc, te['runs'])
        rows.append(dict(task=name, variant=variant, mode=tb,
                         model='gdc-tuned', mean_acc=float(acc.mean()),
                         errors=int(terr), n_predictions=int(total[0]),
                         perfect=int(perf), n_test=n_test,
                         n_states=int(gdc.n_states), **gdc_cfg))
    return rows


def run_dyck_cell(_):
    sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
    from generative_dense_chain import GenerativeDenseChain
    import dyck1
    tr = dyck1.simulate(1000, max_depth=4, length_min=4, length_max=200,
                        seed=42)
    te = dyck1.simulate(200, max_depth=8, length_min=4, length_max=400,
                        seed=123)
    train_seqs = [s.reshape(-1, 1).astype(np.int64) for s in tr['sequences']]
    rows = []
    for tb in ['diffuse', 'absorb']:
        gdc = GenerativeDenseChain(train_seqs, beta=0.0,
                                    initial_dist='sequence_starts',
                                    terminal_behavior=tb,
                                    **DYCK_CFG)
        acc, total, correct = gdc_eval_dyck(gdc, te['sequences'])
        rows.append(dict(task='dyck1', variant='n/a', mode=tb,
                         model='gdc-tuned', mean_acc=float(acc),
                         errors=int(total - correct),
                         n_predictions=int(total), perfect=-1, n_test=200,
                         n_states=int(gdc.n_states), **DYCK_CFG))
    return rows


def main():
    n_workers = max(1, min(12, (os.cpu_count() or 4) - 1))
    print(f"Algorithmic absorb-vs-diffuse comparison: "
          f"{len(TM_TASKS) + len(ADDER_TASKS) + 1} task-variants, "
          f"{n_workers} workers", flush=True)

    all_rows = []; t0 = time.time(); done = 0
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_tm_cell, TM_TASKS):
            all_rows.extend(r); done += 1
            print(f"  TM done {done}/{len(TM_TASKS)} [{time.time()-t0:.0f}s]",
                  flush=True)
        for r in pool.imap_unordered(run_adder_cell, ADDER_TASKS):
            all_rows.extend(r)
            print(f"  adder done [{time.time()-t0:.0f}s]", flush=True)
        for r in pool.imap_unordered(run_dyck_cell, [None]):
            all_rows.extend(r)
            print(f"  dyck done [{time.time()-t0:.0f}s]", flush=True)

    fields = list(all_rows[0].keys())
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}", flush=True)

    # Summary table
    print("\n=== GDC diffuse vs absorb on algorithmic benchmarks ===")
    print(f"{'task':>13s} {'variant':>10s}  "
          f"{'diffuse errors':>16s}  {'absorb errors':>15s}  "
          f"{'diffuse perfect':>17s}  {'absorb perfect':>16s}")
    pairs = {}
    for r in all_rows:
        pairs.setdefault((r['task'], r['variant']), {})[r['mode']] = r
    for (task, variant), modes in sorted(pairs.items()):
        d = modes.get('diffuse'); a = modes.get('absorb')
        if d is None or a is None: continue
        d_err = f"{d['errors']}/{d['n_predictions']}"
        a_err = f"{a['errors']}/{a['n_predictions']}"
        if d['perfect'] >= 0:
            d_perf = f"{d['perfect']}/{d['n_test']}"
            a_perf = f"{a['perfect']}/{a['n_test']}"
        else:
            d_perf = a_perf = 'n/a'
        print(f"{task:>13s} {variant:>10s}  "
              f"{d_err:>16s}  {a_err:>15s}  "
              f"{d_perf:>17s}  {a_perf:>16s}")


if __name__ == "__main__":
    main()
