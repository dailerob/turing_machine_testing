"""ALERGIA on the algorithmic benchmarks — TM tasks (reduced 3-col)
and Dyck-1.

Mirrors the GDC/CHMM evaluation harness in `run_benchmarks.py`:

* TM tasks (parity, increment, reverse, binary_adder), both 'original'
  and 'noread' variants:
    - tokenise (read, write, dir) tuples to integer IDs
    - prepend a START sentinel to every sequence (ALERGIA-MC requires
      a shared initial output)
    - train ALERGIA via AALpy's `run_Alergia`
    - evaluate per-step accuracy of (read, write, dir) restricted to
      candidate next-tuples whose `read` field matches the actual
      next read.

* Dyck-1: train on 1000 train sequences (depth ≤ 4), evaluate
  next-symbol accuracy on 200 test sequences (depth ≤ 8).

ALERGIA runs at default Hoeffding-test `eps=0.05` — same as the
PAutomaC sweep. Optionally we can sweep eps if any task shows large
sensitivity.

Run:
    python algorithmic_benchmarks/run_alergia.py
"""

from __future__ import annotations
import os
import sys
import time
import csv
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import parity_tm, increment_tm, reverse_tm, dyck1  # noqa: E402
from _tm_common import apply_noread_to_runs  # noqa: E402
from binary_alphabet_adder import (  # noqa: E402
    simulate_random_binary_alphabet_adders, BINARY_ALPHABET_ADDER)

from aalpy.learning_algs import run_Alergia  # noqa: E402

EPS_GRID = [0.05]  # default; can extend


# ---------------------------------------------------------------------
# Helpers (mirror run_benchmarks.py)
# ---------------------------------------------------------------------
def reduced_alphabet(runs):
    seen = set()
    for tape in runs:
        for row in tape:
            if int(row[0]) == -1:
                continue
            seen.add((int(row[1]), int(row[2]), int(row[3])))
    id_to_tuple = sorted(seen)
    tuple_to_id = {t: i for i, t in enumerate(id_to_tuple)}
    return tuple_to_id, id_to_tuple


def encode_reduced(tape, tuple_to_id):
    out = []
    for row in tape:
        if int(row[0]) == -1:
            continue
        key = (int(row[1]), int(row[2]), int(row[3]))
        tid = tuple_to_id.get(key)
        if tid is not None:
            out.append(tid)
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------------
# ALERGIA scoring
# ---------------------------------------------------------------------
def alergia_train(seqs_int, alphabet_size, eps=0.05):
    """Train ALERGIA on integer sequences, prepending a shared START."""
    START = alphabet_size  # one past the alphabet
    data = [[START] + [int(t) for t in s] for s in seqs_int if len(s) > 0]
    mc = run_Alergia(data, automaton_type='mc', eps=eps, print_info=False)
    return mc, START


def state_next_dist(state):
    """Return dict {output_token: total_prob} from `state`'s transitions."""
    nxt = defaultdict(float)
    for target, prob in state.transitions:
        nxt[target.output] += prob
    return nxt


def step_state(state, output):
    """Find a transition target whose output matches `output`.
    Returns (target, prob) or (None, 0)."""
    candidates = [(t, p) for t, p in state.transitions if t.output == output]
    if not candidates:
        return None, 0.0
    # Pick the highest-prob (typical in AALpy these are unique)
    return max(candidates, key=lambda tp: tp[1])


def alergia_eval_tm(model, START, test_runs, tuple_to_id, id_to_tuple):
    """Per-position accuracy on (read, write, dir).  Returns
    (acc[3], total[3], tuple_errors, perfect_per_tape)."""
    correct = np.zeros(3, dtype=np.int64)
    total = np.zeros(3, dtype=np.int64)
    tuple_errors, perfect = 0, 0
    by_read = {}
    for tid, tup in enumerate(id_to_tuple):
        by_read.setdefault(tup[0], []).append(tid)

    for tape in test_runs:
        x = encode_reduced(tape, tuple_to_id)
        if len(x) < 2:
            perfect += 1; continue
        # State at time 0: initial state (after consuming START)
        state = model.initial_state
        # Step into x[0]
        ns, _ = step_state(state, int(x[0]))
        if ns is None:
            # Unobserved first symbol — fall back: skip eval for this tape
            perfect += 1
            continue
        state = ns

        tape_err = 0
        for t in range(len(x) - 1):
            # Predict next token from current state
            next_dist = state_next_dist(state)
            actual_next_id = int(x[t + 1])
            actual_tup = id_to_tuple[actual_next_id]
            actual_read = actual_tup[0]
            cands = by_read.get(actual_read, [])
            if not cands:
                # No transition exists with the actual read — register error
                pred_tup = id_to_tuple[max(next_dist, key=next_dist.get)] \
                    if next_dist else (-1, -1, -1)
            else:
                # Score each candidate, picking argmax
                best_tid = max(cands, key=lambda c: next_dist.get(c, 0.0))
                pred_tup = id_to_tuple[best_tid]
            mismatch = False
            for pos in range(3):
                total[pos] += 1
                if pred_tup[pos] == actual_tup[pos]:
                    correct[pos] += 1
                else:
                    mismatch = True
            if mismatch:
                tape_err += 1; tuple_errors += 1
            # Advance state to actual observed token
            ns, _ = step_state(state, actual_next_id)
            if ns is None:
                state = model.initial_state
            else:
                state = ns
        if tape_err == 0:
            perfect += 1
    acc = correct / np.maximum(total, 1)
    return acc, total, tuple_errors, perfect


def alergia_eval_dyck(model, START, test_seqs, alphabet_size):
    """Next-symbol accuracy for Dyck-1.  Skip prediction on positions
    where the actual next symbol is END (alphabet_size index)."""
    correct, total = 0, 0
    for seq in test_seqs:
        if len(seq) < 2: continue
        state = model.initial_state
        ns, _ = step_state(state, int(seq[0]))
        if ns is None:
            continue
        state = ns
        for t in range(len(seq) - 1):
            actual = int(seq[t + 1])
            if actual >= alphabet_size:
                # END symbol — don't count
                ns, _ = step_state(state, actual)
                if ns is None: state = model.initial_state
                else: state = ns
                continue
            next_dist = state_next_dist(state)
            # Restrict to non-END outputs
            cands = {tok: p for tok, p in next_dist.items()
                     if tok < alphabet_size}
            if not cands:
                # No valid prediction
                pred = -1
            else:
                pred = max(cands, key=cands.get)
            total += 1
            if pred == actual:
                correct += 1
            ns, _ = step_state(state, actual)
            if ns is None: state = model.initial_state
            else: state = ns
    return (correct / max(total, 1), total, correct)


# ---------------------------------------------------------------------
# Per-task runners
# ---------------------------------------------------------------------
def run_tm_alergia(name, module, train_range, test_range,
                   n_train, n_test, max_steps, log, variant):
    nr = (variant == 'noread')
    log(f"\n{'='*66}\nALERGIA on {name} ({variant})\n{'='*66}")
    tr = module.simulate(n_train, train_range, max_steps=max_steps,
                         seed=42, noread=nr)
    te = module.simulate(n_test, test_range, max_steps=max_steps * 4,
                         seed=123, noread=nr)
    train_lens = [t.shape[0] for t in tr['runs']]
    test_lens = [t.shape[0] for t in te['runs']]
    log(f"  train: n={n_train}, len min/max/mean: "
        f"{min(train_lens)}/{max(train_lens)}/{np.mean(train_lens):.1f}; "
        f"test: n={n_test}, len mean={np.mean(test_lens):.1f}")
    tuple_to_id, id_to_tuple = reduced_alphabet(tr['runs'])
    nA = len(id_to_tuple)
    log(f"  reduced alphabet size: {nA}")

    train_seqs = [encode_reduced(t, tuple_to_id) for t in tr['runs']]
    train_seqs = [s for s in train_seqs if len(s) > 0]

    rows = []
    for eps in EPS_GRID:
        t0 = time.time()
        model, START = alergia_train(train_seqs, nA, eps=eps)
        train_t = time.time() - t0
        n_states = len(model.states)
        t0 = time.time()
        acc, total, terr, perf = alergia_eval_tm(
            model, START, te['runs'], tuple_to_id, id_to_tuple)
        eval_t = time.time() - t0
        n_pred = int(total[0])
        log(f"  eps={eps}: states={n_states}, "
            f"read={acc[0]:.4f} write={acc[1]:.4f} dir={acc[2]:.4f} "
            f"mean={acc.mean():.4f}, errors={terr}/{n_pred} "
            f"({100*terr/max(n_pred,1):.3f}%), perfect={perf}/{n_test}")
        log(f"    [train={train_t:.1f}s eval={eval_t:.1f}s]")
        rows.append(dict(task=name, variant=variant, model='ALERGIA',
                         K_or_alpha=f'eps={eps}', n_states=n_states,
                         train_s=train_t, eval_s=eval_t,
                         acc_read=acc[0], acc_write=acc[1], acc_dir=acc[2],
                         mean_acc=acc.mean(),
                         tuple_errors=terr, n_predictions=n_pred,
                         perfect_tapes=perf, n_test=n_test))
    return rows


def run_binary_adder_alergia(log, variant, n_train=200, n_test=10):
    log(f"\n{'='*66}\nALERGIA on binary_adder ({variant})\n{'='*66}")
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
        tr_runs, _ = apply_noread_to_runs(
            tr['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)
        te_runs, _ = apply_noread_to_runs(
            te['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)
        tr['runs'] = tr_runs; te['runs'] = te_runs
    train_lens = [t.shape[0] for t in tr['runs']]
    test_lens = [t.shape[0] for t in te['runs']]
    log(f"  train: n={n_train}, len mean={np.mean(train_lens):.1f}; "
        f"test: n={n_test}, len mean={np.mean(test_lens):.1f}")
    tuple_to_id, id_to_tuple = reduced_alphabet(tr['runs'])
    nA = len(id_to_tuple)
    log(f"  reduced alphabet size: {nA}")
    train_seqs = [encode_reduced(t, tuple_to_id) for t in tr['runs']]
    train_seqs = [s for s in train_seqs if len(s) > 0]

    rows = []
    for eps in EPS_GRID:
        t0 = time.time()
        model, START = alergia_train(train_seqs, nA, eps=eps)
        train_t = time.time() - t0
        n_states = len(model.states)
        t0 = time.time()
        acc, total, terr, perf = alergia_eval_tm(
            model, START, te['runs'], tuple_to_id, id_to_tuple)
        eval_t = time.time() - t0
        n_pred = int(total[0])
        log(f"  eps={eps}: states={n_states}, "
            f"read={acc[0]:.4f} write={acc[1]:.4f} dir={acc[2]:.4f} "
            f"mean={acc.mean():.4f}, errors={terr}/{n_pred} "
            f"({100*terr/max(n_pred,1):.3f}%), perfect={perf}/{n_test}")
        log(f"    [train={train_t:.1f}s eval={eval_t:.1f}s]")
        rows.append(dict(task='binary_adder', variant=variant,
                         model='ALERGIA', K_or_alpha=f'eps={eps}',
                         n_states=n_states, train_s=train_t,
                         eval_s=eval_t,
                         acc_read=acc[0], acc_write=acc[1], acc_dir=acc[2],
                         mean_acc=acc.mean(), tuple_errors=terr,
                         n_predictions=n_pred, perfect_tapes=perf,
                         n_test=n_test))
    return rows


def run_dyck_alergia(log):
    log(f"\n{'='*66}\nALERGIA on dyck1\n{'='*66}")
    tr = dyck1.simulate(1000, max_depth=4, length_min=4, length_max=200,
                        seed=42)
    te = dyck1.simulate(200, max_depth=8, length_min=4, length_max=400,
                        seed=123)
    nA = dyck1.ALPHABET_SIZE
    train_seqs = tr['sequences']
    log(f"  train n=1000 max_depth=4, alphabet={nA}")

    rows = []
    for eps in EPS_GRID:
        t0 = time.time()
        model, START = alergia_train(train_seqs, nA, eps=eps)
        train_t = time.time() - t0
        n_states = len(model.states)
        t0 = time.time()
        acc, total, correct = alergia_eval_dyck(model, START,
                                                te['sequences'], nA)
        eval_t = time.time() - t0
        log(f"  eps={eps}: states={n_states}, "
            f"next-symbol accuracy={acc:.4f} ({correct}/{total})")
        log(f"    [train={train_t:.1f}s eval={eval_t:.1f}s]")
        rows.append(dict(task='dyck1', variant='n/a', model='ALERGIA',
                         K_or_alpha=f'eps={eps}', n_states=n_states,
                         train_s=train_t, eval_s=eval_t,
                         acc_read=np.nan, acc_write=np.nan, acc_dir=np.nan,
                         mean_acc=acc, tuple_errors=total - correct,
                         n_predictions=total, perfect_tapes=-1, n_test=200))
    return rows


def main():
    log_lines = []
    def log(msg=""):
        print(msg, flush=True); log_lines.append(str(msg))

    log("=== ALERGIA on the algorithmic benchmarks ===")
    rows = []
    for variant in ('original', 'noread'):
        rows += run_tm_alergia('parity', parity_tm, train_range=(3, 8),
                               test_range=(16, 32), n_train=300, n_test=20,
                               max_steps=200, log=log, variant=variant)
        rows += run_tm_alergia('increment', increment_tm, train_range=(1, 5),
                               test_range=(8, 12), n_train=300, n_test=20,
                               max_steps=200, log=log, variant=variant)
        rows += run_tm_alergia('reverse', reverse_tm, train_range=(3, 6),
                               test_range=(10, 16), n_train=300, n_test=20,
                               max_steps=10000, log=log, variant=variant)
        rows += run_binary_adder_alergia(log=log, variant=variant,
                                         n_train=200, n_test=10)
    rows += run_dyck_alergia(log=log)

    out_csv = os.path.join(HERE, 'alergia_results.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log(f"\nWrote {out_csv}")

    log("\n=== ALERGIA SUMMARY ===")
    log(f"{'task':>14s}  {'variant':>8s}  {'eps':>8s}  {'states':>7s}  "
        f"{'mean_acc':>9s}  {'errors':>14s}  {'perfect':>8s}  {'time_s':>7s}")
    for r in rows:
        terr_str = f"{r['tuple_errors']}/{r['n_predictions']}"
        perfect_str = (f"{r['perfect_tapes']}/{r['n_test']}"
                       if r['perfect_tapes'] >= 0 else '   -   ')
        log(f"{r['task']:>14s}  {r['variant']:>8s}  "
            f"{r['K_or_alpha']:>8s}  {r['n_states']:>7d}  "
            f"{r['mean_acc']:>9.4f}  {terr_str:>14s}  "
            f"{perfect_str:>8s}  "
            f"{r['train_s']+r['eval_s']:>7.1f}")

    log_path = os.path.join(HERE, 'alergia.log')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))


if __name__ == "__main__":
    main()
