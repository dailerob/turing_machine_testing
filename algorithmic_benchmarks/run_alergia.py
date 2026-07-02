"""ALERGIA on the algorithmic benchmarks — TM tasks (reduced 3-col)
and Dyck-1.

Mirrors the val-tuned protocol of `parrot_eval.py` / `hpylm_eval.py`:

* TM tasks: tokenise (read, write, dir) tuples to integer IDs,
  prepend a START sentinel, train ALERGIA via AALpy's `run_Alergia`,
  evaluate per-step accuracy of (read, write, dir) restricted to
  candidate next-tuples whose `read` field matches the actual next read.

* Dyck-1: train on 1000 train sequences (depth ≤ 4), val on depth 6,
  evaluate on depth ≤ 8 next-symbol accuracy on 200 test sequences.

* Hyperparam to tune: Hoeffding-test eps ∈ {0.001, 0.005, 0.05, 0.5}.
  Picked per task by val tuple errors.

Run:
    python algorithmic_benchmarks/run_alergia.py
"""
from __future__ import annotations
import os, sys, time, csv
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import dyck1                                          # noqa: E402
from aalpy.learning_algs import run_Alergia            # noqa: E402
from _tm_task_config import (                          # noqa: E402
    TM_TASKS, TASK_ORDER, simulate_train_val_test, SUFFIX)

EPS_GRID = [0.001, 0.005, 0.05, 0.5]


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


def alergia_train(seqs_int, alphabet_size, eps=0.05):
    START = alphabet_size
    data = [[START] + [int(t) for t in s] for s in seqs_int if len(s) > 0]
    mc = run_Alergia(data, automaton_type='mc', eps=eps, print_info=False)
    return mc, START


def state_next_dist(state):
    nxt = defaultdict(float)
    for target, prob in state.transitions:
        nxt[target.output] += prob
    return nxt


def step_state(state, output):
    candidates = [(t, p) for t, p in state.transitions if t.output == output]
    if not candidates:
        return None, 0.0
    return max(candidates, key=lambda tp: tp[1])


def alergia_eval_tm(model, START, test_runs, tuple_to_id, id_to_tuple):
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
        state = model.initial_state
        ns, _ = step_state(state, int(x[0]))
        if ns is None:
            perfect += 1
            continue
        state = ns

        tape_err = 0
        for t in range(len(x) - 1):
            next_dist = state_next_dist(state)
            actual_next_id = int(x[t + 1])
            actual_tup = id_to_tuple[actual_next_id]
            actual_read = actual_tup[0]
            cands = by_read.get(actual_read, [])
            if not cands:
                pred_tup = id_to_tuple[max(next_dist, key=next_dist.get)] \
                    if next_dist else (-1, -1, -1)
            else:
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
                ns, _ = step_state(state, actual)
                if ns is None: state = model.initial_state
                else: state = ns
                continue
            next_dist = state_next_dist(state)
            cands = {tok: p for tok, p in next_dist.items()
                     if tok < alphabet_size}
            if not cands:
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


def run_tm_alergia(name, log, variant):
    cfg = TM_TASKS[name]
    n_test = cfg['n_test']
    log(f"\n{'='*66}\nALERGIA on {name} ({variant})\n{'='*66}")
    tr_runs, val_runs, te_runs = simulate_train_val_test(name, variant)
    tuple_to_id, id_to_tuple = reduced_alphabet(tr_runs)
    nA = len(id_to_tuple)
    log(f"  alphabet={nA}, train={len(tr_runs)}, val={len(val_runs)}, "
        f"test={len(te_runs)}")
    train_seqs = [encode_reduced(t, tuple_to_id) for t in tr_runs]
    train_seqs = [s for s in train_seqs if len(s) > 0]

    val_results = []
    for eps in EPS_GRID:
        t0 = time.time()
        model, START = alergia_train(train_seqs, nA, eps=eps)
        train_t = time.time() - t0
        n_states = len(model.states)
        _, _, terr_v, _ = alergia_eval_tm(
            model, START, val_runs, tuple_to_id, id_to_tuple)
        log(f"  eps={eps}: states={n_states}, val_errors={terr_v} "
            f"(train={train_t:.1f}s)")
        val_results.append((terr_v, eps, model, START, n_states, train_t))
    val_results.sort(key=lambda x: x[0])
    val_terr, best_eps, best_model, best_START, n_states, train_t = val_results[0]
    log(f"  Val pick: eps={best_eps} (states={n_states}) val_errors={val_terr}")

    t0 = time.time()
    acc, total, terr, perf = alergia_eval_tm(
        best_model, best_START, te_runs, tuple_to_id, id_to_tuple)
    eval_t = time.time() - t0
    n_pred = int(total[0])
    log(f"  ALERGIA test: read={acc[0]:.4f} write={acc[1]:.4f} dir={acc[2]:.4f} "
        f"mean={acc.mean():.4f}, errors={terr}/{n_pred} "
        f"({100*terr/max(n_pred,1):.3f}%), perfect={perf}/{n_test} ({eval_t:.1f}s)")
    return [dict(task=name, variant=variant, model='ALERGIA',
                 eps=best_eps, n_states=n_states,
                 val_tuple_errors=val_terr,
                 train_s=train_t, eval_s=eval_t,
                 acc_read=acc[0], acc_write=acc[1], acc_dir=acc[2],
                 mean_acc=acc.mean(),
                 tuple_errors=terr, n_predictions=n_pred,
                 perfect_tapes=perf, n_test=n_test)]


def run_dyck_alergia(log):
    log(f"\n{'='*66}\nALERGIA on dyck1\n{'='*66}")
    tr = dyck1.simulate(1000, max_depth=4, length_min=4, length_max=200,
                        seed=42)
    val = dyck1.simulate(100, max_depth=6, length_min=4, length_max=300,
                         seed=7)
    te = dyck1.simulate(200, max_depth=8, length_min=4, length_max=400,
                        seed=123)
    nA = dyck1.ALPHABET_SIZE

    val_results = []
    for eps in EPS_GRID:
        t0 = time.time()
        model, START = alergia_train(tr['sequences'], nA, eps=eps)
        train_t = time.time() - t0
        n_states = len(model.states)
        _, total_v, correct_v = alergia_eval_dyck(
            model, START, val['sequences'], nA)
        terr_v = total_v - correct_v
        log(f"  eps={eps}: states={n_states}, val_errors={terr_v} "
            f"(train={train_t:.1f}s)")
        val_results.append((terr_v, eps, model, START, n_states, train_t))
    val_results.sort(key=lambda x: x[0])
    val_terr, best_eps, best_model, best_START, n_states, train_t = val_results[0]
    log(f"  Val pick: eps={best_eps} (states={n_states}) val_errors={val_terr}")

    t0 = time.time()
    acc, total, correct = alergia_eval_dyck(
        best_model, best_START, te['sequences'], nA)
    eval_t = time.time() - t0
    log(f"  ALERGIA dyck1 acc={acc:.4f} ({correct}/{total}, eval={eval_t:.1f}s)")
    return [dict(task='dyck1', variant='n/a', model='ALERGIA',
                 eps=best_eps, n_states=n_states,
                 val_tuple_errors=val_terr,
                 train_s=train_t, eval_s=eval_t,
                 acc_read=np.nan, acc_write=np.nan, acc_dir=np.nan,
                 mean_acc=acc,
                 tuple_errors=total - correct, n_predictions=total,
                 perfect_tapes=-1, n_test=200)]


def main():
    log_lines = []
    def log(msg=""):
        print(msg, flush=True); log_lines.append(str(msg))

    log("=== ALERGIA val-tuned ===")
    rows = []
    for variant in ('original', 'noread'):
        for name in TASK_ORDER:
            rows += run_tm_alergia(name, log, variant)
    rows += run_dyck_alergia(log)

    out_csv = os.path.join(HERE, f'alergia_results{SUFFIX}.csv')
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
            f"{r['eps']:>8.4f}  {r['n_states']:>7d}  "
            f"{r['mean_acc']:>9.4f}  {terr_str:>14s}  "
            f"{perfect_str:>8s}  {r['train_s']+r['eval_s']:>7.1f}")

    log_path = os.path.join(HERE, f'alergia{SUFFIX}.log')
    with open(log_path, 'w') as f: f.write('\n'.join(log_lines))


if __name__ == "__main__":
    main()
