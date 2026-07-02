"""Discrete context-parroting on the algorithmic / Turing-machine
benchmarks.

Mirrors `run_benchmarks.py` exactly: same nine TM tasks plus dyck1,
same train/test sizes, same alphabet construction, same metrics
(tuple-level error count for TM tasks, next-symbol accuracy for
Dyck-1).

Predictor: top-K nearest-neighbour parrot over length-L sliding windows
of the concatenated training corpus (Hamming distance). For TM tasks
we use the existing protocol where the actual next read symbol is
given as a conditional, so the parrot's job is to pick the best
(write, dir) tuple consistent with that read. For Dyck-1: argmax of
K-vote restricted to {OPEN, CLOSE}.

We sweep over (L, K) and pick the per-task best by val tuple errors
on a stretched validation set (input lengths strictly between train
and test). See `_tm_task_config.py` for the canonical splits.

Output: algorithmic_benchmarks/parrot_benchmark_results.csv
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

import dyck1                                          # noqa: E402
from discrete_parrot import DiscreteParrotPool         # noqa: E402
from torch_tm_adapters import TorchTMParrot           # noqa: E402
from _tm_task_config import (                          # noqa: E402
    TM_TASKS, TASK_ORDER, simulate_train_val_test, SUFFIX)

# Variant grid for parrot
LS = [1, 2, 3, 4, 6, 8]
KS = [1, 5, 25]


# --------------------------------------------------------------------
# TM-trace tokenisation (same as run_benchmarks.py)
# --------------------------------------------------------------------
def reduced_alphabet(train_tapes):
    seen = set()
    for tape in train_tapes:
        for row in tape:
            if int(row[0]) == -1:
                continue
            seen.add((int(row[1]), int(row[2]), int(row[3])))
    id_to_tuple = sorted(seen)
    tuple_to_id = {t: i for i, t in enumerate(id_to_tuple)}
    return tuple_to_id, id_to_tuple


def encode_reduced(tape, tuple_to_id):
    out, skipped = [], 0
    for row in tape:
        if int(row[0]) == -1:
            continue
        key = (int(row[1]), int(row[2]), int(row[3]))
        if key in tuple_to_id:
            out.append(tuple_to_id[key])
        else:
            skipped += 1
    return np.asarray(out, dtype=np.int64), skipped


# --------------------------------------------------------------------
# Parrot evaluator on TM tapes (reduced 3-col, conditional on next read)
# --------------------------------------------------------------------
def parrot_eval_tm_reduced(pool, test_tapes, tuple_to_id, id_to_tuple, K):
    """For each test tape, slide through positions and at each t predict
    the next tuple, restricting to candidates whose tuple[0] == actual
    next read. Mirrors chmm_eval_tm_reduced's protocol.

    `pool` may be either a numpy DiscreteParrotPool (per-position loop)
    OR a TorchTMParrot (GPU-batched across all tapes/positions).
    Detected via duck-typing: if it has score_tapes_batched, use it."""
    by_read = {}
    for tid, tup in enumerate(id_to_tuple):
        by_read.setdefault(tup[0], []).append(tid)
    correct = np.zeros(3, dtype=np.int64)
    total = np.zeros(3, dtype=np.int64)
    tuple_errors, perfect = 0, 0
    nA = len(id_to_tuple)

    encoded = [(encode_reduced(t, tuple_to_id)[0]) for t in test_tapes]
    valid_tapes = [(i, x) for i, x in enumerate(encoded) if len(x) >= 2]
    perfect += sum(1 for x in encoded if len(x) < 2)

    if not valid_tapes:
        return correct / np.maximum(total, 1), total, tuple_errors, perfect

    indices, xs = zip(*valid_tapes)
    actuals_per_tape = [
        [id_to_tuple[int(x[t + 1])][0] for t in range(len(x) - 1)]
        for x in xs]

    if hasattr(pool, 'score_tapes_batched'):
        all_preds = pool.score_tapes_batched(list(xs), actuals_per_tape,
                                              by_read, K=K)
    else:
        all_preds = []
        for x, actuals in zip(xs, actuals_per_tape):
            preds = np.empty(len(x) - 1, dtype=np.int64)
            for t in range(len(x) - 1):
                cands = by_read.get(int(actuals[t]), [])
                if not cands:
                    preds[t] = 0; continue
                prefix = x[:t + 1]
                mask = np.zeros(nA, dtype=bool); mask[cands] = True
                preds[t] = pool.predict_argmax(prefix, h=1, K=K, mask=mask,
                                                alpha_prior=1.0)
            all_preds.append(preds)

    for x, preds in zip(xs, all_preds):
        tape_err = 0
        for t in range(len(x) - 1):
            actual_tup = id_to_tuple[int(x[t + 1])]
            pred_tup = id_to_tuple[int(preds[t])]
            for pos in range(3):
                total[pos] += 1
                if pred_tup[pos] == actual_tup[pos]:
                    correct[pos] += 1
            if pred_tup != actual_tup:
                tape_err += 1; tuple_errors += 1
        if tape_err == 0:
            perfect += 1
    acc = correct / np.maximum(total, 1)
    return acc, total, tuple_errors, perfect


def parrot_eval_dyck(pool, test_seqs, K):
    correct, total = 0, 0
    nA = dyck1.ALPHABET_SIZE
    cand_mask = np.zeros(nA, dtype=bool)
    cand_mask[dyck1.OPEN] = True
    cand_mask[dyck1.CLOSE] = True
    for x in test_seqs:
        if len(x) < 2:
            continue
        for t in range(len(x) - 1):
            actual_next = int(x[t + 1])
            if actual_next == dyck1.END:
                continue
            prefix = x[: t + 1]
            best = pool.predict_argmax(prefix, h=1, K=K, mask=cand_mask,
                                        alpha_prior=1.0)
            total += 1
            if best == actual_next:
                correct += 1
    return correct / max(total, 1), total, correct


# --------------------------------------------------------------------
# Per-task runner using shared train/val/test split
# --------------------------------------------------------------------
def run_tm_task(name, log, variant='original'):
    cfg = TM_TASKS[name]
    n_test = cfg['n_test']
    log(f"\n{'='*66}\nTASK: {name} (TM, variant={variant})\n{'='*66}")
    tr_runs, val_runs, te_runs = simulate_train_val_test(name, variant)
    tuple_to_id, id_to_tuple = reduced_alphabet(tr_runs)
    nA = len(id_to_tuple)
    log(f"  alphabet={nA}, train={len(tr_runs)}, val={len(val_runs)}, "
        f"test={len(te_runs)}")

    train_red_seqs = [encode_reduced(t, tuple_to_id)[0] for t in tr_runs]
    train_red_seqs = [s for s in train_red_seqs if len(s) > 0]

    val_scores = []
    for L in LS:
        pool = TorchTMParrot(L=L, K=1, alpha_prior=1.0)
        pool.fit(train_red_seqs, alphabet_size=nA)
        for K in KS:
            _, _, terr, _ = parrot_eval_tm_reduced(
                pool, val_runs, tuple_to_id, id_to_tuple, K=K)
            val_scores.append((terr, L, K, pool))
    # Stable sort: ties broken by (lower L, lower K) which is iteration order
    val_scores.sort(key=lambda x: x[0])
    val_terr, best_L, best_K, best_pool = val_scores[0]
    log(f"  Val pick: L={best_L}, K={best_K}  val_tuple_errors={val_terr}")

    t0 = time.time()
    acc, total, terr, perf = parrot_eval_tm_reduced(
        best_pool, te_runs, tuple_to_id, id_to_tuple, K=best_K)
    eval_t = time.time() - t0
    n_pred = int(total[0])
    log(f"  Parrot acc: read={acc[0]:.4f} write={acc[1]:.4f} "
        f"dir={acc[2]:.4f} mean={acc.mean():.4f}")
    log(f"  Parrot tuple errors: {terr}/{n_pred} "
        f"({100*terr/max(n_pred,1):.3f}%), perfect: {perf}/{n_test}  "
        f"(eval={eval_t:.1f}s)")
    return [dict(task=name, variant=variant, model='parrot',
                 L=best_L, K=best_K, val_tuple_errors=val_terr,
                 acc_read=acc[0], acc_write=acc[1], acc_dir=acc[2],
                 mean_acc=float(acc.mean()),
                 tuple_errors=terr, n_predictions=n_pred,
                 perfect_tapes=perf, n_test=n_test, eval_s=eval_t)]


def run_dyck(log):
    """Dyck-1 keeps its own splitting (no canonical val_range)."""
    name = 'dyck1'
    log(f"\n{'='*66}\nTASK: {name}\n{'='*66}")
    tr = dyck1.simulate(1000, max_depth=4, length_min=4, length_max=200,
                        seed=42)
    te = dyck1.simulate(200, max_depth=8, length_min=4, length_max=400,
                        seed=123)
    # Stretched val: max_depth=6 (between train depth=4 and test depth=8)
    val = dyck1.simulate(100, max_depth=6, length_min=4, length_max=300,
                         seed=7)
    tr_seqs = [s.astype(np.int64) for s in tr['sequences']]
    val_seqs = [s.astype(np.int64) for s in val['sequences']]
    te_seqs = te['sequences']

    nA = dyck1.ALPHABET_SIZE
    val_scores = []
    for L in LS:
        pool = DiscreteParrotPool(tr_seqs, alphabet_size=nA, L=L)
        for K in KS:
            acc, total, _ = parrot_eval_dyck(pool, val_seqs, K=K)
            val_scores.append((total - int(round(acc * total)), L, K, pool))
    val_scores.sort(key=lambda x: x[0])
    val_err, best_L, best_K, best_pool = val_scores[0]
    log(f"  Val pick: L={best_L}, K={best_K}  val_errors={val_err}")
    t0 = time.time()
    acc, total, correct = parrot_eval_dyck(best_pool, te_seqs, K=best_K)
    eval_t = time.time() - t0
    log(f"  Parrot dyck1 accuracy: {acc:.4f} ({correct}/{total}, "
        f"eval={eval_t:.1f}s)")
    return [dict(task=name, variant='n/a', model='parrot',
                 L=best_L, K=best_K, val_tuple_errors=val_err,
                 acc_read=np.nan, acc_write=np.nan, acc_dir=np.nan,
                 mean_acc=float(acc),
                 tuple_errors=int(total - correct), n_predictions=int(total),
                 perfect_tapes=-1, n_test=200, eval_s=eval_t)]


def main():
    lines = []
    def log(msg=""):
        print(msg, flush=True); lines.append(str(msg))
    rows = []
    for variant in ('original', 'noread'):
        for name in TASK_ORDER:
            rows += run_tm_task(name, log, variant=variant)
    rows += run_dyck(log=log)

    out_csv = os.path.join(HERE, f'parrot_benchmark_results{SUFFIX}.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log(f"\nWrote {out_csv}")

    log(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    log(f"{'task':>13s}  {'variant':>8s}  {'L':>2s}  {'K':>3s}  "
        f"{'mean_acc':>8s}  {'errors':>15s}  {'perfect':>9s}")
    for r in rows:
        terr_str = f"{r['tuple_errors']}/{r['n_predictions']}"
        perfect_str = (f"{r['perfect_tapes']}/{r['n_test']}"
                       if r['perfect_tapes'] >= 0 else '   -   ')
        log(f"{r['task']:>13s}  {r['variant']:>8s}  {r['L']:>2d}  {r['K']:>3d}  "
            f"{r['mean_acc']:>8.4f}  {terr_str:>15s}  {perfect_str:>9s}")

    log_path = os.path.join(HERE, f'parrot_benchmark_log{SUFFIX}.txt')
    with open(log_path, 'w') as f: f.write('\n'.join(lines))


if __name__ == "__main__":
    main()
