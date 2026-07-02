"""PPM-D (absolute-discount n-gram backoff) on the algorithmic / TM benchmarks.

Mirrors `parrot_eval.py` exactly so PPM numbers slot into Tables 8+9.

Same protocol:
  - Reduced 3-tuple alphabet per task
  - Train on training tapes, val-tune on stretched val, evaluate on test
  - Tuple-error count, with conditional masking on the actual next read

Variant grid (val-tuned per task):
  - max_depth ∈ {3, 5, 8, 12}
  - discount  ∈ {0.25, 0.5, 0.75}
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

import dyck1                                          # noqa: E402
from discrete_ppm import PPMPool                       # noqa: E402
from _tm_task_config import (                          # noqa: E402
    TM_TASKS, TASK_ORDER, simulate_train_val_test, SUFFIX)

DEPTHS = [3, 5, 8, 12]
DISCOUNTS = [0.25, 0.5, 0.75]


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
    out = []
    for row in tape:
        if int(row[0]) == -1:
            continue
        key = (int(row[1]), int(row[2]), int(row[3]))
        if key in tuple_to_id:
            out.append(tuple_to_id[key])
    return np.asarray(out, dtype=np.int64), 0


def ppm_eval_tm_reduced(pool, test_tapes, tuple_to_id, id_to_tuple):
    by_read = {}
    for tid, tup in enumerate(id_to_tuple):
        by_read.setdefault(tup[0], []).append(tid)
    correct = np.zeros(3, dtype=np.int64)
    total = np.zeros(3, dtype=np.int64)
    tuple_errors, perfect = 0, 0
    nA = len(id_to_tuple)
    for tape in test_tapes:
        x, _ = encode_reduced(tape, tuple_to_id)
        if len(x) < 2:
            perfect += 1; continue
        tape_err = 0
        for t in range(len(x) - 1):
            actual_tup = id_to_tuple[int(x[t + 1])]
            cands = by_read.get(actual_tup[0], [])
            if not cands:
                continue
            prefix = x[: t + 1]
            mask = np.zeros(nA, dtype=bool)
            mask[cands] = True
            best_tid = pool.predict_argmax(prefix, h=1, mask=mask,
                                           alpha_prior=0.01)
            pred_tup = id_to_tuple[int(best_tid)]
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


def ppm_eval_dyck(pool, test_seqs):
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
            best = pool.predict_argmax(prefix, h=1, mask=cand_mask,
                                        alpha_prior=0.01)
            total += 1
            if best == actual_next:
                correct += 1
    return correct / max(total, 1), total, correct


def run_tm_task(name, log, variant='original'):
    cfg = TM_TASKS[name]
    n_test = cfg['n_test']
    log(f"\n{'='*66}\nTASK: {name} (variant={variant})\n{'='*66}")
    tr_runs, val_runs, te_runs = simulate_train_val_test(name, variant)
    tuple_to_id, id_to_tuple = reduced_alphabet(tr_runs)
    nA = len(id_to_tuple)
    log(f"  alphabet={nA}, train={len(tr_runs)}, val={len(val_runs)}, "
        f"test={len(te_runs)}")
    train_seqs = [encode_reduced(t, tuple_to_id)[0] for t in tr_runs]
    train_seqs = [s for s in train_seqs if len(s) > 0]

    best = (float('inf'), None, None)
    for D in DEPTHS:
        for disc in DISCOUNTS:
            pool = PPMPool(train_seqs, alphabet_size=nA,
                            max_depth=D, discount=disc)
            _, _, terr_v, _ = ppm_eval_tm_reduced(pool, val_runs,
                                                  tuple_to_id, id_to_tuple)
            if terr_v < best[0]:
                best = (terr_v, (D, disc), pool)
    val_terr, (best_D, best_disc), best_pool = best
    log(f"  Val pick: D={best_D} d={best_disc} → val_errors={val_terr}")

    t0 = time.time()
    acc, total, terr, perf = ppm_eval_tm_reduced(best_pool, te_runs,
                                                    tuple_to_id, id_to_tuple)
    eval_t = time.time() - t0
    n_pred = int(total[0])
    log(f"  PPM test: tuple errors {terr}/{n_pred} ({100*terr/max(n_pred,1):.3f}%), "
        f"perfect: {perf}/{n_test} ({eval_t:.1f}s)")
    return [dict(task=name, variant=variant, model='ppm',
                 max_depth=best_D, discount=best_disc,
                 val_tuple_errors=val_terr,
                 acc_read=acc[0], acc_write=acc[1], acc_dir=acc[2],
                 mean_acc=float(acc.mean()),
                 tuple_errors=terr, n_predictions=n_pred,
                 perfect_tapes=perf, n_test=n_test, eval_s=eval_t)]


def run_dyck(log):
    name = 'dyck1'
    log(f"\n{'='*66}\nTASK: {name}\n{'='*66}")
    tr = dyck1.simulate(1000, max_depth=4, length_min=4, length_max=200,
                        seed=42)
    val = dyck1.simulate(100, max_depth=6, length_min=4, length_max=300,
                         seed=7)
    te = dyck1.simulate(200, max_depth=8, length_min=4, length_max=400,
                        seed=123)
    tr_seqs = [s.astype(np.int64) for s in tr['sequences']]
    val_seqs = [s.astype(np.int64) for s in val['sequences']]
    nA = dyck1.ALPHABET_SIZE
    best = (float('inf'), None, None)
    for D in DEPTHS:
        for disc in DISCOUNTS:
            pool = PPMPool(tr_seqs, alphabet_size=nA,
                            max_depth=D, discount=disc)
            acc, total, _ = ppm_eval_dyck(pool, val_seqs)
            terr_v = total - int(round(acc * total))
            if terr_v < best[0]:
                best = (terr_v, (D, disc), pool)
    val_terr, (best_D, best_disc), best_pool = best
    log(f"  Val pick: D={best_D} d={best_disc} → val_errors={val_terr}")
    t0 = time.time()
    acc, total, correct = ppm_eval_dyck(best_pool, te['sequences'])
    eval_t = time.time() - t0
    log(f"  PPM test: dyck1 accuracy {acc:.4f} ({correct}/{total}, {eval_t:.1f}s)")
    return [dict(task=name, variant='n/a', model='ppm',
                 max_depth=best_D, discount=best_disc,
                 val_tuple_errors=val_terr,
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

    out_csv = os.path.join(HERE, f'ppm_benchmark_results{SUFFIX}.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log(f"\nWrote {out_csv}")
    log(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    log(f"{'task':>13s}  {'variant':>8s}  {'D':>2s} {'d':>5s}  "
        f"{'mean_acc':>8s}  {'errors':>15s}  {'perfect':>9s}")
    for r in rows:
        terr_str = f"{r['tuple_errors']}/{r['n_predictions']}"
        perfect_str = (f"{r['perfect_tapes']}/{r['n_test']}"
                       if r['perfect_tapes'] >= 0 else '   -   ')
        log(f"{r['task']:>13s}  {r['variant']:>8s}  {r['max_depth']:>2d} {r['discount']:>5.2f}  "
            f"{r['mean_acc']:>8.4f}  {terr_str:>15s}  {perfect_str:>9s}")


if __name__ == "__main__":
    main()
