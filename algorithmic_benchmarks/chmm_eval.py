"""CHMM (Cloned HMM) on the algorithmic / TM benchmarks.

Mirrors `parrot_eval.py` / `hpylm_eval.py` etc. so CHMM numbers slot
into the same TM tuple-error comparison under a leakage-free
train/val/test protocol.

Only hyperparam to tune: K (number of clones per emission).

Same protocol:
  - Reduced 3-tuple alphabet per task
  - Train on training tapes, val-tune on stretched val, eval on test
  - Tuple-error count, with conditional masking on the actual next read

Output: algorithmic_benchmarks/chmm_benchmark_results.csv
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "chmm_tests", "naturecomm_cscg"))

import dyck1                                          # noqa: E402
from chmm_actions import CHMM, forward                # noqa: E402
from _tm_task_config import (                          # noqa: E402
    TM_TASKS, TASK_ORDER, simulate_train_val_test, SUFFIX)

K_GRID = [2, 4, 8]
N_EM_ITERS = 50


# --------------------------------------------------------------------
# TM tokenisation (same as elsewhere)
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
    out = []
    for row in tape:
        if int(row[0]) == -1:
            continue
        key = (int(row[1]), int(row[2]), int(row[3]))
        if key in tuple_to_id:
            out.append(tuple_to_id[key])
    return np.asarray(out, dtype=np.int64), 0


# --------------------------------------------------------------------
# CHMM scoring
# --------------------------------------------------------------------
def chmm_predict_next_dist(model, alpha_t, x_t, n_clones, state_loc):
    gs, ge = int(state_loc[x_t]), int(state_loc[x_t + 1])
    T = model.T[0]
    full = alpha_t @ T[gs:ge, :]
    full = np.maximum(full, 0)
    p_next = np.zeros(len(n_clones), dtype=np.float64)
    for e in range(len(n_clones)):
        s, t = int(state_loc[e]), int(state_loc[e + 1])
        p_next[e] = full[s:t].sum()
    z = p_next.sum()
    if z > 0:
        p_next /= z
    return p_next


def chmm_eval_tm_reduced(model, test_tapes, tuple_to_id, id_to_tuple,
                         n_clones):
    state_loc = np.hstack(([0], n_clones)).cumsum().astype(np.int64)
    by_read = {}
    for tid, tup in enumerate(id_to_tuple):
        by_read.setdefault(tup[0], []).append(tid)
    correct = np.zeros(3, dtype=np.int64)
    total = np.zeros(3, dtype=np.int64)
    tuple_errors, perfect = 0, 0
    for tape in test_tapes:
        x, _ = encode_reduced(tape, tuple_to_id)
        if len(x) < 2:
            perfect += 1; continue
        a = np.zeros_like(x)
        _, mess_fwd = forward(model.T.transpose(0, 2, 1), model.Pi_x,
                              model.n_clones, x, a, store_messages=True)
        mess_loc = np.hstack(([0], n_clones[x])).cumsum().astype(np.int64)
        tape_err = 0
        for t in range(len(x) - 1):
            ms, me = int(mess_loc[t]), int(mess_loc[t + 1])
            alpha_t = mess_fwd[ms:me].astype(np.float64)
            p_next = chmm_predict_next_dist(
                model, alpha_t, int(x[t]), n_clones, state_loc)
            actual_tup = id_to_tuple[int(x[t + 1])]
            cands = by_read.get(actual_tup[0], [])
            if not cands:
                continue
            best_tid = max(cands, key=lambda c: p_next[c])
            pred_tup = id_to_tuple[best_tid]
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


def chmm_eval_dyck(model, test_seqs, n_clones):
    state_loc = np.hstack(([0], n_clones)).cumsum().astype(np.int64)
    correct, total = 0, 0
    for x in test_seqs:
        if len(x) < 2:
            continue
        a = np.zeros_like(x)
        _, mess_fwd = forward(model.T.transpose(0, 2, 1), model.Pi_x,
                              model.n_clones, x, a, store_messages=True)
        mess_loc = np.hstack(([0], n_clones[x])).cumsum().astype(np.int64)
        for t in range(len(x) - 1):
            actual_next = int(x[t + 1])
            if actual_next == dyck1.END:
                continue
            ms, me = int(mess_loc[t]), int(mess_loc[t + 1])
            alpha_t = mess_fwd[ms:me].astype(np.float64)
            p_next = chmm_predict_next_dist(
                model, alpha_t, int(x[t]), n_clones, state_loc)
            cand = [dyck1.OPEN, dyck1.CLOSE]
            pred = max(cand, key=lambda c: p_next[c])
            total += 1
            if pred == actual_next:
                correct += 1
    return correct / max(total, 1), total, correct


# --------------------------------------------------------------------
# Per-task runner
# --------------------------------------------------------------------
def fit_chmm(train_seqs, nA, K):
    n_clones = np.full(nA, K, dtype=np.int64)
    x_train = np.concatenate(train_seqs).astype(np.int64)
    a_train = np.zeros_like(x_train)
    model = CHMM(n_clones=n_clones, x=x_train, a=a_train,
                 pseudocount=1e-3, seed=0)
    model.learn_em_T(x_train, a_train, n_iter=N_EM_ITERS, term_early=True)
    return model, n_clones


def run_tm_task(name, log, variant='original'):
    cfg = TM_TASKS[name]
    n_test = cfg['n_test']
    log(f"\n{'='*66}\nTASK: {name} (CHMM, variant={variant})\n{'='*66}")
    tr_runs, val_runs, te_runs = simulate_train_val_test(name, variant)
    tuple_to_id, id_to_tuple = reduced_alphabet(tr_runs)
    nA = len(id_to_tuple)
    log(f"  alphabet={nA}, train={len(tr_runs)}, val={len(val_runs)}, "
        f"test={len(te_runs)}")
    train_seqs = [encode_reduced(t, tuple_to_id)[0] for t in tr_runs]
    train_seqs = [s for s in train_seqs if len(s) > 0]

    val_scores = []
    for K in K_GRID:
        t0 = time.time()
        model, n_clones = fit_chmm(train_seqs, nA, K)
        train_t = time.time() - t0
        _, _, terr_v, _ = chmm_eval_tm_reduced(
            model, val_runs, tuple_to_id, id_to_tuple, n_clones)
        log(f"  K={K} (n_states={K*nA}): val_errors={terr_v}  train={train_t:.1f}s")
        val_scores.append((terr_v, K, model, n_clones, train_t))
    val_scores.sort(key=lambda x: x[0])
    val_terr, best_K, best_model, best_clones, best_train_t = val_scores[0]
    log(f"  Val pick: K={best_K}  val_errors={val_terr}")

    t0 = time.time()
    acc, total, terr, perf = chmm_eval_tm_reduced(
        best_model, te_runs, tuple_to_id, id_to_tuple, best_clones)
    eval_t = time.time() - t0
    n_pred = int(total[0])
    log(f"  CHMM K={best_K}: read={acc[0]:.4f} write={acc[1]:.4f} "
        f"dir={acc[2]:.4f} mean={acc.mean():.4f}")
    log(f"  CHMM tuple errors: {terr}/{n_pred} "
        f"({100*terr/max(n_pred,1):.3f}%), perfect: {perf}/{n_test}  "
        f"(eval={eval_t:.1f}s)")
    return [dict(task=name, variant=variant, model='CHMM',
                 K=best_K, val_tuple_errors=val_terr,
                 acc_read=acc[0], acc_write=acc[1], acc_dir=acc[2],
                 mean_acc=float(acc.mean()),
                 tuple_errors=terr, n_predictions=n_pred,
                 perfect_tapes=perf, n_test=n_test,
                 train_s=best_train_t, eval_s=eval_t)]


def run_dyck(log):
    log(f"\n{'='*66}\nTASK: dyck1 (CHMM)\n{'='*66}")
    tr = dyck1.simulate(1000, max_depth=4, length_min=4, length_max=200,
                        seed=42)
    val = dyck1.simulate(100, max_depth=6, length_min=4, length_max=300,
                         seed=7)
    te = dyck1.simulate(200, max_depth=8, length_min=4, length_max=400,
                        seed=123)
    nA = dyck1.ALPHABET_SIZE
    train_seqs = [s.astype(np.int64) for s in tr['sequences']]
    val_seqs = [s.astype(np.int64) for s in val['sequences']]
    te_seqs = list(te['sequences'])

    val_scores = []
    for K in K_GRID:
        t0 = time.time()
        model, n_clones = fit_chmm(train_seqs, nA, K)
        train_t = time.time() - t0
        acc_v, total_v, correct_v = chmm_eval_dyck(model, val_seqs, n_clones)
        terr_v = total_v - correct_v
        log(f"  K={K}: val_errors={terr_v}  train={train_t:.1f}s")
        val_scores.append((terr_v, K, model, n_clones, train_t))
    val_scores.sort(key=lambda x: x[0])
    val_terr, best_K, best_model, best_clones, best_train_t = val_scores[0]
    log(f"  Val pick: K={best_K}  val_errors={val_terr}")

    t0 = time.time()
    acc, total, correct = chmm_eval_dyck(best_model, te_seqs, best_clones)
    eval_t = time.time() - t0
    log(f"  CHMM dyck1 acc: {acc:.4f} ({correct}/{total}, eval={eval_t:.1f}s)")
    return [dict(task='dyck1', variant='n/a', model='CHMM',
                 K=best_K, val_tuple_errors=val_terr,
                 acc_read=np.nan, acc_write=np.nan, acc_dir=np.nan,
                 mean_acc=float(acc),
                 tuple_errors=int(total - correct), n_predictions=int(total),
                 perfect_tapes=-1, n_test=200,
                 train_s=best_train_t, eval_s=eval_t)]


def main():
    lines = []
    def log(msg=""):
        print(msg, flush=True); lines.append(str(msg))
    rows = []
    for variant in ('original', 'noread'):
        for name in TASK_ORDER:
            rows += run_tm_task(name, log, variant=variant)
    rows += run_dyck(log=log)

    out_csv = os.path.join(HERE, f'chmm_benchmark_results{SUFFIX}.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log(f"\nWrote {out_csv}")
    log(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    log(f"{'task':>13s}  {'variant':>8s}  {'K':>2s}  "
        f"{'mean_acc':>8s}  {'errors':>15s}  {'perfect':>9s}")
    for r in rows:
        terr_str = f"{r['tuple_errors']}/{r['n_predictions']}"
        perfect_str = (f"{r['perfect_tapes']}/{r['n_test']}"
                       if r['perfect_tapes'] >= 0 else '   -   ')
        log(f"{r['task']:>13s}  {r['variant']:>8s}  {r['K']:>2d}  "
            f"{r['mean_acc']:>8.4f}  {terr_str:>15s}  {perfect_str:>9s}")
    log_path = os.path.join(HERE, f'chmm_benchmark_log{SUFFIX}.txt')
    with open(log_path, 'w') as f: f.write('\n'.join(lines))


if __name__ == "__main__":
    main()
