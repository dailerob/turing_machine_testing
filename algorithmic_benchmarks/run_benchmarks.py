"""GDC vs CHMM benchmark harness across the 4 algorithmic tasks.

For each TM-trace task (parity, increment, reverse) we generate
training tapes at small input sizes and test tapes at larger input
sizes, then evaluate Test 2 (Reduced 3-column): predict
(read, write, dir) tokens 1-step-ahead conditional on the actual
next read symbol.  This is the same protocol as the binary-adder
experiment.

For Dyck-1 we generate sequences at small max_depth and test at
larger max_depth, then evaluate next-token accuracy on the {OPEN,
CLOSE} positions (skipping END).

GDC uses the same hyperparams the binary-adder run used (alpha=0.99,
theta=0.005, transition_type='self_loop_two_step',
initial_dist='sequence_starts').
CHMM is run at K in {2, 4, 8} clones per emission.

Outputs:
    algorithmic_benchmarks/benchmark_results.csv
    stdout / log file
"""

from __future__ import annotations

import os, sys, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "chmm_tests", "naturecomm_cscg"))

import parity_tm, increment_tm, reverse_tm, dyck1  # noqa: E402
from _tm_common import apply_noread_to_runs  # noqa: E402
from generative_dense_chain import GenerativeDenseChain  # noqa: E402
from chmm_actions import CHMM, forward  # noqa: E402
from binary_alphabet_adder import (  # noqa: E402
    simulate_random_binary_alphabet_adders, BINARY_ALPHABET_ADDER)

K_GRID = [2, 4, 8]
N_EM_ITERS = 50
GDC_PARAMS = dict(alpha=0.99, theta=0.005, gamma=0.000,
                  transition_type='self_loop_two_step',
                  initial_dist='sequence_starts')


# --------------------------------------------------------------------
# TM-trace tokenisation (reduced 3-column)
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
# CHMM helpers
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
    by_read = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
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


# --------------------------------------------------------------------
# GDC helpers (TM tasks, Reduced 3-col)
# --------------------------------------------------------------------
def gdc_eval_tm_reduced(gdc, test_tapes):
    correct = np.zeros(3, dtype=np.int64)
    total = np.zeros(3, dtype=np.int64)
    tuple_errors, perfect = 0, 0
    for tape in test_tapes:
        if len(tape) < 2:
            perfect += 1; continue
        tape_red = tape[:, 1:4].astype(np.int64)
        # Skip halt-row at end
        valid_mask = (tape[:, 0] != -1)
        tape_red = tape_red[valid_mask]
        if len(tape_red) < 2:
            perfect += 1; continue
        _, hist = gdc.forward_pass(tape_red, return_history=True)
        tape_err = 0
        for t in range(len(tape_red) - 1):
            state_dist = hist[t]
            forecast = gdc.forecast(state_dist, n_steps=1)
            actual_next = tape_red[t + 1]
            cond = np.array([actual_next[0], np.nan, np.nan])
            pred = gdc.greedy_sample(forecast, conditional=cond)
            mismatched = False
            for pos in range(3):
                if not np.isnan(pred[pos]):
                    total[pos] += 1
                    if int(pred[pos]) == int(actual_next[pos]):
                        correct[pos] += 1
                    else:
                        mismatched = True
            if mismatched:
                tape_err += 1; tuple_errors += 1
        if tape_err == 0:
            perfect += 1
    acc = correct / np.maximum(total, 1)
    return acc, total, tuple_errors, perfect


# --------------------------------------------------------------------
# Dyck-1 evaluation
# --------------------------------------------------------------------
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
                continue  # skip end-of-seq positions
            ms, me = int(mess_loc[t]), int(mess_loc[t + 1])
            alpha_t = mess_fwd[ms:me].astype(np.float64)
            p_next = chmm_predict_next_dist(
                model, alpha_t, int(x[t]), n_clones, state_loc)
            # restrict prediction to {OPEN, CLOSE}
            cand = [dyck1.OPEN, dyck1.CLOSE]
            pred = max(cand, key=lambda c: p_next[c])
            total += 1
            if pred == actual_next:
                correct += 1
    return correct / max(total, 1), total, correct


def gdc_eval_dyck(gdc, test_seqs):
    correct, total = 0, 0
    for x in test_seqs:
        if len(x) < 2:
            continue
        x_col = x.reshape(-1, 1).astype(np.int64)
        _, hist = gdc.forward_pass(x_col, return_history=True)
        for t in range(len(x) - 1):
            actual_next = int(x[t + 1])
            if actual_next == dyck1.END:
                continue
            forecast = gdc.forecast(hist[t], n_steps=1)
            cond = np.array([np.nan])
            pred = gdc.greedy_sample(forecast, conditional=cond)
            if np.isnan(pred[0]):
                continue
            pred_int = int(pred[0])
            # If predicted END, fall back to whichever of OPEN/CLOSE
            # has higher probability under forecast.  Simpler: only
            # count mismatches as wrong.
            total += 1
            if pred_int == actual_next:
                correct += 1
    return correct / max(total, 1), total, correct


# --------------------------------------------------------------------
# Per-task runners
# --------------------------------------------------------------------
def run_tm_task(name, module, train_range, test_range,
                n_train, n_test, max_steps, log, variant='original'):
    nr = (variant == 'noread')
    log(f"\n{'='*66}\nTASK: {name} (TM, reduced 3-col, variant={variant})"
        f"\n{'='*66}")
    tr = module.simulate(n_train, train_range, max_steps=max_steps,
                         seed=42, noread=nr)
    te = module.simulate(n_test, test_range, max_steps=max_steps * 4,
                         seed=123, noread=nr)
    log(f"  train: n={n_train} len_range={train_range}, "
        f"halted={sum(tr['halted_flags'])}/{n_train}, "
        f"correct={sum(tr['correct'])}/{n_train}")
    log(f"  test:  n={n_test}  len_range={test_range}, "
        f"halted={sum(te['halted_flags'])}/{n_test}, "
        f"correct={sum(te['correct'])}/{n_test}")
    train_lens = [t.shape[0] for t in tr['runs']]
    test_lens = [t.shape[0] for t in te['runs']]
    log(f"  trace lens — train: min={min(train_lens)}, "
        f"max={max(train_lens)}, mean={np.mean(train_lens):.1f}; "
        f"test: min={min(test_lens)}, max={max(test_lens)}, "
        f"mean={np.mean(test_lens):.1f}")

    tuple_to_id, id_to_tuple = reduced_alphabet(tr['runs'])
    nA = len(id_to_tuple)
    log(f"  reduced alphabet size: {nA}")
    log(f"    tuples: {id_to_tuple}")

    rows = []

    # GDC
    train_red = [t[:, 1:4][t[:, 0] != -1].astype(np.int64) for t in tr['runs']]
    train_red = [t for t in train_red if len(t) > 0]
    t0 = time.time()
    gdc = GenerativeDenseChain(train_red, **GDC_PARAMS)
    gdc_train_t = time.time() - t0
    log(f"  GDC built: {gdc.n_states} hidden states "
        f"(train={gdc_train_t:.2f}s)")
    t0 = time.time()
    acc, total, terr, perf = gdc_eval_tm_reduced(gdc, te['runs'])
    gdc_eval_t = time.time() - t0
    n_pred = int(total[0])
    log(f"  GDC acc: read={acc[0]:.4f} write={acc[1]:.4f} dir={acc[2]:.4f} "
        f"mean={acc.mean():.4f}")
    log(f"  GDC tuple errors: {terr}/{n_pred} "
        f"({100*terr/max(n_pred,1):.3f}%), perfect tapes: {perf}/{n_test} "
        f"(eval={gdc_eval_t:.1f}s)")
    rows.append(dict(task=name, variant=variant, model='GDC',
                     K_or_alpha='alpha=0.99',
                     n_states=gdc.n_states, train_s=gdc_train_t,
                     eval_s=gdc_eval_t, acc_read=acc[0], acc_write=acc[1],
                     acc_dir=acc[2], mean_acc=acc.mean(),
                     tuple_errors=terr, n_predictions=n_pred,
                     perfect_tapes=perf, n_test=n_test))

    # CHMM
    x_train = np.concatenate(
        [encode_reduced(t, tuple_to_id)[0] for t in tr['runs']]
    ).astype(np.int64)
    a_train = np.zeros_like(x_train)
    log(f"  CHMM training tokens: {len(x_train)}")
    for K in K_GRID:
        n_clones = np.full(nA, K, dtype=np.int64)
        n_states = int(K * nA)
        t0 = time.time()
        model = CHMM(n_clones=n_clones, x=x_train, a=a_train,
                     pseudocount=1e-3, seed=0)
        model.learn_em_T(x_train, a_train, n_iter=N_EM_ITERS,
                         term_early=True)
        chmm_train_t = time.time() - t0
        t0 = time.time()
        acc, total, terr, perf = chmm_eval_tm_reduced(
            model, te['runs'], tuple_to_id, id_to_tuple, n_clones)
        chmm_eval_t = time.time() - t0
        n_pred = int(total[0])
        log(f"  CHMM K={K} (n_states={n_states}): "
            f"read={acc[0]:.4f} write={acc[1]:.4f} dir={acc[2]:.4f} "
            f"mean={acc.mean():.4f}")
        log(f"    errors: {terr}/{n_pred} "
            f"({100*terr/max(n_pred,1):.3f}%), perfect: {perf}/{n_test} "
            f"(train={chmm_train_t:.1f}s, eval={chmm_eval_t:.1f}s)")
        rows.append(dict(task=name, variant=variant, model='CHMM',
                         K_or_alpha=f'K={K}',
                         n_states=n_states, train_s=chmm_train_t,
                         eval_s=chmm_eval_t, acc_read=acc[0],
                         acc_write=acc[1], acc_dir=acc[2],
                         mean_acc=acc.mean(),
                         tuple_errors=terr, n_predictions=n_pred,
                         perfect_tapes=perf, n_test=n_test))
    return rows


def run_binary_adder_task(log, variant='original',
                          n_train=200, n_test=10):
    """Mirror chmm_tests/run_chmm_turing_adder.py but supports variant."""
    name = 'binary_adder'
    log(f"\n{'='*66}\nTASK: {name} (TM, reduced 3-col, variant={variant})"
        f"\n{'='*66}")
    tr = simulate_random_binary_alphabet_adders(
        n_runs=n_train, num_range=(0, 32), max_steps=200_000, seed=42)
    te = simulate_random_binary_alphabet_adders(
        n_runs=n_test, num_range=(0, 1000), max_steps=200_000, seed=123)
    log(f"  train n={n_train} (B in [0,32]); test n={n_test} (B in [0,1000])")
    log(f"  halted train={sum(tr['halted_flags'])}/{n_train}, "
        f"test={sum(te['halted_flags'])}/{n_test}")
    if variant == 'noread':
        # Need to make the SAME symbol_encoding cover both train and test
        # so the NO_READ index is consistent.  Merge encodings first.
        merged_se = dict(tr['symbol_encoding'])
        for k, _v in te['symbol_encoding'].items():
            if k not in merged_se:
                merged_se[k] = len(merged_se)
        merged_st = dict(tr['state_encoding'])
        for k, _v in te['state_encoding'].items():
            if k not in merged_st:
                merged_st[k] = len(merged_st)
        # Re-encode runs through the merged encodings:
        # The numerical encodings differ between train and test if their
        # encodings differ.  In practice, both calls produce identical
        # encodings since the program is the same; we still defensively
        # rebuild via apply_noread_to_runs which uses the encodings as-is.
        tr_runs, train_se = apply_noread_to_runs(
            tr['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)
        te_runs, test_se = apply_noread_to_runs(
            te['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)
        tr['runs'] = tr_runs; te['runs'] = te_runs
    train_lens = [t.shape[0] for t in tr['runs']]
    test_lens = [t.shape[0] for t in te['runs']]
    log(f"  trace lens — train: min={min(train_lens)}, "
        f"max={max(train_lens)}, mean={np.mean(train_lens):.1f}; "
        f"test: min={min(test_lens)}, max={max(test_lens)}, "
        f"mean={np.mean(test_lens):.1f}")

    tuple_to_id, id_to_tuple = reduced_alphabet(tr['runs'])
    nA = len(id_to_tuple)
    log(f"  reduced alphabet size: {nA}")
    log(f"    tuples: {id_to_tuple}")

    rows = []
    train_red = [t[:, 1:4][t[:, 0] != -1].astype(np.int64)
                 for t in tr['runs']]
    train_red = [t for t in train_red if len(t) > 0]
    t0 = time.time()
    gdc = GenerativeDenseChain(train_red, **GDC_PARAMS)
    gdc_train_t = time.time() - t0
    log(f"  GDC built: {gdc.n_states} hidden states "
        f"(train={gdc_train_t:.2f}s)")
    t0 = time.time()
    acc, total, terr, perf = gdc_eval_tm_reduced(gdc, te['runs'])
    gdc_eval_t = time.time() - t0
    n_pred = int(total[0])
    log(f"  GDC acc: read={acc[0]:.4f} write={acc[1]:.4f} "
        f"dir={acc[2]:.4f} mean={acc.mean():.4f}")
    log(f"  GDC errors: {terr}/{n_pred} "
        f"({100*terr/max(n_pred,1):.3f}%), perfect: {perf}/{n_test} "
        f"(eval={gdc_eval_t:.1f}s)")
    rows.append(dict(task=name, variant=variant, model='GDC',
                     K_or_alpha='alpha=0.99',
                     n_states=gdc.n_states, train_s=gdc_train_t,
                     eval_s=gdc_eval_t, acc_read=acc[0],
                     acc_write=acc[1], acc_dir=acc[2],
                     mean_acc=acc.mean(),
                     tuple_errors=terr, n_predictions=n_pred,
                     perfect_tapes=perf, n_test=n_test))
    x_train = np.concatenate(
        [encode_reduced(t, tuple_to_id)[0] for t in tr['runs']]
    ).astype(np.int64)
    a_train = np.zeros_like(x_train)
    log(f"  CHMM training tokens: {len(x_train)}")
    for K in K_GRID:
        n_clones = np.full(nA, K, dtype=np.int64)
        n_states = int(K * nA)
        t0 = time.time()
        model = CHMM(n_clones=n_clones, x=x_train, a=a_train,
                     pseudocount=1e-3, seed=0)
        model.learn_em_T(x_train, a_train, n_iter=N_EM_ITERS,
                         term_early=True)
        chmm_train_t = time.time() - t0
        t0 = time.time()
        acc, total, terr, perf = chmm_eval_tm_reduced(
            model, te['runs'], tuple_to_id, id_to_tuple, n_clones)
        chmm_eval_t = time.time() - t0
        n_pred = int(total[0])
        log(f"  CHMM K={K} (n_states={n_states}): "
            f"read={acc[0]:.4f} write={acc[1]:.4f} dir={acc[2]:.4f} "
            f"mean={acc.mean():.4f}")
        log(f"    errors: {terr}/{n_pred} "
            f"({100*terr/max(n_pred,1):.3f}%), perfect: {perf}/{n_test} "
            f"(train={chmm_train_t:.1f}s, eval={chmm_eval_t:.1f}s)")
        rows.append(dict(task=name, variant=variant, model='CHMM',
                         K_or_alpha=f'K={K}',
                         n_states=n_states, train_s=chmm_train_t,
                         eval_s=chmm_eval_t, acc_read=acc[0],
                         acc_write=acc[1], acc_dir=acc[2],
                         mean_acc=acc.mean(),
                         tuple_errors=terr, n_predictions=n_pred,
                         perfect_tapes=perf, n_test=n_test))
    return rows


def run_dyck_task(log):
    name = 'dyck1'
    log(f"\n{'='*66}\nTASK: {name} (sequence)\n{'='*66}")
    tr = dyck1.simulate(1000, max_depth=4, length_min=4, length_max=200,
                        seed=42)
    te = dyck1.simulate(200, max_depth=8, length_min=4, length_max=400,
                        seed=123)
    train_lens = [len(s) for s in tr['sequences']]
    test_lens = [len(s) for s in te['sequences']]
    log(f"  train: n=1000 max_depth=4, len min/max/mean: "
        f"{min(train_lens)}/{max(train_lens)}/{np.mean(train_lens):.1f}")
    log(f"  test:  n=200  max_depth=8, len min/max/mean: "
        f"{min(test_lens)}/{max(test_lens)}/{np.mean(test_lens):.1f}")

    rows = []
    nA = dyck1.ALPHABET_SIZE

    # GDC: each sequence is column-shaped
    train_seqs = [s.reshape(-1, 1).astype(np.int64) for s in tr['sequences']]
    t0 = time.time()
    gdc = GenerativeDenseChain(train_seqs, **GDC_PARAMS)
    gdc_train_t = time.time() - t0
    log(f"  GDC built: {gdc.n_states} hidden states "
        f"(train={gdc_train_t:.2f}s)")
    t0 = time.time()
    acc, total, correct = gdc_eval_dyck(gdc, te['sequences'])
    gdc_eval_t = time.time() - t0
    log(f"  GDC next-symbol accuracy: {acc:.4f} "
        f"({correct}/{total}, eval={gdc_eval_t:.1f}s)")
    rows.append(dict(task=name, variant='n/a', model='GDC',
                     K_or_alpha='alpha=0.99',
                     n_states=gdc.n_states, train_s=gdc_train_t,
                     eval_s=gdc_eval_t,
                     acc_read=np.nan, acc_write=np.nan, acc_dir=np.nan,
                     mean_acc=acc, tuple_errors=total - correct,
                     n_predictions=total, perfect_tapes=-1, n_test=200))

    # CHMM
    x_train = np.concatenate(tr['sequences']).astype(np.int64)
    a_train = np.zeros_like(x_train)
    log(f"  CHMM training tokens: {len(x_train)}")
    for K in K_GRID:
        n_clones = np.full(nA, K, dtype=np.int64)
        n_states = int(K * nA)
        t0 = time.time()
        model = CHMM(n_clones=n_clones, x=x_train, a=a_train,
                     pseudocount=1e-3, seed=0)
        model.learn_em_T(x_train, a_train, n_iter=N_EM_ITERS,
                         term_early=True)
        chmm_train_t = time.time() - t0
        t0 = time.time()
        acc, total, correct = chmm_eval_dyck(model, te['sequences'],
                                             n_clones)
        chmm_eval_t = time.time() - t0
        log(f"  CHMM K={K} (n_states={n_states}): "
            f"next-symbol accuracy: {acc:.4f} "
            f"({correct}/{total}, train={chmm_train_t:.1f}s, "
            f"eval={chmm_eval_t:.1f}s)")
        rows.append(dict(task=name, variant='n/a', model='CHMM',
                         K_or_alpha=f'K={K}',
                         n_states=n_states, train_s=chmm_train_t,
                         eval_s=chmm_eval_t,
                         acc_read=np.nan, acc_write=np.nan, acc_dir=np.nan,
                         mean_acc=acc, tuple_errors=total - correct,
                         n_predictions=total, perfect_tapes=-1, n_test=200))
    return rows


def main():
    lines = []

    def log(msg=""):
        print(msg, flush=True); lines.append(str(msg))

    rows = []
    for variant in ('original', 'noread'):
        rows += run_tm_task('parity', parity_tm, train_range=(3, 8),
                            test_range=(16, 32), n_train=300, n_test=20,
                            max_steps=200, log=log, variant=variant)
        rows += run_tm_task('increment', increment_tm, train_range=(1, 5),
                            test_range=(8, 12), n_train=300, n_test=20,
                            max_steps=200, log=log, variant=variant)
        rows += run_tm_task('reverse', reverse_tm, train_range=(3, 6),
                            test_range=(10, 16), n_train=300, n_test=20,
                            max_steps=10000, log=log, variant=variant)
        rows += run_binary_adder_task(log=log, variant=variant,
                                      n_train=200, n_test=10)
    rows += run_dyck_task(log=log)

    out_csv = os.path.join(HERE, 'benchmark_results.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log(f"\nWrote {out_csv}")

    log(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    log(f"{'task':>13s}  {'variant':>8s}  {'model':>5s}  {'config':>10s}  "
        f"{'states':>7s}  {'mean_acc':>8s}  {'errors':>13s}  "
        f"{'perfect':>7s}")
    for r in rows:
        terr_str = f"{r['tuple_errors']}/{r['n_predictions']}"
        perfect_str = (f"{r['perfect_tapes']}/{r['n_test']}"
                       if r['perfect_tapes'] >= 0 else '   -   ')
        log(f"{r['task']:>13s}  {r['variant']:>8s}  {r['model']:>5s}  "
            f"{r['K_or_alpha']:>10s}  {r['n_states']:>7d}  "
            f"{r['mean_acc']:>8.4f}  {terr_str:>13s}  "
            f"{perfect_str:>7s}")

    log_path = os.path.join(HERE, 'benchmark_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Wrote {log_path}", flush=True)


if __name__ == "__main__":
    main()
