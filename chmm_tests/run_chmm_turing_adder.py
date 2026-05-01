"""CHMM analogue of the binary-alphabet Turing-adder forecasting test.

Mirrors `test_turing_binary_alphabet_adder_forecasting.py` (Test 2 —
Reduced 3-column).  Tokenises each tape row (read, write, dir) into
a single discrete symbol; trains CHMM at K in {2, 4, 8} clones per
emission; evaluates per-step prediction conditional on the observed
next read symbol; reports per-position accuracy on (read, write, dir).

Numbers compare directly to GDC's reduced 3-column run on the same
splits:

    GDC (N_TRAIN=400, alpha=0.99): mean 0.999, write err 0.13%, 1/10 perfect

We use the same NUM_RANGE_TRAIN = (0, 32), NUM_RANGE_TEST = (0, 1000)
and TRAIN_SEED / TEST_SEED, but optionally a smaller N_TRAIN to keep
CHMM EM tractable.  N_TRAIN can be set via the env var CHMM_N_TRAIN
(default 200).

Run:
    python chmm_tests/run_chmm_turing_adder.py

Outputs:
    chmm_turing_adder_results.csv
    chmm_turing_adder_log.txt   (stdout dump)
"""

from __future__ import annotations

import os
import sys
import csv
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, "naturecomm_cscg"))

from binary_alphabet_adder import simulate_random_binary_alphabet_adders  # noqa: E402
from chmm_actions import CHMM, forward  # noqa: E402

N_TRAIN = int(os.environ.get("CHMM_N_TRAIN", 200))
N_TEST = 10
NUM_RANGE_TRAIN = (0, 32)
NUM_RANGE_TEST = (0, 1000)
MAX_STEPS = 200_000
TRAIN_SEED = 42
TEST_SEED = 123
K_GRID = [2, 4, 8]
N_EM_ITERS = 50


def reduced_token_alphabet(train_tapes):
    """Tokenise (read, write, dir) tuples seen in training to a small
    int alphabet.  Returns:
        tuple_to_id : dict (read, write, dir) -> id
        id_to_tuple : list of (read, write, dir) per id
    """
    seen = set()
    for tape in train_tapes:
        for row in tape:
            if int(row[0]) == -1:
                continue
            seen.add((int(row[1]), int(row[2]), int(row[3])))
    id_to_tuple = sorted(seen)
    tuple_to_id = {t: i for i, t in enumerate(id_to_tuple)}
    return tuple_to_id, id_to_tuple


def encode_tape_reduced(tape, tuple_to_id):
    """Reduced 3-col tokens for one tape (drop halt-marker rows; drop
    rows whose tuple is not in the training alphabet)."""
    out = []
    skipped = 0
    for row in tape:
        if int(row[0]) == -1:
            continue
        key = (int(row[1]), int(row[2]), int(row[3]))
        tid = tuple_to_id.get(key)
        if tid is None:
            skipped += 1
            continue
        out.append(tid)
    return np.asarray(out, dtype=np.int64), skipped


def predict_next_distribution(model, alpha_t, x_t, n_clones, state_loc):
    """Given the smoothed alpha over clones-of-x_t, propagate one step
    via T (single dummy action) to get a distribution over all clones,
    then marginalise to a distribution over emission symbols.

    Returns p_next of shape (nA,).
    """
    gs, ge = int(state_loc[x_t]), int(state_loc[x_t + 1])
    # T[0, i, j] = P(j | i) under our single-dummy-action setup
    T = model.T[0]                  # (nS_total, nS_total)
    full = alpha_t @ T[gs:ge, :]    # (nS_total,)
    full = np.maximum(full, 0)
    # Marginalise over clones for each emission
    p_next = np.zeros(len(n_clones), dtype=np.float64)
    for e in range(len(n_clones)):
        s, t = int(state_loc[e]), int(state_loc[e + 1])
        p_next[e] = full[s:t].sum()
    z = p_next.sum()
    if z > 0:
        p_next /= z
    return p_next, full


def evaluate_chmm_reduced(model, test_tapes, tuple_to_id, id_to_tuple,
                          n_clones):
    """Per-position accuracy on (read, write, dir) restricted to
    candidate next-tokens whose 'read' matches the actual next read."""
    n_states = int(np.sum(n_clones))
    state_loc = np.hstack(([0], n_clones)).cumsum().astype(np.int64)
    nA = len(n_clones)

    by_read = {0: [], 1: [], 2: []}
    for tid, tup in enumerate(id_to_tuple):
        by_read[tup[0]].append(tid)

    correct = np.zeros(3, dtype=np.int64)
    total = np.zeros(3, dtype=np.int64)
    perfect_per_tape = []
    errors_per_tape = []
    total_tuple_errors = 0
    skipped_total = 0
    err_records = []  # (tape_idx, step, actual_tuple, predicted_tuple)

    for tape_idx, tape in enumerate(test_tapes):
        x, skipped = encode_tape_reduced(tape, tuple_to_id)
        skipped_total += skipped
        if len(x) < 2:
            perfect_per_tape.append(True)
            continue
        a = np.zeros_like(x)
        log2_lik, mess_fwd = forward(
            model.T.transpose(0, 2, 1), model.Pi_x, model.n_clones,
            x, a, store_messages=True,
        )
        mess_loc = np.hstack(([0], n_clones[x])).cumsum().astype(np.int64)

        tape_errors = 0
        for t in range(len(x) - 1):
            ms, me = int(mess_loc[t]), int(mess_loc[t + 1])
            alpha_t = mess_fwd[ms:me].astype(np.float64)
            p_next, _ = predict_next_distribution(
                model, alpha_t, int(x[t]), n_clones, state_loc)
            actual_tup = id_to_tuple[int(x[t + 1])]
            actual_read = actual_tup[0]
            cands = by_read[actual_read]
            if not cands:
                continue
            best_tid = max(cands, key=lambda c: p_next[c])
            pred_tup = id_to_tuple[best_tid]
            for pos in range(3):
                total[pos] += 1
                if pred_tup[pos] == actual_tup[pos]:
                    correct[pos] += 1
            if pred_tup != actual_tup:
                tape_errors += 1
                total_tuple_errors += 1
                if len(err_records) < 30:
                    err_records.append((tape_idx, t + 1, actual_tup, pred_tup))
        perfect_per_tape.append(tape_errors == 0)
        errors_per_tape.append(tape_errors)

    acc = correct / np.maximum(total, 1)
    return (acc, total, perfect_per_tape, errors_per_tape,
            total_tuple_errors, skipped_total, err_records)


def main():
    print(f"=== CHMM Turing-adder forecasting (Test 2 — Reduced 3-col) ===",
          flush=True)
    print(f"N_TRAIN={N_TRAIN}, N_TEST={N_TEST}, K_GRID={K_GRID}", flush=True)

    print(f"\n[1/3] Generating {N_TRAIN} training tapes "
          f"(num_range={NUM_RANGE_TRAIN}, seed={TRAIN_SEED})...", flush=True)
    tr = simulate_random_binary_alphabet_adders(
        n_runs=N_TRAIN, num_range=NUM_RANGE_TRAIN,
        max_steps=MAX_STEPS, seed=TRAIN_SEED)
    train_tapes = tr['runs']
    train_halted = tr['halted_flags']
    print(f"  Done. Halted: {sum(train_halted)}/{N_TRAIN}", flush=True)

    print(f"\n[2/3] Generating {N_TEST} test tapes "
          f"(num_range={NUM_RANGE_TEST}, seed={TEST_SEED})...", flush=True)
    te = simulate_random_binary_alphabet_adders(
        n_runs=N_TEST, num_range=NUM_RANGE_TEST,
        max_steps=MAX_STEPS, seed=TEST_SEED)
    test_tapes = te['runs']
    test_halted = te['halted_flags']
    print(f"  Done. Halted: {sum(test_halted)}/{N_TEST}", flush=True)

    tuple_to_id, id_to_tuple = reduced_token_alphabet(train_tapes)
    nA = len(id_to_tuple)
    print(f"\nReduced alphabet size: {nA}", flush=True)
    for tid, tup in enumerate(id_to_tuple):
        print(f"   {tid}: {tup}", flush=True)

    train_x = np.concatenate(
        [encode_tape_reduced(t, tuple_to_id)[0] for t in train_tapes]
    ).astype(np.int64)
    train_a = np.zeros_like(train_x)
    print(f"\nTotal training tokens: {len(train_x)}", flush=True)

    train_lens = [t.shape[0] for t in train_tapes]
    test_lens = [t.shape[0] for t in test_tapes]
    print(f"Train tape lens: min={min(train_lens)}, max={max(train_lens)}, "
          f"mean={np.mean(train_lens):.1f}", flush=True)
    print(f"Test  tape lens: min={min(test_lens)}, max={max(test_lens)}, "
          f"mean={np.mean(test_lens):.1f}", flush=True)

    rows = []
    for K in K_GRID:
        n_clones = np.full(nA, K, dtype=np.int64)
        n_states = int(K * nA)
        print(f"\n--- CHMM K={K} (n_states={n_states}) ---", flush=True)
        t0 = time.time()
        model = CHMM(n_clones=n_clones, x=train_x, a=train_a,
                     pseudocount=1e-3, seed=0)
        conv = model.learn_em_T(train_x, train_a, n_iter=N_EM_ITERS,
                                term_early=True)
        train_time = time.time() - t0
        print(f"  EM done in {train_time:.1f}s ({len(conv)} iters), "
              f"final train_bps={float(np.asarray(conv[-1]).mean()):.3f}",
              flush=True)

        t0 = time.time()
        (acc, total, perfect, errors_per_tape, total_tuple_errors,
         skipped, err_recs) = evaluate_chmm_reduced(
            model, test_tapes, tuple_to_id, id_to_tuple, n_clones)
        eval_time = time.time() - t0
        n_perfect = sum(perfect)
        n_predictions = int(total[0])
        print(f"  eval done in {eval_time:.1f}s, "
              f"skipped (unseen tuples): {skipped}", flush=True)
        print(f"  per-position accuracy: "
              f"read={acc[0]:.4f}  write={acc[1]:.4f}  dir={acc[2]:.4f}",
              flush=True)
        print(f"  mean accuracy: {acc.mean():.4f}", flush=True)
        print(f"  total tuple errors: {total_tuple_errors} / "
              f"{n_predictions} ({100*total_tuple_errors/max(n_predictions,1):.3f}%)",
              flush=True)
        print(f"  errors per tape: {errors_per_tape}", flush=True)
        print(f"  perfect tapes: {n_perfect}/{len(test_tapes)}", flush=True)
        if err_recs:
            print(f"  first error sample (actual -> predicted):", flush=True)
            for rec in err_recs[:6]:
                print(f"    tape{rec[0]} step{rec[1]}: "
                      f"{rec[2]} -> {rec[3]}", flush=True)
        rows.append({
            'K': K, 'n_states': n_states,
            'train_time_s': train_time, 'eval_time_s': eval_time,
            'em_iters': len(conv),
            'final_train_bps': float(np.asarray(conv[-1]).mean()),
            'acc_read': acc[0], 'acc_write': acc[1], 'acc_dir': acc[2],
            'mean_acc': acc.mean(), 'perfect_tapes': n_perfect,
            'total_tuple_errors': total_tuple_errors,
            'n_predictions': n_predictions,
            'skipped_unseen': skipped,
        })

    out_csv = os.path.join(HERE, 'chmm_turing_adder_results.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out_csv}", flush=True)

    print("\n=== summary (CHMM Turing adder) ===", flush=True)
    print(f"{'K':>3}  {'states':>6}  {'read':>6}  {'write':>6}  "
          f"{'dir':>6}  {'mean':>6}  {'perfect':>7}  {'train_s':>7}",
          flush=True)
    for r in rows:
        print(f"{r['K']:>3d}  {r['n_states']:>6d}  "
              f"{r['acc_read']:>6.3f}  {r['acc_write']:>6.3f}  "
              f"{r['acc_dir']:>6.3f}  {r['mean_acc']:>6.3f}  "
              f"{r['perfect_tapes']:>3d}/{N_TEST:<3d}  "
              f"{r['train_time_s']:>7.1f}", flush=True)


if __name__ == "__main__":
    main()
