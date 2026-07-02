"""Kneser-Ney 3-gram on the algorithmic / Turing-machine benchmarks.

Mirrors parrot_eval.py / hpylm_eval.py / ppm_eval.py exactly so KN-3
slots into the same TM tuple-error comparison.

KN-3gram: standard interpolated Kneser-Ney with discount D.
For TM tasks: predict the next tuple-id given the prefix of past tuple-ids,
restricting candidates to those whose tuple[0] (the read symbol) matches
the actual next read.

Variant grid (val-tuned per task):
  - discount D ∈ {0.5, 0.75, 0.9}

Output: algorithmic_benchmarks/kn3_benchmark_results.csv
"""
from __future__ import annotations
import os, sys, csv, time
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

import dyck1                                          # noqa: E402
from _tm_task_config import (                          # noqa: E402
    TM_TASKS, TASK_ORDER, simulate_train_val_test, SUFFIX)

DISCOUNTS = [0.5, 0.75, 0.9]


# --------------------------------------------------------------------
# KN-3gram model (reduced-alphabet predictor)
# --------------------------------------------------------------------
class KN3Model:
    """Interpolated KN-3gram over a finite int alphabet [0, V)."""

    def __init__(self, V: int, discount: float = 0.75):
        self.V = int(V)
        self.D = float(discount)
        self.START = self.V    # sentinel; not in candidate set
        # Counts
        self.c3 = defaultdict(lambda: defaultdict(int))
        self.c2 = defaultdict(lambda: defaultdict(int))
        self.cont1 = defaultdict(set)        # distinct h1 preceding w
        self.cont2 = defaultdict(set)        # distinct w following (h2,h1)
        self.c3_sum = {}
        self.c2_sum = {}
        self.p_cont1 = None

    def fit(self, sequences):
        for s in sequences:
            tokens = [self.START, self.START] + list(int(t) for t in s)
            for t in range(2, len(tokens)):
                w = tokens[t]; h1 = tokens[t - 1]; h2 = tokens[t - 2]
                self.c3[(h2, h1)][w] += 1
                self.c2[h1][w] += 1
                self.cont1[w].add(h1)
                self.cont2[(h2, h1)].add(w)
        self.c3_sum = {k: sum(v.values()) for k, v in self.c3.items()}
        self.c2_sum = {k: sum(v.values()) for k, v in self.c2.items()}
        # Continuation-base unigram
        n_bigram_types = sum(len(s) for s in self.cont1.values())
        if n_bigram_types == 0:
            n_bigram_types = 1
        self.p_cont1 = np.full(self.V + 1, 1e-12)
        for w in range(self.V):
            self.p_cont1[w] = len(self.cont1.get(w, set())) / n_bigram_types
        s = self.p_cont1[:self.V].sum()
        if s > 0:
            self.p_cont1[:self.V] /= s

    def _kn1(self, h1, w):
        D = self.D
        cnt2 = self.c2.get(h1)
        if cnt2 is None or self.c2_sum.get(h1, 0) == 0:
            return float(self.p_cont1[w])
        total = self.c2_sum[h1]
        c = cnt2.get(w, 0)
        n_unique = len(cnt2)
        return max(c - D, 0) / total \
               + (D * n_unique / total) * float(self.p_cont1[w])

    def _kn2(self, h2, h1, w):
        D = self.D
        cnt3 = self.c3.get((h2, h1))
        if cnt3 is None or self.c3_sum.get((h2, h1), 0) == 0:
            return self._kn1(h1, w)
        total = self.c3_sum[(h2, h1)]
        c = cnt3.get(w, 0)
        n_unique = len(cnt3)
        return max(c - D, 0) / total \
               + (D * n_unique / total) * self._kn1(h1, w)

    def predict_distribution(self, prefix, candidates=None):
        """Return P(w | prefix) for w in candidates (or all if None)."""
        if len(prefix) == 0:
            h2 = self.START; h1 = self.START
        elif len(prefix) == 1:
            h2 = self.START; h1 = int(prefix[-1])
        else:
            h2 = int(prefix[-2]); h1 = int(prefix[-1])
        if candidates is None:
            cands = range(self.V)
        else:
            cands = candidates
        out = np.zeros(self.V)
        for w in cands:
            out[w] = self._kn2(h2, h1, int(w))
        s = out.sum()
        if s > 0:
            out = out / s
        else:
            out = np.full(self.V, 1.0 / self.V)
        return out

    def predict_argmax(self, prefix, candidates):
        """Argmax over candidates only."""
        if len(prefix) == 0:
            h2 = self.START; h1 = self.START
        elif len(prefix) == 1:
            h2 = self.START; h1 = int(prefix[-1])
        else:
            h2 = int(prefix[-2]); h1 = int(prefix[-1])
        best = -1; best_p = -1.0
        for w in candidates:
            p = self._kn2(h2, h1, int(w))
            if p > best_p:
                best_p = p; best = int(w)
        return best


# --------------------------------------------------------------------
# Reduced-alphabet TM helpers (copy of parrot_eval.py's helpers)
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


def kn3_eval_tm_reduced(model: KN3Model, test_tapes, tuple_to_id,
                       id_to_tuple):
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
        tape_err = 0
        for t in range(len(x) - 1):
            actual_tup = id_to_tuple[int(x[t + 1])]
            cands = by_read.get(actual_tup[0], [])
            if not cands:
                continue
            prefix = x[: t + 1]
            best_tid = model.predict_argmax(prefix, cands)
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


def kn3_eval_dyck(model: KN3Model, test_seqs):
    correct, total = 0, 0
    cands = [dyck1.OPEN, dyck1.CLOSE]
    for x in test_seqs:
        if len(x) < 2:
            continue
        for t in range(len(x) - 1):
            actual_next = int(x[t + 1])
            if actual_next == dyck1.END:
                continue
            prefix = x[: t + 1]
            best = model.predict_argmax(prefix, cands)
            total += 1
            if best == actual_next:
                correct += 1
    return correct / max(total, 1), total, correct


def run_tm_task(name, log, variant='original'):
    cfg = TM_TASKS[name]
    n_test = cfg['n_test']
    log(f"\n{'='*66}\nTASK: {name} (TM, KN3, variant={variant})\n{'='*66}")
    tr_runs, val_runs, te_runs = simulate_train_val_test(name, variant)
    tuple_to_id, id_to_tuple = reduced_alphabet(tr_runs)
    nA = len(id_to_tuple)
    log(f"  alphabet={nA}, train={len(tr_runs)}, val={len(val_runs)}, "
        f"test={len(te_runs)}")
    train_red_seqs = [encode_reduced(t, tuple_to_id)[0] for t in tr_runs]
    train_red_seqs = [s for s in train_red_seqs if len(s) > 0]

    val_scores = []
    for D in DISCOUNTS:
        m = KN3Model(V=nA, discount=D)
        m.fit(train_red_seqs)
        _, _, terr, _ = kn3_eval_tm_reduced(m, val_runs, tuple_to_id,
                                             id_to_tuple)
        val_scores.append((terr, D, m))
    val_scores.sort(key=lambda x: x[0])
    val_terr, best_D, best_m = val_scores[0]
    log(f"  Val pick: D={best_D}  val_tuple_errors={val_terr}")

    t0 = time.time()
    acc, total, terr, perf = kn3_eval_tm_reduced(
        best_m, te_runs, tuple_to_id, id_to_tuple)
    eval_t = time.time() - t0
    n_pred = int(total[0])
    log(f"  KN3 acc: read={acc[0]:.4f} write={acc[1]:.4f} "
        f"dir={acc[2]:.4f} mean={acc.mean():.4f}")
    log(f"  KN3 tuple errors: {terr}/{n_pred} "
        f"({100*terr/max(n_pred,1):.3f}%), perfect: {perf}/{n_test}  "
        f"(eval={eval_t:.1f}s)")
    return [dict(task=name, variant=variant, model='kn3',
                 discount=best_D, val_tuple_errors=val_terr,
                 acc_read=acc[0], acc_write=acc[1], acc_dir=acc[2],
                 mean_acc=float(acc.mean()),
                 tuple_errors=terr, n_predictions=n_pred,
                 perfect_tapes=perf, n_test=n_test, eval_s=eval_t)]


def run_dyck1(log):
    log(f"\n{'='*66}\nTASK: dyck1 (KN3)\n{'='*66}")
    tr = dyck1.simulate(1000, max_depth=4, length_min=4, length_max=200,
                        seed=42)
    val = dyck1.simulate(100, max_depth=6, length_min=4, length_max=300,
                         seed=7)
    te = dyck1.simulate(200, max_depth=8, length_min=4, length_max=400,
                        seed=123)
    tr_seqs = [s.astype(np.int64) for s in tr['sequences']]
    val_seqs = [s.astype(np.int64) for s in val['sequences']]
    te_seqs = te['sequences']
    nA = dyck1.ALPHABET_SIZE

    val_scores = []
    for D in DISCOUNTS:
        m = KN3Model(V=nA, discount=D)
        m.fit(tr_seqs)
        acc, total, correct = kn3_eval_dyck(m, val_seqs)
        val_scores.append((-correct, D, m))
    val_scores.sort(key=lambda x: x[0])
    val_neg_correct, best_D, best_m = val_scores[0]
    log(f"  Val pick: D={best_D}")

    t0 = time.time()
    acc, total, correct = kn3_eval_dyck(best_m, te_seqs)
    eval_t = time.time() - t0
    log(f"  KN3 dyck1 acc: {acc:.4f} ({correct}/{total})  "
        f"(eval={eval_t:.1f}s)")
    return [dict(task='dyck1', variant='n/a', model='kn3',
                 discount=best_D, val_tuple_errors=-val_neg_correct,
                 acc_read=float('nan'), acc_write=float('nan'),
                 acc_dir=float('nan'), mean_acc=float(acc),
                 tuple_errors=int(total - correct), n_predictions=int(total),
                 perfect_tapes=-1, n_test=len(te_seqs), eval_s=eval_t)]


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    out_csv = os.path.join(HERE, f'kn3_benchmark_results{SUFFIX}.csv')
    log_lines = []
    def log(msg=''):
        print(msg, flush=True); log_lines.append(str(msg))

    rows = []
    for variant in ('original', 'noread'):
        for name in TASK_ORDER:
            rows.extend(run_tm_task(name, log, variant=variant))
    rows.extend(run_dyck1(log))

    fields = ['task', 'variant', 'model', 'discount', 'val_tuple_errors',
              'acc_read', 'acc_write', 'acc_dir', 'mean_acc',
              'tuple_errors', 'n_predictions', 'perfect_tapes',
              'n_test', 'eval_s']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    log(f"\nWrote {out_csv}")
    log_path = out_csv.replace('.csv', '.log')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))


if __name__ == "__main__":
    main()
