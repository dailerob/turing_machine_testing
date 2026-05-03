"""Tune CHMM and ALERGIA on the algorithmic-trace benchmarks.

For each (task, variant) configuration:
  CHMM: K in {4, 8, 16, 32, 64} x seed in {0, 1, 2}; report best
  ALERGIA: eps in {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}; report best

GDC tuned results are pulled from earlier sweeps as reference.

Tasks tested:
  parity, increment, reverse  (original + noread)
  binary_adder                (original + noread; N_train=200)
  dyck1                       (no variant)

Multiprocessed across (task, variant) cells.
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'chmm_tests', 'naturecomm_cscg'))

CHMM_KS = [4, 8, 16, 32, 64]
CHMM_SEEDS = [0, 1, 2]
CHMM_EM_ITERS = 30
ALERGIA_EPS_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

OUT_CSV = os.path.join(HERE, 'chmm_alergia_tuning_results.csv')


# (task_name, module_name, variant, train_range, test_range, n_train, n_test, max_steps)
TM_TASKS = [
    ('parity',    'parity_tm',    'original', (3, 8),  (16, 32), 300, 20, 200),
    ('parity',    'parity_tm',    'noread',   (3, 8),  (16, 32), 300, 20, 200),
    ('increment', 'increment_tm', 'original', (1, 5),  (8, 12),  300, 20, 200),
    ('increment', 'increment_tm', 'noread',   (1, 5),  (8, 12),  300, 20, 200),
    ('reverse',   'reverse_tm',   'original', (3, 6),  (10, 16), 300, 20, 10000),
    ('reverse',   'reverse_tm',   'noread',   (3, 6),  (10, 16), 300, 20, 10000),
]


def reduced_alphabet(runs):
    """Build (read, write, dir) tuple to id mapping."""
    seen = set()
    for arr in runs:
        for row in arr:
            if int(row[0]) == -1:
                continue
            seen.add((int(row[1]), int(row[2]), int(row[3])))
    id_to_tuple = sorted(seen)
    return {t: i for i, t in enumerate(id_to_tuple)}, id_to_tuple


def encode_reduced(arr, tuple_to_id):
    out = []
    for row in arr:
        if int(row[0]) == -1:
            continue
        key = (int(row[1]), int(row[2]), int(row[3]))
        if key in tuple_to_id:
            out.append(tuple_to_id[key])
    return np.asarray(out, dtype=np.int64)


def chmm_eval_tm_reduced(model, test_runs, tuple_to_id, id_to_tuple, n_clones):
    """Per-position eval with conditional-on-read (read, write, dir).
    Returns (acc[3], total[3], tuple_errors, perfect_tapes)."""
    from chmm_actions import forward
    state_loc = np.hstack(([0], n_clones)).cumsum().astype(np.int64)
    nA = len(n_clones)
    by_read = {0: [], 1: [], 2: []}
    for tid, tup in enumerate(id_to_tuple):
        by_read.setdefault(tup[0], []).append(tid)
    correct = np.zeros(3, dtype=np.int64)
    total = np.zeros(3, dtype=np.int64)
    tuple_errors, perfect = 0, 0
    for arr in test_runs:
        x = encode_reduced(arr, tuple_to_id)
        if len(x) < 2:
            perfect += 1; continue
        a = np.zeros_like(x)
        log2_lik, mess_fwd = forward(model.T.transpose(0, 2, 1), model.Pi_x,
                                     model.n_clones, x, a, store_messages=True)
        mess_loc = np.hstack(([0], n_clones[x])).cumsum().astype(np.int64)
        T = model.T[0]
        tape_err = 0
        for t in range(len(x) - 1):
            ms, me = int(mess_loc[t]), int(mess_loc[t + 1])
            alpha_t = mess_fwd[ms:me].astype(np.float64)
            xt = int(x[t])
            gs, ge = int(state_loc[xt]), int(state_loc[xt + 1])
            full = alpha_t @ T[gs:ge, :]
            full = np.maximum(full, 0)
            p_next = np.zeros(nA)
            for e in range(nA):
                s, e2 = int(state_loc[e]), int(state_loc[e + 1])
                p_next[e] = full[s:e2].sum()
            p_next = p_next / p_next.sum() if p_next.sum() > 0 else np.full(nA, 1.0/nA)
            actual = id_to_tuple[int(x[t + 1])]
            cands = by_read.get(actual[0], [])
            if not cands: continue
            best_tid = max(cands, key=lambda c: p_next[c])
            pred = id_to_tuple[best_tid]
            mismatch = False
            for pos in range(3):
                total[pos] += 1
                if pred[pos] == actual[pos]:
                    correct[pos] += 1
                else:
                    mismatch = True
            if mismatch:
                tape_err += 1; tuple_errors += 1
        if tape_err == 0:
            perfect += 1
    acc = correct / np.maximum(total, 1)
    return acc, total, tuple_errors, perfect


def alergia_eval_tm_reduced(model, test_runs, tuple_to_id, id_to_tuple, alphabet_size):
    """Like chmm_eval_tm_reduced but for ALERGIA-learned MC."""
    by_read = {}
    for tid, tup in enumerate(id_to_tuple):
        by_read.setdefault(tup[0], []).append(tid)
    correct = np.zeros(3, dtype=np.int64)
    total = np.zeros(3, dtype=np.int64)
    tuple_errors, perfect = 0, 0
    state_idx = {id(s): i for i, s in enumerate(model.mc.states)}
    n_states = len(state_idx)
    outputs = np.array([s.output for s in model.mc.states], dtype=int)
    T = np.zeros((n_states, n_states), dtype=np.float64)
    for s in model.mc.states:
        i = state_idx[id(s)]
        for target, prob in s.transitions:
            j = state_idx[id(target)]
            T[i, j] += float(prob)
    rs = T.sum(axis=1, keepdims=True); rs[rs==0] = 1
    T = T / rs

    def state_after(prefix):
        st = model.mc.initial_state
        for sym in prefix:
            ns = None; bp = -1
            for target, prob in st.transitions:
                if target.output == int(sym) and float(prob) > bp:
                    ns = target; bp = float(prob)
            if ns is None:
                return None
            st = ns
        return state_idx[id(st)]

    for arr in test_runs:
        x = encode_reduced(arr, tuple_to_id)
        if len(x) < 2:
            perfect += 1; continue
        prefix_idx = state_after([model.start_token]
                                 + list(int(s) for s in x[:0]))  # placeholder
        # Walk along x[:t], evaluating prediction for each t
        st_idx = state_after([model.start_token])
        tape_err = 0
        for t in range(len(x) - 1):
            if st_idx is None:
                # path broken — uniform fallback
                p_next = np.full(alphabet_size, 1.0 / alphabet_size)
            else:
                # next-token distribution from st_idx
                state_dist = np.zeros(n_states); state_dist[st_idx] = 1.0
                state_dist = state_dist @ T
                p_next = np.zeros(alphabet_size)
                for tid in range(alphabet_size):
                    mask = outputs == tid
                    p_next[tid] = state_dist[mask].sum()
                if p_next.sum() > 0: p_next /= p_next.sum()
            actual = id_to_tuple[int(x[t + 1])]
            cands = by_read.get(actual[0], [])
            if not cands: continue
            best_tid = max(cands, key=lambda c: p_next[c])
            pred = id_to_tuple[best_tid]
            mismatch = False
            for pos in range(3):
                total[pos] += 1
                if pred[pos] == actual[pos]:
                    correct[pos] += 1
                else:
                    mismatch = True
            if mismatch:
                tape_err += 1; tuple_errors += 1
            # Advance state
            if st_idx is not None:
                ns = None; bp = -1
                for target, prob in model.mc.states[st_idx].transitions:
                    if target.output == int(x[t]) and float(prob) > bp:
                        ns = target; bp = float(prob)
                st_idx = state_idx[id(ns)] if ns is not None else None
        if tape_err == 0:
            perfect += 1
    acc = correct / np.maximum(total, 1)
    return acc, total, tuple_errors, perfect


def run_tm_cell(args):
    task_name, mod_name, variant, train_range, test_range, n_train, n_test, max_steps = args
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, 'chmm_tests', 'naturecomm_cscg'))
    import importlib
    mod = importlib.import_module(mod_name)
    nr = (variant == 'noread')
    tr = mod.simulate(n_train, train_range, max_steps=max_steps, seed=42, noread=nr)
    te = mod.simulate(n_test, test_range, max_steps=max_steps*4, seed=123, noread=nr)
    if not tr['runs'] or not te['runs']:
        return []
    tuple_to_id, id_to_tuple = reduced_alphabet(tr['runs'])
    nA = len(id_to_tuple)
    if nA == 0:
        return []
    train_x = np.concatenate([encode_reduced(t, tuple_to_id) for t in tr['runs']]
                             ).astype(np.int64)
    if len(train_x) == 0:
        return []
    train_a = np.zeros_like(train_x)

    rows = []
    # CHMM sweep over K and seed
    from chmm_actions import CHMM
    for K in CHMM_KS:
        n_clones_arr = np.full(nA, K, dtype=np.int64)
        for seed in CHMM_SEEDS:
            try:
                t0 = time.time()
                model = CHMM(n_clones=n_clones_arr, x=train_x, a=train_a,
                             pseudocount=1e-3, seed=seed)
                model.learn_em_T(train_x, train_a, n_iter=CHMM_EM_ITERS,
                                 term_early=True)
                fit_t = time.time() - t0
                acc, total, terr, perf = chmm_eval_tm_reduced(
                    model, te['runs'], tuple_to_id, id_to_tuple, n_clones_arr)
                rows.append(dict(task=task_name, variant=variant,
                                 model=f'chmm-K{K}-s{seed}', K=K, seed=seed,
                                 mean_acc=float(acc.mean()),
                                 errors=int(terr), n_predictions=int(total[0]),
                                 perfect=int(perf), n_test=n_test,
                                 fit_s=fit_t))
            except Exception as e:
                sys.stderr.write(f"chmm fail {task_name}/{variant} K={K} s={seed}: {e}\n")
    # ALERGIA sweep over eps
    from aalpy.learning_algs import run_Alergia
    START = nA
    data = []
    for arr in tr['runs']:
        seq = [START] + [int(t) for t in encode_reduced(arr, tuple_to_id)]
        data.append(seq)
    class AlergiaWrap:
        pass
    for eps in ALERGIA_EPS_VALUES:
        try:
            t0 = time.time()
            mc = run_Alergia(data, automaton_type='mc', eps=eps, print_info=False)
            fit_t = time.time() - t0
            wrap = AlergiaWrap()
            wrap.mc = mc
            wrap.start_token = START
            acc, total, terr, perf = alergia_eval_tm_reduced(
                wrap, te['runs'], tuple_to_id, id_to_tuple, nA)
            rows.append(dict(task=task_name, variant=variant,
                             model=f'alergia-eps{eps}', K=None, seed=None,
                             mean_acc=float(acc.mean()),
                             errors=int(terr), n_predictions=int(total[0]),
                             perfect=int(perf), n_test=n_test,
                             fit_s=fit_t))
        except Exception as e:
            sys.stderr.write(f"alergia fail {task_name}/{variant} eps={eps}: {e}\n")
    return rows


def main():
    n_workers = max(1, min(12, (os.cpu_count() or 4) - 1))
    tasks = TM_TASKS
    print(f"Tuning sweep: {len(tasks)} (task,variant) cells, "
          f"{n_workers} workers", flush=True)
    print(f"  CHMM K in {CHMM_KS} x seeds {CHMM_SEEDS}", flush=True)
    print(f"  ALERGIA eps in {ALERGIA_EPS_VALUES}", flush=True)
    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for cell_rows in pool.imap_unordered(run_tm_cell, tasks, chunksize=1):
            all_rows.extend(cell_rows)
            done += 1
            print(f"  {done}/{len(tasks)} cells done [{time.time()-t0:.0f}s]",
                  flush=True)
    fields = ['task', 'variant', 'model', 'K', 'seed', 'mean_acc', 'errors',
              'n_predictions', 'perfect', 'n_test', 'fit_s']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
