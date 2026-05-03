"""Extend the CHMM/ALERGIA tuning to:
  - much smaller ALERGIA eps values (0.00001 to 0.001 plus 'auto')
  - binary_adder (both variants)
  - dyck1 sequence task

Also re-tests Reverse with the tighter eps grid since the previous best
ALERGIA was at the smallest tested eps (0.005) and clearly hadn't found
the optimum.
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'chmm_tests', 'naturecomm_cscg'))

# Reuse helpers
from tune_chmm_alergia import (  # noqa: E402
    reduced_alphabet, encode_reduced,
    chmm_eval_tm_reduced, alergia_eval_tm_reduced)

CHMM_KS = [4, 8, 16, 32, 64]
CHMM_SEEDS = [0, 1, 2]
CHMM_EM_ITERS = 30
ALERGIA_EPS_VALUES = [0.00001, 0.0001, 0.001, 0.005, 0.05, 'auto']

OUT_CSV = os.path.join(HERE, 'chmm_alergia_tuning_extra_results.csv')


def run_reverse_extra(args):
    """Re-tune ALERGIA on reverse with much tighter eps values."""
    task_name, mod_name, variant, train_range, test_range, n_train, n_test, max_steps = args
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, 'chmm_tests', 'naturecomm_cscg'))
    import importlib
    mod = importlib.import_module(mod_name)
    nr = (variant == 'noread')
    tr = mod.simulate(n_train, train_range, max_steps=max_steps, seed=42, noread=nr)
    te = mod.simulate(n_test, test_range, max_steps=max_steps*4, seed=123, noread=nr)
    if not tr['runs']: return []
    tuple_to_id, id_to_tuple = reduced_alphabet(tr['runs'])
    nA = len(id_to_tuple)
    if nA == 0: return []
    rows = []
    from aalpy.learning_algs import run_Alergia
    START = nA
    data = []
    for arr in tr['runs']:
        seq = [START] + [int(t) for t in encode_reduced(arr, tuple_to_id)]
        data.append(seq)
    class W: pass
    for eps in ALERGIA_EPS_VALUES:
        try:
            t0 = time.time()
            mc = run_Alergia(data, automaton_type='mc', eps=eps, print_info=False)
            wrap = W(); wrap.mc = mc; wrap.start_token = START
            acc, total, terr, perf = alergia_eval_tm_reduced(
                wrap, te['runs'], tuple_to_id, id_to_tuple, nA)
            rows.append(dict(task=task_name, variant=variant,
                             model=f'alergia-eps{eps}', K=None, seed=None,
                             mean_acc=float(acc.mean()),
                             errors=int(terr), n_predictions=int(total[0]),
                             perfect=int(perf), n_test=n_test,
                             fit_s=time.time()-t0))
        except Exception as e:
            sys.stderr.write(f"alergia fail {task_name}/{variant} eps={eps}: {e}\n")
    return rows


def run_binary_adder(args):
    variant, n_train, n_test = args
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, 'chmm_tests', 'naturecomm_cscg'))
    from binary_alphabet_adder import (
        simulate_random_binary_alphabet_adders, BINARY_ALPHABET_ADDER)
    from _tm_common import apply_noread_to_runs
    tr = simulate_random_binary_alphabet_adders(n_runs=n_train,
        num_range=(0, 32), max_steps=200_000, seed=42)
    te = simulate_random_binary_alphabet_adders(n_runs=n_test,
        num_range=(0, 1000), max_steps=200_000, seed=123)
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
    tuple_to_id, id_to_tuple = reduced_alphabet(tr['runs'])
    nA = len(id_to_tuple)
    train_x = np.concatenate(
        [encode_reduced(t, tuple_to_id) for t in tr['runs']]).astype(np.int64)
    train_a = np.zeros_like(train_x)
    rows = []
    from chmm_actions import CHMM
    for K in CHMM_KS:
        if K * nA > 200:
            continue  # skip very large K to keep runtime reasonable
        n_clones_arr = np.full(nA, K, dtype=np.int64)
        for seed in CHMM_SEEDS:
            try:
                t0 = time.time()
                model = CHMM(n_clones=n_clones_arr, x=train_x, a=train_a,
                             pseudocount=1e-3, seed=seed)
                model.learn_em_T(train_x, train_a, n_iter=CHMM_EM_ITERS,
                                 term_early=True)
                acc, total, terr, perf = chmm_eval_tm_reduced(
                    model, te['runs'], tuple_to_id, id_to_tuple, n_clones_arr)
                rows.append(dict(task='binary_adder', variant=variant,
                                 model=f'chmm-K{K}-s{seed}', K=K, seed=seed,
                                 mean_acc=float(acc.mean()),
                                 errors=int(terr), n_predictions=int(total[0]),
                                 perfect=int(perf), n_test=n_test,
                                 fit_s=time.time()-t0))
            except Exception as e:
                sys.stderr.write(f"chmm fail binary_adder/{variant} K={K} s={seed}: {e}\n")
    from aalpy.learning_algs import run_Alergia
    START = nA
    data = []
    for arr in tr['runs']:
        seq = [START] + [int(t) for t in encode_reduced(arr, tuple_to_id)]
        data.append(seq)
    class W: pass
    for eps in ALERGIA_EPS_VALUES:
        try:
            t0 = time.time()
            mc = run_Alergia(data, automaton_type='mc', eps=eps, print_info=False)
            wrap = W(); wrap.mc = mc; wrap.start_token = START
            acc, total, terr, perf = alergia_eval_tm_reduced(
                wrap, te['runs'], tuple_to_id, id_to_tuple, nA)
            rows.append(dict(task='binary_adder', variant=variant,
                             model=f'alergia-eps{eps}', K=None, seed=None,
                             mean_acc=float(acc.mean()),
                             errors=int(terr), n_predictions=int(total[0]),
                             perfect=int(perf), n_test=n_test,
                             fit_s=time.time()-t0))
        except Exception as e:
            sys.stderr.write(f"alergia fail binary_adder/{variant} eps={eps}: {e}\n")
    return rows


def run_dyck1(args):
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, 'chmm_tests', 'naturecomm_cscg'))
    import dyck1
    tr = dyck1.simulate(1000, max_depth=4, length_min=4, length_max=200, seed=42)
    te = dyck1.simulate(200, max_depth=8, length_min=4, length_max=400, seed=123)
    nA = dyck1.ALPHABET_SIZE
    rows = []
    train_x = np.concatenate(tr['sequences']).astype(np.int64)
    train_a = np.zeros_like(train_x)
    from chmm_actions import CHMM
    for K in CHMM_KS:
        if K * nA > 200: continue
        n_clones_arr = np.full(nA, K, dtype=np.int64)
        for seed in CHMM_SEEDS:
            try:
                t0 = time.time()
                model = CHMM(n_clones=n_clones_arr, x=train_x, a=train_a,
                             pseudocount=1e-3, seed=seed)
                model.learn_em_T(train_x, train_a, n_iter=CHMM_EM_ITERS,
                                 term_early=True)
                # Eval: next-symbol accuracy on test sequences (exclude END)
                from chmm_actions import forward
                state_loc = np.hstack(([0], n_clones_arr)).cumsum().astype(np.int64)
                T = model.T[0]
                correct, total = 0, 0
                for seq in te['sequences']:
                    x = seq.astype(np.int64)
                    if len(x) < 2: continue
                    a = np.zeros_like(x)
                    log2_lik, mess_fwd = forward(model.T.transpose(0, 2, 1),
                                                  model.Pi_x, model.n_clones,
                                                  x, a, store_messages=True)
                    mess_loc = np.hstack(([0], model.n_clones[x])).cumsum().astype(np.int64)
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
                        if p_next.sum() == 0: continue
                        # Skip END-token positions (last one)
                        actual = int(x[t + 1])
                        if actual == nA - 1:  # assume END is last
                            continue
                        pred = int(np.argmax(p_next[:nA - 1]))
                        total += 1
                        if pred == actual:
                            correct += 1
                acc = correct / max(total, 1)
                rows.append(dict(task='dyck1', variant='n/a',
                                 model=f'chmm-K{K}-s{seed}', K=K, seed=seed,
                                 mean_acc=float(acc),
                                 errors=int(total - correct),
                                 n_predictions=int(total),
                                 perfect=-1, n_test=200,
                                 fit_s=time.time()-t0))
            except Exception as e:
                sys.stderr.write(f"chmm fail dyck1 K={K} s={seed}: {e}\n")
    return rows


def main():
    n_workers = max(1, min(12, (os.cpu_count() or 4) - 1))
    # Reverse re-run with extra eps; binary adder both variants; dyck1
    reverse_tasks = [
        ('reverse', 'reverse_tm', 'original', (3, 6),  (10, 16), 300, 20, 10000),
        ('reverse', 'reverse_tm', 'noread',   (3, 6),  (10, 16), 300, 20, 10000),
    ]
    binary_tasks = [('original', 200, 10), ('noread', 200, 10)]

    all_rows = []
    t0 = time.time()
    print(f"Running {len(reverse_tasks)} reverse-eps + {len(binary_tasks)} adder + 1 dyck1",
          flush=True)
    with mp.Pool(processes=n_workers) as pool:
        for cell_rows in pool.imap_unordered(run_reverse_extra, reverse_tasks):
            all_rows.extend(cell_rows)
            print(f"  reverse done [{time.time()-t0:.0f}s, {len(all_rows)} rows]",
                  flush=True)
        for cell_rows in pool.imap_unordered(run_binary_adder, binary_tasks):
            all_rows.extend(cell_rows)
            print(f"  binary_adder done [{time.time()-t0:.0f}s, {len(all_rows)} rows]",
                  flush=True)
        for cell_rows in pool.imap_unordered(run_dyck1, [None]):
            all_rows.extend(cell_rows)
            print(f"  dyck1 done [{time.time()-t0:.0f}s, {len(all_rows)} rows]",
                  flush=True)
    fields = ['task', 'variant', 'model', 'K', 'seed', 'mean_acc', 'errors',
              'n_predictions', 'perfect', 'n_test', 'fit_s']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
