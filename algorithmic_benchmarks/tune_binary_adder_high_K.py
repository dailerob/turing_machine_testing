"""Push CHMM K higher on binary_adder original to confirm the unimodal
ladder.  Tests K in {16, 24, 32, 48} x 3 seeds.

Earlier sweep capped at K * nA > 200 (so K <= 16 for nA=10).  Removing
the cap to test K up to 48 (480 hidden states).
"""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'chmm_tests', 'naturecomm_cscg'))

from tune_chmm_alergia import (  # noqa: E402
    reduced_alphabet, encode_reduced, chmm_eval_tm_reduced)

CHMM_KS = [16, 24, 32, 48]
CHMM_SEEDS = [0, 1, 2]
CHMM_EM_ITERS = 30

OUT_CSV = os.path.join(HERE, 'binary_adder_high_K_results.csv')


def run_one(args):
    K, seed = args
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, 'chmm_tests', 'naturecomm_cscg'))
    from binary_alphabet_adder import simulate_random_binary_alphabet_adders
    from chmm_actions import CHMM
    tr = simulate_random_binary_alphabet_adders(n_runs=200,
        num_range=(0, 32), max_steps=200_000, seed=42)
    te = simulate_random_binary_alphabet_adders(n_runs=10,
        num_range=(0, 1000), max_steps=200_000, seed=123)
    tuple_to_id, id_to_tuple = reduced_alphabet(tr['runs'])
    nA = len(id_to_tuple)
    train_x = np.concatenate(
        [encode_reduced(t, tuple_to_id) for t in tr['runs']]).astype(np.int64)
    train_a = np.zeros_like(train_x)
    n_clones_arr = np.full(nA, K, dtype=np.int64)
    t0 = time.time()
    model = CHMM(n_clones=n_clones_arr, x=train_x, a=train_a,
                 pseudocount=1e-3, seed=seed)
    model.learn_em_T(train_x, train_a, n_iter=CHMM_EM_ITERS, term_early=True)
    fit_t = time.time() - t0
    acc, total, terr, perf = chmm_eval_tm_reduced(
        model, te['runs'], tuple_to_id, id_to_tuple, n_clones_arr)
    return dict(K=K, seed=seed, mean_acc=float(acc.mean()),
                errors=int(terr), n_predictions=int(total[0]),
                perfect=int(perf), fit_s=fit_t)


def main():
    n_workers = max(1, min(8, (os.cpu_count() or 4) - 1))
    tasks = [(K, seed) for K in CHMM_KS for seed in CHMM_SEEDS]
    print(f"binary_adder high-K sweep: {len(tasks)} configs, "
          f"{n_workers} workers", flush=True)
    rows = []
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_one, tasks):
            rows.append(r)
            print(f"  K={r['K']} s={r['seed']}: errors={r['errors']}/72217 "
                  f"perfect={r['perfect']}/10 [fit={r['fit_s']:.1f}s, "
                  f"total={time.time()-t0:.0f}s]", flush=True)
    fields = ['K', 'seed', 'mean_acc', 'errors', 'n_predictions', 'perfect', 'fit_s']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
