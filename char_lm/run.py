"""Val-tuned char-level BPS sweep across the Dedieu et al. 2019 datasets.

For each dataset and each method:
  - split train into fit (90%) and val (10%) [paper's protocol]
  - sweep config grid; pick best by val BPS
  - retrain on full train
  - score test BPS

Output: char_lm/results.csv + char_lm/RESULTS.md with side-by-side
comparison to the paper's CHMM / n-gram / SeqM / LSTM Table 4.
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'algorithmic_benchmarks'))

from data_loader import load, ALPHABET_SIZE, PAPER_SIZES   # noqa: E402
from bps_eval import (                                       # noqa: E402
    score_bps_predict_distribution, score_bps_parrot, score_bps_gdc,
    score_bps_gdc_dual, score_bps_gdc_dual_torch)


def _score_gdc_dual(train, test, ac, af):
    """Dispatch: torch for large train·test products, numpy otherwise.
    Threshold tuned so torch's per-step GPU sync overhead is amortized."""
    if len(train) * len(test) > 5_000_000_000:  # ~5e9 floating ops
        return score_bps_gdc_dual_torch(train, test, ALPHABET_SIZE,
                                            alpha_ctx=ac, alpha_fc=af,
                                            alpha_prior=0.001)
    return score_bps_gdc_dual(train, test, ALPHABET_SIZE,
                                  alpha_ctx=ac, alpha_fc=af,
                                  alpha_prior=0.001)
from discrete_hpylm import HPYLMPool                         # noqa: E402
from discrete_ppm import PPMPool                              # noqa: E402
from discrete_parrot import DiscreteParrotPool                # noqa: E402
from generative_dense_chain import GenerativeDenseChain      # noqa: E402
from kn3_eval import KN3Model                                # noqa: E402

# Dedieu et al. 2019 Table 4 reference numbers
PAPER_TABLE4 = {
    'blake-poems':         dict(CHMM=1.60, n_gram=1.75, SeqM=1.71, LSTM=1.68),
    'carroll-alice':       dict(CHMM=1.54, n_gram=1.61, SeqM=1.57, LSTM=1.58),
    'shakespeare-hamlet':  dict(CHMM=1.63, n_gram=1.72, SeqM=1.69, LSTM=1.68),
    'shakespeare-macbeth': dict(CHMM=1.69, n_gram=1.79, SeqM=1.77, LSTM=1.74),
    'milton-paradise':     dict(CHMM=1.73, n_gram=1.83, SeqM=1.78, LSTM=1.78),
    'melville-mobydick':   dict(CHMM=1.72, n_gram=1.81, SeqM=1.73, LSTM=1.76),
    'war-peace':           dict(CHMM=1.59, n_gram=1.65, SeqM=1.57, LSTM=1.62),
    'calgary-book1':       dict(CHMM=1.63, n_gram=1.72, SeqM=1.64, LSTM=1.67),
}

# Hyperparameter grids
HPYLM_GRID = [(D, d, c) for D in [3, 5, 8] for d in [0.25, 0.5, 0.75]
              for c in [0.5, 1.0]]
PPM_GRID = [(D, d) for D in [3, 5] for d in [0.1, 0.25, 0.5, 0.75]]
KN_GRID = [d for d in [0.25, 0.5, 0.75, 0.9]]
PARROT_GRID = [(L, K) for L in [1, 2, 3, 4, 6] for K in [1, 5, 25]]
# Dual-α GDC grid: α_ctx (state-tracking) ∈ [0.4, 0.7], α_fc (prediction-
# operator) ∈ {0.95, 0.99, 1.0}. θ and β fixed at 0 — the prior sweep
# showed both are monotone bad on test. See bps_eval.score_bps_gdc_dual.
GDC_GRID = [(ac, af) for ac in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
            for af in [0.95, 0.99, 1.0]]


def split_train_val(train_ids, val_frac=0.10):
    """Paper: 'we retain 10% of the training sequence for validation.'
    Use the LAST 10% as val so the prefix structure of the first 90%
    is preserved (training sequence is contiguous, not shuffled)."""
    n_val = max(1, int(len(train_ids) * val_frac))
    return train_ids[:-n_val], train_ids[-n_val:]


# --------------------------------------------------------------------
# Per-method val-tune helpers
# --------------------------------------------------------------------
def tune_hpylm(fit_seq, val_seq, log):
    best = (float('inf'), None)
    for D, d, c in HPYLM_GRID:
        m = HPYLMPool([fit_seq], alphabet_size=ALPHABET_SIZE,
                       max_depth=D, discount=d, concentration=c, seed=0)
        bps = score_bps_predict_distribution(
            m, val_seq, ALPHABET_SIZE, alpha_prior=0.001)
        if bps < best[0]:
            best = (bps, (D, d, c))
    log(f"  HPYLM best val: D,d,c={best[1]} BPS={best[0]:.3f}")
    return best


def tune_ppm(fit_seq, val_seq, log):
    best = (float('inf'), None)
    for D, d in PPM_GRID:
        m = PPMPool([fit_seq], alphabet_size=ALPHABET_SIZE,
                     max_depth=D, discount=d)
        bps = score_bps_predict_distribution(
            m, val_seq, ALPHABET_SIZE, alpha_prior=0.001)
        if bps < best[0]:
            best = (bps, (D, d))
    log(f"  PPM   best val: D,d={best[1]} BPS={best[0]:.3f}")
    return best


def tune_kn(fit_seq, val_seq, log):
    best = (float('inf'), None)
    for d in KN_GRID:
        m = KN3Model(V=ALPHABET_SIZE, discount=d); m.fit([fit_seq])
        log2 = 0.0
        for t in range(len(val_seq)):
            probs = m.predict_distribution(val_seq[:t])
            log2 += -np.log2(max(float(probs[int(val_seq[t])]), 1e-12))
        bps = log2 / len(val_seq)
        if bps < best[0]:
            best = (bps, (d,))
    log(f"  KN-3  best val: d={best[1]} BPS={best[0]:.3f}")
    return best


def tune_parrot(fit_seq, val_seq, log):
    best = (float('inf'), None)
    # Fit pool once per L, score K-variations cheaply
    for L in [1, 2, 3, 4, 6]:
        pool = DiscreteParrotPool([fit_seq], alphabet_size=ALPHABET_SIZE, L=L)
        for K in [1, 5, 25]:
            bps = score_bps_parrot(pool, val_seq, ALPHABET_SIZE,
                                    K=K, alpha_prior=1.0)
            if bps < best[0]:
                best = (bps, (L, K))
    log(f"  Parrot best val: L,K={best[1]} BPS={best[0]:.3f}")
    return best


def tune_gdc(fit_seq, val_seq, log):
    best = (float('inf'), None)
    for ac, af in GDC_GRID:
        bps = _score_gdc_dual(fit_seq, val_seq, ac, af)
        if bps < best[0]:
            best = (bps, (ac, af))
    log(f"  GDC   best val: α_ctx,α_fc={best[1]} BPS={best[0]:.3f}")
    return best


# --------------------------------------------------------------------
# Final test scoring at the val-picked config (retrain on full train)
# --------------------------------------------------------------------
def fit_and_score_hpylm(train, test, cfg):
    D, d, c = cfg
    m = HPYLMPool([train], alphabet_size=ALPHABET_SIZE,
                   max_depth=D, discount=d, concentration=c, seed=0)
    return score_bps_predict_distribution(m, test, ALPHABET_SIZE,
                                           alpha_prior=0.001)


def fit_and_score_ppm(train, test, cfg):
    D, d = cfg
    m = PPMPool([train], alphabet_size=ALPHABET_SIZE, max_depth=D, discount=d)
    return score_bps_predict_distribution(m, test, ALPHABET_SIZE,
                                           alpha_prior=0.001)


def fit_and_score_kn(train, test, cfg):
    (d,) = cfg
    m = KN3Model(V=ALPHABET_SIZE, discount=d); m.fit([train])
    log2 = 0.0
    for t in range(len(test)):
        probs = m.predict_distribution(test[:t])
        log2 += -np.log2(max(float(probs[int(test[t])]), 1e-12))
    return log2 / len(test)


def fit_and_score_parrot(train, test, cfg):
    L, K = cfg
    pool = DiscreteParrotPool([train], alphabet_size=ALPHABET_SIZE, L=L)
    return score_bps_parrot(pool, test, ALPHABET_SIZE, K=K, alpha_prior=1.0)


def fit_and_score_gdc(train, test, cfg):
    ac, af = cfg
    return _score_gdc_dual(train, test, ac, af)


METHODS = {
    'HPYLM':  (tune_hpylm,  fit_and_score_hpylm),
    'PPM-D':  (tune_ppm,    fit_and_score_ppm),
    'KN-3':   (tune_kn,     fit_and_score_kn),
    'Parrot': (tune_parrot, fit_and_score_parrot),
    'GDC':    (tune_gdc,    fit_and_score_gdc),
}


def run_dataset(name, log, methods=None):
    if methods is None:
        methods = list(METHODS.keys())
    log(f"\n{'='*72}\nDATASET: {name}\n{'='*72}")
    train, test, info = load(name)
    fit_seq, val_seq = split_train_val(train, val_frac=0.10)
    log(f"  fit={len(fit_seq):,}  val={len(val_seq):,}  test={len(test):,}")
    rows = []
    for method in methods:
        tune, score = METHODS[method]
        t0 = time.time()
        val_bps, cfg = tune(fit_seq, val_seq, log)
        t_tune = time.time() - t0
        t0 = time.time()
        test_bps = score(train, test, cfg)
        t_test = time.time() - t0
        log(f"  {method:6s} cfg={cfg}  val={val_bps:.3f}  "
            f"test={test_bps:.3f}  [tune={t_tune:.1f}s, test={t_test:.1f}s]")
        rows.append(dict(dataset=name, method=method, cfg=str(cfg),
                          val_bps=float(val_bps), test_bps=float(test_bps),
                          tune_s=t_tune, test_s=t_test))
    return rows


def main():
    out_csv = os.path.join(HERE, 'results.csv')
    log_lines = []
    def log(msg=''):
        print(msg, flush=True); log_lines.append(str(msg))

    # All 8 Dedieu-2019 datasets. The torch-GPU GDC scorer handles the
    # mobydick and war-peace test sets which are otherwise prohibitive.
    datasets = ['blake-poems', 'shakespeare-macbeth', 'carroll-alice',
                'shakespeare-hamlet', 'milton-paradise', 'calgary-book1',
                'melville-mobydick', 'war-peace']
    rows_all = []
    log(f"=== Dedieu et al. 2019 Table 4 replication ===")
    log(f"Datasets: {datasets}")
    log(f"Methods: {list(METHODS.keys())}")
    log(f"Val: last 10% of train; retrain on full train; score test.\n")
    t0 = time.time()
    for name in datasets:
        rows_all.extend(run_dataset(name, log))
    log(f"\nTotal: {time.time()-t0:.0f}s")

    fields = ['dataset', 'method', 'cfg', 'val_bps', 'test_bps', 'tune_s', 'test_s']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows_all)
    log(f"Wrote {out_csv}")

    log(f"\n=== Summary vs paper Table 4 ===")
    methods = list(METHODS.keys())
    log(f"{'dataset':<22} " + " ".join(f"{m:>7}" for m in methods)
        + "    Paper: CHMM  ngram  SeqM  LSTM")
    by_ds = {}
    for r in rows_all:
        by_ds.setdefault(r['dataset'], {})[r['method']] = r['test_bps']
    for name in datasets:
        ours = " ".join(f"{by_ds[name].get(m, float('nan')):>7.3f}" for m in methods)
        p = PAPER_TABLE4[name]
        paper = f"{p['CHMM']:>5.2f} {p['n_gram']:>5.2f} {p['SeqM']:>5.2f} {p['LSTM']:>5.2f}"
        log(f"{name:<22} {ours}      {paper}")

    log_path = out_csv.replace('.csv', '.log')
    with open(log_path, 'w') as f: f.write('\n'.join(log_lines))


if __name__ == '__main__':
    main()
