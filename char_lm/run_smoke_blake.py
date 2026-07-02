"""Smoke-test on blake-poems: fit each baseline + GDC at a single
config, score BPS, print vs paper Table 4.

The point is to validate that our preprocessing pipeline + BPS scorer
land in the right ballpark of the paper's reported numbers — not to
beat them. Once the protocol checks out, run.py does the proper
val-tuning sweep.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from data_loader import load, ALPHABET_SIZE          # noqa: E402
from bps_eval import (                                # noqa: E402
    score_bps_predict_distribution, score_bps_parrot,
    score_bps_gdc)
from discrete_hpylm import HPYLMPool                  # noqa: E402
from discrete_ppm import PPMPool                       # noqa: E402
from discrete_parrot import DiscreteParrotPool         # noqa: E402
from generative_dense_chain import GenerativeDenseChain  # noqa: E402

# Add KN-3 from algorithmic_benchmarks
sys.path.insert(0, os.path.join(ROOT, 'algorithmic_benchmarks'))
from kn3_eval import KN3Model                         # noqa: E402

PAPER_BLAKE = dict(CHMM=1.60, n_gram=1.75, SeqM=1.71, LSTM=1.68)


def main():
    name = 'blake-poems'
    print(f"=== Smoke test: {name} ===\n")
    train, test, info = load(name)
    print(f"Train chars: {len(train):,}  Test chars: {len(test):,}  "
          f"Alphabet: {info['alphabet_size']}\n")

    rows = []

    # ----- HPYLM (closest analog of SeqM in the paper) -----
    t0 = time.time()
    print("Fitting HPYLM (depth=8, d=0.5, conc=1.0)...")
    hpylm = HPYLMPool([train], alphabet_size=ALPHABET_SIZE,
                       max_depth=8, discount=0.5, concentration=1.0, seed=0)
    bps = score_bps_predict_distribution(
        hpylm, test, ALPHABET_SIZE, alpha_prior=0.001)
    print(f"  HPYLM BPS = {bps:.3f}  (paper SeqM = {PAPER_BLAKE['SeqM']:.3f})  "
          f"[{time.time()-t0:.1f}s]\n")
    rows.append(('HPYLM', 8, bps))

    # ----- PPM-D -----
    t0 = time.time()
    print("Fitting PPM-D (depth=8, d=0.5)...")
    ppm = PPMPool([train], alphabet_size=ALPHABET_SIZE,
                   max_depth=8, discount=0.5)
    bps = score_bps_predict_distribution(
        ppm, test, ALPHABET_SIZE, alpha_prior=0.001)
    print(f"  PPM-D BPS = {bps:.3f}  (no direct paper analog) "
          f"[{time.time()-t0:.1f}s]\n")
    rows.append(('PPM-D', 8, bps))

    # ----- KN-3 (closest analog of paper's kylm n-gram, but trigram) -----
    t0 = time.time()
    print("Fitting KN-3 (discount=0.75)...")
    kn3 = KN3Model(V=ALPHABET_SIZE, discount=0.75)
    kn3.fit([train])
    log2_total = 0.0
    for t in range(len(test)):
        prefix = test[:t]
        probs = kn3.predict_distribution(prefix)
        log2_total += -np.log2(max(float(probs[int(test[t])]), 1e-12))
    bps = log2_total / len(test)
    print(f"  KN-3 BPS = {bps:.3f}  (paper n-gram = {PAPER_BLAKE['n_gram']:.3f}) "
          f"[{time.time()-t0:.1f}s]\n")
    rows.append(('KN-3', 3, bps))

    # ----- Parrot -----
    t0 = time.time()
    print("Fitting Parrot (L=4, K=25)...")
    parrot = DiscreteParrotPool([train], alphabet_size=ALPHABET_SIZE, L=4)
    bps = score_bps_parrot(parrot, test, ALPHABET_SIZE, K=25, alpha_prior=1.0)
    print(f"  Parrot BPS = {bps:.3f}  (no direct paper analog) "
          f"[{time.time()-t0:.1f}s]\n")
    rows.append(('Parrot', 4, bps))

    # ----- GDC -----
    t0 = time.time()
    print("Fitting GDC (alpha=0.99, theta=0.005)...")
    gdc = GenerativeDenseChain([train.reshape(-1, 1)],
                                alpha=0.99, theta=0.005, gamma=0.0,
                                transition_type='self_loop',
                                initial_dist='sequence_starts')
    bps = score_bps_gdc(gdc, test, ALPHABET_SIZE, alpha_prior=0.001)
    print(f"  GDC BPS = {bps:.3f}  (no paper baseline; ours)  "
          f"[{time.time()-t0:.1f}s]\n")
    rows.append(('GDC', '0.99/0.005', bps))

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Method':<10} {'cfg':<14} {'BPS':>6}  Paper")
    print("-" * 60)
    paper_map = {'HPYLM': PAPER_BLAKE['SeqM'], 'PPM-D': '-',
                 'KN-3': PAPER_BLAKE['n_gram'], 'Parrot': '-',
                 'GDC': '-'}
    for m, cfg, b in rows:
        ref = paper_map.get(m, '-')
        ref_s = f"{ref:.2f}" if isinstance(ref, float) else ref
        print(f"{m:<10} {str(cfg):<14} {b:>6.3f}  {ref_s}")
    print(f"\nPaper Table 4 for {name}: CHMM={PAPER_BLAKE['CHMM']:.2f}, "
          f"n-gram={PAPER_BLAKE['n_gram']:.2f}, SeqM={PAPER_BLAKE['SeqM']:.2f}, "
          f"LSTM={PAPER_BLAKE['LSTM']:.2f}")


if __name__ == '__main__':
    main()
