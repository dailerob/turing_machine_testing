"""Run baselines + CHMM + GDC across PAutomaC problems.

Usage:
    python pautomac/run_eval.py --problems 1,2,3
    python pautomac/run_eval.py --problems 1-10
    python pautomac/run_eval.py --problems all   # all 48 (slow)
"""

from __future__ import annotations
import os, sys, argparse, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_loader import load_problem, summary  # noqa: E402
from scoring import pautomac_score  # noqa: E402
from models import (  # noqa: E402
    UniformModel, UnigramModel, BigramModel, CHMMModel, GDCModel,
    KneserNey3gramModel)
from baselines import SpectralOOMModel, AlergiaModel  # noqa: E402
from fast_gdc import BatchedGDCScorer  # noqa: E402


def parse_problems(s):
    if s == 'all':
        return list(range(1, 49))
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if '-' in tok:
            a, b = tok.split('-'); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def get_models():
    return [
        UniformModel(),
        UnigramModel(),
        BigramModel(),
        KneserNey3gramModel(discount=0.75),
        SpectralOOMModel(max_basis_length=3, rank=50, prob_mode='abs'),
        AlergiaModel(eps=0.05),
        CHMMModel(K=2, n_em_iters=50),
        CHMMModel(K=4, n_em_iters=50),
        CHMMModel(K=8, n_em_iters=50),
        # GDC: a small set of (alpha, theta, transition) configs.
        # Use BatchedGDCScorer for ~3× faster eval on large problems.
        BatchedGDCScorer(alpha=0.95, theta=0.05, transition_type='self_loop'),
        BatchedGDCScorer(alpha=0.50, theta=0.005, transition_type='self_loop'),
    ]


def evaluate_model(model, problem, log):
    train, test, true_probs, A = (problem['train'], problem['test'],
                                  problem['true_probs'],
                                  problem['alphabet_size'])
    t0 = time.time()
    model.fit(train, A)
    train_t = time.time() - t0
    t0 = time.time()
    if hasattr(model, 'score_test_set'):
        log_probs = model.score_test_set(test)
    else:
        log_probs = np.empty(len(test), dtype=np.float64)
        for i, seq in enumerate(test):
            log_probs[i] = model.log_prob(seq)
    eval_t = time.time() - t0
    r = pautomac_score(log_probs, true_probs)
    log(f"  {model.name:>22s}  score={r['score']:>9.3f}  "
        f"floor={r['entropy_floor']:>8.3f}  gap={r['gap']:>8.3f}  "
        f"lift={r['lift']:>+5.3f}  "
        f"[fit={train_t:5.1f}s  eval={eval_t:5.1f}s]")
    return dict(model=model.name, **r,
                train_s=train_t, eval_s=eval_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problems', default='1,3,7',
                    help='Comma-separated or ranges, or "all"')
    ap.add_argument('--out', default=os.path.join(HERE, 'results',
                                                  'pautomac_results.csv'))
    args = ap.parse_args()

    problems = parse_problems(args.problems)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    log_lines = []
    def log(msg=""):
        print(msg, flush=True); log_lines.append(str(msg))

    log(f"=== PAutomaC eval on problems {problems} ===")
    rows = []
    for pi in problems:
        try:
            problem = load_problem(pi)
        except FileNotFoundError as e:
            log(f"\nproblem {pi}: missing data ({e}); skipping")
            continue
        s = summary(problem)
        log(f"\n--- problem {pi}: A={s['alphabet_size']}, "
            f"n_train={s['n_train']} (mean_len={s['train_len_mean']:.1f}, "
            f"max={s['train_len_max']}), "
            f"n_test={s['n_test']} (mean_len={s['test_len_mean']:.1f}) ---")
        for model in get_models():
            try:
                r = evaluate_model(model, problem, log)
                r['problem'] = pi
                r['alphabet_size'] = s['alphabet_size']
                rows.append(r)
            except Exception as e:
                log(f"  {model.name}: FAILED ({e})")

    if rows:
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        log(f"\nWrote {args.out}")

    log_path = args.out.replace('.csv', '.log')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"Wrote {log_path}", flush=True)


if __name__ == "__main__":
    main()
