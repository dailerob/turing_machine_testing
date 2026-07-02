"""Classify each PAutomaC target machine as DPFA / HMM / PFA, and break the
leakage-free GDC results (Table 14) down by type (Table 14b).

The 48 PAutomaC problems were generated from three model classes, 16 each
(Verwer et al. 2014). We recover the type from each released model file's
transition section `T: (state,symbol,state) prob`:

  - DPFA : deterministic — every (state,symbol) has exactly one next state.
  - HMM  : the next-state distribution is independent of the emitted symbol
           (P(dst | s, a) = P(dst | s) for all a present in state s).
  - PFA  : neither (non-deterministic, symbol-dependent transitions).

Then joins with results/pautomac_leakage_free.csv (GDC LF + fixed gaps) and
the per-problem perplexity table to report median/mean gap and win counts
per type per method.
"""
from __future__ import annotations
import os, re, csv
from collections import defaultdict, Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'PAutomaC-competition_sets')


def parse_T(path):
    """state -> {symbol -> {dst -> prob}} from the T: section."""
    T = defaultdict(lambda: defaultdict(dict))
    in_T = False
    with open(path) as f:
        for line in f:
            if line.startswith('T:'):
                in_T = True
                continue
            if in_T:
                m = re.match(r'\s*\((\d+),(\d+),(\d+)\)\s+([\d.eE+-]+)', line)
                if m:
                    s, a, d, p = (int(m.group(1)), int(m.group(2)),
                                  int(m.group(3)), float(m.group(4)))
                    T[s][a][d] = p
    return T


def classify(T):
    # DPFA: every (s,a) leads to exactly one dst.
    if all(len(T[s][a]) == 1 for s in T for a in T[s]):
        return 'DPFA'
    # HMM: per state, the (normalised) next-state distribution is the same
    # across all emitted symbols.
    for s in T:
        dists = []
        for a in T[s]:
            tot = sum(T[s][a].values())
            if tot <= 0:
                continue
            dists.append({k: v / tot for k, v in T[s][a].items()})
        for i in range(1, len(dists)):
            keys = set(dists[0]) | set(dists[i])
            if any(abs(dists[0].get(k, 0) - dists[i].get(k, 0)) > 1e-4
                   for k in keys):
                return 'PFA'
    return 'HMM'


def main():
    types = {}
    for i in range(1, 49):
        path = os.path.join(DATA, f'{i}.pautomac_model.txt')
        if not os.path.exists(path):
            continue
        types[i] = classify(parse_T(path))

    out = os.path.join(HERE, 'results', 'pautomac_types.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['problem', 'type'])
        for i in sorted(types):
            w.writerow([i, types[i]])
    print("Type counts:", dict(Counter(types.values())))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
