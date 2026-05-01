"""Summarise the full v2 PAutomaC sweep, including the new off-the-shelf
baselines (KN3 trigram, Spectral OOM, ALERGIA)."""
from __future__ import annotations
import os, csv, re
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'results', 'full_sweep_v2.csv')
LB_HTML = os.path.join(HERE, 'data', 'leaderboard.html')


def parse_leaderboard():
    html = open(LB_HTML).read()
    tables = re.findall(r'<table[^>]*>(.*?)</table>', html, flags=re.DOTALL)
    cells = []
    for t in tables:
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', t, flags=re.DOTALL)
        for r in rows:
            cs = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', r, flags=re.DOTALL)
            cells.extend(re.sub(r'<[^>]+>', '', c).strip() for c in cs)
    by_p = {}
    pending = None
    floor_re = re.compile(r"Minimal perplexity \(solution\):\s*([0-9eE.+-]+)")
    score_re = re.compile(r"(First|Second|Third|Fourth):\s*([^()]+?)\s*\(([0-9eE.+-]+)\)")
    for c in cells:
        m = re.match(r"^Problem\s+(\d+)\s*$", c)
        if m:
            pending = int(m.group(1)); continue
        if pending is None: continue
        mf = floor_re.search(c)
        if not mf:
            pending = None; continue
        subs = [(rk, t.strip(), float(s)) for (rk, t, s) in score_re.findall(c)]
        by_p[pending] = {'floor': float(mf.group(1)), 'submissions': subs}
        pending = None
    return by_p


def main():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append(dict(model=r['model'], problem=int(r['problem']),
                             A=int(r['alphabet_size']),
                             score=float(r['score']),
                             floor=float(r['entropy_floor']),
                             gap=float(r['gap']),
                             train_s=float(r['train_s']),
                             eval_s=float(r['eval_s'])))
    by_p = defaultdict(dict)
    for r in rows:
        by_p[r['problem']][r['model']] = r
    models = sorted({r['model'] for r in rows})
    n_problems = len(by_p)

    # Per-model summary
    print(f"=== Per-model gap statistics across {n_problems} problems ===")
    print(f"{'model':>32s}  {'mean':>9s}  {'median':>9s}  {'max':>9s}  "
          f"{'gmean':>9s}  {'wins':>5s}  {'<floor+0.1':>10s}  "
          f"{'fit_s':>7s}  {'eval_s':>7s}")
    summary = {}
    for m in models:
        gaps, train_ts, eval_ts = [], [], []
        for p in sorted(by_p):
            r = by_p[p].get(m)
            if r is None: continue
            gaps.append(max(r['gap'], 1e-9))
            train_ts.append(r['train_s']); eval_ts.append(r['eval_s'])
        gaps = np.array(gaps)
        gmean = float(np.exp(np.log(gaps).mean()))
        n_wins = 0
        for p in sorted(by_p):
            best = min(by_p[p].items(), key=lambda kv: kv[1]['gap'])
            if best[0] == m:
                n_wins += 1
        n_near = int(np.sum(gaps < 0.1))
        summary[m] = (float(gaps.mean()), float(np.median(gaps)),
                      float(gaps.max()), gmean, n_wins, n_near)
        print(f"{m:>32s}  {gaps.mean():>9.3f}  {np.median(gaps):>9.3f}  "
              f"{gaps.max():>9.3f}  {gmean:>9.4f}  {n_wins:>5d}  "
              f"{n_near:>10d}  {np.mean(train_ts):>7.2f}  "
              f"{np.mean(eval_ts):>7.2f}")

    # Comparison vs leaderboard winner
    lb = parse_leaderboard()
    print(f"\n=== Comparison vs PAutomaC competition winners ===")
    print(f"{'model':>32s}  {'mean(gap)':>9s}  {'median':>9s}  "
          f"{'gmean':>9s}  {'mean delta-winner':>18s}  "
          f"{'beats-winner':>12s}")
    winners = {}
    for p in sorted(lb):
        winner = next((s for s in lb[p]['submissions'] if s[0] == 'First'), None)
        if winner: winners[p] = (winner[1], winner[2])
    for m in models:
        deltas, gaps = [], []
        for p in sorted(by_p):
            if p not in winners: continue
            r = by_p[p].get(m)
            if r is None: continue
            winner_score = winners[p][1]
            winner_gap = winner_score - lb[p]['floor']
            our_gap = r['gap']
            deltas.append(our_gap - winner_gap)
            gaps.append(our_gap)
        if not gaps:
            continue
        gaps_arr = np.maximum(gaps, 1e-9)
        gmean = float(np.exp(np.log(gaps_arr).mean()))
        print(f"{m:>32s}  {np.mean(gaps):>9.3f}  {np.median(gaps):>9.3f}  "
              f"{gmean:>9.4f}  {np.mean(deltas):>+18.3f}  "
              f"{int(sum(d <= 0 for d in deltas)):>12d}")

    # Models ranked by gmean, plus comparison to competition top-4
    print(f"\n=== Final ladder (gmean of gap-to-floor; lower is better) ===")
    competition_team_gmean = {}
    for team in ['Shibata Yoshinaka', 'Mans Hulden', 'David Llorens',
                 'Raphael Bailly', 'Fabio Kepler']:
        gs = []
        for p in sorted(lb):
            for rk, t, s in lb[p]['submissions']:
                if t == team:
                    gs.append(s - lb[p]['floor'])
        if gs:
            gs = np.maximum(gs, 1e-9)
            gmean = float(np.exp(np.log(gs).mean()))
            competition_team_gmean[team] = (gmean, len(gs))

    # Combine our models with competition teams
    ladder = []
    for m, (mean_, med_, max_, gmean_, wins_, near_) in summary.items():
        ladder.append((gmean_, m, 'ours', wins_))
    for team, (gmean_, n) in competition_team_gmean.items():
        ladder.append((gmean_, f"{team} (competition, n={n})", 'competition', None))
    ladder.sort()
    for gmean_, name, kind, wins in ladder:
        wins_str = f"  {wins} wins" if wins is not None else ""
        print(f"  {gmean_:>9.4f}  {name:<55s}  [{kind}]{wins_str}")


if __name__ == "__main__":
    main()
