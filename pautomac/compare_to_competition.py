"""Parse the PAutomaC competition leaderboard and compare to our GDC
and CHMM results.

Leaderboard source: https://grammarlearning.org/pautomac/
Competition winner overall: Shibata Yoshinaka (212 pts; 5/3/2/1
scoring per problem).  Other top teams: Mans Hulden, David Llorens,
Raphael Bailly (spectral), Fabio Kepler.

For each problem we extract the floor (minimum perplexity) and the
best of the four submitted scores; then compare against ours.
"""
from __future__ import annotations
import os, re, csv
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LB_HTML = os.path.join(HERE, 'data', 'leaderboard.html')
SWEEP_CSV = os.path.join(HERE, 'results', 'full_sweep.csv')


def parse_leaderboard():
    html = open(LB_HTML).read()
    tables = re.findall(r'<table[^>]*>(.*?)</table>', html, flags=re.DOTALL)
    cells_text = []
    for t in tables:
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', t, flags=re.DOTALL)
        for r in rows:
            cs = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', r, flags=re.DOTALL)
            cs = [re.sub(r'<[^>]+>', '', c).strip() for c in cs]
            cells_text.extend(cs)
    by_problem = {}
    floor_re = re.compile(r"Minimal perplexity \(solution\):\s*([0-9eE.+-]+)")
    score_re = re.compile(r"(First|Second|Third|Fourth):\s*([^()]+?)\s*\(([0-9eE.+-]+)\)")
    # Cells alternate between "Problem N" labels and the data cell
    # ("Minimal ... First ... Second ... ").  Pair them up.
    pending_problem = None
    for cell in cells_text:
        if not cell:
            continue
        m_problem = re.match(r"^Problem\s+(\d+)\s*$", cell)
        if m_problem:
            pending_problem = int(m_problem.group(1))
            continue
        if pending_problem is None:
            continue
        m_floor = floor_re.search(cell)
        if not m_floor:
            pending_problem = None
            continue
        scores = []
        for m in score_re.finditer(cell):
            scores.append((m.group(1), m.group(2).strip(), float(m.group(3))))
        by_problem[pending_problem] = {
            'floor': float(m_floor.group(1)),
            'submissions': scores,
        }
        pending_problem = None
    return by_problem


def main():
    lb = parse_leaderboard()
    print(f"Parsed {len(lb)} leaderboard entries")

    # Load our results
    rows = []
    with open(SWEEP_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(dict(model=r['model'], problem=int(r['problem']),
                             score=float(r['score']),
                             floor=float(r['entropy_floor']),
                             gap=float(r['gap'])))
    by_p = defaultdict(dict)
    for r in rows:
        by_p[r['problem']][r['model']] = r

    print(f"\n{'p':>3s}  {'floor':>10s}  {'compete winner':>16s}  "
          f"{'score':>10s}  {'gap':>9s}  | "
          f"{'best CHMM':>10s}  {'gap':>8s}  | "
          f"{'best GDC':>10s}  {'gap':>8s}  | "
          f"{'GDC vs winner':>14s}")
    out_rows = []
    for pi in sorted(lb):
        ent = lb[pi]
        if pi not in by_p:
            continue
        # Winner = entry labelled "First"
        winner = next((s for s in ent['submissions'] if s[0] == 'First'), None)
        if winner is None:
            continue
        winner_team, winner_score = winner[1], winner[2]
        winner_gap = winner_score - ent['floor']

        # Our best CHMM and GDC
        our = by_p[pi]
        chmm = [(m, our[m]) for m in our if 'chmm' in m]
        gdc = [(m, our[m]) for m in our if 'gdc' in m]
        best_chmm = min(chmm, key=lambda x: x[1]['gap']) if chmm else None
        best_gdc = min(gdc, key=lambda x: x[1]['gap']) if gdc else None

        gdc_score = best_gdc[1]['score']
        gdc_gap = best_gdc[1]['gap']
        chmm_gap = best_chmm[1]['gap'] if best_chmm else float('nan')
        chmm_score = best_chmm[1]['score'] if best_chmm else float('nan')
        gdc_vs_winner = gdc_score - winner_score
        out_rows.append({
            'problem': pi, 'floor': ent['floor'],
            'winner_team': winner_team, 'winner_score': winner_score,
            'winner_gap': winner_gap,
            'best_chmm_score': chmm_score, 'best_chmm_gap': chmm_gap,
            'best_gdc_score': gdc_score, 'best_gdc_gap': gdc_gap,
            'gdc_minus_winner': gdc_vs_winner,
        })

        team_short = winner_team[:14]
        print(f"{pi:>3d}  {ent['floor']:>10.3f}  "
              f"{team_short:>16s}  {winner_score:>10.3f}  "
              f"{winner_gap:>9.4f}  | "
              f"{chmm_score:>10.3f}  {chmm_gap:>8.3f}  | "
              f"{gdc_score:>10.3f}  {gdc_gap:>8.3f}  | "
              f"{gdc_vs_winner:>+14.3f}")

    # Summary
    out_csv = os.path.join(HERE, 'results', 'compete_compare.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"\nWrote {out_csv}")

    print("\n=== Summary across 48 problems ===")
    winner_gaps = np.array([r['winner_gap'] for r in out_rows])
    chmm_gaps = np.array([r['best_chmm_gap'] for r in out_rows])
    gdc_gaps = np.array([r['best_gdc_gap'] for r in out_rows])
    floors = np.array([r['floor'] for r in out_rows])

    print(f"  competition winner    mean_gap={winner_gaps.mean():>8.3f}  "
          f"median={np.median(winner_gaps):>8.3f}  "
          f"max={winner_gaps.max():>8.3f}")
    print(f"  best CHMM             mean_gap={chmm_gaps.mean():>8.3f}  "
          f"median={np.median(chmm_gaps):>8.3f}  "
          f"max={chmm_gaps.max():>8.3f}")
    print(f"  best GDC              mean_gap={gdc_gaps.mean():>8.3f}  "
          f"median={np.median(gdc_gaps):>8.3f}  "
          f"max={gdc_gaps.max():>8.3f}")

    # gmean(gap / floor)
    gmean_winner = np.exp(np.mean(np.log(np.maximum(winner_gaps, 1e-9))))
    gmean_chmm = np.exp(np.mean(np.log(np.maximum(chmm_gaps, 1e-9))))
    gmean_gdc = np.exp(np.mean(np.log(np.maximum(gdc_gaps, 1e-9))))
    print(f"\n  gmean(gap):  winner={gmean_winner:.4f}  "
          f"CHMM={gmean_chmm:.4f}  GDC={gmean_gdc:.4f}")

    # ----- per-team breakdown across all 48 problems -----
    print("\n=== Per-team gap distributions (competition entries) ===")
    teams = {}
    for pi in sorted(lb):
        for rank, team, score in lb[pi]['submissions']:
            teams.setdefault(team, []).append(score - lb[pi]['floor'])
    for team, gs in sorted(teams.items(),
                           key=lambda kv: -np.mean(kv[1])):
        gs = np.asarray(gs)
        gm = np.exp(np.log(np.maximum(gs, 1e-9)).mean())
        print(f"  {team:>20s}  n={len(gs):>3d}  "
              f"mean={gs.mean():>7.4f}  median={np.median(gs):>7.4f}  "
              f"max={gs.max():>7.4f}  gmean(gap)={gm:>8.5f}")

    # how often is GDC within X of competition winner
    deltas = gdc_gaps - winner_gaps
    print(f"\n  GDC vs winner per-problem gap delta:")
    print(f"    mean delta:    {deltas.mean():+.3f}")
    print(f"    median delta:  {np.median(deltas):+.3f}")
    print(f"    GDC <= winner: {int((deltas <= 0).sum())} / 48")
    print(f"    GDC < winner-1.0: {int((deltas < -1.0).sum())} / 48")
    print(f"    GDC < winner-0.1: {int((deltas < -0.1).sum())} / 48")
    print(f"    GDC > winner+1.0: {int((deltas > 1.0).sum())} / 48")
    print(f"    GDC > winner+10.0: {int((deltas > 10.0).sum())} / 48")

    # Top 5 worst problems for GDC vs winner
    rank = sorted(range(48), key=lambda i: -deltas[i])
    print("\n  Top 10 problems where GDC is worst-vs-winner:")
    for i in rank[:10]:
        r = out_rows[i]
        print(f"    p{r['problem']:>3d}: floor={r['floor']:>8.2f}  "
              f"winner={r['winner_score']:>8.3f} ({r['winner_team']})  "
              f"GDC={r['best_gdc_score']:>8.3f}  delta={deltas[i]:>+8.3f}")
    print("\n  Top 10 problems where GDC is best-vs-winner:")
    for i in rank[-10:][::-1]:
        r = out_rows[i]
        print(f"    p{r['problem']:>3d}: floor={r['floor']:>8.2f}  "
              f"winner={r['winner_score']:>8.3f} ({r['winner_team']})  "
              f"GDC={r['best_gdc_score']:>8.3f}  delta={deltas[i]:>+8.3f}")


if __name__ == "__main__":
    main()
