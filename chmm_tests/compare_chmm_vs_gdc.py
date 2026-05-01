"""Side-by-side comparison plot of CHMM vs GDC across the 9
topologies.  Reads:

    hmm_comparison/paper_topology_best.csv   (GDC)
    chmm_tests/chmm_topology_best.csv        (CHMM)

and emits:

    chmm_tests/fig_chmm_vs_gdc_lift.png
    chmm_tests/fig_chmm_vs_gdc_bps.png
"""

from __future__ import annotations
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def main():
    gdc_rows = load_csv(os.path.join(ROOT, 'hmm_comparison',
                                     'paper_topology_best.csv'))
    chmm_rows = load_csv(os.path.join(HERE, 'chmm_topology_best.csv'))

    gdc = {r['topology']: float(r['mean_lift_3_seeds']) for r in gdc_rows}
    chmm_lift = {r['topology']: float(r['mean_lift']) for r in chmm_rows}
    chmm_bps = {r['topology']: float(r['chmm_bps']) for r in chmm_rows}
    bayes_bps = {r['topology']: float(r['bayes_bps']) for r in chmm_rows}
    chmm_K = {r['topology']: int(r['best_K']) for r in chmm_rows}

    topos = list(chmm_lift.keys())

    # --- Lift bar plot ---
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(topos)); w = 0.36
    g_vals = [min(gdc[t], 2.6) for t in topos]
    c_vals = [min(chmm_lift[t], 2.6) for t in topos]
    b1 = ax.bar(x - w / 2, g_vals, w, label='GDC',
                color='steelblue', edgecolor='black')
    b2 = ax.bar(x + w / 2, c_vals, w, label='CHMM',
                color='darkorange', edgecolor='black')
    for i, t in enumerate(topos):
        ax.text(i - w / 2, min(gdc[t], 2.6) + 0.04,
                f'{gdc[t]:.2f}', ha='center', fontsize=8)
        ax.text(i + w / 2, min(chmm_lift[t], 2.6) + 0.04,
                f'{chmm_lift[t]:.2f}\nK={chmm_K[t]}',
                ha='center', fontsize=8)
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.5,
               label='Bayes-optimal')
    ax.set_xticks(x); ax.set_xticklabels(topos, rotation=20, ha='right')
    ax.set_ylabel('hidden-state alignment lift (mean of 3 seeds)')
    ax.set_title('CHMM vs GDC: hidden-state alignment lift across 9 HMM topologies')
    ax.set_ylim(0, 2.8)
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(HERE, 'fig_chmm_vs_gdc_lift.png')
    plt.savefig(out1, dpi=130); plt.close()
    print('Wrote', out1)

    # --- BPS gap plot ---
    fig, ax = plt.subplots(figsize=(11, 5))
    chmm_gap = [chmm_bps[t] - bayes_bps[t] for t in topos]
    bars = ax.bar(topos, chmm_gap, color='darkorange',
                  edgecolor='black')
    for i, t in enumerate(topos):
        ax.text(i, chmm_gap[i] + 0.0015,
                f'{chmm_gap[i]:.3f}', ha='center', fontsize=8)
    ax.set_ylabel('CHMM bps − Bayes-ceiling bps (lower is better)')
    ax.set_title('CHMM forecasting gap to Bayes ceiling per topology')
    ax.set_xticklabels(topos, rotation=20, ha='right')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(HERE, 'fig_chmm_vs_gdc_bps.png')
    plt.savefig(out2, dpi=130); plt.close()
    print('Wrote', out2)

    # Console summary
    print()
    print(f'{"topology":>17s}  {"GDC lift":>9s}  {"CHMM lift":>9s}  '
          f'{"CHMM K":>6s}  {"CHMM bps":>9s}  {"bayes bps":>9s}  {"gap":>6s}')
    for t in topos:
        gap = chmm_bps[t] - bayes_bps[t]
        print(f'{t:>17s}  {gdc[t]:>9.3f}  {chmm_lift[t]:>9.3f}  '
              f'{chmm_K[t]:>6d}  {chmm_bps[t]:>9.3f}  {bayes_bps[t]:>9.3f}  '
              f'{gap:>+6.3f}')


if __name__ == '__main__':
    main()
