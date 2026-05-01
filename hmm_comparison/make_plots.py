"""
Generate figures from the sweep + hypothesis-test CSVs.

Figures:
  fig1_horizon_curves.png       MSE vs horizon (OOM-clip, OOM-soft, GDC, uniform)
  fig2_grid_ratio.png           Heatmap of log(OOM_soft / GDC) at h=1 over (nS, nA)
  fig3_h1_rank.png              H1: OOM vs GDC MSE vs transition-matrix rank
  fig4_h2_emission.png          H2: MSE vs emission Dirichlet concentration
  fig5_h4_topology.png          H4: dense vs sparse topology box summary
"""

from __future__ import annotations
import os
import csv
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(_THIS_DIR, 'main_sweep_results.csv')
H1   = os.path.join(_THIS_DIR, 'h1_rank_results.csv')
H2   = os.path.join(_THIS_DIR, 'h2_emission_concentration_results.csv')
H4   = os.path.join(_THIS_DIR, 'h4_topology_results.csv')


def load(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def figure_horizon_curves():
    rows = load(MAIN)
    horizons = sorted({int(r['horizon']) for r in rows})
    curves = {key: [] for key in ['oom_clip_mse', 'oom_softmax_mse', 'gdc_mse', 'uni_mse']}
    for h in horizons:
        sub = [r for r in rows if int(r['horizon']) == h]
        for key in curves:
            curves[key].append(np.mean([float(r[key]) for r in sub]))

    plt.figure(figsize=(6, 4))
    plt.plot(horizons, curves['oom_clip_mse'],    'o-', label='OOM (clip)')
    plt.plot(horizons, curves['oom_softmax_mse'], 's-', label='OOM (softmax)')
    plt.plot(horizons, curves['gdc_mse'],         'D-', label='GDC')
    plt.plot(horizons, curves['uni_mse'],         ':', label='Uniform baseline', color='grey')
    plt.yscale('log')
    plt.xlabel('Forecasting horizon h')
    plt.ylabel('Mean MSE (next-symbol)')
    plt.title('MSE vs forecasting horizon — averaged over (nS,nA) grid')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig1_horizon_curves.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('Wrote', out)


def figure_grid_ratio():
    rows = load(MAIN)
    h1 = [r for r in rows if int(r['horizon']) == 1]
    grid = collections.defaultdict(list)
    for r in h1:
        grid[(int(r['nS']), int(r['nA']))].append(
            (float(r['oom_softmax_mse']), float(r['gdc_mse']))
        )

    nS_vals = sorted({k[0] for k in grid.keys()})
    nA_vals = sorted({k[1] for k in grid.keys()})
    ratio = np.zeros((len(nS_vals), len(nA_vals)))
    for i, nS in enumerate(nS_vals):
        for j, nA in enumerate(nA_vals):
            pairs = grid[(nS, nA)]
            oom_mean = np.mean([p[0] for p in pairs])
            gdc_mean = np.mean([p[1] for p in pairs])
            ratio[i, j] = np.log10(oom_mean / max(gdc_mean, 1e-12))

    plt.figure(figsize=(6, 5))
    vmax = max(abs(ratio.min()), abs(ratio.max()))
    im = plt.imshow(ratio, cmap='RdBu_r', aspect='auto',
                    origin='lower', vmin=-vmax, vmax=vmax,
                    extent=[min(nA_vals)-0.5, max(nA_vals)+0.5,
                            min(nS_vals)-0.5, max(nS_vals)+0.5])
    plt.colorbar(im, label='log10(MSE_OOM / MSE_GDC) at h=1')
    plt.xticks(nA_vals)
    plt.yticks(nS_vals)
    plt.xlabel('Alphabet size nA')
    plt.ylabel('State count nS')
    plt.title('Relative OOM vs GDC MSE, h=1\n(red = OOM worse, blue = OOM better)')
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig2_grid_ratio.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('Wrote', out)


def figure_h1_rank():
    rows = load(H1)
    ranks = sorted({int(r['rank']) for r in rows})
    oom_h1, gdc_h1 = [], []
    oom_h5, gdc_h5 = [], []
    for r in ranks:
        h1_rows = [x for x in rows if int(x['rank']) == r and int(x['horizon']) == 1]
        h5_rows = [x for x in rows if int(x['rank']) == r and int(x['horizon']) == 5]
        oom_h1.append(np.mean([float(x['oom_mse']) for x in h1_rows]))
        gdc_h1.append(np.mean([float(x['gdc_mse']) for x in h1_rows]))
        oom_h5.append(np.mean([float(x['oom_mse']) for x in h5_rows]))
        gdc_h5.append(np.mean([float(x['gdc_mse']) for x in h5_rows]))

    plt.figure(figsize=(6, 4))
    plt.plot(ranks, oom_h1, 'o-', label='OOM (softmax) h=1')
    plt.plot(ranks, gdc_h1, 'D-', label='GDC h=1')
    plt.plot(ranks, oom_h5, 'o--', label='OOM (softmax) h=5', alpha=0.6)
    plt.plot(ranks, gdc_h5, 'D--', label='GDC h=5', alpha=0.6)
    plt.xlabel('Transition-matrix rank (nS=10)')
    plt.ylabel('Mean MSE')
    plt.title('H1: Does low-rank T favour OOM? (nS=10, nA=6)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig3_h1_rank.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('Wrote', out)


def figure_h2_emission():
    rows = load(H2)
    cs = sorted({float(r['E_concentration']) for r in rows})
    oom_h1, gdc_h1, oom_h5, gdc_h5 = [], [], [], []
    for c in cs:
        h1_rows = [x for x in rows if float(x['E_concentration']) == c and int(x['horizon']) == 1]
        h5_rows = [x for x in rows if float(x['E_concentration']) == c and int(x['horizon']) == 5]
        oom_h1.append(np.mean([float(x['oom_mse']) for x in h1_rows]))
        gdc_h1.append(np.mean([float(x['gdc_mse']) for x in h1_rows]))
        oom_h5.append(np.mean([float(x['oom_mse']) for x in h5_rows]))
        gdc_h5.append(np.mean([float(x['gdc_mse']) for x in h5_rows]))

    plt.figure(figsize=(6, 4))
    plt.plot(cs, oom_h1, 'o-', label='OOM h=1')
    plt.plot(cs, gdc_h1, 'D-', label='GDC h=1')
    plt.plot(cs, oom_h5, 'o--', label='OOM h=5', alpha=0.6)
    plt.plot(cs, gdc_h5, 'D--', label='GDC h=5', alpha=0.6)
    plt.xscale('log')
    plt.xlabel('Emission Dirichlet concentration (↑ = more uniform emissions)')
    plt.ylabel('Mean MSE')
    plt.title('H2: Near-deterministic emissions favour GDC? (nS=6, nA=6)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig4_h2_emission.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('Wrote', out)


def figure_h4_topology():
    rows = load(H4)
    topos = sorted({r['topology'] for r in rows})
    width = 0.35
    horizons = [1, 2, 5, 10, 20]
    plt.figure(figsize=(7, 4))
    x = np.arange(len(horizons))
    colors = {'dense': 'tab:blue', 'sparse2': 'tab:orange'}
    for i, topo in enumerate(topos):
        oom_vals = [np.mean([float(r['oom_mse']) for r in rows
                             if r['topology'] == topo and int(r['horizon']) == h])
                    for h in horizons]
        gdc_vals = [np.mean([float(r['gdc_mse']) for r in rows
                             if r['topology'] == topo and int(r['horizon']) == h])
                    for h in horizons]
        plt.plot(horizons, oom_vals, 'o-', color=colors[topo],
                 label=f'OOM ({topo})')
        plt.plot(horizons, gdc_vals, 'D--', color=colors[topo],
                 label=f'GDC ({topo})', alpha=0.6)
    plt.yscale('log')
    plt.xlabel('Horizon h')
    plt.ylabel('Mean MSE')
    plt.title('H4: Sparse vs dense transition topology (nS=8, nA=4)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_THIS_DIR, 'fig5_h4_topology.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('Wrote', out)


if __name__ == '__main__':
    figure_horizon_curves()
    figure_grid_ratio()
    figure_h1_rank()
    figure_h2_emission()
    figure_h4_topology()
