"""Generate summary figures for the M4 hourly write-up.

  fig_smape_distribution.png     — histogram & ECDF of per-series sMAPE
  fig_smape_scatter.png          — per-series GDC-tuned vs naive scatter
  fig_best_worst_forecasts.png   — example forecasts (best, median, worst)
  fig_config_selection.png       — pie/bar of validation-picked configs
"""
from __future__ import annotations
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
M4_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(M4_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, M4_ROOT)
sys.path.insert(0, ROOT)

import data_loader as dl
from v0_basic_gdc import smape, naive_seasonal_forecast, H_HORIZON
from v5_gdc_proper import gdc_proper_forecast


def load_results():
    rows = []
    with open(os.path.join(HERE, 'val_tuned_results.csv')) as f:
        for r in csv.DictReader(f):
            rows.append(dict(sid=r['sid'],
                             window_len=int(r['window_len']),
                             sigma_frac=float(r['sigma_frac']),
                             val_smape=float(r['val_smape']),
                             test_smape=float(r['test_smape']),
                             smape_naive=float(r['smape_naive'])))
    return rows


def fig_distribution(rows, out_path):
    gdc = np.array([r['test_smape'] for r in rows])
    naive = np.array([r['smape_naive'] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Histogram (clipped at 50% for readability)
    bins = np.linspace(0, 50, 51)
    axes[0].hist(np.clip(naive, 0, 50), bins=bins, alpha=0.5,
                 color='gray', label=f'naive (mean={naive.mean():.2f}%)')
    axes[0].hist(np.clip(gdc, 0, 50), bins=bins, alpha=0.6,
                 color='steelblue', label=f'GDC val-tuned (mean={gdc.mean():.2f}%)')
    axes[0].axvline(naive.mean(), color='gray', linestyle='--', linewidth=1)
    axes[0].axvline(gdc.mean(), color='steelblue', linestyle='--', linewidth=1)
    axes[0].set_xlabel('sMAPE (%)'); axes[0].set_ylabel('# of series')
    axes[0].set_title('Per-series sMAPE distribution (clipped at 50%)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # ECDF
    axes[1].plot(np.sort(naive),
                 np.arange(1, len(naive) + 1) / len(naive),
                 color='gray', linewidth=1.5,
                 label=f'naive (median={np.median(naive):.2f}%)')
    axes[1].plot(np.sort(gdc),
                 np.arange(1, len(gdc) + 1) / len(gdc),
                 color='steelblue', linewidth=1.5,
                 label=f'GDC val-tuned (median={np.median(gdc):.2f}%)')
    axes[1].set_xlim(0, 50)
    axes[1].set_xlabel('sMAPE (%)')
    axes[1].set_ylabel('cumulative fraction of series')
    axes[1].set_title('ECDF of per-series sMAPE')
    axes[1].grid(True, alpha=0.3); axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Wrote {out_path}")


def fig_scatter(rows, out_path):
    gdc = np.array([r['test_smape'] for r in rows])
    naive = np.array([r['smape_naive'] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 7))
    upper = max(gdc.max(), naive.max())
    upper = min(upper, 80)
    ax.scatter(naive, gdc, s=10, alpha=0.5, color='steelblue')
    ax.plot([0, upper], [0, upper], color='black', linestyle='--',
            linewidth=1, label='y=x (tied)')
    n_better = int((gdc < naive).sum())
    n_worse = int((gdc > naive).sum())
    ax.set_xlabel('naive seasonal sMAPE (%)')
    ax.set_ylabel('GDC val-tuned sMAPE (%)')
    ax.set_title(f'Per-series sMAPE: GDC vs naive\n'
                 f'GDC better on {n_better}/{len(rows)} series, '
                 f'worse on {n_worse}/{len(rows)}')
    ax.set_xlim(0, upper); ax.set_ylim(0, upper)
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Wrote {out_path}")


def fig_examples(rows, train_d, test_d, out_path):
    """Plot best-case, median-case, worst-case forecasts."""
    rows_sorted = sorted(rows, key=lambda r: r['test_smape'])
    n = len(rows_sorted)
    picks = [
        ('best',   rows_sorted[5]),    # 5th-best (avoid trivially perfect)
        ('25%-ile', rows_sorted[n // 4]),
        ('median', rows_sorted[n // 2]),
        ('75%-ile', rows_sorted[3 * n // 4]),
        ('worst',  rows_sorted[-5]),
    ]
    fig, axes = plt.subplots(len(picks), 1, figsize=(13, 2.5 * len(picks)))
    for ax, (label, r) in zip(axes, picks):
        sid = r['sid']
        train = train_d[sid]; test = test_d[sid]
        n_train = len(train)
        zoom = 4 * 24
        start = max(0, n_train - zoom)
        # Refit and forecast with the chosen config
        cfg = dict(window_len=r['window_len'], sigma_frac=r['sigma_frac'])
        pred, _ = gdc_proper_forecast(train, **cfg, h=H_HORIZON)
        naive = naive_seasonal_forecast(train, H_HORIZON, period=24)
        ax.plot(np.arange(start, n_train), train[start:],
                color='steelblue', linewidth=0.7, label='train')
        ax.plot(np.arange(n_train, n_train + H_HORIZON), test,
                color='black', linewidth=1.4, label='test (actual)')
        ax.plot(np.arange(n_train, n_train + H_HORIZON), pred,
                color='salmon', linewidth=1.2,
                label=f'GDC L={cfg["window_len"]} '
                      f'σ%={cfg["sigma_frac"]:.2f} '
                      f'sMAPE={r["test_smape"]:.1f}%')
        ax.plot(np.arange(n_train, n_train + H_HORIZON), naive,
                color='green', linestyle='--', linewidth=0.9,
                label=f'naive sMAPE={r["smape_naive"]:.1f}%')
        ax.axvline(n_train, color='black', linestyle=':', linewidth=0.8)
        ax.set_title(f'{label}: {sid}')
        ax.set_xlabel('hour'); ax.set_ylabel('value')
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Wrote {out_path}")


def fig_config_selection(rows, out_path):
    sel = Counter((r['window_len'], r['sigma_frac']) for r in rows)
    items = sorted(sel.items(), key=lambda x: -x[1])
    labels = [f'L={k[0]}, σ%={k[1]:.2f}' for k, _ in items]
    counts = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels[::-1], counts[::-1], color='steelblue',
                   edgecolor='black')
    total = sum(counts)
    for bar, c in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{c} ({100*c/total:.1f}%)', va='center', fontsize=8)
    ax.set_xlabel('# of series picking this config')
    ax.set_title(f'Validation-selected configs across {total} series')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Wrote {out_path}")


def main():
    rows = load_results()
    train_d = dl.load_train()
    test_d = dl.load_test()
    fig_distribution(rows, os.path.join(HERE, 'fig_smape_distribution.png'))
    fig_scatter(rows, os.path.join(HERE, 'fig_smape_scatter.png'))
    fig_examples(rows, train_d, test_d,
                 os.path.join(HERE, 'fig_best_worst_forecasts.png'))
    fig_config_selection(rows, os.path.join(HERE, 'fig_config_selection.png'))


if __name__ == "__main__":
    main()
