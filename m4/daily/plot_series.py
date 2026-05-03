"""Plot a representative subset of M4 daily series to understand their
structure before any modelling.

Daily has 4,227 series with varying lengths (93 - 9,919 days, median
~2,940). Categories include Demographic, Macro, Industry, Finance,
Other (per M4-info.csv).
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
M4_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(M4_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, M4_ROOT)
sys.path.insert(0, ROOT)

import data_loader as dl


def main():
    train = dl.load_train("Daily")
    test = dl.load_test("Daily")
    horizon = dl.horizon("Daily")

    # Pick a spread: short, medium, long; from different parts of the index
    ids = ["D1", "D100", "D500", "D1000", "D2500", "D4000"]
    fig, axes = plt.subplots(len(ids), 2, figsize=(14, 2.6 * len(ids)))
    for row, sid in enumerate(ids):
        s_train = train[sid]
        s_test = test[sid]
        # Left: full training series with test continuation
        ax = axes[row, 0]
        ax.plot(np.arange(len(s_train)), s_train, color='steelblue',
                linewidth=0.4, label='train')
        ax.plot(np.arange(len(s_train), len(s_train) + len(s_test)),
                s_test, color='salmon', linewidth=1.0, label='test (14d)')
        ax.set_title(f'{sid} — full series (n={len(s_train)} days train)')
        ax.set_xlabel('day'); ax.set_ylabel('value')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        # Right: zoom into last ~120 days of train + test
        zoom = 120
        start = max(0, len(s_train) - zoom)
        ax = axes[row, 1]
        ax.plot(np.arange(start, len(s_train)), s_train[start:],
                color='steelblue', linewidth=0.7, label='train')
        ax.plot(np.arange(len(s_train), len(s_train) + len(s_test)),
                s_test, color='salmon', linewidth=1.2, label='test (14d)')
        ax.axvline(len(s_train), color='black', linestyle=':', linewidth=0.8)
        ax.set_title(f'{sid} — last 120 days of train + 14d test')
        ax.set_xlabel('day'); ax.set_ylabel('value')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('M4 Daily: representative time series', fontsize=12, y=1.0)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig_m4_daily_sample.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
