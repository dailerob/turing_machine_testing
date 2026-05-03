"""Plot a representative subset of M4 monthly series."""
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
sys.path.insert(0, HERE); sys.path.insert(0, M4_ROOT); sys.path.insert(0, ROOT)
import data_loader as dl


def main():
    train = dl.load_train("Monthly"); test = dl.load_test("Monthly")
    ids = ["M1", "M5000", "M15000", "M25000", "M35000", "M47000"]
    fig, axes = plt.subplots(len(ids), 2, figsize=(14, 2.6 * len(ids)))
    for row, sid in enumerate(ids):
        s_train = train[sid]; s_test = test[sid]
        ax = axes[row, 0]
        ax.plot(np.arange(len(s_train)), s_train, color='steelblue',
                linewidth=0.5, label='train')
        ax.plot(np.arange(len(s_train), len(s_train) + len(s_test)),
                s_test, color='salmon', linewidth=1.0, label='test (18m)')
        ax.set_title(f'{sid} - full series (n={len(s_train)} months train)')
        ax.set_xlabel('month'); ax.set_ylabel('value')
        ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
        zoom = 60  # 5 yrs
        start = max(0, len(s_train) - zoom)
        ax = axes[row, 1]
        ax.plot(np.arange(start, len(s_train)), s_train[start:],
                color='steelblue', linewidth=0.7, label='train')
        ax.plot(np.arange(len(s_train), len(s_train) + len(s_test)),
                s_test, color='salmon', linewidth=1.2, label='test (18m)')
        ax.axvline(len(s_train), color='black', linestyle=':', linewidth=0.8)
        ax.set_title(f'{sid} - last 5y of train + 18m test')
        ax.set_xlabel('month'); ax.set_ylabel('value')
        ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle('M4 Monthly: representative series', fontsize=12, y=1.0)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig_m4_monthly_sample.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
