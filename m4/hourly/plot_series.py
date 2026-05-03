"""Plot a representative subset of M4 hourly series to understand
their structure before building any model."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_loader as dl

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    train = dl.load_train()
    test = dl.load_test()
    ids = ["H1", "H2", "H10", "H50", "H150", "H300"]

    fig, axes = plt.subplots(len(ids), 2, figsize=(14, 2.6 * len(ids)))
    for row, sid in enumerate(ids):
        s_train = train[sid]
        s_test = test[sid]
        # Left: full training series with test continuation
        ax = axes[row, 0]
        ax.plot(np.arange(len(s_train)), s_train, color='steelblue',
                linewidth=0.6, label='train')
        ax.plot(np.arange(len(s_train), len(s_train) + len(s_test)),
                s_test, color='salmon', linewidth=1.0, label='test (48h)')
        ax.set_title(f'{sid} — full series (train+test)')
        ax.set_xlabel('hour'); ax.set_ylabel('value')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        # Right: zoom into last ~10 days of train + test
        zoom = 10 * 24
        start = max(0, len(s_train) - zoom)
        ax = axes[row, 1]
        ax.plot(np.arange(start, len(s_train)), s_train[start:],
                color='steelblue', linewidth=0.8, label='train')
        ax.plot(np.arange(len(s_train), len(s_train) + len(s_test)),
                s_test, color='salmon', linewidth=1.2, label='test (48h)')
        ax.axvline(len(s_train), color='black', linestyle=':', linewidth=0.8)
        ax.set_title(f'{sid} — last 10d of train + test')
        ax.set_xlabel('hour'); ax.set_ylabel('value')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('M4 hourly: representative time series', fontsize=12, y=1.0)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig_m4_hourly_sample.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    main()
