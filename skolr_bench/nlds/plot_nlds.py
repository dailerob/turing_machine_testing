"""Plot the 4 NLDS trajectories — visual sanity check."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'nlds_data')

systems = ['pendulum', 'duffing', 'lotka_volterra', 'lorenz63']
fig, axes = plt.subplots(len(systems), 2, figsize=(14, 2.6 * len(systems)))
for row, name in enumerate(systems):
    d = np.load(os.path.join(DATA_DIR, f'{name}_seed0.npz'), allow_pickle=True)
    train = d['train']; val = d['val']; test = d['test']
    full = np.concatenate([train, val, test])
    # Left: full series, dim 0
    ax = axes[row, 0]
    ax.plot(full[:, 0], color='steelblue', linewidth=0.4)
    ax.axvline(len(train), color='gray', linestyle=':', linewidth=0.6)
    ax.axvline(len(train) + len(val), color='gray', linestyle=':', linewidth=0.6)
    ax.set_title(f'{name} - dim 0 (full 20k)')
    ax.grid(True, alpha=0.3)
    # Right: zoom to first 1000 steps
    ax = axes[row, 1]
    for i in range(full.shape[1]):
        ax.plot(full[:1000, i], linewidth=0.7, label=f'dim {i}')
    ax.set_title(f'{name} - first 1000 steps')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(HERE, 'fig_nlds_sample.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f"Wrote {out}")
