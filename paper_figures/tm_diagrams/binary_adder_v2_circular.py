"""
v2 — circular layout.

The 6 cycling states (FIND_SEP, CZ0, CZ1, DEC, GOTO_A, INC) sit evenly
around a ring in execution order; H sits off to the upper-right since it
only has the single CZ0 -> H edge. Self-loops point radially outward.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import COLORS, apply_style  # noqa: E402

from _common import (  # noqa: E402
    CYCLE, PHASE_DARK, PHASE_LIGHT, PHASE_OF,
    PRETTY, TRANSITIONS, draw_edge, draw_node, draw_self_loop, is_writing,
    save, split_edges,
)

apply_style()

NODE_R = 0.42
RING_R = 2.6

POSITIONS: dict[str, tuple[float, float]] = {}
LOOP_DIR: dict[str, tuple[float, float]] = {}
for i, name in enumerate(CYCLE):
    angle = np.pi / 2 - 2 * np.pi * i / len(CYCLE)  # start at top, go clockwise
    POSITIONS[name] = (RING_R * np.cos(angle), RING_R * np.sin(angle))
    LOOP_DIR[name] = (np.cos(angle), np.sin(angle))  # radial outward

POSITIONS['H'] = (RING_R + 1.6, RING_R + 0.4)


def edge_color_for(labels):
    return COLORS['red_dark'] if any(is_writing(r, w) for r, w, _ in labels) else COLORS['muted']


def main():
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    cross, loops = split_edges(TRANSITIONS)

    for (src, dst), labels in cross.items():
        draw_edge(ax, POSITIONS[src], POSITIONS[dst], labels,
                  r_src=NODE_R, r_dst=NODE_R, edge_color=edge_color_for(labels))

    for s, labels in loops.items():
        draw_self_loop(ax, POSITIONS[s], LOOP_DIR[s], labels,
                       node_radius=NODE_R, edge_color=edge_color_for(labels))

    for name, pos in POSITIONS.items():
        ph = PHASE_OF[name]
        draw_node(ax, pos, PRETTY[name], radius=NODE_R,
                  face=PHASE_LIGHT[ph], edge_color=PHASE_DARK[ph])

    ax.set_xlim(-RING_R - 1.7, RING_R + 2.5)
    ax.set_ylim(-RING_R - 1.7, RING_R + 1.7)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.suptitle('Binary-alphabet adder TM — circular layout', fontsize=10,
                 color=COLORS['text'], y=0.97)

    save(fig, Path(__file__).with_suffix(''))


if __name__ == '__main__':
    main()
