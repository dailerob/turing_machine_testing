"""
v1 — original design.

Rectangular layout: top row scan phase, bottom row modify phase, halt above.
Phase-coloured filled nodes. Walking edges grey, writing edges red.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import COLORS, apply_style  # noqa: E402

from _common import (  # noqa: E402
    PHASE_DARK, PHASE_LIGHT, PHASE_OF,
    PRETTY, TRANSITIONS, draw_edge, draw_node, draw_self_loop, is_writing,
    save, split_edges,
)

apply_style()

NODE_R = 0.42

POSITIONS = {
    'FIND_SEP': (0.0, 2.4),
    'CZ0':      (3.0, 2.4),
    'CZ1':      (6.0, 2.4),
    'DEC':      (6.0, 0.0),
    'GOTO_A':   (3.0, 0.0),
    'INC':      (0.0, 0.0),
    'H':        (3.0, 4.2),
}

LOOP_DIR = {
    'FIND_SEP': ( 0.0,  1.0),
    'CZ0':      (-0.95, 0.32),
    'CZ1':      ( 0.0,  1.0),
    'DEC':      ( 0.0, -1.0),
    'GOTO_A':   ( 0.0, -1.0),
    'INC':      ( 0.0, -1.0),
}


def edge_color_for(labels):
    return COLORS['red_dark'] if any(is_writing(r, w) for r, w, _ in labels) else COLORS['muted']


def main():
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
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

    ax.set_xlim(-1.6, 7.6)
    ax.set_ylim(-1.4, 5.0)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.suptitle('Binary-alphabet adder Turing machine', fontsize=10,
                 color=COLORS['text'], y=0.97)

    save(fig, Path(__file__).with_suffix(''))


if __name__ == '__main__':
    main()
