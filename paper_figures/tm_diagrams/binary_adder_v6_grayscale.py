"""
v6 — greyscale + linestyle.

B&W-print-safe. All nodes are light grey with dark borders, halt is a
double ring. The walk-vs-write distinction is carried by linestyle
(walking = thin solid grey, writing = thicker dashed black) instead of
hue, so the figure stays informative when printed monochrome or viewed
by a colour-blind reader.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import COLORS, apply_style  # noqa: E402

from _common import (  # noqa: E402
    PRETTY, TRANSITIONS, draw_edge, draw_node, draw_self_loop, is_writing,
    save, split_edges,
)

apply_style()

NODE_R = 0.46

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


def edge_style(labels):
    if any(is_writing(r, w) for r, w, _ in labels):
        return dict(edge_color=COLORS['text'], edge_lw=1.3, linestyle=(0, (4, 1.6)))
    return dict(edge_color=COLORS['grey'], edge_lw=0.9, linestyle='-')


def main():
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    cross, loops = split_edges(TRANSITIONS)

    for (src, dst), labels in cross.items():
        draw_edge(ax, POSITIONS[src], POSITIONS[dst], labels,
                  r_src=NODE_R, r_dst=NODE_R, **edge_style(labels))

    for s, labels in loops.items():
        draw_self_loop(ax, POSITIONS[s], LOOP_DIR[s], labels,
                       node_radius=NODE_R, **edge_style(labels))

    for name, pos in POSITIONS.items():
        draw_node(
            ax, pos, PRETTY[name], radius=NODE_R,
            face=COLORS['grey_light'], edge_color=COLORS['text'], edge_lw=1.0,
            text_color=COLORS['text'], text_size=8.5,
            double_ring=(name == 'H'),
        )

    # Legend strip at the bottom
    ax.plot([-0.4, 0.4], [-1.15, -1.15], color=COLORS['grey'], lw=0.9, solid_capstyle='round')
    ax.text(0.6, -1.15, 'walk (read = write)', fontsize=7.5,
            color=COLORS['text'], va='center', ha='left')
    ax.plot([3.4, 4.2], [-1.15, -1.15], color=COLORS['text'], lw=1.3,
            linestyle=(0, (4, 1.6)), solid_capstyle='round')
    ax.text(4.4, -1.15, 'write (read ≠ write)', fontsize=7.5,
            color=COLORS['text'], va='center', ha='left')

    ax.set_xlim(-1.6, 7.6)
    ax.set_ylim(-1.6, 5.0)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.suptitle('Binary-alphabet adder TM — greyscale (B&W safe)',
                 fontsize=10, color=COLORS['text'], y=0.97)

    save(fig, Path(__file__).with_suffix(''))


if __name__ == '__main__':
    main()
