"""
v5 — phase-grouped background panels.

Two soft background panels group the scan-phase nodes (top row) and the
modify-phase nodes (bottom row). The two-pass structure of the algorithm —
forward scan, then backward modify — is the most important fact about it,
and the panels make that obvious before the reader looks at any edges.

CZ0's self-loop points down (into the gap between rows) so it doesn't
collide with the FIND-SEP -> CZ0 edge label or the CZ0 -> H edge above.
The halt state H uses the textbook double-ring convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import COLORS, apply_style  # noqa: E402

from _common import (  # noqa: E402
    PHASE_DARK, PHASE_LIGHT, PHASE_OF,
    PRETTY, TRANSITIONS, draw_edge, draw_node, draw_self_loop, is_writing,
    save, split_edges,
)

apply_style()

NODE_R = 0.48
COL_X = (0.0, 3.4, 6.8)
ROW_Y = (0.0, 2.6)

POSITIONS = {
    'FIND_SEP': (COL_X[0], ROW_Y[1]),
    'CZ0':      (COL_X[1], ROW_Y[1]),
    'CZ1':      (COL_X[2], ROW_Y[1]),
    'DEC':      (COL_X[2], ROW_Y[0]),
    'GOTO_A':   (COL_X[1], ROW_Y[0]),
    'INC':      (COL_X[0], ROW_Y[0]),
    'H':        (COL_X[1], ROW_Y[1] + 1.85),
}

# CZ0's self-loop dives into the inter-row gap so it stays out of the
# top-row edge labels and the CZ0 -> H edge above.
LOOP_DIR = {
    'FIND_SEP': ( 0.0,  1.0),
    'CZ0':      ( 0.0, -1.0),
    'CZ1':      ( 0.0,  1.0),
    'DEC':      ( 0.0, -1.0),
    'GOTO_A':   ( 0.0, -1.0),
    'INC':      ( 0.0, -1.0),
}


def edge_color_for(labels):
    return COLORS['red_dark'] if any(is_writing(r, w) for r, w, _ in labels) else COLORS['muted']


def _phase_panel(ax, x0, y0, x1, y1, *, face, edge):
    box = FancyBboxPatch(
        (x0, y0), (x1 - x0), (y1 - y0),
        boxstyle='round,pad=0.0,rounding_size=0.28',
        facecolor=face, edgecolor=edge, linewidth=0.6, zorder=0,
    )
    ax.add_patch(box)


def main():
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    pad_x = 0.95
    pad_y = 0.75

    scan_x0 = COL_X[0] - pad_x
    scan_x1 = COL_X[2] + pad_x
    scan_y0 = ROW_Y[1] - pad_y
    scan_y1 = ROW_Y[1] + pad_y
    mod_y0 = ROW_Y[0] - pad_y
    mod_y1 = ROW_Y[0] + pad_y

    _phase_panel(ax, scan_x0, scan_y0, scan_x1, scan_y1,
                 face='#EAF1F8', edge=COLORS['blue_light'])
    _phase_panel(ax, scan_x0, mod_y0, scan_x1, mod_y1,
                 face='#FBEDE9', edge=COLORS['red_light'])

    # Phase labels sit above/below the panels in the gap between the
    # FIND-SEP / INC self-loop labels (column x=0) and the H column (x=3.4),
    # so they don't collide with either.
    ax.text(2.0, scan_y1 + 0.20, '1.  forward scan',
            fontsize=10.0, color=COLORS['blue_dark'],
            ha='center', va='bottom', style='italic')
    ax.text(2.0, mod_y0 - 0.20, '2.  backward modify',
            fontsize=10.0, color=COLORS['red_dark'],
            ha='center', va='top', style='italic')

    cross, loops = split_edges(TRANSITIONS)
    for (src, dst), labels in cross.items():
        draw_edge(ax, POSITIONS[src], POSITIONS[dst], labels,
                  r_src=NODE_R, r_dst=NODE_R, edge_color=edge_color_for(labels))
    for s, labels in loops.items():
        draw_self_loop(ax, POSITIONS[s], LOOP_DIR[s], labels,
                       node_radius=NODE_R, edge_color=edge_color_for(labels),
                       label_bg=COLORS['bg'], label_bg_pad=1.2)

    for name, pos in POSITIONS.items():
        ph = PHASE_OF[name]
        draw_node(
            ax, pos, PRETTY[name], radius=NODE_R,
            face=PHASE_LIGHT[ph], edge_color=PHASE_DARK[ph],
            text_size=8.5, double_ring=(name == 'H'),
        )

    ax.set_xlim(scan_x0 - 0.4, scan_x1 + 0.4)
    ax.set_ylim(mod_y0 - 0.9, POSITIONS['H'][1] + 0.7)
    ax.set_aspect('equal')
    ax.set_axis_off()

    fig.suptitle('Binary-alphabet adder Turing machine',
                 fontsize=10.5, color=COLORS['text'], y=0.965)

    save(fig, Path(__file__).with_suffix(''))


if __name__ == '__main__':
    main()
