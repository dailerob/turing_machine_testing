"""
Shared data and drawing primitives for binary-adder Turing-machine diagrams.

Each variant file in this folder picks its own layout / node style / label
style / color scheme but uses the helpers below for the heavy lifting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import COLORS  # noqa: E402


# ---------------------------------------------------------------------------
# TM data (mirrors binary_alphabet_adder.BINARY_ALPHABET_ADDER)
# ---------------------------------------------------------------------------
TRANSITIONS: list[tuple[str, str, str, str, str]] = [
    ('FIND_SEP', '0', '0', 'R', 'FIND_SEP'),
    ('FIND_SEP', '1', '1', 'R', 'FIND_SEP'),
    ('FIND_SEP', '_', '_', 'R', 'CZ0'),
    ('CZ0', '0', '0', 'R', 'CZ0'),
    ('CZ0', '1', '1', 'R', 'CZ1'),
    ('CZ0', '_', '_', 'L', 'H'),
    ('CZ1', '0', '0', 'R', 'CZ1'),
    ('CZ1', '1', '1', 'R', 'CZ1'),
    ('CZ1', '_', '_', 'L', 'DEC'),
    ('DEC', '1', '0', 'L', 'GOTO_A'),
    ('DEC', '0', '1', 'L', 'DEC'),
    ('GOTO_A', '0', '0', 'L', 'GOTO_A'),
    ('GOTO_A', '1', '1', 'L', 'GOTO_A'),
    ('GOTO_A', '_', '_', 'L', 'INC'),
    ('INC', '0', '1', 'R', 'FIND_SEP'),
    ('INC', '1', '0', 'L', 'INC'),
    ('INC', '_', '1', 'R', 'FIND_SEP'),
]

PRETTY: dict[str, str] = {
    'FIND_SEP': 'FIND‐SEP',
    'CZ0':      'CZ₀',
    'CZ1':      'CZ₁',
    'DEC':      'DEC',
    'GOTO_A':   'GOTO‐A',
    'INC':      'INC',
    'H':        'H',
}

CYCLE: list[str] = ['FIND_SEP', 'CZ0', 'CZ1', 'DEC', 'GOTO_A', 'INC']
HALT_STATE = 'H'

# Phase categorisation (used by phase-coloured variants)
PHASE_OF: dict[str, str] = {
    'FIND_SEP': 'scan',
    'CZ0':      'scan',
    'CZ1':      'scan',
    'DEC':      'modify',
    'GOTO_A':   'walk',
    'INC':      'modify',
    'H':        'halt',
}

PHASE_COLOR = {
    'scan':   COLORS['blue'],
    'modify': COLORS['red'],
    'walk':   COLORS['amber'],
    'halt':   COLORS['grey'],
}
PHASE_DARK = {
    'scan':   COLORS['blue_dark'],
    'modify': COLORS['red_dark'],
    'walk':   COLORS['amber_dark'],
    'halt':   COLORS['text'],
}
PHASE_LIGHT = {
    'scan':   COLORS['blue_light'],
    'modify': COLORS['red_light'],
    'walk':   COLORS['amber_light'],
    'halt':   COLORS['grey_light'],
}


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def is_writing(read: str, write: str) -> bool:
    return read != write


def label_default(read: str, write: str, direction: str, blank: str = '_') -> str:
    r = blank if read == '_' else read
    w = blank if write == '_' else write
    return f'{r} / {w}, {direction}'


def label_arrow(read: str, write: str, direction: str) -> str:
    r = r'\sqcup' if read == '_' else read
    w = r'\sqcup' if write == '_' else write
    arrow = r'\triangleright' if direction == 'R' else r'\triangleleft'
    return rf'${r} \to {w} \;\; {arrow}$'


def group_edges(transitions: Iterable[tuple[str, str, str, str, str]]
                ) -> dict[tuple[str, str], list[tuple[str, str, str]]]:
    g: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for src, r, w, d, dst in transitions:
        g.setdefault((src, dst), []).append((r, w, d))
    return g


def trim_endpoints(p_src: tuple[float, float], p_dst: tuple[float, float],
                   r_src: float, r_dst: float):
    sx, sy = p_src
    dx, dy = p_dst
    vx, vy = dx - sx, dy - sy
    L = np.hypot(vx, vy)
    if L == 0:
        return p_src, p_dst
    ux, uy = vx / L, vy / L
    return (sx + ux * r_src, sy + uy * r_src), (dx - ux * r_dst, dy - uy * r_dst)


def draw_node(ax, pos, text, *,
              radius: float = 0.42,
              face: str = '#FFFFFF',
              edge_color: str = COLORS['text'],
              edge_lw: float = 1.0,
              text_color: str = COLORS['text'],
              text_size: float = 8.5,
              double_ring: bool = False,
              ring_gap: float = 0.085) -> None:
    x, y = pos
    if double_ring:
        outer = Circle((x, y), radius, facecolor=face, edgecolor=edge_color,
                       linewidth=edge_lw, zorder=3)
        ax.add_patch(outer)
        inner = Circle((x, y), radius - ring_gap, facecolor='none',
                       edgecolor=edge_color, linewidth=edge_lw, zorder=4)
        ax.add_patch(inner)
    else:
        circ = Circle((x, y), radius, facecolor=face, edgecolor=edge_color,
                      linewidth=edge_lw, zorder=3)
        ax.add_patch(circ)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=text_size, color=text_color, zorder=5)


def draw_edge(ax, p_src, p_dst, labels, *,
              r_src: float, r_dst: float,
              edge_color: str = COLORS['muted'],
              edge_lw: float = 0.9,
              label_color: str = COLORS['text'],
              label_size: float = 7.0,
              label_format: Callable = label_default,
              label_offset: float = 0.18,
              label_bg: str = COLORS['bg'],
              label_bg_pad: float = 1.0,
              linestyle: str = '-',
              curvature: float = 0.0) -> None:
    a, b = trim_endpoints(p_src, p_dst, r_src, r_dst)
    arrow = FancyArrowPatch(
        a, b, arrowstyle='-|>', mutation_scale=10,
        linewidth=edge_lw, color=edge_color, linestyle=linestyle,
        connectionstyle=f'arc3,rad={curvature}', zorder=2,
    )
    ax.add_patch(arrow)

    mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
    vx, vy = b[0] - a[0], b[1] - a[1]
    L = np.hypot(vx, vy) or 1.0
    px, py = -vy / L, vx / L
    if curvature != 0.0:
        offset = 0.35 * np.sign(curvature)
        mx += px * offset
        my += py * offset
    else:
        mx += px * label_offset
        my += py * label_offset

    text = '\n'.join(label_format(r, w, d) for r, w, d in labels)
    bbox = dict(facecolor=label_bg, edgecolor='none', pad=label_bg_pad) if label_bg else None
    ax.text(mx, my, text, ha='center', va='center',
            fontsize=label_size, color=label_color,
            bbox=bbox, zorder=6)


def draw_self_loop(ax, pos, dir_vec, labels, *,
                   node_radius: float = 0.42,
                   loop_radius: float = 0.30,
                   edge_color: str = COLORS['muted'],
                   edge_lw: float = 0.9,
                   label_color: str = COLORS['text'],
                   label_size: float = 7.0,
                   label_format: Callable = label_default,
                   label_extra_offset: float = 0.18,
                   label_bg: str | None = None,
                   label_bg_pad: float = 1.0,
                   linestyle: str = '-') -> None:
    cx, cy = pos
    dx, dy = dir_vec
    norm = np.hypot(dx, dy)
    dx, dy = dx / norm, dy / norm

    lcx = cx + dx * (node_radius + loop_radius)
    lcy = cy + dy * (node_radius + loop_radius)
    base_deg = np.degrees(np.arctan2(dy, dx))
    touch_deg = base_deg + 180.0
    gap = 25.0

    arc = Arc(
        (lcx, lcy), 2 * loop_radius, 2 * loop_radius,
        angle=0.0,
        theta1=touch_deg + gap, theta2=touch_deg + 360.0 - gap,
        color=edge_color, linewidth=edge_lw, linestyle=linestyle, zorder=2,
    )
    ax.add_patch(arc)

    end_deg = touch_deg + 360.0 - gap
    end_rad = np.radians(end_deg)
    end_x = lcx + loop_radius * np.cos(end_rad)
    end_y = lcy + loop_radius * np.sin(end_rad)
    tx, ty = -np.sin(end_rad), np.cos(end_rad)
    head = FancyArrowPatch(
        (end_x - tx * 0.001, end_y - ty * 0.001), (end_x, end_y),
        arrowstyle='-|>', mutation_scale=10,
        linewidth=edge_lw, color=edge_color, zorder=2,
    )
    ax.add_patch(head)

    apex_x = cx + dx * (node_radius + 2 * loop_radius + label_extra_offset)
    apex_y = cy + dy * (node_radius + 2 * loop_radius + label_extra_offset)
    text = '\n'.join(label_format(r, w, d) for r, w, d in labels)
    bbox = dict(facecolor=label_bg, edgecolor='none', pad=label_bg_pad) if label_bg else None
    ax.text(apex_x, apex_y, text, ha='center', va='center',
            fontsize=label_size, color=label_color, bbox=bbox, zorder=5)


def split_edges(transitions):
    """Split (src,dst,labels) into self-loops and cross-edges."""
    g = group_edges(transitions)
    cross = {(s, d): lab for (s, d), lab in g.items() if s != d}
    loops = {s: lab for (s, d), lab in g.items() if s == d}
    return cross, loops


def save(fig, stem_path: Path) -> None:
    fig.savefig(stem_path.with_suffix('.pdf'))
    fig.savefig(stem_path.with_suffix('.png'))
    print(f'wrote {stem_path.with_suffix(".pdf")}')
    print(f'wrote {stem_path.with_suffix(".png")}')
