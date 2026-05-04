"""
Shared style for paper figures (NeurIPS-targeted).

Three primary hues — red, blue, amber — each available in light / medium /
dark tones for accents. Plus neutral text/grid/background colors. Use the
``COLORS`` dict for explicit color choices and ``PALETTE`` for cycling
through the three primaries when plotting series.

Usage
-----
    from style import COLORS, PALETTE, apply_style
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, y, color=COLORS['blue'])
"""

from __future__ import annotations

import matplotlib as mpl


COLORS: dict[str, str] = {
    'red':         '#B0604F',
    'red_light':   '#E8C0B5',
    'red_dark':    '#7A3325',

    'blue':        '#4A7BA8',
    'blue_light':  '#B5D0E5',
    'blue_dark':   '#284C72',

    'amber':       '#D49A4A',
    'amber_light': '#F0D9A8',
    'amber_dark':  '#8A622A',

    'text':        '#2A2A2A',
    'muted':       '#7A7A7A',
    'grey':        '#9A9A9A',
    'grey_light':  '#BFBFBF',
    'grid':        '#E2E2E2',
    'panel':       '#F7F5F2',
    'bg':          '#FFFFFF',
}

PALETTE: list[str] = [COLORS['blue'], COLORS['red'], COLORS['amber']]
PALETTE_DARK: list[str] = [COLORS['blue_dark'], COLORS['red_dark'], COLORS['amber_dark']]
PALETTE_LIGHT: list[str] = [COLORS['blue_light'], COLORS['red_light'], COLORS['amber_light']]


def apply_style() -> None:
    """Apply the shared matplotlib rcParams. Idempotent — safe to call repeatedly."""
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'STIX Two Text', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.titleweight': 'normal',
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,

        'axes.linewidth': 0.7,
        'axes.edgecolor': COLORS['text'],
        'axes.labelcolor': COLORS['text'],
        'axes.titlecolor': COLORS['text'],
        'axes.spines.top': True,
        'axes.spines.right': True,

        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',

        'lines.linewidth': 1.1,
        'lines.markersize': 3.5,

        'grid.color': COLORS['grid'],
        'grid.linewidth': 0.5,

        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'savefig.transparent': False,
        'figure.dpi': 110,
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['bg'],

        'axes.prop_cycle': mpl.cycler(color=PALETTE),
    })
