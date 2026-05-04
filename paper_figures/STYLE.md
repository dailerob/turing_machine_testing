# Paper figure style guide

Shared palette and matplotlib defaults for all figures in `paper_figures/`.
Targeted at NeurIPS-style camera-ready figures (serif fonts, tight bounding
boxes, 300 dpi PDF + PNG).

## Palette

Three primary hues — **red**, **blue**, **amber** — each with a `light`
and `dark` variant for accents and emphasis. Plus neutral text / grid /
panel-background colors.

| name          | hex       | use                                         |
|---------------|-----------|---------------------------------------------|
| `blue`        | `#4A7BA8` | primary series 1 (e.g. training)            |
| `blue_light`  | `#B5D0E5` | filled regions, light accents               |
| `blue_dark`   | `#284C72` | strong accents, marker edges                |
| `red`         | `#B0604F` | primary series 2 (e.g. context)             |
| `red_light`   | `#E8C0B5` | filled regions                              |
| `red_dark`    | `#7A3325` | strong accents                              |
| `amber`       | `#D49A4A` | primary series 3 (e.g. forecast)            |
| `amber_light` | `#F0D9A8` | filled regions                              |
| `amber_dark`  | `#8A622A` | strong accents                              |
| `text`        | `#2A2A2A` | axis labels, titles, tick text              |
| `muted`       | `#7A7A7A` | secondary text                              |
| `grey`        | `#9A9A9A` | medium grey (de-emphasized lines)           |
| `grey_light`  | `#BFBFBF` | light grey (background reference series)    |
| `grid`        | `#E2E2E2` | grid lines                                  |
| `panel`       | `#F7F5F2` | optional soft panel background              |
| `bg`          | `#FFFFFF` | figure background                           |

## Conventions

- **Cycling order**: `PALETTE = [blue, red, amber]`. When showing
  multiple series, follow this order.
- **Light / dark accents**: use `_light` for filled regions (likelihood
  bars, error bands) paired with the medium tone for outlines, and
  `_dark` for marker edges or emphasis lines.
- **Markers**: connected scatter (line + markers) is the default for
  small sequence plots. Marker face = medium tone, marker edge =
  matching `_dark` tone, edge width 0.4 pt.
- **Spines**: all four spines on, 0.7 pt, color `text`. Ticks are
  inward-facing.
- **Fonts**: serif (Times New Roman, falling back to STIX / DejaVu).
  9 pt body, 8 pt ticks.

## Usage

```python
from style import COLORS, PALETTE, apply_style
apply_style()

fig, ax = plt.subplots(figsize=(3.5, 2.0))
ax.plot(x, y, color=COLORS['blue'], marker='o',
        markerfacecolor=COLORS['blue'],
        markeredgecolor=COLORS['blue_dark'],
        markeredgewidth=0.4)
fig.savefig('out.pdf')
```

Always call `apply_style()` at the top of a figure script. Save both
`.pdf` (for the paper) and `.png` (for previews) to the same directory
as the script.
