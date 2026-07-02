"""Aggregate per-frequency parrot OWA + sMAPE + MASE into a comparison table
against the GDC numbers in paper/tables.tex and published M4 baselines."""
from __future__ import annotations
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))

FREQS = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']

# GDC results from paper/tables.tex
GDC_OWA = {
    'Yearly':    (0.814, 0.811),  # (per-series, by-freq)
    'Quarterly': (0.908, 0.910),
    'Monthly':   (0.949, 0.957),
    'Weekly':    (0.759, 0.800),
    'Daily':     (0.985, 0.987),
    'Hourly':    (0.543, 0.534),
}

# Published M4 references (paper/tables.tex)
M4_TOP3 = {  # series-weighted-total OWAs from published M4 supplementary
    'Yearly':    {'Smyl': 0.778, 'M-M': 0.799, 'Pawl': 0.820},
    'Quarterly': {'Smyl': 0.847, 'M-M': 0.847, 'Pawl': 0.855},
    'Monthly':   {'Smyl': 0.836, 'M-M': 0.858, 'Pawl': 0.867},
    'Weekly':    {'Smyl': 0.851, 'M-M': 0.796, 'Pawl': 0.766},
    'Daily':     {'Smyl': 1.046, 'M-M': 1.019, 'Pawl': 0.806},
    'Hourly':    {'Smyl': 0.440, 'M-M': 0.484, 'Pawl': 0.444},
}

# Best statistical benchmark per frequency (paper/tables.tex)
M4_STAT_BEST = {
    'Yearly':    ('Comb',  0.867),
    'Quarterly': ('Comb',  0.890),
    'Monthly':   ('ARIMA', 0.903),
    'Weekly':    ('Damped',0.917),
    'Daily':     ('Comb',  0.978),
    'Hourly':    ('ARIMA', 0.577),
}

# n series per frequency
NSER = {'Yearly': 23000, 'Quarterly': 24000, 'Monthly': 48000,
        'Weekly': 359, 'Daily': 4227, 'Hourly': 414}


def parse_summary(freq):
    log_path = os.path.join(HERE, freq.lower(), 'parrot_eval.log')
    if not os.path.exists(log_path):
        return None
    text = open(log_path).read()
    out = {}
    m = re.search(r"\(1\) per-series val-tuned: sMAPE=([\d.]+), MASE=([\d.]+), OWA=([\d.]+)", text)
    if m:
        out['ps_smape'], out['ps_mase'], out['ps_owa'] = map(float, m.groups())
    m = re.search(r"\(2\) global pick by val-sMAPE \[(.*?)\]: OWA = ([\d.]+)", text)
    if m:
        out['gs_name'], out['gs_owa'] = m.group(1), float(m.group(2))
    m = re.search(r"\(2'\) global pick by val-MASE \[(.*?)\]: OWA = ([\d.]+)", text)
    if m:
        out['gm_name'], out['gm_owa'] = m.group(1), float(m.group(2))
    return out


def main():
    rows = []
    for freq in FREQS:
        s = parse_summary(freq)
        if s is None:
            print(f"  [skip] {freq}: parrot_eval.log not found")
            continue
        rows.append((freq, s))

    print(f"\n{'Freq':<11s} {'n':>6s}  {'GDC ps':>6s} {'GDC gf':>6s}  "
          f"{'Parrot ps':>9s} {'Parrot gf':>9s}  {'best stat':>14s}  {'best top3':>14s}")
    print("-" * 100)
    parrot_wins = 0; gdc_wins = 0; ties = 0
    for freq, s in rows:
        gdc_ps, gdc_gf = GDC_OWA[freq]
        p_ps = s.get('ps_owa', float('nan'))
        p_gs = s.get('gs_owa', float('nan'))
        p_gm = s.get('gm_owa', float('nan'))
        # Best parrot global (which OWA pick is lower)
        best_parrot_global = min(p_gs, p_gm)
        # Compare per-series
        delta = (p_ps - gdc_ps) / gdc_ps * 100
        if delta < -2: tag='parrot wins'; parrot_wins += 1
        elif delta > 2: tag='GDC wins'; gdc_wins += 1
        else: tag='tied'; ties += 1
        stat_name, stat_owa = M4_STAT_BEST[freq]
        best_top3 = min(M4_TOP3[freq].items(), key=lambda x: x[1])
        print(f"{freq:<11s} {NSER[freq]:>6d}  "
              f"{gdc_ps:.3f}  {gdc_gf:.3f}    "
              f"{p_ps:.3f}     {best_parrot_global:.3f}    "
              f"{stat_name:<10s}{stat_owa:.3f}  {best_top3[0]:<10s}{best_top3[1]:.3f}    [{tag}]")

    print()
    print(f"Per-series OWA across {len(rows)} freqs: parrot wins {parrot_wins}, "
          f"GDC wins {gdc_wins}, tied {ties}")


if __name__ == "__main__":
    main()
