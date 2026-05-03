# M4 forecasting competition

GDC results across all six M4 frequencies (Hourly, Daily, Weekly,
Monthly, Quarterly, Yearly) with leakage-free OWA-vs-Naive-2
evaluation.

**Headline writeup**: [`summary/M4_SUMMARY.md`](summary/M4_SUMMARY.md)
— series-weighted OWA across the full 100,000 series puts our GDC at
~0.89, between the M4 winners (0.83–0.85) and the official
statistical benchmarks (Theta/Comb/ETS at 0.91–0.92).

## Folder layout

```
m4/
├── README.md                   # this file
├── data/                       # CSVs go here (NOT committed; see below)
├── data_loader.py              # generic loader for all six frequencies
├── naive2.py                   # M4 Naive 2 reproduction (verified)
├── clean_eval.py               # leakage-free per-frequency val + test sweep
├── owa_total.py                # series-weighted total OWA across freqs
├── owa_select.py               # variant: pick configs by val OWA jointly
├── extract_published.py        # parses M4 supplementary doc for benchmark numbers
├── summary/
│   └── M4_SUMMARY.md           # cross-frequency summary writeup
└── {hourly,daily,weekly,monthly,quarterly,yearly}/
    ├── M4_*_RESULTS.md         # per-frequency writeup
    ├── plot_series.py          # sample-trajectory plot
    ├── v0_baselines.py         # naive baselines
    ├── v1..vN_*.py             # progressive recipe iterations
    └── clean_eval_summary.md   # per-frequency leakage-free results
```

## Download data

The M4 dataset is **not committed** (large; ~250 MB total). Download
from the official M4 competition repo:

```bash
mkdir -p m4/data
cd m4/data
for freq in Yearly Quarterly Monthly Weekly Daily Hourly; do
  curl -O "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/${freq}-train.csv"
  curl -O "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/${freq}-test.csv"
done
# M4-info.csv has metadata (frequency labels, etc.)
curl -O https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/M4-info.csv
```

Source: https://github.com/Mcompetitions/M4-methods/tree/master/Dataset

## Reproduce

```bash
# Verify Naive 2 reproduces published M4 numbers (sanity check)
python m4/naive2.py

# Per-frequency leakage-free val + test sweep
python m4/clean_eval.py Hourly Daily Weekly Monthly Quarterly Yearly

# Series-weighted total OWA against published Naive 2
python m4/owa_total.py
python m4/owa_select.py     # alternative: jointly val-OWA pick

# Re-parse benchmark numbers from the M4 supplementary doc (optional;
# requires the docx at the path expected by the script)
python m4/extract_published.py
```

Per-frequency writeups in each subfolder explain the recipe
progression, picks per channel/series, and how the GDC config was
chosen.
