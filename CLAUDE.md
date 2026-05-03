# Agent handoff

This file is a structured snapshot of where each line of work stands,
so a fresh agent (or human collaborator) can pick up without
re-discovering what was tried. **Read your folder's writeup first**,
then this for cross-cutting context.

## Quick orientation

- The repo's main subject is the **Generative Dense Chain (GDC)**, a
  non-parametric prefix-memorising sequence model. Discrete in
  `generative_dense_chain.py`, continuous (Gaussian emission) in
  `generative_dense_chain_timeseries.py`.
- Major recent addition: **`terminal_behavior='absorb'` mode** in
  GDC-TS (and discrete GDC). Used by all forecasting work to avoid
  the "diffuse smearing" of the original GDC at the trained
  manifold's terminus. See
  [`algorithmic_benchmarks/ABSORB_RESULTS.md`](algorithmic_benchmarks/ABSORB_RESULTS.md)
  for a clean derivation of when absorb vs diffuse matter.
- Two speed kernels mirror `forecast_gdc_style` exactly:
  - **Numba**: `skolr_bench/forecast/gdc_numba.py` — CPU parallel,
    used by HMM-comparison and SKOLR-NLDS work.
  - **PyTorch**: `skolr_bench/forecast/gdc_torch.py` — GPU, fp32
    matches numba to ~1e-6, fp64 to machine precision; ~3-9× faster
    than the numba version on RTX 5090.

## Per-area status

### M4 forecasting (`m4/`)

**Done.** All six frequencies have leakage-free, val-tuned GDC
results plus a series-weighted-total OWA against published M4 Naive
2. Headline:

| protocol | total OWA |
|---|---:|
| GDC per-series val-tune | 0.887 |
| GDC global val-OWA pick | 0.888 |
| Naive 2 reference | 1.000 |
| Top-3 M4 winners (Smyl/M-M/Pawl) | 0.833–0.849 |
| Statistical baselines (Theta/Comb/ETS/ARIMA) | 0.904–0.913 |

GDC sits cleanly **between top-3 winners and statistical benchmarks**
on the official M4 series-weighted total. Full writeup:
[`m4/summary/M4_SUMMARY.md`](m4/summary/M4_SUMMARY.md).

Files (all leakage-free, no test-set picking):
- `m4/naive2.py` — verified reproduction of M4 Naive 2 numbers
- `m4/clean_eval.py` — per-frequency val + test sweep
- `m4/owa_total.py`, `m4/owa_select.py` — series-weighted total OWA
- `m4/extract_published.py` — parses M4 supplementary doc for
  reference numbers
- Per-frequency: `m4/{hourly,daily,weekly,monthly,quarterly,yearly}/`
  has its own writeup `M4_*_RESULTS.md`

### Algorithmic benchmarks (`algorithmic_benchmarks/`)

**Done; mostly stable.** Recent addition: absorb-mode comparison.
Key finding: on argmax-greedy-with-conditional algorithmic prediction,
absorb mode gives **identical predictions to diffuse mode** (proven
in [`ABSORB_RESULTS.md`](algorithmic_benchmarks/ABSORB_RESULTS.md)).
Headline tasks: parity, increment, reverse, binary adder, dyck1.
Per-task tuned configs are in
[`TUNED_GDC_RESULTS.md`](algorithmic_benchmarks/TUNED_GDC_RESULTS.md).

### HMM comparison (`hmm_comparison/`)

**Done.** GDC vs CHMM vs ALERGIA on HMM-generated forecasting tasks
across regimes (dense/sparse, small/large, deterministic/stochastic).
Comprehensive table in
[`comprehensive_table.md`](hmm_comparison/comprehensive_table.md);
high-level summary in
[`HMM_EXPERIMENTS_SUMMARY.md`](hmm_comparison/HMM_EXPERIMENTS_SUMMARY.md).
Per-experiment writeups for forecasting, dimensionality, diffusion,
hidden alignment, topology, etc.

### CHMM tests (`chmm_tests/`)

**Done.** Topology comparisons against CHMM. Requires the CHMM
reference repo cloned to `chmm_tests/naturecomm_cscg/` (instructions
in [`chmm_tests/README.md`](chmm_tests/README.md)).

### PAutomaC (`pautomac/`)

**Done.** Earlier work; intact since prior session.

### SKOLR forecasting (`skolr_bench/forecast/`) — IN PROGRESS

**Last working state**: GDC val-tuned on ETTh1 (univariate, OT) at
all 5 Informer horizons:

| T | GDC | ARIMA pub | Informer pub | GDC vs Informer |
|---:|---:|---:|---:|---:|
| 24 | 0.030 | 0.108 | 0.098 | **3.0× better** |
| 48 | 0.046 | 0.175 | 0.158 | **3.4× better** |
| 168 | 0.084 | 0.396 | 0.183 | **2.2× better** |
| 336 | 0.107 | 0.468 | 0.222 | **2.0× better** |
| 720 | 0.447 | 0.659 | 0.269 | **1.7× WORSE** |

Pipeline validated: Informer on the same data reproduces published
~0.10 MSE at T=24 to within seed-noise.

**Open issues:**
1. **T=720 collapse**: val pick chose `raw σ=0.10 α=1.0` (val MSE
   0.168), test MSE was 0.447 — 3× val/test gap. Hypotheses: diff
   recipe accumulates errors over 720 steps; val and test distributions
   diverge at long horizons; raw matching with N=11520 state space
   isn't enough at this horizon. Worth exploring: longer L (we've
   been using L=720; try L=1440), ensembling over top-K val picks,
   diff with damped α<1.
2. **Run sweep on remaining datasets**: ETTh2, ETTm1/2, Weather, ECL,
   Traffic, ILI. Universal sweep script not yet built — need to
   adapt `gdc_etth1_full_sweep.py` to take a dataset arg.
3. **ARIMA / Prophet baselines**: implementations exist
   (`arima_baseline.py`, `prophet_baseline.py`), validated to
   reproduce within ~3× of published numbers on ETTh1 T=24
   (i.e. our auto_arima is much stronger than the paper's published
   ARIMA — paper's baselines were under-tuned). Not yet run on full
   horizon × dataset grid.
4. **Informer reproduction**: clone needed, see
   [`skolr_bench/forecast/README.md`](skolr_bench/forecast/README.md).
   Already validated for ETTh1 T=24.

**Recipe insight**: GDC's `diff` variant (forecast 1-step changes,
cumsum onto last value) wins 4 of 5 horizons. State space =
`train + val` (fixed at inference time, no test-prefix leakage).
Lookback L=720 to match Informer's universal lookback.

### SKOLR NLDS (`skolr_bench/nlds/`) — DONE

| system | GDC | SKOLR | KooPA |
|---|---:|---:|---:|
| Pendulum | 0.0003 ± 0.0002 | 0.0001 | 0.0039 |
| Duffing | **0.0005 ± 0.0000** | 0.0047 | 0.0365 |
| Lotka-Volterra | **0.0000 ± 0.0000** | 0.0018 | 0.0178 |
| Lorenz '63 | 1.171 ± 0.103 | 0.974 | 1.094 |

GDC wins 2 of 4 outright (Duffing, Lotka-Volterra), ties on Pendulum,
loses on chaotic Lorenz. Full writeup:
[`skolr_bench/nlds/NLDS_RESULTS.md`](skolr_bench/nlds/NLDS_RESULTS.md).

## Recent reference papers (PDFs not committed)

- SKOLR (arXiv:2506.14113) — at the user's
  `OneDrive/Documents/Research/GDC lit review/koopman operator time series.pdf`
- Informer (arXiv:2012.07436) — re-downloadable; we used it to find
  the exact reproducibility setup (Appendix E)

## Things that didn't pan out

- **Multivariate (channel-coupled) GDC for ETT**: skipped; paper-style
  channel-independence is standard and our univariate-target results
  match the Informer paper's univariate column.
- **Larger config grids on M4**: caused val-overfitting (tried 85
  configs on monthly; val-pick lost to a single fixed config). Final
  M4 sweep uses 5-9 candidate configs per frequency.
- **NumPy-batched GDC**: at large N (~12k state space), NumPy memory
  bandwidth dominated and batching was actually *slower* than per-prime.
  Numba serial + GPU parallel are the actual wins. See
  [`skolr_bench/forecast/gdc_batched.py`](skolr_bench/forecast/gdc_batched.py)
  for the (deprecated) numpy batched implementation kept for reference.

## Style conventions

- Leakage-free everywhere: val data is used only for hyperparameter
  selection; test data is never seen during model selection.
- All forecasting metrics on standardized data (StandardScaler fit
  on train only).
- Where we report a single config (not val-picked), it's flagged as
  an oracle / leaky comparison and qualified accordingly.
- Per-folder writeups (`*_RESULTS.md`) are the source of truth;
  result CSVs are not committed (regenerate from scripts; headline
  numbers are in the writeups).
