# turing_machine_testing

Research repo for the **Generative Dense Chain (GDC)** model family — a
non-parametric prefix-memorising sequence/time-series model — and its
evaluation across a wide range of benchmarks (algorithmic Turing
machines, HMM comparisons, M4 forecasting competition, SKOLR Koopman
benchmarks, dysts chaotic systems).

A two-page summary of what GDC is, how it works, and an honest
benchmark eval lives in [`paper/GDC_OVERVIEW.md`](paper/GDC_OVERVIEW.md).

## Headline results

| Benchmark | Headline | Writeup |
|---|---|---|
| **dysts (131 chaotic systems, univariate)** | GDC #3 of 20 by median (sMAPE 68.9); ahead of pyEDM/AnDA/Parrot at 72.5–73.0; #6 by mean | [dysts_bench/RESULTS.md](dysts_bench/RESULTS.md) |
| **M4 (100k series)** | GDC OWA 0.887 — between top-3 winners (0.83–0.85) and statistical benchmarks (0.91–0.92) | [m4/summary/M4_SUMMARY.md](m4/summary/M4_SUMMARY.md) |
| **SKOLR / Informer (univariate)** | GDC matches or beats Autoformer at most horizons on ETTm2, Exchange; beats every available statistical baseline on ILI/Traffic/ECL (no published deep-learning baselines for these splits) | [paper/tables.tex](paper/tables.tex) Tables 1, 4 |
| **SKOLR NLDS (Pendulum/Duffing/LV/Lorenz)** | GDC wins 2 of 4 outright (Duffing, LV); ties Pendulum; loses Lorenz to SKOLR | [skolr_bench/nlds/NLDS_RESULTS.md](skolr_bench/nlds/NLDS_RESULTS.md) |
| **HMM forecasting (6 regimes × 3 N)** | GDC at the entropy floor on dense regimes; beaten by CHMM on sparse | [hmm_comparison/HMM_EXPERIMENTS_SUMMARY.md](hmm_comparison/HMM_EXPERIMENTS_SUMMARY.md) |
| **Algorithmic / Turing machine** | GDC achieves zero errors on parity / increment / reverse / binary_adder under no-read trace; lone losses on dyck1 OOD | [algorithmic_benchmarks/TUNED_GDC_RESULTS.md](algorithmic_benchmarks/TUNED_GDC_RESULTS.md) |

A val-tuned **context-parroting baseline** (Zhang & Gilpin 2025) is also
run on every benchmark above for a clean "what does the kernel iteration
buy you over hard nearest-prefix?" ablation. Across 62 evaluation cells:
GDC 26 wins, Parrot 13 wins, 23 ties.

## What's in here

- **GDC core** — `generative_dense_chain.py` (discrete) and
  `generative_dense_chain_timeseries.py` (continuous, Gaussian
  emissions, with `terminal_behavior='absorb'` mode for honest
  finite-horizon forecasting).
- **Speed kernels** — `skolr_bench/forecast/gdc_numba.py` (Numba CPU
  parallel) and `skolr_bench/forecast/gdc_torch.py` (PyTorch GPU,
  fp32/fp64). Both produce numerically identical outputs to the
  reference NumPy implementation.
- **Discrete parrot core** — `discrete_parrot.py` (HMM/algorithmic
  benchmarks) plus `skolr_bench/forecast/parrot_torch.py` (continuous,
  GPU). Used as the canonical "context-parroting" baseline (Zhang &
  Gilpin 2025) on every benchmark.
- **Benchmark suites** — each in its own folder with a writeup:
  - `dysts_bench/` — Gilpin (2021) 131-system chaotic benchmark with
    full leaderboard. Writeup:
    [`dysts_bench/RESULTS.md`](dysts_bench/RESULTS.md).
  - `m4/` — M4 forecasting competition (all six frequencies),
    OWA-vs-Naive 2 evaluation. Writeup:
    [`m4/summary/M4_SUMMARY.md`](m4/summary/M4_SUMMARY.md).
  - `skolr_bench/forecast/` — SKOLR/Informer time-series benchmarks
    (ETTh1, ETTm2, Exchange, ECL, Traffic, ILI; multi-horizon under
    both Informer and Autoformer protocols). Headline numbers in
    [`paper/tables.tex`](paper/tables.tex) Tables 1, 4.
  - `skolr_bench/nlds/` — SKOLR nonlinear dynamical systems
    (Pendulum, Duffing, Lotka-Volterra, Lorenz '63). Writeup:
    [`skolr_bench/nlds/NLDS_RESULTS.md`](skolr_bench/nlds/NLDS_RESULTS.md).
  - `algorithmic_benchmarks/` — Turing-machine-style algorithmic
    sequence prediction (parity, increment, reverse, binary adder,
    Dyck-1). Writeups: [`TUNED_GDC_RESULTS.md`](algorithmic_benchmarks/TUNED_GDC_RESULTS.md),
    [`ABSORB_RESULTS.md`](algorithmic_benchmarks/ABSORB_RESULTS.md), and others.
  - `hmm_comparison/` — comparison of GDC against CHMM and ALERGIA on
    HMM-generated forecasting tasks across regimes. Writeup:
    [`HMM_EXPERIMENTS_SUMMARY.md`](hmm_comparison/HMM_EXPERIMENTS_SUMMARY.md).
  - `chmm_tests/` — CHMM-specific topology comparisons. Writeup:
    [`CHMM_TOPOLOGY_COMPARISON.md`](chmm_tests/CHMM_TOPOLOGY_COMPARISON.md).
  - `pautomac/` — PAutomaC competition tasks. Writeup:
    [`COMPETITION_COMPARISON.md`](pautomac/COMPETITION_COMPARISON.md).
- **Paper tables** — [`paper/tables.tex`](paper/tables.tex) — 11
  ready-to-drop NeurIPS-format tables covering ETTm2, Exchange, ILI,
  Traffic, ECL, M4 (overall + per-frequency), HMM forecasting, TM
  standard + no-read variants, and dysts (leaderboard + Lorenz).

## Quickstart

### Setup environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU acceleration of the GDC kernel (used in
`skolr_bench/forecast/`), install a CUDA wheel of PyTorch separately;
the default `pip install torch` from `requirements.txt` is CPU-only:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu129
```

(use the CUDA version matching your driver — `cu128`, `cu129`, etc.)

### Datasets

No datasets are committed. Each benchmark folder's README explains
where to download the data:

- **M4** — `m4/data/`: download from [Mcompetitions/M4-methods](https://github.com/Mcompetitions/M4-methods/tree/master/Dataset).
  See [`m4/README.md`](m4/README.md) for exact files.
- **SKOLR / Informer** (ETT, ECL, Weather, ILI, Exchange, Traffic) —
  `skolr_bench/data_original/`: download from the [Tsinghua Cloud
  bundle](https://cloud.tsinghua.edu.cn/f/b8f4a78a39874ac9893e/?dl=1)
  (used by Koopa/SKOLR/TSLib). For the canonical original-source ETT
  data, see [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset).
- **NLDS** — generated locally:
  `python skolr_bench/nlds/nlds_generate.py` (~20s).
- **dysts** — small 3 MB pull from `williamgilpin/dysts_data`; see
  [`dysts_bench/RESULTS.md`](dysts_bench/RESULTS.md) for the curl
  commands.
- **HMM benchmark / Algorithmic / PAutomaC** — synthetic, generated
  by the experiment scripts.

### Run something

```bash
# dysts 131-system univariate (GDC + Parrot, full leaderboard, ~36s on H200)
python dysts_bench/run_parrot_gdc.py

# M4 OWA evaluation (fast — uses precomputed leakage-free splits)
python m4/clean_eval.py Hourly Daily Weekly Monthly Quarterly Yearly
python m4/owa_total.py
python m4/naive2.py    # verifies our Naive 2 reproduces the published numbers

# SKOLR NLDS (4 systems × 5 seeds)
python skolr_bench/nlds/nlds_generate.py
python skolr_bench/nlds/nlds_eval.py

# SKOLR univariate forecasting (ETTm2 / Exchange / ILI / Traffic / ECL)
python skolr_bench/forecast/gdc_ettm2_autoformer.py
python skolr_bench/forecast/parrot_valtuned_sweep.py

# Algorithmic / TM sweep (parity, increment, reverse, binary_adder, dyck1)
python algorithmic_benchmarks/run_benchmarks.py
python algorithmic_benchmarks/parrot_eval.py
```

## Repo map

```
turing_machine_testing/
├── generative_dense_chain.py              # GDC (discrete)
├── generative_dense_chain_timeseries.py   # GDC-TS (continuous + absorb mode)
├── discrete_parrot.py                     # discrete top-K parrot baseline
├── chat_model.py / pacman_*.py / etc.     # earlier toy work
├── paper/                                 # ready-to-drop NeurIPS tables
│   ├── tables.tex                         # 11 tables across all benchmarks
│   └── GDC_OVERVIEW.md                    # 1–2 page summary of GDC + eval
├── dysts_bench/                           # Gilpin (2021) chaotic-systems benchmark
│   ├── RESULTS.md                         # writeup with leaderboard + Lorenz
│   ├── run_parrot_gdc.py                  # full sweep driver
│   └── data/                              # symlinked from main repo
├── algorithmic_benchmarks/                # Turing-machine-style prediction tasks
├── hmm_comparison/                        # GDC vs CHMM vs ALERGIA on HMM forecasting
├── chmm_tests/                            # CHMM-specific topology experiments
├── pautomac/                              # PAutomaC competition
├── m4/                                    # M4 forecasting competition
│   ├── data/                              # (re-downloadable; not committed)
│   ├── summary/M4_SUMMARY.md              # cross-frequency summary + tables
│   ├── naive2.py                          # Naive 2 reproduction
│   ├── clean_eval.py                      # leakage-free GDC per-frequency
│   ├── parrot_eval.py                     # leakage-free parrot per-frequency
│   ├── owa_total.py / owa_total_parrot.py # series-weighted OWA aggregations
│   └── {hourly,daily,...}/                # per-frequency scripts + writeups
└── skolr_bench/                           # SKOLR (ICML 2025) benchmarks
    ├── data_original/                     # (re-downloadable; not committed)
    ├── data_provider/                     # Koopa-derived data loaders
    ├── nlds/                              # nonlinear dynamical systems
    │   ├── NLDS_RESULTS.md                # writeup
    │   ├── nlds_generate.py               # generate trajectories (5 seeds × 4 systems)
    │   ├── nlds_eval.py                   # GDC sweep + sliding-window test
    │   └── parrot_eval.py                 # autoregressive in-context parrot
    └── forecast/                          # forecasting benchmarks
        ├── informer_loaders.py            # Informer / Autoformer-protocol loaders
        ├── gdc_numba.py                   # Numba CPU-parallel GDC kernel
        ├── gdc_torch.py                   # PyTorch GPU GDC kernel (fp32/fp64)
        ├── parrot_torch.py                # PyTorch GPU parrot kernel
        ├── parrot_valtuned_sweep.py       # leakage-free parrot sweep across datasets
        ├── arima_baseline.py              # auto_arima reference
        ├── prophet_baseline.py            # Prophet reference
        ├── gdc_etth1_full_sweep.py        # ETTh1 sweep (Informer protocol)
        ├── gdc_ettm2_autoformer.py        # ETTm2 sweep (Autoformer protocol)
        ├── gdc_exchange_autoformer.py     # Exchange sweep
        └── (other per-dataset sweeps)
```

## Cited / cloned external work

- **SKOLR paper**: Zhang et al., *SKOLR: Structured Koopman Operator
  Linear RNN for Time-Series Forecasting* (ICML 2025), arXiv:2506.14113.
- **Informer paper**: Zhou et al., *Informer: Beyond Efficient
  Transformer for Long Sequence Time-Series Forecasting* (AAAI 2021),
  arXiv:2012.07436. Used for the original ETT benchmark and ARIMA /
  Prophet / LSTMa / DeepAR baseline numbers.
- **Autoformer paper**: Wu et al., *Autoformer: Decomposition
  Transformers with Auto-Correlation* (NeurIPS 2021), arXiv:2106.13008.
  Used for the univariate ETTm2 + Exchange long-horizon comparisons.
- **dysts paper**: Gilpin, *Chaos as an Interpretable Benchmark for
  Forecasting and Data-Driven Modelling* (NeurIPS 2021),
  arXiv:2110.05266. The 131-system chaotic-systems benchmark; baseline
  prediction JSONs lifted from
  [williamgilpin/dysts_data](https://github.com/williamgilpin/dysts_data).
- **Context Parroting paper**: Zhang & Gilpin, *Context Parroting: A
  Simple but Tough-to-Beat Baseline for Foundation Models in
  Scientific Machine Learning* (arXiv:2505.11349, 2025). Our parrot
  baselines implement this protocol.
- **CHMM (Naturecomms paper)**: George et al., *Clone-structured graph
  representations enable flexible learning and vicarious evaluation
  of cognitive maps* (Nature Communications 2021). Clone instructions
  in [`chmm_tests/README.md`](chmm_tests/README.md).
- **M4 competition**: Makridakis, Spiliotis, Assimakopoulos (2020).
  Data + reference Naive 2 from
  [Mcompetitions/M4-methods](https://github.com/Mcompetitions/M4-methods).

## For agents picking up the work

See [`CLAUDE.md`](CLAUDE.md) for a structured handoff: what's been
tried, current state of each benchmark, known gaps, and recommended
next steps. See [`paper/GDC_OVERVIEW.md`](paper/GDC_OVERVIEW.md) for a
1–2 page summary of the model itself.
