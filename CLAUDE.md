# Agent handoff

This file is a structured snapshot of where each line of work stands,
so a fresh agent (or human collaborator) can pick up without
re-discovering what was tried. **Read your folder's writeup first**,
then this for cross-cutting context. For a model-level summary of GDC
and an honest benchmark eval, see
[`paper/GDC_OVERVIEW.md`](paper/GDC_OVERVIEW.md).

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
- **Context-parroting baseline** (Zhang & Gilpin, arXiv:2505.11349,
  2025) is now run on every benchmark below. Implementations:
  `discrete_parrot.py` (HMM/algorithmic) and
  `skolr_bench/forecast/parrot_torch.py` (continuous). Used as a clean
  "what does the kernel iteration buy you over hard nearest-prefix?"
  ablation.
- **HPYLM** (Wood et al. ICML 2009 fixed-depth Sequence Memoizer
  approximation) and **PPM-D** (Howard 1993 absolute-discount n-gram
  backoff) baselines on the discrete (HMM + TM) benchmarks.
  Implementations: `discrete_hpylm.py` and `discrete_ppm.py`. PPM-D
  is the strongest non-CHMM, non-GDC method on TM state-propagation
  tasks, well ahead of HPYLM despite a similar fixed-depth backoff
  structure (PPM updates counts at every n-gram level; HPYLM's CRP
  franchise propagation only fires on table-opening events).
- **Dual-α scoring**: GDC's forecast roll-out can use a different
  $(\alpha_{\mathrm{fc}}, \theta_{\mathrm{fc}})$ than the prefix
  forward pass. Setting $\alpha_{\mathrm{fc}}{=}1$ (deterministic
  walk-forward) while $\alpha_{\mathrm{ctx}}{<}1$ is the canonical
  Table 7 HMM forecasting setting and wins on dysts (multi-IC val),
  char-LM Dedieu, and PAutomaC. Implemented as
  `forecast_many_torch_dual` (per-prime), `forecast_many_torch_dual_batched`
  (multi-α-config), and `forecast_dual_xseries` (cross-series batched)
  in `skolr_bench/forecast/gdc_torch.py`.
- Single venv lives at `/home/roberto/turing_machine_testing/.venv`
  (gitignored, shared across main checkout and worktrees).

## Per-area status

### dysts chaotic systems (`dysts_bench/`) — DONE (univariate, multi-IC val)

**Done.** 131-system Gilpin (2021) NeurIPS chaotic-systems forecasting
benchmark, univariate protocol (`pts_per_period=15, periods=12`,
30-step forecast). All numbers below on the **130-system intersection**
common to all methods (excludes 2 systems missing from released
baselines, 1 where pyEDM val-tuning failed). Our four methods (GDC,
Parrot, pyEDM, AnDA) use **multi-IC val**: 3 sliding 90-point fit
windows averaged per config (reduces Lyapunov-divergence val noise).
GDC ranks #4 of 20 methods by median sMAPE; pyEDM (also multi-IC)
jumps to #2.

| Rank | Method | Median sMAPE | Mean sMAPE |
|---:|---|---:|---:|
| 1 | NBEATS | 49.21 | 61.12 |
| **2** | **pyEDM (multi-IC val-tuned)** | **61.92** | **68.79** |
| 3 | RNN | 66.72 | 72.52 |
| **4** | **GDC (dual-α, multi-IC val-tuned)** | **67.37** | 72.79 |
| **5** | **AnDA full (multi-IC val-tuned)** | 70.46 | 73.96 |
| **6** | **Parrot (multi-IC val-tuned)** | 74.06 | 77.58 |
| 7 | RandomForest | 88.15 | 86.13 |
| ... | (13 more methods 93+ median sMAPE) | | |

**Single-IC → multi-IC val deltas on our four methods** (130-system
intersection): pyEDM −11.40 median (huge — 74% of picks change), GDC
−1.55 (dual-α grid + noise reduction), AnDA −2.18, Parrot +1.08
(roughly flat). Multi-IC val is the methodological contribution that
lifts the EDM-family methods substantially; GDC's gain comes from a
36-config dual-α grid (single-α α∈{1, 0.99} + 4 dual-α α_ctx∈{0.8,
0.9, 0.95, 0.99} with α_fc=1) that requires multi-IC val to avoid
val-overfitting.

**Caveat: by mean, GDC stays at #4** (72.79). pyEDM mean (68.79) is
below its median; GDC's mean is above its median by 5.4.

**Outright wins (130-system set)**: NBEATS 33, pyEDM 23, Parrot 19,
RNN 14, AnDA 8, GDC 8, Transformer 7, RandomForest 7, others 11.

Lorenz '63 single-system under multi-IC val: GDC and Parrot both
converge to `diff/L=16/α=1` and score 16.46 (tied), beating pyEDM
31.55, NBEATS 76.48. pyEDM Simplex with fixed E=5 (no val-tuning)
gets 14.48 — multi-IC val on the train trajectory now closes most
of the gap to this oracle reference for GDC/Parrot.

Total multi-IC val runtime: GDC ~6 min, Parrot ~12 s, pyEDM ~15 min,
AnDA ~4 min. All on a single H200. Writeup:
[`dysts_bench/RESULTS.md`](dysts_bench/RESULTS.md). Tables 10+11 of
`paper/tables.tex`.

**Open issues / next steps:**
- **Multivariate variant** (`pts_per_period=100, periods=12`):
  ~1–2 hr compute, channel-independent per dim. Adds NBEATS, NHiTS,
  DLinear, NLinear, BlockRNN, KalmanForecaster, XGB to the
  comparison.
- **Foundation-model baselines** (Chronos, Panda zero-shot): public
  checkpoints, ~4–8 hr including downloads.
- **Noise variant**: investigated and shelved. The released noise
  data files in `dysts_data` are different noise realizations than
  the released noise baselines, so an exact comparison to published
  noise numbers isn't reproducible without the original seeds.
- **pyEDM-on-diffs option in val-tuning**: investigated, results
  recorded in `pyedm_dysts.csv`. Adding diff configs to the val-
  tuning sweep slightly *worsens* the pyEDM median (72.45 → 73.32)
  because val-tuning overfits across the wider config grid. We
  report the pure-raw pyEDM result.

### M4 forecasting (`m4/`) — DONE

**Done.** All six frequencies have leakage-free, val-tuned GDC and
parrot results. Plus a series-weighted-total OWA against published M4
Naive 2.

| Method | sMAPE | MASE | OWA |
|---|---:|---:|---:|
| Top-3 winners (Smyl/M-M/Pawl, deep ensembles) | 11.4–11.8 | 1.55 | 0.833–0.849 |
| **GDC (per-series val-OWA)** | **12.65** | **1.61** | **0.887** |
| Parrot (by-freq val-OWA) | 13.58 | 1.67 | 0.938 |
| Naive 2 (reference) | 13.56 | 1.91 | 1.000 |
| Statistical baselines (Theta/Comb/ETS/ARIMA) | 12.3–12.7 | 1.66–1.70 | 0.904–0.913 |
| Parrot (per-series val-tune) | 14.80 | 1.86 | 1.032 |

GDC wins 5/6 frequencies vs parrot (parrot wins only Hourly
0.522 vs GDC 0.534). Full writeup:
[`m4/summary/M4_SUMMARY.md`](m4/summary/M4_SUMMARY.md). Tables 5+6
of `paper/tables.tex`.

Files (all leakage-free, no test-set picking):
- `m4/naive2.py` — verified reproduction of M4 Naive 2 numbers
- `m4/clean_eval.py` — per-frequency GDC val + test sweep
- `m4/parrot_eval.py` — per-frequency parrot val + test sweep
- `m4/owa_total.py`, `m4/owa_total_parrot.py` — series-weighted total OWA
- `m4/owa_select.py`, `m4/extract_published.py` — OWA aggregations + reference parsing
- Per-frequency: `m4/{hourly,daily,weekly,monthly,quarterly,yearly}/`
  has its own writeup `M4_*_RESULTS.md`

### SKOLR / Informer forecasting (`skolr_bench/forecast/`) — DONE

**Done.** Six datasets covered under both Informer and Autoformer
univariate protocols. Headline numbers in `paper/tables.tex` Tables
1 (ETTm2 + Exchange) and 4 (ILI + Traffic + ECL).

**ETTm2 (Autoformer protocol, $L=96$):**

| $T$ | GDC | Parrot | Autoformer | Best published |
|---:|---:|---:|---:|---|
| 96 | **0.074** | 0.092 | 0.065 | Autoformer 0.065 |
| 192 | **0.111** | 0.134 | 0.118 | GDC 0.111 |
| 336 | **0.150** | 0.185 | 0.154 | GDC 0.150 |
| 720 | 0.254 | 0.256 | 0.182 | LogTrans 0.160 |

**Exchange (Autoformer protocol, $L=96$):**

| $T$ | GDC | Parrot | Autoformer | Best published |
|---:|---:|---:|---:|---|
| 96 | 0.093 | **0.086** | 0.241 | Parrot 0.086 |
| 192 | 0.207 | **0.200** | 0.273 | Parrot 0.200 |
| 336 | **0.442** | 0.458 | 0.508 | GDC 0.442 |
| 720 | 1.757 | 2.063 | 0.991 | Autoformer 0.991 |

**ILI** (no published deep-learning univariate baselines): GDC and
Parrot trade wins by horizon; both beat the available statistical
baselines (Naive 1 / Seasonal Naive / Naive 2 / ARIMA / Prophet) by
30–80%, but with no learned-model competitor for these splits, this
is a comparison against weak baselines only.

**Traffic / ECL:** GDC beats Parrot by 5–15% on Traffic mid/long
horizons; Parrot wins ECL short horizons. Both beat the statistical
baselines comfortably; same caveat about no published deep-learning
baselines for these splits.

**ETTh1 (Informer protocol, $L=720$):** GDC 0.030 / 0.046 / 0.084 /
0.107 / 0.447 at $T \in \{24, 48, 168, 336, 720\}$. Parrot at L=720
catastrophically loses on short horizons (3–5× worse) — the cleanest
"kernel iteration matters" demonstration in the paper.

**Recipe insight**: GDC's `diff` variant (forecast 1-step changes,
cumsum onto last value) wins 4 of 5 horizons. State space =
`train + val` (fixed at inference time, no test-prefix leakage).

**Open issue: long-horizon val/test gap.** ETTh1 T=720 (GDC=0.447)
and Exchange T=720 (GDC=1.757) both show severe val/test divergence
where val tuning picks the wrong σ. The pathology reproduces in
parrot at the same horizons (Exchange T=720: parrot val 0.137 →
test 2.063). It's structural to long-horizon univariate forecasting
on near-random-walk data, not specific to GDC.

### SKOLR NLDS (`skolr_bench/nlds/`) — DONE

| System | GDC | Parrot | SKOLR | KooPA |
|---|---:|---:|---:|---:|
| Pendulum | 0.0003 ± 0.0002 | 0.0003 ± 0.0001 | 0.0001 | 0.0039 |
| Duffing | **0.0005 ± 0.0000** | **0.0004 ± 0.0000** | 0.0047 | 0.0365 |
| Lotka-Volterra | **0.0000 ± 0.0000** | **0.0000 ± 0.0000** | 0.0018 | 0.0178 |
| Lorenz '63 | **1.171 ± 0.103** | 1.427 ± 0.124 | 0.974 | 1.094 |

GDC wins 2 of 4 outright vs SKOLR (Duffing, LV); ties on Pendulum;
loses on chaotic Lorenz. Parrot ties or beats GDC on the three smooth
systems and loses on Lorenz (matching the dysts pattern). Full writeup:
[`skolr_bench/nlds/NLDS_RESULTS.md`](skolr_bench/nlds/NLDS_RESULTS.md).

### Algorithmic / TM benchmarks (`algorithmic_benchmarks/`) — DONE

**Done.** Nine TM tasks (parity, increment, reverse, binary_adder,
shift_left, bit_count_mod3, anbn, palindrome, subtraction) × two
variants (original, noread). Six methods compared on a leakage-free
protocol: train / val / test ranges defined once in
[`_tm_task_config.py`](algorithmic_benchmarks/_tm_task_config.py),
val drawn from a stretched range that sits between train and test
(may overlap with test on length but uses different seeds — val is
used only for hyperparameter selection). Each method's hyperparams
are val-tuned per task; test errors are reported only for the chosen
config. **Headline numbers below are at 4× the original training
budget** (n_train=1200 for most tasks, 800 for binary_adder).

Tuple errors / total predictions, **bold** = row-best:

| Task | Variant | GDC | LSTM | CHMM | Parrot | HPYLM | PPM-D | KN-3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| parity | orig | 11/506 | **9** | 10 | 10 | **9** | 12 | 13 |
| increment | orig | **0/266** | 2 | **0** | **0** | **0** | **0** | **0** |
| reverse | orig | **150/13646** | 1351 | 301 | 573 | 588 | 476 | 547 |
| binary_adder | orig | **3/72217** | 1012 | 10 | 381 | 375 | 178 | 2194 |
| shift_left | orig | **0/526** | **0** | **0** | **0** | **0** | **0** | **0** |
| bit_count_mod3 | orig | 10/526 | **2** | 12 | 14 | 13 | 13 | 12 |
| anbn | orig | **2/934** | 5 | 4 | 4 | 4 | 4 | 4 |
| palindrome | orig | 8/1574 | **3** | 6 | 9 | 9 | 8 | 9 |
| subtraction | orig | **857/33433** | 936 | 1572 | 1476 | 1608 | 1608 | 1622 |
| parity | nr | 11/506 | 12 | 10 | 10 | **9** | 12 | 13 |
| increment | nr | **0/266** | **0** | **0** | **0** | **0** | **0** | **0** |
| reverse | nr | **0/13646** | 2035 | 121 | 349 | 313 | 313 | 415 |
| binary_adder | nr | **0/72217** | 1263 | **0** | 193 | 375 | 375 | 740 |
| shift_left | nr | **0/526** | **0** | **0** | **0** | **0** | **0** | **0** |
| bit_count_mod3 | nr | **10/526** | 11 | 12 | 14 | 13 | 13 | 12 |
| anbn | nr | **0/934** | 9 | **0** | 9 | 3 | 3 | 3 |
| palindrome | nr | 13/1574 | 29 | 9 | **8** | **8** | **8** | 9 |
| subtraction | nr | **0/33433** | 2302 | 966 | 1777 | 1777 | 1558 | 1862 |

**Wins / 18 (row-best, ties counted):** GDC **13**, HPYLM 7, LSTM 6,
CHMM 6, Parrot 5, PPM-D 5, KN-3 4. (GDC's dual-α prediction step —
$\alpha_{\mathrm{fc}}{=}1$, added in the P1 retest — takes binary_adder
-original 59→3 and subtraction-original 1132→857, claiming both cells
from CHMM and LSTM respectively; it also improves anbn-original 3→2 and
palindrome-original 16→8. Dual-α is val-confirmed with 0 regressions.)

**LSTM (2-layer × 256 hidden, ~860k params, trained 60 epochs ADAM lr=1e-3
on the same reduced-tuple-ID sequences GDC sees)** — added as a learned-
model baseline. The pattern is clean: **LSTM wins or ties on short-OOD
tasks (parity, increment, shift_left, bit_count_mod3, palindrome, anbn —
OOD ratio ≤ 4×); GDC dominates on long-OOD tasks (reverse, binary_adder,
subtraction noread — OOD ratio 6×–28×, where the LSTM hidden state loses
fidelity over thousands of rollout steps).** Specifically: reverse-noread
trace lengths go from train mean 116 → test mean 684 (5.9×), and the LSTM
makes 2035 errors where GDC makes 0; binary_adder-noread goes 273 → 7224
(26.5×), LSTM 1263 vs GDC 0; subtraction-noread 60 → 1674 (28×), LSTM 2302
vs GDC 0. The LSTM training shows non-trivial seed variance (some cells
vary ~2× across runs); reported numbers are single-seed.

Headline:

- **GDC takes 13/18 cells** (with dual-α; was 11 single-α), including
  six perfect-zero noread tasks (increment, reverse, binary_adder,
  shift_left, anbn, subtraction). No other method reaches more than 5
  perfect-zero noread tasks.
- **LSTM (new column) takes 6/18 cells**, all at short-OOD ratios
  (≤4× length extrapolation): parity, increment, shift_left, the
  bit_count_mod3/palindrome/anbn family. The LSTM still beats GDC
  outright on bit_count_mod3-orig (2 vs 10) and palindrome-orig
  (3 vs 8); GDC's dual-α now edges LSTM on subtraction-orig (857 vs
  936). The LSTM matches or beats every non-deep-learning baseline at
  short OOD.
- **LSTM fails catastrophically at long OOD.** On reverse-noread
  (5.9× length OOD) the LSTM gets 2035 errors vs GDC's 0; on
  binary_adder-noread (26.5× OOD) LSTM gets 1263 vs GDC 0; on
  subtraction-noread (28× OOD) LSTM gets 2302 vs GDC 0. The LSTM
  hidden state loses fidelity over thousands of rollout steps.
- **GDC scales with training data; most baselines saturate.** Going
  from 1× to 4× training, GDC strictly improves on bit_count_mod3
  (16→10), anbn-original (5→2), anbn-noread (4→0), subtraction
  -original (1234→857), subtraction-noread (1→0). (The 4× endpoints
  are the dual-α defaults; dual-α further improves the deterministic
  state-propagation cells on top of the data-scaling gain.) PPM-D,
  HPYLM, KN-3, Parrot are essentially flat.
- **CHMM also benefits modestly from more data** (reverse-original
  329→301, reverse-noread 140→121) but doesn't close the long-OOD
  gap to GDC.
- **n-gram-style methods** (HPYLM, PPM-D, KN-3) and Parrot are
  competitive on Markov-ish tasks but degrade by 100×–1000× on long
  state-propagation tasks. HPYLM wins parity (9) under widened val.
- **The load-bearing claim**: GDC's iterated-transition mechanism
  is decisive on long-OOD deterministic state-propagation
  (subtraction-noread 0/33,433 vs LSTM 2,302; reverse-noread
  0/13,646 vs LSTM 2,035; binary_adder-noread 0/72,217 vs
  LSTM 1,263). The LSTM matches at short OOD but cannot maintain
  hidden-state fidelity across long-trace extrapolation; GDC's
  chain-position matching has no length-dependent decay.

**Sequential-training scaling** (`binary_adder_scaling.py`): with
the same GDC config (α=0.95, θ=0.05, self_loop), as few as **2
training tapes** (~5-bit operands, chain N=535 positions) suffice
to reach 0/80,333 errors on 5-10 bit binary_adder-noread test.
**K=10 training tapes** (chain N=3,096) reach 0/1,034,130 errors on
11-13 bit additions — direct evidence that the chain extracts
algorithmic structure rather than statistical regularity.

Older Dyck-1 numbers (CHMM 4497/10920 best, GDC 5118 at fixed
α=0.99) remain in `benchmark_results.csv` but were collected before
the unified protocol — Dyck-1 has no canonical OOD val_range yet.
ALERGIA was run at 1× only; at 4× the state-merging algorithm's
n²-in-strings runtime made it impractical. PAutomaC results
(`pautomac/`) and HMM forecasting (`hmm_comparison/`) characterize
GDC on stochastic finite-state regimes — both move to supplementary;
GDC is competitive with kNN-style methods there but not distinctive,
and CHMM dominates sparse-topology cells.

Tables 8 + 9 of `paper/tables.tex`. Writeups:
[`LEAKAGE_FREE_RESULTS.md`](algorithmic_benchmarks/LEAKAGE_FREE_RESULTS.md)
(canonical 1× table),
[`TUNED_GDC_RESULTS.md`](algorithmic_benchmarks/TUNED_GDC_RESULTS.md),
[`NOREAD_VARIANT_RESULTS.md`](algorithmic_benchmarks/NOREAD_VARIANT_RESULTS.md),
[`ABSORB_RESULTS.md`](algorithmic_benchmarks/ABSORB_RESULTS.md).

### HMM comparison (`hmm_comparison/`) — DONE

**Done.** GDC vs CHMM vs ALERGIA vs Parrot vs HPYLM vs PPM-D vs KN-3
vs Freq on synthetic-HMM next-token forecasting. **Canonical Table 7**
uses 4 structurally distinct regimes (cyclic, reset_chain, bimodal,
sparse topology), $n_S{=}20$ hidden states, $n_A{=}4$ symbols,
$N{=}25$ training sequences × length 50 (= 1{,}250 chars per HMM),
20 test HMMs per regime, 100 test prefixes of length 20. **Leakage-free**:
each method picks its best config per $(\text{regime}, h)$ on a disjoint
set of 20 validation HMMs (different random draws), reported on the test
HMMs. GDC uses dual-α ($\alpha_{\mathrm{forecast}}{=}1$,
$\theta_{\mathrm{forecast}}{=}0$ override on the prefix's $\alpha$) as a
candidate in the grid. **Reproducible via the committed
`hmm_comparison/gen_table7_forecasting.py`** — written 2026-06 to
replace the lost original generator (the paper numbers were faithful but
unreproducible; the reconstruction matches them within ~0.01–0.02, with
Freq matching to ~0.01; see PROTOCOL_STANDARDIZATION.md §7).

**Excess perplexity at $h{=}1$** (lower-bound 1.000; **bold** = column-best):

| Regime | GDC | CHMM | ALERGIA | Parrot | HPYLM | PPM-D | KN-3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cyclic | 1.033 | **1.027** | 1.098 | 1.044 | 1.032 | 1.034 | 1.044 |
| reset_chain | **1.026** | 1.028 | 1.074 | 1.042 | 1.029 | 1.029 | 1.035 |
| bimodal | **1.007** | 1.016 | 1.020 | 1.014 | 1.021 | 1.019 | 1.027 |
| sparse topology | 1.158 | 1.150 | 1.330 | 1.170 | **1.147** | 1.161 | 1.191 |

GDC wins **reset_chain and bimodal** at $h{=}1$ outright (1.026, 1.007);
trails CHMM at cyclic (1.033 vs 1.027); at sparse, HPYLM is column-best
(1.147) with GDC (1.158) close behind CHMM (1.150). Across $h \in
\{1,...,5\}$ GDC holds up best on reset_chain and stays competitive
everywhere except bimodal at $h{\geq}2$ where ALERGIA/CHMM hold the
column-best.

**Data-scaling (Table 13)**: same 4 regimes at $N \in \{1, 3, 5, 10, 25\}$
training sequences (leakage-free; reconstructed via
`hmm_comparison/gen_table13_scaling.py`; the $N{=}25$ column reproduces
Table 7's $h{=}1$ column exactly). GDC wins **14 / 20 column-best
cells**, concentrated at the low-data end (reset_chain: all 5 $N$);
CHMM/HPYLM retake several $N{=}25$ cells (and cyclic $N{=}1$, by a 0.014
margin — a coarse-grid near-tie at the single-sequence scale). The
bimodal $N{=}1$ column is
a genuine degenerate case (GDC/KN-3/Freq blow up to ~$10^{5}$ because
one length-50 sticky-bimodal sample visits only one cluster, so the
other cluster's symbols get ~0 probability); smoothing methods stay
bounded, and from $N{=}3$ everything is well-behaved.

**Product-HMM data-scaling (Table 12)**: a Kronecker-product HMM with
$n_S{=}27$ and $n_A{=}27$ (3 independent ternary components, each near-
deterministic with sharp emissions; reconstructed via
`hmm_comparison/gen_table12_product_hmm.py`). GDC's fixed config
($\alpha{=}0.85$, $\theta{=}0.005$, $\beta{=}0.075$) is the GDC row at
all scales and improves monotonically with data (1.477 → 1.202 → 1.183
at $h{=}1$ for $N \in \{40, 160, 640\}$), holding the $h{=}1$
column-best at $4\times$ and $16\times$. Parrot is column-best at small
scale ($h{=}1{-}2$, $1\times$); CHMM (val-picked $K$) retakes long
horizons as data grows. The fixed config is validation-best at
$16\times$ and near-best at smaller scales (which marginally prefer
$\alpha{=}0.7$).

**Principled grid reduction (applied 2026-06).** The HMM GDC sweeps were
the only oversized grids in the project. They are now reduced and the
reductions empirically validated: Table 7/13 use a **32-config** grid
($\alpha\in\{0.3,0.5,0.7,0.9\}\times\theta\in\{0,0.1\}\times\beta\in\{0,0.005\}
\times\alpha_{\mathrm{fc}}\in\{\alpha,1\}$, down from **464**) — it
reproduces the full sweep to within 0.004 excess-PP at $N{=}25$; at the
single-sequence $N{=}1$ scale it is coarser (≤0.03; one cyclic $N{=}1$
near-tie flips GDC→CHMM, giving the 14/20 tally above). Table 12 uses an
**18-config** grid (down from **462**; $\beta$ range raised to
$\{0.05,0.075,0.15\}$ for the 27-symbol stochastic emission — a
regime-justified difference), with the reported fixed-config numbers
unchanged. The original full grids remain reachable via `--full` /
`GDC_TABLE12_FULL=1`. Rationale + the universal-grid proposal are in
PROTOCOL_STANDARDIZATION.md §9–10.

Per-experiment writeups for forecasting, dimensionality, diffusion,
hidden alignment, topology, etc. in `hmm_comparison/HMM_*_EXPERIMENT.md`;
high-level summary in
[`HMM_EXPERIMENTS_SUMMARY.md`](hmm_comparison/HMM_EXPERIMENTS_SUMMARY.md).
Tables 7, 12, 13 of `paper/tables.tex`. The older 6-regime sweep
(dense_small/large, det_small/large, sparse_small/large) is in
[`comprehensive_table.md`](hmm_comparison/comprehensive_table.md) but
is no longer the canonical reference — the new 4-regime + dual-α
setup is paper Table 7.

### CHMM tests (`chmm_tests/`)

**Done.** Topology comparisons against CHMM. Requires the CHMM
reference repo cloned to `chmm_tests/naturecomm_cscg/` (instructions
in [`chmm_tests/README.md`](chmm_tests/README.md)).

### Character-LM (Dedieu et al. 2019) (`char_lm/`) — DONE

**Done.** Dedieu et al. (2019) Table 4 protocol on 8 character-level
datasets (blake-poems, shakespeare-macbeth, carroll-alice,
shakespeare-hamlet, milton-paradise, calgary-book1, melville-mobydick,
war-peace). Val-tuned bits-per-symbol (BPS); last 10% of train used
as val; each method's best-on-val config refit on full train and
scored on test. GDC uses dual-α: $\alpha_{\mathrm{ctx}} \in \{0.40,
0.45, ..., 0.70\}$, $\alpha_{\mathrm{fc}} \in \{0.95, 0.99, 1.0\}$,
$\theta{=}0$, $\beta{=}0$. Torch-GPU scorer dispatches when
$N \cdot T > 5 \times 10^9$.

| Dataset | HPYLM | PPM-D | KN-3 | Parrot | GDC (α_ctx, α_fc) | CHMM (paper) |
|---|---:|---:|---:|---:|---|---:|
| blake-poems | 1.680 | **1.663** | 1.878 | 2.485 | 1.724 (0.5, 1.0) | 1.60 |
| shakespeare-macbeth | 1.772 | **1.736** | 2.076 | 2.556 | 1.802 (0.55, 1.0) | 1.69 |
| carroll-alice | 1.791 | 1.753 | 2.199 | 2.479 | **1.718** (0.55, 1.0) | 1.54 |
| shakespeare-hamlet | 1.785 | **1.747** | 2.129 | 2.493 | 1.823 (0.55, 1.0) | 1.63 |
| milton-paradise | 2.003 | **1.960** | 2.424 | 2.598 | 2.006 (0.6, 1.0) | 1.73 |
| calgary-book1 | **1.848** | 1.985 | 2.492 | — | 1.889 (0.6, 1.0) | 1.63 |
| melville-mobydick | **1.921** | 2.015 | 2.495 | — | 1.954 (0.6, 1.0) | 1.72 |
| war-peace | **1.788** | 1.845 | 2.490 | — | 1.822 (0.65, 1.0) | 1.59 |

GDC's results:
- **Wins carroll-alice** outright (1.718 vs PPM-D 1.753).
- **Trails best non-GDC by 0.02–0.08** on the other 7 datasets;
  HPYLM and PPM-D split the remaining wins.
- **Underperforms CHMM (paper-reported)** uniformly by 0.13–0.25 —
  CHMM stays the best on every dataset; we do not have a re-run.
- **Optimal $\alpha_{\mathrm{ctx}}$ grows with dataset size**: 0.50
  (blake, 30k chars) → 0.55 (medium) → 0.60–0.65 (large, ≥350k).
  $\alpha_{\mathrm{fc}}{=}1$ is universal.

Parrot omitted on the 3 largest datasets (calgary, mobydick,
war-peace) — $O(N \cdot T)$ cost = 4–150 hours per dataset. Where it
runs, Parrot is 30–50% worse than every other method.

Files:
- `char_lm/bps_eval.py` — `score_bps_gdc_dual` (numpy) and
  `score_bps_gdc_dual_torch` (GPU); dual-α recipe.
- `char_lm/run.py` — dispatch + val sweep.
- `char_lm/RESULTS.md` — full writeup.

### PAutomaC competition (`pautomac/`) — DONE

**Done (now leakage-free).** 48-problem PAutomaC competition (Verwer
et al. 2014; probabilistic-automaton learning). The competition ships
only train + test(+solution) — no validation split — so config
selection is made **leakage-free**: each problem's train sequences are
split 80/20, the 7 GDC configs are fit on the 80% and ranked by
**held-out negative log-likelihood** on the 20% (no true distribution
needed); the lowest-NLL config is refit on full train and scored on
test. The 7 configs: 2 single-α at $\alpha \in \{0.95, 0.50\}$ plus 5
dual-α at $\alpha_{\mathrm{fc}}{=}0.9999$, $\alpha_{\mathrm{ctx}} \in
\{0.30, 0.50, 0.70, 0.85, 0.95\}$. ($\alpha_{\mathrm{fc}}{=}0.9999$
instead of 1.0 is a numerical safety floor — 9 problems gave
$\log(0){=}-\infty$ at $\alpha_{\mathrm{fc}}{=}1$.) Run by
`pautomac/run_leakage_free.py` (resumable). Compared against ALERGIA+
and MDI (Verwer & Hammerschmidt 2022 FlexFringe Table 2), KN-3,
Parrot.

Summary statistics on **gap above the entropy floor** (gap = score
$-$ floor; 48 problems):

| Method | Median gap | Mean gap | Max gap | Wins |
|---|---:|---:|---:|---:|
| **ALERGIA+** | **0.12** | **0.63** | **6.04** | **28** |
| MDI | 0.37 | 1.09 | 7.64 | 11 |
| GDC val-tuned (leakage-free) | 0.73 | 1.85 | 12.54 | 0 |
| GDC fixed (α_ctx=0.85, α_fc=0.9999) | 0.80 | 1.44 | 10.22 | 9 |
| KN-3 | 4.86 | 13.27 | 132.49 | 0 |
| Parrot | 50.15 | 104.10 | 1006.07 | 0 |

GDC ranks **#3 of 6** by median gap (0.73), behind ALERGIA+ (0.12,
28 wins) and MDI (0.37, 11 wins) — both FlexFringe state-merging PDFA
learners specifically designed for the PAutomaC distribution. Under
the leakage-free protocol GDC val-tuned takes **0** outright wins:
held-out NLL on a 20% split is a noisy proxy for test perplexity at
this train size, so it occasionally mis-picks. It attains a slightly
lower *median* gap than the single fixed config (0.73 vs 0.80) but a
higher *mean* (1.85 vs 1.44), and the fixed $\alpha_{\mathrm{ctx}}{=}
0.85$ config actually takes 9 outright wins (small-floor problems) —
most of the value of config selection is already captured by that one
config. KN-3 and Parrot are far behind. Table 14 of
`paper/tables.tex` has the full per-problem perplexity matrix.

**Note**: the older [`pautomac/FULL_SWEEP_RESULTS.md`](pautomac/FULL_SWEEP_RESULTS.md)
reports "GDC wins 43/48" — that comparison was vs CHMM, not vs
ALERGIA+/MDI. The current paper Table 14 uses the FlexFringe
competition baselines as the comparator and GDC sits behind both.

## Aggregate cell-by-cell parrot ablation

Across 62 evaluation cells from 5 benchmark suites (SKOLR/Informer
univariate × all horizons; SKOLR NLDS × 4 systems; M4 × 6 frequencies;
HMM forecasting × 18 cells; algorithmic × 9 task-variant pairs):

| Outcome | Count | Fraction |
|---|---:|---:|
| GDC wins | 26 | 42% |
| Parrot wins | 13 | 21% |
| Tied (within ±2%) | 23 | 37% |

The wins map to a clean interpretable pattern: **GDC wins on
algorithmic state propagation, smooth-with-trends time series, and
chaotic dynamics; Parrot wins on near-random-walk data, pure-seasonality,
and OOD-syntactic tasks where the mode of similar prefixes is
informative.** See
[`paper/GDC_OVERVIEW.md`](paper/GDC_OVERVIEW.md) for the prose
version.

**Note**: the 62-cell count is for the original 5-suite ablation
and does not include the more recently added suites — char-LM (8
datasets; GDC vs Parrot is 5W/0L on the cells where Parrot is run,
Parrot is omitted on the 3 largest), PAutomaC (48 problems; GDC
beats Parrot in all 48 by orders of magnitude), and dysts (133
systems; GDC 67.37 vs Parrot 74.06 by median sMAPE). On the dysts
130-system set per-system breakdown, GDC outperforms Parrot on the
substantial majority of systems but Parrot is non-trivial — see
`dysts_bench/RESULTS.md` for the per-system tally.

## Recent reference papers (PDFs not committed)

- SKOLR (arXiv:2506.14113) — at the user's
  `OneDrive/Documents/Research/GDC lit review/koopman operator time series.pdf`
- Informer (arXiv:2012.07436) — re-downloadable; we used it to find
  the exact reproducibility setup (Appendix E).
- Autoformer (arXiv:2106.13008) — used for the ETTm2 + Exchange
  univariate baseline numbers.
- dysts (arXiv:2110.05266, Gilpin 2021) — used for the chaotic-systems
  benchmark in `dysts_bench/`.
- Context Parroting (arXiv:2505.11349, Zhang & Gilpin 2025) — used as
  the parrot baseline across every benchmark.
- Sequence Memoizer (Wood et al. ICML 2009) — used as the HPYLM
  fixed-depth baseline on HMM and TM benchmarks.
- PPM-D (Howard 1993, "The design and analysis of efficient lossless
  data compression systems") — used as the PPM-D fixed-depth baseline
  on HMM and TM benchmarks.

## Things that didn't pan out

- **Multivariate (channel-coupled) GDC for ETT**: skipped; paper-style
  channel-independence is standard and our univariate-target results
  match the published univariate columns.
- **Larger config grids on M4**: caused val-overfitting (tried 85
  configs on monthly; val-pick lost to a single fixed config). Final
  M4 sweep uses 5–9 candidate configs per frequency.
- **NumPy-batched GDC**: at large N (~12k state space), NumPy memory
  bandwidth dominated and batching was actually *slower* than per-prime.
  Numba serial + GPU parallel are the actual wins. See
  [`skolr_bench/forecast/gdc_batched.py`](skolr_bench/forecast/gdc_batched.py)
  for the (deprecated) numpy batched implementation kept for reference.
- **Exact-reproduction of dysts noise baselines**: the released noise
  data files don't correspond to the same noise realization the noise
  baselines were scored against; without the original numpy seed an
  apples-to-apples comparison isn't reproducible.

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
- **Cross-benchmark protocol + sweep audit**:
  [`paper/PROTOCOL_STANDARDIZATION.md`](paper/PROTOCOL_STANDARDIZATION.md)
  tabulates the train/val/test split, selection metric, and exact GDC
  sweep grid for all 8 benchmarks (with file:line cites), proposes one
  canonical sweep per emission regime (continuous-forecast vs
  discrete-next-token), and judges which per-benchmark differences are
  justified. Key open items: (1) dual-α is only swept in 4 of 8
  benchmarks (absent in M4, SKOLR-forecast, SKOLR-NLDS, TM) — the top
  retest priority since adding it is leakage-free-safe; (2) **paper HMM
  Tables 7/12/13 had no committed generator** — the originals were absent
  from every branch (verified by git pickaxe). **All three are now
  reconstructed, leakage-free, and reproducible**: Table 7 →
  `gen_table7_forecasting.py`, Table 13 → `gen_table13_scaling.py`,
  Table 12 → `gen_table12_product_hmm.py` (each with a matching
  `build_table*_latex.py`). The reconstructed numbers replaced the
  paper's (Table 7 matched the old within ~0.01–0.02). See §7 of
  PROTOCOL_STANDARDIZATION.md for the evidence and status.
