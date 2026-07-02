# GDC on the dysts chaotic-systems benchmark

## TL;DR

GDC-TS evaluated on the **dysts** 131-system chaotic-systems forecasting
benchmark of [Gilpin (2021, NeurIPS)](https://arxiv.org/abs/2110.05266)
using the canonical univariate protocol (`pts_per_period=15`,
`periods=12`, 30-step forecast horizon). All results below aggregated
on the **130-system intersection** common to every method (excludes
`AtmosphericRegime` and `LidDrivenCavityFlow` not in the released
baselines, and `PiecewiseCircuit` where pyEDM val-tuning failed).

Our four val-tuned methods (GDC, Parrot, pyEDM, AnDA) use a
**multi-IC val** protocol: per system, val sMAPE is averaged across
3 sliding 90-point fit windows within the train trajectory instead of
scored on a single 150-point fit at the end. Multi-IC val noticeably
reduces the noise that Lyapunov-divergent val targets otherwise inject
into per-system config picking. Published baselines (NBEATS, RNN, etc.)
use their own protocols as released; the comparison's caveat is that
the val protocol differs for our four methods only.

**GDC dual-α ranks #4 of 20 methods by median sMAPE** (67.37), behind
NBEATS (49.21), pyEDM (61.92, multi-IC), and RNN (66.72). pyEDM is the
biggest beneficiary of multi-IC val and the new strongest non-deep-
learning baseline; GDC's iterated-kernel-propagation contribution
still keeps it ahead of AnDA (70.46) and Parrot (74.06).

| Rank | Method | Median sMAPE | Mean sMAPE |
|---:|---|---:|---:|
| 1 | NBEATS | **49.21** | **61.12** |
| **2** | **pyEDM (ours, multi-IC val-tuned)** | **61.92** | **68.79** |
| 3 | RNN | 66.72 | 72.52 |
| **4** | **GDC (ours, dual-α, multi-IC val-tuned)** | **67.37** | **72.79** |
| **5** | **AnDA full (ours, multi-IC val-tuned)** | **70.46** | 73.96 |
| **6** | **Parrot (ours, multi-IC val-tuned)** | **74.06** | 77.58 |
| 7 | RandomForest | 88.15 | 86.13 |
| 8 | Transformer | 93.39 | 91.97 |
| 9 | LinearRegression | 104.76 | 99.42 |
| 10 | ARIMA | 110.36 | 99.86 |
| 11 | AutoARIMA | 113.79 | 107.84 |
| 12 | FFT | 115.91 | 105.93 |
| 13 | Theta | 120.09 | 108.45 |
| 14 | FourTheta | 120.09 | 108.45 |
| 15 | NaiveSeasonal | 125.64 | 113.96 |
| 16 | NaiveDrift | 127.56 | 115.11 |
| 17 | TCN | 130.74 | 117.20 |
| 18 | Prophet | 130.46 | 118.45 |
| 19 | ExponentialSmoothing | 141.67 | 126.25 |
| 20 | NaiveMean | 153.25 | 129.57 |

**Single-IC → multi-IC val deltas** for our four methods on the same
130-system intersection:

| Method | Old median | New median | Δ median | Old mean | New mean | Δ mean |
|---|---:|---:|---:|---:|---:|---:|
| GDC | 68.91 | **67.37** | **−1.55** | 74.10 | 72.79 | **−1.31** |
| Parrot | 72.98 | 74.06 | +1.08 | 75.09 | 77.58 | +2.48 |
| pyEDM | 73.32 | **61.92** | **−11.40** | 72.78 | 68.79 | **−3.99** |
| AnDA | 72.64 | **70.46** | **−2.18** | 72.29 | 73.96 | +1.67 |

Multi-IC val helps 3 of 4 of our methods; pyEDM is the biggest mover
by far, with 74% of its per-system config picks changing and a fat
tail of large gains on systems where the single-IC val was selecting
configs that blew up on the test trajectory.

**Honest caveats**:

- **Different val protocol for our methods vs published baselines.**
  Multi-IC val is our methodology, not part of Gilpin's released
  benchmark. The published baselines (NBEATS through ExponentialSmoothing
  in the table) use their original protocols as released. The fair
  comparison is "our methods, with our val protocol, vs released
  baselines as released" — same as in the original Gilpin (2021)
  comparison, but our methods are now stronger.
- **Mean ranking re-orders the top 6**: by mean sMAPE, pyEDM is #2
  (68.79), NBEATS still #1 (61.12), RNN at #3 (72.52), GDC #4 (72.79).
  Median is the standard aggregation in this benchmark family because
  per-system sMAPE is heavy-tailed.
- **Outright wins on the 130-system set**: NBEATS 33, pyEDM 23,
  Parrot 19, RNN 14, AnDA 8, GDC 8, Transformer 7, RandomForest 7,
  others 11. GDC has fewer outright wins than under the single-IC
  protocol (10 → 8); the wins it loses go primarily to pyEDM, which
  now wins systems like Lorenz '63 outright with `S-Map E=3 θ=10`.

## Lorenz '63 spot-check

The single-system result on Lorenz '63 reverses substantially under
multi-IC val: both GDC and Parrot now find `diff`-recipe configs that
match within rounding of each other and clearly beat pyEDM.

| Method | sMAPE (multi-IC) | sMAPE (old single-IC) |
|---|---:|---:|
| **GDC (val-pick: diff / σ=0.05 / α=1.0)** | **16.46** | 41.41 |
| **Parrot (val-pick: diff / L=16 / k=1)** | **16.46** | 96.55 |
| pyEDM (val-pick: S-Map / E=3 / θ=10) | 31.55 | 31.55 |
| AnDA (val-pick: local_linear / E=5 / k=10) | 68.81 | 159.16 |
| NBEATS | 76.48 | 76.48 |
| RandomForest | 96.42 | 96.42 |
| AutoARIMA | 115.92 | 115.92 |
| Prophet | 119.29 | 119.29 |
| ARIMA | 119.89 | 119.89 |
| Transformer | 120.75 | 120.75 |
| NaiveSeasonal | 122.32 | 122.32 |
| FourTheta | 123.52 | 123.52 |
| Theta | 123.59 | 123.59 |
| NaiveDrift | 128.83 | 128.83 |
| ExponentialSmoothing | 130.24 | 130.24 |
| TCN | 130.29 | 130.29 |
| LinearRegression | 135.70 | 135.70 |
| RNN | 136.02 | 136.02 |
| NaiveMean | 179.86 | 179.86 |
| FFT | 181.24 | 181.24 |
| _Reference (no val-tuning, fixed config):_ | | |
| pyEDM Simplex / E=5 (fixed, oracle pick) | 14.48 | 14.48 |

Multi-IC val on Lorenz brings both GDC and Parrot within 14% of the
oracle `E=5` Simplex reference (sMAPE 14.48). With the canonical
single-IC val, Parrot picked the wrong `(L=4, k=5)` config and scored
96.55; GDC picked the wrong σ=0.10 / α=0.99 and scored 41.41. Under
multi-IC val both converge on `diff / L=16 / α=1` (effectively
"deterministic 1-NN walk in the differenced 16-point delay
embedding"), which is the right inductive bias for the Lorenz attractor
at this sampling rate.

## Protocol

For each of 131 chaotic dynamical systems (the canonical dysts
univariate set, drawn from the first coordinate of each system):

- **Train trajectory**: 180 points (= 15 pts/period × 12 periods).
- **Test trajectory**: 180 points, *independent initial conditions*
  (not a continuation of train).
- **Multi-IC val tuning** (our protocol): for each candidate config,
  fit on `train[s-90:s]` and forecast 30 steps for $s\in\{90, 120, 150\}$;
  score sMAPE against `train[s:s+29]` for each window; pick the config
  with the lowest mean val sMAPE across the 3 windows. All three
  windows use the same fit_len=90 to hold the model's effective
  state-space size constant across val replicates.
- **Test eval**: apply val-picked config to `test[:150]` → forecast
  30 steps → score first 29 vs `test[150:179]`.
- **Metric**: sMAPE, then median across systems (heavy-tailed).

The "first 29 of 30" oddity is the protocol convention from the
original `dysts_data` benchmarks — the released JSONs store length-30
predictions but a length-29 truth, so we match.

**Why multi-IC val.** Chaotic systems have positive Lyapunov exponents.
Train and test trajectories share the same attractor but evolve from
independent ICs, so by step 30 their per-coordinate values have
diverged. With a single val window, val sMAPE on the train trajectory
is a noisy 29-point estimate of a config's generalization to the test
trajectory. Spearman rank correlation between val and test sMAPEs is
~0.15 on a 12-config GDC grid and collapses to ~0 on a 36-config grid;
i.e., adding more configs to val-pick from is dominated by val noise.
Averaging val sMAPE across 3 fixed-fit-len windows lifts Spearman ρ
back toward 0.15 and lets larger config grids actually help. The
benefit is most pronounced for methods whose configs have very
different generalization profiles (pyEDM E/θ, AnDA regression/E/k);
methods whose configs are robustly similar (Parrot L/k) gain little.

## GDC config grid (36 candidates)

```
recipe ∈ {raw, diff}
σ_frac ∈ {0.05, 0.10, 0.25}
α: 12 (α_ctx, α_fc) pairs:
   single-α (α_ctx == α_fc): (1.0, 1.0), (0.99, 0.99)
   dual-α   (α_fc = 1.0):    (0.0, 1.0), (0.5, 1.0), (0.7, 1.0),
                              (0.95, 1.0), (0.975, 1.0), (0.99, 1.0),
                              (0.999, 1.0)
                              (plus α_ctx ∈ {0.8, 0.9}: see below)
fixed: terminal_behavior='absorb', initial_dist='uniform'
       (the canonical forecasting choice across our work)
```

Effective α grid: 2 single-α + 4 dual-α (`α_ctx ∈ {0.8, 0.9, 0.95, 0.99}`,
α_fc = 1.0) × 2 recipes × 3 σ = **36 configs**.

Pick distribution across 133 systems (top 8):

| n | pick |
|---:|---|
| 16 | `raw_s0.05_ac1.0_afc1.0` |
| 13 | `raw_s0.05_ac0.99_afc1.0` (dual-α) |
| 12 | `raw_s0.1_ac1.0_afc1.0` |
| 9 | `raw_s0.25_ac1.0_afc1.0` |
| 7 | `raw_s0.05_ac0.8_afc1.0` (dual-α) |
| 6 | `raw_s0.25_ac0.99_afc1.0` (dual-α) |
| 5 | `diff_s0.05_ac0.99_afc0.99` |
| 5 | `raw_s0.1_ac0.99_afc0.99` |

Roughly half the picks are dual-α; the most popular dual variants are
`α_ctx=0.99` and `α_ctx=0.8` (both with `α_fc=1.0`). The `α_ctx=0.0`
extreme (essentially uniform-prior context, deterministic walk-forward)
is rarely the val-picked option after multi-IC val averaging.

## Parrot config grid (16 variants)

```
mode ∈ {raw, diff}
L    ∈ {2, 4, 8, 16}     # lookback length for nearest-prefix match
K    ∈ {1, 5}            # top-K average
```

Parrot serves as the "what does pure prefix matching get you" baseline.
Its multi-IC val median (74.06) is essentially unchanged from the
single-IC val (72.98) at aggregate, but on Lorenz specifically it drops
from 96.55 → 16.46 because multi-IC val lets it pick `diff/L=16/k=1`
instead of an overfit `raw/L=4/k=5`.

## Compute

| Phase | Time on H200 |
|---|---|
| Per-system GDC val-tune (36 configs × 3 windows) + test | ~0.7 s |
| Per-system Parrot val-tune (16 variants × 3 windows) + test | ~0.1 s |
| Per-system pyEDM val-tune (18 configs × 3 windows) + test | ~6.7 s |
| Per-system AnDA val-tune (36 configs × 3 windows) + test | ~1.5 s |
| **Total: 133 systems, all four methods** | ~20 min |

## Files

```
dysts_bench/
├── RESULTS.md                                # this file
├── run_parrot_gdc.py                         # legacy single-IC val driver (Parrot + GDC)
├── run_gdc_dual.py                           # GDC dual-α with multi-IC val
├── run_parrot_multiIC.py                     # Parrot with multi-IC val
├── run_pyedm.py                              # legacy single-IC pyEDM driver
├── run_pyedm_multiIC.py                      # pyEDM with multi-IC val
├── run_anda.py                               # legacy single-IC AnDA driver
├── run_anda_multiIC.py                       # AnDA with multi-IC val
├── data/                                     # symlinked from main repo
│   ├── train.json                            # 180-pt training trajectories per system
│   ├── test.json                             # 180-pt test trajectories per system
│   └── released_baselines.json               # Gilpin 2021 published predictions/metrics
└── results/
    ├── parrot_gdc_dysts.csv                  # legacy single-IC val (Parrot + GDC)
    ├── pyedm_dysts.csv                       # legacy single-IC val (pyEDM)
    ├── anda_dysts_full.csv                   # legacy single-IC val (AnDA)
    ├── gdc_dual_dysts_multiIC3_fitlen90.csv  # GDC dual-α + multi-IC val
    ├── parrot_dysts_multiIC3_fitlen90.csv    # Parrot + multi-IC val
    ├── pyedm_dysts_multiIC3_fitlen90.csv     # pyEDM + multi-IC val
    └── anda_dysts_full_multiIC3_fitlen90.csv # AnDA + multi-IC val
```

## Reproduce

```bash
# Pull data files (one-time, ~3 MB)
mkdir -p dysts_bench/data && cd dysts_bench/data
curl -L "https://github.com/williamgilpin/dysts_data/raw/main/dysts_data/data/train_univariate__pts_per_period_15__periods_12.json.gz" \
  | gunzip > train.json
curl -L "https://github.com/williamgilpin/dysts_data/raw/main/dysts_data/data/test_univariate__pts_per_period_15__periods_12.json.gz" \
  | gunzip > test.json
curl -L "https://raw.githubusercontent.com/williamgilpin/dysts_data/main/dysts_data/benchmarks/results/results_test_univariate__pts_per_period_15__periods_12.json" \
  -o released_baselines.json
cd -

# Run multi-IC val sweeps for our four methods
python dysts_bench/run_gdc_dual.py       # GDC dual-α, ~6 min
python dysts_bench/run_parrot_multiIC.py # Parrot, ~12 s
python dysts_bench/run_pyedm_multiIC.py  # pyEDM, ~15 min
python dysts_bench/run_anda_multiIC.py   # AnDA, ~4 min
```

## Caveats and notes

1. **Independent train/test trajectories**. The dysts protocol does
   *not* continue the train trajectory in test — both are independent
   draws of the same system from different initial conditions. Models
   are val-tuned on train, then applied to test using its own first
   150 points as the fitting context.
2. **133 vs 131 systems**: our methods score 133 systems (the count in
   the train+test JSONs), but the released baselines from Gilpin
   (2021) only report 131. Median/mean aggregations in the leaderboard
   table are on the 130-system intersection.
3. **Val fit_len=90 ≠ test fit_len=150**. Multi-IC val uses 90-point
   fit windows for noise reduction; test eval uses 150-point fit
   windows. The val_smape and test_smape distributions are at different
   absolute scales as a result (val median ~65, test median ~67 for
   GDC) but the within-system *ranking* of configs is what drives the
   pick, and that ranking is more stable than at fit_len=150 because
   it's averaged across 3 replicates instead of computed on a single
   fit.
4. **sMAPE on chaotic systems is heavy-tailed**. The mean sMAPE values
   (rightmost columns above) are dominated by a few systems where every
   method fails (the Lyapunov horizon is much shorter than 29 steps).
   Median is the robust statistic that matches the Gilpin paper's
   distribution-plot reporting.
5. **No noise variant**: the dysts release also includes Brownian-noise
   variants (`*_noise.json.gz`) for measuring robustness, which we
   skipped here. The released noise data is a different random
   realisation from the released noise baselines, so an apples-to-
   apples noise comparison isn't reproducible without the original
   seeds.

## Next steps

- **Multivariate variant** (`pts_per_period=100, periods=12`,
  channel-independent forecasting per dim). Estimated ~1–2 hours
  compute on H200.
- **Foundation-model baselines**: Chronos and Panda zero-shot. Code
  is publicly available; ~4–8 hours including downloads.
- **VPT (Valid Prediction Time)**: the 2024/25 zero-shot protocol of
  Zhang & Gilpin uses Lyapunov-time-normalised VPT instead of fixed
  sMAPE. Adding this would put GDC on the same axis as the foundation-
  model literature.
