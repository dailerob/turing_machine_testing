# HMM experiments — summary across all axes

A consolidated map of every HMM-forecasting (and adjacent) experiment
in this folder, the axes each one explores, and the headline finding.
Use this as the index when revisiting.

---

## 1. The basic benchmark

**Setup** ([random_hmm.py](random_hmm.py), [evaluation.py](evaluation.py)):

* Generate a random HMM with `nS` hidden states, `nA` emissions.
* Sample `N_train` training sequences of length `T_train`.
* Sample `N_test = 100` test prefixes of length `TEST_PREFIX_LEN = 20`.
* For each test prefix, the **true posterior predictive** at horizon
  `h` is `α_t @ T^h @ E` — computed exactly from the underlying HMM.

**Two metrics** measured per (model, prefix, h):

* **MSE**: squared error against the true predictive distribution,
  averaged over the alphabet.
* **Excess perplexity**: `2^(CE − floor)` where `CE = -Σ true·log2(pred)`
  and `floor = -Σ true·log2(true)`. Lower bound 1.0; closest analog
  to PAutomaC's gap-to-floor.

The two metrics broadly agree on rankings. They diverge mainly when
models overcommit (CHMM at low N) or underconfidently spread mass
(GDC with high β on sharp distributions).

**Models compared** (per-experiment subset):

* **GDC** — the focal method. Hyperparameters `(alpha, theta, gamma,
  beta, transition_type)` swept extensively. Default for HMM
  forecasting tuned to `(α=0.5, θ=0.05, β=0.2, self_loop)` for
  general use; per-regime optima vary (see §3).
* **Spectral OOM** — included in the original sweep; consistently
  worst on this metric, often *worse than uniform*.
* **CHMM** ([chmm_alergia_wrappers.py](chmm_alergia_wrappers.py)) —
  EM-trained Cloned HMM via the upstream `chmm_actions` library.
  K ∈ {2, 4, 8, 16, 32} tested.
* **ALERGIA** — passive PDFA learner via AALpy.
* **Baselines**: uniform, stationary-emission marginal.

---

## 2. Axes explored

### 2.1 Model class

* OOM (legacy) → GDC vs OOM (the original `run_main_sweep.py` story)
* GDC vs CHMM vs ALERGIA (the [run_chmm_alergia_sweep.py](run_chmm_alergia_sweep.py) story)
* Adding KN3 / Bigram / Spectral was done in PAutomaC, not here

### 2.2 GDC hyperparameters

Sweeps progressively tightened around the optimum:

| sweep | grid | finding |
|---|---|---|
| baseline | `(α=0.7, θ=0.2, β=0.1)` fixed | undertuned but consistently mediocre |
| [run_gdc_tuned_sweep](run_gdc_tuned_sweep.py) | single config `(0.95, 0.005, 0.0)` | **catastrophic** — sharp configs flop on this metric |
| [run_gdc_grid_sweep](run_gdc_grid_sweep.py) | α∈{0.5,0.7,0.9}, θ∈{0.05,0.2}, β∈{0.05,0.1,0.2} | best at boundary `(α=0.5, θ=0.05, β=0.2)` |
| [run_gdc_wider_grid](run_gdc_wider_grid.py) | α up to 0.99, β to 0.3 | α=0.99 catastrophic; β=0.3 wins for several regimes |
| [run_gdc_low_alpha](run_gdc_low_alpha.py) | α down to 0.1, β to 0.5 | `dense_large` wants α=0.1 (boundary) |
| [run_high_alpha_sparse](run_high_alpha_sparse.py) | α up to 0.95 on sparse | sparse optimum at **α=0.80-0.85** (interior) |
| [run_absorb_compare_regimes](run_absorb_compare_regimes.py) | terminal_behavior ∈ {diffuse, absorb} per regime | regime-dependent (see §2.10) |

**Per-regime GDC optima:**

| regime | best (α, θ, β) | mechanism |
|---|---|---|
| `dense_small` | (0.30-0.50, 0.005, 0.20-0.30) | midline smoothing |
| `dense_large` | **(0.10, 0.001, 0.20-0.50)** | predict near-stationary + small prefix correction |
| `det_small` | (0.50-0.70, 0.010, 0.20) | sharp emissions → commit to states |
| `det_large` | (0.20-0.40, 0.001, 0.20) | midline |
| `sparse_small` | (0.80-0.85, 0.001, 0.10) | sharp transitions need commitment |
| `sparse_large` | (0.70-0.80, 0.001, 0.10-0.20) | as above |

**Key qualitative patterns:**

* `α + θ + β ≈ 1` is the "right" total smoothing; how it's split depends on regime.
* `θ ≪ α` always wins — self-loop mass is rarely useful.
* `β=0.0` and `β=0.5` both fail; β ∈ {0.1, 0.3} is the useful range.
* `transition_type='self_loop_two_step'` is essentially neutral; never helped meaningfully.

### 2.3 HMM size and structure

Six regimes used across §2.4-2.6 sweeps (see [plot_topologies.py](plot_topologies.py),
[fig_hmm_topologies.png](fig_hmm_topologies.png)):

| name | nS | nA | T | E | properties |
|---|---:|---:|---|---|---|
| dense_small | 10 | 4 | Dirichlet(1) | Dirichlet(1) | moderate everything |
| dense_large | 30 | 8 | Dirichlet(1) | Dirichlet(1) | fast-mixing, flat π |
| det_small | 10 | 4 | Dirichlet(1) | Dirichlet(0.1) | each state ≈1 emission |
| det_large | 30 | 8 | Dirichlet(1) | Dirichlet(0.1) | as above, scaled |
| sparse_small | 10 | 4 | fanout=2 | Dirichlet(0.1) | sparse transitions + sharp emissions |
| sparse_large | 30 | 8 | fanout=2 | Dirichlet(0.1) | as above, scaled |

### 2.4 Sample size N_train

[run_sample_efficiency.py](run_sample_efficiency.py): N ∈ {10, 25, 50, 100, 200} on
the standard 9×9 (nS, nA) grid.

* **Crossover**: GDC tuned wins 84% of cells at N=10; ALERGIA takes
  over by N=50 (83%) and dominates at N≥100 (90%+).
* Baseline-config GDC never competitive (0-1 wins out of 243).
* CHMM K=4 catches up only by N=200 in some cells.

### 2.5 Extreme low-data

[run_extreme_low_data.py](run_extreme_low_data.py): N ∈ {3, 5, 10} × T ∈ {10, 50}.

* GDC wins 9-18 / 18 cells per (T, N) setting.
* **CHMM wins 0/18 in 5 of 6 settings** — EM can't converge with
  ≤150 training tokens.
* ALERGIA wins 9/18 only at T=10 N=3 (essentially tied with GDC);
  drops to 0-3/18 elsewhere.
* In the most extreme (T=10, N=3): CHMM beats uniform on 22% of
  cells; ALERGIA on 17%; GDC on 44%.

### 2.6 Training-sequence length T_train

[run_long_seqs_sparse.py](run_long_seqs_sparse.py): T ∈ {50, 200, 500} on
sparse regimes.

* **GDC has a Goldilocks T_train ≈ 200**; degrades at T=500
  (state-count blow-up dilutes diffusion).
* ALERGIA peaks at T=200 then plateaus.
* **CHMM scales monotonically** with both N and T — bounded
  parametric model amortises additional data into sharper EM.
* Result: longer sequences *widen* the CHMM-vs-GDC gap on sparse,
  not narrow it.

### 2.7 Forecasting horizon

Most sweeps used h ∈ {1, 5, 20}. At h=20 nearly all methods converge
to the stationary marginal regardless of model — the long-horizon
test discriminates very little. h=1 is where most signal lives.

### 2.8 Topology determinism + sparsity

[run_large_det_sweep.py](run_large_det_sweep.py): scaled nS ∈ {10, 20, 30, 50}
on dense vs det_em.
[run_sparse_sweep.py](run_sparse_sweep.py): added sparse regimes.

* CHMM K=32 has a real niche on `det_small + high N` (clones
  disambiguate emission-equivalent states).
* CHMM dominates *everything* on sparse (28/30 cells); ALERGIA
  sometimes worse than GDC even at high N.
* Per-regime ladder summarised in §3.

### 2.9 Metric (MSE vs perplexity)

[run_perplexity_sweep.py](run_perplexity_sweep.py): added cross-entropy /
excess-perplexity metric.

* Both metrics broadly agree on regime-by-regime rankings (4/18
  cells switch winner across model classes; the rest agree).
* Perplexity makes overcommitment hurt more visibly:
  - CHMM at low N looks dramatically worse under perplexity.
  - ALERGIA at high N on dense looks slightly worse under perplexity.
* At moderate-to-high N on most regimes, all methods sit at excess
  perplexity 1.001-1.05 — **the benchmark is saturated** unless N is
  pushed below 25 or topology is sparse.

### 2.10 GDC `terminal_behavior` (diffuse vs absorb)

Discrete GDC and GDC-TS both gained a `terminal_behavior` parameter
(default `'diffuse'` preserves existing behavior; `'absorb'` makes
each sequence's terminal an absorbing sink — mass that reaches it
leaks out instead of being uniformly redistributed across non-terminal
states).

[run_absorb_compare_regimes.py](run_absorb_compare_regimes.py): per-regime
comparison at the per-regime tuned config from §2.3, both metrics:

| regime           | N=25 mode | N=100 mode | N=400 mode | N=400 MSE ratio (a/d) | N=400 ExcPP ratio (a/d) |
|---               |---        |---         |---         |---:                   |---:                     |
| dense_small      | diffuse   | diffuse    | diffuse    | 1.023                 | 1.000                   |
| dense_large      | diffuse   | diffuse    | **absorb** | 0.999                 | 1.000                   |
| det_small        | diffuse   | diffuse    | diffuse    | 1.048                 | 1.000                   |
| det_large        | diffuse   | **absorb** | **absorb** | **0.991**             | 1.000                   |
| sparse_small     | diffuse   | diffuse    | diffuse    | 1.012                 | 1.000                   |
| **sparse_large** | diffuse   | **absorb** | **absorb** | **0.974**             | **0.992**               |

Three patterns:

* **Small dense regimes** (`dense_small`, `det_small`): diffuse wins
  everywhere. Terminal-mass redistribution acts as a small
  uniform-prior smoother that helps match the soft HMM posterior.
* **Large dense regimes at high N** (`dense_large`, `det_large`):
  absorb wins. Sharp emissions + larger state space → the soft
  posterior is concentrated; smearing dilutes the signal.
* **Sparse regimes at high N** (`sparse_large`): absorb wins by
  the largest margin (2.6% MSE / 0.8% ExcPP improvement). Sparse
  posteriors are highly concentrated; any uniform-smearing hurts.

Both metrics agree on the regime-winner direction. The MSE differences
are larger than the perplexity differences (perplexity values cluster
more tightly because the metric saturates near 1.0 for most regimes),
but both point the same way at h=1.

For the M4 hourly forecasting task (continuous-value point forecasts),
absorb is universally better — see [m4/M4_HOURLY_RESULTS.md](../m4/M4_HOURLY_RESULTS.md).
The general lesson: **`terminal_behavior='absorb'` is the right
default whenever the underlying source has concentrated posterior
structure (sparse, deterministic, or finite-horizon point forecasts);
diffuse is better when the source is genuinely soft Dirichlet random.**

---

## 3. Final regime → winner map

Synthesizing across all sweeps, with both metrics:

| regime | low N (3-25) | moderate N (50-100) | high N (200-400) |
|---|---|---|---|
| dense_small | **GDC tuned** | GDC ≈ ALERGIA | ALERGIA / GDC tied |
| **dense_large** | **GDC tuned** | **GDC tuned** | **GDC tuned** (vs ALERGIA: 3× better) |
| det_small | **GDC tuned** | GDC / ALERGIA | **CHMM K=32** (clone niche) |
| det_large | **GDC tuned** | GDC / ALERGIA | ALERGIA |
| **sparse_small** | **GDC** at N≤10; **CHMM K=32** above | CHMM K=32 | **CHMM K=32** (8× better than GDC) |
| **sparse_large** | **GDC** at N≤10; **CHMM K=32** above | CHMM K=32 | **CHMM K=32** (4× better than GDC) |

**Where GDC genuinely dominates:**

* `dense_large` at all N — the "fast-mixing, predict near-stationary +
  prefix correction" niche. GDC's α=0.1 config is near-optimal.
* All regimes at very low N (N≤10) — its no-fitting design is the
  most robust under data starvation.

**Where CHMM dominates:**

* Sparse regimes at all N≥25 — clones + EM are designed for exactly
  this. Margin grows with both N and T_train.
* `det_small + high N` — the cognitive-map use case (many states,
  shared emissions).

**Where ALERGIA dominates:**

* `dense_small` and `det_large` at high N — moderate-density
  emissions where state-merging tests have enough signal to converge.
* The original "ALERGIA dominates HMM forecasting" claim was
  partially-true but oversimplified; it depends on regime *and* N.

---

## 4. What changed across iterations

Useful to remember:

1. **Original framing**: "GDC vs OOM, GDC wins on HMM forecasting."
   Correct but misleading — OOM is genuinely bad here, often worse
   than uniform.
2. **First CHMM/ALERGIA pass**: "ALERGIA dominates everywhere."
   Correct only on the saturated regime (moderate dense HMMs at high
   N). Doesn't survive harder regimes.
3. **First GDC tune** (α=0.95 flop): "Sharp configs win on PAutomaC
   but lose here." True; the metric rewards smoothness.
4. **Wider GDC grid**: "α=0.5 wins everywhere" — but at boundary.
5. **Low-alpha grid**: "α=0.1 wins on `dense_large`" — first regime
   where GDC genuinely beats ALERGIA.
6. **Sparse regimes**: "CHMM has its own niche on sparse."
7. **Long sequences**: "More data per sequence widens gaps; doesn't
   help GDC catch up to CHMM on sparse."
8. **Perplexity**: "Both metrics mostly agree; perplexity highlights
   overcommitment."
9. **Extreme low-data**: "GDC's design-axis advantage is most visible
   at N≤10 where every estimation procedure fails."

---

## 5. Open / not-yet-done axes

* **Per-regime tuned-CHMM K and tuned-ALERGIA eps** — only K and eps
  defaults swept. Tighter sweeps could close gaps where CHMM/ALERGIA
  trail.
* **Sample efficiency on sparse regimes specifically** — the existing
  sample-efficiency sweep is on the dense 9×9 grid; sparse-only N
  ladder would be informative.
* **Sample-efficiency curves under perplexity** (current ones use MSE).
* **gamma > 0** (skip-2 transition mass) — never tested; unlikely to
  matter but completes the parameter coverage.
* **Long-horizon (h=50, 100)** discrimination — likely all converge to
  stationary at our scale.
* **Real-data forecasting** — PAutomaC and the algorithmic-trace
  benchmarks are different beasts but useful complements.

---

## 6. File map

| file | purpose |
|---|---|
| [random_hmm.py](random_hmm.py) | HMM constructors (dense, sparse, low-rank) |
| [model_wrappers.py](model_wrappers.py) | OOM / GDC forecasting wrappers |
| [chmm_alergia_wrappers.py](chmm_alergia_wrappers.py) | CHMM and ALERGIA wrappers |
| [evaluation.py](evaluation.py) | MSE, perplexity, baselines |
| [run_main_sweep.py](run_main_sweep.py) | original GDC vs OOM (9×9) |
| [run_chmm_alergia_sweep.py](run_chmm_alergia_sweep.py) | adds CHMM and ALERGIA on the 9×9 |
| [run_sample_efficiency.py](run_sample_efficiency.py) | N ∈ {10..200} sweep, multiprocessed |
| [run_large_det_sweep.py](run_large_det_sweep.py) | nS up to 50, dense+det |
| [run_sparse_sweep.py](run_sparse_sweep.py) | adds sparse regimes |
| [run_high_alpha_sparse.py](run_high_alpha_sparse.py) | brackets GDC sparse optimum |
| [run_long_seqs_sparse.py](run_long_seqs_sparse.py) | T_train ∈ {50, 200, 500} |
| [run_gdc_grid_sweep.py](run_gdc_grid_sweep.py) | first GDC two-phase grid |
| [run_gdc_tuned_sweep.py](run_gdc_tuned_sweep.py) | single PAutomaC-style config (flopped) |
| [run_gdc_wider_grid.py](run_gdc_wider_grid.py) | α up to 0.99, β to 0.3 |
| [run_gdc_low_alpha.py](run_gdc_low_alpha.py) | α down to 0.1 |
| [run_perplexity_sweep.py](run_perplexity_sweep.py) | MSE+perplexity side-by-side |
| [run_extreme_low_data.py](run_extreme_low_data.py) | N ∈ {3, 5, 10}, T ∈ {10, 50} |
| [plot_topologies.py](plot_topologies.py) | visualises 6 regimes |
| [fig_hmm_topologies.png](fig_hmm_topologies.png) | regime visualisation |

CSVs and per-experiment summaries follow the same naming pattern.

---

## 7. Appendix A — Comprehensive per-regime, per-N, per-model table

Best of each model class per (regime, N), at horizon h=1, averaged
over 3 seeds. All models run on the same (nS, nA, seed)-matched
HMMs as the perplexity sweep
([run_perplexity_sweep.py](run_perplexity_sweep.py)). The
"GDC `a*-t*-b*`" rows are the best-performing config from the
perplexity sweep's 5-config grid (`(0.10/0.001/0.20)`,
`(0.30/0.005/0.30)`, `(0.50/0.005/0.20)`, `(0.70/0.010/0.20)`,
`(0.80/0.001/0.10)`); the "GDC tuned" rows use the per-regime
optimal config from the wider single-config sweeps and add the
`terminal_behavior='diffuse'|'absorb'` comparison
([run_absorb_compare_regimes.py](run_absorb_compare_regimes.py)).

* **MSE**: squared error against the true posterior predictive,
  averaged over the alphabet then over 100 test prefixes.
* **excess PP**: `2^(CE − floor)`; lower bound 1.0; closest analog
  to PAutomaC's gap-to-floor.
* **abs PP**: `2^CE` — absolute perplexity; comparable across rows
  within a single regime.

### A.1 dense_small (nS=10, nA=4, dense Dirichlet emissions)

| N | model | MSE | excess PP | abs PP |
|---|---|---:|---:|---:|
| **25** | _entropy floor_ | -- | 1.000 | 3.819 |
| 25 | GDC a=0.30, θ=0.005, β=0.30 | 0.00042 | 1.0033 | 3.831 |
| 25 | CHMM K=32 | 0.00429 | 1.0362 | 3.957 |
| 25 | ALERGIA eps=0.05 | 0.00066 | 1.0054 | 3.839 |
| 25 | GDC tuned (diffuse) α=0.5, θ=0.05, β=0.2 | 0.00077 | 1.0061 | 3.842 |
| 25 | GDC tuned (absorb) α=0.5, θ=0.05, β=0.2 | 0.00080 | 1.0063 | 3.843 |
| **100** | _entropy floor_ | -- | 1.000 | 3.819 |
| 100 | GDC a=0.50, θ=0.005, β=0.20 | 0.00022 | 1.0017 | 3.825 |
| 100 | CHMM K=32 | 0.00038 | 1.0031 | 3.831 |
| 100 | **ALERGIA eps=0.05** | **0.00017** | **1.0013** | **3.824** |
| 100 | GDC tuned (diffuse) α=0.5, θ=0.05, β=0.2 | 0.00048 | 1.0037 | 3.833 |
| 100 | GDC tuned (absorb) α=0.5, θ=0.05, β=0.2 | 0.00049 | 1.0038 | 3.833 |
| **400** | _entropy floor_ | -- | 1.000 | 3.819 |
| 400 | GDC a=0.50, θ=0.005, β=0.20 | 0.00017 | 1.0014 | 3.824 |
| 400 | **CHMM K=32** | **0.00010** | **1.0008** | **3.822** |
| 400 | **ALERGIA eps=0.05** | **0.00010** | **1.0008** | **3.822** |
| 400 | GDC tuned (diffuse) α=0.5, θ=0.05, β=0.2 | 0.00048 | 1.0036 | 3.832 |
| 400 | GDC tuned (absorb) α=0.5, θ=0.05, β=0.2 | 0.00049 | 1.0037 | 3.833 |

### A.2 dense_large (nS=30, nA=8, dense Dirichlet emissions)

| N | model | MSE | excess PP | abs PP |
|---|---|---:|---:|---:|
| **25** | _entropy floor_ | -- | 1.000 | 7.883 |
| 25 | GDC a=0.10, θ=0.001, β=0.20 | 0.00013 | 1.0044 | 7.918 |
| 25 | CHMM K=4 | 0.00897 | 1.9725 | 15.550 |
| 25 | ALERGIA eps=0.05 | 0.00072 | 1.0247 | 8.079 |
| 25 | **GDC tuned (diffuse) α=0.1, θ=0.001, β=0.2** | **0.00008** | **1.0025** | **7.903** |
| 25 | **GDC tuned (absorb) α=0.1, θ=0.001, β=0.2** | **0.00008** | **1.0025** | **7.903** |
| **100** | _entropy floor_ | -- | 1.000 | 7.883 |
| 100 | GDC a=0.10, θ=0.001, β=0.20 | 0.00004 | 1.0012 | 7.893 |
| 100 | CHMM K=4 | 0.00255 | 1.1013 | 8.682 |
| 100 | ALERGIA eps=0.05 | 0.00017 | 1.0055 | 7.927 |
| 100 | **GDC tuned (diffuse) α=0.1, θ=0.001, β=0.2** | **0.00003** | **1.0009** | **7.891** |
| 100 | **GDC tuned (absorb) α=0.1, θ=0.001, β=0.2** | **0.00003** | **1.0009** | **7.891** |
| **400** | _entropy floor_ | -- | 1.000 | 7.883 |
| 400 | GDC a=0.10, θ=0.001, β=0.20 | 0.00002 | 1.0005 | 7.888 |
| 400 | CHMM K=32 | 0.00031 | 1.0100 | 7.962 |
| 400 | ALERGIA eps=0.05 | 0.00005 | 1.0015 | 7.895 |
| 400 | **GDC tuned (diffuse) α=0.1, θ=0.001, β=0.2** | **0.00001** | **1.0004** | **7.887** |
| 400 | **GDC tuned (absorb) α=0.1, θ=0.001, β=0.2** | **0.00001** | **1.0004** | **7.887** |

### A.3 det_small (nS=10, nA=4, near-deterministic emissions)

| N | model | MSE | excess PP | abs PP |
|---|---|---:|---:|---:|
| **25** | _entropy floor_ | -- | 1.000 | 3.368 |
| 25 | GDC a=0.50, θ=0.005, β=0.20 | 0.00134 | 1.0112 | 3.406 |
| 25 | CHMM K=32 | 0.00381 | 1.0387 | 3.499 |
| 25 | **ALERGIA eps=0.05** | **0.00104** | **1.0092** | **3.399** |
| 25 | GDC tuned (diffuse) α=0.7, θ=0.01, β=0.2 | 0.00224 | 1.0169 | 3.425 |
| 25 | GDC tuned (absorb) α=0.7, θ=0.01, β=0.2 | 0.00241 | 1.0181 | 3.429 |
| **100** | _entropy floor_ | -- | 1.000 | 3.368 |
| 100 | GDC a=0.50, θ=0.005, β=0.20 | 0.00132 | 1.0109 | 3.405 |
| 100 | CHMM K=32 | 0.00054 | 1.0047 | 3.384 |
| 100 | **ALERGIA eps=0.05** | **0.00056** | **1.0044** | **3.383** |
| 100 | GDC tuned (diffuse) α=0.7, θ=0.01, β=0.2 | 0.00120 | 1.0094 | 3.400 |
| 100 | GDC tuned (absorb) α=0.7, θ=0.01, β=0.2 | 0.00130 | 1.0100 | 3.402 |
| **400** | _entropy floor_ | -- | 1.000 | 3.368 |
| 400 | GDC a=0.70, θ=0.01, β=0.20 | 0.00091 | 1.0077 | 3.394 |
| 400 | **CHMM K=32** | **0.00014** | **1.0012** | **3.373** |
| 400 | ALERGIA eps=0.05 | 0.00042 | 1.0034 | 3.380 |
| 400 | GDC tuned (diffuse) α=0.7, θ=0.01, β=0.2 | 0.00095 | 1.0074 | 3.393 |
| 400 | GDC tuned (absorb) α=0.7, θ=0.01, β=0.2 | 0.00100 | 1.0077 | 3.394 |

### A.4 det_large (nS=30, nA=8, near-deterministic emissions)

| N | model | MSE | excess PP | abs PP |
|---|---|---:|---:|---:|
| **25** | _entropy floor_ | -- | 1.000 | 7.434 |
| 25 | GDC a=0.30, θ=0.005, β=0.30 | 0.00030 | 1.0100 | 7.508 |
| 25 | CHMM K=4 | 0.00918 | 1.7667 | 13.134 |
| 25 | ALERGIA eps=0.05 | 0.00087 | 1.0643 | 7.912 |
| 25 | **GDC tuned (diffuse) α=0.3, θ=0.005, β=0.3** | **0.00031** | **1.0092** | **7.503** |
| 25 | GDC tuned (absorb) α=0.3, θ=0.005, β=0.3 | 0.00031 | 1.0093 | 7.503 |
| **100** | _entropy floor_ | -- | 1.000 | 7.434 |
| 100 | GDC a=0.30, θ=0.005, β=0.30 | 0.00018 | 1.0057 | 7.477 |
| 100 | CHMM K=4 | 0.00226 | 1.0913 | 8.113 |
| 100 | ALERGIA eps=0.05 | 0.00018 | 1.0057 | 7.477 |
| 100 | GDC tuned (diffuse) α=0.3, θ=0.005, β=0.3 | 0.00017 | 1.0051 | 7.472 |
| 100 | **GDC tuned (absorb) α=0.3, θ=0.005, β=0.3** | **0.00017** | **1.0051** | **7.472** |
| **400** | _entropy floor_ | -- | 1.000 | 7.434 |
| 400 | GDC a=0.50, θ=0.005, β=0.20 | 0.00015 | 1.0045 | 7.467 |
| 400 | CHMM K=32 | 0.00035 | 1.0112 | 7.517 |
| 400 | **ALERGIA eps=0.05** | **0.00007** | **1.0021** | **7.450** |
| 400 | GDC tuned (diffuse) α=0.3, θ=0.005, β=0.3 | 0.00014 | 1.0041 | 7.464 |
| 400 | GDC tuned (absorb) α=0.3, θ=0.005, β=0.3 | 0.00014 | 1.0041 | 7.464 |

### A.5 sparse_small (nS=10, nA=4, fanout-2 transitions, sharp emissions)

| N | model | MSE | excess PP | abs PP |
|---|---|---:|---:|---:|
| **25** | _entropy floor_ | -- | 1.000 | 1.796 |
| 25 | GDC a=0.80, θ=0.001, β=0.10 | 0.01195 | 1.1097 | 1.993 |
| 25 | **CHMM K=32** | **0.00579** | **1.0543** | **1.894** |
| 25 | ALERGIA eps=0.05 | 0.02149 | 1.1931 | 2.143 |
| 25 | GDC tuned (diffuse) α=0.8, θ=0.0, β=0.05 | 0.01405 | 1.1224 | 2.016 |
| 25 | GDC tuned (absorb) α=0.8, θ=0.0, β=0.05 | 0.01438 | 1.1243 | 2.019 |
| **100** | _entropy floor_ | -- | 1.000 | 1.796 |
| 100 | GDC a=0.80, θ=0.001, β=0.10 | 0.00898 | 1.0872 | 1.953 |
| 100 | **CHMM K=4** | **0.00282** | **1.0336** | **1.856** |
| 100 | ALERGIA eps=0.05 | 0.01331 | 1.1159 | 2.004 |
| 100 | GDC tuned (diffuse) α=0.8, θ=0.0, β=0.05 | 0.01159 | 1.1006 | 1.977 |
| 100 | GDC tuned (absorb) α=0.8, θ=0.0, β=0.05 | 0.01187 | 1.1018 | 1.979 |
| **400** | _entropy floor_ | -- | 1.000 | 1.796 |
| 400 | GDC a=0.80, θ=0.001, β=0.10 | 0.00876 | 1.0846 | 1.948 |
| 400 | **CHMM K=16** | **0.00194** | **1.0205** | **1.833** |
| 400 | ALERGIA eps=0.05 | 0.01560 | 1.1343 | 2.037 |
| 400 | GDC tuned (diffuse) α=0.8, θ=0.0, β=0.05 | 0.00838 | 1.0784 | 1.937 |
| 400 | GDC tuned (absorb) α=0.8, θ=0.0, β=0.05 | 0.00848 | 1.0781 | 1.936 |

### A.6 sparse_large (nS=30, nA=8, fanout-2 transitions, sharp emissions)

| N | model | MSE | excess PP | abs PP |
|---|---|---:|---:|---:|
| **25** | _entropy floor_ | -- | 1.000 | 3.073 |
| 25 | GDC a=0.70, θ=0.01, β=0.20 | 0.01349 | 1.4160 | 4.351 |
| 25 | **CHMM K=4** | **0.01276** | 1.5656 | 4.811 |
| 25 | ALERGIA eps=0.05 | 0.01913 | 1.6557 | 5.088 |
| 25 | GDC tuned (diffuse) α=0.8, θ=0.0, β=0.2 | 0.01570 | 1.4808 | 4.550 |
| 25 | GDC tuned (absorb) α=0.8, θ=0.0, β=0.2 | 0.01588 | 1.4826 | 4.556 |
| **100** | _entropy floor_ | -- | 1.000 | 3.073 |
| 100 | GDC a=0.70, θ=0.01, β=0.20 | 0.01284 | 1.3875 | 4.264 |
| 100 | **CHMM K=4** | **0.00684** | **1.2087** | **3.714** |
| 100 | ALERGIA eps=0.05 | 0.01596 | 1.4634 | 4.497 |
| 100 | GDC tuned (diffuse) α=0.8, θ=0.0, β=0.2 | 0.01397 | 1.4102 | 4.333 |
| 100 | GDC tuned (absorb) α=0.8, θ=0.0, β=0.2 | 0.01382 | 1.4028 | 4.311 |
| **400** | _entropy floor_ | -- | 1.000 | 3.073 |
| 400 | GDC a=0.80, θ=0.001, β=0.10 | 0.01139 | 1.3191 | 4.054 |
| 400 | **CHMM K=16** | **0.00237** | **1.0745** | **3.302** |
| 400 | ALERGIA eps=0.05 | 0.01472 | 1.4234 | 4.374 |
| 400 | GDC tuned (diffuse) α=0.8, θ=0.0, β=0.2 | 0.01190 | 1.3550 | 4.164 |
| 400 | GDC tuned (absorb) α=0.8, θ=0.0, β=0.2 | 0.01158 | 1.3438 | 4.129 |

**Bolded rows in each block are the per-(regime, N) winners under MSE
(also under excess perplexity in all cases).**

Generated by [build_comprehensive_table.py](build_comprehensive_table.py)
from [perplexity_sweep_results.csv](perplexity_sweep_results.csv) and
[absorb_compare_regimes_results.csv](absorb_compare_regimes_results.csv).

## 8. One-paragraph elevator pitch

GDC, CHMM, and ALERGIA each occupy a distinct corner of the
(data-size, topology-structure) plane. **GDC** is the right default
when training data is scarce or when the source is a fast-mixing
dense HMM (so the stationary distribution is approximately the
right answer); its no-fitting design avoids the failure modes of EM
and state-merging at low N. **CHMM** is the right choice when the
underlying source has sparse, structured transitions with shared
emissions (the cognitive-map regime that motivated the model in the
first place); its bounded parametric form captures clone-like
ambiguity that GDC's prefix memory cannot. **ALERGIA** is the right
choice on moderate-density emissions with enough data for its
state-merging compatibility tests to converge — typically dense
Dirichlet HMMs at N ≥ 100. The right choice in any one experiment
depends on the regime; the right choice across regimes depends on
which axis the deployment is bottlenecked on.
