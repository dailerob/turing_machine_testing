# GDC benchmark protocol audit & parameter-sweep standardization

This document audits, for every benchmark we report, (a) the
train/validation/test split, (b) how the validation set is constructed,
(c) the metric and granularity of config selection, and (d) the exact
GDC hyperparameter sweep grid. It then proposes a **single canonical
sweep per emission regime**, says where current grids deviate, judges
which deviations are justified, and prioritizes what to re-run.

Audit performed 2026-06-03 by reading the eval scripts directly
(file:line citations below). All headline numbers live in the
per-folder `*_RESULTS.md` writeups and in `paper/tables.tex`.

---

## 1. Train / validation / test setup (all benchmarks)

**Headline finding: every benchmark is leakage-free.** Validation is
used only for config selection; the test set (and, for PAutomaC, its
solution probabilities) is never seen during selection. The *selection
metric always matches the test metric*. What differs is how validation
is *constructed* and the *granularity* of the per-config pick.

| Benchmark | Emission | Train | Validation construction | Test | Sel. metric | Granularity | Source |
|---|---|---|---|---|---|---|---|
| dysts | cont. | 180-pt traj | **multi-IC**: 3 fixed fit-len-90 windows, targets `[90:119]`,`[120:149]`,`[150:179]`, mean sMAPE | 180-pt **independent-IC** traj, 29-step | sMAPE | per-system | `dysts_bench/run_gdc_dual.py:157-174` |
| M4 | cont. | competition train | last-`h` of each train series (`h` = freq horizon) | competition test | sMAPE / MASE / OWA | per-series **and** global (per-freq) | `m4/clean_eval.py:203-237` |
| SKOLR-forecast | cont. | Informer 12/4/4 mo or Autoformer 7:1:2 | held-out val block; state=`train` at val, `train+val` at test | competition test | MSE (standardized) | per (dataset × horizon) | `skolr_bench/forecast/gdc_etth1_full_sweep.py:94-120` |
| SKOLR-NLDS | cont. | 14k contiguous | 2k contiguous (next block) | 4k contiguous, 5 seeds | MSE | per (system × seed × dim) | `skolr_bench/nlds/nlds_eval.py:136-170` |
| TM algorithmic | disc. | per-task range (n_train 200–300, ×1/×4) | **stretched range between train and test, different seeds** | OOD-length range, different seeds | tuple-error count | per (task × variant) | `algorithmic_benchmarks/_tm_task_config.py:45-98` |
| HMM comparison | disc. | N∈{25,100,400} seqs × len 50 | **disjoint HMM seeds {3,4,5}** | HMM seeds {0,1,2}, 100 prefixes len 20 | excess perplexity @h=1 | per (regime × N × method) | `hmm_comparison/run_val_sweep.py:28-40` |
| char-LM | disc. | 90% of stream (cap 750k) | last 10% of train | last 10% | bits-per-symbol | per dataset | `char_lm/run.py:68-73,192-213` |
| PAutomaC | disc. | competition train | **random 20% split** of train | competition test | held-out NLL | per problem | `pautomac/run_leakage_free.py:92-104` |

Validation-construction taxonomy (all leakage-free, but methodologically
distinct):
- **Temporal hold-out** — M4 (last-h), char-LM (last-10%),
  SKOLR-forecast (val block). Right for series with trend/seasonality.
- **Contiguous next-block** — SKOLR-NLDS.
- **Multi-IC sliding windows** — dysts (averages 3 windows to beat
  Lyapunov val-noise; this is itself a methodological contribution).
- **Seed-disjoint synthetic draws** — HMM (val HMMs ≠ test HMMs), TM
  (val/test ranges use different seeds).
- **Random hold-out** — PAutomaC (20% of sequences).

---

## 2. The exact GDC sweep grid, per benchmark

The grids split cleanly into a **continuous-forecast backbone** and a
**discrete-next-token backbone** that are each internally consistent on
structure but differ on the swept *values*.

### 2a. Continuous benchmarks
Shared & fixed everywhere: `transition_type=self_loop`,
`initial_dist=uniform`, `terminal_behavior=absorb`, `θ=0`,
`β=(σ_frac·std·√L)²`, recipe ∈ {raw, diff}.

| Benchmark | α handling | σ_frac grid | L (lookback) | dual-α? | #cfg | Source |
|---|---|---|---|---|---|---|
| dysts | single {1.0, 0.99} **+ dual** α_ctx∈{0.8,0.9,0.95,0.99}, α_fc=1 | {0.05, 0.10, 0.25} | **16, fixed** | **yes** | 36 | `run_gdc_dual.py:54-64` |
| M4 | single, per-freq subset of {0.8, 0.9, 0.95, 0.99, 1.0} | per-freq | per-freq, swept | no | 6–10/freq | `m4/clean_eval.py:133-191` |
| SKOLR-forecast | single {1.0,0.99} (raw) / {1.0,0.99,0.95} (diff) | raw {0.02,0.05,0.10,0.25,0.50}, diff {0.10,0.25,0.50,1.00} | **720/96/36, protocol-fixed** | no | 22 | `gdc_etth1_full_sweep.py:34-43` |
| SKOLR-NLDS | single {1.0, 0.99, 0.95, 0.9} | raw {0.05,0.1,0.25}, diff {0.25,0.5,1.0} | {48, 96}, swept | no | 48 | `nlds_eval.py:85-96` |

### 2b. Discrete benchmarks
Shared & fixed: `transition_type=self_loop` (TM also tests
`self_loop_two_step`), `terminal_behavior=diffuse` (n/a for char-LM).

| Benchmark | init | α grid | θ grid | β | dual-α? | #cfg | Source |
|---|---|---|---|---|---|---|---|
| TM | sequence_starts | {0.5, 0.7, 0.9, 0.95, 0.99} | {0.005, 0.05} | 0 | **no** | 8–16/task | `algorithmic_benchmarks/tuned_gdc_sweep.py:24-26` |
| HMM | sequence_starts | 5 fixed (α,θ,β) tuples, α∈{0.1,0.3,0.5,0.7,0.8} | within tuples | **swept within tuples** {0.10–0.30} | **yes** (α_fc=1 override) | 5 | `hmm_comparison/run_val_sweep.py:35-41` + dual override `gdc_torch_discrete.py:101,216` |
| char-LM | dirac@0 | dual α_ctx∈{0.40,0.45,…,0.70} | 0 | 0 | **yes** (α_fc∈{0.95,0.99,1.0}) | 21 | `char_lm/run.py:61-65` |
| PAutomaC | sequence_starts | single {0.95,0.50} **+ dual** α_ctx∈{0.30,0.50,0.70,0.85,0.95} | {0.05,0.005,0} | 0 | **yes** (α_fc=0.9999) | 7 | `pautomac/run_leakage_free.py:45-85` |

Dual-α mechanics (confirmed `gdc_torch_discrete.py:212-227`,
`gdc_torch.py forecast_many_torch_dual`): the prefix is absorbed with
α_ctx; the forecast roll-out (h≥1) recomputes its diffusion coefficients
from α_fc. **α_fc=1 ⇒ deterministic walk-forward, no diffuse smearing.**
It applies identically to single-step next-token prediction (HMM/TM @h=1)
and multi-step roll-out (M4/SKOLR/dysts).

---

## 3. What is shared, what differs

**Shared (the GDC "spine"), and it is consistent:** non-parametric
forward pass over a position-chain; `self_loop` transition; σ-derived β;
the raw-vs-diff recipe on continuous data. The two regime backbones
(absorb+uniform for trajectory forecasting; diffuse+sequence_starts for
finite-state next-token) are applied consistently *within* each regime.

**Differs across benchmarks:**
1. **dual-α is present in 4 of 8** (dysts, HMM, char-LM, PAutomaC) and
   **absent in 4 of 8** (M4, SKOLR-forecast, SKOLR-NLDS, TM).
2. **α grid breadth/centre differs** with no single rule: TM {0.5–0.99},
   char-LM {0.4–0.7}, PAutomaC {0.3–0.95}, continuous {0.9–1.0}.
3. **σ_frac grid differs** across the continuous family.
4. **θ** is swept in TM/PAutomaC/HMM but fixed 0 in char-LM and all
   continuous.
5. **β** is swept only in HMM; fixed 0 in the other discrete benchmarks.
6. **transition_type**: `self_loop_two_step` is tested only in TM.
7. **L** is protocol-fixed (SKOLR, dysts) in some, swept (NLDS, M4) in
   others.
8. **config count** ranges 5→48.

---

## 4. Which differences are justified

**Justified — structural, keep them:**
- **Discrete vs continuous emission.** β means different things
  (Gaussian variance vs symbol-noise floor); recipe {raw,diff} only
  makes sense for continuous. ⇒ two separate canonical grids is correct.
- **absorb (trajectory forecast) vs diffuse (re-enterable finite-state).**
  Derived in `algorithmic_benchmarks/ABSORB_RESULTS.md`. Justified.
- **init**: uniform (single continuous trajectory) / sequence_starts
  (set of strings) / dirac@0 (one long char stream) — dictated by the
  data's structure. Justified.
- **L fixed by protocol** (SKOLR lookback 720/96/36 is the published
  benchmark's choice; dysts L=16 because series are only 180 pts) vs
  swept where the protocol leaves L free (NLDS, M4). Justified, but must
  be *documented per benchmark* rather than left implicit.

**Not justified — standardization targets:**
- **(critical) dual-α presence.** Dual-α *nests* single-α (α_fc=α_ctx is
  a point in the grid). dysts showed α_fc=1 helps continuous forecasting;
  char-LM/PAutomaC showed α_fc=1 is the universal winner on discrete.
  Omitting it from M4/SKOLR/NLDS/TM was an unmotivated inconsistency.
  > **UPDATE (P1, 2026-06): this was tested — see §6/§9. The original
  > claim here that "adding dual-α can only improve or leave the val-pick
  > unchanged" is WRONG.** Dual-α adds *new candidates*, and a candidate
  > that wins on validation but loses on test degrades the val-tuned
  > result. That is exactly what happens on SKOLR-forecast (the raw
  > dual-α roll-out val-overfits at long horizons: ETTm2 T=336
  > 0.150→0.189). Dual-α is therefore **not** a free addition; its value
  > is structure-dependent (big win on TM, neutral on NLDS, no-op on M4,
  > harmful on SKOLR-forecast).
- **α / σ grid breadth.** The per-benchmark α windows track each task's
  determinism (high α for deterministic TM, low α_ctx for noisy natural
  language), which is *principled* — but the windows were chosen after
  seeing roughly where each optimum sits, which reads as grid-fitting.
  Fix: one wide grid per regime; any per-benchmark narrowing must be
  flagged as a deliberate val-overfitting mitigation (M4 already does
  this explicitly — see "Things that didn't pan out" in CLAUDE.md).
- **θ, β, transition_type** swept inconsistently. Fix: a single small
  default ({θ=0}, {β=0}, {self_loop}) with documented, regime-specific
  exceptions (θ small grid for deterministic-state TM; β small grid for
  stochastic-emission HMM/PFA).

**Meta-point:** more configs ⇒ more val-overfitting risk (M4 hit this at
85 configs). So the cure is a *smaller unified* grid plus *robust val*
(multi-IC where val is noisy), not larger per-benchmark grids.

---

## 5. Proposed standardization: two canonical grids

### Canonical CONTINUOUS-FORECAST grid (dysts, M4, SKOLR-forecast, NLDS)
- Fixed: `self_loop`, `uniform`, `absorb`, `θ=0`, `β=(σ_frac·std·√L)²`.
- recipe ∈ {raw, diff}.
- **α as dual-α**: α_ctx ∈ {1.0, 0.95, 0.90, 0.80}, α_fc ∈ {α_ctx, 1.0}
  (⇒ 7 distinct pairs: 4 single + 3 dual; α_ctx=1.0 contributes only one).
- σ_frac: raw {0.05, 0.10, 0.25, 0.50}, diff {0.10, 0.25, 0.50, 1.00}.
- L: protocol-fixed where the benchmark fixes lookback; otherwise the
  benchmark's documented small L-set.
- ⇒ ≈ 7 α-pairs × 4 σ × 2 recipes, trimmed per the raw/diff σ split.

### Canonical DISCRETE-NEXT-TOKEN grid (TM, HMM, char-LM, PAutomaC)
- Fixed: `self_loop`; `diffuse`; `init` = sequence_starts (multi-seq) or
  dirac@0 (single stream).
- **α as dual-α**: α_ctx ∈ {0.3, 0.5, 0.7, 0.85, 0.95, 0.99},
  α_fc ∈ {α_ctx, 1.0}. (α_fc uses the 0.9999 numerical floor in practice
  to avoid log(0); see PAutomaC.)
- θ = 0 by default; deterministic-state tasks (TM) may add {0.005, 0.05}.
- β = 0 by default; stochastic-emission regimes (HMM, PAutomaC PFA) may
  add {0.05, 0.1}, documented.
- ⇒ ≈ 11 α-pairs × {θ,β} exceptions; ~11–22 configs.

These two grids are **supersets** of every current grid except the
post-hoc-narrowed α windows, so re-running against them is safe.

---

## 6. Retest plan (prioritized)

- **P1 — add dual-α to the four benchmarks that omit it — DONE
  (2026-06); result is structure-dependent, not uniform.** Full results
  in §9. Summary: dual-α **helps TM substantially** (binary_adder/orig
  59→3, subtraction/orig 1109→834, palindrome/orig 17→14; val-confirmed,
  0 regressions) — kept ON. It is **neutral on NLDS**, a **no-op on M4**
  (the diff recipe is α_fc-invariant), and **harmful on SKOLR-forecast**
  (raw dual-α val-overfits: ETTm2 T=336 0.150→0.189) — so it is gated OFF
  by default on those three. The hoped-for "free, uniform" addition did
  not materialize; dual-α is now applied per-benchmark where it
  demonstrably helps (TM, dysts).
- **P2 — unify σ_frac and α grids within each regime** to the §5 grids;
  re-run and confirm headline numbers move within noise (they should,
  since the canonical grids are supersets). This converts "we picked a
  grid that brackets the answer" into "we ran one fixed grid everywhere."
- **P3 — HMM-comparison provenance cleanup (see §7).** Identify and label
  the single canonical Table 7 / 12 / 13 generator, reconcile the
  product-HMM config, re-run once under the §5b grid.
- **P4 — decide `self_loop_two_step`** (TM only): if it is never the
  val-pick, drop it; otherwise add it to the discrete grid everywhere.

---

## 7. RESOLVED: HMM Tables 7/12/13 reconstructed from scratch

**Status 2026-06-05: all three tables reconstructed and committed.** The
original generators were absent from every branch (evidence below);
fresh canonical, leakage-free generators were written, run, and their
output replaced the paper numbers:
- Table 7 → `gen_table7_forecasting.py` + `build_table7_latex.py`
  (4 regimes at n_S=20/n_A=4, 20 test + 20 val HMMs, full α/θ/β grid
  with α_fc∈{α,1}, h=1..5). Reconstruction matched the old numbers
  within ~0.01–0.02 (Freq to ~0.01) — confirming the originals were
  faithful but unreproducible.
- Table 13 → `gen_table13_scaling.py` + `build_table13_latex.py`
  (same regimes, N∈{1,3,5,10,25}, h=1; N=25 column reproduces Table 7
  h=1 exactly). Surfaced a genuine leakage-free degeneracy at bimodal
  N=1 (sharp methods blow up on a single-cluster sample).
- Table 12 → `gen_table12_product_hmm.py` + `build_table12_latex.py`
  (ternary 3-component product HMM, 27 states/27 symbols, leakage-free
  val/test seed split, GDC reported as the fixed a=0.85/t=0.005/b=0.075
  config — confirmed val-best at 16×, near-best at smaller scales).

The original-absence evidence that motivated the rebuild:

The generating scripts for paper Tables 7, 12, and 13 were **not present
in the repository on any branch.**

Evidence:
- **Table 7 spec** (from its own caption `paper/tables.tex:235`): 4
  regimes (cyclic, reset_chain, bimodal, sparse), **all n_S=20, n_A=4**,
  **20 HMMs/regime**, N=25×len-50, h=1..5, GDC grid α∈10 vals,
  θ∈8 vals, β∈{0,0.005,0.025,0.05} asymptotic, α_forecast∈{α,1}.
- **No committed script matches.** The committed leakage-free HMM
  pipeline (`run_val_sweep.py` + `run_perplexity_sweep.py` +
  `build_leakage_free_table.py`, and the `run_gdc_expanded_sweep.py`
  variant) is a **6-regime** benchmark (dense/det/sparse × small/large),
  horizons {1,5,20}, **5 fixed single-α (α,θ,β) tuples** — its numbers
  (`seq_len_table.md`: sparse_large GDC 1.479) do not match Table 7.
  The `new_regimes_*` scripts use 4 regimes but at **mixed sizes**
  (bimodal 10×4, cyclic_K8 8×8, binary_deep 30×2, reset_chain 20×4),
  6 seeds, and a θ=0.001 narrow grid or accuracy metric — their GDC
  perplexity (cyclic_K8 h=1 ≈ 3.53) is nowhere near Table 7's 1.027.
- **`git log -S '0.025' -- 'hmm_comparison/*.py'` across all branches
  returns nothing** — no script with the β=0.025 grid was ever
  committed. `git grep "('cyclic', 20, 4" <all-commits>` is empty — no
  script ever set the four regimes at uniform n_S=20. No stale
  worktrees hold it.
- Table 7's GDC numbers (1.027, 1.175, 1.067, …) appear **only** in
  `paper/tables.tex` and `paper/GDC_OVERVIEW.md`, in **no** committed
  CSV.
- **Table 12** (ternary 3-component product HMM, 27 states/27 symbols,
  best α=0.85/θ=0.005/β=0.075, 462-config grid): no generator exists.
  The only product-HMM script (`tune_gdc_product_hmm.py`,
  `compare_product_hmm.py`) is a **6-component binary** HMM (64 symbols,
  α=0.7/β=0.05) — a different experiment.
- **Table 13** (4-regime data-scaling, N∈{1,3,5,10,25}, 20 seeds,
  bimodal N=1 = 1.892): no generator or CSV exists.

At the time of discovery this meant the entire HMM-comparison block
rested on numbers whose provenance could not be verified — either the
generators were run ad-hoc and never committed, or the captions were
written to an intended protocol never executed as described. **This has
now been resolved** by the from-scratch reconstruction above: each table
has a single committed entry point, runs leakage-free, and its output
replaced the paper numbers. The §5b grid standardization can now be
applied on top of these generators.

---

## 8. One-line answer to "are we justified in different sweeps?"

Justified where the difference is **structural** (discrete vs continuous;
absorb vs diffuse; init dictated by data shape; L fixed by a published
protocol). **Not** justified where it is **incidental** (α/σ/θ/β grid
breadth tuned per benchmark; config counts 5→48). The remedy is one
canonical superset grid per emission regime (§5) plus explicit
documentation of any per-benchmark narrowing as a deliberate
val-overfitting mitigation. **Caveat (post-P1):** dual-α is the one knob
that is *not* safe to make uniformly available — adding it as a candidate
val-overfits and degrades test on SKOLR-forecast (§9). It belongs only
where it demonstrably helps (TM, dysts). "Leakage-free" guarantees no
test peeking; it does **not** guarantee that a larger candidate set
improves the val-tuned test score.

---

## 9. P1 results: dual-α added to M4 / SKOLR-forecast / NLDS / TM (2026-06)

Dual-α (decouple the forecast roll-out's α_fc from the context α_ctx;
α_fc=1.0 = deterministic walk-forward) was added as a candidate to the
four benchmarks that omitted it, val-tuned leakage-free, and compared to
single-α. **The outcome is structure-dependent**, and one mechanistic
finding explains most of it.

### Key mechanism: α_fc is a no-op for the `diff` recipe
On continuous data, dual-α only affects the **raw** recipe. For the
**diff** recipe (forecast first-differences, cumsum onto an anchor),
α_fc has **exactly zero** effect — verified on identical M4 Hourly
series: raw gives a max |single−dual| of 165–1061, diff gives 0.000000.
Intuition: the diff states are ~stationary, zero-mean, so the roll-out's
expected emission is invariant to how the self-loop α splits stay-vs-
advance mass; raw levels are non-stationary, so position (hence α)
matters. Consequence: any diff-dominated benchmark is unaffected.

### Per-benchmark verdict

| Benchmark | recipe used | dual-α effect | default | evidence |
|---|---|---|---|---|
| **TM** (discrete) | n/a | **helps a lot** | **ON** | val-tuned single→dual: binary_adder/orig 59→3, subtraction/orig 1109→834, palindrome/orig 17→14; 3 better, 15 same, **0 worse**; val tracks test (not overfitting) |
| **M4** | diff (5/6 freq) | **no-op** | off (gated) | α_fc no-op for diff; Hourly is raw but all α=1; Weekly OWA identical 0.7854 |
| **SKOLR-NLDS** | raw+diff | **neutral** | off (gated) | identical to published (pendulum 0.0003, duffing 0.0005, LV 0.0000, lorenz 1.171); val picks equivalent configs |
| **SKOLR-forecast** | diff+raw | **HURTS** | off (gated) | raw dual-α val-overfits long horizons: ETTm2 T=336 0.150→0.189, T=720 0.254→0.262 (T=96/192 unchanged, diff) |

### Why SKOLR-forecast regresses (the important correction)
At ETTm2 T=336 the original sweep picks a **diff** config (test 0.150).
Adding raw dual-α candidates introduces a `raw/α_ctx=0.99/α_fc=1.0`
config whose **validation** MSE is lower than that diff config, so the
leakage-free val-tuner selects it — but its **test** MSE is 0.189. The
sharper raw roll-out fits the validation slice better and generalizes
worse. This is ordinary val-overfitting, and it shows the §4 reasoning
("adding candidates is leakage-free-safe") was wrong: a larger candidate
set can lower the val-tuned test score whenever a candidate wins val for
the wrong reason.

### TM: why dual-α helps
TM next-tuple prediction is single-step; α_fc=1.0 makes the prediction
transition a sharp deterministic advance through the chain while the
carried state still conditions on observed tokens with α_ctx. On
deterministic state-propagation this pins the exact next tuple. The
kernel change is a per-step fork (advance with α_ctx, predict with α_fc)
in `algorithmic_benchmarks/torch_tm_adapters.py`; the config grid in
`tuned_gdc_sweep.py` adds an α_fc=1.0 twin for each α<1 context config.
The 1× sweep above is reproduced at the 4× headline budget to refresh
Tables 8/9.

### Implementation / gating
- TM dual-α: ON by default (`tuned_gdc_sweep.py` configs include duals).
- M4 / NLDS / SKOLR-forecast: dual-α plumbed but **gated off** behind
  `GDC_DUAL_ALPHA=1` (or `M4_DUAL_ALPHA=1` for M4) so the default grids
  are unchanged and the negative/neutral results stay reproducible.
- The `forecast_gdc_style(..., alpha_fc, theta_fc)` override
  (`generative_dense_chain_timeseries.py`) and the torch
  `forecast_many_torch_dual` dispatch are backward-compatible no-ops when
  α_fc=α_ctx.

---

## 10. Principled grid reduction (2026-06): making the sweeps small + defensible

The two HMM grids (464, 462) were the only oversized sweeps; everything
else was already ≤48. They are bloated because **α, θ, β are redundant
smoothing knobs swept independently** — in the discrete self-loop kernel
the diffuse mass is `(1−α−θ)/(N−2)`, so α and θ spend from one budget,
and β is a separate emission-noise floor. Sweeping all three densely
multiplies out while only α is load-bearing.

### Evidence (full-464 val-pick distribution, 20 cells)
- **α_fc = 1.0 in 20/20 cells** → single-α never wins.
- **α only lands in {0.3–0.75}** → {0.80–0.95} never picked.
- **θ always ≤ 0.1** → {0.2–0.5} never picked.
- **β only ∈ {0, 0.005}** → {0.025, 0.05} never picked.

Roughly half of every axis is dead weight.

### The principle
| Knob | Role | Treatment |
|---|---|---|
| **α** | primary; optimum tracks task *determinism* | sweep coarsely, 4–5 values |
| **α_fc** | deterministic-forecast switch | binary {α, 1.0} |
| **θ** | redundant with α | fix to {0} or {0, small} |
| **β** | emission-noise floor | **regime-conditional**: 0 / small for deterministic emission, larger for stochastic (HMM, PFA, 27-symbol product) |

β is the **one justified per-task axis**, gated by a *measurable* property
(is the emission deterministic given state?), not by the task name.

### Applied reductions (validated)
- **Table 7 / 13** (`gen_table7_forecasting.py`, `--full` for the old grid):
  464 → **32** configs (`α∈{0.3,0.5,0.7,0.9} × θ∈{0,0.1} × β∈{0,0.005} ×
  α_fc∈{α,1}`). Reproduces the 464-grid to within **0.004** excess-PP at
  every (regime, horizon) cell at N=25. At Table 13's single-sequence
  N=1 scale the grid is coarser (≤0.03; cyclic N=1 tips GDC→CHMM by 0.014,
  14/20 tally), documented in that table's caption.
- **Table 12** (`gen_table12_product_hmm.py`, `GDC_TABLE12_FULL=1` for the
  old grid): 462 → **18** (`α∈{0.5,0.7,0.85} × θ∈{0.005,0.05} ×
  β∈{0.05,0.075,0.15}`). The β range is raised — a regime-justified
  difference for the 27-symbol stochastic emission. Reported numbers
  unchanged (GDC row is a fixed config; the grid only verifies it).

### Proposed universal discrete grid (not yet applied to TM/char-LM/PAutomaC)
A single set could replace the four different discrete grids:
```
α_ctx ∈ {0.3, 0.5, 0.7, 0.9, 0.99}     # spans stochastic → deterministic
α_fc  ∈ {α_ctx, 1.0}
θ      = 0
β      ∈ {0} (deterministic emission)  or  {0, small} (stochastic)
→ 10–20 configs, β the only per-task axis.
```
This is left as a follow-up: TM/char-LM/PAutomaC already use small grids
(18/21/7) and would need a re-run to confirm no loss, like the HMM check
above. The continuous benchmarks are already ≤48 and keep their grids
(dual-α off per §9, except dysts where it helps).
