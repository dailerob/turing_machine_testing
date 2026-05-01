# HMM Forecasting: Spectral OOM vs GDC

An experiment comparing two sequence models on their ability to predict the
future-symbol distribution of a **random hidden Markov model**, measured by
mean squared error against the HMM's *exact* posterior predictive
distribution.

Models compared:
- **Spectral OOM** — Observable Operator Model / Weighted Finite Automaton
  learned via Hankel-matrix SVD (`spectral_oom.SpectralOOM`).
- **GDC** — Generative Dense Chain, the repository's non-parametric prefix-
  memorising model (`generative_dense_chain.GenerativeDenseChain`).

This complements the earlier binary-adder experiment: there, both models were
asked to forecast a deterministic Turing machine trace. Here the data is
genuinely stochastic, so MSE against the true posterior predictive
distribution is the right metric.

---

## 1. Set-up

### 1.1 Random HMM

Three random-HMM constructors in `random_hmm.py`, all producing valid
stochastic matrices:

- `random_dense_hmm(nS, nA, T_concentration=1, E_concentration=1)`: rows of
  T and E each drawn from `Dirichlet(concentration)`.
- `random_sparse_topology_hmm(nS, nA, fanout=2)`: each row of T supports
  exactly `fanout` successors, weights drawn from Dirichlet.
- `random_lowrank_hmm(nS, nA, rank)`: `T = mix @ V` where `V` has `rank`
  Dirichlet rows and `mix` mixes them — guarantees `rank(T) <= rank`.

The ground truth for any prefix `o_1..o_t` is computed exactly:
```
alpha_t = forward_filter(o_1..o_t)              # posterior over states
P(o_{t+h} = a | prefix) = (alpha_t @ T^h @ E)[a]
```

### 1.2 Models

#### Spectral OOM

Substring-count Hankel formulation (Hsu–Kakade–Zhang / Balle–Mohri):
```
H    [u, v] = # occurrences of substring u·v in training
H_a  [u, v] = # occurrences of substring u·a·v in training
basis = ε ∪ {all substrings of length 1..L appearing in training}
H = U Σ Vᵀ                                  (truncated SVD)
A_a = Uᵀ H_a V diag(1/σ)
α_0 = U[ε, :]
α_∞ = σ ⊙ V[ε, :]
```

Horizon-h one-step prediction marginalises over intermediate symbols:
```
state_t = α_0ᵀ · A_{o_1} · … · A_{o_t}
A_total = Σ_a A_a
score_h[a] = state_t · A_total^{h-1} · A_a · α_∞       # extended in spectral_oom.py
```

Because `score_h[a]` can be negative (the spectral factorisation does not
enforce probability), we tested **four** projection rules:

| mode | formula | notes |
|---|---|---|
| `clip` | `max(s, 0)` then normalise | standard in OOM literature |
| `abs` | `|s|` then normalise | preserves magnitude ranking |
| `softmax` | `softmax(s / max|s|)` | log-linear, always valid |
| `simplex` | Euclidean projection onto the simplex | Duchi et al. 2008 |

A quick sanity comparison on 5 random HMMs showed `softmax` to be the most
consistent; `clip` was catastrophic on small alphabets (losing a whole mass
to clipping). The main sweep reports **both `clip` (vanilla) and `softmax`
(projected)** as the OOM variants.

#### GDC

Used with `alpha=0.7, theta=0.2, beta=0.1, transition_type='self_loop',
initial_dist='sequence_starts'`. These are looser than the adder settings —
HMM data has genuine emission noise and no natural "step ordering" so a
moderate self-loop + non-zero beta is appropriate.

To get a symbol distribution at horizon h from GDC's state distribution:
```
state_dist_h = gdc.forecast(state_dist, n_steps=h)
symbol_dist[a] = Σ_{i : states[i] = a} state_dist_h[i]
```
This works because each GDC hidden state is tied to one observed scalar.

### 1.3 Metric

Per prefix, at each horizon h:
```
MSE = mean_a (predicted[a] - true[a])²
```
Averaged across 100 test prefixes × 3 HMM seeds per configuration. The
**uniform** baseline (`1/nA`) is reported alongside as a floor.

### 1.4 Training / Test

- Training: 200 sequences of length 50, sampled from the ground-truth HMM.
- Test: 100 independent length-20 prefixes from the same HMM.
- Horizons evaluated: h ∈ {1, 2, 5, 10, 20}.

---

## 2. Hypotheses and Predictions

Before running the sweep, four hypotheses were written down:

- **H1 — Low-rank transition matrices favour OOM.** Because OOM's SVD
  truncation natively represents low-rank operators, giving a well-
  parameterised model when the true `T` is low-rank.
- **H2 — Near-deterministic emissions favour GDC.** When each hidden state
  emits essentially one symbol, the observed prefix *is* the hidden state
  chain, and GDC's prefix memorisation is near-optimal.
- **H3 — Horizon effects.** Conjectured OOM would degrade more gracefully
  because it applies operator powers directly, whereas GDC's smoothed
  transition matrix may drift from truth over many steps.
- **H4 — Sparse topologies favour GDC.** Fewer distinct prefixes →
  easier memorisation; dense topology → more equivalence classes, harder
  for GDC.

Three of four predictions turned out to be wrong. See §4.

---

## 3. How to Reproduce

From `hmm_comparison/`:

```bash
python -u run_main_sweep.py         # 243 configs, ~25 min, writes main_sweep_results.csv
python -u run_hypothesis_tests.py   # ~5 min total, writes h{1,2,4}_*.csv
python -u make_plots.py             # fig1..fig5 PNGs
```

All randomness is seeded deterministically from `(seed, nS, nA)` tuples so
re-runs produce byte-identical CSV rows. Only numpy and matplotlib are
required.

---

## 4. Results

### 4.1 Main sweep (nS × nA grid, 3 seeds, 5 horizons)

Mean MSE across all 81 (nS, nA) configurations × 3 seeds:

| h | OOM-clip | OOM-softmax | **GDC** | Uniform |
|---|---|---|---|---|
| 1  | 0.0949 | 0.0327 | **0.0074** | 0.0100 |
| 2  | 0.0919 | 0.0347 | **0.0022** | 0.0097 |
| 5  | 0.0904 | 0.0338 | **0.00071** | 0.0097 |
| 10 | 0.0901 | 0.0341 | **0.00019** | 0.0096 |
| 20 | 0.0897 | 0.0341 | **0.000040** | 0.0096 |

`fig1_horizon_curves.png` plots these curves on a log y-axis.

**GDC wins the overall sweep by a wide margin**, and the margin grows with
horizon. Two clean observations:

- GDC's MSE drops ~200× from h=1 to h=20. As h grows, the true posterior
  predictive converges to the emission marginal under T's stationary
  distribution, which GDC approximates well because its smoothed transitions
  drive its state distribution toward uniform-over-hidden-states → the
  training's overall emission histogram.
- **OOM's MSE is flat across horizon**. The matrix `A_total = Σ_a A_a` is
  *not* stochastic and its powers don't converge to a useful stationary
  projection — the OOM state drifts without mixing, so information about
  "what happens many steps from now" is effectively lost.

### 4.2 OOM wins, when it wins

Of the 243 (nS, nA, seed) configurations, OOM-softmax beats GDC at **h=1**
in **62 cases**, and at h≥2 in **0 cases**.

Per-(nS, nA) wins at h=1 (out of 3 seeds each):
```
        nA=2 nA=3 nA=4 nA=5 nA=6 nA=7 nA=8 nA=9 nA=10
 nS=2     0    0    0    0    0    0    0    0    1
 nS=3     0    0    0    0    0    0    0    3    3
 nS=4     0    0    0    0    0    1    0    0    3
 nS=5     0    0    0    0    0    1    0    3    3
 nS=6     0    0    0    0    0    0    2    2    3
 nS=7     0    0    0    0    0    0    3    3    3
 nS=8     0    0    0    0    0    0    3    3    3
 nS=9     0    0    0    0    0    2    3    3    3
 nS=10    0    0    0    0    0    0    2    3    3
```

OOM's only regime of dominance is **high-alphabet**, **high-state** HMMs
(nA ≥ 8 and nS ≥ 6) at **horizon 1 only**. `fig2_grid_ratio.png` visualises
log(MSE_OOM / MSE_GDC) across the grid; the blue (OOM-better) cells cluster
in the upper-right corner.

### 4.3 H1 — Low-rank T should favour OOM

**Rejected.** With nS=10, nA=6 and rank(T) swept 1..10 (5 seeds each),
MSE is essentially flat for both models:

| rank | OOM h=1 | GDC h=1 | OOM h=5 | GDC h=5 |
|---|---|---|---|---|
| 1  | 0.0118 | 0.0086 | 0.0107 | 0.00096 |
| 5  | 0.0110 | 0.0081 | 0.0111 | 0.00086 |
| 10 | 0.0105 | 0.0083 | 0.0108 | 0.00076 |

See `fig3_h1_rank.png`. Two likely reasons:
1. The SVD-derived OOM rank is chosen from the *Hankel* spectrum, not the
   true T rank, and the Hankel has plenty of its own noise from finite
   sampling. The 200 × 50 training stream is enough to populate most of the
   Hankel regardless of T's rank.
2. GDC does not "know" about rank at all, yet its performance is rank-
   independent too — because its error is dominated by smoothing bias, not
   the underlying stochastic rank of T.

### 4.4 H2 — Near-deterministic emissions should favour GDC

**Supported.** With nS=6, nA=6 and emission concentration c swept from
0.05 (near-deterministic) to 10 (near-uniform), 5 seeds each:

| c | OOM h=1 | GDC h=1 | OOM/GDC ratio |
|---|---|---|---|
| 0.05 | 0.0350 | 0.00520 | 6.7× |
| 0.3  | 0.0182 | 0.00775 | 2.3× |
| 1.0  | 0.0117 | 0.00847 | 1.4× |
| 3.0  | 0.0104 | 0.00790 | 1.3× |
| 10.0 | 0.0097 | 0.00839 | 1.2× |

Monotone: when emissions are sharp, the observed prefix ≈ the hidden state
chain, exactly the regime GDC is designed for. The OOM/GDC ratio degrades by
a factor of ~5 as we move from uniform to near-deterministic emissions.
`fig4_h2_emission.png`.

### 4.5 H3 — Horizon scaling should favour OOM

**Rejected in the opposite direction.** As shown in §4.1, GDC's MSE
*collapses* at long horizons while OOM's is flat. The intuition that drove
H3 (clean operator extrapolation vs noisy smoothed transitions) was wrong:
under finite-rank truncation, OOM's `A_total^h` doesn't converge to
anything sensible because the spectral radius of `A_total` isn't pinned to
1.

### 4.6 H4 — Sparse topology should favour GDC

**Rejected.** GDC has lower MSE than OOM on *both* topologies — it wins in
absolute terms regardless. The hypothesis was about *relative* position: H4
predicted that GDC's lead over OOM would be **larger** on sparse-topology
HMMs (where prefix memorisation should really pay off). The data shows the
opposite — the OOM/GDC gap *narrows* under sparse topology.

With nS=8, nA=4, 5 seeds each:

| topology | OOM h=1 | GDC h=1 | ratio | OOM h=5 | GDC h=5 | ratio |
|---|---|---|---|---|---|---|
| dense   | 0.0347 | 0.00766 | **4.5×** | 0.0353 | 0.00064 | **55×** |
| sparse2 | 0.0371 | 0.01263 | **2.9×** | 0.0285 | 0.00332 | **8.6×** |

Why the ratio shrinks on sparse:
1. GDC's absolute MSE *increases* on sparse (0.008 → 0.013 at h=1; 0.0006
   → 0.003 at h=5). Sparse-topology HMMs emit long runs that cycle within
   small subsets of states, so the true predictive distribution peaks on a
   few symbols. GDC's smoothed transitions (`alpha=0.7, theta=0.2`) push
   mass uniformly across its ~10,000 hidden states → away from those peaks
   toward uniform-over-emissions, which is the wrong direction.
2. OOM's MSE actually *improves slightly* on sparse at h=5 (0.035 → 0.029).
   Sparse transitions have lower effective rank, which the Hankel SVD can
   capture more cleanly.

Combined, the sparse regime is the one topology where GDC's smoothing-bias
failure mode overlaps with OOM's mild advantage — so the gap narrows even
though GDC still wins. `fig5_h4_topology.png`.

> **Note on non-monotonic MSE vs horizon.** On `sparse2`, all three rows
> (OOM, GDC, uniform baseline) have lower MSE at h=20 than at h=1.
> Intuition says forecast error should grow with horizon — but for
> *distributional* MSE against an ergodic HMM's true posterior predictive,
> it can shrink. As h → ∞, `P(o_{t+h} | prefix) →` stationary emission
> marginal independent of prefix. When that marginal is close to uniform
> (random HMMs with moderate Dirichlet concentration), the true
> distribution collapses onto a predictable point as h grows. The uniform
> baseline going from 0.0137 at h=1 to 0.0084 at h=20 on sparse (a 39%
> reduction) is the diagnostic — uniform can't be "learning," so the only
> explanation is that truth itself is moving toward uniform. The effect is
> strongest on sparse-topology HMMs because they mix slowly, so the h=1
> distribution is far from stationary and has plenty of room to collapse.
> Dense HMMs mix fast — their h=1 distribution is already near stationary
> — so uniform-baseline MSE is flat across horizons (0.0122 → 0.0122).

---

## 5. Discussion

### 5.1 Why GDC wins so decisively

GDC's hidden-state count on this data is ~10,000 per configuration (200
sequences × 50 timesteps). That is a *lot* of capacity. Because each
hidden state is tied to a specific observed scalar, GDC effectively runs a
non-parametric estimator over observed n-grams (with the smoothing knobs
controlling how that estimator generalises). On enough training data
relative to the alphabet size, this beats a rank-50-ish spectral
factorisation.

Conversely, GDC's smoothed transitions (`alpha, theta` on a fixed training-
order chain) impose a strong prior: mass flows uniformly-ish across the
hidden-state chain. At long horizons, that prior effectively predicts the
marginal emission distribution — which *is* the correct long-horizon limit
for ergodic HMMs. So GDC is accidentally well-calibrated for horizon
extrapolation on dense-transition HMMs.

### 5.2 Why OOM plateaus

Two separable issues:

1. **Sign / probability projection.** The clip mode loses entire symbols;
   even softmax is a distortion. None of the tested projections recovers a
   proper probability when the underlying spectral operator has
   non-negligible negative scores.
2. **Non-ergodic operator powers.** `A_total = Σ_a A_a` is not stochastic;
   its powers can grow or shrink without converging to a rank-1 projection.
   State-renormalisation in the forward pass preserves argmax but not
   scale-calibration. This is why horizon scaling is flat.

An EM refinement (use spectral OOM for initialisation, then run
expectation-maximisation on the training sequences to project parameters
onto proper probabilities) is the textbook remedy and was not attempted
here.

### 5.3 When would OOM realistically beat GDC?

The sweep identifies one niche: **large alphabets + medium-to-large state
counts, short-horizon prediction**. Intuitively:

- Large nA makes each prefix rarely-seen, so GDC has less to memorise and
  its prior hurts more.
- Short horizon avoids the OOM's non-ergodic power problem.

A second plausible niche not tested here: settings where training sequences
are *abundantly* long (so the Hankel is well-populated) but have very
high intrinsic rank (so GDC can't memorise it all). The current sweep uses
short training sequences (length 50), which under-exercises this advantage.

---

## 6. File Map

All files in `hmm_comparison/`.

- `random_hmm.py` — RandomHMM class, dense/sparse/low-rank constructors.
- `model_wrappers.py` — `OOMForecaster`, `GDCForecaster`, `fit_oom`, `fit_gdc`.
- `evaluation.py` — `mse_at_horizons`, `uniform_baseline_mse`,
  `stationary_baseline_mse`.
- `run_main_sweep.py` — the 243-config grid sweep.
- `run_hypothesis_tests.py` — H1, H2, H4 scripts.
- `make_plots.py` — generates the 5 figures.

**Raw results**
- `main_sweep_results.csv` — 1,215 rows (243 configs × 5 horizons).
- `h1_rank_results.csv` — 250 rows.
- `h2_emission_concentration_results.csv` — 125 rows.
- `h4_topology_results.csv` — 50 rows.
- `main_sweep.log`, `hypotheses.log` — stdout from the runs.

**Figures**
- `fig1_horizon_curves.png` — MSE vs horizon (all models).
- `fig2_grid_ratio.png` — log(OOM/GDC) heatmap over the (nS, nA) grid.
- `fig3_h1_rank.png` — H1 result.
- `fig4_h2_emission.png` — H2 result.
- `fig5_h4_topology.png` — H4 result.

One dependency note: `spectral_oom.py` (in the repo root) was extended with
`predict_next_scores(..., horizon=h)` and multiple `predict_next_probs`
projection modes (`clip`, `abs`, `softmax`, `simplex`) specifically for
this experiment. The Turing-machine adder experiment continues to work
unchanged (defaults to `mode='clip'`).
