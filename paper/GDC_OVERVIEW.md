# The Generative Dense Chain (GDC): a one-page overview

## What it is

The **Generative Dense Chain** is a hidden markov model with one hidden state per element in the training data. This lets it function as a non-parametric, training-free
next-symbol predictor for sequences of any form. 

So fart this model has been tested on:

- **Continuous time-series forecasting** — M4 (100k series across 6
  frequencies), and the dysts 131-system chaotic attractor benchmark.
- **Discrete sequence learning** — synthetic HMM forecasting (single
  and product/factored), the PAutomaC competition (Verwer et al. 2014)
  for probabilistic-automaton learning.
- **algorithimic trace learning** - Turing-machine
  algorithmic-trace prediction (parity, increment, reverse, binary
  adder, Dyck-1).


Given a corpus $x_1, x_2, \ldots, x_N$ (the "state space"), a GDC is
a $N$-state hidden Markov chain whose state $i$ is **tied** to the
observation $x_i$ — discretely for symbolic data, or
through a continous distribution (ex. Gaussian) for continuous data.
Given this setup, simple parameterization of emission and transitions can be constructed to provide excellent performance without training.
The work below often uses a fixed three-parameter form: stay in place
with probability $\theta$, advance one step (the "natural successor")
with probability $\alpha$, or diffuse uniformly with probability
$1 - \alpha - \theta$, and emission spread term $\beta$.

Prediction from a lookback window $p_1, \ldots, p_L$ is a forward pass through this chain. Iterating the transition
matrix $T$ steps forward gives the predictive distribution over the
state space at horizon $T$


`terminal_behavior='absorb'` mode: after every training sequence an absorbing state is added, so probability mass that propagates past the trained
manifold can no longer diffuse. This allows for clean integration of subsequences shorter and longer into training data.



Free parameters: $\sigma$ / $\beta$ (Gaussian bandwidth for continuous, probability to emit a different token for discrete), $\alpha$
(advance rate, typically 0.99–1.0), $\theta$ (self-loop, typically 0).
For continuous tasks, two recipes — `raw` and `diff` — are val-picked
per dataset.

## Interpretability

Because GDC is a literal Bayesian inference over an $N$-state hidden
Markov chain whose states are tied to specific training-corpus
positions, **every step of a forecast produces a full posterior
distribution over training-corpus positions** $P(\text{state}_t \mid
x_{1..t})$. At any timestep you can ask: "which positions in my
training data is the model currently treating as most likely matches?"
It surfaces:

- **Coverage gaps** — when the posterior is diffuse (no training
  position dominates), the model is signaling "I haven't seen
  anything like this." Useful for OOD detection.
- **Failure modes** — when the posterior concentrates on the wrong
  training positions, the prediction will be wrong in a predictable
  way, and the wrong positions reveal which surface feature the model
  latched onto.
- **Provenance** — every forecast value is a weighted average over
  specific training observations, with weights given by the posterior.
  Forecasts are fully attributable to source data.


## Benchmark results

### Discrete sequence learning

Noted comparisons (used across the discrete-sequence experiments below):

- **Parrot** — context-parroting baseline (Zhang & Gilpin
  2025); the original paper just set k to 1 with a fixed lookup length. In this work the comparison is a validation tuned top-$K$ kNN over multiple fixed-length prefixes.
- **CHMM** — cloned HMM (Dileep George); overcomplete clone-per-emission
  HMM, EM-trained to recover transition structure.
- **ALERGIA / ALERGIA+** — passive PDFA learner with Hoeffding-bound
  state-merging consistency check. ALERGIA+ refers to the FlexFringe
  implementation with sinks, pooling, and low-frequency-count handling
  (Verwer & Hammerschmidt 2022).
- **MDI** — Minimum Description Length state-merging PDFA learner;
  also from the FlexFringe family.
- **HPYLM** — fixed-depth Hierarchical Pitman-Yor language model;
  closest analog to the Sequence Memoizer (Wood et al. 2009).
- **PPM-D** — absolute-discount n-gram backoff compressor (Howard 1993).
- **KN-3** — interpolated Kneser-Ney trigram.
- **Freq** — unigram baseline that predicts the empirical training
  symbol frequency at every step (no prefix conditioning).

#### HMM forecasting (Table 7)

Excess perplexity ($2^{\,\mathrm{CE} - \mathrm{floor}}$, lower bound
1.000) on synthetic-HMM next-token forecasting at horizons
$h \in \{1, \ldots, 5\}$ across four structurally distinct regimes.
All regimes use $n_S{=}20$ hidden states and $n_A{=}4$ symbols;
$N{=}25$ training sequences of length 50 (1{,}250 chars per HMM);
100 test prefixes of length 20; 20 test HMMs per regime, with
per-method config selection done leakage-free on a disjoint set of 20
validation HMMs (different random draws). Each regime is constructed to vary the dominant structural
property:

- **cyclic** ($\mathrm{advance\_prob}{=}0.95$): each state $i$ transitions
  to $(i{+}1) \bmod n_S$ with probability 0.95, emits symbol
  $i \bmod n_A$ deterministically. Multiple cycle positions share each
  symbol; the prefix disambiguates which position you are at.
- **reset_chain** ($\mathrm{advance\_prob}{=}0.90$,
  $\mathrm{reset\_prob}{=}0.05$): linear chain with periodic resets to
  state 0. Resets break long-range correlation.
- **bimodal** ($\mathrm{sticky\_prob}{=}0.95$): two equal-size state
  clusters with disjoint emission supports; cross-cluster transition
  mass = 0.05.
- **sparse topology** (fanout=2, $E_{\mathrm{conc}}{=}0.1$): each row
  of $T$ is supported on exactly two random successors; emissions
  drawn from $\mathrm{Dirichlet}(0.1)$.

Methods: Freq (unigram baseline), KN-3, PPM-D, ALERGIA, Parrot
(Zhang & Gilpin 2025), HPYLM (fixed-depth Sequence Memoizer of Wood et
al. 2009), GDC (ours), CHMM. Each method picks its best config per
$(\text{regime}, h)$ by lowest mean excess PP on the validation HMMs,
reported on the test HMMs. GDC tunes a separate
$\alpha_{\mathrm{forecast}}$ axis: the prefix forward pass uses the
swept $\alpha$, while the forecast roll-out can override to
$\alpha_{\mathrm{forecast}}{=}1.0$, $\theta_{\mathrm{forecast}}{=}0$
(deterministic walk-forward through the chain); this dual variant is a
candidate alongside single-$\alpha$ in the grid. GDC's grid is a
principled 32-config set ($\alpha_{\mathrm{ctx}} \in \{0.3, 0.5, 0.7,
0.9\}$, $\theta \in \{0, 0.1\}$, $\beta \in \{0, 0.005\}$, $\alpha_{\mathrm{fc}}
\in \{\alpha_{\mathrm{ctx}}, 1.0\}$) that reproduces a 464-config sweep to
within 0.004 excess-PP per cell. Full grids in Table 7's caption.

Excess PP per regime (best per column **bold**):

| | $h{=}1$ | $h{=}2$ | $h{=}3$ | $h{=}4$ | $h{=}5$ |
|---|---:|---:|---:|---:|---:|
| **(a) cyclic** | | | | | |
| Freq | 2.878 | — | — | — | — |
| ALERGIA | 1.098 | 1.082 | 1.085 | 1.092 | 1.095 |
| KN-3 | 1.044 | 37.946$^\dagger$ | 42.756$^\dagger$ | 36.684$^\dagger$ | 1.206 |
| Parrot | 1.044 | 1.041 | 1.040 | 1.039 | 1.040 |
| PPM-D | 1.034 | 1.044 | 1.055 | 1.072 | 1.102 |
| GDC | 1.033 | 1.031 | 1.032 | 1.034 | 1.036 |
| HPYLM | 1.032 | 1.028 | 1.029 | 1.040 | 1.042 |
| CHMM | **1.027** | **1.025** | **1.023** | **1.022** | **1.023** |
| **(b) reset_chain** | | | | | |
| Freq | 2.581 | — | — | — | — |
| ALERGIA | 1.074 | 1.057 | 1.052 | 1.050 | 1.049 |
| Parrot | 1.042 | 1.037 | 1.032 | 1.027 | 1.024 |
| KN-3 | 1.035 | 20.721$^\dagger$ | 13.644$^\dagger$ | 17.581$^\dagger$ | 1.370 |
| HPYLM | 1.029 | 1.028 | 1.041 | 1.031 | 1.047 |
| PPM-D | 1.029 | 1.068 | 1.109 | 1.162 | 1.212 |
| CHMM | 1.028 | **1.022** | **1.019** | **1.016** | **1.015** |
| GDC | **1.026** | 1.026 | 1.021 | 1.018 | 1.019 |
| **(c) bimodal** | | | | | |
| Freq | 1.663 | — | — | — | — |
| KN-3 | 1.027 | 1.072 | 1.134 | 1.208 | 1.289 |
| HPYLM | 1.021 | 1.012 | 1.023 | 1.025 | 1.021 |
| ALERGIA | 1.020 | 1.010 | **1.008** | **1.008** | **1.008** |
| PPM-D | 1.019 | 1.020 | 1.045 | 1.085 | 1.131 |
| CHMM | 1.016 | 1.010 | 1.009 | **1.008** | **1.008** |
| Parrot | 1.014 | 1.010 | 1.013 | 1.013 | 1.014 |
| GDC | **1.007** | **1.009** | 1.011 | 1.011 | 1.012 |
| **(d) sparse topology** | | | | | |
| Freq | 1.556 | — | — | — | — |
| ALERGIA | 1.330 | 1.249 | 1.164 | 1.111 | 1.078 |
| KN-3 | 1.191 | 3.604$^\dagger$ | 1.740$^\dagger$ | 2.588$^\dagger$ | 1.567$^\dagger$ |
| Parrot | 1.170 | 1.119 | 1.086 | 1.059 | 1.042 |
| PPM-D | 1.161 | 1.240 | 1.262 | 1.295 | 1.308 |
| GDC | 1.158 | 1.117 | 1.076 | 1.053 | 1.040 |
| CHMM | 1.150 | **1.096** | **1.057** | **1.040** | **1.027** |
| HPYLM | **1.147** | 1.112 | 1.096 | 1.086 | 1.088 |

$^\dagger$ KN-3 is a 1-step trigram; we reuse its single-step
prediction at every horizon, which fails on near-deterministic
regimes at $h{>}1$ where the trigram assigns near-zero probability to
symbols that appear naturally at later horizons.

GDC's per-regime results:

- **cyclic**: GDC at 1.033 ($h{=}1$), 1.031–1.035 across $h{=}2{-}5$.
  CHMM holds the column-best at every horizon (1.027 at $h{=}1$); GDC
  and HPYLM trail by ~0.005.
- **reset_chain**: GDC wins $h{=}1$ outright (1.026, vs CHMM 1.028);
  CHMM retakes the column-best at $h{\geq}2$, GDC second throughout.
- **bimodal**: GDC wins $h{=}1$ and $h{=}2$ (1.007, 1.009); ALERGIA and
  CHMM tie for the column-best at $h{=}3{-}5$ (1.008).
- **sparse topology**: HPYLM is column-best at $h{=}1$ (1.147), with
  GDC (1.158) just behind CHMM (1.150); CHMM takes $h{\geq}2$. GDC,
  CHMM, Parrot bunch within 0.02 here.

The Freq baseline sits at 1.556–2.878 ($h{=}1$) across the four
regimes, substantially above the structured methods.

#### HMM forecasting: data scaling (Table 13)

Same four regimes as Table 7 (cyclic, reset_chain, bimodal, sparse
fanout-2), varying the number of training sequences
$N \in \{1, 3, 5, 10, 25\}$ (each of length 50, so 50–1{,}250 training
chars per HMM). $h{=}1$ excess perplexity; 20 test HMMs (leakage-free
selection on 20 disjoint validation HMMs); 100 test prefixes of length
20 per cell. $N{=}25$ reproduces Table 7's $h{=}1$ column exactly.
**Bold** =
best per column.

| Method | $N{=}1$ | $N{=}3$ | $N{=}5$ | $N{=}10$ | $N{=}25$ |
|---|---:|---:|---:|---:|---:|
| **(a) cyclic** | | | | | |
| Freq | 2.890 | 2.885 | 2.882 | 2.879 | 2.878 |
| KN-3 | 1.133 | 1.147 | 1.119 | 1.081 | 1.044 |
| PPM-D | 1.270 | 1.144 | 1.102 | 1.055 | 1.034 |
| ALERGIA | 2.822 | 2.357 | 1.893 | 1.321 | 1.098 |
| Parrot | 1.685 | 1.085 | 1.063 | 1.052 | 1.044 |
| HPYLM | 1.158 | 1.096 | 1.075 | 1.047 | 1.032 |
| GDC | 1.100 | **1.061** | **1.049** | 1.040 | 1.033 |
| CHMM | **1.086** | 1.063 | 1.053 | **1.038** | **1.027** |
| **(b) reset_chain** | | | | | |
| Freq | 2.597 | 2.592 | 2.588 | 2.585 | 2.581 |
| KN-3 | 1.144 | 1.148 | 1.113 | 1.070 | 1.035 |
| PPM-D | 1.275 | 1.134 | 1.091 | 1.050 | 1.029 |
| ALERGIA | 2.627 | 2.107 | 1.622 | 1.210 | 1.074 |
| Parrot | 1.611 | 1.090 | 1.064 | 1.051 | 1.042 |
| HPYLM | 1.207 | 1.099 | 1.073 | 1.045 | 1.029 |
| GDC | **1.106** | **1.058** | **1.044** | **1.036** | **1.026** |
| CHMM | 1.117 | 1.065 | 1.054 | 1.038 | 1.028 |
| **(c) bimodal** | | | | | |
| Freq | ~7×10²$^\ddagger$ | 1.768 | 1.708 | 1.678 | 1.663 |
| KN-3 | ~1×10⁵$^\ddagger$ | 1.099 | 1.082 | 1.061 | 1.027 |
| PPM-D | 1.367 | 1.148 | 1.108 | 1.062 | 1.019 |
| ALERGIA | 1.838 | 1.585 | 1.368 | 1.123 | 1.020 |
| Parrot | 1.860 | 1.098 | 1.064 | 1.025 | 1.014 |
| HPYLM | **1.323** | 1.115 | 1.079 | 1.049 | 1.021 |
| GDC | ~1×10⁵$^\ddagger$ | **1.050** | **1.032** | **1.020** | **1.007** |
| CHMM | 1.343 | 1.149 | 1.095 | 1.040 | 1.016 |
| **(d) sparse topology** | | | | | |
| Freq | 1.659 | 1.577 | 1.559 | 1.564 | 1.556 |
| KN-3 | 1.594 | 1.337 | 1.273 | 1.241 | 1.191 |
| PPM-D | 1.460 | 1.280 | 1.241 | 1.239 | 1.161 |
| ALERGIA | 1.983 | 1.787 | 1.573 | 1.414 | 1.330 |
| Parrot | 1.450 | 1.311 | 1.263 | 1.216 | 1.170 |
| HPYLM | **1.420** | 1.272 | 1.248 | 1.199 | **1.147** |
| GDC | 1.524 | **1.265** | **1.219** | **1.188** | 1.158 |
| CHMM | 1.504 | 1.379 | 1.313 | 1.259 | 1.150 |

$^\ddagger$ The bimodal $N{=}1$ column is degenerate for the methods
whose predictions are sharp on a single training sequence (GDC's
val-pick selects $\alpha_{\mathrm{fc}}{=}1$; KN-3 and Freq have zero
counts for the unseen cluster): one length-50 sample from a sticky
(0.95) bimodal HMM visits only one of the two clusters, so the other
cluster's symbols get ~0 probability and the test perplexity blows up.
The smoothing methods (CHMM, ALERGIA, Parrot, HPYLM, PPM-D) stay
bounded; from $N{=}3$ onward every method is well-behaved.

GDC's per-regime results across the data-scale axis:

- **cyclic**: GDC is column-best at $N \in \{2, 3\}$ (1.061 / 1.049);
  CHMM holds $N{=}1$ (1.086, by a 0.014 margin) and $N \in \{10, 25\}$
  (1.038 / 1.027).
- **reset_chain**: GDC is column-best at **every** $N$ (1.106 → 1.026),
  the cleanest data-scaling win.
- **bimodal**: $N{=}1$ is degenerate (see above); from $N{=}3$ GDC
  holds the column-best at every scale (1.050 / 1.032 / 1.020 / 1.007).
- **sparse topology**: HPYLM is column-best at $N{=}1$ and $N{=}25$
  (1.420 / 1.147); GDC takes the middle scales $N \in \{3, 5, 10\}$
  (1.265 / 1.219 / 1.188).

GDC takes **14 of the 20 column-best cells** (cyclic 2 + reset 5 +
bimodal 4 + sparse 3), concentrated at the low-data end; CHMM and HPYLM
retake several $N{=}25$ cells (and cyclic $N{=}1$, by a hair).

#### Product-HMM data-scaling experiment (Table 12)

A focused experiment on the **near-deterministic, structurally rich**
end of the HMM regime spectrum. We construct a *product HMM* by
sampling 3 independent component HMMs, each with $n_S{=}3$ hidden
states and a ternary alphabet ($n_A{=}3$); per-component transitions
are drawn from $\mathrm{Dirichlet}(0.1)$ (near-deterministic rows) and
per-component emissions are state-preferred ($E[i,\,i\bmod n_A]\ge
0.7$). The components are combined by Kronecker product into a single
product HMM with 27 hidden states and a 27-symbol alphabet. The data-
generating process is structurally rich (independent latent factors,
near-deterministic dynamics, sharp emissions) but stochastic — so it
sits between the easy-mixing dense regimes of Table 7 and the
algorithmic state-propagation tasks of Tables 8–9.

We then sweep training-data size at $N \in \{40, 160, 640\}$ sequences
(800 / 3,200 / 12,800 training chars), holding the test set fixed at
20 sequences of length 20. Excess perplexity reported at horizons
$h{=}1{,}\ldots{,}5$, over 3 test product-HMM seeds with leakage-free
config selection on 3 disjoint validation seeds. 

The GDC row is the single **fixed** config $\alpha{=}0.85$,
$\theta{=}0.005$, $\beta{=}0.075$ (chosen a priori, applied at every
scale — leakage-free by construction); CHMM (best $K$) and Parrot (best
$L,K$) are val-picked per horizon on disjoint validation seeds. **Bold**
= best per column. KN-3$^\dagger$ reuses its single-step prediction at
every horizon and blows up at $h \in \{2,4\}$.

| Method | $h{=}1$ | $h{=}2$ | $h{=}3$ | $h{=}4$ | $h{=}5$ |
|---|---:|---:|---:|---:|---:|
| **(a) $N{=}40$ sequences = 800 chars (1$\times$)** | | | | | |
| GDC ($\alpha{=}0.85$, $\theta{=}0.005$, $\beta{=}0.075$) | 1.477 | 1.415 | 1.420 | 1.416 | 1.434 |
| CHMM (best $K$) | 1.984 | 1.553 | **1.281** | **1.264** | **1.240** |
| Parrot (best $L,K$) | **1.297** | **1.325** | 1.366 | 1.368 | 1.358 |
| Freq (unigram) | 1.615 | — | — | — | — |
| HPYLM ($D{=}3$) | 1.611 | 1.551 | 1.499 | 1.479 | 1.436 |
| PPM-D ($D{=}3$) | 1.837 | 1.744 | 1.543 | 1.563 | 1.541 |
| KN-3$^\dagger$ | 1.852 | 3.517 | 1.888 | 3.237 | 1.927 |
| **(b) $N{=}160$ sequences = 3,200 chars (4$\times$)** | | | | | |
| GDC (same config) | **1.202** | **1.197** | **1.202** | 1.217 | 1.225 |
| CHMM (best $K$) | 1.607 | 1.249 | 1.209 | **1.210** | **1.217** |
| Parrot (best $L,K$) | 1.243 | 1.287 | 1.312 | 1.335 | 1.324 |
| Freq | 1.462 | — | — | — | — |
| HPYLM | 1.441 | 1.376 | 1.377 | 1.310 | 1.331 |
| PPM-D | 1.634 | 1.484 | 1.480 | 1.359 | 1.468 |
| KN-3$^\dagger$ | 1.382 | 3.182 | 1.462 | 2.818 | 1.517 |
| **(c) $N{=}640$ sequences = 12,800 chars (16$\times$)** | | | | | |
| GDC (same config) | **1.183** | 1.192 | 1.208 | 1.209 | 1.217 |
| CHMM (best $K$) | 1.249 | **1.181** | **1.130** | **1.172** | **1.179** |
| Parrot (best $L,K$) | 1.197 | 1.250 | 1.262 | 1.277 | 1.292 |
| Freq | 1.458 | — | — | — | — |
| HPYLM | 1.341 | 1.330 | 1.268 | 1.259 | 1.272 |
| PPM-D | 1.394 | 1.385 | 1.288 | 1.286 | 1.303 |
| KN-3$^\dagger$ | 1.258 | 3.064 | 1.338 | 2.683 | 1.393 |

GDC's results across the data-scale axis:

- **$1\times$ ($N{=}40$)**: GDC (fixed config) at 1.477 / 1.415 / 1.420
  / 1.416 / 1.434. Parrot is column-best at $h{=}1{-}2$ (1.297 /
  1.325); CHMM is column-best at $h{=}3{-}5$. GDC second at short
  horizons — the smallest scale is where the fixed config is least
  favored.
- **$4\times$ ($N{=}160$)**: GDC column-best at $h{=}1{-}3$ (1.202 /
  1.197 / 1.202); CHMM retakes $h{=}4{-}5$.
- **$16\times$ ($N{=}640$)**: GDC column-best at $h{=}1$ (1.183, with
  Parrot 1.197 close); CHMM takes $h{\geq}2$. GDC improves
  monotonically with scale (1.477 → 1.202 → 1.183 at $h{=}1$).

The fixed config $\alpha{=}0.85$, $\theta{=}0.005$, $\beta{=}0.075$ is
**validation-best at $16\times$ and near-best at the smaller scales**
(where validation marginally prefers $\alpha{=}0.7$); reporting one
config across all scales is the leakage-free, no-per-scale-tuning
choice.


#### PAutomaC competition (Table 14)

The PAutomaC competition (Verwer et al. 2014) is a 48-problem
benchmark for probabilistic-automaton learning. Each problem provides
training sequences sampled from a hidden target machine (a PDFA, HMM,
or PNFA) and a held-out test set of unique traces. Submissions assign
probabilities to each test trace; scoring is the perplexity
$2^{-\sum_x p_T(x) \log_2 p_M(x)}$, lower-bounded by the entropy
$2^{H(p_T)}$ of the target.

Six methods compared per-problem:

- **GDC val-tuned (leakage-free)**: selects among 7 GDC configs per
  problem (2 single-$\alpha$ + 5 dual-$\alpha$ at
  $\alpha_{\mathrm{fc}}{=}0.9999$, $\alpha_{\mathrm{ctx}} \in
  \{0.30, 0.50, 0.70, 0.85, 0.95\}$) by lowest **held-out negative
  log-likelihood** on a 20% split of the training sequences. The test
  set and its solution probabilities are never used for selection; the
  chosen config is then refit on full train and scored on test. The
  competition ships no validation split, so this held-out-NLL protocol
  is what makes the selection leakage-free.
- **GDC fixed**: single best config across the 48 problems — dual
  $\alpha_{\mathrm{ctx}}{=}0.85$, $\alpha_{\mathrm{fc}}{=}0.9999$, no
  per-problem tuning.
- **ALERGIA+**, **MDI**: Verwer & Hammerschmidt (2022) FlexFringe Table
  2. State-merging PDFA learners with sinks, pooling, low-frequency
  counts; untuned confidence-bound.
- **KN-3**: interpolated Kneser-Ney trigram, discount $d{=}0.75$.
- **Parrot**: best of $(L{=}2, K{=}5, \alpha_p{=}1.0)$ and
  $(L{=}4, K{=}25, \alpha_p{=}0.1)$ context-parroting configs.

Per-problem perplexity (lower = closer to entropy floor):

| P | floor | GDC val-tune (LF) | GDC fixed | ALERGIA+ | MDI | KN-3 | Parrot |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29.90 | 31.38 | **30.52** | 31.98 | 31.20 | 33.49 | 65.50 |
| 2 | 168.33 | 176.24 | 177.91 | **168.43** | 168.96 | 178.20 | 413.94 |
| 3 | 49.96 | 53.65 | 51.47 | 51.35 | **51.21** | 68.27 | 98.24 |
| 4 | 80.82 | 81.37 | 82.35 | 80.95 | **80.89** | 101.38 | 170.67 |
| 5 | 33.24 | 33.92 | 34.03 | **33.24** | 33.31 | 45.66 | 42.76 |
| 6 | 66.99 | 69.95 | 69.95 | **67.01** | 67.54 | 110.28 | 147.72 |
| 7 | 51.22 | 52.63 | 52.63 | **51.24** | 51.46 | 57.32 | 153.11 |
| 8 | 81.38 | 84.90 | 82.59 | 83.01 | **82.05** | 106.19 | 1087.44 |
| 9 | 20.84 | 21.53 | 22.17 | **20.85** | 20.99 | 71.63 | 356.49 |
| 10 | 33.30 | 36.10 | 34.39 | **33.65** | 35.04 | 45.43 | 102.81 |
| 11 | 31.81 | 33.46 | 33.03 | **31.84** | 33.56 | 38.17 | 286.37 |
| 12 | 21.66 | 22.28 | 22.15 | **21.68** | 22.49 | 25.25 | 115.60 |
| 13 | 62.81 | 63.54 | 64.83 | 64.76 | **62.87** | 158.07 | 238.03 |
| 14 | 116.79 | 119.14 | 120.23 | **116.84** | 117.13 | 125.84 | 191.08 |
| 15 | 44.24 | 47.07 | 45.96 | **45.10** | 46.80 | 48.34 | 112.23 |
| 16 | 30.71 | 31.37 | 31.00 | **30.72** | 30.78 | 41.22 | 73.60 |
| 17 | 47.31 | 50.34 | 48.61 | **48.03** | 51.13 | 51.91 | 415.21 |
| 18 | 57.33 | 58.01 | 57.87 | **57.33** | 57.39 | 65.97 | 113.53 |
| 19 | 17.88 | 18.32 | 18.03 | 17.97 | **17.92** | 21.14 | 37.29 |
| 20 | 90.97 | 103.51 | 101.19 | **92.36** | 98.61 | 109.08 | 148.49 |
| 21 | 30.52 | 36.49 | 35.63 | **35.25** | 37.31 | 46.95 | 239.37 |
| 22 | 25.98 | 26.79 | **26.22** | 26.56 | 26.61 | 29.92 | 193.27 |
| 23 | 18.41 | 18.71 | 18.52 | 18.49 | **18.47** | 20.82 | 33.06 |
| 24 | 38.73 | 40.09 | 40.09 | **38.73** | 38.91 | 48.27 | 90.75 |
| 25 | 65.74 | 73.34 | 70.03 | 67.26 | **66.83** | 86.36 | 149.52 |
| 26 | 80.74 | 84.26 | 84.26 | **80.89** | 83.52 | 213.23 | 364.23 |
| 27 | 42.43 | 44.32 | 43.70 | **42.46** | 43.49 | 46.98 | 259.17 |
| 28 | 52.74 | 58.36 | 54.10 | 53.77 | **53.55** | 65.50 | 181.62 |
| 29 | 24.03 | 24.76 | 24.76 | **24.20** | 24.58 | 43.98 | 35.21 |
| 30 | 22.93 | 24.16 | 23.46 | 23.47 | **23.33** | 25.32 | 47.99 |
| 31 | 41.21 | 42.86 | **42.01** | 42.08 | 42.27 | 46.33 | 74.45 |
| 32 | 32.61 | 33.24 | 33.25 | **32.62** | 32.65 | 53.04 | 65.81 |
| 33 | 31.87 | 32.49 | 32.75 | **31.96** | 32.64 | 34.56 | 75.46 |
| 34 | 19.96 | 21.05 | **20.99** | 25.99 | 26.50 | 23.66 | 78.58 |
| 35 | 33.78 | 35.08 | 34.96 | **33.80** | 36.81 | 42.87 | 339.33 |
| 36 | 37.99 | 38.64 | **38.24** | 38.87 | 38.29 | 39.69 | 52.22 |
| 37 | 20.98 | 21.08 | **21.03** | 21.19 | 21.11 | 21.10 | 22.54 |
| 38 | 21.45 | 21.70 | 21.69 | 21.84 | **21.49** | 21.70 | 35.10 |
| 39 | 10.00 | 10.19 | 10.13 | **10.00** | 10.05 | 10.28 | 15.84 |
| 40 | 8.20 | 8.39 | 8.33 | **8.26** | 8.52 | 8.88 | 13.15 |
| 41 | 13.91 | 13.98 | **13.94** | 14.02 | 13.98 | 14.01 | 16.03 |
| 42 | 16.00 | 16.32 | 16.19 | **16.01** | 16.05 | 16.45 | 29.85 |
| 43 | 32.64 | 32.85 | 32.91 | 33.14 | **32.85** | 32.85 | 36.99 |
| 44 | 11.71 | 12.00 | **11.97** | 12.70 | 12.04 | 12.09 | 14.64 |
| 45 | 24.04 | 24.22 | 24.42 | **24.04** | 24.24 | 24.14 | 32.14 |
| 46 | 11.98 | 12.37 | **12.32** | 12.50 | 12.89 | 12.32 | 26.86 |
| 47 | 4.12 | 4.16 | 4.14 | **4.12** | 4.13 | 4.68 | 8.49 |
| 48 | 8.04 | 8.32 | 8.33 | **8.04** | 8.24 | 8.45 | 14.99 |

Summary statistics on the gap above the entropy floor (gap = score $-$ floor; 48 problems):

| Method | median gap | mean gap | max gap | wins |
|---|---:|---:|---:|---:|
| **ALERGIA+** | **0.12** | **0.63** | **6.04** | **28** |
| MDI | 0.37 | 1.09 | 7.64 | 11 |
| GDC val-tune (LF) | 0.73 | 1.85 | 12.54 | 0 |
| GDC fixed (α_ctx=0.85, α_fc=0.9999) | 0.80 | 1.44 | 10.22 | 9 |
| KN-3 | 4.86 | 13.27 | 132.49 | 0 |
| Parrot | 50.15 | 104.10 | 1006.07 | 0 |

GDC's results:

- **GDC val-tuned (leakage-free)**: median gap 0.73, mean gap 1.85,
  max gap 12.54, 0 outright wins. Held-out-NLL selection on a 20%
  train split occasionally mis-picks — it attains a slightly *lower*
  median gap than the fixed config (0.73 vs 0.80) but a *higher* mean
  (1.85 vs 1.44), and the per-problem winners always go to ALERGIA+,
  MDI, or the fixed GDC config rather than the val-tuned pick.
- **GDC fixed** ($\alpha_{\mathrm{ctx}}{=}0.85$, $\alpha_{\mathrm{fc}}
  {=}0.9999$, no per-problem tuning): median gap 0.80, mean gap 1.44,
  max gap 10.22, 9 outright wins (P1, P22, P31, P34, P36, P37, P41,
  P44, P46 — the small-floor problems where GDC's prefix matching is
  closest to the entropy floor). The single fixed config beats
  leakage-free val-tuning on both mean gap and win count: most of the
  value of config selection is already captured by the one
  $\alpha_{\mathrm{ctx}}{=}0.85$ config, and held-out NLL is a noisy
  proxy for test perplexity at this train size.
- **Ranking by median gap** across the six compared methods: ALERGIA+
  (0.12, 28 wins), MDI (0.37, 11 wins), GDC val-tuned LF (0.73, 0
  wins), GDC fixed (0.80, 9 wins), KN-3 (4.86, 0 wins), Parrot (50.15,
  0 wins). Both FlexFringe state-merging learners (ALERGIA+, MDI),
  purpose-built for the PAutomaC distribution, stay ahead of both GDC
  variants.

**Breakdown by generative model type (Table 14b).** The competition
was deliberately built from 16 problems of each of three target-machine
types — deterministic PFA (DPFA), hidden Markov model (HMM), and
non-deterministic PFA — and classifying each released target machine
(DPFA = deterministic transition per (state, symbol); HMM = next-state
distribution independent of the emitted symbol; PFA = neither) recovers
exactly the 16/16/16 split. The overall ranking is almost entirely a
*deterministic-automaton* story:

| Method | DPFA median (mean), wins | HMM median (mean), wins | PFA median (mean), wins |
|---|---|---|---|
| GDC val-tune (LF) | 0.71 (1.14), 0 | 0.67 (2.93), 0 | 0.95 (1.47), 0 |
| GDC fixed | 1.20 (1.15), 0 | 0.71 (2.36), 4 | 0.77 (**0.81**), 5 |
| ALERGIA+ | **0.02** (**0.15**), **15** | **0.25** (**0.85**), **7** | **0.53** (0.91), **6** |
| MDI | 0.19 (0.66), 1 | 0.34 (1.28), 5 | 0.75 (1.34), 5 |
| KN-3 | 8.87 (24.95), 0 | 3.43 (7.11), 0 | 4.02 (7.76), 0 |
| Parrot | 68.47 (123.0), 0 | 27.51 (60.17), 0 | 53.45 (129.1), 0 |

- **ALERGIA+'s dominance is concentrated on DPFA.** It wins 15 of 16
  deterministic problems with a median gap of 0.02 — essentially
  recovering the target exactly. State-merging is built for
  deterministic PFA, and 15 of ALERGIA+'s 28 total wins come from this
  one type. MDI (same FlexFringe family) behaves the same way.
- **GDC is most competitive on non-deterministic PFA.** There GDC fixed
  has the **lowest mean gap of any method (0.81), beating ALERGIA+
  (0.91)**, the lowest max (1.72 — it never blows up), and ties MDI for
  second in wins (5 vs ALERGIA+ 6). When the target is not
  deterministic, the state-merging structural advantage shrinks and
  GDC's prefix matching is on par.
- **HMM is GDC's worst type by tail risk.** GDC's single worst problem
  across all 48 (max gap 12.54) is an HMM (P20, an 11-state machine);
  its HMM *mean* is inflated by that tail even though its HMM *median*
  (0.67) is close to MDI's (0.34). GDC fixed still takes 4 HMM wins.
- **Val-tuning's value is type-dependent.** Leakage-free selection
  helps on DPFA (median 0.71 vs fixed 1.20) but *hurts* on PFA (mean
  1.47 vs 0.81) and HMM (mean 2.93 vs 2.36) — the held-out-NLL proxy
  mis-picks most on the stochastic/non-deterministic types, which is
  why the single fixed config beats GDC-LF on overall mean and wins.

Net: GDC's weakness relative to the FlexFringe learners is specifically
on *deterministic* targets, where exact structure recovery is possible
and GDC's soft prefix matching cannot compete; on non-deterministic PFA
GDC fixed is essentially tied with the best method.

#### Turing-machine algorithmic traces (Tables 8 + 9)

Nine tasks (parity, increment, reverse, binary adder, shift_left,
bit_count_mod3, anbn, palindrome, subtraction) × two variants
(original, noread) under a leakage-free protocol: train, val, and
test ranges defined once in
[`_tm_task_config.py`](../algorithmic_benchmarks/_tm_task_config.py);
val drawn from a stretched range that sits between train and test
(may overlap test on length but uses different seeds; val is used
only for hyperparameter selection). Each method's hyperparams are
val-tuned per task; test errors are reported only for the chosen
config. **Numbers below are at 4× the original training budget**
(n_train = 1200 for all tasks; binary_adder is 800; n_test = 20,
binary_adder n_test = 10). ALERGIA omitted from the 4× table because
its O(n² in strings) state-merging is impractical at the larger
training budget.

Held-out tape tuple-error counts (read, write, dir prediction).
**Bold = best in row.** KN-3 (interpolated Kneser-Ney trigram) is
included as a third n-gram baseline alongside HPYLM and PPM-D.

| Task | Variant | GDC | CHMM | Parrot | HPYLM | PPM-D | KN-3 |
|---|---|---:|---:|---:|---:|---:|---:|
| parity | orig | 11/506 | 10 | 10 | **9** | 12 | 13 |
| increment | orig | **0/266** | **0** | **0** | **0** | **0** | **0** |
| reverse | orig | **150/13646** | 301 | 573 | 588 | 476 | 547 |
| binary_adder | orig | **3/72217** | 10 | 381 | 375 | 178 | 2194 |
| shift_left | orig | **0/526** | **0** | **0** | **0** | **0** | **0** |
| bit_count_mod3 | orig | **10/526** | 12 | 14 | 13 | 13 | 12 |
| anbn | orig | **2/934** | 4 | 4 | 4 | 4 | 4 |
| palindrome | orig | 8/1574 | **6** | 9 | 9 | 8 | 9 |
| subtraction | orig | **857/33433** | 1572 | 1476 | 1608 | 1608 | 1622 |
| parity | nr | 11/506 | 10 | 10 | **9** | 12 | 13 |
| increment | nr | **0/266** | **0** | **0** | **0** | **0** | **0** |
| reverse | nr | **0/13646** | 121 | 349 | 313 | 313 | 415 |
| binary_adder | nr | **0/72217** | **0** | 193 | 375 | 375 | 740 |
| shift_left | nr | **0/526** | **0** | **0** | **0** | **0** | **0** |
| bit_count_mod3 | nr | **10/526** | 12 | 14 | 13 | 13 | 12 |
| anbn | nr | **0/934** | **0** | 9 | 3 | 3 | 3 |
| palindrome | nr | 13/1574 | 9 | **8** | **8** | **8** | 9 |
| subtraction | nr | **0/33433** | 966 | 1777 | 1777 | 1558 | 1862 |

Wins / 18 (row-best, ties counted): GDC 14, CHMM 7, HPYLM 7, Parrot
5, PPM-D 5, KN-3 4. (GDC's dual-α prediction step takes binary_adder
-original 59→3 — claiming it from CHMM — and improves subtraction
-original 1132→857, anbn-original 3→2, palindrome-original 16→8; this
6-method panel omits the LSTM column that appears in paper Table 8,
where GDC's dual-α also reclaims subtraction-original from the LSTM.)
For binary_adder the test numbers go 1–10× larger
than training (length-OOD generalization); for the other tasks the
test input lengths are 2–4× the training range.

GDC's per-task results:

- **Zero-error tasks (noread)**: increment, reverse, binary_adder,
  shift_left, anbn, subtraction. Six of the nine noread variants.
- **Zero-error tasks (original)**: increment, shift_left.
- **Best in row but non-zero (original)**: reverse (150/13,646),
  binary_adder (3/72,217), bit_count_mod3 (10/526), anbn (2/934),
  subtraction (857/33,433). The last four use the dual-α prediction
  step ($\alpha_{\mathrm{fc}}{=}1$); binary_adder, anbn and subtraction
  all improve over single-α (59, 3, 1132) and binary_adder reclaims the
  row from CHMM.
- **Best in row but non-zero (noread)**: bit_count_mod3 (10/526),
  palindrome (13/1,574 — not best; HPYLM/PPM-D/KN-3 tie at 8).
- **Not best in row**: parity (HPYLM at 9/506 vs GDC at 11/506),
  palindrome-original (CHMM at 6 vs GDC at 8) and palindrome-noread
  (tie at 8 vs GDC at 13).

GDC's change from 1× to 4× training budget (4× = dual-α default):
bit_count_mod3 16→10, anbn-original 5→2, anbn-noread 4→0,
subtraction-original 1,234→857, subtraction-noread 1→0.

A separate **sequential-training scaling experiment**
(`binary_adder_scaling.py`) varies the number of training tapes:
with the same GDC config ($\alpha{=}0.95$, $\theta{=}0.05$,
transition=self_loop), 2 training tapes (~5-bit operands, chain
length 535) reach 0/80,333 errors on 5–10 bit binary_adder-noread
test (~2× longer test inputs); 10 training tapes (chain $N{=}3{,}096$)
reach 0/1,034,130 errors on 11–13 bit additions (~2.6× longer than
training). The error count is 0 at $K{=}2$, dips to 2 at
$K{\in}\{3,4,5\}$, then returns to 0 from $K{=}6$ onwards.


### Continuous time-series forecasting

#### dysts chaotic-systems benchmark (Table 10, top of leaderboard)

Median sMAPE across the 130-system **intersection** common to all
methods (excludes `AtmosphericRegime`, `LidDrivenCavityFlow` not
scored in Gilpin's released baselines, and `PiecewiseCircuit` where
pyEDM val-tuning failed). Univariate protocol of Gilpin (2021),
released baseline numbers re-aggregated from `dysts_data` prediction
JSONs. **pyEDM** (Sugihara lab; Simplex of Sugihara & May, 1990 and
S-Map of Sugihara, 1994; implemented as the rEDM/pyEDM packages) and
**AnDA** (Lguensat et al., 2017; Analog Data Assimilation framework)
are included as canonical EDM / nearest-neighbour-in-embedding
baselines. Parrot is the kNN-in-prefix-space baseline of Zhang &
Gilpin (2025). All four of our methods (GDC, Parrot, pyEDM, AnDA)
share the same **multi-IC val** protocol: per system, val sMAPE is
averaged across 3 sliding val windows within the train trajectory at
a fixed 90-point fit length, instead of being scored on a single
150-point fit. Multi-IC val reduces single-trajectory val noise
caused by Lyapunov divergence between train and test ICs, and
substantially lifts the EDM-family methods (pyEDM, AnDA) and GDC.
The reported AnDA number is its **best-of-three-regression-modes**
result: the val-tuner picks per-system from a unified pool of
(`locally_constant`, `increment`, `local_linear`) × $E$ × $k$.
Median is the standard aggregation in this benchmark family
(Gilpin 2021; Zhang & Gilpin 2025) because the per-system sMAPE
distribution is heavy-tailed.

| Rank | Method | Median sMAPE | Mean sMAPE |
|---:|---|---:|---:|
| 1 | NBEATS | **49.21** | **61.12** |
| **2** | **pyEDM (ours, multi-IC val-tuned)** | **61.92** | **68.79** |
| 3 | RNN | 66.72 | 72.52 |
| **4** | **GDC (ours, dual-α, multi-IC val-tuned)** | **67.37** | **72.79** |
| **5** | **AnDA (ours, multi-IC val-tuned)** | **70.46** | 73.96 |
| **6** | **Parrot (ours, multi-IC val-tuned)** | **74.06** | 77.58 |
| 7 | Random Forest | 88.15 | 86.13 |
| 8 | Transformer | 93.39 | 91.97 |
| 9 | Linear Regression | 104.76 | 99.42 |
| 10 | ARIMA | 110.36 | 99.86 |
| 11–20 | (AutoARIMA, FFT, Theta, Naive variants, TCN, Prophet, ExpSmoothing, NaiveMean) | 113–153 | 106–130 |

GDC's results:

- **Median sMAPE**: 67.37 (4th of 20 methods, behind NBEATS at 49.21,
  pyEDM at 61.92, and RNN at 66.72; ahead of AnDA at 70.46 and
  Parrot at 74.06).
- **Mean sMAPE**: 72.79 (4th of 20). pyEDM's mean of 68.79 also
  sits well below its median; GDC's mean is above its median by 5.4.
- **Single-system outright wins (130-system intersection)**: 8 for
  GDC. NBEATS wins outright on 33 systems, pyEDM on 23, Parrot on
  19, RNN on 14, AnDA on 8, Transformer on 7, Random Forest on 7.
- **GDC's α-grid**: 36 configs combining 2 single-α
  ($\alpha\in\{1.0, 0.99\}$) with 4 dual-α
  ($\alpha_\text{ctx}\in\{0.8, 0.9, 0.95, 0.99\}, \alpha_\text{fc}=1$),
  crossed with 2 recipes (raw, diff) and 3 σ values
  ({0.05, 0.10, 0.25}). Roughly half the val-picks are dual-α.
- **√L emission-variance scaling ablation**: GDC's continuous variant
  uses $\beta = (\sigma_\text{per-step} \cdot \sqrt{L})^2$ at inference
  time. Removing the √L scaling raises GDC's median sMAPE on the
  earlier single-IC val protocol from 68.91 to 74.95.


#### M4 forecasting competition (Table 5)

Series-weighted total OWA across all 100,000 M4 series.

| Method | sMAPE | MASE | OWA |
|---|---:|---:|---:|
| Smyl (M4 1st place, hybrid ES+RNN)        | 11.375 | 1.536 | 0.833 |
| Montero-Manso (M4 2nd, FFORMA)            | 11.720 | 1.551 | 0.847 |
| Pawlikowski (M4 3rd)                      | 11.845 | 1.547 | 0.849 |
| **GDC (per-series val-OWA)**              | **12.645** | **1.611** | **0.887** |
| Parrot (by-freq val-pick)                 | 13.582 | 1.673 | 0.938 |
| ARIMA                                     | 12.669 | 1.666 | 0.904 |
| Theta                                     | 12.309 | 1.697 | 0.906 |
| Comb (mean of SES/Holt/Damped)            | 12.555 | 1.663 | 0.906 |
| Naive 2 (reference)                       | 13.564 | 1.912 | 1.000 |

GDC's results: sMAPE 12.645, MASE 1.611, OWA 0.887. OWA is below the
M4 top-3 deep-learning ensembles (0.83–0.85), above the best
statistical benchmarks (ARIMA 0.904, Theta 0.906, Comb 0.906), and
above Parrot (0.938).

#### SKOLR / Informer univariate (Tables 1, 4)

ETTm2 + Exchange (Autoformer protocol, $L=96$, MSE on standardized
data, **bold = best in {GDC, Parrot, ARIMA, Prophet}**, **red = best
overall**):

| Dataset | $T$ | GDC | Parrot | DeepAR | N-BEATS | Informer | Autoformer |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTm2 | 96 | **0.074** | 0.092 | 0.099 | 0.082 | 0.088 | **0.065** (red) |
| ETTm2 | 192 | **0.111** (red) | 0.134 | 0.154 | 0.120 | 0.132 | 0.118 |
| ETTm2 | 336 | **0.150** (red) | 0.185 | 0.277 | 0.226 | 0.180 | 0.154 |
| ETTm2 | 720 | 0.254 | **0.256** (MAE win) | 0.332 | 0.188 | 0.300 | 0.182 |
| Exchange | 96 | 0.093 | **0.086** (red) | 0.417 | 0.156 | 0.591 | 0.241 |
| Exchange | 192 | 0.207 | **0.200** (red) | 0.813 | 0.669 | 1.183 | 0.273 |
| Exchange | 336 | **0.442** (red) | 0.458 | 1.331 | 0.611 | 1.367 | 0.508 |
| Exchange | 720 | **1.757** | 2.063 | 1.894 | 1.111 | 1.872 | **0.991** (red) |


