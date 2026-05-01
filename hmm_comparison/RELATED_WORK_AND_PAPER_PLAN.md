# Related work and paper plan for GDC

A consolidated review of the literature surrounding the Generative Dense
Chain (GDC), and a concrete plan for a paper. This document supersedes
all earlier drafts and is organised as:

1. **Background areas** — the subfields a GDC paper sits in and must
   engage with.
2. **GDC's place in the landscape** — the one-paragraph elevator
   pitch and the closest-cousin map.
3. **Benchmarks** — the datasets and tasks the paper will use.
4. **Models** — the baselines, grouped by tier.
5. **Models × benchmarks matrix** — what gets tested where, and the
   expected outcome of each cell.
6. **Paper narrative** — how the experiments wrap together as a
   single argument.
7. **What GDC is and is not** — honest framing.
8. **Sources**.

---

## 1. Background areas

A GDC paper sits at the intersection of ten subfields. We list each
with the canonical references, the recent (2022–2026) developments
that change framing, and an explicit note on how GDC relates.

### 1.1 Classical automata learning (active and passive)

* **Active (L\*-family)** — Angluin (1987) L\*; Isberner et al (2014)
  TTT; Vaandrager et al (2022) L#; Muskardin et al (2022) AALpy.
* **Passive DFA learning** — Oncina & García (1992) RPNI;
  Lang–Pearlmutter–Price (1998) EDSM; Heule & Verwer (2013) DFASAT.
* **Passive PDFA learning** — Carrasco & Oncina (1994) ALERGIA;
  Thollard et al (2000) MDI; Verwer & Hammerschmidt (LMCS 2025)
  **FlexFringe** — modern, maintained C++ implementation.

**GDC's relation.** GDC is a *passive* learner. It competes most
directly with FlexFringe / ALERGIA on positive-sequence PDFA
inference, but does not perform state merging — it keeps the full
prefix tree and smooths over it.

**Could GDC be used for active query learning?** Yes, in principle:
GDC's posterior provides a natural uncertainty signal for query
selection, and the Turing-adder result (with state-tape features as
input) shows GDC can be near-perfect when the underlying machine is
fully observable. But honestly: L*/TTT/L# are essentially optimal for
deterministic targets and probabilistic-L\* variants are tighter for
stochastic ones. GDC is unlikely to outperform them at active
learning of small machines.

The one defensible active-learning angle is **GDC-as-oracle for
downstream L\* extraction**: train GDC passively on a dataset, then
use it as the membership/equivalence oracle that L\* queries. This
is the Weiss-Goldberg-Yahav (ICML 2018) RNN-extraction trick applied
to a non-parametric base learner. It is a plausible appendix
section, not a centrepiece.

### 1.2 Variable-order Markov / suffix-tree predictors

The closest classical neighbours of GDC.

* Cleary & Witten (1984) PPM; PPM-D, PPM\*, PPM-II.
* Willems et al (1995) Context Tree Weighting (CTW).
* Ron–Singer–Tishby (1996) Probabilistic Suffix Trees (PSTs).
* Begleiter–El-Yaniv–Yona (JAIR 2004) — the canonical empirical
  comparison of PPM, CTW, PST, LZ, PSA. **The natural empirical
  reference frame for a GDC paper.**

### 1.3 Bayesian non-parametric sequence models

* **Sequence Memoizer (SM)** — Wood et al (ICML 2009 / CACM 2011);
  Gasthaus & Wood (NIPS 2010) improvements; Bartlett & Wood
  "Forgetting Counts" for online inference. Hierarchical Pitman–Yor
  prior over a suffix tree; same memorise-everything philosophy as
  GDC, but with a principled posterior, power-law tail, and compact
  context-tree representation.
* **Bayesian Context Trees (BCT)** — Kontoyiannis et al (JRSS-B 2022;
  Bayesian Analysis 2024). Modern Bayesian replacement for CTW with
  exact, deterministic, linear-time inference.
* **BCT-AR / Soft-BCT / BCT-X** — Lungu & Kontoyiannis (2021–2023);
  arXiv 2308.00913, 2601.11079. Real-valued extensions of BCT.

**GDC's relation.** GDC is a non-parametric prefix-memoriser without
a Bayesian posterior. It is faster and has a simpler smoothing
parameterisation, but on natural-language-like data with Zipfian
vocabulary SM and BCT compress and back-off better. **GDC's
qualitative advantage over SM and BCT** is that the continuous-
emission extension is trivial (replace `P(o|s)` with a similarity
kernel), whereas SM and BCT-X require substantial reformulation.

### 1.4 HMM learning

* **Likelihood-based** — Baum–Welch (1970), Rabiner (1989),
  Variational HMMs (MacKay 1997, Beal 2003).
* **Spectral / method-of-moments** — Hsu–Kakade–Zhang (JCSS 2012);
  Anandkumar et al (2014) tensor decomposition;
  Boots–Siddiqi–Gordon (IJRR 2011); Glaude–Pietquin (ICML 2016);
  Zhao–Poupart (2014) "A sober look at spectral learning."
* **Predictive State Representations / Observable Operator Models** —
  Littman–Sutton–Singh (NIPS 2001) PSRs; Jaeger (Neural Comp 2000)
  OOMs. **The OOM is the spectral baseline already in our forecasting
  experiment.**

### 1.5 Cloned HMMs and Clone-Structured Causal Graphs

The single closest *parametric* cousin to GDC, missed in the first
draft.

* **Cloned HMM (CHMM)** — Dedieu, Gothoskar, Swingle, Lehrach,
  Lázaro-Gredilla, George (arXiv 1905.00507, 2019). Sparse HMM where
  each emission is duplicated into K hidden "clones"; EM diversifies
  clones to capture different temporal contexts of the same
  observation. Trains on GPU at >1B parameters; explicitly positioned
  against n-grams and the Sequence Memoizer.
* **Clone-Structured Causal/Cognitive Graph (CSCG)** — George, Rikhye,
  Gothoskar, Guntupalli, Dedieu, Lázaro-Gredilla (Nature
  Communications 2021). Extends CHMM to graph-structured cognitive
  maps, with action-conditioned transitions; provides a unified
  account of hippocampal place / splitter / lap cells.
* Code: `github.com/vicariousinc/naturecomm_cscg`.

**GDC's relation.** *GDC is the maximum-fidelity end of the design
axis whose maximum-compression end is CHMM/CSCG.* Both expand the
hidden state space to disambiguate temporal contexts; CHMM does it
via EM-trained transitions over `K × nA` clones, GDC does it by
keeping one state per training-sequence position. CHMMs must appear
as a Tier-1 baseline.

### 1.6 Modern parametric sequence models

* **Recurrent / gated** — LSTM (1997), GRU (2014).
* **State-space models** — Gu et al S4 (ICLR 2022); Gu & Dao Mamba
  (ICLR 2024); Dao & Gu Mamba-2 / SSD (ICML 2024).
* **Transformers** — Vaswani et al (2017); Liu et al (2022)
  "Transformers learn shortcuts to automata."
* **Theoretical equivalence** — Borenstein et al (EMNLP 2023):
  finite-precision RNN-LMs are PFSAs.

These share the *graphical-model skeleton* of HMMs (a Markov chain
over latents, observations conditional on the chain) but use
deterministic gradient-trained transitions.

### 1.7 In-context learning of Markovian sources (2024–2026)

A dense recent cluster that asks exactly the questions a GDC paper
asks, but for parametric models.

* **Bondaschi et al (ICLR 2025)** — "From Markov to Laplace: How
  Mamba In-Context Learns Markov Chains." A single-layer Mamba
  realises Bayes/minimax-optimal Laplacian smoothing for k-th order
  Markov chains. **Direct theoretical comparison target.**
* **Edelman et al (NeurIPS 2024)** — "The Evolution of Statistical
  Induction Heads." Transformers form bigram-induction-heads;
  training passes through uniform → unigram → bigram phases.
* **Rajaraman et al (ICLR 2025)** — Generalises induction heads to
  variable-order Markov chains.
* **Dai et al (NeurIPS 2025)** — "Pre-trained LLMs Learn HMMs
  In-context." GPT-4-class models converge to the theoretical
  optimum on synthetic HMMs and do well on animal-decision data.
* **Yang et al (ACL 2025)** — "Finite State Automata Inside
  Transformers with Chain-of-Thought." Late-MLP-layer FSA embedding.

**Take-away for framing.** GDC is the *optimisation-free,
non-parametric* analogue of what Mamba and Transformers are doing
in-context — and it matches Bayes-optimal hidden-state inference
within a few percentage points without any training.

### 1.8 Memory-augmented and retrieval-augmented neural models

The neural side of "store everything you saw."

* **kNN-LM** — Khandelwal et al (ICLR 2020); Xu et al (ICML 2023)
  "Why do kNN-LMs work?". A trained LM with a (key, next-token)
  cache from training data, retrieved at inference.
* **RetoMaton** — Alon et al (ICML 2022). Builds an automaton on top
  of the kNN-LM datastore by clustering keys into states with
  pointers between consecutive entries. **The closest neural-symbolic
  cousin to GDC.**
* **Neural Turing Machines / Differentiable Neural Computer** —
  Graves et al (2014, Nature 2016). External memory + controller.

### 1.9 Automaton extraction from neural networks

The interpretability bridge between neural models and the FSA world.

* Weiss–Goldberg–Yahav (ICML 2018, ML 2024) — DFAs from RNNs via L\*.
* Wickramasinghe et al (IJCCI 2025) — DFAs from RNNs via hyperplane
  partitioning.
* Aichernig et al (2024, 2025) — automata from Transformers; robust
  register automata from NNs.
* **TAYSIR competition** — Eyraud et al (ICGI 2023). Currently the
  most active benchmark venue for "extract a small interpretable
  model from a trained neural network." Two tracks
  (binary classification / real regression). **Recommended target.**

### 1.10 Continuous time-series modelling

The non-trivial regime where GDC has the cleanest novelty story.

* **k-NN time-series predictors** (`tsfknn` R package; multiple
  ad-hoc papers 2017–2024). Lack explicit transition / state
  structure; GDC subsumes them as a smoothed special case.
* **BCT-AR / Soft-BCT / BCT-X** — Kontoyiannis line. Closest
  *interpretable non-parametric probabilistic* competitor in the
  continuous regime.
* **Echo State Networks / reservoir computing** — random recurrent
  reservoir + linear readout. Different philosophy.
* **Classical** — ARIMA, ETS, TBATS, Gaussian processes.
* **Foundation models** — Chronos, TimeGPT, Lag-Llama, Moirai
  (2023–2025). Parametric, accuracy-dominant, opaque, expensive.

---

## 2. GDC's place in the landscape

**One-line summary.** GDC is a non-parametric prefix-memoriser with a
smoothed forward-filter inference rule. It is to PSTs what a smoothed
HMM forward filter is to a deterministic context match — and what
RetoMaton is to a generic kNN-LM — and what an "all-clones-kept"
limit is to CHMM.

**Closest-cousin map.**

| corner of the design space | closest cousin to GDC |
|---|---|
| variable-order Markov | PST, CTW, BCT |
| Bayesian non-parametric | Sequence Memoizer |
| latent-variable parametric | Cloned HMM (CHMM) |
| spectral latent-variable | Spectral OOM / PSR |
| memory-augmented neural | kNN-LM + RetoMaton |
| modern parametric | Mamba-1L (in the ICL-of-Markov regime) |
| symbolic FSA learning | FlexFringe / ALERGIA |
| neuroscience-grounded | CSCG (cognitive map) |

**GDC's distinctive points** in the design space:

| dimension | GDC | PST/CTW | ALERGIA | CHMM | RetoMaton | SM |
|---|---|---|---|---|---|---|
| state allocation | one per training pos | suffix-tree node | merge-based | K clones / emission | one per training pos | suffix-tree node |
| training | `vstack` | tree-build | iterative merging | EM | LM training + cache | HPYP inference |
| inference | smoothed forward filter | tree back-off | one PDFA path | HMM forward | retrieval + LM mix | tree back-off |
| latent variable | yes | yes (suffix node) | yes (PDFA state) | yes (clone) | partial (cluster) | yes (suffix node) |
| continuous emissions | trivial | no | no | reformulation | (via NN representation) | no |
| neural | no | no | no | no | yes | no |

---

## 3. Benchmarks

The paper covers six benchmark families. Each maps to one paper
section.

### 3.1 Random-HMM grid (already implemented)

The 9 topologies × 3 seeds × parameter sweep in
`paper_topology_and_samples.py`. Used for forecasting-NLL,
hidden-state alignment, and sample-efficiency sections. **Owned
benchmark — direct ground truth available.**

### 3.2 PAutomaC (Verwer–Eyraud–de la Higuera, ML 2014)

48 artificial PFA / PNFA / HMM benchmarks with leaderboard. Standard
reference for forecasting-NLL on probabilistic automata.

### 3.3 SPiCe 2016 (Balle et al, ICGI 2016)

15 sequence-prediction problems, mixed real/synthetic. Smaller,
older, but the data is public and several baseline numbers exist.

### 3.4 TAYSIR 2023 (Eyraud et al, ICGI 2023)

Extract-from-neural-network benchmark, two tracks (binary / real-
valued). The most active modern benchmark in this corner of the
field. **Most novel positioning available**: frame GDC as a "reference
simple model" that any extracted automaton must beat to claim it
captured the network's behaviour.

### 3.5 Algorithmic / Turing-machine traces

* **Binary-alphabet adder** (already in repo;
  `BINARY_ALPHABET_ADDER_EXPERIMENT.md`). GDC at 99.87% per-step.
* **Tomita 1–7** and **Reber** grammars — the "MNIST of automaton
  learning."
* **Parity-of-bits in window N** — canonical long-range probe.
* **Dyck-1 / Dyck-2** — counter automata.
* **Cellular automata** (Rule 30, 110, 184) — discrete trajectories
  with rich emergent structure.
* **Multi-tape TM traces** — multiplication, sorting, regex matching.
* **Length-extrapolation curves** — train at length N, test at
  {N, 1.5N, 2N, 4N}.

### 3.6 Continuous time-series

* **M4 competition** — 100 000 series across seasonal patterns.
* **Monash time-series archive** — short-horizon diverse datasets.
* **ETT / electricity / traffic** — short-horizon univariate.
* **Synthetic continuous Markov-jump processes** with known hidden
  states — to validate hidden-state alignment in the continuous
  regime.
* **Mackey–Glass / Lorenz / Rössler** — chaotic dynamics, ESN's
  home turf.

---

## 4. Models

The baselines, grouped by what they tell us. GDC is the focal
method; the others are arranged so each row of the matrix in §5
isolates one comparison axis.

### Tier A — latent-variable parametric (the closest competitors)

1. **Spectral OOM / PSR** at ranks `r ∈ {nS, nS+2, nS+5}`.
2. **EM-HMM** at correct `nS` (Baum–Welch from spectral init).
3. **EM-HMM at misspecified `nS`** (`nS_true ± 2`).
4. **Cloned HMM (CHMM)** — Dedieu et al 2019. Public code.
5. **Clone-Structured Causal Graph (CSCG)** — George et al 2021.
   Public code. Used on tasks where graph structure matters.

### Tier B — non-parametric variable-order

6. **PPM-D**.
7. **CTW** at depth 8 and 16.
8. **Bayesian Context Trees (BCT)** — R package `BCT`.
9. **PST** — Ron–Singer–Tishby; PST R package.
10. **Sequence Memoizer (SM)** — perfect-sampling variant where
    available.

### Tier C — symbolic automaton learners

11. **FlexFringe (ALERGIA-family)** — modern PDFA learner.
12. **AALpy ALERGIA** as a sanity check.

### Tier D — modern parametric (the ICL-of-Markov peers)

13. **Mamba-1L** — single-layer selective SSM. The Bondaschi et al
    Laplacian-smoothing-equivalence regime.
14. **Small Transformer** (~10k–100k params, 2 layers) — mirrors
    Edelman / Rajaraman induction-head literature.
15. **LSTM-small** (~1k–10k params).

### Tier E — memory-augmented neural

16. **kNN-LM** with a small LM backbone — direct philosophical
    competitor.
17. **RetoMaton** with same backbone — closest neural-symbolic
    competitor.

### Tier F — continuous-emission baselines (used only in §3.6)

18. **ARIMA / ETS / TBATS** (`forecast` R / `statsmodels`).
19. **k-NN time-series** (`tsfknn`).
20. **BCT-AR / Soft-BCT / BCT-X**.
21. **Echo State Network**.
22. **Small Transformer / Mamba time-series** (PatchTST,
    Chronos-tiny).

### Tier G — extras (used opportunistically)

23. **Lempel–Ziv predictor**.
24. **n-gram with Kneser–Ney smoothing**.
25. **Neural Turing Machine / DNC** — algorithmic-task ceiling.

---

## 5. Models × benchmarks matrix

Cells are tested → expected outcome. `—` = not applicable. The
matrix is what the paper's results section is structured around.

| Model | Random HMM (§3.1) | PAutomaC (§3.2) | SPiCe (§3.3) | TAYSIR (§3.4) | Algorithmic (§3.5) | Continuous TS (§3.6) |
|---|---|---|---|---|---|---|
| **GDC** (focal) | strong on most topologies; weakest on sparse-branching FSAs | competitive with Tier A; sample-efficient at small N | competitive on dynamics-driven problems | competitive reference simple model | **near-perfect with state features**; controlled length-extrapolation degradation | competitive vs ARIMA/BCT-AR; unique hidden-state lift |
| Spectral OOM | weak at sparse topologies (H4) | mid-pack; calibration issues | similar | not applicable | weak | — |
| EM-HMM (true nS) | Bayes ceiling on hidden-state lift; strong NLL | strong if nS guessed right | strong | — | weak without features | — |
| EM-HMM (misspec nS) | degrades with nS error | degrades | degrades | — | — | — |
| **CHMM** | **Tier-1 competitor**; slightly tighter hidden-state alignment than GDC at large N | strong | strong | competitive | strong on in-distribution algorithmic; better length extrap than GDC | reformulation needed |
| **CSCG** | only when graph structure matters (cognitive-map subset) | partial | partial | partial | strong on spatial / navigation tasks | reformulation needed |
| PPM-D | mid | mid | mid | — | strong in-dist; fails length extrap | — |
| CTW (D=8, 16) | mid | strong (well-known) | strong | — | strong in-dist; fails length extrap | — |
| **BCT** | mid; principled posterior | strong | strong | partial | — | — |
| PST | mid | mid | mid | — | — | — |
| **Sequence Memoizer** | mid | strong on Zipfian | strong | — | strong in-dist | — |
| FlexFringe (ALERGIA) | strong; produces small interpretable PDFA | leaderboard-class | strong | strong (interpretable extraction) | — | — |
| **Mamba-1L** | matches Bayes-Laplacian on bigram order; weaker on higher-order | mid (small model) | mid | — | scaffolded with curriculum | competitive (PatchTST-class) |
| Small Transformer | induction-head regime | mid | mid | central (TAYSIR is about extracting from these) | length-extrap with curriculum | mid |
| LSTM-small | weakest of neural | weak | weak | central | requires curriculum | weak |
| kNN-LM | — (overkill for HMM) | weak (small data) | — | partial | — | — |
| **RetoMaton** | partial via cluster-state | partial | partial | central (cluster automaton extraction) | — | — |
| ARIMA / ETS / TBATS | — | — | — | — | — | strong baseline |
| k-NN TS | — | — | — | — | — | weak |
| **BCT-AR / BCT-X** | — | — | — | — | — | strong; closest competitor |
| Echo State Network | — | — | — | — | — | strong on chaotic |
| PatchTST / Chronos-tiny | — | — | — | — | — | accuracy-leader; opaque |
| NTM / DNC | — | — | — | — | algorithmic ceiling | — |

### Headline results table (§4 of the paper)

| Method | Forecast NLL | Hidden-state lift | Train time | Inference time | Memory | Params |
|---|---|---|---|---|---|---|
| Spectral OOM (r = nS+2) | mid | mid | fast | fast | small | small |
| EM-HMM (true nS) | strong | **Bayes ceiling** | mid | fast | small | small |
| **CHMM** | strong | strong | mid–slow (EM, GPU helps) | fast | mid | small–large |
| CSCG | task-dependent | strong on graph tasks | slow | mid | mid | large |
| Mamba-1L | mid | partial (probe) | slow | fast | mid | small |
| Transformer-2L | mid | partial (probe) | slow | mid | mid | small |
| PPM-D | mid | — | fast | fast | mid | nonparam |
| CTW(D=8) | strong | — | fast | fast | mid | nonparam |
| BCT | strong | partial | mid | mid | mid | nonparam |
| PST | mid | partial | fast | fast | mid | nonparam |
| FlexFringe | strong | yes (PDFA state) | mid | fast | small | nonparam |
| Sequence Memoizer | strong | — | slow | mid | small (compressed) | nonparam |
| LSTM-small | mid | partial | slow | fast | small | small |
| kNN-LM | mid | — | slow | slow | LM + DB | small + DB |
| RetoMaton | strong | partial | slow | mid | LM + DB | small + DB |
| **GDC** | strong | **yes (this paper)** | **fastest** | fast | large | nonparam |

Hidden-state lift is meaningful only for methods with an explicit or
extractable latent state.

### Expected per-section verdicts

* **§4 Forecasting NLL** — GDC is competitive across PAutomaC, SPiCe,
  TAYSIR, and the random-HMM grid. It is *not* expected to beat
  CTW/BCT on natural-language-flavoured data; it is expected to beat
  spectral-OOM on most topologies and match or modestly beat CHMM on
  small-data regimes.
* **§5 Hidden-state lift** — GDC is within a few pp of the Bayes
  ceiling on most topologies; CHMM is comparable, slight CHMM edge at
  large N, slight GDC edge at small N. Mamba-1L probed for hidden
  state should land below both.
* **§6 Sample efficiency** — GDC saturates at N_train ≈ 25; PPM/CTW/
  BCT/SM saturate at similar small N; Spectral-OOM and EM-HMM
  saturate at moderate N; LSTM/Mamba/Transformer need much more.
* **§7 Cost** — GDC trains in `O(observations)` via `vstack`; this
  is the cheapest method in the comparison by orders of magnitude.
  Inference is one matrix–vector multiply per step.
* **§8 Dimensionality / interpretability** — naive SVD on GDC fails
  to reveal `nS`; aggregation-by-emission-context recovers it; the
  same diagnostic protocol applied to Mamba/Transformer hidden
  states should reveal a parallel dimensionality structure.
* **§9 Algorithmic learning** — with state-tape features, GDC is
  near-perfect in-distribution; CHMM and CTW are comparable; length-
  extrapolation degradation curves separate the methods qualitatively.
* **§10 Continuous emissions** — GDC matches or beats ARIMA / ETS on
  short benchmarks, competes with BCT-AR on Markov-jump synthetic
  data, and is alone in offering a hidden-state alignment story for
  continuous data.

---

## 6. Paper narrative

The paper is one argument in nine sections, each grounded in a
specific experiment, each with a narrow expected finding.

```
1. Introduction
   - The non-parametric prefix-memoriser corner of sequence modelling.
   - GDC as the optimisation-free analogue of Mamba's in-context
     Laplacian smoothing on Markov chains (Bondaschi et al ICLR 2025).
   - GDC as the all-clones-kept limit of CHMM (Dedieu et al 2019).
   - Six findings: forecasting-strength, dimensionality opacity under
     naive SVD, transparency under context-aggregation, hidden-state
     alignment near Bayes, sample efficiency saturating at N≈25,
     and a clean structural failure on branching FSAs.

2. Related work (§§1–2 above, condensed to ~2 pages).

3. The model
   - Definition; training; inference.
   - Connections to PSTs, BCT, RetoMaton, smoothed HMMs, CHMM.
   - Hyperparameters (alpha, theta, beta, gamma, d) and their roles.

4. Forecasting benchmarks (§3.1–3.4)
   - PAutomaC, SPiCe, TAYSIR, random-HMM grid.
   - Tier A–E baselines.

5. Hidden-state recoverability (§3.1)
   - Posterior-alignment metric (lift over Bayes).
   - Cross-topology results.
   - GDC vs Mamba-1L / Transformer-2L / EM-HMM / CHMM /
     RetoMaton-clusters on the same metric.

6. Sample efficiency and computational cost
   - GDC saturates at N_train ≈ 25.
   - Curves vs Mamba, Transformer, EM-HMM, CHMM.

7. Dimensionality and interpretability
   - Naive SVD does not reveal nS; diffusion sweep collapses to nA;
     aggregation-by-emission-context reveals nS.
   - Same probing protocol applied to Mamba hidden states.

8. Theoretical context: GDC as optimisation-free Laplacian smoothing
   - Connect to Bondaschi et al (ICLR 2025).
   - GDC's predictive distribution as a mixture-of-Markov-chains
     estimator at small d.

9. Algorithmic learning with state-visibility features
   - Turing-adder (99.87% per-step) as worked example.
   - Parity-of-bits, Dyck-1/2, cellular automata, multi-tape TM.
   - Length-extrapolation degradation curves as a controlled
     limitation.

10. Continuous-emission extension
    - Kernel-based emission likelihood; everything else of GDC is
      unchanged.
    - M4 / Monash / ETT vs ARIMA, ETS, BCT-AR, kNN-TS, ESN, PatchTST.
    - Hidden-state alignment on synthetic Markov-jump data.

11. Limitations and failure modes
    - Mealy / deterministic-branching FSAs.
    - Length extrapolation.
    - Absorbing chains (metric pathology).
    - Memory cost on long training sequences.
    - Natural-language data (where Sequence Memoizer wins).

12. Discussion: when GDC is the right tool.

13. Conclusion.

Appendices: reproducibility, full numbers, failure-mode catalogue.
```

**Length.** ~9 pages main + appendix for an ML conference; 25–30 for
journal. **First version targets ICGI 2026** (smallest, most relevant
audience; PAutomaC/SPiCe/TAYSIR live there). A NeurIPS/ICML version
framed against the ICL-of-Markov literature can follow.

### A particularly natural cross-cutting experiment

**Use GDC's posterior to initialise CHMM's EM**, instead of random
init. CHMM authors note the credit-diffusion problem of HMMs; GDC's
surface-form posterior provides a fully-resolved starting point that
EM only has to merge from, not diffuse to. Small, novel, easy to
run, links three sections of the paper.

---

## 7. What GDC is and is not

**GDC is:**

* A sample-efficient, structurally-transparent, **optimisation-free**
  predictor for sequences from HMM-like sources, **including
  real-valued and vector-valued time series** — a regime where the
  closest competitors (Sequence Memoizer, classical CTW, PSTs) do
  not natively operate.
* A method that **matches Bayes-optimal hidden-state inference within
  a few percentage points** without any training, on most random HMM
  topologies.
* **A competitive in-distribution learner of algorithmic structure**
  given appropriate state-visibility features (Turing-adder: 99.87%
  per-step accuracy).
* The **all-clones-kept limit of CHMM** — sharing CHMM's emission-
  cloning intuition without EM's compression and credit-assignment.
* The **optimisation-free, non-parametric analogue** of what Mamba
  and Transformers do in-context on Markov chains.
* A useful **diagnostic** for sequence data — its structural failure
  modes (branching FSAs, length extrapolation) are themselves
  informative.
* A **simple reference baseline** that any "interpretable sequence
  model" or "in-context learning of Markov sources" paper should
  beat before claiming novelty.

**GDC is not:**

* A better LM than Transformers on real natural-language text.
  Sequence Memoizer / Pitman–Yor handles Zipfian vocabularies
  better; Transformer / Mamba dominate at scale.
* A learner that **extrapolates to sequence lengths beyond training**.
  Its prefix form cannot represent patterns that only appear longer
  than what it has seen. This is a controlled, characterisable
  limitation, not a hidden one.
* A learner of small interpretable automata. FlexFringe does
  state-merging that GDC cannot.
* A serious active query learner. L\* / TTT / L# / probabilistic-L\*
  are tighter for that task.

---

## 8. Sources (web-searched, April 2026)

- [PAutomaC: a probabilistic automata and HMM learning competition (Verwer et al, ML 2013/14)](https://link.springer.com/article/10.1007/s10994-013-5409-9)
- [Results of the SPiCe Sequence Prediction Challenge (Balle et al, ICGI 2016)](http://proceedings.mlr.press/v57/balle16.html)
- [TAYSIR competition site (Eyraud et al, ICGI 2023)](https://remieyraud.github.io/TAYSIR/)
- [TAYSIR proceedings paper](https://proceedings.mlr.press/v217/eyraud23a.html)
- [ICGI 2023 proceedings (PMLR vol 217)](https://proceedings.mlr.press/v217/)
- [Hsu, Kakade, Zhang — A Spectral Algorithm for Learning HMMs](https://arxiv.org/abs/0811.4413)
- [Kontoyiannis et al — Bayesian Context Trees, JRSS-B 2022](https://arxiv.org/abs/2007.14900)
- [Papageorgiou & Kontoyiannis — Posterior Representations for BCT (Bayesian Analysis 2024)](https://projecteuclid.org/journals/bayesian-analysis/advance-publication/Posterior-Representations-for-Bayesian-Context-Trees--Sampling-Estimation-and/10.1214/23-BA1362.full)
- [BCT R package on CRAN](https://cran.r-project.org/web/packages/BCT/BCT.pdf)
- [Begleiter, El-Yaniv, Yona — On Prediction Using Variable Order Markov Models, JAIR](https://arxiv.org/pdf/1107.0051)
- [PST R package update](https://r-forge.r-universe.dev/PST)
- [Wood et al — A Stochastic Memoizer for Sequence Data, ICML 2009](https://icml.cc/Conferences/2009/papers/319.pdf)
- [Wood et al — The Sequence Memoizer, CACM 2011](https://dl.acm.org/doi/10.1145/1897816.1897842)
- [Gasthaus & Wood — Improvements to the Sequence Memoizer (NIPS 2010)](http://papers.neurips.cc/paper/3938-improvements-to-the-sequence-memoizer.pdf)
- [Gasthaus & Wood — Lossless compression based on Sequence Memoizer (DCC 2010)](https://www.cs.ubc.ca/~fwood/papers/Gasthaus-DCC-2010.pdf)
- [Perfect Sampling for Hierarchical Pitman-Yor (PMC 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10597554/)
- [Lungu, Kontoyiannis — CTW for real-valued time series](https://arxiv.org/pdf/2106.03023)
- [Soft Bayesian Context Tree Models for Real-Valued Time Series (arXiv 2601.11079)](https://arxiv.org/html/2601.11079)
- [The Bayesian Context Trees State Space Model (arXiv 2308.00913, IJF 2025)](https://www.sciencedirect.com/science/article/pii/S0169207025000688)
- [FlexFringe — Verwer & Hammerschmidt (LMCS 2025)](https://arxiv.org/html/2203.16331v5)
- [AALpy — active automata learning library](https://link.springer.com/article/10.1007/s11334-022-00449-3)
- [Borenstein et al — Recurrent NLMs as PFSAs, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.502.pdf)
- [Mamba (Gu, Dao, ICLR 2024)](https://arxiv.org/abs/2312.00752)
- [Mamba-2 / SSD blog (Goomba Lab, 2024)](https://goombalab.github.io/blog/2024/mamba2-part1-model/)
- [Relation of State Space Models and HMMs (arXiv 2601.13357)](https://arxiv.org/html/2601.13357)
- [Bondaschi et al — From Markov to Laplace, ICLR 2025](https://arxiv.org/abs/2502.10178)
- [Edelman et al — Statistical Induction Heads, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/75b0edb869e2cd509d64d0e8ff446bc1-Paper-Conference.pdf)
- [Rajaraman et al — Transformers Learn Variable-order Markov Chains in-Context (OpenReview)](https://openreview.net/forum?id=TdgAtxP6G2)
- [Dai et al — Pre-trained LLMs Learn HMMs In-context, NeurIPS 2025](https://arxiv.org/abs/2506.07298)
- [Yang et al — FSAs Inside Transformers with CoT, ACL 2025](https://aclanthology.org/2025.acl-long.668.pdf)
- [Khandelwal et al — kNN-LM, ICLR 2020](https://arxiv.org/abs/1911.00172)
- [Xu et al — Why do kNN-LMs work, ICML 2023](https://arxiv.org/abs/2301.02828)
- [Alon et al — RetoMaton, ICML 2022](https://arxiv.org/abs/2201.12431)
- [RetoMaton code](https://github.com/neulab/retomaton)
- [Weiss, Goldberg, Yahav — Extracting Automata from RNNs (ML 2024)](https://link.springer.com/article/10.1007/s10994-022-06163-2)
- [Wickramasinghe et al — DFAs from RNNs via Hyperplane Partitioning, IJCCI 2025](https://link.springer.com/chapter/10.1007/978-3-032-15638-9_8)
- [Aichernig et al — Automata Extraction from Transformers, arXiv 2406.05564](https://arxiv.org/html/2406.05564)
- [Aichernig et al — Robust Register Automata from NNs (Nov 2025)](https://arxiv.org/html/2511.19100)
- [Predictive State Representations — Wikipedia](https://en.wikipedia.org/wiki/Predictive_state_representation)
- [Boots, Siddiqi, Gordon — Closing the learning-planning loop with PSRs](https://www.cs.cmu.edu/~ggordon/boots-siddiqi-gordon-closing-loop-psrs.pdf)
- [Dedieu et al — Cloned HMMs (arXiv 1905.00507, 2019)](https://arxiv.org/abs/1905.00507)
- [George et al — Clone-Structured Cognitive Graphs, Nat Comm 2021](https://www.nature.com/articles/s41467-021-22559-5)
- [Memorize-Generalize: online algorithm for cloned HMMs (bioRxiv 2019)](https://www.biorxiv.org/content/10.1101/764456v1)
- [naturecomm_cscg code repository](https://github.com/vicariousinc/naturecomm_cscg)
- [Dileep George publications page](https://dileeplearning.github.io/)
- [tsfknn — KNN time-series forecasting R package](https://cran.r-project.org/web/packages/tsfknn/vignettes/tsfknn.html)
- [Survey on Bayesian non-parametric time series (Frontiers 2023)](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2023.1287516/full)
- [Attraos — non-parametric phase-space-reconstruction time-series memory (NeurIPS 2024)](https://dl.acm.org/doi/10.5555/3737916.3738571)
