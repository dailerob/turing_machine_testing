# Comparison to the PAutomaC 2012 competition leaderboard

The PAutomaC competition leaderboard is publicly maintained at
[grammarlearning.org/pautomac](https://grammarlearning.org/pautomac/).
For each of the 48 synthetic problems it lists the **minimum
achievable perplexity** (entropy floor) and the four submitted scores
ranked First / Second / Third / Fourth. The four competing teams were:

| rank | team | overall pts |
|---:|---|---:|
| 1 | Shibata Yoshinaka | 212 |
| 2 | Mans Hulden | 124 |
| 3 | David Llorens | 122 |
| 4 | Raphael Bailly (spectral / Hsu-Kakade-Zhang style) | 75 |
| 5 | Fabio Kepler | 14 |

Scoring is 5 / 3 / 2 / 1 points per problem for First / Second /
Third / Fourth.

## Headline summary

For each of the 48 problems we compare the *gap to entropy floor*
(score − floor; lower is better):

| | mean | median | max | gmean(gap) |
|---|---:|---:|---:|---:|
| **competition winner** (1st place per problem) | **0.025** | **0.013** | **0.097** | **0.0125** |
| best CHMM (ours, K ∈ {2, 4, 8}) | 40.94 | 20.32 | 421.04 | 15.75 |
| best GDC (ours, 2 configs swept) | 7.097 | 4.934 | 33.86 | 3.43 |

**Competition winners are essentially AT the entropy floor on every
problem** — gap of 0.001 to 0.1 perplexity, never more than 0.1.
That is the SOTA bar. Our GDC averages a 7-perplexity gap; CHMM
averages 41. The winners are roughly:

* **~300× closer to optimal than GDC** in geometric-mean gap (0.0125 vs 3.43).
* **~1300× closer to optimal than CHMM** in geometric-mean gap (0.0125 vs 15.75).

GDC is never within 0.1 perplexity of any winner's score on any of
the 48 problems. The closest is problem 47: GDC 4.285 vs winner 4.119
(delta +0.17).

## Per-problem extremes (GDC vs competition winner)

GDC's worst-vs-winner problems:

| problem | A | floor | winner | GDC | gap |
|---:|---:|---:|---:|---:|---:|
| 21 | 23 | 30.5 | 30.57 | 64.4 | **+33.8** |
| 26 | 6 | 80.7 | 80.83 | 108.9 | +28.1 |
| 20 | 18 | 91.0 | 91.00 | 118.7 | +27.7 |
| 25 | 10 | 65.7 | 65.78 | 89.1 | +23.3 |
| 17 | 13 | 47.3 | 47.35 | 67.8 | +20.5 |

GDC's best-vs-winner problems:

| problem | A | floor | winner | GDC | gap |
|---:|---:|---:|---:|---:|---:|
| 47 | 15 | 4.12 | 4.119 | 4.285 | +0.17 |
| 41 | 7 | 13.9 | 13.92 | 14.12 | +0.20 |
| 37 | 8 | 21.0 | 21.00 | 21.21 | +0.21 |
| 45 | 19 | 24.0 | 24.05 | 24.57 | +0.53 |
| 44 | 13 | 11.7 | 11.73 | 12.27 | +0.55 |

GDC does best (relative to the winner) on problems with **low entropy
floors and short test sequences** (problems 47, 41, 37, 45, 44 — all
with floor ≤ 24). On these, the ground truth is close to a
fixed-distribution random source, and a simple prefix-memoriser is
nearly enough.

GDC does worst on problems with **larger alphabets and moderate
floors** (problems 21 / A=23, 20 / A=18, 17 / A=13, 25 / A=10) — the
regime where the underlying machine has lots of latent structure
that competition methods exploit but our smoothed prefix filter
doesn't.

## What the competition winners actually did

The top teams used heavily engineered, problem-specific methods
combining several techniques:

* **Shibata-Yoshinaka** (1st): state-merging with EM refinement and
  multi-seed restart. Won 33 / 48 problems.
* **Mans Hulden** (2nd): probabilistic SAT-based methods + n-gram
  baselines.
* **David Llorens** (3rd): spectral / SVM-based methods, particularly
  effective on the larger-alphabet problems.
* **Raphael Bailly** (4th): pure Hsu-Kakade-Zhang spectral OOM. Strong
  on dense-PFA problems, degraded sharply on sparse transition
  matrices.

These solutions are described in the Verwer-Eyraud-de la Higuera 2014
ML-journal paper. They typically:

* Sweep many hyperparameters per problem (we only tested 2 GDC
  configs).
* Use validation splits to select per-problem hyperparameters.
* Combine multiple methods (e.g. n-gram blended with state-merging).
* Take wallclock time of hours, sometimes days, per problem.

Our GDC training-and-eval is on the order of *seconds per
configuration per problem*, with no per-problem tuning beyond
choosing one of 2 fixed `(α, θ, transition)` triples.

## Honest framing for the paper

Three honest takeaways:

1. **GDC is not competitive with the SOTA on PAutomaC** at the
   entropy-floor-precision level the competition operates at. The
   competition winners are within 0.1 perplexity of optimal; GDC
   averages 7. This is a real and large gap.
2. **GDC dominates CHMM** on this benchmark by ~5× in gap-to-floor.
   The earlier finding that "GDC wins 43 / 48 vs CHMM" remains
   correct but should be read as "between two non-state-of-the-art
   methods, GDC is the better one."
3. **Like-for-like comparison.** The competition top entries are
   not directly comparable to GDC: they use per-problem hyperparameter
   tuning and method ensembling. A fairer comparison would be GDC
   against ALERGIA / FlexFringe with similarly minimal tuning, or
   against the 4-place spectral baseline (Bailly), which is a single
   well-defined method. We have not yet built those baselines.

## Suggested next steps

1. **Add ALERGIA / FlexFringe as a fairer single-method baseline.**
   Modern open-source FlexFringe ([Verwer & Hammerschmidt LMCS 2025](https://arxiv.org/html/2203.16331v5))
   is the natural single-method comparison.
2. **Run a wider GDC hyperparameter sweep.** The algorithmic
   benchmarks taught us that 2 configs are too few. An 18-point grid
   per problem would close some of the gap, especially on the
   worst-performing problems (21, 26, 20).
3. **Sweep larger CHMM K** (16, 32, 64) and run multi-seed EM. K=8
   was clearly capacity-limited on problems 8, 9, 26.
4. **Build the fixed-baseline spectral OOM comparison.** A from-scratch
   Hsu-Kakade-Zhang implementation (ranks like Bailly's submission)
   would give us a directly-comparable single-method baseline against
   which GDC's relative position becomes clearer.

## Reproduce

```bash
python pautomac/compare_to_competition.py
```

Outputs:
* `results/compete_compare.csv` — per-problem winner / CHMM / GDC scores
* `results/compete_compare.txt` — full stdout
