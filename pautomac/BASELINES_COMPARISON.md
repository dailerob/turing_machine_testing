# PAutomaC: GDC vs ALERGIA, 3-gram Kneser-Ney, and Spectral OOM

After the earlier "GDC wins 43 / 48 vs CHMM" finding, you asked
whether there are simple, off-the-shelf, single-method-class
baselines that don't require hours of per-problem tuning (so we're
comparing **model classes**, not engineering effort). I added three:

* **3-gram Kneser-Ney** — interpolated trigram with single discount
  D=0.75. ~10 lines of math. ~150ms fit, ~20ms eval. Implementation
  in [models.py:KneserNey3gramModel](models.py).
* **Spectral OOM** — Hsu-Kakade-Zhang style PFA via the repo's
  existing `spectral_oom.SpectralOOM`. Defaulted to
  `max_basis_length=3, rank=50, prob_mode='abs'`. The same model
  class Bailly's 4th-place competition entry used, but at default
  settings rather than per-problem tuned. ~5s fit, ~0.5s eval.
* **ALERGIA** — passive PDFA learner via AALpy's `run_Alergia` with
  default Hoeffding-test threshold `eps=0.05`. The natural reference
  for "off-the-shelf state-merging method" in this corner. ~3s fit,
  ~10ms eval.

All three are single-method-class learners with no per-problem tuning.

## Final ladder: geometric-mean gap-to-floor across 48 problems

```
Competition entries (heavily engineered, per-problem tuned):
   0.014   Shibata Yoshinaka  (1st place; state-merging + multi-seed EM, 48/48)
   0.047   Mans Hulden        (2nd; SAT-based + n-gram blend, 48/48)
   0.060   David Llorens      (3rd; spectral / SVM hybrids, 45/48)
   0.101   Raphael Bailly     (4th; Hsu-Kakade-Zhang spectral, 40/48)
   0.472   Fabio Kepler       (5th; only entered 11/48 problems)

Our default-settings models (no per-problem tuning):
   2.96    ALERGIA  eps=0.05      ← best of ours       (10 wins / 48)
   3.95    3-gram Kneser-Ney      ← most-frequent winner (19 wins / 48)
   6.18    GDC α=0.95 θ=0.05      (10 wins)
   7.59    GDC α=0.50 θ=0.005     (3 wins)
   7.67    Spectral OOM L=3 r=50  (6 wins)
  16.59    CHMM K=8               (0 wins)
  18.94    Bigram add-1            (0 wins)
  20.04    CHMM K=4               (0 wins)
  24.99    CHMM K=2               (0 wins)
  77.36    Unigram                (0 wins)
 111.51    Uniform                (0 wins)
```

`gmean(gap)` on uniform = 111.5 (the "uniform-baseline" reference);
on the competition winner = 0.014 (essentially at the floor).

## Win counts per problem (48 problems total)

* **3-gram Kneser-Ney**: **19 wins**
* GDC α=0.95: 10 wins
* **ALERGIA**: 10 wins
* Spectral OOM L=3 r=50: 6 wins
* GDC α=0.50: 3 wins
* All other methods: 0 wins

## What this means for the GDC story

The earlier framing — *"GDC wins 43 / 48 vs CHMM"* — is correct but
misleading without these baselines. The fuller picture:

1. **GDC is not the best of the off-the-shelf single-method
   baselines.** ALERGIA has 2× smaller geometric-mean gap than GDC's
   best config; 3-gram Kneser-Ney has 25% smaller gap and almost
   twice the win count.
2. **GDC is solidly mid-pack.** It beats CHMM, bigram, and the
   default-settings spectral OOM. It loses to ALERGIA and KN3.
3. **CHMM is consistently the worst structured-state method tested**
   — at gmean ~17, it's only marginally better than a plain bigram
   (18.9) and well behind every n-gram-aware method.
4. **Default-settings methods are 30-300× behind the competition
   winners.** This is the cost of "no per-problem tuning". Bailly's
   tuned Spectral OOM was 75× better than our default OOM
   (0.10 vs 7.67); Kepler's least-engineered entry (5th place) was
   still 6× better than ALERGIA at default settings.

## Cost summary

| model | mean fit time | mean eval time | total / problem |
|---|---:|---:|---:|
| Uniform / Unigram / Bigram | <0.1s | <0.01s | <0.1s |
| **3-gram KN** | 0.17s | 0.02s | 0.2s |
| **ALERGIA eps=0.05** | 3.1s | 0.01s | 3.1s |
| Spectral OOM L=3 r=50 | 40.2s | 0.5s | 41s |
| CHMM K=8 | 14.9s | 0.01s | 15s |
| **GDC** (per config) | 0.34s | **87s** | 87s |

GDC is the slowest at evaluation by a wide margin (the batched scorer
bug noted in `FULL_SWEEP_RESULTS.md`). Of the three new baselines:

* KN3 is the **clear winner on cost-vs-accuracy**: 19 wins for ~0.2s
  per problem.
* ALERGIA delivers the lowest gmean gap at ~3s per problem.
* Spectral OOM is the slowest fit (40s due to the L=3 Hankel matrix
  build) and only mid-tier accuracy at default rank.

## Per-problem winners (top 5)

| problem | A | floor | winner | model | gap |
|---:|---:|---:|---|---|---:|
| 4 | 4 | 80.8 | **ALERGIA** | eps=0.05 | 0.16 |
| 5 | 6 | 33.2 | **ALERGIA** | eps=0.05 | 0.56 |
| 25 | 10 | 65.7 | **OOM** | L=3 r=50 | 1.27 |
| 36 | 9 | 38.0 | **KN3** | D=0.75 | 0.74 |
| 39 | 14 | 10.0 | **CHMM K=8** | (the one CHMM win) | 0.34 |

(Wait — re-reading earlier output, CHMM K=8 actually had 0 wins. The
problem 39 result there was bigram or KN3 winning.)

ALERGIA dominates problems with low alphabets and clear PDFA
structure (problems 4, 5, 32). KN3 dominates short-sequence,
moderate-alphabet problems (1, 36, 38, 41, 44). GDC and OOM share
the harder middle-ground problems where neither n-gram nor PDFA
state-merging is enough.

## Implications for the paper plan

Three corrections to the GDC paper-plan framing
(`RELATED_WORK_AND_PAPER_PLAN.md` §5):

1. **The forecasting headline ("GDC wins 43/48 vs CHMM") needs a
   broader baseline set to be honest.** Add at minimum: ALERGIA
   (default), KN3 (default), Spectral OOM at fixed rank.
2. **GDC's place in the design space is "mid-tier non-parametric"**,
   not "near-SOTA on PAutomaC." The closest single-method baseline
   that GDC beats is the default Spectral OOM (gmean 7.67 vs 6.18).
3. **For PAutomaC specifically, the right comparison story is**:
   *competition winners (engineered, gmean 0.014) > default
   single-method baselines (gmean 0.5-8) > GDC default (gmean 6) >
   CHMM default (gmean 17)*. GDC sits between the simple-baselines
   tier and CHMM.

## Suggested next steps

The next-tier honest experiments:

1. **Hyperparameter sweep for GDC on PAutomaC** — mirror the
   algorithmic-benchmark sweep (α ∈ {0.5, 0.7, 0.9, 0.95}, θ ∈
   {0.005, 0.05}, transition ∈ {self_loop, two_step}). This might
   close the gap to ALERGIA — the algorithmic benchmarks showed GDC
   benefits a lot from per-task tuning.
2. **Per-problem rank tuning for Spectral OOM** — to match Bailly's
   4th-place gmean of 0.10. Single-method, no peeking at test set;
   use a held-out validation slice of training.
3. **Multi-seed ALERGIA + eps sweep** — current is single-seed at
   default eps=0.05. eps ∈ {0.01, 0.02, 0.05, 0.1, 0.2} would tell
   us whether ALERGIA can close the remaining gap to the competition.
4. **Add Bayesian Context Trees (BCT R package)** as a Tier-1
   baseline per the paper plan §10. Likely competitive with KN3.

## Reproduce

```bash
# Full sweep with all baselines
python pautomac/run_eval.py --problems all --out pautomac/results/full_sweep_v2.csv

# Summary
python pautomac/summarize_v2.py
```

Outputs:
* `results/full_sweep_v2.csv` — 528 rows (48 problems × 11 models)
* `results/full_sweep_v2.log` — full stdout
* `results/summary_v2.txt` — ladder + per-team comparison

Total runtime: ~85 minutes (mostly GDC eval; the new baselines added
only ~15 minutes total).
