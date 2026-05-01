# PAutomaC initial results — problems 1-5

First sweep: PAutomaC problems 1-5, all baselines + CHMM at K∈{2,4,8}
+ GDC at two hyperparameter configs.

## Headline table

`gap` is `score - entropy_floor` (perplexity units; lower is better;
`0` = optimal). `lift` is `(uniform_baseline − score) / (uniform_baseline − floor)`,
where `uniform_baseline = N_test = 1000`.

| problem | A | n_train | floor | uniform | unigram | bigram | CHMM K=2 | CHMM K=4 | CHMM K=8 | GDC α=.95 θ=.05 | **GDC α=.50 θ=.005** | best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 20k | 29.9 | 72.9 | 69.8 | 45.3 | 44.0 | 46.6 | 48.0 | 37.4 | **36.4** | GDC |
| 2 | 18 | 20k | 168.3 | 484.1 | 447.5 | 233.4 | 230.3 | 255.3 | 223.9 | 306.4 | **173.8** | GDC |
| 3 | 4 | 20k | 49.9 | 145.0 | 102.1 | 87.5 | 76.0 | 84.0 | 71.2 | **56.4** | 70.6 | GDC |
| 4 | 4 | 100k | 80.8 | 541.5 | 508.4 | 228.9 | 115.3 | 107.5 | 91.7 | **82.4** | 149.6 | GDC |
| 5 | 6 | 20k | 33.2 | 2228.3 | 493.1 | 201.5 | 154.2 | 129.2 | 96.4 | **42.4** | 89.2 | GDC |

GDC wins all five. The two hyperparameter configs we tried split
3 / 2 — α=0.50 wins on problems 1 and 2, α=0.95 wins on problems
3, 4, 5 — confirming again that the GDC optimum is task-dependent
and a sweep is required for paper-grade numbers.

## Gap-to-floor (lower = closer to optimal)

| problem | best CHMM gap | best GDC gap | GDC vs CHMM |
|---:|---:|---:|---|
| 1 | 14.1 | **6.5** | 2.2× better |
| 2 | 55.5 | **5.4** | **10.2× better** |
| 3 | 21.2 | **6.5** | 3.3× better |
| 4 | 10.8 | **1.6** | **6.7× better** |
| 5 | 63.1 | **9.2** | **6.9× better** |

GDC's gap to the floor is consistently 2–10× smaller than CHMM's
best K. On problem 4 GDC reaches `gap = 1.6` against `floor = 80.8`
— **essentially optimal**.

## What's going on

PAutomaC problems are i.i.d. samples from non-deterministic
probabilistic automata with 5–80 hidden states. The training sets
are large (20k–100k sequences). At this scale, GDC's prefix-memory
strategy is near-ideal: it has enough training prefixes to interpolate
all common transitions, and `theta`-driven self-loop smoothing
covers the long-tail.

CHMM with K ∈ {2, 4, 8} has hidden-state count `K × (A+1)`. For
problem 2 (A=18) that's 38 / 76 / 152 states, well below the typical
PAutomaC target's 30-80 states — and CHMM's EM is a single seed,
which we already know is fragile (cf the K=4 anomalies in
`algorithmic_benchmarks/`). A larger K and multi-seed EM would close
some of the gap.

## A note on the `uniform` row

The `uniform` model in this run computes `pM(t) = (1/(A+1))^(T+1)`
— a length-aware uniform that penalises long sequences. After
normalising over the test set, the longer test sequences get *more*
probability mass than uniform-over-test-set (1/N), which is **worse
than the official uniform baseline** when the true test distribution
favours short sequences (problem 5 in particular).

The official PAutomaC `uniform baseline` is `score = N_test`, set in
the `lift` denominator. The `score` column in the table is the model's
absolute score; the `lift` column compares against the official
`N_test` baseline. So `uniform-row.score > 1000` simply means
"length-aware uniform is worse than equal-mass-over-test-set."

Both interpretations are valid baselines; the official one is what
the lift uses.

## Cost summary

| model | fit time (mean) | eval time (mean) | notes |
|---|---:|---:|---|
| uniform / unigram / bigram | <0.1s | <0.1s | trivial |
| CHMM K=2 | ~6s | <1s | numba-JIT'd |
| CHMM K=4 | ~7s | <1s | |
| CHMM K=8 | ~9s | <1s | |
| GDC | ~0.2s | **~110s** | one transition matvec per test step |

GDC eval is the bottleneck. At 100s/problem × 48 problems = ~80 min
for a full sweep. Could be made faster by caching the transition
matrix and batching forecast across test sequences, but for the
initial benchmark this is fine.

## Next steps

1. Sweep more GDC hyperparameter configs (mirror what the algorithmic
   benchmarks taught us — `α ∈ {0.5, 0.7, 0.9, 0.95}`, `θ ∈ {0.005,
   0.05}`, `transition ∈ {self_loop, two_step}`).
2. Larger CHMM K (16, 32) and multi-seed EM.
3. Run the full 48-problem set.
4. Report the headline metric the PAutomaC leaderboard uses
   (perplexity gap to entropy floor, geometric mean across all 48).
5. (paper-plan tier) add baselines listed in §10 of the paper plan:
   PPM, CTW, BCT, Sequence Memoizer, FlexFringe.

## Reproduce

```bash
python pautomac/download.py
python pautomac/run_eval.py --problems 1-5
```

Outputs:
* `pautomac/results/pautomac_results.csv` — per (problem, model) row
* `pautomac/results/pautomac_results.log` — full stdout
* `pautomac/run_5.log` — same content (tee'd)

Total runtime ~12 minutes (GDC eval is the bottleneck).
