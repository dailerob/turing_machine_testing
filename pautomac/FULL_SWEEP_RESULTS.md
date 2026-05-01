# Full PAutomaC sweep — 48 problems × 8 models

## Headline

**GDC (best of 2 configs) wins 43 / 48 problems** against CHMM (best of
K ∈ {2, 4, 8}). Bigram wins 1 problem; CHMM-K8 wins 4.

| metric | uniform | unigram | bigram | CHMM K=2 | CHMM K=4 | CHMM K=8 | GDC α=.95 θ=.05 | GDC α=.50 θ=.005 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean gap (perplexity) | 1271 | 636 | 66.6 | 71.4 | 55.5 | 42.1 | **19.3** | **20.7** |
| median gap | 107.0 | 82.6 | 27.8 | 31.4 | 23.0 | 20.3 | **5.86** | **10.05** |
| **gmean(gap / uniform_gap)** | 1.000 | 0.694 | 0.170 | 0.224 | 0.180 | 0.149 | **0.055** | **0.068** |
| n problems won | 0 | 0 | 1 | 0 | 1 | 4 | **23** | **19** |
| mean train time | 0s | 0s | 0s | 9s | 10s | 13s | 0.3s | 0.3s |
| mean eval time | 0s | 0s | 0s | 0s | 0s | 0s | 84s | 84s |

The cleanest single-number summary is **`gmean(gap / uniform_gap)`** —
the geometric mean across 48 problems of "what fraction of the
uniform→floor distance is left as model error." Lower is better.

* GDC α=0.95: **5.5%** of the gap remains.
* GDC α=0.50: **6.8%** remains.
* CHMM K=8: 14.9% remains.
* Bigram: 17.0% remains.

## Head-to-head: best-GDC vs best-CHMM

| | count |
|---|---:|
| GDC wins (gap smaller by > 1e-6) | **43 / 48** |
| CHMM wins | 5 / 48 |
| ties | 0 / 48 |
| mean (CHMM_gap − GDC_gap) | +33.8 |
| median (CHMM_gap − GDC_gap) | +9.5 |

The 5 problems where CHMM wins:

| problem | A | floor | best CHMM | best GDC | margin |
|---:|---:|---:|---:|---:|---:|
| 7 | 13 | 51.2 | **8.4 (K=8)** | 14.4 | -6.0 |
| 17 | 13 | 47.3 | **10.7 (K=8)** | 20.5 | -9.9 |
| 18 | 20 | 57.3 | **1.9 (K=8)** | 4.6 | -2.7 |
| 21 | 23 | 30.5 | **26.5 (K=4)** | 33.9 | -7.3 |
| 39 | 14 | 10.0 | **0.34 (K=8)** | 1.45 | -1.1 |

These tend to be moderate-alphabet (13-23) problems where CHMM-K=8's
80-180 hidden states happen to recover the target structure
particularly well. Problem 39 has a very low entropy floor (10.0) and
CHMM K=8 nearly reaches it (0.34); GDC's two configs give 0.34→1.45
and 1.98 — close but not as close.

The 1 problem where bigram wins (45) is also one where GDC α=0.50 is
within 0.02 of bigram. Essentially a tie.

## Configuration sweet spots

* **GDC α=0.95, θ=0.05** wins 23 / 48. Tends to win on problems with
  longer sequences and more emission ambiguity (the high-α regime
  preserves a sharper prefix-memorising filter).
* **GDC α=0.50, θ=0.005** wins 19 / 48. Tends to win on problems with
  lower entropy floors and more bigram-like structure (lower α
  diffuses faster through the prefix tree, behaving closer to a
  smoothed n-gram).
* **CHMM K=8** wins 4 / 48 — but is consistently the best CHMM config
  (mean gap 42.1 vs 71.4 for K=2). K=4 only wins 1 / 48.

The GDC-α split confirms what the algorithmic-benchmark sweep
showed: the GDC optimum is task-dependent and a 2-config sweep is the
minimum honest evaluation.

## Cost analysis

| stage | cost |
|---|---:|
| CHMM full sweep (3 K × 48 problems) | ~25 minutes |
| GDC full sweep (2 configs × 48 problems) | **~135 minutes** |
| Bigram / unigram / uniform | seconds |
| **Total** | ~160 minutes |

GDC eval was the bottleneck. Per-problem GDC eval averaged 84s, vs
near-zero for everything else. **This was due to a bug** in
[fast_gdc.py](fast_gdc.py): the batched scorer ran every sequence to
`max_test_len ≈ 70` timesteps when the *mean* test length is ~10. So
the batched version did roughly **7× more total work** than the naive
per-sequence scorer (which only runs each sequence for its actual
length). The naive scorer would have averaged ~12s/problem, halving
the total runtime to ~80 minutes.

## On the "sparse linear operator" detour

The original three optimisations I outlined were:

1. ✓ Cache per-call invariants — small win (<2×).
2. ✗ **Pack the transition into a sparse matmul** — *redundant*. The
   existing `_transition_self_loop` is already O(n) via the rank-1
   perturbation trick (`diffuse · sum · 1_n − diffuse · per_element`)
   plus an array-slice shift. There is no `n × n` matrix to compress.
3. ◐ **Batch test sequences** — works in principle, but the current
   implementation runs every sequence to `max_test_len` instead of
   shrinking the active set as sequences finish. Net: 7× *more* total
   work than naive.

The proper batching fix:

```python
# Sort test sequences by length descending; drop rows from the (B, n)
# state matrix as they finish.
order = np.argsort([-len(s) for s in test_seqs])
S = ...  # (B, n)
for t in range(max_len):
    if t > 0:
        S = transition_batch(S)
    active_now = lengths_sorted > t
    if not active_now.any(): break
    # Condition active rows on this step's symbols
    ...
    # Shrink S to drop rows that are now inactive
    S = S[active_now]
```

With shrinking, total work is `Σ_t |active_at_t| · n ≈ mean_len · B · n`
— the same as naive — but each step is one wide vector op instead of
B narrow ones. Realistic speedup over naive: ~3–5× from BLAS
broadcasting, *without* the 7× regression we hit here.

Worth implementing if we want a faster GDC sweep across more
hyperparameter configs (the algorithmic benchmarks showed an 18-point
config grid was useful). For the current 2-config sweep on 48
problems, the existing results are already conclusive.

## Per-problem detail

See [results/full_sweep.csv](results/full_sweep.csv) and
[results/summary.txt](results/summary.txt) for per-problem gap/lift
numbers across all 8 models × 48 problems.

## Reproduce

```bash
python pautomac/run_eval.py --problems all --out pautomac/results/full_sweep.csv
python pautomac/summarize.py
```
