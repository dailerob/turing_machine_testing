# GDC `terminal_behavior='absorb'` on algorithmic benchmarks

## TL;DR

**Absorb mode and diffuse mode produce IDENTICAL predictions on every
algorithmic benchmark we tested.** Same error counts, same perfect
tapes, byte-for-byte equivalent. This is not a bug — it's a
structural property of the argmax-greedy prediction protocol used on
these tasks. The mathematical explanation is in §3 below.

## Per-task comparison

Per-task tuned configs from [TUNED_GDC_RESULTS.md](TUNED_GDC_RESULTS.md);
diffuse vs absorb measured side by side.

| task | variant | best config | diffuse errors | absorb errors | diffuse perfect | absorb perfect |
|---|---|---|---:|---:|---:|---:|
| parity | original | α=.5, θ=.05, two-step | 8/506 | 8/506 | 12/20 | 12/20 |
| parity | noread | same | 8/506 | 8/506 | 12/20 | 12/20 |
| increment | original | α=.5, θ=.005, self_loop | 0/266 | 0/266 | 20/20 | 20/20 |
| increment | noread | same | 0/266 | 0/266 | 20/20 | 20/20 |
| reverse | original | α=.95, θ=.05, self_loop | 149/13646 | 149/13646 | 6/20 | 6/20 |
| reverse | noread | same | 0/13646 | 0/13646 | 20/20 | 20/20 |
| binary_adder | original | α=.5, θ=.005, self_loop | 59/72217 | 59/72217 | 0/10 | 0/10 |
| binary_adder | noread | α=.9, θ=.05, two-step | 0/72217 | 0/72217 | 10/10 | 10/10 |
| dyck1 | n/a | α=.95, θ=.05, self_loop | 5118/11120 | 5118/11120 | n/a | n/a |

**Total runtime: 415s on 12 workers** (the binary adder cells dominate
because of the 100k-state GDC; everything else finishes in seconds).

## Why they're identical: argmax invariance

The transition kernels in the two modes differ in how they handle
mass at terminal training positions:

* **Diffuse**: terminal mass redistributes uniformly to all
  non-terminal states (`+diffuse_t · terminal_sum` to each non-terminal).
* **Absorb**: terminal mass leaks out of the active distribution.

Critically, the diffuse-mode contribution is a **constant additive
term per non-terminal state**: every non-terminal candidate receives
exactly `diffuse_t · terminal_sum` more probability mass. The relative
ordering of candidate states is **unchanged** by adding the same
constant to all of them.

Algorithmic benchmarks use **greedy argmax with conditional**
prediction: at each step, take the argmax over hidden states whose
read field matches the actual next read. The argmax is **invariant
to additive constants**. So:

* Diffuse adds a constant (terminal mass / # non-terminals) to each
  non-terminal candidate.
* The argmax over candidates is the same with or without that constant.
* Predictions are byte-identical.

This holds whenever predictions go through `argmax`. It would NOT
hold for predictions based on probability-weighted sums (continuous
expected values, MSE-against-distribution metrics) — see the HMM
forecasting and M4 hourly results, where absorb genuinely differs
from diffuse.

## Where absorb DOES help (for context)

| benchmark style | argmax-based? | absorb vs diffuse |
|---|---|---|
| **Algorithmic / Turing trace prediction** | **yes** (greedy + conditional) | **identical** |
| HMM forecasting (MSE against soft posterior) | no (full distribution) | regime-dependent — see [HMM_EXPERIMENTS_SUMMARY.md §2.10](../hmm_comparison/HMM_EXPERIMENTS_SUMMARY.md) |
| M4 hourly (continuous E[next value]) | no (probability-weighted sum) | absorb wins universally — see [M4_HOURLY_RESULTS.md](../m4/M4_HOURLY_RESULTS.md) |

## Verification

A direct test confirms the absorb implementation is genuinely
different from diffuse on these inputs (not silently disabled):

```
3 sequences of length 3, terminals at indices [2, 5, 8]
Mass at terminal (idx 2):
  diffuse: [0.119, 0.119, 0.05, ...] sum=1.0000
  absorb:  [0.0,   0.0,   0.05, ...] sum=0.0500
  ARE IDENTICAL: False  ← they differ in raw distribution
  argmax DIFFUSE: 1, argmax ABSORB: 1  ← same prediction
```

The state distributions are demonstrably different; the argmax
collapses them to the same prediction.

## Reproduce

```bash
python algorithmic_benchmarks/run_absorb_compare.py
```

Outputs:

* `absorb_results.csv` — per (task, variant, mode) row
* `absorb_compare.log` — full stdout
* `ABSORB_RESULTS.md` — this file
