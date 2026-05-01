# ALERGIA on the algorithmic benchmarks

After ALERGIA's strong showing on PAutomaC (lowest geometric-mean
gap-to-floor of all our default-settings models), we ran the same
default-config ALERGIA (`aalpy.run_Alergia(eps=0.05)`) on the four
TM tasks plus Dyck-1 from `algorithmic_benchmarks/`.

## Headline table — ALERGIA alongside tuned GDC and best CHMM

`mean acc` is per-position accuracy on `(read, write, dir)` for TM
tasks (or next-symbol accuracy for Dyck-1). All errors are
`tuple_errors / total_predictions`. **Bold** = best model in row.

| task | variant | model | states | mean acc | errors | perfect |
|---|---|---|---:|---:|---:|---:|
| parity | original | ALERGIA eps=0.05 | **5** | 0.9921 | 12 / 506 | 8 / 20 |
| | | GDC tuned (α=0.5, θ=0.05, 2-step) | 1,901 | **0.9934** | **8 / 506** | **12 / 20** |
| | | CHMM K=2 | 8 | 0.9921 | 12 / 506 | 8 / 20 |
| parity | noread | ALERGIA | 5 | 0.9921 | 12 / 506 | 8 / 20 |
| | | GDC tuned | 1,901 | **0.9934** | **8 / 506** | **12 / 20** |
| | | CHMM K=2 | 8 | 0.9921 | 12 / 506 | 8 / 20 |
| increment | original | ALERGIA | 30 | 0.9950 | 3 / 266 | 18 / 20 |
| | | GDC / CHMM | 12-2073 | **1.0000** | **0 / 266** | **20 / 20** |
| increment | noread | **ALERGIA** | **21** | **1.0000** | **0 / 266** | **20 / 20** |
| | | GDC / CHMM | 10-2073 | 1.0000 | 0 / 266 | 20 / 20 |
| reverse | original | ALERGIA | 72 | 0.9803 | 601 / 13,646 | 0 / 20 |
| | | **GDC tuned** (α=0.95, θ=0.05) | 34,818 | **0.9918** | **149 / 13,646** | **6 / 20** |
| | | CHMM K=8 | 120 | 0.9852 | 391 / 13,646 | 0 / 20 |
| reverse | **noread** | ALERGIA | 272 | 0.8492 | **6,011 / 13,646** | 0 / 20 |
| | | **GDC tuned** | 34,818 | **1.0000** | **0 / 13,646** | **20 / 20** |
| | | CHMM K=8 | 72 | 0.9966 | 140 / 13,646 | 0 / 20 |
| binary_adder | original | ALERGIA | 18 | 0.9675 | 5,579 / 72,217 | 0 / 10 |
| | | GDC tuned (α=0.5, self_loop) | 54,991 | 0.9999 | 59 / 72,217 | 0 / 10 |
| | | **CHMM K=4** | 40 | **1.0000** | **10 / 72,217** | 0 / 10 |
| binary_adder | noread | ALERGIA | 16 | 0.9932 | 1,466 / 72,217 | 0 / 10 |
| | | **GDC tuned** (α=0.9, θ=0.05, 2-step) | 54,991 | **1.0000** | **0 / 72,217** | **10 / 10** |
| | | CHMM K=4 | 40 | 1.0000 | 0 / 72,217 | 10 / 10 |
| dyck1 | n/a | ALERGIA | 12 | 0.5003 | 5,557 / 11,120 | — |
| | | GDC | 17,072 | 0.5367 | 5,059 / 10,920 | — |
| | | **CHMM K=8** | 24 | **0.5882** | **4,497 / 10,920** | — |

ALERGIA training + eval times across all 9 task-variants total **<2 seconds**.

## Per-task readings

### Parity — ALERGIA matches CHMM K=2; both behind tuned GDC

* ALERGIA produces a 5-state PDFA — exactly the right size for the
  Parity TM (states SCAN0, SCAN1, plus halt/sink). It captures the
  same structure as CHMM K=2 (8 states) and gets identical accuracy
  (0.9921, 12 errors, 8/20 perfect).
* Tuned GDC (α=0.50, θ=0.05, 2-step) still leads with 8 errors / 12
  perfect tapes thanks to its prefix-memorising bias.
* No-read makes no difference (Parity has no pass-through transitions).

### Increment — ALERGIA hits 100% under no-read, slightly behind under original

* Original variant: ALERGIA misses 3 / 266 (98.5% perfect tapes) —
  off by 1 vs the perfect GDC/CHMM solutions.
* **No-read variant: ALERGIA reaches 100% / 20 of 20 perfect tapes**
  with a 21-state model. This is the regime where ALERGIA's
  state-merging is tightest — pass-through transitions are collapsed
  into a single emission, and ALERGIA's Hoeffding test merges them
  cleanly.

### Reverse — original: ALERGIA mid-pack; no-read: ALERGIA collapses

* Original: ALERGIA 4.4% errors with 72 states, between CHMM K=8
  (2.87%, 120 states) and GDC tuned (1.09%, 34,818 states).
* **No-read: ALERGIA's accuracy collapses to 85% mean (44% tuple
  errors)** despite having more states (272). The state-merging is
  apparently making the wrong decisions on the no-read variant —
  likely because the increased pass-through aliasing puts more
  pressure on the Hoeffding test, and at default `eps=0.05` it
  over-merges.
* Tuned GDC reaches 100% / 20 of 20 perfect tapes on this variant —
  the structural advantage of prefix memory dominates.

### Binary adder — ALERGIA underperforms on both variants

* Original: ALERGIA 7.7% errors with only 18 states. CHMM K=4 gets
  0.014% with 40 states. ALERGIA is **wildly under-merging** here —
  the Hoeffding test merges more states than the underlying TM
  actually has.
* No-read: ALERGIA improves to 2.0% errors but still 200× worse than
  the perfect GDC tuned / CHMM K=4 solutions.
* The binary adder has 7 underlying TM states and 10 reduced-tuple
  emissions. ALERGIA's ~16-state PDFA is the right *order* but
  evidently misses the long-range structure (B-decrement and
  A-increment carry chains) that CHMM's clones and GDC's prefix
  memory both capture.

### Dyck-1 — ALERGIA at random baseline

* 50.0% next-symbol accuracy with 12 states. Effectively a random
  predictor.
* GDC (53%) is mildly better; CHMM K=8 (59%) is the only model that
  approaches the depth-counter dynamics.
* Counter languages are fundamentally beyond ALERGIA's regular-PDFA
  inductive bias — same limitation as GDC and CHMM, but ALERGIA's
  small state count makes it especially exposed.

## What this changes about the bigger story

The PAutomaC sweep made ALERGIA look like the strongest off-the-shelf
baseline (gmean 2.96, 10 wins). On the algorithmic benchmarks the
picture is more mixed:

| benchmark family | ALERGIA verdict |
|---|---|
| **PAutomaC** (i.i.d. PFA samples, 20k+ training) | **best off-the-shelf** (gmean 2.96) |
| **Small TMs** (Parity, Increment) | ties with CHMM, slightly behind tuned GDC |
| **Long-trace TMs** (Reverse, binary_adder) | substantially worse than tuned GDC/CHMM |
| **Counter languages** (Dyck-1) | random-baseline, like GDC |

The interpretation: ALERGIA is **excellent at recovering compact
PDFA structure when training data are abundant relative to state
count**. PAutomaC is exactly that regime. Algorithmic-trace tasks
have the OPPOSITE regime — small training sets (300 sequences), long
sequences (50–600 steps), and underlying structure that's *not* a
small PDFA but a deterministic algorithm with stack-like or
counter-like state. ALERGIA's state-merging is too aggressive there.

This dovetails with the existing GDC vs CHMM finding: prefix-memory
(GDC) wins on long-trace deterministic algorithms; clone-merging
(CHMM, ALERGIA) wins on i.i.d. PFA samples. The default-settings
ALERGIA in particular is **especially fragile** on long traces with
default `eps=0.05` — a sweep over `eps` would help but won't change
the qualitative picture.

## Updated comparison ladder for algorithmic benchmarks

Across all 9 task-variants, "winner" of each:

| task / variant | winner | runner-up | ALERGIA rank |
|---|---|---|---|
| parity (both) | GDC tuned | CHMM K=2 ≈ ALERGIA (tied) | tied 2nd |
| increment original | GDC = CHMM (perfect) | — | ~2nd (3/266 vs 0) |
| **increment noread** | **GDC = CHMM = ALERGIA (all perfect)** | — | tied 1st |
| reverse original | GDC tuned | CHMM K=8 | 3rd |
| reverse noread | **GDC tuned (perfect)** | CHMM K=8 | 4th (worst) |
| binary_adder original | CHMM K=4 | GDC tuned | 4th (worst) |
| binary_adder noread | **GDC = CHMM (both perfect)** | — | 3rd |
| dyck1 | CHMM K=8 | GDC | tied 3rd |

ALERGIA at default eps wins / ties on **2 of 9** task-variants, both
on the easiest tasks (parity, increment-noread). On the harder
algorithmic tasks (reverse, binary_adder) it's clearly behind both
tuned GDC and best CHMM.

## Reproduce

```bash
python algorithmic_benchmarks/run_alergia.py
```

Outputs:

* `algorithmic_benchmarks/alergia_results.csv` — per task-variant row
* `algorithmic_benchmarks/alergia_run.log` — full stdout

Total runtime: <2 seconds. ALERGIA is by far the cheapest method
tested on these benchmarks.

## Caveats

* Single eps value (0.05). The reverse-noread collapse to 85% accuracy
  suggests `eps` matters a lot here. A sweep `eps ∈ {0.01, 0.02, 0.05,
  0.1, 0.2}` would clarify.
* Single-seed; ALERGIA is deterministic given training data and eps,
  but a sweep over different shuffles of training data could check
  robustness.
* The binary_adder ALERGIA has only 16-18 states — strikingly low.
  Almost certainly under-merging; a tighter eps (e.g. 0.01) would
  let more states survive.
