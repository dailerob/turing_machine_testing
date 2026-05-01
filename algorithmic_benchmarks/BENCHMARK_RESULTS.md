# GDC vs CHMM on algorithmic tasks — initial benchmark

Four tasks, all with **out-of-distribution length / depth**:
training inputs are short, test inputs are several times longer.
Protocol matches the binary-adder experiment: TM-trace tasks are
evaluated 1-step-ahead conditional on the actual next read symbol,
on the reduced 3-column `(read, write, dir)` view (so the hidden
TM state must be inferred). Dyck-1 is a sequence task; we report
next-symbol accuracy on positions where the next token is `(` or `)`
(skipping `END`).

GDC: `alpha=0.99, theta=0.005, transition_type=self_loop_two_step,
initial_dist=sequence_starts` (the reduced-3-col config from the
binary-adder writeup). CHMM: `K ∈ {2, 4, 8}` clones per emission,
50 EM iters, single dummy action channel.

## Headline table

| task | model | config | hidden states | mean acc | errors | perfect tapes |
|---|---|---|---:|---:|---:|---:|
| parity (test 16–32, train 3–8) | GDC | α=0.99 | 1,901 | 0.9934 | 10 / 506 | 10 / 20 |
| | CHMM | K=2 | **8** | 0.9921 | 12 / 506 | 8 / 20 |
| | CHMM | K=4 | 16 | 0.9921 | 12 / 506 | 8 / 20 |
| | CHMM | K=8 | 32 | 0.9921 | 12 / 506 | 8 / 20 |
| increment (test 8–12 bits, train 1–5) | GDC | α=0.99 | 2,073 | **1.0000** | **0 / 266** | **20 / 20** |
| | CHMM | K=2 | **12** | **1.0000** | **0 / 266** | **20 / 20** |
| | CHMM | K=4 | 24 | 1.0000 | 0 / 266 | 20 / 20 |
| | CHMM | K=8 | 48 | 1.0000 | 0 / 266 | 20 / 20 |
| reverse (test 10–16 bits, train 3–6) | **GDC** | α=0.99 | 34,818 | **0.9918** | **252 / 13,646** | 0 / 20 |
| | CHMM | K=2 | 30 | 0.9865 | 329 / 13,646 | 0 / 20 |
| | CHMM | K=4 | 60 | 0.9836 | 449 / 13,646 | 0 / 20 |
| | CHMM | K=8 | 120 | 0.9852 | 391 / 13,646 | 0 / 20 |
| dyck1 (test depth 8, train depth 4) | GDC | α=0.99 | 17,072 | 0.5313 | 5,118 / 10,920 | — |
| | CHMM | K=2 | 6 | 0.5444 | 4,975 / 10,920 | — |
| | CHMM | K=4 | 12 | 0.5777 | 4,612 / 10,920 | — |
| | **CHMM** | **K=8** | **24** | **0.5882** | **4,497 / 10,920** | — |

## Per-task interpretation

### Parity TM — effectively tied; CHMM 60–240× more compact

CHMM K=2 already saturates: at 8 hidden states it matches K=4 (16
states) and K=8 (32 states) byte-for-byte. The Parity machine has
exactly 2 reachable hidden states (`SCAN0`, `SCAN1`) plus halt, so
K=2 has *exactly enough capacity to clone every TM state*.

Both models miss ~2% of writes. The errors are at the final step of
many test tapes: the parity bit must be written at the trailing
blank, but training tapes had blank at positions 4–9, while test
tapes have blanks at positions 17–33. Whichever clone got the
"writing-at-blank" emission during EM was conditioned on the very
short training contexts; long contexts at test time push posterior
mass to the wrong clone. GDC has the same problem (10 errors, all
at the same boundary).

**Verdict.** Tied on accuracy. CHMM K=2 wins on parsimony — 8
states vs GDC's 1,901, a 240× ratio.

### Increment TM — both perfect; CHMM 170× more compact

20/20 perfect tapes for every model. The increment machine is
small (4 states) and the carry-chain pattern is fully exposed in
training (most training inputs already contain a few trailing 1s).
This is the cleanest "compression-is-correct" case in the suite:
CHMM at K=2 (12 states) is structurally enough; GDC's 2,073 prefix
states deliver no extra information.

**Verdict.** Tied on accuracy at 100%. CHMM K=2 wins decisively
on parsimony.

### Reverse TM — GDC modestly better; CHMM still 290× more compact

GDC's 34,818 hidden states give it a small but real edge (1.85%
errors vs CHMM's best 2.41%). The reverse machine has 10 hidden
states sharing 15 emission tuples; multiple states emit the same
(read, write, dir) under the reduced view. With training inputs
3–6 bits but test inputs 10–16 bits, mean test trace length is
**684 steps vs 117 steps in training** — a 6× length gap.

A surprising sub-finding: **CHMM K=4 is worse than K=2** (449 vs
329 errors). The K=2 model is forced into a more constrained
clone-to-state mapping; K=4 has more degrees of freedom and EM
appears to converge to a slightly worse local optimum on this
limited training set. K=8 partially recovers (391). This is a real
hyperparameter-sensitivity issue we'd want to address with
multi-seed initialisation in a paper-grade run.

**Verdict.** GDC wins. The reverse machine is the kind of
"deep-prefix-coordination" task where GDC's memorisation of the
exact bit pattern gives it more usable signal than CHMM's
EM-merged clones.

### Dyck-1 — both weak; CHMM consistently leads, gap grows with K

Both models are barely above the 50% random baseline. Neither has
a true counter, so generalising from depth 4 to depth 8 is
fundamentally beyond the architectures.

That said, CHMM **consistently leads** GDC, and the gap grows
monotonically with K: 54.4% (K=2) → 57.8% (K=4) → 58.8% (K=8) vs
GDC's 53.1%. CHMM's clones are giving it a soft approximation of
"where in the depth stack we are", and more clones = better
approximation. GDC's prefix memory plateaus around training depth.

**Verdict.** CHMM wins. Counter dynamics are exactly the regime
where compressed latent state beats prefix memorisation, even when
both methods are weak in absolute terms.

## Why the binary-adder result is special

On the binary adder (the original CHMM-vs-GDC headline), CHMM K=4
made 10 errors total vs GDC's 94 — a 9× win for CHMM. Now we have
a fuller picture:

| task | winner | gap |
|---|---|---|
| binary-adder | CHMM K=4 | 9× fewer errors |
| reverse | GDC | 1.3× fewer errors |
| parity | GDC (slight) | 1.2× fewer errors |
| increment | tied | both perfect |
| dyck-1 | CHMM K=8 | ~1.14× more correct |

The binary adder is the **largest** of the algorithmic tasks (107k
training tokens vs 2k–35k for the others). CHMM benefits from EM
having enough training data to fully resolve clone identities. On
small-data tasks (parity, increment, reverse), EM has less signal
and either hits a local optimum (reverse K=4) or doesn't pick up
the rare emissions (parity boundary write).

GDC's prefix memorisation is more robust on small data — it makes
no compression compromises and just stores everything. The cost is
state count (1.9k–35k vs CHMM's 8–120) and a small accuracy gap
on tasks where prefix identity is genuinely informative.

**Updated paper-plan framing.** §1.5 of `RELATED_WORK_AND_PAPER_PLAN.md`
already says CHMM and GDC are "the maximum-compression and
maximum-fidelity ends of the same design axis". This benchmark
makes that axis quantitative:

* **Compression-is-correct + plenty of data** → CHMM dominates
  (binary adder).
* **Compression-is-correct + small data** → tied or slight CHMM
  edge in compactness, GDC edge in accuracy (parity, increment,
  reverse).
* **Counter / depth-dependent** → CHMM has a small edge by
  approximating the counter as a Markov chain over clones; GDC's
  prefix memory caps at training depth.

## Caveats and follow-ups

* **N_train**: 300 tapes for the TM tasks is small. Sample-efficiency
  curves (à la `paper_topology_and_samples.py` EXP2) would tell us
  whether CHMM's reverse-task gap closes with more data.
* **EM seed sensitivity**: the Reverse-TM K=4 < K=2 result wants
  multi-seed averages. A single seed shouldn't determine which K
  is "best".
* **Length extrapolation curves**: train at fixed length, test at
  geometric ladder of lengths. Both models are expected to degrade,
  but the *shape* of the degradation will differ qualitatively
  between GDC's prefix memory and CHMM's clone graph.
* **No multiplication TM yet**: dropped from this initial pass for
  scope. Follow-up.
* **Dyck-2 (two paren types)** would be a stronger counter test;
  Dyck-1 is too easy to memorise at small depth.

## Reproduce

```bash
python algorithmic_benchmarks/test_generators.py    # sanity tests
python algorithmic_benchmarks/run_benchmarks.py     # ~30 seconds
```

Outputs:
* `benchmark_results.csv` — one row per (task, model, config)
* `benchmark_log.txt` — full stdout
* `BENCHMARK_RESULTS.md` (this file)
