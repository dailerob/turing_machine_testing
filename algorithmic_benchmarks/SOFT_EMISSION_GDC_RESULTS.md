# Soft-emission GDC on the algorithmic benchmarks

## What I added

The default GDC enforces a deterministic emission: state `s` emits
its stored vector `states[s, :]` exactly. The only "softness" in the
existing code is the `beta` parameter inside
[`_emission_probabilities`](../generative_dense_chain.py:340), which
softens the **likelihood** function used during forward-pass posterior
updates — but not the **prediction** step.

The new `SoftEmissionGDC` wrapper
([soft_gdc.py](soft_gdc.py)) adds a per-position emission-noise
parameter `eta ∈ [0, 1]` used at *prediction* time:

    P(emit_i = a | state s) = (1 - eta) · 1{states[s, i] == a}
                            + eta / V_i

where `V_i` is the alphabet size at position `i`. With `eta = 0` this
reduces to deterministic emission.

For each test step, the wrapper:

1. Computes a **soft posterior** over states given the conditional
   positions (using the eta-noise emission as the likelihood):
   `log_post[s] = log state_dist[s] + Σ_{i ∈ cond} log P(emit_i = c_i | s)`
2. Enumerates candidate tuples seen in training that match the
   conditional positions.
3. For each candidate `t`, computes the joint probability
   `P(emit t | history) = Σ_s posterior[s] · ∏_i P(emit_i = t_i | s)`.
4. Returns the `argmax` candidate tuple.

## Results: eta sweep across all 9 task-variants

Tuned GDC hyperparameters from `TUNED_GDC_RESULTS.md`. eta ∈ {0.0,
0.01, 0.05, 0.1, 0.2, 0.3}. Errors reported as
tuple-errors / total-positions.

### Original variant

| task | eta=0.0 | eta=0.01 | eta=0.05 | eta=0.1 | eta=0.2 | **eta=0.3** | best eta |
|---|---:|---:|---:|---:|---:|---:|---|
| parity | 11/506 | 11 | 11 | 11 | 11 | 12 | 0.0–0.2 (tied) |
| increment | 1/266 | 1 | 1 | 1 | 1 | 1 | unchanged |
| reverse | **162** / 13,646 | 582 | 622 | 631 | 658 | 691 | **0.0** (eta hurts) |
| binary_adder | 1,108 / 72,217 | 1,101 | 1,048 | 971 | 770 | **518** | **0.3** (eta helps) |

### No-read variant

| task | eta=0.0 | eta=0.01 | eta=0.05 | eta=0.1 | eta=0.2 | eta=0.3 |
|---|---:|---:|---:|---:|---:|---:|
| parity | 11/506 | 11 | 11 | 11 | 11 | 12 |
| increment | **0/266 (perfect)** | 0 | 0 | 0 | 0 | 0 |
| reverse | **0/13,646 (perfect)** | 0 | 0 | 0 | 0 | 0 |
| binary_adder | **0/72,217 (perfect)** | 0 | 0 | 0 | 0 | 0 |

For the no-read variant on the 3 hardest TM tasks, eta=0 is already
at the floor — soft emission can't help. Higher eta either ties or
slightly degrades.

## Where soft emission helps and hurts

* **Helps decisively on binary_adder original** (the OOD-content
  regime): eta=0.3 reduces errors from 1108 → 518, a **53%
  reduction**. The mechanism: when test traces contain bit patterns
  unseen in training (longer numbers), soft emission lets states
  "vote for" tuples they don't exactly store, which gives the model
  graceful degradation under content extrapolation.
* **Hurts on reverse original**: eta from 0 to 0.3 grows errors
  3-4×. The reverse trace has many states sharing the same emission;
  softening the emission lets *wrong* states contribute mass to
  candidate tuples and the argmax flips.
* **Doesn't matter on parity / increment**: too easy to be sensitive
  to eta.

## Important caveat: joint-tuple vs single-state inference

The eta=0 numbers in the table above **don't match** the earlier
`TUNED_GDC_RESULTS.md` numbers exactly. For example:

| task / variant | TUNED_GDC report (greedy_sample) | This run, eta=0 |
|---|---:|---:|
| parity original | 8/506 | 11/506 |
| reverse original | 149/13,646 | 162/13,646 |
| binary_adder original | 59/72,217 | 1,108/72,217 |

This is because my soft predict uses **joint-tuple-mass argmax**
(marginalising posterior across all states with the same tuple),
while the original `greedy_sample` uses **single-state argmax** (pick
the single highest-mass state's tuple).

For per-position Hamming loss, joint-tuple argmax is *theoretically*
the optimal decision rule. But empirically on these tasks — especially
binary_adder original — single-state argmax wins by a large margin
(59 vs 1108 errors at the same hyperparameters). The interpretation:
GDC's posterior places its mass primarily on the *correct* training-
prefix state, but spurious mass leaks onto unrelated states with the
same emission. Marginalising over states sharing a tuple averages in
this spurious mass and degrades the decision.

This means **two changes are confounded** in the soft-emission run:

1. The decision rule changed from single-state argmax to tuple-
   marginal argmax. (This is a regression on its own at eta=0.)
2. Soft per-position emission noise was added.

The cleanest soft-emission experiment would be:

* **Variant A** (single-state, the original greedy_sample): pick
  best state from soft posterior, output its stored tuple. No per-
  position eta in the OUTPUT (only in the likelihood used for
  posterior). This matches the original at eta=0.
* **Variant B** (this implementation): joint-tuple argmax over
  candidates, with soft per-position emission contributing to
  candidate scoring.

I implemented (B). The binary_adder result (518 errors at eta=0.3
vs 1108 at eta=0) shows soft emission still helps within (B), but
the absolute ceiling is bound by the inferior decision rule. Variant
(A) at eta=0 already gives 59 errors; whether soft emission lifts it
further is open.

## Practical conclusion

Adding soft emissions to GDC is **possible and partly useful**:

* On the binary_adder original, soft emissions cut errors by ~50%
  within a fixed decision rule.
* On most other tasks, soft emissions don't help, and on Reverse
  original they hurt.
* The biggest gain is *not* soft emissions — it's keeping the
  original single-state-argmax decision rule. Joint-tuple
  marginalisation, while theoretically tighter, leaks spurious mass
  on these long-trace tasks.

A stronger experiment would re-implement single-state argmax with
soft-posterior weighting (Variant A above) and sweep eta there. That
would isolate the "soft emission helps OOD content" claim from the
"joint-tuple argmax loses" regression. Worth doing if we want a
crisp claim about soft emissions, but the headline finding ("soft
emission helps the binary adder by ~50% within (B)") is already
visible.

## Cost

| task / variant | GDC train | eta=0 eval | eta=0.3 eval |
|---|---:|---:|---:|
| parity | <0.01s | 0.1s | 0.1s |
| increment | <0.01s | 0.04s | 0.04s |
| reverse | 0.02s | 33-36s | 33s |
| binary_adder | 0.03s | 263s | 269s |

Soft predict is ~3× slower than original `greedy_sample` due to the
per-tuple `per_state` computation across all `n_states`. Could be
made faster by precomputing the per-state×per-position indicator
matrix once and re-using per step.

## Reproduce

```bash
python algorithmic_benchmarks/run_soft_gdc.py
```

Outputs:

* `algorithmic_benchmarks/soft_gdc_results.csv` — per (task, variant, eta)
* `algorithmic_benchmarks/soft_gdc_run_v2.log` — full stdout
* `algorithmic_benchmarks/soft_gdc.log` — pretty summary
