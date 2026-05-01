# CHMM tests

Sandbox for training and evaluating Cloned HMMs (CHMM, Dedieu et al
2019; George et al Nat Comm 2021) on the same synthetic HMM
sequences used by the GDC experiments in `../hmm_comparison/`.

The point: CHMM is GDC's nearest *parametric* cousin
(see §1.5 of `../hmm_comparison/RELATED_WORK_AND_PAPER_PLAN.md`).
This folder gets the upstream code running locally and produces a
first set of comparison numbers.

## Layout

* `naturecomm_cscg/` — upstream code, cloned shallow from
  https://github.com/vicariousinc/naturecomm_cscg. Contains
  `chmm_actions.py` (the CHMM implementation) plus the original
  intro notebook. License MIT (see `naturecomm_cscg/LICENSE`).
* `run_chmm_basic.py` — sanity test: generate sequences from a known
  random HMM, train CHMM with various clone counts `K`, report
  per-symbol negative log-likelihood (bps) on held-out data
  vs the true-HMM Bayes-optimal bps and the stationary-prior bps.

## Usage

```bash
python chmm_tests/run_chmm_basic.py
```

## Notes on the upstream API

* `CHMM(n_clones, x, a, ...)` — `n_clones` is shape `(nA,)`, with
  `n_clones[i]` = number of hidden clones assigned to emission `i`.
  `x` is the int64 observation sequence; `a` is the int64 action
  sequence (same length).
* For a vanilla HMM (no actions), pass `a = np.zeros_like(x)` —
  this gives a single dummy action and recovers a standard CHMM.
* `learn_em_T(x, a, n_iter=...)` runs Baum–Welch on `T` with the
  emission matrix held fixed (each clone deterministically emits
  its parent observation).
* `bps(x, a)` returns negative log2-likelihood per symbol.
* The library is built around a single concatenated sequence. To
  train on multiple short sequences we concatenate them; the
  boundary terms add a small bias but it's negligible at the lengths
  we use here.
