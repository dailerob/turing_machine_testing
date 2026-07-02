# HMM forecasting — sequence-length scaling

How does each method scale as the **per-sequence training length** is
varied? The canonical Table-7 setup uses TRAIN_LEN=50; here we rerun
the full 6-method benchmark at TRAIN_LEN ∈ {25, 50, 100, 200} (half /
canonical / double / quadruple).

## Setup

Identical protocol to `seq_len_sweep.py` (mirror of the Table 7
sweep), only TRAIN_LEN varies. Same regimes, same RNG-derived HMMs,
same val/test seed split, same val-tuning grids.

- 6 regimes × 3 N_train ∈ {25, 100, 400} × 6 seeds (val 3,4,5; test 0,1,2)
- 100 test prefixes per cell, each length 20
- Horizon h=1
- HMM seed depends only on (regime, seed) — NOT on TRAIN_LEN — so the
  same HMM is re-sampled at each TRAIN_LEN for a given (regime, seed),
  isolating the sequence-length effect.
- Methods + val grids:
  - **GDC**: 5 fixed (α, θ, β) configs (the same 5 that sit in
    `run_perplexity_sweep.py`)
  - **CHMM**: $K \in \{4, 16, 32\}$, 50 EM iters
  - **ALERGIA**: ε = 0.05 (single config, no val tune)
  - **Parrot**: $L \in \{1,2,3,4\}$, $K \in \{1,5,25,100,400\}$,
    $\alpha_{\text{prior}} \in \{0.1, 1.0\}$ — 40 configs
  - **HPYLM**: depth ∈ {2,3,4,6}, discount ∈ {0.25,0.5,0.75},
    concentration ∈ {0.5,1.0,5.0} — 36 configs
  - **PPM-D**: depth ∈ {2,3,4,6}, discount ∈ {0.25,0.5,0.75} — 12 configs

Total runtime on 20 CPU workers: TL=25 127 s, TL=50 208 s, TL=100
364 s, TL=200 698 s. Roughly $O(\text{TL})$ — slowest per-cell are
HPYLM/PPM-D (linear-pass counting at every depth) and CHMM EM at K=32.

## Headline: per-method win-count vs TRAIN_LEN

Win counts at horizon $h{=}1$ across all 18 (regime × $N$) cells per
TRAIN_LEN. "Win" = lowest excess perplexity (within 1e-4 ties).

| Method | TL=25 | TL=50 | TL=100 | TL=200 |
|---|---:|---:|---:|---:|
| **GDC**       | **7** | 6 | 5 | 5 |
| **CHMM**      | 4 | **7** | **8** | **9** |
| **ALERGIA**   | 4 | 6 | 3 | 5 |
| **Parrot**    | 0 | 0 | 1 | 0 |
| **HPYLM**     | 3 | 1 | 1 | 0 |
| **PPM-D**     | 1 | 0 | 0 | 0 |

**Three trends:**

1. **GDC is most-wins at TL=25 and stays competitive at every length.**
   No training, so longer sequences don't improve its representation
   directly — the gain is only that the same training-corpus position
   pool is built from richer data.
2. **CHMM scales hardest with TRAIN_LEN.** Its EM needs enough state
   transitions per sequence to identify the latent topology; at TL=25
   it ties for 4 wins (mostly losing dense-and-det regimes to GDC),
   then takes 7→8→9 as the EM has more data to fit. By TL=200 it wins
   every sparse regime outright.
3. **HPYLM is a "small-data" specialist.** At TL=25 its
   Bayesian-nonparametric smoothing wins 3 cells (sparse_small/sparse_large
   at low N); by TL=100+ those wins go to CHMM as the latent-state
   estimator has enough data, and HPYLM falls to 1 then 0 wins. PPM-D
   shows the same pattern more weakly.
4. **Parrot never wins more than 1 cell.** Its best cell across all
   TRAIN_LEN is det_small N=25 at TL=100 (1.0070, marginal beat over
   ALERGIA's 1.0073). Plain kNN-in-prefix-space simply does not match
   any of the structure-aware methods on these regimes.

## Per-regime trends

The story differs by regime:

- **dense regimes** (parameters drawn from broad Dirichlets): GDC
  dominates dense_large at all TRAIN_LEN; ALERGIA edges out dense_small
  at high N. **CHMM never catches GDC on dense_large** — even at
  TL=200, GDC=1.0011/1.0005/1.0003 vs CHMM=1.0823/1.0101/1.0009 at
  N=25/100/400. The dense topology gives CHMM no structural advantage.
- **det regimes** (concentrated emissions): GDC wins at low N every
  length. At TL≥50, CHMM and ALERGIA take det_small / det_large at
  high N as their better estimators benefit from cleaner data.
- **sparse regimes** (topology with fanout=2): **CHMM dominates at
  every TRAIN_LEN ≥ 50.** At TL=25, the EM hasn't seen enough data
  to recover the sparse topology and HPYLM wins one cell; from TL=50
  on, CHMM takes 5 of 6 sparse cells, growing to all 6 by TL=200.
  This is the cleanest "sparse needs latent state" story in the suite.

## Mean "distance to the floor" per method × TRAIN_LEN

Win counts only count "best in cell" outcomes; they hide *how badly*
a method loses when it isn't the winner. The mean log₂ excess
perplexity (averaged over the 18 cells) tells the complementary story
— how far each method sits from the entropy floor on average:

| TRAIN_LEN | GDC | CHMM | ALERGIA | Parrot | HPYLM | PPM-D |
|---:|---:|---:|---:|---:|---:|---:|
| 25  | 0.119 | 0.240 | 0.200 | 0.109 | **0.099** | 0.141 |
| 50  | 0.102 | 0.180 | 0.142 | 0.087 | **0.084** | 0.110 |
| 100 | 0.090 | 0.091 | 0.130 | 0.067 | **0.063** | 0.075 |
| 200 | 0.092 | **0.045** | 0.138 | 0.067 | 0.059 | 0.068 |

This reframes everything. Despite **never winning a cell** (Parrot)
or barely doing so (HPYLM/PPM-D), the n-gram-style methods sit
*closest* to the floor on average across the 18 cells at TL=25,50,100
because they never collapse: GDC and CHMM both have catastrophic
sparse-regime cells (CHMM at TL=25 hits 1.71 on dense_large,
sparse_large; GDC at every TL sits at 1.30+ on sparse_large) that
the n-gram methods avoid. **GDC's win count is high because it
dominates the easy regimes, not because it's broadly closest to the
floor.**

Three crossovers as TRAIN_LEN grows:

- **GDC and CHMM cross between TL=50 and TL=100.** At TL=50 GDC is
  ahead of CHMM in mean (0.10 vs 0.18); they tie at TL=100 (0.09);
  CHMM pulls 2× ahead by TL=200 (0.045). The driver is CHMM's EM
  catching up in sparse regimes.
- **CHMM overtakes the n-gram methods between TL=100 and TL=200.**
  HPYLM was the lowest-mean method through TL=100; CHMM passes it
  at TL=200 (0.045 vs 0.059).
- **GDC's mean stops improving after TL=100.** GDC at TL=100→200
  goes 0.090 → 0.092, basically flat. The training-corpus pool is
  already saturated for the chain length scale; more data per
  sequence does not help an N-state chain that already has 400 ×
  100 = 40,000 positions.

Practical implication: at the canonical TL=50 setup, GDC's win-count
edge **understates** how broadly competitive the n-gram methods are.
At the longer TL=200 setup, CHMM is the broadly best method,
**doubling** GDC's edge over it from "slight" (TL=50) to "decisive"
(TL=200). HPYLM is the most consistent method overall (lowest mean
at three of four TRAIN_LEN values).

## Files

- `seq_len_sweep.py` — unified sweep driver, takes TRAIN_LEN as CLI arg
- `build_seq_len_table.py` — aggregator (val-pick on val seeds,
  test mean on test seeds)
- `seq_len_<TL>_results.csv` — raw per-(seed, model, horizon) outputs
- `seq_len_table.csv` — long-format per-(TL, regime, N, model_class)
  table with picks and val/test perplexities
- `seq_len_table.md` — full markdown table dump (this file is the
  prose summary)
