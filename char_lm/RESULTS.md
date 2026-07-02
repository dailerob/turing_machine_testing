# Character-LM benchmark (Dedieu et al. 2019, Table 4 protocol)

Val-tuned bits-per-symbol on the 8 datasets of the Dedieu et al. 2019
char-LM benchmark. Last 10% of train is held out as validation;
each method picks its best config on val, then is retrained on
full train and scored on test. Methods:

- **GDC**: this work; dual-α scorer (α_ctx for state-tracking
  transition, α_fc for prediction-time transition; θ=0, β=0).
  Grid: α_ctx ∈ {0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}, 
  α_fc ∈ {0.95, 0.99, 1.0}. Uses torch-GPU scorer when N·T > 5×10⁹.
- **HPYLM**: fixed-depth Hierarchical Pitman-Yor LM (Wood et al. 2009).
  Grid: depth ∈ {3,5,8}, discount ∈ {0.25,0.5,0.75}, concentration ∈ {0.5,1.0}.
- **PPM-D**: absolute-discount n-gram (Howard 1993).
  Grid: depth ∈ {3,5}, discount ∈ {0.1,0.25,0.5,0.75}.
- **KN-3**: interpolated Kneser-Ney trigram. Grid: discount ∈ {0.25,0.5,0.75,0.9}.
- **Parrot**: top-K nearest-prefix kNN (Zhang & Gilpin 2025).
  Grid: L ∈ {1,2,3,4,6}, K ∈ {1,5,25}.
  *Omitted on calgary/moby/war-peace* (O(N·T) cost = 4-150 hours per dataset).

Paper Table 4 reference numbers shown for context (CHMM, n-gram=KN30
in their kylm setup, SeqM = Sequence Memoizer, LSTM).

## Test BPS per dataset

| Dataset | HPYLM | PPM-D | KN-3 | Parrot | GDC (α_ctx, α_fc) | CHMM | n-gram | SeqM | LSTM |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| blake-poems | 1.680 | 1.663 | 1.878 | 2.485 | 1.724 (0.5,1.0) | 1.60 | 1.75 | 1.71 | 1.68 |
| shakespeare-macbeth | 1.772 | 1.736 | 2.076 | 2.556 | 1.802 (0.55,1.0) | 1.69 | 1.79 | 1.77 | 1.74 |
| carroll-alice | 1.791 | 1.753 | 2.199 | 2.479 | 1.718 (0.55,1.0) | 1.54 | 1.61 | 1.57 | 1.58 |
| shakespeare-hamlet | 1.785 | 1.747 | 2.129 | 2.493 | 1.823 (0.55,1.0) | 1.63 | 1.72 | 1.69 | 1.68 |
| milton-paradise | 2.003 | 1.960 | 2.424 | 2.598 | 2.006 (0.6,1.0) | 1.73 | 1.83 | 1.78 | 1.78 |
| calgary-book1 | 1.848 | 1.985 | 2.492 | — | 1.889 (0.6,1.0) | 1.63 | 1.72 | 1.64 | 1.67 |
| melville-mobydick | 1.921 | 2.015 | 2.495 | — | 1.954 (0.6,1.0) | 1.72 | 1.81 | 1.73 | 1.76 |
| war-peace | 1.788 | 1.845 | 2.490 | — | 1.822 (0.65,1.0) | 1.59 | 1.65 | 1.57 | 1.62 |

## Notes

- **carroll-alice**: GDC at 1.718 is the best of our methods (beats PPM-D by 0.035).
- **Other 7**: HPYLM and PPM-D win, with GDC trailing the best by 0.02-0.08.
- **vs paper CHMM**: GDC is uniformly 0.13-0.25 above CHMM (which is still the best on every dataset).
- **Optimal α_ctx grows with dataset size**: 0.50 (blake, 30k) → 0.55 (medium) → 0.60-0.65 (large, ≥350k). α_fc=1.0 is universal.
- **calgary-book1 test size**: our preprocessing produces a 71,567-char test set (10× the paper's 7,116). Cross-method comparison within this row is valid; the paper-reference columns aren't directly comparable for calgary.

## Dual-α recipe summary

For each test position $t$:

1. Apply `transition(dist, α_fc, θ_fc=0)` once → `pred_state`. Marginalize over emissions → score $-\log_2 P(\text{test}[t])$.
2. Apply `transition(dist, α_ctx, θ_ctx=0)` once to advance state-tracking distribution.
3. Filter on `test[t]` with sharp emission (β=0) → updated `dist`.

Implemented in `char_lm/bps_eval.py` as `score_bps_gdc_dual` (numpy)
and `score_bps_gdc_dual_torch` (GPU). `char_lm/run.py` dispatches to
torch when N·T > 5×10⁹.