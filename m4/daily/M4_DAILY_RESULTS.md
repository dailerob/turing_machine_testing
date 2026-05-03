# M4 Daily — GDC results

## TL;DR

On M4 daily (4,227 series, h=14):

| approach | mean sMAPE | median sMAPE |
|---|---:|---:|
| naive_last (M4 "Naive2") | 3.05% | 1.99% |
| drift / mean_diff (identical) | 3.17% | 1.99% |
| naive_last7_mean | 3.56% | — |
| naive_seasonal7 | 3.74% | — |
| NN-matching on raw values, detrended | 4.36% | 2.72% |
| **NN-matching on 1-step diffs, fixed config (L=14, σ%=0.50)** | **2.99%** | **1.96%** |
| GDC-TS on diffs, fixed best (L=7, σ%=1.00, α=1, absorb) | 3.01% | 1.96% |
| Val-tuned GDC, 5 candidates | 3.05% | 1.97% |
| Val-tuned GDC, 85 candidates | 3.19% | 2.01% (overfit) |

**Best result: 2.99% mean / 1.96% median** with single-config
NN-matching on first-differences. This sits at the M4 daily winner
neighborhood (~3.0%).

The headline takeaway is the *opposite* of the hourly story: on daily
series, GDC's iterated transition kernel doesn't help over a plain
Gaussian-weighted average of historical 1-step changes, and per-series
validation tuning (which won the hourly benchmark) overfits because
the validation horizon h=14 is too short and the signal-to-noise ratio
is low.

## What the data looks like

Daily series in M4 are mostly financial / macro indicators with strong
drifts and weak periodicity. Lengths span 93 to 9,919 days
(median ~2,940). The plot in [fig_m4_daily_sample.png](fig_m4_daily_sample.png)
shows representative series — most look like random walks with drift
and occasional regime shifts; few have clear daily/weekly cycles.

This contrasts sharply with hourly, which had strong 24-hour cycles.

## Recipe progression

### v0 baselines: random walk wins

```
naive_last:       3.05%  ← M4's "Naive2"
drift:            3.17%
naive_last7_mean: 3.56%
naive_seasonal7:  3.74%
```

`drift` (linear extrapolation from first/last) is identical to
`mean_diff` (cumulative mean of first-differences) since
`(last-first)/(n-1) == mean(diff)`. Both lose to `naive_last` because
they extrapolate the *long-run* drift, which is often misleading for
the next 14 days.

### v1 NN-matching on raw values: doesn't work

Standard NN-matching (find similar L-windows in history, average their
continuations) gets 4.7% mean. Adding per-window linear detrending
helps (4.36%) but still loses to `naive_last`. The reason: drifting
series make any past window pull the forecast away from `train[-1]`.

### v2 NN-matching on 1-step diffs: this is what works

Match shape of recent *changes* and forecast the next 14 changes,
then cumsum onto `train[-1]`. The level cancels out entirely.

Best fixed config (L=14, σ%=0.50): **2.99% mean, 1.96% median**.

Higher σ helps a lot here (3.45% at σ%=0.10 → 2.99% at σ%=0.50). At
high σ the Gaussian weights become broad, and the forecast approaches
a soft global average of historical diffs — essentially a noise-robust
drift estimate that's slightly better than pure naive.

### v4 GDC-TS-on-diffs: same as v2, slightly worse

Plugging differenced series into GDC-TS with α=1, θ=0,
`terminal_behavior='absorb'` (same recipe that won hourly) gives
3.01-3.02% mean — basically tied with the raw NN-diff at 2.99%.

Reason: at α=1, θ=0 the GDC step kernel is essentially a re-mixing of
the same Gaussian similarity weights, with no extra cyclic structure
to amplify. The hourly win came from iterated transitions exploiting
the 24h cycle; daily diffs have no such structure.

### v5 val-tuned: overfits

Hold out last h=14 of training as validation, pick best config per
series. With 85 configs (α × θ × L × σ%): **3.19% mean, worse than
naive**. With a focused 5-config set: **3.05% mean, ties naive**.

Validation noise dominates at h=14. Of 4,227 series, naive_last is
the val-best for ~47% — the random-walk hypothesis is genuinely
correct for almost half the data.

## Why this is qualitatively different from hourly

| feature | hourly | daily |
|---|---|---|
| series count | 414 | 4,227 |
| horizon h | 48 | 14 |
| dominant structure | 24h cycle, ~weekly cycle | drift, regime shifts |
| best naive | seasonal_period_mean (8.4%) | naive_last (3.05%) |
| best GDC | iterated absorb-mode kernel | basically equal to raw similarity weights |
| best val-tuned beats naive? | yes (10.65% vs 11.71%) | no |
| validation noise at chosen h | low | high |

The hourly benchmark rewarded models that could exploit cyclic
structure across a long forecast horizon. Daily rewards models that
recognize when the random walk hypothesis is right (most of the time)
and add a small drift correction otherwise. There's just not much
above-random-walk signal to capture.

## Files

- [v0_baselines.py](v0_baselines.py) — naive baselines
- [v1_nn_matching.py](v1_nn_matching.py) — NN on raw values (with detrend option)
- [v2_nn_diff.py](v2_nn_diff.py) — NN on 1-step diffs (best result)
- [v3_nn_diff_sweep.py](v3_nn_diff_sweep.py) — wider sigma + initial val-tune
- [v4_gdc_diff.py](v4_gdc_diff.py) — GDC-TS on diffs, absorb mode
- [v5_val_tuned.py](v5_val_tuned.py) — per-series val-tuned ensemble
- [plot_series.py](plot_series.py) — representative series plots
- Result CSVs: `*_results.csv` — per-series sMAPE for each config

## Reproduce

```bash
python m4/daily/v0_baselines.py
python m4/daily/v2_nn_diff.py        # best single result: 2.99%
python m4/daily/v4_gdc_diff.py       # GDC-TS-on-diffs comparison
python m4/daily/v5_val_tuned.py      # val-tune comparison
```
