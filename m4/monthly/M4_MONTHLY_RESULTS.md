# M4 Monthly — GDC results

## TL;DR

On M4 monthly (48,000 series, h=18, season=12):

| approach | mean sMAPE | median sMAPE |
|---|---:|---:|
| naive_last (random walk) | 15.26% | 8.61% |
| drift | 15.48% | 8.25% |
| naive_seasonal12 ("Naive2") | 15.99% | 9.95% |
| naive_seasonal12 + drift | 17.33% | 9.20% |
| NN-matching on diffs (best fixed L=6, sigma%=0.50) | 14.37% | 7.84% |
| **GDC-TS on diffs (L=12, sigma%=0.25, alpha=0.95, absorb)** | **13.96%** | **7.54%** |

**Best result: 13.96% mean / 7.54% median.** ~9% relative gain over
naive_last on mean. M4 monthly leaderboard winners are around 12-13%.

## Surprises

1. **Seasonal naive is *worse* than naive_last on monthly.** Despite
   season=12 being a real annual cycle, repeating last-12-months
   loses 0.7 abs sMAPE to just repeating train[-1]. Many M4 monthly
   series have weak periodicity or strong drifts that dominate the
   cycle.

2. **Smaller windows win.** Best L is 6 (NN-diff) or 12 (GDC), not
   18 or 24. Recent local structure beats long historical context for
   monthly. Contrast: weekly preferred L=26 ≈ 2h.

3. **alpha=0.95 (more damping than weekly) is optimal.** Weekly's best
   was alpha=0.99; monthly wants alpha=0.95. The longer h=18 horizon
   benefits from more aggressive smoothing through the kernel.

## Val-tuning starts to work at h=18

For the first time, per-series val-tuning is competitive. On the
long-series subset (37,231 series with `len >= 78`):

| model | mean | median |
|---|---:|---:|
| gdc_L12_s0.25_a0.95 (best fixed) | 10.56% | 5.09% |
| val_tuned (8 candidates) | 10.67% | 5.15% |
| naive_last | 11.74% | 5.92% |

(These numbers are lower than the headline because the 10,769 short
series excluded here are the high-sMAPE long-tail.)

Val-tuned is essentially tied with the best fixed config. Compare to
daily (h=14) where val-tuned overfit and lost by 0.06-0.2 abs to
fixed. With h=18 there's enough validation signal to start
distinguishing genuinely different regimes.

## Recipe

```python
def gdc_diff_forecast(train, window_len=12, sigma_frac=0.25,
                      alpha=0.95, theta=0.0, h=18):
    d = np.diff(train)
    sigma_per_step = float(np.std(d)) * sigma_frac
    sigma_gdc = sigma_per_step * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = d.reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform',
    )
    prime = d[-window_len:].reshape(-1, 1)
    _, state_dists = gdc.forecast_gdc_style(prime, n_steps=h)
    nt_mask = (~gdc.terminal_mask).astype(float)
    sd_nt = state_dists * nt_mask[None, :]
    sd_nt_norm = sd_nt / np.where(sd_nt.sum(axis=1, keepdims=True) > 1e-12,
                                   sd_nt.sum(axis=1, keepdims=True), 1.0)
    forecast_d = (sd_nt_norm @ gdc.states)[:, 0]
    return train[-1] + np.cumsum(forecast_d)
```

## Files

- [v0_baselines.py](v0_baselines.py)
- [v1_nn_diff.py](v1_nn_diff.py)
- [v2_gdc_diff.py](v2_gdc_diff.py) — best fixed result (13.96%)
- [v3_val_tuned.py](v3_val_tuned.py) — val-tuned ensemble
- [plot_series.py](plot_series.py)

## Reproduce

```bash
python m4/monthly/v0_baselines.py
python m4/monthly/v1_nn_diff.py
python m4/monthly/v2_gdc_diff.py    # best: 13.96%
python m4/monthly/v3_val_tuned.py
```
