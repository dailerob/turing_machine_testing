# GDC on ETTm2 (Autoformer convention)

## TL;DR

GDC val-tuned univariate ETTm2 forecasting at the modern Autoformer
convention (input length I=96, horizons T ∈ {96, 192, 336, 720}).
Comparison against all baselines published in Autoformer Table 2:

| T | **GDC (ours)** | ARIMA | Prophet | DeepAR | N-BEATS | Reformer | LogTrans | Informer | Autoformer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.074 / 0.196 | 0.211 / 0.362 | 0.287 / 0.456 | 0.099 / 0.237 | 0.082 / 0.219 | 0.108 / 0.244 | 0.075 / 0.208 | 0.088 / 0.225 | **0.065 / 0.189** |
| 192 | **0.111 / 0.249** | 0.261 / 0.406 | 0.312 / 0.483 | 0.154 / 0.310 | 0.120 / 0.268 | 0.175 / 0.296 | 0.129 / 0.275 | 0.132 / 0.283 | 0.118 / 0.256 |
| 336 | **0.150 / 0.294** | 0.317 / 0.448 | 0.331 / 0.474 | 0.277 / 0.428 | 0.226 / 0.370 | 0.396 / 0.491 | 0.154 / 0.302 | 0.180 / 0.336 | 0.154 / 0.305 |
| 720 | 0.254 / 0.399 | 0.366 / 0.487 | 0.534 / 0.593 | 0.332 / 0.468 | 0.188 / 0.338 | 0.468 / 0.540 | **0.160 / 0.322** | 0.300 / 0.435 | 0.182 / 0.335 |

(format: MSE / MAE on standardized data, lower is better; **bold** = best per row.)

**GDC achieves the best published MSE on T=192 and T=336**, beating
the previous SOTA Autoformer. At T=96 GDC trails Autoformer by 14%
relative; at T=720 GDC degrades and falls behind LogTrans by 59% (the
long-horizon collapse pattern previously seen on ETTh1).

## GDC vs each baseline category

| comparison | T=96 | T=192 | T=336 | T=720 |
|---|---|---|---|---|
| vs ARIMA | 2.9× better | 2.4× better | 2.1× better | 1.4× better |
| vs Prophet | 3.9× better | 2.8× better | 2.2× better | 2.1× better |
| vs DeepAR | 1.3× better | 1.4× better | 1.8× better | 1.3× better |
| vs N-BEATS | tied (10% worse) | tied (8% better) | better (34% better) | worse (35% worse) |
| vs Reformer | 1.5× better | 1.6× better | 2.6× better | 1.8× better |
| vs LogTrans | tied (1% worse) | better (14%) | tied (3% better) | **worse (59%)** |
| vs Informer | 1.2× better | 1.2× better | 1.2× better | 1.2× better |
| vs Autoformer | tied (14% worse) | better (6%) | better (3%) | worse (40%) |

**GDC dominates every classical baseline (ARIMA/Prophet) and most
deep-learning baselines (DeepAR, Reformer, Informer) by clear margins
across all four horizons.** Against the strongest published methods
(Autoformer, LogTrans, N-BEATS), GDC is competitive at short horizons
but loses ground at T=720.

## Selected configurations (val-tuned)

| T | recipe | σ | α | val MSE | test MSE | test MAE | runtime |
|---:|---|---:|---:|---:|---:|---:|---:|
| 96 | diff | 0.10 | 1.0 | 0.102 | 0.074 | 0.196 | 251s |
| 192 | diff | 0.10 | 1.0 | 0.160 | 0.111 | 0.249 | 409s |
| 336 | diff | 0.10 | 1.0 | 0.213 | 0.150 | 0.294 | 621s |
| 720 | raw | 0.10 | 1.0 | 0.246 | 0.254 | 0.399 | 1099s |

The diff recipe (forecast 1-step changes, then cumsum onto the
last observation) wins at T ≤ 336; at T=720 raw matching takes over.
The crossover matches what we previously observed on ETTh1.

## Protocol

- Univariate target = OT (oil temperature)
- Splits = 12 / 4 / 4 months at 15-min granularity (34560 / 11520 / 11520 rows)
- StandardScaler fit on train only
- Lookback I=96 (matches Autoformer / TimesNet / iTransformer / Koopa convention)
- Val tuning: GDC state space = train; pick best of 22 configs by val MSE
- Test eval: GDC state space = train + val; apply val-picked config
- Stride=1 rolling test windows
- MSE / MAE on standardized data
- GPU PyTorch fp32 kernel (matches fp64 to ~3e-7 relative)

Total runtime: 40 min on RTX 5090 (22 configs × 4 horizons).

## Reproduce

```bash
python skolr_bench/forecast/gdc_ettm2_autoformer.py
```

Result CSV: `skolr_bench/forecast/results/gdc_ettm2_autoformer.csv`.

Note: ETTm2 is the only ETT subset for which Autoformer reports
univariate baselines (Table 2 footnote: "ETT means the ETTm2"). For
ETTh1, ETTh2, ETTm1 univariate at this convention, no published
baselines exist for direct comparison.
