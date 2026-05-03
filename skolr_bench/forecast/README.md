# SKOLR / Informer forecasting benchmarks

GDC evaluated on the original Informer paper's univariate
forecasting protocol (and the SKOLR paper's variant of it).

## Headline result so far

ETTh1 univariate (target=OT), leakage-free val-tuned GDC vs the
Informer paper's Table 1 baselines:

| T | GDC | ARIMA | Prophet | LSTMa | DeepAR | Informer | best other | GDC vs best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24  | **0.030** | 0.108 | 0.115 | 0.114 | 0.107 | 0.098 | 0.092 (Informer-stack) | **3.0× better** |
| 48  | **0.046** | 0.175 | 0.168 | 0.193 | 0.162 | 0.158 | 0.158 (Informer)       | **3.4× better** |
| 168 | **0.084** | 0.396 | 1.224 | 0.236 | 0.239 | 0.183 | 0.183 (Informer)       | **2.2× better** |
| 336 | **0.107** | 0.468 | 1.549 | 0.590 | 0.445 | 0.222 | 0.215 (Informer-stack) | **2.0× better** |
| 720 | 0.447     | 0.659 | 2.735 | 0.683 | 0.658 | 0.269 | 0.257 (Informer-stack) | **1.7× WORSE**  |

GDC dominates 4 of 5 horizons by 50-71%; T=720 collapses (3× val/test
gap from a bad pick). Open work in CLAUDE.md.

## Files

### Core kernels (drop-in replacements)
- `gdc_numba.py` — Numba CPU-parallel forecast kernel (~10-30× faster
  than the pure-NumPy reference).
- `gdc_torch.py` — PyTorch GPU kernel, fp32 / fp64 (~3-9× faster than
  numba on RTX 5090).
- `gdc_batched.py` — earlier numpy-batched attempt; deprecated due to
  CPU memory bandwidth dominance at large N. Kept as reference.

### Data loaders
- `loaders.py` — multivariate, SKOLR/Koopa/TSLib convention.
- `informer_loaders.py` — univariate, original Informer Table 1
  convention. Use this for direct comparison to published numbers.

### Eval scripts
- `gdc_etth1_full_sweep.py` — leakage-free val-tuned GDC across all 5
  Informer horizons for ETTh1. Uses the GPU kernel.
- `val_tune_etth1_t24.py` — single-horizon (T=24) version with
  diagnostic plots. Good template for new datasets.
- `smoke_informer.py` — quick GDC sanity test at one horizon, no val
  tuning.

### Baselines (statistical, refit per window)
- `arima_baseline.py` — pmdarima auto_arima. Result on ETTh1 T=24
  L=720 is 0.034 MSE — much better than the Informer paper's
  published 0.108 (their ARIMA appears under-tuned).
- `prophet_baseline.py` — Facebook Prophet, weekly+daily seasonality.
  ETTh1 T=24 L=720 gives 0.063 MSE vs published 0.115.

### Comparison helpers
- `fair_compare_etth1_t24.py` — runs three GDC variants (with and
  without train+val state space, short / long lookback) plus reads
  ARIMA and Informer numbers for an apples-to-apples ETTh1 T=24
  comparison.
- `bench_torch_horizons.py` — GPU kernel timing benchmark.

### Earlier experimental versions (kept for reference)
- `eval_forecast.py` — multivariate val-tuned baseline (v1).
- `eval_v2.py` — wider config grid (overfit val).
- `eval_v3.py` — long-context (L>2T) experiment.
- `eval_v4_longctx.py`, `eval_v5_fullstate.py`, `eval_v6_numba.py` —
  iterations exploring different state-space scoping. Superseded by
  `gdc_etth1_full_sweep.py`. See CLAUDE.md for the chronology.

### Data setup
See `../README.md` for dataset download instructions.

## Reproducing Informer (optional)

To verify our evaluation pipeline matches the published Informer
numbers, clone Informer's official repo into
`skolr_bench/Informer2020/` (gitignored) and patch one numpy 2.0
incompatibility:

```bash
cd skolr_bench
git clone --depth 1 https://github.com/zhouhaoyi/Informer2020.git
# Symlink dataset into the expected location
ln -s ../data_original Informer2020/dataset

# numpy 2.0 compat patch (np.Inf was removed)
sed -i "s/np\.Inf/np.inf/" Informer2020/utils/tools.py

# ETTh1 univariate T=24, GPU
cd Informer2020
python main_informer.py --model informer --data ETTh1 \
  --root_path ./dataset/ETT-small/ --features S \
  --seq_len 720 --label_len 168 --pred_len 24 \
  --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 \
  --train_epochs 6 --use_gpu True
```

Expected: test MSE ~0.10, MAE ~0.25. Matches the published 0.098/0.247
to within seed noise. Training takes ~50s on RTX 5090.

## Reproduce GDC

```bash
# Single-horizon smoke test (~5s)
python skolr_bench/forecast/smoke_informer.py

# Single-horizon val-tuned with diagnostic plots (~30s)
python skolr_bench/forecast/val_tune_etth1_t24.py

# Full ETTh1 sweep (~12 min on GPU fp64)
python skolr_bench/forecast/gdc_etth1_full_sweep.py

# ARIMA baseline (refit per window, parallelized)
python skolr_bench/forecast/arima_baseline.py ETTh1 --T 24 --L 720

# Prophet baseline
python skolr_bench/forecast/prophet_baseline.py ETTh1 --T 24 --L 720
```

## Status

See [`../../CLAUDE.md`](../../CLAUDE.md) for the open issues and next
steps (T=720 collapse, remaining datasets to sweep, full
ARIMA/Prophet grid).
