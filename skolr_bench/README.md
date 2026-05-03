# SKOLR benchmark suite

GDC evaluated on the benchmarks from the SKOLR paper (Zhang et al.,
*SKOLR: Structured Koopman Operator Linear RNN for Time-Series
Forecasting*, ICML 2025; arXiv:2506.14113), plus the original
Informer (Zhou et al., AAAI 2021; arXiv:2012.07436) protocol.

Two sub-benchmarks:

- **`forecast/`** — long-horizon multivariate / univariate
  forecasting on 8 standard datasets (ETTh1/h2, ETTm1/m2, ECL,
  Traffic, Weather, ILI). Compares to the Informer paper's
  ARIMA / Prophet / LSTMa / DeepAR / Reformer / LogTrans / Informer
  baselines. See [`forecast/README.md`](forecast/README.md).
- **`nlds/`** — 4 nonlinear dynamical systems (Pendulum, Duffing,
  Lotka-Volterra, Lorenz '63) from SKOLR Appendix E. See
  [`nlds/NLDS_RESULTS.md`](nlds/NLDS_RESULTS.md).

## Datasets

The forecasting CSVs are **not committed** (large; ~250 MB total).
Two equivalent download paths:

### Option 1: Tsinghua Cloud bundle (used by Koopa/SKOLR/TSLib)

Single zip containing ETT, ECL, Weather, Exchange, ILI, Traffic.

```bash
mkdir -p skolr_bench/data_original
cd skolr_bench
curl -L "https://cloud.tsinghua.edu.cn/f/b8f4a78a39874ac9893e/?dl=1" -o datasets.zip
unzip -q datasets.zip -d data_original
mv data_original/dataset/* data_original/ && rmdir data_original/dataset
rm datasets.zip
```

The expected layout under `skolr_bench/data_original/`:

```
ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
electricity/electricity.csv
exchange_rate/exchange_rate.csv
illness/national_illness.csv
traffic/traffic.csv
weather/weather.csv
```

### Option 2: Original sources (more work; equivalent ETT data)

- ETT (ETTh1/h2/m1/m2): https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small
  (the four `.csv` files in `ETT-small/`). These are byte-identical
  to the ETT files inside the Tsinghua bundle.
- ECL (electricity): https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
- Weather (Max-Planck Biogeochemistry): https://www.bgc-jena.mpg.de/wetter/
- ILI: https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html
- Traffic (Caltrans PeMS): http://pems.dot.ca.gov

The Tsinghua bundle is much easier — use that unless you need the
canonical original-source files.

## Loaders

Two loader modules live in `forecast/`:

- `loaders.py` — multivariate (all-channels) loader matching the
  SKOLR / Koopa / TSLib data-split convention (12/4/4 months for ETT
  hourly, 70/10/20 ratio for the others).
- `informer_loaders.py` — univariate loader matching the **original
  Informer paper Table 1** convention (12/4/4 months for ETT, target
  `OT` for ETT, `MT_320` for ECL, etc.).

Both fit StandardScaler on the train split only.

For independent verification, you can also clone the official Koopa
data_provider into `skolr_bench/data_provider/` and use it directly
(requires PyTorch):

```bash
mkdir -p skolr_bench/data_provider
cd skolr_bench/data_provider
for f in __init__.py data_factory.py data_loader.py; do
  curl -O "https://raw.githubusercontent.com/thuml/Koopa/main/data_provider/$f"
done
```

Our `loaders.py` was validated to match the windowed-sample counts
produced by the Koopa loader exactly.
