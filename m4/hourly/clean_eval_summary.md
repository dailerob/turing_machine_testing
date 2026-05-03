# Clean (leakage-free) evaluation: Hourly

- Series: 414, h = 48
- Candidate configs: 10 (all GDC-TS absorb + naive fallback)

## OWA vs published M4 Naive 2 (sMAPE=18.383, MASE=2.3950)

- **(1) Per-series val-tuned OWA = 0.5438**
- (2) Global pick by val-sMAPE [gdc_L168_s0.10_a1.0] OWA = 0.5667
- (2') Global pick by val-MASE [gdc_L168_s0.05_a1.0] OWA = 0.5342

## (SMAPE) (1) Per-series val-tuned

- Test mean: **10.64%**
- Test median: **5.09%**

Picks:

| config | n picks |
|---|---:|
| gdc_L168_s0.05_a1.0 | 109 |
| gdc_L168_s0.10_a1.0 | 59 |
| gdc_L24_s0.05_a1.0 | 49 |
| gdc_L72_s0.05_a1.0 | 43 |
| gdc_L48_s0.20_a1.0 | 37 |
| gdc_L24_s0.10_a1.0 | 37 |
| gdc_L72_s0.10_a1.0 | 29 |
| gdc_L48_s0.05_a1.0 | 26 |
| gdc_L48_s0.10_a1.0 | 24 |
| naive_last | 1 |

## (SMAPE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L168_s0.10_a1.0 | 10.46% | 9.60% | 5.22% |
| gdc_L168_s0.05_a1.0 | 10.93% | 10.14% | 6.01% |
| gdc_L72_s0.10_a1.0 | 11.19% | 11.37% | 5.08% |
| gdc_L48_s0.10_a1.0 | 11.44% | 11.09% | 5.10% |
| gdc_L72_s0.05_a1.0 | 11.48% | 11.64% | 5.62% |
| gdc_L48_s0.05_a1.0 | 11.73% | 11.34% | 5.38% |
| gdc_L48_s0.20_a1.0 | 12.35% | 11.77% | 5.00% |
| gdc_L24_s0.05_a1.0 | 12.74% | 12.20% | 5.18% |
| gdc_L24_s0.10_a1.0 | 12.84% | 11.90% | 4.91% |
| naive_last | 41.40% | 43.00% | 19.88% |

**Picked by val: `gdc_L168_s0.10_a1.0` -> test mean = 9.60%, median = 5.22%**

## (MASE) (1) Per-series val-tuned

- Test mean: **1.2186**
- Test median: **1.0557**

Picks:

| config | n picks |
|---|---:|
| gdc_L168_s0.05_a1.0 | 112 |
| gdc_L168_s0.10_a1.0 | 64 |
| gdc_L24_s0.05_a1.0 | 52 |
| gdc_L72_s0.05_a1.0 | 40 |
| gdc_L48_s0.20_a1.0 | 38 |
| gdc_L24_s0.10_a1.0 | 34 |
| gdc_L72_s0.10_a1.0 | 28 |
| gdc_L48_s0.05_a1.0 | 25 |
| gdc_L48_s0.10_a1.0 | 21 |

## (MASE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L168_s0.05_a1.0 | 1.2537 | 1.2373 | 1.0557 |
| gdc_L72_s0.05_a1.0 | 1.2689 | 1.3363 | 1.1929 |
| gdc_L48_s0.05_a1.0 | 1.2803 | 1.2782 | 1.1317 |
| gdc_L24_s0.05_a1.0 | 1.3037 | 1.2863 | 1.1535 |
| gdc_L168_s0.10_a1.0 | 1.4912 | 1.4631 | 1.0733 |
| gdc_L72_s0.10_a1.0 | 1.5266 | 1.5840 | 1.3185 |
| gdc_L48_s0.10_a1.0 | 1.5376 | 1.5368 | 1.1841 |
| gdc_L24_s0.10_a1.0 | 1.5728 | 1.5467 | 1.1825 |
| gdc_L48_s0.20_a1.0 | 2.1491 | 2.1367 | 1.2209 |
| naive_last | 11.5323 | 11.6077 | 3.6849 |

**Picked by val: `gdc_L168_s0.05_a1.0` -> test mean = 1.2373, median = 1.0557**

## (MSE) (1) Per-series val-tuned

- Test mean: **3.689e+06**
- Test median: **401.4**

Picks:

| config | n picks |
|---|---:|
| gdc_L168_s0.05_a1.0 | 113 |
| gdc_L168_s0.10_a1.0 | 65 |
| gdc_L24_s0.05_a1.0 | 50 |
| gdc_L48_s0.20_a1.0 | 43 |
| gdc_L72_s0.05_a1.0 | 41 |
| gdc_L24_s0.10_a1.0 | 28 |
| gdc_L72_s0.10_a1.0 | 27 |
| gdc_L48_s0.05_a1.0 | 24 |
| gdc_L48_s0.10_a1.0 | 23 |

## (MSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L168_s0.10_a1.0 | 8.85e+06 | 3.374e+06 | 295.9 |
| gdc_L168_s0.05_a1.0 | 9.057e+06 | 4.717e+06 | 337.2 |
| gdc_L72_s0.05_a1.0 | 9.779e+06 | 6.331e+06 | 459.7 |
| gdc_L24_s0.05_a1.0 | 1.059e+07 | 2.011e+06 | 496.8 |
| gdc_L72_s0.10_a1.0 | 1.073e+07 | 5.469e+06 | 442.9 |
| gdc_L48_s0.05_a1.0 | 1.099e+07 | 2.258e+06 | 530.1 |
| gdc_L24_s0.10_a1.0 | 1.103e+07 | 1.495e+06 | 472.5 |
| gdc_L48_s0.10_a1.0 | 1.112e+07 | 1.969e+06 | 510.5 |
| gdc_L48_s0.20_a1.0 | 1.379e+07 | 2.878e+06 | 494.7 |
| naive_last | 5.42e+07 | 5.754e+07 | 3770 |

**Picked by val: `gdc_L168_s0.10_a1.0` -> test mean = 3.374e+06, median = 295.9**

## (NRMSE) (1) Per-series val-tuned

- Test mean: **0.9248**
- Test median: **0.6094**

Picks:

| config | n picks |
|---|---:|
| gdc_L168_s0.05_a1.0 | 113 |
| gdc_L168_s0.10_a1.0 | 65 |
| gdc_L24_s0.05_a1.0 | 50 |
| gdc_L48_s0.20_a1.0 | 43 |
| gdc_L72_s0.05_a1.0 | 41 |
| gdc_L24_s0.10_a1.0 | 28 |
| gdc_L72_s0.10_a1.0 | 27 |
| gdc_L48_s0.05_a1.0 | 24 |
| gdc_L48_s0.10_a1.0 | 23 |

## (NRMSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L72_s0.10_a1.0 | 0.9489 | 1.0170 | 0.6616 |
| gdc_L168_s0.10_a1.0 | 0.9529 | 0.9061 | 0.5828 |
| gdc_L48_s0.10_a1.0 | 0.9586 | 0.9615 | 0.6502 |
| gdc_L24_s0.10_a1.0 | 0.9928 | 0.9645 | 0.6601 |
| gdc_L72_s0.05_a1.0 | 0.9946 | 1.0542 | 0.6442 |
| gdc_L48_s0.05_a1.0 | 1.0018 | 0.9815 | 0.6497 |
| gdc_L24_s0.05_a1.0 | 1.0094 | 0.9835 | 0.6800 |
| gdc_L168_s0.05_a1.0 | 1.0257 | 0.9927 | 0.5395 |
| gdc_L48_s0.20_a1.0 | 1.0392 | 1.0367 | 0.8078 |
| naive_last | 3.7496 | 3.7704 | 3.9291 |

**Picked by val: `gdc_L72_s0.10_a1.0` -> test mean = 1.0170, median = 0.6616**

