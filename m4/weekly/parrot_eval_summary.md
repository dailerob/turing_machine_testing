# Parrot evaluation: Weekly

- Series: 359, h = 13
- Candidate configs: 13 (raw/diff × k∈{1,5} × per-freq L grid + naive_last)

## OWA vs published M4 Naive 2 (sMAPE=9.161, MASE=2.7770)

- **(1) Per-series val-tuned OWA = 0.9142**
- (2) Global pick by val-sMAPE [naive_last] OWA = 1.0001
- (2') Global pick by val-MASE [parrot_diff_L13_k5] OWA = 0.8152

## (SMAPE) (1) Per-series val-tuned

- Test mean: **8.11%**
- Test median: **5.09%**

Picks:

| config | n picks |
|---|---:|
| parrot_diff_L13_k5 | 67 |
| parrot_diff_L52_k5 | 45 |
| parrot_diff_L13_k1 | 32 |
| parrot_diff_L26_k5 | 30 |
| parrot_raw_L13_k5 | 30 |
| parrot_raw_L13_k1 | 26 |
| naive_last | 25 |
| parrot_diff_L26_k1 | 24 |
| parrot_raw_L52_k5 | 23 |
| parrot_diff_L52_k1 | 21 |
| parrot_raw_L52_k1 | 13 |
| parrot_raw_L26_k5 | 13 |
| parrot_raw_L26_k1 | 10 |

## (SMAPE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| naive_last | 9.20% | 9.16% | 5.18% |
| parrot_diff_L52_k5 | 9.36% | 7.90% | 4.49% |
| parrot_diff_L13_k5 | 9.58% | 7.92% | 4.27% |
| parrot_diff_L26_k5 | 9.76% | 7.59% | 4.47% |
| parrot_raw_L52_k1 | 10.34% | 12.25% | 7.97% |
| parrot_raw_L52_k5 | 10.45% | 12.25% | 8.57% |
| parrot_diff_L52_k1 | 10.46% | 9.61% | 5.06% |
| parrot_diff_L13_k1 | 11.97% | 9.83% | 5.03% |
| parrot_diff_L26_k1 | 12.10% | 10.12% | 4.90% |
| parrot_raw_L13_k1 | 12.23% | 10.57% | 7.46% |
| parrot_raw_L13_k5 | 12.26% | 9.45% | 7.03% |
| parrot_raw_L26_k1 | 13.72% | 10.54% | 7.23% |
| parrot_raw_L26_k5 | 13.78% | 9.99% | 7.67% |

**Picked by val: `naive_last` -> test mean = 9.16%, median = 5.18%**

## (MASE) (1) Per-series val-tuned

- Test mean: **2.6187**
- Test median: **1.4515**

Picks:

| config | n picks |
|---|---:|
| parrot_diff_L13_k5 | 69 |
| parrot_diff_L52_k5 | 44 |
| parrot_diff_L13_k1 | 32 |
| parrot_raw_L13_k5 | 32 |
| parrot_diff_L26_k5 | 29 |
| naive_last | 24 |
| parrot_raw_L13_k1 | 24 |
| parrot_diff_L26_k1 | 23 |
| parrot_diff_L52_k1 | 23 |
| parrot_raw_L52_k5 | 22 |
| parrot_raw_L52_k1 | 14 |
| parrot_raw_L26_k5 | 12 |
| parrot_raw_L26_k1 | 11 |

## (MASE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_diff_L13_k5 | 2.4979 | 2.1254 | 1.4005 |
| parrot_diff_L52_k5 | 2.5099 | 2.3100 | 1.4076 |
| parrot_diff_L26_k5 | 2.5986 | 2.1476 | 1.4727 |
| naive_last | 2.8975 | 2.7773 | 1.9384 |
| parrot_diff_L52_k1 | 2.9702 | 2.7060 | 1.6151 |
| parrot_diff_L13_k1 | 3.0608 | 2.7959 | 1.8018 |
| parrot_diff_L26_k1 | 3.1718 | 2.4876 | 1.7246 |
| parrot_raw_L52_k1 | 4.5448 | 4.7125 | 2.5417 |
| parrot_raw_L13_k1 | 4.6821 | 4.2462 | 2.3301 |
| parrot_raw_L52_k5 | 4.6825 | 5.0084 | 2.5189 |
| parrot_raw_L13_k5 | 4.7848 | 4.2853 | 2.3054 |
| parrot_raw_L26_k1 | 5.0585 | 4.5993 | 2.2611 |
| parrot_raw_L26_k5 | 5.1453 | 4.8178 | 2.3099 |

**Picked by val: `parrot_diff_L13_k5` -> test mean = 2.1254, median = 1.4005**

## (MSE) (1) Per-series val-tuned

- Test mean: **4.589e+05**
- Test median: **5.438e+04**

Picks:

| config | n picks |
|---|---:|
| parrot_diff_L13_k5 | 67 |
| parrot_diff_L52_k5 | 48 |
| parrot_diff_L13_k1 | 33 |
| parrot_diff_L26_k5 | 30 |
| parrot_raw_L13_k1 | 29 |
| parrot_raw_L13_k5 | 29 |
| parrot_raw_L52_k5 | 24 |
| parrot_diff_L52_k1 | 23 |
| parrot_diff_L26_k1 | 22 |
| naive_last | 21 |
| parrot_raw_L26_k5 | 16 |
| parrot_raw_L52_k1 | 9 |
| parrot_raw_L26_k1 | 8 |

## (MSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_diff_L52_k5 | 3.505e+05 | 3.238e+05 | 3.598e+04 |
| parrot_diff_L13_k5 | 3.549e+05 | 3.938e+05 | 3.447e+04 |
| naive_last | 3.7e+05 | 4.535e+05 | 7.325e+04 |
| parrot_diff_L26_k5 | 3.973e+05 | 3.459e+05 | 3.319e+04 |
| parrot_diff_L52_k1 | 4.457e+05 | 5.246e+05 | 4.274e+04 |
| parrot_diff_L13_k1 | 5.603e+05 | 7.829e+05 | 5.438e+04 |
| parrot_raw_L13_k5 | 6.485e+05 | 5.86e+05 | 8.593e+04 |
| parrot_raw_L52_k5 | 7.598e+05 | 1.089e+06 | 1.008e+05 |
| parrot_diff_L26_k1 | 7.692e+05 | 6.477e+05 | 5.032e+04 |
| parrot_raw_L26_k5 | 9.445e+05 | 9.241e+05 | 9.469e+04 |
| parrot_raw_L13_k1 | 1.056e+06 | 7e+05 | 8.777e+04 |
| parrot_raw_L26_k1 | 1.058e+06 | 1.1e+06 | 8.56e+04 |
| parrot_raw_L52_k1 | 1.183e+06 | 1.183e+06 | 9.836e+04 |

**Picked by val: `parrot_diff_L52_k5` -> test mean = 3.238e+05, median = 3.598e+04**

## (NRMSE) (1) Per-series val-tuned

- Test mean: **1.8037**
- Test median: **1.2002**

Picks:

| config | n picks |
|---|---:|
| parrot_diff_L13_k5 | 67 |
| parrot_diff_L52_k5 | 48 |
| parrot_diff_L13_k1 | 33 |
| parrot_diff_L26_k5 | 30 |
| parrot_raw_L13_k5 | 30 |
| parrot_raw_L13_k1 | 29 |
| parrot_raw_L52_k5 | 24 |
| parrot_diff_L52_k1 | 23 |
| parrot_diff_L26_k1 | 22 |
| naive_last | 21 |
| parrot_raw_L26_k5 | 15 |
| parrot_raw_L52_k1 | 9 |
| parrot_raw_L26_k1 | 8 |

## (NRMSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_diff_L13_k5 | 1.7209 | 1.4718 | 1.1385 |
| parrot_diff_L52_k5 | 1.7361 | 1.5174 | 1.1208 |
| parrot_diff_L26_k5 | 1.7782 | 1.4767 | 1.1584 |
| naive_last | 1.9651 | 1.9075 | 1.4643 |
| parrot_diff_L52_k1 | 1.9696 | 1.7639 | 1.3121 |
| parrot_diff_L13_k1 | 2.0572 | 1.9273 | 1.4263 |
| parrot_diff_L26_k1 | 2.1374 | 1.7706 | 1.4180 |
| parrot_raw_L52_k1 | 2.6993 | 2.8292 | 1.9191 |
| parrot_raw_L52_k5 | 2.7407 | 2.9611 | 1.9795 |
| parrot_raw_L13_k5 | 2.8266 | 2.6186 | 1.5973 |
| parrot_raw_L13_k1 | 2.8940 | 2.6420 | 1.6868 |
| parrot_raw_L26_k5 | 3.0828 | 2.8606 | 1.7374 |
| parrot_raw_L26_k1 | 3.1158 | 2.7800 | 1.7508 |

**Picked by val: `parrot_diff_L13_k5` -> test mean = 1.4718, median = 1.1385**

