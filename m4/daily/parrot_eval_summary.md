# Parrot evaluation: Daily

- Series: 4227, h = 14
- Candidate configs: 13 (raw/diff × k∈{1,5} × per-freq L grid + naive_last)

## OWA vs published M4 Naive 2 (sMAPE=3.045, MASE=3.2780)

- **(1) Per-series val-tuned OWA = 1.2096**
- (2) Global pick by val-sMAPE [naive_last] OWA = 1.0001
- (2') Global pick by val-MASE [naive_last] OWA = 1.0001

## (SMAPE) (1) Per-series val-tuned

- Test mean: **3.68%**
- Test median: **2.42%**

Picks:

| config | n picks |
|---|---:|
| naive_last | 602 |
| parrot_diff_L7_k1 | 453 |
| parrot_diff_L28_k5 | 429 |
| parrot_diff_L7_k5 | 428 |
| parrot_diff_L14_k5 | 418 |
| parrot_diff_L28_k1 | 359 |
| parrot_diff_L14_k1 | 352 |
| parrot_raw_L7_k5 | 345 |
| parrot_raw_L14_k5 | 262 |
| parrot_raw_L7_k1 | 201 |
| parrot_raw_L28_k5 | 166 |
| parrot_raw_L14_k1 | 117 |
| parrot_raw_L28_k1 | 95 |

## (SMAPE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| naive_last | 2.79% | 3.05% | 1.99% |
| parrot_diff_L14_k5 | 3.08% | 3.32% | 2.13% |
| parrot_diff_L7_k5 | 3.09% | 3.29% | 2.14% |
| parrot_diff_L28_k5 | 3.10% | 3.23% | 2.07% |
| parrot_diff_L28_k1 | 3.62% | 3.75% | 2.47% |
| parrot_diff_L14_k1 | 3.65% | 3.88% | 2.57% |
| parrot_diff_L7_k1 | 3.77% | 3.98% | 2.72% |
| parrot_raw_L7_k5 | 4.11% | 4.40% | 2.72% |
| parrot_raw_L14_k5 | 4.49% | 4.83% | 3.02% |
| parrot_raw_L7_k1 | 4.59% | 4.82% | 3.07% |
| parrot_raw_L14_k1 | 4.86% | 5.15% | 3.30% |
| parrot_raw_L28_k5 | 5.05% | 5.36% | 3.37% |
| parrot_raw_L28_k1 | 5.26% | 5.60% | 3.56% |

**Picked by val: `naive_last` -> test mean = 3.05%, median = 1.99%**

## (MASE) (1) Per-series val-tuned

- Test mean: **3.9707**
- Test median: **2.8500**

Picks:

| config | n picks |
|---|---:|
| naive_last | 603 |
| parrot_diff_L7_k1 | 456 |
| parrot_diff_L28_k5 | 430 |
| parrot_diff_L7_k5 | 422 |
| parrot_diff_L14_k5 | 418 |
| parrot_diff_L28_k1 | 362 |
| parrot_diff_L14_k1 | 351 |
| parrot_raw_L7_k5 | 347 |
| parrot_raw_L14_k5 | 262 |
| parrot_raw_L7_k1 | 200 |
| parrot_raw_L28_k5 | 164 |
| parrot_raw_L14_k1 | 118 |
| parrot_raw_L28_k1 | 94 |

## (MASE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| naive_last | 651931461019.1388 | 3.2784 | 2.3557 |
| parrot_diff_L14_k5 | 651931461019.4026 | 3.4649 | 2.4946 |
| parrot_diff_L28_k5 | 651931461019.4045 | 3.4243 | 2.3644 |
| parrot_diff_L7_k5 | 651931461019.4221 | 3.4762 | 2.4735 |
| parrot_diff_L28_k1 | 651931461019.9230 | 3.9640 | 2.8838 |
| parrot_diff_L14_k1 | 651931461020.0057 | 4.1027 | 3.0083 |
| parrot_diff_L7_k1 | 651931461020.1521 | 4.2682 | 3.1691 |
| parrot_raw_L7_k5 | 651931461020.7380 | 4.9548 | 3.1803 |
| parrot_raw_L14_k5 | 651931461021.1257 | 5.3822 | 3.4963 |
| parrot_raw_L7_k1 | 651931461021.2283 | 5.3475 | 3.5985 |
| parrot_raw_L14_k1 | 651931461021.5342 | 5.6792 | 3.8374 |
| parrot_raw_L28_k5 | 651931461021.6487 | 5.9654 | 3.9568 |
| parrot_raw_L28_k1 | 651931461021.8414 | 6.1538 | 4.2327 |

**Picked by val: `naive_last` -> test mean = 3.2784, median = 2.3557**

## (MSE) (1) Per-series val-tuned

- Test mean: **5.966e+05**
- Test median: **1.773e+04**

Picks:

| config | n picks |
|---|---:|
| naive_last | 615 |
| parrot_diff_L7_k1 | 446 |
| parrot_diff_L14_k5 | 434 |
| parrot_diff_L7_k5 | 434 |
| parrot_diff_L28_k5 | 433 |
| parrot_raw_L7_k5 | 358 |
| parrot_diff_L28_k1 | 354 |
| parrot_diff_L14_k1 | 328 |
| parrot_raw_L14_k5 | 252 |
| parrot_raw_L7_k1 | 198 |
| parrot_raw_L28_k5 | 176 |
| parrot_raw_L14_k1 | 112 |
| parrot_raw_L28_k1 | 87 |

## (MSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| naive_last | 2.829e+05 | 4.976e+05 | 1.175e+04 |
| parrot_diff_L7_k5 | 2.854e+05 | 5.82e+05 | 1.443e+04 |
| parrot_diff_L14_k5 | 2.956e+05 | 6.423e+05 | 1.352e+04 |
| parrot_diff_L28_k5 | 3.125e+05 | 6.319e+05 | 1.27e+04 |
| parrot_raw_L7_k5 | 5.286e+05 | 6.591e+05 | 2.151e+04 |
| parrot_raw_L14_k5 | 5.624e+05 | 7.174e+05 | 2.547e+04 |
| parrot_raw_L28_k5 | 5.936e+05 | 7.814e+05 | 3.188e+04 |
| parrot_raw_L14_k1 | 6.239e+05 | 7.881e+05 | 3.33e+04 |
| parrot_raw_L7_k1 | 6.317e+05 | 7.81e+05 | 3.029e+04 |
| parrot_raw_L28_k1 | 6.7e+05 | 8.166e+05 | 3.731e+04 |
| parrot_diff_L7_k1 | 8.512e+05 | 5.905e+05 | 2.293e+04 |
| parrot_diff_L14_k1 | 8.652e+05 | 1.378e+06 | 2.077e+04 |
| parrot_diff_L28_k1 | 9.68e+05 | 1.385e+06 | 1.902e+04 |

**Picked by val: `naive_last` -> test mean = 4.976e+05, median = 1.175e+04**

## (NRMSE) (1) Per-series val-tuned

- Test mean: **2.9232**
- Test median: **2.1868**

Picks:

| config | n picks |
|---|---:|
| naive_last | 615 |
| parrot_diff_L7_k1 | 446 |
| parrot_diff_L14_k5 | 434 |
| parrot_diff_L7_k5 | 434 |
| parrot_diff_L28_k5 | 433 |
| parrot_raw_L7_k5 | 358 |
| parrot_diff_L28_k1 | 354 |
| parrot_diff_L14_k1 | 328 |
| parrot_raw_L14_k5 | 252 |
| parrot_raw_L7_k1 | 198 |
| parrot_raw_L28_k5 | 176 |
| parrot_raw_L14_k1 | 112 |
| parrot_raw_L28_k1 | 87 |

## (NRMSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| naive_last | 2.3102 | 2.4453 | 1.7988 |
| parrot_diff_L14_k5 | 2.5076 | 2.5784 | 1.9458 |
| parrot_diff_L28_k5 | 2.5082 | 2.5470 | 1.8411 |
| parrot_diff_L7_k5 | 2.5159 | 2.5795 | 1.9263 |
| parrot_diff_L28_k1 | 2.9260 | 2.9851 | 2.2630 |
| parrot_diff_L14_k1 | 2.9919 | 3.0866 | 2.3697 |
| parrot_diff_L7_k1 | 3.0628 | 3.1702 | 2.4877 |
| parrot_raw_L7_k5 | 3.3361 | 3.5130 | 2.4204 |
| parrot_raw_L14_k5 | 3.5849 | 3.7983 | 2.6268 |
| parrot_raw_L7_k1 | 3.7186 | 3.8454 | 2.8012 |
| parrot_raw_L14_k1 | 3.9096 | 4.0613 | 2.9358 |
| parrot_raw_L28_k5 | 3.9107 | 4.1523 | 2.9603 |
| parrot_raw_L28_k1 | 4.0862 | 4.3276 | 3.1940 |

**Picked by val: `naive_last` -> test mean = 2.4453, median = 1.7988**

