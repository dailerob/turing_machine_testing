# Clean (leakage-free) evaluation: Daily

- Series: 4227, h = 14
- Candidate configs: 7 (all GDC-TS absorb + naive fallback)

## OWA vs published M4 Naive 2 (sMAPE=3.045, MASE=3.2780)

- **(1) Per-series val-tuned OWA = 0.9923**
- (2) Global pick by val-sMAPE [naive_last] OWA = 1.0001
- (2') Global pick by val-MASE [naive_last] OWA = 1.0001

## (SMAPE) (1) Per-series val-tuned

- Test mean: **3.04%**
- Test median: **1.97%**

Picks:

| config | n picks |
|---|---:|
| naive_last | 1933 |
| gdc_L28_s0.50_a1.0 | 836 |
| gdc_L7_s0.50_a1.0 | 485 |
| gdc_L14_s0.50_a1.0 | 474 |
| gdc_L7_s1.00_a1.0 | 234 |
| gdc_L14_s1.00_a1.0 | 225 |
| gdc_L14_s0.50_a0.99 | 40 |

## (SMAPE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| naive_last | 2.79% | 3.05% | 1.99% |
| gdc_L7_s0.50_a1.0 | 2.83% | 3.03% | 1.96% |
| gdc_L7_s1.00_a1.0 | 2.85% | 3.01% | 1.96% |
| gdc_L14_s0.50_a0.99 | 2.85% | 2.97% | 1.95% |
| gdc_L14_s0.50_a1.0 | 2.86% | 3.02% | 1.96% |
| gdc_L14_s1.00_a1.0 | 2.86% | 3.03% | 1.96% |
| gdc_L28_s0.50_a1.0 | 2.88% | 3.02% | 1.95% |

**Picked by val: `naive_last` -> test mean = 3.05%, median = 1.99%**

## (MASE) (1) Per-series val-tuned

- Test mean: **3.2375**
- Test median: **2.3384**

Picks:

| config | n picks |
|---|---:|
| naive_last | 1936 |
| gdc_L28_s0.50_a1.0 | 837 |
| gdc_L7_s0.50_a1.0 | 485 |
| gdc_L14_s0.50_a1.0 | 472 |
| gdc_L7_s1.00_a1.0 | 233 |
| gdc_L14_s1.00_a1.0 | 225 |
| gdc_L14_s0.50_a0.99 | 39 |

## (MASE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| naive_last | 651931461019.1385 | 3.2784 | 2.3557 |
| gdc_L7_s0.50_a1.0 | 651931461019.1561 | 3.2100 | 2.3087 |
| gdc_L14_s0.50_a1.0 | 651931461019.1617 | 3.2102 | 2.2941 |
| gdc_L7_s1.00_a1.0 | 651931461019.1636 | 3.2067 | 2.3025 |
| gdc_L14_s0.50_a0.99 | 651931461019.1689 | 3.1922 | 2.2962 |
| gdc_L14_s1.00_a1.0 | 651931461019.1697 | 3.2155 | 2.2997 |
| gdc_L28_s0.50_a1.0 | 651931461019.1869 | 3.2159 | 2.2872 |

**Picked by val: `naive_last` -> test mean = 3.2784, median = 2.3557**

## (MSE) (1) Per-series val-tuned

- Test mean: **5.409e+05**
- Test median: **1.169e+04**

Picks:

| config | n picks |
|---|---:|
| naive_last | 1955 |
| gdc_L28_s0.50_a1.0 | 834 |
| gdc_L7_s0.50_a1.0 | 497 |
| gdc_L14_s0.50_a1.0 | 468 |
| gdc_L7_s1.00_a1.0 | 223 |
| gdc_L14_s1.00_a1.0 | 212 |
| gdc_L14_s0.50_a0.99 | 38 |

## (MSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L7_s0.50_a1.0 | 2.139e+05 | 5.138e+05 | 1.147e+04 |
| gdc_L14_s0.50_a1.0 | 2.615e+05 | 5.333e+05 | 1.155e+04 |
| gdc_L14_s0.50_a0.99 | 2.674e+05 | 5.131e+05 | 1.145e+04 |
| gdc_L7_s1.00_a1.0 | 2.681e+05 | 5.114e+05 | 1.145e+04 |
| naive_last | 2.829e+05 | 4.976e+05 | 1.175e+04 |
| gdc_L14_s1.00_a1.0 | 2.834e+05 | 5.177e+05 | 1.144e+04 |
| gdc_L28_s0.50_a1.0 | 3.012e+05 | 5.273e+05 | 1.153e+04 |

**Picked by val: `gdc_L7_s0.50_a1.0` -> test mean = 5.138e+05, median = 1.147e+04**

## (NRMSE) (1) Per-series val-tuned

- Test mean: **2.4182**
- Test median: **1.7704**

Picks:

| config | n picks |
|---|---:|
| naive_last | 1955 |
| gdc_L28_s0.50_a1.0 | 834 |
| gdc_L7_s0.50_a1.0 | 497 |
| gdc_L14_s0.50_a1.0 | 468 |
| gdc_L7_s1.00_a1.0 | 223 |
| gdc_L14_s1.00_a1.0 | 212 |
| gdc_L14_s0.50_a0.99 | 38 |

## (NRMSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| naive_last | 2.3102 | 2.4453 | 1.7988 |
| gdc_L7_s1.00_a1.0 | 2.3317 | 2.3923 | 1.7511 |
| gdc_L7_s0.50_a1.0 | 2.3337 | 2.3966 | 1.7558 |
| gdc_L14_s1.00_a1.0 | 2.3349 | 2.3974 | 1.7453 |
| gdc_L14_s0.50_a1.0 | 2.3359 | 2.3933 | 1.7494 |
| gdc_L14_s0.50_a0.99 | 2.3365 | 2.3835 | 1.7448 |
| gdc_L28_s0.50_a1.0 | 2.3484 | 2.3986 | 1.7342 |

**Picked by val: `naive_last` -> test mean = 2.4453, median = 1.7988**

