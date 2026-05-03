# Clean (leakage-free) evaluation: Weekly

- Series: 359, h = 13
- Candidate configs: 7 (all GDC-TS absorb + naive fallback)

## OWA vs published M4 Naive 2 (sMAPE=9.161, MASE=2.7770)

- **(1) Per-series val-tuned OWA = 0.7854**
- (2) Global pick by val-sMAPE [gdc_L26_s0.10_a0.99] OWA = 0.7998
- (2') Global pick by val-MASE [gdc_L26_s0.10_a0.99] OWA = 0.7998

## (SMAPE) (1) Per-series val-tuned

- Test mean: **7.39%**
- Test median: **4.19%**

Picks:

| config | n picks |
|---|---:|
| gdc_L26_s0.10_a0.99 | 110 |
| gdc_L13_s0.25_a0.99 | 73 |
| naive_last | 72 |
| gdc_L52_s0.25_a0.99 | 45 |
| gdc_L26_s0.50_a0.99 | 32 |
| gdc_L26_s0.25_a0.99 | 17 |
| gdc_L26_s0.25_a0.95 | 10 |

## (SMAPE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L26_s0.10_a0.99 | 8.38% | 7.22% | 4.23% |
| gdc_L52_s0.25_a0.99 | 8.56% | 7.33% | 4.44% |
| naive_last | 9.20% | 9.16% | 5.18% |
| gdc_L26_s0.25_a0.95 | 9.32% | 7.60% | 4.59% |
| gdc_L26_s0.25_a0.99 | 9.48% | 7.13% | 4.21% |
| gdc_L26_s0.50_a0.99 | 9.61% | 7.98% | 4.83% |
| gdc_L13_s0.25_a0.99 | 9.99% | 7.53% | 4.59% |

**Picked by val: `gdc_L26_s0.10_a0.99` -> test mean = 7.22%, median = 4.23%**

## (MASE) (1) Per-series val-tuned

- Test mean: **2.1205**
- Test median: **1.4604**

Picks:

| config | n picks |
|---|---:|
| gdc_L26_s0.10_a0.99 | 111 |
| naive_last | 72 |
| gdc_L13_s0.25_a0.99 | 71 |
| gdc_L52_s0.25_a0.99 | 45 |
| gdc_L26_s0.50_a0.99 | 32 |
| gdc_L26_s0.25_a0.99 | 18 |
| gdc_L26_s0.25_a0.95 | 10 |

## (MASE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L26_s0.10_a0.99 | 2.5158 | 2.2534 | 1.5390 |
| gdc_L13_s0.25_a0.99 | 2.5274 | 2.1757 | 1.4729 |
| gdc_L52_s0.25_a0.99 | 2.5854 | 2.2063 | 1.4929 |
| gdc_L26_s0.50_a0.99 | 2.6179 | 2.3393 | 1.6292 |
| gdc_L26_s0.25_a0.99 | 2.6186 | 2.1609 | 1.4885 |
| gdc_L26_s0.25_a0.95 | 2.6433 | 2.2740 | 1.5851 |
| naive_last | 2.8975 | 2.7773 | 1.9384 |

**Picked by val: `gdc_L26_s0.10_a0.99` -> test mean = 2.2534, median = 1.5390**

## (MSE) (1) Per-series val-tuned

- Test mean: **3.207e+05**
- Test median: **3.799e+04**

Picks:

| config | n picks |
|---|---:|
| gdc_L26_s0.10_a0.99 | 112 |
| gdc_L13_s0.25_a0.99 | 77 |
| naive_last | 70 |
| gdc_L52_s0.25_a0.99 | 42 |
| gdc_L26_s0.50_a0.99 | 34 |
| gdc_L26_s0.25_a0.99 | 18 |
| gdc_L26_s0.25_a0.95 | 6 |

## (MSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L26_s0.10_a0.99 | 2.922e+05 | 3.36e+05 | 4.094e+04 |
| gdc_L52_s0.25_a0.99 | 3.224e+05 | 3.114e+05 | 4.061e+04 |
| gdc_L26_s0.25_a0.95 | 3.419e+05 | 3.536e+05 | 4.984e+04 |
| gdc_L26_s0.25_a0.99 | 3.459e+05 | 3.276e+05 | 4.245e+04 |
| gdc_L26_s0.50_a0.99 | 3.679e+05 | 3.581e+05 | 5.289e+04 |
| naive_last | 3.7e+05 | 4.535e+05 | 7.325e+04 |
| gdc_L13_s0.25_a0.99 | 3.929e+05 | 3.623e+05 | 4.231e+04 |

**Picked by val: `gdc_L26_s0.10_a0.99` -> test mean = 3.36e+05, median = 4.094e+04**

## (NRMSE) (1) Per-series val-tuned

- Test mean: **1.4960**
- Test median: **1.1441**

Picks:

| config | n picks |
|---|---:|
| gdc_L26_s0.10_a0.99 | 112 |
| gdc_L13_s0.25_a0.99 | 77 |
| naive_last | 70 |
| gdc_L52_s0.25_a0.99 | 42 |
| gdc_L26_s0.50_a0.99 | 34 |
| gdc_L26_s0.25_a0.99 | 18 |
| gdc_L26_s0.25_a0.95 | 6 |

## (NRMSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L26_s0.10_a0.99 | 1.7241 | 1.5519 | 1.1777 |
| gdc_L13_s0.25_a0.99 | 1.7555 | 1.5246 | 1.1834 |
| gdc_L52_s0.25_a0.99 | 1.7640 | 1.5257 | 1.2102 |
| gdc_L26_s0.25_a0.99 | 1.7965 | 1.5080 | 1.2011 |
| gdc_L26_s0.25_a0.95 | 1.8097 | 1.5969 | 1.2604 |
| gdc_L26_s0.50_a0.99 | 1.8165 | 1.6239 | 1.2840 |
| naive_last | 1.9651 | 1.9075 | 1.4643 |

**Picked by val: `gdc_L26_s0.10_a0.99` -> test mean = 1.5519, median = 1.1777**

