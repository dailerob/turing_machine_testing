# Clean (leakage-free) evaluation: Monthly

- Series: 48000, h = 18
- Candidate configs: 8 (all GDC-TS absorb + naive fallback)

## OWA vs published M4 Naive 2 (sMAPE=14.427, MASE=1.0630)

- **(1) Per-series val-tuned OWA = 0.9685**
- (2) Global pick by val-sMAPE [naive_last] OWA = 1.0956
- (2') Global pick by val-MASE [gdc_L12_s0.25_a0.99] OWA = 0.9597

## (SMAPE) (1) Per-series val-tuned

- Test mean: **14.32%**
- Test median: **7.48%**

Picks:

| config | n picks |
|---|---:|
| naive_last | 15342 |
| gdc_L12_s0.25_a0.99 | 12960 |
| gdc_L6_s0.25_a0.95 | 7658 |
| gdc_L12_s0.50_a0.95 | 3764 |
| gdc_L6_s0.50_a0.95 | 2654 |
| gdc_L18_s0.25_a0.95 | 2317 |
| gdc_L12_s0.25_a0.9 | 2049 |
| gdc_L12_s0.25_a0.95 | 1256 |

## (SMAPE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| naive_last | 13.99% | 15.26% | 8.61% |
| gdc_L12_s0.25_a0.9 | 15.41% | 14.06% | 7.74% |
| gdc_L6_s0.25_a0.95 | 15.41% | 14.12% | 7.68% |
| gdc_L12_s0.25_a0.95 | 15.46% | 13.96% | 7.54% |
| gdc_L18_s0.25_a0.95 | 15.51% | 14.03% | 7.57% |
| gdc_L6_s0.50_a0.95 | 15.78% | 14.23% | 7.81% |
| gdc_L12_s0.25_a0.99 | 16.13% | 14.31% | 7.47% |
| gdc_L12_s0.50_a0.95 | 16.17% | 14.28% | 7.76% |

**Picked by val: `naive_last` -> test mean = 15.26%, median = 8.61%**

## (MASE) (1) Per-series val-tuned

- Test mean: **1.0039**
- Test median: **0.7841**

Picks:

| config | n picks |
|---|---:|
| naive_last | 15268 |
| gdc_L12_s0.25_a0.99 | 13015 |
| gdc_L6_s0.25_a0.95 | 7676 |
| gdc_L12_s0.50_a0.95 | 3775 |
| gdc_L6_s0.50_a0.95 | 2659 |
| gdc_L18_s0.25_a0.95 | 2327 |
| gdc_L12_s0.25_a0.9 | 2024 |
| gdc_L12_s0.25_a0.95 | 1256 |

## (MASE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L12_s0.25_a0.99 | 1.0102 | 0.9859 | 0.7695 |
| gdc_L12_s0.25_a0.95 | 1.0219 | 1.0049 | 0.7744 |
| gdc_L18_s0.25_a0.95 | 1.0298 | 1.0122 | 0.7771 |
| gdc_L6_s0.25_a0.95 | 1.0339 | 1.0212 | 0.7873 |
| gdc_L12_s0.25_a0.9 | 1.0538 | 1.0404 | 0.7946 |
| gdc_L12_s0.50_a0.95 | 1.0776 | 1.0488 | 0.8012 |
| gdc_L6_s0.50_a0.95 | 1.0877 | 1.0565 | 0.8037 |
| naive_last | 1.1908 | 1.2051 | 0.9022 |

**Picked by val: `gdc_L12_s0.25_a0.99` -> test mean = 0.9859, median = 0.7695**

## (MSE) (1) Per-series val-tuned

- Test mean: **2.161e+06**
- Test median: **9.627e+04**

Picks:

| config | n picks |
|---|---:|
| naive_last | 14996 |
| gdc_L12_s0.25_a0.99 | 13564 |
| gdc_L6_s0.25_a0.95 | 7591 |
| gdc_L12_s0.50_a0.95 | 3786 |
| gdc_L6_s0.50_a0.95 | 2664 |
| gdc_L18_s0.25_a0.95 | 2232 |
| gdc_L12_s0.25_a0.9 | 1999 |
| gdc_L12_s0.25_a0.95 | 1168 |

## (MSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L6_s0.25_a0.95 | 2.23e+06 | 2.167e+06 | 9.579e+04 |
| gdc_L12_s0.25_a0.95 | 2.232e+06 | 2.121e+06 | 9.379e+04 |
| gdc_L12_s0.25_a0.9 | 2.239e+06 | 2.153e+06 | 9.778e+04 |
| gdc_L18_s0.25_a0.95 | 2.257e+06 | 2.142e+06 | 9.46e+04 |
| gdc_L6_s0.50_a0.95 | 2.319e+06 | 2.207e+06 | 1.003e+05 |
| gdc_L12_s0.25_a0.99 | 2.337e+06 | 2.166e+06 | 9.328e+04 |
| naive_last | 2.426e+06 | 2.501e+06 | 1.284e+05 |
| gdc_L12_s0.50_a0.95 | 2.435e+06 | 2.216e+06 | 9.914e+04 |

**Picked by val: `gdc_L6_s0.25_a0.95` -> test mean = 2.167e+06, median = 9.579e+04**

## (NRMSE) (1) Per-series val-tuned

- Test mean: **2.4014**
- Test median: **1.5450**

Picks:

| config | n picks |
|---|---:|
| naive_last | 14996 |
| gdc_L12_s0.25_a0.99 | 13564 |
| gdc_L6_s0.25_a0.95 | 7591 |
| gdc_L12_s0.50_a0.95 | 3786 |
| gdc_L6_s0.50_a0.95 | 2664 |
| gdc_L18_s0.25_a0.95 | 2232 |
| gdc_L12_s0.25_a0.9 | 1999 |
| gdc_L12_s0.25_a0.95 | 1168 |

## (NRMSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L12_s0.25_a0.95 | 2.3048 | 2.3725 | 1.5010 |
| gdc_L18_s0.25_a0.95 | 2.3120 | 2.3804 | 1.5130 |
| gdc_L6_s0.25_a0.95 | 2.3142 | 2.3925 | 1.5228 |
| gdc_L12_s0.25_a0.9 | 2.3282 | 2.4089 | 1.5391 |
| gdc_L12_s0.25_a0.99 | 2.3430 | 2.3816 | 1.5038 |
| gdc_L6_s0.50_a0.95 | 2.3569 | 2.4162 | 1.5627 |
| gdc_L12_s0.50_a0.95 | 2.3693 | 2.4151 | 1.5554 |
| naive_last | 2.4575 | 2.7316 | 1.7998 |

**Picked by val: `gdc_L12_s0.25_a0.95` -> test mean = 2.3725, median = 1.5010**

