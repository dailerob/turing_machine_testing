# Clean (leakage-free) evaluation: Quarterly

- Series: 24000, h = 8
- Candidate configs: 7 (all GDC-TS absorb + naive fallback)

## OWA vs published M4 Naive 2 (sMAPE=11.012, MASE=1.3710)

- **(1) Per-series val-tuned OWA = 0.9119**
- (2) Global pick by val-sMAPE [gdc_L12_s0.25_a0.9] OWA = 0.9136
- (2') Global pick by val-MASE [gdc_L12_s0.25_a0.95] OWA = 0.9098

## (SMAPE) (1) Per-series val-tuned

- Test mean: **10.47%**
- Test median: **5.77%**

Picks:

| config | n picks |
|---|---:|
| naive_last | 7816 |
| gdc_L12_s0.25_a0.95 | 4096 |
| gdc_L8_s0.25_a0.9 | 3562 |
| gdc_L4_s0.50_a0.9 | 3241 |
| gdc_L8_s0.50_a0.9 | 2647 |
| gdc_L6_s0.50_a0.9 | 1661 |
| gdc_L12_s0.25_a0.9 | 975 |
| naive_last (fallback) | 2 |

## (SMAPE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L12_s0.25_a0.9 | 11.00% | 10.53% | 5.69% |
| gdc_L8_s0.25_a0.9 | 11.02% | 10.55% | 5.69% |
| gdc_L12_s0.25_a0.95 | 11.04% | 10.54% | 5.67% |
| gdc_L4_s0.50_a0.9 | 11.05% | 10.58% | 5.75% |
| gdc_L6_s0.50_a0.9 | 11.09% | 10.63% | 5.76% |
| gdc_L8_s0.50_a0.9 | 11.11% | 10.67% | 5.79% |
| naive_last | 11.85% | 11.61% | 7.10% |

**Picked by val: `gdc_L12_s0.25_a0.9` -> test mean = 10.53%, median = 5.69%**

## (MASE) (1) Per-series val-tuned

- Test mean: **1.1971**
- Test median: **0.9306**

Picks:

| config | n picks |
|---|---:|
| naive_last | 7840 |
| gdc_L12_s0.25_a0.95 | 4110 |
| gdc_L8_s0.25_a0.9 | 3556 |
| gdc_L4_s0.50_a0.9 | 3235 |
| gdc_L8_s0.50_a0.9 | 2641 |
| gdc_L6_s0.50_a0.9 | 1644 |
| gdc_L12_s0.25_a0.9 | 972 |
| naive_last (fallback) | 2 |

## (MASE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L12_s0.25_a0.95 | 34377864823.3227 | 1.1824 | 0.9224 |
| gdc_L12_s0.25_a0.9 | 34377864823.3294 | 1.1939 | 0.9330 |
| gdc_L8_s0.25_a0.9 | 34377864823.3299 | 1.1925 | 0.9304 |
| gdc_L4_s0.50_a0.9 | 34377864823.3335 | 1.1998 | 0.9377 |
| gdc_L6_s0.50_a0.9 | 34377864823.3389 | 1.2066 | 0.9434 |
| gdc_L8_s0.50_a0.9 | 34377864823.3420 | 1.2117 | 0.9464 |
| naive_last | 34377864823.5500 | 1.4770 | 1.1639 |

**Picked by val: `gdc_L12_s0.25_a0.95` -> test mean = 1.1824, median = 0.9224**

## (MSE) (1) Per-series val-tuned

- Test mean: **1.943e+06**
- Test median: **9.671e+04**

Picks:

| config | n picks |
|---|---:|
| naive_last | 7755 |
| gdc_L12_s0.25_a0.95 | 4259 |
| gdc_L8_s0.25_a0.9 | 3518 |
| gdc_L4_s0.50_a0.9 | 3229 |
| gdc_L8_s0.50_a0.9 | 2669 |
| gdc_L6_s0.50_a0.9 | 1643 |
| gdc_L12_s0.25_a0.9 | 925 |
| naive_last (fallback) | 2 |

## (MSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L12_s0.25_a0.9 | 1.96e+06 | 1.94e+06 | 9.923e+04 |
| gdc_L12_s0.25_a0.95 | 1.967e+06 | 1.937e+06 | 9.71e+04 |
| gdc_L8_s0.25_a0.9 | 1.97e+06 | 1.942e+06 | 9.978e+04 |
| gdc_L4_s0.50_a0.9 | 1.98e+06 | 1.966e+06 | 9.951e+04 |
| gdc_L6_s0.50_a0.9 | 1.984e+06 | 1.99e+06 | 1.02e+05 |
| gdc_L8_s0.50_a0.9 | 1.993e+06 | 2.016e+06 | 1.025e+05 |
| naive_last | 2.138e+06 | 2.27e+06 | 1.484e+05 |

**Picked by val: `gdc_L12_s0.25_a0.9` -> test mean = 1.94e+06, median = 9.923e+04**

## (NRMSE) (1) Per-series val-tuned

- Test mean: **2.0384**
- Test median: **1.4133**

Picks:

| config | n picks |
|---|---:|
| naive_last | 7755 |
| gdc_L12_s0.25_a0.95 | 4259 |
| gdc_L8_s0.25_a0.9 | 3518 |
| gdc_L4_s0.50_a0.9 | 3229 |
| gdc_L8_s0.50_a0.9 | 2669 |
| gdc_L6_s0.50_a0.9 | 1643 |
| gdc_L12_s0.25_a0.9 | 925 |
| naive_last (fallback) | 2 |

## (NRMSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L4_s0.50_a0.9 | 1.9484 | 2.0248 | 1.4041 |
| gdc_L12_s0.25_a0.9 | 1.9548 | 2.0228 | 1.3994 |
| gdc_L8_s0.25_a0.9 | 1.9575 | 2.0201 | 1.4015 |
| gdc_L6_s0.50_a0.9 | 1.9590 | 2.0331 | 1.4090 |
| gdc_L12_s0.25_a0.95 | 1.9595 | 2.0187 | 1.4024 |
| gdc_L8_s0.50_a0.9 | 1.9630 | 2.0405 | 1.4108 |
| naive_last | 2.4576 | 2.6576 | 1.6872 |

**Picked by val: `gdc_L4_s0.50_a0.9` -> test mean = 2.0248, median = 1.4041**

