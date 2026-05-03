# Clean (leakage-free) evaluation: Yearly

- Series: 23000, h = 6
- Candidate configs: 7 (all GDC-TS absorb + naive fallback)

## OWA vs published M4 Naive 2 (sMAPE=16.342, MASE=3.9740)

- **(1) Per-series val-tuned OWA = 0.8188**
- (2) Global pick by val-sMAPE [gdc_L3_s0.50_a0.8] OWA = 0.8110
- (2') Global pick by val-MASE [gdc_L3_s0.50_a0.8] OWA = 0.8110

## (SMAPE) (1) Per-series val-tuned

- Test mean: **13.99%**
- Test median: **8.27%**

Picks:

| config | n picks |
|---|---:|
| gdc_L8_s0.50_a0.9 | 5708 |
| naive_last | 5420 |
| naive_last (fallback) | 2923 |
| gdc_L3_s0.50_a0.8 | 2904 |
| gdc_L8_s0.25_a0.8 | 2415 |
| gdc_L4_s0.50_a0.8 | 1477 |
| gdc_L6_s0.50_a0.8 | 1248 |
| gdc_L8_s0.50_a0.8 | 905 |

## (SMAPE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L3_s0.50_a0.8 | 14.89% | 14.04% | 8.00% |
| gdc_L4_s0.50_a0.8 | 14.94% | 14.03% | 8.00% |
| gdc_L6_s0.50_a0.8 | 14.98% | 13.96% | 8.00% |
| gdc_L8_s0.50_a0.9 | 15.35% | 14.10% | 8.19% |
| gdc_L8_s0.50_a0.8 | 15.39% | 14.05% | 8.24% |
| gdc_L8_s0.25_a0.8 | 15.39% | 14.10% | 8.28% |
| naive_last | 18.86% | 16.34% | 11.36% |

**Picked by val: `gdc_L3_s0.50_a0.8` -> test mean = 14.04%, median = 8.00%**

## (MASE) (1) Per-series val-tuned

- Test mean: **3.1052**
- Test median: **2.1295**

Picks:

| config | n picks |
|---|---:|
| gdc_L8_s0.50_a0.9 | 5743 |
| naive_last | 5451 |
| naive_last (fallback) | 2923 |
| gdc_L3_s0.50_a0.8 | 2885 |
| gdc_L8_s0.25_a0.8 | 2399 |
| gdc_L4_s0.50_a0.8 | 1468 |
| gdc_L6_s0.50_a0.8 | 1237 |
| gdc_L8_s0.50_a0.8 | 894 |

## (MASE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L3_s0.50_a0.8 | 508044030486.0966 | 3.0320 | 2.0820 |
| gdc_L4_s0.50_a0.8 | 508044030486.1110 | 3.0274 | 2.0776 |
| gdc_L6_s0.50_a0.8 | 508044030486.1487 | 3.0216 | 2.0712 |
| gdc_L8_s0.50_a0.9 | 508044030486.2323 | 3.0660 | 2.1196 |
| gdc_L8_s0.50_a0.8 | 508044030486.2742 | 3.0776 | 2.1007 |
| gdc_L8_s0.25_a0.8 | 508044030486.2774 | 3.0959 | 2.1093 |
| naive_last | 508044030487.4659 | 3.9744 | 2.9370 |

**Picked by val: `gdc_L3_s0.50_a0.8` -> test mean = 3.0320, median = 2.0820**

## (MSE) (1) Per-series val-tuned

- Test mean: **3.371e+06**
- Test median: **2.096e+05**

Picks:

| config | n picks |
|---|---:|
| gdc_L8_s0.50_a0.9 | 5937 |
| naive_last | 5530 |
| naive_last (fallback) | 2923 |
| gdc_L3_s0.50_a0.8 | 2801 |
| gdc_L8_s0.25_a0.8 | 2323 |
| gdc_L4_s0.50_a0.8 | 1400 |
| gdc_L6_s0.50_a0.8 | 1187 |
| gdc_L8_s0.50_a0.8 | 899 |

## (MSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L3_s0.50_a0.8 | 2.435e+06 | 3.251e+06 | 1.978e+05 |
| gdc_L4_s0.50_a0.8 | 2.439e+06 | 3.246e+06 | 1.963e+05 |
| gdc_L6_s0.50_a0.8 | 2.446e+06 | 3.24e+06 | 1.964e+05 |
| gdc_L8_s0.50_a0.8 | 2.473e+06 | 3.326e+06 | 2.035e+05 |
| gdc_L8_s0.50_a0.9 | 2.475e+06 | 3.327e+06 | 2.047e+05 |
| gdc_L8_s0.25_a0.8 | 2.491e+06 | 3.351e+06 | 2.066e+05 |
| naive_last | 3.096e+06 | 4.066e+06 | 3.697e+05 |

**Picked by val: `gdc_L3_s0.50_a0.8` -> test mean = 3.251e+06, median = 1.978e+05**

## (NRMSE) (1) Per-series val-tuned

- Test mean: **3.4460**
- Test median: **2.3937**

Picks:

| config | n picks |
|---|---:|
| gdc_L8_s0.50_a0.9 | 5937 |
| naive_last | 5530 |
| naive_last (fallback) | 2923 |
| gdc_L3_s0.50_a0.8 | 2801 |
| gdc_L8_s0.25_a0.8 | 2323 |
| gdc_L4_s0.50_a0.8 | 1400 |
| gdc_L6_s0.50_a0.8 | 1187 |
| gdc_L8_s0.50_a0.8 | 899 |

## (NRMSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| gdc_L3_s0.50_a0.8 | 2.9279 | 3.3148 | 2.3689 |
| gdc_L4_s0.50_a0.8 | 2.9518 | 3.3103 | 2.3638 |
| gdc_L6_s0.50_a0.8 | 3.0033 | 3.3059 | 2.3560 |
| gdc_L8_s0.50_a0.9 | 3.1127 | 3.3843 | 2.3656 |
| gdc_L8_s0.50_a0.8 | 3.1564 | 3.4035 | 2.3611 |
| gdc_L8_s0.25_a0.8 | 3.1622 | 3.4220 | 2.3759 |
| naive_last | 4.5749 | 4.7368 | 3.2591 |

**Picked by val: `gdc_L3_s0.50_a0.8` -> test mean = 3.3148, median = 2.3689**

