# Parrot evaluation: Quarterly

- Series: 24000, h = 8
- Candidate configs: 17 (raw/diff × k∈{1,5} × per-freq L grid + naive_last)

## OWA vs published M4 Naive 2 (sMAPE=11.012, MASE=1.3710)

- **(1) Per-series val-tuned OWA = 1.0609**
- (2) Global pick by val-sMAPE [parrot_diff_L4_k5] OWA = 0.9616
- (2') Global pick by val-MASE [parrot_diff_L4_k5] OWA = 0.9616

## (SMAPE) (1) Per-series val-tuned

- Test mean: **12.32%**
- Test median: **6.65%**

Picks:

| config | n picks |
|---|---:|
| naive_last | 4096 |
| parrot_diff_L4_k1 | 2741 |
| parrot_diff_L4_k5 | 2731 |
| parrot_diff_L12_k5 | 2356 |
| parrot_diff_L6_k5 | 1887 |
| parrot_diff_L8_k5 | 1756 |
| parrot_diff_L6_k1 | 1330 |
| parrot_raw_L4_k5 | 1305 |
| parrot_diff_L12_k1 | 1111 |
| parrot_diff_L8_k1 | 1024 |
| parrot_raw_L4_k1 | 1011 |
| parrot_raw_L6_k5 | 663 |
| parrot_raw_L12_k5 | 605 |
| parrot_raw_L8_k5 | 579 |
| parrot_raw_L6_k1 | 333 |
| parrot_raw_L12_k1 | 246 |
| parrot_raw_L8_k1 | 224 |
| naive_last (fallback) | 2 |

## (SMAPE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_diff_L4_k5 | 11.71% | 11.27% | 6.01% |
| parrot_diff_L6_k5 | 11.83% | 11.39% | 6.03% |
| naive_last | 11.85% | 11.61% | 7.10% |
| parrot_diff_L8_k5 | 11.96% | 11.45% | 6.11% |
| parrot_diff_L12_k5 | 12.21% | 11.71% | 6.22% |
| parrot_diff_L4_k1 | 14.15% | 13.91% | 7.36% |
| parrot_diff_L6_k1 | 14.31% | 13.89% | 7.33% |
| parrot_diff_L8_k1 | 14.40% | 13.84% | 7.36% |
| parrot_diff_L12_k1 | 14.63% | 14.14% | 7.50% |
| parrot_raw_L4_k5 | 17.30% | 16.87% | 11.47% |
| parrot_raw_L6_k5 | 17.61% | 17.25% | 11.61% |
| parrot_raw_L8_k5 | 17.84% | 17.52% | 11.73% |
| parrot_raw_L4_k1 | 17.90% | 17.42% | 11.31% |
| parrot_raw_L6_k1 | 18.07% | 17.68% | 11.37% |
| parrot_raw_L12_k5 | 18.08% | 17.91% | 11.84% |
| parrot_raw_L8_k1 | 18.19% | 17.85% | 11.41% |
| parrot_raw_L12_k1 | 18.30% | 18.04% | 11.33% |

**Picked by val: `parrot_diff_L4_k5` -> test mean = 11.27%, median = 6.01%**

## (MASE) (1) Per-series val-tuned

- Test mean: **1.3756**
- Test median: **1.0648**

Picks:

| config | n picks |
|---|---:|
| naive_last | 4105 |
| parrot_diff_L4_k1 | 2739 |
| parrot_diff_L4_k5 | 2712 |
| parrot_diff_L12_k5 | 2347 |
| parrot_diff_L6_k5 | 1891 |
| parrot_diff_L8_k5 | 1750 |
| parrot_raw_L4_k5 | 1329 |
| parrot_diff_L6_k1 | 1323 |
| parrot_diff_L12_k1 | 1107 |
| parrot_diff_L8_k1 | 1023 |
| parrot_raw_L4_k1 | 1021 |
| parrot_raw_L6_k5 | 670 |
| parrot_raw_L12_k5 | 611 |
| parrot_raw_L8_k5 | 576 |
| parrot_raw_L6_k1 | 323 |
| parrot_raw_L12_k1 | 247 |
| parrot_raw_L8_k1 | 224 |
| naive_last (fallback) | 2 |

## (MASE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_diff_L4_k5 | 34377864823.3745 | 1.2333 | 0.9689 |
| parrot_diff_L6_k5 | 34377864823.3893 | 1.2449 | 0.9744 |
| parrot_diff_L8_k5 | 34377864823.4033 | 1.2497 | 0.9813 |
| parrot_diff_L12_k5 | 34377864823.4163 | 1.2738 | 1.0008 |
| naive_last | 34377864823.5500 | 1.4770 | 1.1639 |
| parrot_diff_L4_k1 | 34377864823.6190 | 1.4875 | 1.1834 |
| parrot_diff_L6_k1 | 34377864823.6296 | 1.4908 | 1.1951 |
| parrot_diff_L8_k1 | 34377864823.6450 | 1.4868 | 1.1891 |
| parrot_diff_L12_k1 | 34377864823.6595 | 1.5153 | 1.2068 |
| parrot_raw_L4_k1 | 34377864824.3184 | 2.2228 | 1.7889 |
| parrot_raw_L12_k1 | 34377864824.3198 | 2.2604 | 1.8185 |
| parrot_raw_L6_k1 | 34377864824.3309 | 2.2468 | 1.8083 |
| parrot_raw_L8_k1 | 34377864824.3371 | 2.2522 | 1.8158 |
| parrot_raw_L4_k5 | 34377864824.3659 | 2.2939 | 1.7743 |
| parrot_raw_L6_k5 | 34377864824.3927 | 2.3281 | 1.8092 |
| parrot_raw_L8_k5 | 34377864824.4057 | 2.3488 | 1.8338 |
| parrot_raw_L12_k5 | 34377864824.4068 | 2.3790 | 1.8694 |

**Picked by val: `parrot_diff_L4_k5` -> test mean = 1.2333, median = 0.9689**

## (MSE) (1) Per-series val-tuned

- Test mean: **2.581e+06**
- Test median: **1.256e+05**

Picks:

| config | n picks |
|---|---:|
| naive_last | 4200 |
| parrot_diff_L4_k1 | 2711 |
| parrot_diff_L4_k5 | 2710 |
| parrot_diff_L12_k5 | 2406 |
| parrot_diff_L6_k5 | 1878 |
| parrot_diff_L8_k5 | 1758 |
| parrot_raw_L4_k5 | 1326 |
| parrot_diff_L6_k1 | 1313 |
| parrot_diff_L12_k1 | 1095 |
| parrot_diff_L8_k1 | 1034 |
| parrot_raw_L4_k1 | 949 |
| parrot_raw_L6_k5 | 677 |
| parrot_raw_L12_k5 | 622 |
| parrot_raw_L8_k5 | 566 |
| parrot_raw_L6_k1 | 295 |
| parrot_raw_L12_k1 | 241 |
| parrot_raw_L8_k1 | 217 |
| naive_last (fallback) | 2 |

## (MSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| naive_last | 2.138e+06 | 2.27e+06 | 1.484e+05 |
| parrot_diff_L4_k5 | 2.146e+06 | 2.148e+06 | 1.056e+05 |
| parrot_diff_L6_k5 | 2.19e+06 | 2.189e+06 | 1.082e+05 |
| parrot_diff_L12_k5 | 2.227e+06 | 2.279e+06 | 1.125e+05 |
| parrot_diff_L8_k5 | 2.252e+06 | 2.186e+06 | 1.094e+05 |
| parrot_diff_L4_k1 | 2.917e+06 | 3.185e+06 | 1.602e+05 |
| parrot_diff_L8_k1 | 3.059e+06 | 3.088e+06 | 1.632e+05 |
| parrot_diff_L6_k1 | 3.11e+06 | 3.241e+06 | 1.611e+05 |
| parrot_diff_L12_k1 | 3.143e+06 | 3.237e+06 | 1.647e+05 |
| parrot_raw_L4_k5 | 3.532e+06 | 3.507e+06 | 3.141e+05 |
| parrot_raw_L6_k5 | 3.636e+06 | 3.626e+06 | 3.244e+05 |
| parrot_raw_L8_k5 | 3.731e+06 | 3.763e+06 | 3.316e+05 |
| parrot_raw_L12_k5 | 3.816e+06 | 3.852e+06 | 3.457e+05 |
| parrot_raw_L6_k1 | 4.14e+06 | 4.144e+06 | 3.294e+05 |
| parrot_raw_L4_k1 | 4.175e+06 | 3.991e+06 | 3.219e+05 |
| parrot_raw_L8_k1 | 4.183e+06 | 4.225e+06 | 3.303e+05 |
| parrot_raw_L12_k1 | 4.244e+06 | 4.338e+06 | 3.32e+05 |

**Picked by val: `naive_last` -> test mean = 2.27e+06, median = 1.484e+05**

## (NRMSE) (1) Per-series val-tuned

- Test mean: **2.3505**
- Test median: **1.6565**

Picks:

| config | n picks |
|---|---:|
| naive_last | 4200 |
| parrot_diff_L4_k5 | 2711 |
| parrot_diff_L4_k1 | 2711 |
| parrot_diff_L12_k5 | 2406 |
| parrot_diff_L6_k5 | 1878 |
| parrot_diff_L8_k5 | 1757 |
| parrot_raw_L4_k5 | 1337 |
| parrot_diff_L6_k1 | 1313 |
| parrot_diff_L12_k1 | 1095 |
| parrot_diff_L8_k1 | 1034 |
| parrot_raw_L4_k1 | 949 |
| parrot_raw_L6_k5 | 673 |
| parrot_raw_L12_k5 | 620 |
| parrot_raw_L8_k5 | 561 |
| parrot_raw_L6_k1 | 295 |
| parrot_raw_L12_k1 | 241 |
| parrot_raw_L8_k1 | 217 |
| naive_last (fallback) | 2 |

## (NRMSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_diff_L4_k5 | 2.0690 | 2.1279 | 1.5080 |
| parrot_diff_L6_k5 | 2.1054 | 2.1465 | 1.5150 |
| parrot_diff_L8_k5 | 2.1343 | 2.1595 | 1.5252 |
| parrot_diff_L12_k5 | 2.1531 | 2.2112 | 1.5417 |
| naive_last | 2.4576 | 2.6576 | 1.6872 |
| parrot_diff_L4_k1 | 2.4887 | 2.5657 | 1.8661 |
| parrot_diff_L6_k1 | 2.5170 | 2.5644 | 1.8805 |
| parrot_diff_L8_k1 | 2.5529 | 2.5661 | 1.8856 |
| parrot_diff_L12_k1 | 2.5733 | 2.6176 | 1.8887 |
| parrot_raw_L12_k1 | 3.9839 | 4.2469 | 2.6635 |
| parrot_raw_L4_k1 | 4.0266 | 4.2068 | 2.5878 |
| parrot_raw_L6_k1 | 4.0380 | 4.2324 | 2.6291 |
| parrot_raw_L8_k1 | 4.0410 | 4.2414 | 2.6525 |
| parrot_raw_L4_k5 | 4.2136 | 4.4675 | 2.4761 |
| parrot_raw_L12_k5 | 4.2304 | 4.5842 | 2.6393 |
| parrot_raw_L6_k5 | 4.2478 | 4.5160 | 2.5358 |
| parrot_raw_L8_k5 | 4.2544 | 4.5459 | 2.5707 |

**Picked by val: `parrot_diff_L4_k5` -> test mean = 2.1279, median = 1.5080**

