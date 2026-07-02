# Parrot evaluation: Yearly

- Series: 23000, h = 6
- Candidate configs: 17 (raw/diff × k∈{1,5} × per-freq L grid + naive_last)

## OWA vs published M4 Naive 2 (sMAPE=16.342, MASE=3.9740)

- **(1) Per-series val-tuned OWA = 0.8985**
- (2) Global pick by val-sMAPE [parrot_diff_L3_k5] OWA = 0.8413
- (2') Global pick by val-MASE [parrot_diff_L3_k5] OWA = 0.8413

## (SMAPE) (1) Per-series val-tuned

- Test mean: **15.52%**
- Test median: **8.91%**

Picks:

| config | n picks |
|---|---:|
| parrot_diff_L3_k1 | 3741 |
| naive_last | 3699 |
| naive_last (fallback) | 2923 |
| parrot_diff_L3_k5 | 2863 |
| parrot_diff_L6_k5 | 1652 |
| parrot_diff_L8_k5 | 1544 |
| parrot_diff_L4_k5 | 1481 |
| parrot_diff_L6_k1 | 1261 |
| parrot_diff_L4_k1 | 1148 |
| parrot_diff_L8_k1 | 786 |
| parrot_raw_L3_k5 | 608 |
| parrot_raw_L3_k1 | 526 |
| parrot_raw_L6_k5 | 221 |
| parrot_raw_L4_k5 | 178 |
| parrot_raw_L8_k5 | 130 |
| parrot_raw_L4_k1 | 88 |
| parrot_raw_L6_k1 | 81 |
| parrot_raw_L8_k1 | 70 |

## (SMAPE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_diff_L3_k5 | 15.46% | 14.67% | 8.30% |
| parrot_diff_L4_k5 | 15.62% | 14.78% | 8.31% |
| parrot_diff_L6_k5 | 15.76% | 14.94% | 8.40% |
| parrot_diff_L8_k5 | 16.19% | 14.91% | 8.61% |
| parrot_diff_L6_k1 | 17.19% | 16.91% | 9.44% |
| parrot_diff_L3_k1 | 17.30% | 16.74% | 9.47% |
| parrot_diff_L4_k1 | 17.36% | 16.98% | 9.54% |
| parrot_diff_L8_k1 | 17.70% | 16.75% | 9.55% |
| naive_last | 18.86% | 16.34% | 11.36% |
| parrot_raw_L8_k1 | 28.46% | 26.91% | 20.75% |
| parrot_raw_L6_k1 | 29.32% | 27.28% | 21.05% |
| parrot_raw_L4_k1 | 30.06% | 27.54% | 21.14% |
| parrot_raw_L3_k1 | 30.64% | 27.55% | 21.14% |
| parrot_raw_L8_k5 | 32.77% | 31.15% | 25.66% |
| parrot_raw_L6_k5 | 33.95% | 31.83% | 26.34% |
| parrot_raw_L4_k5 | 34.95% | 32.07% | 26.60% |
| parrot_raw_L3_k5 | 35.59% | 32.06% | 26.56% |

**Picked by val: `parrot_diff_L3_k5` -> test mean = 14.67%, median = 8.30%**

## (MASE) (1) Per-series val-tuned

- Test mean: **3.3657**
- Test median: **2.3582**

Picks:

| config | n picks |
|---|---:|
| parrot_diff_L3_k1 | 3780 |
| naive_last | 3711 |
| naive_last (fallback) | 2923 |
| parrot_diff_L3_k5 | 2831 |
| parrot_diff_L6_k5 | 1626 |
| parrot_diff_L8_k5 | 1529 |
| parrot_diff_L4_k5 | 1469 |
| parrot_diff_L6_k1 | 1257 |
| parrot_diff_L4_k1 | 1156 |
| parrot_diff_L8_k1 | 790 |
| parrot_raw_L3_k5 | 633 |
| parrot_raw_L3_k1 | 539 |
| parrot_raw_L6_k5 | 219 |
| parrot_raw_L4_k5 | 183 |
| parrot_raw_L8_k5 | 126 |
| parrot_raw_L4_k1 | 85 |
| parrot_raw_L6_k1 | 75 |
| parrot_raw_L8_k1 | 68 |

## (MASE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_diff_L3_k5 | 508044030486.1638 | 3.1181 | 2.2063 |
| parrot_diff_L4_k5 | 508044030486.1797 | 3.1212 | 2.2104 |
| parrot_diff_L6_k5 | 508044030486.2225 | 3.1463 | 2.2447 |
| parrot_diff_L8_k5 | 508044030486.3395 | 3.1856 | 2.2573 |
| parrot_diff_L6_k1 | 508044030486.4218 | 3.4721 | 2.5317 |
| parrot_diff_L4_k1 | 508044030486.4258 | 3.4721 | 2.5372 |
| parrot_diff_L3_k1 | 508044030486.4282 | 3.4496 | 2.5084 |
| parrot_diff_L8_k1 | 508044030486.5322 | 3.4890 | 2.5368 |
| naive_last | 508044030487.4661 | 3.9744 | 2.9370 |
| parrot_raw_L8_k1 | 508044030489.4973 | 6.2833 | 4.9803 |
| parrot_raw_L6_k1 | 508044030489.6897 | 6.3359 | 5.0426 |
| parrot_raw_L4_k1 | 508044030489.7980 | 6.3747 | 5.0846 |
| parrot_raw_L3_k1 | 508044030489.8726 | 6.3732 | 5.0969 |
| parrot_raw_L8_k5 | 508044030490.6113 | 7.4845 | 5.9717 |
| parrot_raw_L6_k5 | 508044030490.8794 | 7.5911 | 6.1913 |
| parrot_raw_L4_k5 | 508044030491.0680 | 7.6277 | 6.3066 |
| parrot_raw_L3_k5 | 508044030491.1588 | 7.6235 | 6.2978 |

**Picked by val: `parrot_diff_L3_k5` -> test mean = 3.1181, median = 2.2063**

## (MSE) (1) Per-series val-tuned

- Test mean: **3.939e+06**
- Test median: **2.473e+05**

Picks:

| config | n picks |
|---|---:|
| naive_last | 3768 |
| parrot_diff_L3_k1 | 3741 |
| naive_last (fallback) | 2923 |
| parrot_diff_L3_k5 | 2809 |
| parrot_diff_L6_k5 | 1684 |
| parrot_diff_L8_k5 | 1524 |
| parrot_diff_L4_k5 | 1445 |
| parrot_diff_L6_k1 | 1269 |
| parrot_diff_L4_k1 | 1134 |
| parrot_diff_L8_k1 | 800 |
| parrot_raw_L3_k5 | 669 |
| parrot_raw_L3_k1 | 489 |
| parrot_raw_L6_k5 | 226 |
| parrot_raw_L4_k5 | 161 |
| parrot_raw_L8_k5 | 140 |
| parrot_raw_L4_k1 | 81 |
| parrot_raw_L6_k1 | 69 |
| parrot_raw_L8_k1 | 68 |

## (MSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_diff_L3_k5 | 2.529e+06 | 3.386e+06 | 2.173e+05 |
| parrot_diff_L4_k5 | 2.559e+06 | 3.415e+06 | 2.191e+05 |
| parrot_diff_L6_k5 | 2.61e+06 | 3.468e+06 | 2.226e+05 |
| parrot_diff_L8_k5 | 2.66e+06 | 3.501e+06 | 2.308e+05 |
| parrot_diff_L4_k1 | 3.009e+06 | 4.255e+06 | 2.878e+05 |
| parrot_diff_L3_k1 | 3.035e+06 | 4.136e+06 | 2.806e+05 |
| naive_last | 3.096e+06 | 4.066e+06 | 3.697e+05 |
| parrot_diff_L6_k1 | 3.102e+06 | 4.304e+06 | 2.901e+05 |
| parrot_diff_L8_k1 | 3.183e+06 | 4.228e+06 | 2.939e+05 |
| parrot_raw_L8_k1 | 5.103e+06 | 7.615e+06 | 9.377e+05 |
| parrot_raw_L6_k1 | 5.123e+06 | 7.725e+06 | 9.586e+05 |
| parrot_raw_L4_k1 | 5.203e+06 | 7.795e+06 | 9.698e+05 |
| parrot_raw_L3_k1 | 5.294e+06 | 7.823e+06 | 9.719e+05 |
| parrot_raw_L8_k5 | 5.982e+06 | 9.189e+06 | 1.227e+06 |
| parrot_raw_L6_k5 | 6.107e+06 | 9.416e+06 | 1.281e+06 |
| parrot_raw_L4_k5 | 6.214e+06 | 9.491e+06 | 1.287e+06 |
| parrot_raw_L3_k5 | 6.318e+06 | 9.478e+06 | 1.277e+06 |

**Picked by val: `parrot_diff_L3_k5` -> test mean = 3.386e+06, median = 2.173e+05**

## (NRMSE) (1) Per-series val-tuned

- Test mean: **3.6652**
- Test median: **2.6175**

Picks:

| config | n picks |
|---|---:|
| naive_last | 3768 |
| parrot_diff_L3_k1 | 3741 |
| naive_last (fallback) | 2923 |
| parrot_diff_L3_k5 | 2809 |
| parrot_diff_L6_k5 | 1684 |
| parrot_diff_L8_k5 | 1524 |
| parrot_diff_L4_k5 | 1445 |
| parrot_diff_L6_k1 | 1269 |
| parrot_diff_L4_k1 | 1134 |
| parrot_diff_L8_k1 | 800 |
| parrot_raw_L3_k5 | 671 |
| parrot_raw_L3_k1 | 489 |
| parrot_raw_L6_k5 | 224 |
| parrot_raw_L4_k5 | 161 |
| parrot_raw_L8_k5 | 140 |
| parrot_raw_L4_k1 | 81 |
| parrot_raw_L6_k1 | 69 |
| parrot_raw_L8_k1 | 68 |

## (NRMSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_diff_L3_k5 | 2.9720 | 3.3930 | 2.4636 |
| parrot_diff_L4_k5 | 2.9975 | 3.3930 | 2.4802 |
| parrot_diff_L6_k5 | 3.0506 | 3.4155 | 2.5078 |
| parrot_diff_L8_k5 | 3.2039 | 3.4950 | 2.5079 |
| parrot_diff_L3_k1 | 3.2367 | 3.7362 | 2.8153 |
| parrot_diff_L4_k1 | 3.2411 | 3.7523 | 2.8317 |
| parrot_diff_L6_k1 | 3.2422 | 3.7501 | 2.8430 |
| parrot_diff_L8_k1 | 3.3911 | 3.8085 | 2.8421 |
| naive_last | 4.5749 | 4.7368 | 3.2591 |
| parrot_raw_L8_k1 | 6.2136 | 7.0026 | 4.9950 |
| parrot_raw_L6_k1 | 6.3939 | 7.0520 | 5.0808 |
| parrot_raw_L4_k1 | 6.4950 | 7.0888 | 5.1302 |
| parrot_raw_L3_k1 | 6.5671 | 7.0916 | 5.1453 |
| parrot_raw_L8_k5 | 7.3298 | 8.4239 | 5.8234 |
| parrot_raw_L6_k5 | 7.5962 | 8.5565 | 6.0032 |
| parrot_raw_L4_k5 | 7.7899 | 8.6083 | 6.0959 |
| parrot_raw_L3_k5 | 7.8857 | 8.6100 | 6.0892 |

**Picked by val: `parrot_diff_L3_k5` -> test mean = 3.3930, median = 2.4636**

