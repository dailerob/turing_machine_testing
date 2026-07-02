# Parrot evaluation: Hourly

- Series: 414, h = 48
- Candidate configs: 17 (raw/diff × k∈{1,5} × per-freq L grid + naive_last)

## OWA vs published M4 Naive 2 (sMAPE=18.383, MASE=2.3950)

- **(1) Per-series val-tuned OWA = 0.5219**
- (2) Global pick by val-sMAPE [parrot_raw_L168_k1] OWA = 0.5471
- (2') Global pick by val-MASE [parrot_diff_L48_k5] OWA = 0.6028

## (SMAPE) (1) Per-series val-tuned

- Test mean: **12.01%**
- Test median: **5.23%**

Picks:

| config | n picks |
|---|---:|
| parrot_diff_L24_k1 | 58 |
| parrot_raw_L24_k1 | 37 |
| parrot_diff_L24_k5 | 32 |
| parrot_raw_L168_k5 | 30 |
| parrot_diff_L168_k5 | 29 |
| parrot_diff_L48_k1 | 26 |
| parrot_raw_L72_k5 | 26 |
| parrot_diff_L72_k5 | 25 |
| parrot_raw_L24_k5 | 25 |
| parrot_diff_L48_k5 | 24 |
| parrot_raw_L48_k5 | 24 |
| parrot_raw_L48_k1 | 21 |
| parrot_diff_L168_k1 | 21 |
| parrot_diff_L72_k1 | 16 |
| parrot_raw_L168_k1 | 12 |
| parrot_raw_L72_k1 | 8 |

## (SMAPE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_raw_L168_k1 | 11.16% | 10.45% | 6.34% |
| parrot_raw_L168_k5 | 11.17% | 10.03% | 5.71% |
| parrot_raw_L48_k5 | 11.66% | 11.53% | 5.16% |
| parrot_raw_L72_k5 | 11.70% | 11.65% | 5.73% |
| parrot_raw_L72_k1 | 11.96% | 12.14% | 6.13% |
| parrot_raw_L24_k5 | 12.09% | 12.15% | 4.93% |
| parrot_raw_L48_k1 | 12.46% | 11.87% | 6.03% |
| parrot_raw_L24_k1 | 12.79% | 12.96% | 5.90% |
| parrot_diff_L168_k5 | 14.16% | 13.39% | 4.96% |
| parrot_diff_L72_k5 | 14.41% | 15.18% | 4.81% |
| parrot_diff_L48_k5 | 15.02% | 14.74% | 4.43% |
| parrot_diff_L24_k5 | 15.04% | 14.71% | 4.11% |
| parrot_diff_L168_k1 | 15.50% | 14.40% | 6.13% |
| parrot_diff_L48_k1 | 16.87% | 16.81% | 6.15% |
| parrot_diff_L72_k1 | 16.94% | 17.32% | 6.12% |
| parrot_diff_L24_k1 | 17.40% | 17.43% | 5.82% |
| naive_last | 41.40% | 43.00% | 19.88% |

**Picked by val: `parrot_raw_L168_k1` -> test mean = 10.45%, median = 6.34%**

## (MASE) (1) Per-series val-tuned

- Test mean: **0.9352**
- Test median: **0.7881**

Picks:

| config | n picks |
|---|---:|
| parrot_diff_L24_k1 | 61 |
| parrot_raw_L24_k1 | 39 |
| parrot_diff_L24_k5 | 34 |
| parrot_raw_L168_k5 | 32 |
| parrot_diff_L48_k5 | 29 |
| parrot_diff_L72_k5 | 27 |
| parrot_diff_L168_k5 | 25 |
| parrot_diff_L168_k1 | 25 |
| parrot_diff_L48_k1 | 23 |
| parrot_raw_L24_k5 | 22 |
| parrot_raw_L48_k1 | 21 |
| parrot_raw_L48_k5 | 21 |
| parrot_raw_L72_k5 | 21 |
| parrot_diff_L72_k1 | 16 |
| parrot_raw_L168_k1 | 11 |
| parrot_raw_L72_k1 | 7 |

## (MASE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_diff_L48_k5 | 0.9871 | 0.9677 | 0.7920 |
| parrot_diff_L72_k5 | 0.9891 | 0.9901 | 0.8656 |
| parrot_diff_L168_k5 | 0.9925 | 0.9499 | 0.7810 |
| parrot_diff_L24_k5 | 1.0001 | 0.9674 | 0.8296 |
| parrot_diff_L168_k1 | 1.0588 | 1.0210 | 0.8515 |
| parrot_diff_L48_k1 | 1.0680 | 1.0836 | 0.9227 |
| parrot_diff_L72_k1 | 1.0886 | 1.1097 | 0.9395 |
| parrot_diff_L24_k1 | 1.1091 | 1.1260 | 0.9801 |
| parrot_raw_L72_k1 | 1.2694 | 1.3387 | 1.2297 |
| parrot_raw_L168_k1 | 1.2779 | 1.2592 | 1.0836 |
| parrot_raw_L48_k1 | 1.2834 | 1.3145 | 1.1832 |
| parrot_raw_L24_k1 | 1.3119 | 1.3181 | 1.1796 |
| parrot_raw_L48_k5 | 1.6545 | 1.6925 | 1.1782 |
| parrot_raw_L72_k5 | 1.6644 | 1.7294 | 1.2265 |
| parrot_raw_L168_k5 | 1.6688 | 1.6532 | 1.0816 |
| parrot_raw_L24_k5 | 1.6733 | 1.6729 | 1.1386 |
| naive_last | 11.5323 | 11.6077 | 3.6849 |

**Picked by val: `parrot_diff_L48_k5` -> test mean = 0.9677, median = 0.7920**

## (MSE) (1) Per-series val-tuned

- Test mean: **2.752e+06**
- Test median: **459.5**

Picks:

| config | n picks |
|---|---:|
| parrot_diff_L24_k1 | 56 |
| parrot_diff_L24_k5 | 52 |
| parrot_raw_L24_k1 | 38 |
| parrot_diff_L48_k5 | 33 |
| parrot_diff_L168_k5 | 32 |
| parrot_raw_L168_k5 | 30 |
| parrot_diff_L72_k5 | 29 |
| parrot_raw_L48_k5 | 22 |
| parrot_diff_L48_k1 | 21 |
| parrot_diff_L168_k1 | 20 |
| parrot_raw_L24_k5 | 20 |
| parrot_raw_L72_k5 | 19 |
| parrot_raw_L48_k1 | 15 |
| parrot_diff_L72_k1 | 14 |
| parrot_raw_L168_k1 | 8 |
| parrot_raw_L72_k1 | 5 |

## (MSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_diff_L72_k5 | 4.109e+06 | 2.633e+06 | 703.6 |
| parrot_diff_L168_k5 | 4.118e+06 | 2.987e+06 | 371.7 |
| parrot_diff_L48_k5 | 4.181e+06 | 1.849e+06 | 708.4 |
| parrot_diff_L24_k5 | 5.787e+06 | 1.172e+06 | 613.1 |
| parrot_diff_L24_k1 | 7.968e+06 | 2.17e+06 | 730.5 |
| parrot_raw_L168_k5 | 8.359e+06 | 5.216e+06 | 296.5 |
| parrot_raw_L168_k1 | 8.958e+06 | 6.23e+06 | 389.2 |
| parrot_raw_L72_k1 | 1.031e+07 | 6.641e+06 | 520.9 |
| parrot_raw_L24_k1 | 1.049e+07 | 2.755e+06 | 573 |
| parrot_raw_L48_k5 | 1.136e+07 | 5.584e+06 | 475.3 |
| parrot_raw_L48_k1 | 1.142e+07 | 5.807e+06 | 616.2 |
| parrot_diff_L48_k1 | 1.17e+07 | 2.331e+06 | 832.8 |
| parrot_diff_L72_k1 | 1.222e+07 | 5.096e+06 | 960.1 |
| parrot_diff_L168_k1 | 1.227e+07 | 7.015e+06 | 509.5 |
| parrot_raw_L72_k5 | 1.236e+07 | 6.509e+06 | 404.2 |
| parrot_raw_L24_k5 | 1.292e+07 | 1.585e+06 | 486 |
| naive_last | 5.42e+07 | 5.754e+07 | 3770 |

**Picked by val: `parrot_diff_L72_k5` -> test mean = 2.633e+06, median = 703.6**

## (NRMSE) (1) Per-series val-tuned

- Test mean: **0.8860**
- Test median: **0.5922**

Picks:

| config | n picks |
|---|---:|
| parrot_diff_L24_k1 | 56 |
| parrot_diff_L24_k5 | 52 |
| parrot_raw_L24_k1 | 38 |
| parrot_diff_L48_k5 | 33 |
| parrot_diff_L168_k5 | 32 |
| parrot_raw_L168_k5 | 30 |
| parrot_diff_L72_k5 | 29 |
| parrot_raw_L48_k5 | 22 |
| parrot_diff_L48_k1 | 21 |
| parrot_diff_L168_k1 | 20 |
| parrot_raw_L24_k5 | 20 |
| parrot_raw_L72_k5 | 19 |
| parrot_raw_L48_k1 | 15 |
| parrot_diff_L72_k1 | 14 |
| parrot_raw_L168_k1 | 8 |
| parrot_raw_L72_k1 | 5 |

## (NRMSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_diff_L48_k5 | 0.9178 | 0.8619 | 0.7021 |
| parrot_diff_L168_k5 | 0.9199 | 0.8544 | 0.6102 |
| parrot_diff_L72_k5 | 0.9308 | 0.8930 | 0.6992 |
| parrot_diff_L24_k5 | 0.9314 | 0.8783 | 0.6394 |
| parrot_raw_L48_k5 | 0.9920 | 1.0160 | 0.7222 |
| parrot_raw_L72_k5 | 0.9939 | 1.0515 | 0.7355 |
| parrot_raw_L72_k1 | 1.0024 | 1.0760 | 0.7129 |
| parrot_raw_L168_k1 | 1.0097 | 0.9970 | 0.5923 |
| parrot_raw_L168_k5 | 1.0110 | 0.9691 | 0.6558 |
| parrot_raw_L24_k5 | 1.0245 | 0.9859 | 0.7259 |
| parrot_raw_L48_k1 | 1.0309 | 1.0428 | 0.7255 |
| parrot_diff_L48_k1 | 1.0507 | 1.0563 | 0.8197 |
| parrot_raw_L24_k1 | 1.0662 | 1.0511 | 0.7885 |
| parrot_diff_L168_k1 | 1.0890 | 1.0271 | 0.6611 |
| parrot_diff_L72_k1 | 1.0934 | 1.1061 | 0.8382 |
| parrot_diff_L24_k1 | 1.1145 | 1.0807 | 0.8526 |
| naive_last | 3.7496 | 3.7704 | 3.9291 |

**Picked by val: `parrot_diff_L48_k5` -> test mean = 0.8619, median = 0.7021**

