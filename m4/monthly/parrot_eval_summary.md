# Parrot evaluation: Monthly

- Series: 48000, h = 18
- Candidate configs: 13 (raw/diff × k∈{1,5} × per-freq L grid + naive_last)

## OWA vs published M4 Naive 2 (sMAPE=14.427, MASE=1.0630)

- **(1) Per-series val-tuned OWA = 1.0576**
- (2) Global pick by val-sMAPE [naive_last] OWA = 1.0956
- (2') Global pick by val-MASE [parrot_diff_L6_k5] OWA = 1.0161

## (SMAPE) (1) Per-series val-tuned

- Test mean: **15.51%**
- Test median: **8.42%**

Picks:

| config | n picks |
|---|---:|
| naive_last | 10540 |
| parrot_diff_L6_k5 | 5590 |
| parrot_diff_L12_k5 | 5315 |
| parrot_diff_L18_k5 | 5101 |
| parrot_diff_L6_k1 | 4708 |
| parrot_raw_L6_k5 | 3332 |
| parrot_diff_L12_k1 | 3309 |
| parrot_diff_L18_k1 | 2292 |
| parrot_raw_L12_k5 | 2210 |
| parrot_raw_L18_k5 | 2135 |
| parrot_raw_L6_k1 | 1963 |
| parrot_raw_L12_k1 | 926 |
| parrot_raw_L18_k1 | 579 |

## (SMAPE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| naive_last | 13.99% | 15.26% | 8.61% |
| parrot_diff_L6_k5 | 17.85% | 15.19% | 8.02% |
| parrot_raw_L6_k5 | 18.27% | 18.79% | 11.85% |
| parrot_raw_L18_k5 | 18.46% | 19.19% | 12.43% |
| parrot_raw_L12_k5 | 18.64% | 18.95% | 12.09% |
| parrot_raw_L6_k1 | 19.34% | 20.04% | 12.92% |
| parrot_diff_L12_k5 | 19.44% | 15.36% | 7.85% |
| parrot_raw_L18_k1 | 19.51% | 20.10% | 12.99% |
| parrot_raw_L12_k1 | 19.59% | 19.95% | 12.86% |
| parrot_diff_L18_k5 | 20.45% | 15.95% | 7.94% |
| parrot_diff_L6_k1 | 21.13% | 19.17% | 9.67% |
| parrot_diff_L12_k1 | 22.24% | 18.84% | 9.37% |
| parrot_diff_L18_k1 | 23.14% | 19.54% | 9.42% |

**Picked by val: `naive_last` -> test mean = 15.26%, median = 8.61%**

## (MASE) (1) Per-series val-tuned

- Test mean: **1.1060**
- Test median: **0.8598**

Picks:

| config | n picks |
|---|---:|
| naive_last | 10647 |
| parrot_diff_L6_k5 | 5645 |
| parrot_diff_L12_k5 | 5337 |
| parrot_diff_L18_k5 | 5083 |
| parrot_diff_L6_k1 | 4679 |
| parrot_diff_L12_k1 | 3314 |
| parrot_raw_L6_k5 | 3286 |
| parrot_diff_L18_k1 | 2295 |
| parrot_raw_L12_k5 | 2201 |
| parrot_raw_L18_k5 | 2115 |
| parrot_raw_L6_k1 | 1923 |
| parrot_raw_L12_k1 | 906 |
| parrot_raw_L18_k1 | 569 |

## (MASE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| parrot_diff_L6_k5 | 1.0771 | 1.0412 | 0.8203 |
| parrot_diff_L12_k5 | 1.0886 | 1.0307 | 0.8112 |
| parrot_diff_L18_k5 | 1.1098 | 1.0438 | 0.8278 |
| naive_last | 1.1908 | 1.2051 | 0.9022 |
| parrot_diff_L12_k1 | 1.2717 | 1.2191 | 0.9966 |
| parrot_diff_L6_k1 | 1.2750 | 1.2533 | 1.0239 |
| parrot_diff_L18_k1 | 1.2883 | 1.2330 | 1.0108 |
| parrot_raw_L6_k5 | 1.5510 | 1.5584 | 1.1798 |
| parrot_raw_L12_k5 | 1.5641 | 1.5749 | 1.1725 |
| parrot_raw_L18_k5 | 1.5841 | 1.5975 | 1.1881 |
| parrot_raw_L12_k1 | 1.6116 | 1.6213 | 1.2538 |
| parrot_raw_L6_k1 | 1.6163 | 1.6300 | 1.2807 |
| parrot_raw_L18_k1 | 1.6251 | 1.6324 | 1.2590 |

**Picked by val: `parrot_diff_L6_k5` -> test mean = 1.0412, median = 0.8203**

## (MSE) (1) Per-series val-tuned

- Test mean: **2.491e+06**
- Test median: **1.179e+05**

Picks:

| config | n picks |
|---|---:|
| naive_last | 10588 |
| parrot_diff_L6_k5 | 5724 |
| parrot_diff_L12_k5 | 5404 |
| parrot_diff_L18_k5 | 5213 |
| parrot_diff_L6_k1 | 4576 |
| parrot_raw_L6_k5 | 3418 |
| parrot_diff_L12_k1 | 3278 |
| parrot_diff_L18_k1 | 2290 |
| parrot_raw_L12_k5 | 2196 |
| parrot_raw_L18_k5 | 2107 |
| parrot_raw_L6_k1 | 1830 |
| parrot_raw_L12_k1 | 840 |
| parrot_raw_L18_k1 | 536 |

## (MSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| naive_last | 2.426e+06 | 2.501e+06 | 1.284e+05 |
| parrot_diff_L6_k5 | 2.543e+06 | 2.27e+06 | 1.083e+05 |
| parrot_diff_L12_k5 | 2.801e+06 | 2.339e+06 | 1.037e+05 |
| parrot_raw_L6_k5 | 2.861e+06 | 3.073e+06 | 2.205e+05 |
| parrot_raw_L18_k5 | 2.865e+06 | 3.192e+06 | 2.251e+05 |
| parrot_diff_L18_k5 | 2.965e+06 | 2.412e+06 | 1.072e+05 |
| parrot_raw_L12_k5 | 2.965e+06 | 3.127e+06 | 2.185e+05 |
| parrot_diff_L6_k1 | 3.387e+06 | 3.185e+06 | 1.655e+05 |
| parrot_raw_L6_k1 | 3.448e+06 | 3.69e+06 | 2.548e+05 |
| parrot_raw_L18_k1 | 3.457e+06 | 3.753e+06 | 2.506e+05 |
| parrot_raw_L12_k1 | 3.481e+06 | 3.632e+06 | 2.455e+05 |
| parrot_diff_L12_k1 | 3.625e+06 | 3.162e+06 | 1.543e+05 |
| parrot_diff_L18_k1 | 3.738e+06 | 3.258e+06 | 1.565e+05 |

**Picked by val: `naive_last` -> test mean = 2.501e+06, median = 1.284e+05**

## (NRMSE) (1) Per-series val-tuned

- Test mean: **2.6901**
- Test median: **1.7121**

Picks:

| config | n picks |
|---|---:|
| naive_last | 10588 |
| parrot_diff_L6_k5 | 5724 |
| parrot_diff_L12_k5 | 5404 |
| parrot_diff_L18_k5 | 5213 |
| parrot_diff_L6_k1 | 4576 |
| parrot_raw_L6_k5 | 3422 |
| parrot_diff_L12_k1 | 3278 |
| parrot_diff_L18_k1 | 2290 |
| parrot_raw_L12_k5 | 2202 |
| parrot_raw_L18_k5 | 2097 |
| parrot_raw_L6_k1 | 1830 |
| parrot_raw_L12_k1 | 840 |
| parrot_raw_L18_k1 | 536 |

## (NRMSE) (2) Global single config picked by mean val

| config | mean val | mean test | median test |
|---|---:|---:|---:|
| naive_last | 2.4575 | 2.7316 | 1.7998 |
| parrot_diff_L6_k5 | 2.5552 | 2.5349 | 1.6213 |
| parrot_diff_L12_k5 | 2.6678 | 2.5381 | 1.5870 |
| parrot_diff_L18_k5 | 2.7354 | 2.5822 | 1.6131 |
| parrot_diff_L6_k1 | 3.0299 | 3.0690 | 1.9992 |
| parrot_diff_L12_k1 | 3.0863 | 2.9971 | 1.9412 |
| parrot_diff_L18_k1 | 3.1412 | 3.0387 | 1.9664 |
| parrot_raw_L6_k5 | 3.8481 | 4.0167 | 2.2609 |
| parrot_raw_L6_k1 | 3.9360 | 4.1351 | 2.4832 |
| parrot_raw_L12_k5 | 3.9397 | 4.1138 | 2.2754 |
| parrot_raw_L18_k5 | 3.9542 | 4.1631 | 2.3302 |
| parrot_raw_L12_k1 | 3.9898 | 4.1738 | 2.4427 |
| parrot_raw_L18_k1 | 4.0080 | 4.1999 | 2.4754 |

**Picked by val: `naive_last` -> test mean = 2.7316, median = 1.7998**

