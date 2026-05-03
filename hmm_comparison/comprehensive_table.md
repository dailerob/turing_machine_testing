# Comprehensive HMM forecasting results @ h=1

Best of each model class per (regime, N), at horizon h=1, averaged over 3 seeds.  All models are run on the same (nS, nA, seed)-matched HMMs as the perplexity sweep (see [run_perplexity_sweep.py](run_perplexity_sweep.py)).

## dense_small

| N | model | MSE | excess PP | abs PP |
|---|---|---:|---:|---:|
| **25** | _entropy floor_ | -- | 1.000 | 3.819 |
| 25 | GDC a0.3-t0.005-b0.3 | 0.00042 | 1.0033 | 3.831 |
| 25 | CHMM K=32 | 0.00429 | 1.0362 | 3.957 |
| 25 | ALERGIA eps=0.05 | 0.00066 | 1.0054 | 3.839 |
| 25 | GDC tuned (diffuse) alpha=0.5 theta=0.05 beta=0.2 | 0.00077 | 1.0061 | 3.842 |
| 25 | GDC tuned (absorb) alpha=0.5 theta=0.05 beta=0.2 | 0.00080 | 1.0063 | 3.843 |
| **100** | _entropy floor_ | -- | 1.000 | 3.819 |
| 100 | GDC a0.5-t0.005-b0.2 | 0.00022 | 1.0017 | 3.825 |
| 100 | CHMM K=32 | 0.00038 | 1.0031 | 3.831 |
| 100 | ALERGIA eps=0.05 | 0.00017 | 1.0013 | 3.824 |
| 100 | GDC tuned (diffuse) alpha=0.5 theta=0.05 beta=0.2 | 0.00048 | 1.0037 | 3.833 |
| 100 | GDC tuned (absorb) alpha=0.5 theta=0.05 beta=0.2 | 0.00049 | 1.0038 | 3.833 |
| **400** | _entropy floor_ | -- | 1.000 | 3.819 |
| 400 | GDC a0.5-t0.005-b0.2 | 0.00017 | 1.0014 | 3.824 |
| 400 | CHMM K=32 | 0.00010 | 1.0008 | 3.822 |
| 400 | ALERGIA eps=0.05 | 0.00010 | 1.0008 | 3.822 |
| 400 | GDC tuned (diffuse) alpha=0.5 theta=0.05 beta=0.2 | 0.00048 | 1.0036 | 3.832 |
| 400 | GDC tuned (absorb) alpha=0.5 theta=0.05 beta=0.2 | 0.00049 | 1.0037 | 3.833 |

## dense_large

| N | model | MSE | excess PP | abs PP |
|---|---|---:|---:|---:|
| **25** | _entropy floor_ | -- | 1.000 | 7.883 |
| 25 | GDC a0.1-t0.001-b0.2 | 0.00013 | 1.0044 | 7.918 |
| 25 | CHMM K=4 | 0.00897 | 1.9725 | 15.550 |
| 25 | ALERGIA eps=0.05 | 0.00072 | 1.0247 | 8.079 |
| 25 | GDC tuned (diffuse) alpha=0.1 theta=0.001 beta=0.2 | 0.00008 | 1.0025 | 7.903 |
| 25 | GDC tuned (absorb) alpha=0.1 theta=0.001 beta=0.2 | 0.00008 | 1.0025 | 7.903 |
| **100** | _entropy floor_ | -- | 1.000 | 7.883 |
| 100 | GDC a0.1-t0.001-b0.2 | 0.00004 | 1.0012 | 7.893 |
| 100 | CHMM K=4 | 0.00255 | 1.1013 | 8.682 |
| 100 | ALERGIA eps=0.05 | 0.00017 | 1.0055 | 7.927 |
| 100 | GDC tuned (diffuse) alpha=0.1 theta=0.001 beta=0.2 | 0.00003 | 1.0009 | 7.891 |
| 100 | GDC tuned (absorb) alpha=0.1 theta=0.001 beta=0.2 | 0.00003 | 1.0009 | 7.891 |
| **400** | _entropy floor_ | -- | 1.000 | 7.883 |
| 400 | GDC a0.1-t0.001-b0.2 | 0.00002 | 1.0005 | 7.888 |
| 400 | CHMM K=32 | 0.00031 | 1.0100 | 7.962 |
| 400 | ALERGIA eps=0.05 | 0.00005 | 1.0015 | 7.895 |
| 400 | GDC tuned (diffuse) alpha=0.1 theta=0.001 beta=0.2 | 0.00001 | 1.0004 | 7.887 |
| 400 | GDC tuned (absorb) alpha=0.1 theta=0.001 beta=0.2 | 0.00001 | 1.0004 | 7.887 |

## det_small

| N | model | MSE | excess PP | abs PP |
|---|---|---:|---:|---:|
| **25** | _entropy floor_ | -- | 1.000 | 3.368 |
| 25 | GDC a0.5-t0.005-b0.2 | 0.00134 | 1.0112 | 3.406 |
| 25 | CHMM K=32 | 0.00381 | 1.0387 | 3.499 |
| 25 | ALERGIA eps=0.05 | 0.00104 | 1.0092 | 3.399 |
| 25 | GDC tuned (diffuse) alpha=0.7 theta=0.01 beta=0.2 | 0.00224 | 1.0169 | 3.425 |
| 25 | GDC tuned (absorb) alpha=0.7 theta=0.01 beta=0.2 | 0.00241 | 1.0181 | 3.429 |
| **100** | _entropy floor_ | -- | 1.000 | 3.368 |
| 100 | GDC a0.5-t0.005-b0.2 | 0.00132 | 1.0109 | 3.405 |
| 100 | CHMM K=32 | 0.00054 | 1.0047 | 3.384 |
| 100 | ALERGIA eps=0.05 | 0.00056 | 1.0044 | 3.383 |
| 100 | GDC tuned (diffuse) alpha=0.7 theta=0.01 beta=0.2 | 0.00120 | 1.0094 | 3.400 |
| 100 | GDC tuned (absorb) alpha=0.7 theta=0.01 beta=0.2 | 0.00130 | 1.0100 | 3.402 |
| **400** | _entropy floor_ | -- | 1.000 | 3.368 |
| 400 | GDC a0.7-t0.01-b0.2 | 0.00091 | 1.0077 | 3.394 |
| 400 | CHMM K=32 | 0.00014 | 1.0012 | 3.373 |
| 400 | ALERGIA eps=0.05 | 0.00042 | 1.0034 | 3.380 |
| 400 | GDC tuned (diffuse) alpha=0.7 theta=0.01 beta=0.2 | 0.00095 | 1.0074 | 3.393 |
| 400 | GDC tuned (absorb) alpha=0.7 theta=0.01 beta=0.2 | 0.00100 | 1.0077 | 3.394 |

## det_large

| N | model | MSE | excess PP | abs PP |
|---|---|---:|---:|---:|
| **25** | _entropy floor_ | -- | 1.000 | 7.434 |
| 25 | GDC a0.3-t0.005-b0.3 | 0.00030 | 1.0100 | 7.508 |
| 25 | CHMM K=4 | 0.00918 | 1.7667 | 13.134 |
| 25 | ALERGIA eps=0.05 | 0.00087 | 1.0643 | 7.912 |
| 25 | GDC tuned (diffuse) alpha=0.3 theta=0.005 beta=0.3 | 0.00031 | 1.0092 | 7.503 |
| 25 | GDC tuned (absorb) alpha=0.3 theta=0.005 beta=0.3 | 0.00031 | 1.0093 | 7.503 |
| **100** | _entropy floor_ | -- | 1.000 | 7.434 |
| 100 | GDC a0.3-t0.005-b0.3 | 0.00018 | 1.0057 | 7.477 |
| 100 | CHMM K=4 | 0.00226 | 1.0913 | 8.113 |
| 100 | ALERGIA eps=0.05 | 0.00018 | 1.0057 | 7.477 |
| 100 | GDC tuned (diffuse) alpha=0.3 theta=0.005 beta=0.3 | 0.00017 | 1.0051 | 7.472 |
| 100 | GDC tuned (absorb) alpha=0.3 theta=0.005 beta=0.3 | 0.00017 | 1.0051 | 7.472 |
| **400** | _entropy floor_ | -- | 1.000 | 7.434 |
| 400 | GDC a0.5-t0.005-b0.2 | 0.00015 | 1.0045 | 7.467 |
| 400 | CHMM K=32 | 0.00035 | 1.0112 | 7.517 |
| 400 | ALERGIA eps=0.05 | 0.00007 | 1.0021 | 7.450 |
| 400 | GDC tuned (diffuse) alpha=0.3 theta=0.005 beta=0.3 | 0.00014 | 1.0041 | 7.464 |
| 400 | GDC tuned (absorb) alpha=0.3 theta=0.005 beta=0.3 | 0.00014 | 1.0041 | 7.464 |

## sparse_small

| N | model | MSE | excess PP | abs PP |
|---|---|---:|---:|---:|
| **25** | _entropy floor_ | -- | 1.000 | 1.796 |
| 25 | GDC a0.8-t0.001-b0.1 | 0.01195 | 1.1097 | 1.993 |
| 25 | CHMM K=32 | 0.00579 | 1.0543 | 1.894 |
| 25 | ALERGIA eps=0.05 | 0.02149 | 1.1931 | 2.143 |
| 25 | GDC tuned (diffuse) alpha=0.8 theta=0.0 beta=0.05 | 0.01405 | 1.1224 | 2.016 |
| 25 | GDC tuned (absorb) alpha=0.8 theta=0.0 beta=0.05 | 0.01438 | 1.1243 | 2.019 |
| **100** | _entropy floor_ | -- | 1.000 | 1.796 |
| 100 | GDC a0.8-t0.001-b0.1 | 0.00898 | 1.0872 | 1.953 |
| 100 | CHMM K=4 | 0.00282 | 1.0336 | 1.856 |
| 100 | ALERGIA eps=0.05 | 0.01331 | 1.1159 | 2.004 |
| 100 | GDC tuned (diffuse) alpha=0.8 theta=0.0 beta=0.05 | 0.01159 | 1.1006 | 1.977 |
| 100 | GDC tuned (absorb) alpha=0.8 theta=0.0 beta=0.05 | 0.01187 | 1.1018 | 1.979 |
| **400** | _entropy floor_ | -- | 1.000 | 1.796 |
| 400 | GDC a0.8-t0.001-b0.1 | 0.00876 | 1.0846 | 1.948 |
| 400 | CHMM K=16 | 0.00194 | 1.0205 | 1.833 |
| 400 | ALERGIA eps=0.05 | 0.01560 | 1.1343 | 2.037 |
| 400 | GDC tuned (diffuse) alpha=0.8 theta=0.0 beta=0.05 | 0.00838 | 1.0784 | 1.937 |
| 400 | GDC tuned (absorb) alpha=0.8 theta=0.0 beta=0.05 | 0.00848 | 1.0781 | 1.936 |

## sparse_large

| N | model | MSE | excess PP | abs PP |
|---|---|---:|---:|---:|
| **25** | _entropy floor_ | -- | 1.000 | 3.073 |
| 25 | GDC a0.7-t0.01-b0.2 | 0.01349 | 1.4160 | 4.351 |
| 25 | CHMM K=4 | 0.01276 | 1.5656 | 4.811 |
| 25 | ALERGIA eps=0.05 | 0.01913 | 1.6557 | 5.088 |
| 25 | GDC tuned (diffuse) alpha=0.8 theta=0.0 beta=0.2 | 0.01570 | 1.4808 | 4.550 |
| 25 | GDC tuned (absorb) alpha=0.8 theta=0.0 beta=0.2 | 0.01588 | 1.4826 | 4.556 |
| **100** | _entropy floor_ | -- | 1.000 | 3.073 |
| 100 | GDC a0.7-t0.01-b0.2 | 0.01284 | 1.3875 | 4.264 |
| 100 | CHMM K=4 | 0.00684 | 1.2087 | 3.714 |
| 100 | ALERGIA eps=0.05 | 0.01596 | 1.4634 | 4.497 |
| 100 | GDC tuned (diffuse) alpha=0.8 theta=0.0 beta=0.2 | 0.01397 | 1.4102 | 4.333 |
| 100 | GDC tuned (absorb) alpha=0.8 theta=0.0 beta=0.2 | 0.01382 | 1.4028 | 4.311 |
| **400** | _entropy floor_ | -- | 1.000 | 3.073 |
| 400 | GDC a0.8-t0.001-b0.1 | 0.01139 | 1.3191 | 4.054 |
| 400 | CHMM K=16 | 0.00237 | 1.0745 | 3.302 |
| 400 | ALERGIA eps=0.05 | 0.01472 | 1.4234 | 4.374 |
| 400 | GDC tuned (diffuse) alpha=0.8 theta=0.0 beta=0.2 | 0.01190 | 1.3550 | 4.164 |
| 400 | GDC tuned (absorb) alpha=0.8 theta=0.0 beta=0.2 | 0.01158 | 1.3438 | 4.129 |

