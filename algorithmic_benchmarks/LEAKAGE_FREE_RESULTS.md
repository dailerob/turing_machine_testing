# Leakage-free TM benchmark — all methods on equal footing

All methods use the same canonical train / val / test split defined in
`_tm_task_config.py`. Val data is drawn from a stretched range strictly
between train and test (informative for length extrapolation, never
overlapping test). Hyperparameters are picked per task by val tuple errors;
test errors are reported only after the chosen config is fixed.

Tuple errors / total predictions on test set, **bold** = row-best (ties shown).

| Task | Variant | n_pred | GDC | CHMM | ALERGIA | Parrot | HPYLM | PPM-D | KN-3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parity | original | 506 | **10** | 12 | 12 | **10** | 11 | 11 | 13 |
| increment | original | 266 | **0** | **0** | **0** | **0** | 16 | **0** | **0** |
| reverse | original | 13646 | **147** | 329 | 572 | 582 | 590 | 474 | 556 |
| binary_adder | original | 72217 | 59 | **10** | 5579 | 381 | 375 | 178 | 2194 |
| shift_left | original | 526 | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| bit_count_mod3 | original | 526 | 16 | **12** | **12** | 15 | **12** | 14 | **12** |
| anbn | original | 934 | 5 | 5 | 19 | 4 | 4 | 4 | **3** |
| palindrome | original | 1574 | 14 | **6** | 16 | 9 | 9 | 8 | 9 |
| subtraction | original | 33433 | 1234 | 1479 | **736** | 1476 | 1608 | 1856 | 1629 |
| parity | noread | 506 | **10** | 12 | 12 | **10** | 11 | 11 | 13 |
| increment | noread | 266 | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| reverse | noread | 13646 | **0** | 140 | 6260 | 349 | 317 | 317 | 415 |
| binary_adder | noread | 72217 | **0** | **0** | 1466 | 193 | 375 | 375 | 740 |
| shift_left | noread | 526 | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| bit_count_mod3 | noread | 526 | 16 | **12** | **12** | 15 | **12** | 14 | **12** |
| anbn | noread | 934 | 4 | **0** | **0** | 9 | 3 | 3 | 3 |
| palindrome | noread | 1574 | 13 | 9 | 75 | **8** | **8** | **8** | 9 |
| subtraction | noread | 33433 | **1** | 966 | 1597 | 1777 | 1777 | 1563 | 1869 |


## Wins by method (row-best, ties counted for all winners)

| Method | Wins / 18 |
|---|---:|
| GDC | 10 |
| CHMM | 10 |
| ALERGIA | 8 |
| Parrot | 7 |
| HPYLM | 6 |
| PPM-D | 5 |
| KN-3 | 7 |

## Protocol summary

- Train / val / test all drawn fresh per task at the seeds in `_tm_task_config.py`
  (train=42, val=7, test=123).
- Val size = 10% of train.
- Val_range strictly between train_range and test_range (e.g. reverse:
  train (3,6) → val (6,10) → test (10,16)).
- Hyperparameter sweep grids:
    - GDC: alpha × theta = {0.5, 0.7, 0.9, 0.95, 0.99} × {0.005, 0.05},
      transition fixed self_loop, beta=0, terminal=diffuse
    - CHMM: K (clones per emission) ∈ {2, 4, 8}, 50 EM iters, pseudocount=1e-3
    - ALERGIA: eps ∈ {0.001, 0.005, 0.05, 0.5}
    - Parrot: L × K = {1,2,3,4,6,8} × {1, 5, 25}
    - HPYLM: depth × discount × concentration = {3,5,8,12} × {0.25,0.5,0.75} × {0.5,1.0}
    - PPM-D: depth × discount = {3,5,8,12} × {0.25,0.5,0.75}
    - KN-3: discount ∈ {0.5, 0.75, 0.9}, fixed depth 3
- Best config per (method, task, variant) is picked by val tuple errors;
  test is computed only for the winner.
