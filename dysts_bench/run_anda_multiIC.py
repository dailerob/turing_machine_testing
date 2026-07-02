"""AnDA on dysts univariate with fixed-fit-len multi-IC val.

Same "full" config grid as run_anda.py (3 regressions × 3 E × 4 K per E
= 36 configs) but val sMAPE is averaged across 3 sliding val windows
within the train trajectory, all at fixed fit_len=90.

Output: results/anda_dysts_full_multiIC3_fitlen90.csv
"""
from __future__ import annotations
import os, sys, json, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANDA_PATH = '/home/roberto/turing_machine_testing/external/AnDA'
sys.path.insert(0, ANDA_PATH)

from AnDA_codes.AnDA_analog_forecasting import AnDA_analog_forecasting

DATA_DIR   = os.path.join(HERE, 'data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.json')
TEST_PATH  = os.path.join(DATA_DIR, 'test.json')

TRAIN_FIT_LEN  = 150
TEST_TRUTH_LEN = 29
PRED_LEN       = 30

VAL_FIT_LEN = 90
VAL_STARTS  = [90, 120, 150]


class _Catalog:
    def __init__(self, analogs, successors):
        self.analogs = analogs
        self.successors = successors


class _AF:
    def __init__(self, catalog, k, regression, sampling, E):
        self.catalog = catalog
        self.k = k
        self.regression = regression
        self.sampling = sampling
        self.neighborhood = np.ones((E, E), dtype=np.int64)


def _build_catalog(series, E):
    series = np.asarray(series, dtype=np.float64)
    n_pairs = len(series) - E
    if n_pairs < 2:
        return None
    analogs = np.zeros((n_pairs, E))
    successors = np.zeros((n_pairs, E))
    for i in range(n_pairs):
        analogs[i] = series[i:i + E]
        successors[i] = series[i + 1:i + E + 1]
    return _Catalog(analogs, successors)


def _forecast_one(series_fit, E, k, regression, n_steps=PRED_LEN,
                   sampling='gaussian'):
    catalog = _build_catalog(series_fit, E)
    if catalog is None or catalog.analogs.shape[0] < k:
        return None
    af = _AF(catalog, k=k, regression=regression, sampling=sampling, E=E)
    cur = np.asarray(series_fit[-E:], dtype=np.float64).reshape(1, E)
    preds = []
    for _ in range(n_steps):
        try:
            _, xf_mean = AnDA_analog_forecasting(cur, af)
        except Exception:
            return None
        if not np.all(np.isfinite(xf_mean)):
            return None
        next_val = float(xf_mean[0, -1])
        preds.append(next_val)
        cur = np.column_stack([cur[:, 1:], np.array([[next_val]])])
    return np.asarray(preds, dtype=np.float64)


def smape(t, p):
    t = np.asarray(t, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    denom = (np.abs(t) + np.abs(p)) / 2.0
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return float(100 * np.mean(np.abs(t - p) / denom))


def mse(t, p):
    return float(np.mean((np.asarray(t, dtype=np.float64) -
                          np.asarray(p, dtype=np.float64)) ** 2))


def mae(t, p):
    return float(np.mean(np.abs(np.asarray(t, dtype=np.float64) -
                                np.asarray(p, dtype=np.float64))))


def run_one_system(system, train_traj, test_traj, configs):
    train = np.asarray(train_traj, dtype=np.float64)
    test  = np.asarray(test_traj,  dtype=np.float64)
    if len(train) < TRAIN_FIT_LEN + TEST_TRUTH_LEN \
       or len(test) < TRAIN_FIT_LEN + TEST_TRUTH_LEN:
        return None

    val_specs = []
    for vs in VAL_STARTS:
        fit_start = vs - VAL_FIT_LEN
        if fit_start < 0:
            continue
        fit_w = train[fit_start:vs]
        target = train[vs:vs + TEST_TRUTH_LEN]
        if len(fit_w) < VAL_FIT_LEN or len(target) < TEST_TRUTH_LEN:
            continue
        val_specs.append((fit_w, target))
    if not val_specs:
        return None

    val_results = []
    for regression, E, k in configs:
        sms = []
        for fit_w, target in val_specs:
            fc = _forecast_one(fit_w, E=E, k=k, regression=regression)
            if fc is None or not np.all(np.isfinite(fc)):
                sms = None
                break
            sms.append(smape(target, fc[:TEST_TRUTH_LEN]))
        if sms:
            val_results.append((float(np.mean(sms)), regression, E, k))
    if not val_results:
        return None
    val_results.sort(key=lambda x: x[0])
    val_sm, best_reg, best_E, best_k = val_results[0]

    test_fit = test[:TRAIN_FIT_LEN]
    test_truth = test[TRAIN_FIT_LEN:TRAIN_FIT_LEN + TEST_TRUTH_LEN]
    fc = _forecast_one(test_fit, E=best_E, k=best_k, regression=best_reg)
    if fc is None:
        return None
    pred = fc[:TEST_TRUTH_LEN]
    return dict(
        system=system,
        anda_pick=f'{best_reg}_E{best_E}_k{best_k}',
        anda_val_smape=val_sm,
        anda_smape=smape(test_truth, pred),
        anda_mse=mse(test_truth, pred),
        anda_mae=mae(test_truth, pred),
    )


def main():
    np.random.seed(0)
    print(f"=== AnDA on dysts univariate (multi-IC val) ===")
    print(f"Val starts: {VAL_STARTS}  fit_len={VAL_FIT_LEN}  "
          f"({len(VAL_STARTS)} windows averaged)")
    train_data = json.load(open(TRAIN_PATH))
    test_data  = json.load(open(TEST_PATH))
    systems = sorted(set(train_data) & set(test_data))
    print(f"Systems: {len(systems)}")

    REGRESSIONS_FULL = ('locally_constant', 'increment', 'local_linear')
    ES = (3, 5, 8)
    KS_PER_E = {3: (4, 10, 25, 50), 5: (6, 10, 25, 50), 8: (9, 10, 25, 50)}
    full_configs = [(r, E, k) for r in REGRESSIONS_FULL for E in ES
                    for k in KS_PER_E[E]]
    print(f"Configs per system: {len(full_configs)}")

    rows, skipped = [], []
    t0 = time.time()
    for i, sys_name in enumerate(systems):
        try:
            train_v = (train_data[sys_name].get('values')
                       if isinstance(train_data[sys_name], dict)
                       else train_data[sys_name])
            test_v  = (test_data[sys_name].get('values')
                       if isinstance(test_data[sys_name], dict)
                       else test_data[sys_name])
            r = run_one_system(sys_name, train_v, test_v, full_configs)
            if r is None:
                skipped.append(sys_name)
            else:
                rows.append(r)
        except Exception as e:
            skipped.append(f"{sys_name} ({e})")
        if (i + 1) % 25 == 0 or i == len(systems) - 1:
            print(f"  {i+1}/{len(systems)} systems  [{time.time()-t0:.1f}s]")
    print(f"\nTotal: {time.time() - t0:.1f}s, scored {len(rows)}, skipped {len(skipped)}")

    out = os.path.join(HERE, 'results', 'anda_dysts_full_multiIC3_fitlen90.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = ['system', 'anda_pick', 'anda_val_smape',
              'anda_smape', 'anda_mse', 'anda_mae']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}")

    sm = np.array([r['anda_smape'] for r in rows
                    if np.isfinite(r['anda_smape'])])
    print(f"\nAnDA (multi-IC val): median={np.median(sm):.2f}  mean={sm.mean():.2f}  "
          f"n={len(sm)}")


if __name__ == "__main__":
    main()
