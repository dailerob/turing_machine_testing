"""pyEDM on dysts univariate with fixed-fit-len multi-IC val.

Same config grid as run_pyedm.py (Simplex × 3 E + SMap × 2 E × 3 θ +
their diff variants) but val sMAPE is averaged across 3 sliding val
windows within the train trajectory, all at fixed fit_len=90.

Output: results/pyedm_dysts_multiIC3_fitlen90.csv
"""
from __future__ import annotations
import os, sys, json, csv, time
import numpy as np
import pandas as pd
import pyEDM

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

DATA_DIR   = os.path.join(HERE, 'data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.json')
TEST_PATH  = os.path.join(DATA_DIR, 'test.json')

TRAIN_FIT_LEN  = 150
TEST_TRUTH_LEN = 29
PRED_LEN       = 30

VAL_FIT_LEN = 90
VAL_STARTS  = [90, 120, 150]

SIMPLEX_ES  = (3, 5, 8)
SMAP_ES     = (3, 5)
SMAP_THETAS = (2.0, 5.0, 10.0)


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


def _make_df(values):
    return pd.DataFrame({'time': np.arange(len(values)),
                         'x': np.asarray(values, dtype=np.float64)})


def _simplex_forecast(df_lib, E, n_steps=PRED_LEN):
    try:
        out = pyEDM.Simplex(
            dataFrame=df_lib, columns='x', target='x',
            lib=f'1 {len(df_lib)}', pred=f'{len(df_lib)-1} {len(df_lib)}',
            E=E, Tp=1, tau=-1, generateSteps=n_steps, showPlot=False,
        )
        pred = np.asarray(out['Predictions'].values, dtype=np.float64)
        if pred.size < n_steps or not np.all(np.isfinite(pred[:n_steps])):
            return None
        return pred[:n_steps]
    except Exception:
        return None


def _smap_forecast(df_lib, E, theta, n_steps=PRED_LEN):
    try:
        out = pyEDM.SMap(
            dataFrame=df_lib, columns='x', target='x',
            lib=f'1 {len(df_lib)}', pred=f'{len(df_lib)-1} {len(df_lib)}',
            E=E, Tp=1, tau=-1, theta=theta,
            generateSteps=n_steps, showPlot=False,
        )
        pred = np.asarray(out['predictions']['Predictions'].values,
                          dtype=np.float64)
        if pred.size < n_steps or not np.all(np.isfinite(pred[:n_steps])):
            return None
        return pred[:n_steps]
    except Exception:
        return None


def _simplex_diff_forecast(df_lib_raw, E, n_steps=PRED_LEN):
    raw = df_lib_raw['x'].values
    if len(raw) < 2:
        return None
    diffs = np.diff(raw)
    df_diff = pd.DataFrame({'time': np.arange(len(diffs)), 'x': diffs})
    pred_diffs = _simplex_forecast(df_diff, E, n_steps=n_steps)
    if pred_diffs is None:
        return None
    return raw[-1] + np.cumsum(pred_diffs)


def _smap_diff_forecast(df_lib_raw, E, theta, n_steps=PRED_LEN):
    raw = df_lib_raw['x'].values
    if len(raw) < 2:
        return None
    diffs = np.diff(raw)
    df_diff = pd.DataFrame({'time': np.arange(len(diffs)), 'x': diffs})
    pred_diffs = _smap_forecast(df_diff, E, theta, n_steps=n_steps)
    if pred_diffs is None:
        return None
    return raw[-1] + np.cumsum(pred_diffs)


def _all_configs():
    for E in SIMPLEX_ES:
        yield (f'simplex_raw_E{E}',
               lambda df, E=E: _simplex_forecast(df, E))
    for E in SMAP_ES:
        for th in SMAP_THETAS:
            yield (f'smap_raw_E{E}_t{th}',
                   lambda df, E=E, th=th: _smap_forecast(df, E, th))
    for E in SIMPLEX_ES:
        yield (f'simplex_diff_E{E}',
               lambda df, E=E: _simplex_diff_forecast(df, E))
    for E in SMAP_ES:
        for th in SMAP_THETAS:
            yield (f'smap_diff_E{E}_t{th}',
                   lambda df, E=E, th=th: _smap_diff_forecast(df, E, th))


def run_one_system(system, train_traj, test_traj):
    train = np.asarray(train_traj, dtype=np.float64)
    test  = np.asarray(test_traj,  dtype=np.float64)
    if len(train) < TRAIN_FIT_LEN + TEST_TRUTH_LEN \
       or len(test) < TRAIN_FIT_LEN + TEST_TRUTH_LEN:
        return None

    # Build val specs: fixed fit_len=90, 3 sliding val starts.
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

    configs = list(_all_configs())
    val_results = []
    for label, fn in configs:
        sms = []
        for fit_w, target in val_specs:
            df_w = _make_df(fit_w)
            fc = fn(df_w)
            if fc is None or not np.all(np.isfinite(fc)):
                sms = None
                break
            sms.append(smape(target, fc[:TEST_TRUTH_LEN]))
        if sms:
            val_results.append((float(np.mean(sms)), label, fn))
    if not val_results:
        return None
    val_results.sort(key=lambda x: x[0])
    val_sm, pick, fn = val_results[0]

    test_fit = test[:TRAIN_FIT_LEN]
    test_truth = test[TRAIN_FIT_LEN:TRAIN_FIT_LEN + TEST_TRUTH_LEN]
    df_test = _make_df(test_fit)
    fc = fn(df_test)
    if fc is None:
        return None
    pred = fc[:TEST_TRUTH_LEN]
    return dict(
        system=system, pyedm_pick=pick, pyedm_val_smape=val_sm,
        pyedm_smape=smape(test_truth, pred),
        pyedm_mse=mse(test_truth, pred),
        pyedm_mae=mae(test_truth, pred),
    )


def main():
    print(f"=== pyEDM on dysts univariate (multi-IC val) ===")
    print(f"Val starts: {VAL_STARTS}  fit_len={VAL_FIT_LEN}  "
          f"({len(VAL_STARTS)} windows averaged)")
    configs = list(_all_configs())
    print(f"Configs per system: {len(configs)}")
    train_data = json.load(open(TRAIN_PATH))
    test_data  = json.load(open(TEST_PATH))
    systems = sorted(set(train_data) & set(test_data))
    print(f"Systems: {len(systems)}\n")

    rows = []
    skipped = []
    t0 = time.time()
    for i, sys_name in enumerate(systems):
        try:
            train_v = (train_data[sys_name].get('values')
                       if isinstance(train_data[sys_name], dict)
                       else train_data[sys_name])
            test_v  = (test_data[sys_name].get('values')
                       if isinstance(test_data[sys_name], dict)
                       else test_data[sys_name])
            r = run_one_system(sys_name, train_v, test_v)
            if r is None:
                skipped.append(sys_name)
            else:
                rows.append(r)
        except Exception as e:
            skipped.append(f"{sys_name} ({e})")
        if (i + 1) % 25 == 0 or i == len(systems) - 1:
            print(f"  {i+1}/{len(systems)} systems  [{time.time()-t0:.1f}s]")
    print(f"\nTotal: {time.time() - t0:.1f}s, scored {len(rows)}, skipped {len(skipped)}")

    out = os.path.join(HERE, 'results', 'pyedm_dysts_multiIC3_fitlen90.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = ['system', 'pyedm_pick', 'pyedm_val_smape',
              'pyedm_smape', 'pyedm_mse', 'pyedm_mae']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}")

    sm = np.array([r['pyedm_smape'] for r in rows
                    if np.isfinite(r['pyedm_smape'])])
    print(f"\npyEDM (multi-IC val): median={np.median(sm):.2f}  mean={sm.mean():.2f}  "
          f"n={len(sm)}")


if __name__ == "__main__":
    main()
