"""pyEDM (Sugihara lab) on the dysts univariate forecasting benchmark.

Matches the protocol used in `run_parrot_gdc.py` exactly so the pyEDM
numbers slot into the same leaderboard:
  - 131 chaotic systems, univariate view
  - val tune on train trajectory (fit on train[:150], score 30-step rollout
    vs train[150:179]); pick best config per system
  - test eval: fit on test[:150], score 30-step rollout vs test[150:179]
  - sMAPE on the first 29 of 30 generated predictions, per the released
    JSON convention

Methods (val-picked per system):
  - Simplex (E ∈ {3, 5, 8})
  - SMap    (E ∈ {3, 5} × theta ∈ {2, 5, 10})

The smoke test showed Simplex is flat in E on Lorenz (the canonical
EDM target), so we keep the grid small. SMap needs theta≥2 to be
competitive on chaotic data; we keep three values.

Output: results/pyedm_dysts.csv (per-system) + side-by-side comparison
to the existing parrot/GDC leaderboard.
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
BASELINES  = os.path.join(DATA_DIR, 'released_baselines.json')

# Protocol constants (mirror run_parrot_gdc.py)
TRAIN_FIT_LEN  = 150  # 5/6 of 180
TEST_TRUTH_LEN = 29   # length of "values" field in released_baselines.json
PRED_LEN       = 30   # generateSteps

# Variant grid (val-tuned per system)
SIMPLEX_ES = (3, 5, 8)
SMAP_ES    = (3, 5)
SMAP_THETAS = (2.0, 5.0, 10.0)


def smape(t, p):
    t = np.asarray(t, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
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
    """pyEDM input: DataFrame with a time column + the data column."""
    return pd.DataFrame({
        'time': np.arange(len(values)),
        'x': np.asarray(values, dtype=np.float64),
    })


def _simplex_forecast(df_lib, E, n_steps=PRED_LEN):
    """Run Simplex with generative rollout; return n_steps predictions or
    None on failure."""
    try:
        out = pyEDM.Simplex(
            dataFrame=df_lib, columns='x', target='x',
            lib=f'1 {len(df_lib)}', pred=f'{len(df_lib)-1} {len(df_lib)}',
            E=E, Tp=1, tau=-1,
            generateSteps=n_steps, showPlot=False,
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
        # SMap returns dict; the predictions DataFrame is at key 'predictions'
        pred = np.asarray(out['predictions']['Predictions'].values,
                          dtype=np.float64)
        if pred.size < n_steps or not np.all(np.isfinite(pred[:n_steps])):
            return None
        return pred[:n_steps]
    except Exception:
        return None


# ---- Diff-mode wrappers: differentiate, forecast, cumsum back ----------------
def _simplex_diff_forecast(df_lib_raw, E, n_steps=PRED_LEN):
    """Run Simplex on the differenced series, cumsum 30 predicted diffs onto
    the last raw value to obtain raw-space forecasts."""
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
    """Yield (label, fn) where fn(df_lib) -> 30-step prediction array (raw space)."""
    # Raw-mode: classic pyEDM on the trajectory directly
    for E in SIMPLEX_ES:
        yield (f'simplex_raw_E{E}',
               lambda df, E=E: _simplex_forecast(df, E))
    for E in SMAP_ES:
        for th in SMAP_THETAS:
            yield (f'smap_raw_E{E}_t{th}',
                   lambda df, E=E, th=th: _smap_forecast(df, E, th))
    # Diff-mode: pyEDM on np.diff of trajectory, cumsum back onto last raw value
    for E in SIMPLEX_ES:
        yield (f'simplex_diff_E{E}',
               lambda df, E=E: _simplex_diff_forecast(df, E))
    for E in SMAP_ES:
        for th in SMAP_THETAS:
            yield (f'smap_diff_E{E}_t{th}',
                   lambda df, E=E, th=th: _smap_diff_forecast(df, E, th))


def run_one_system(system, train_traj, test_traj):
    """Returns dict with pyEDM test sMAPE/MSE/MAE + picked config, or None."""
    train = np.asarray(train_traj, dtype=np.float64)
    test  = np.asarray(test_traj,  dtype=np.float64)
    if len(train) < TRAIN_FIT_LEN + TEST_TRUTH_LEN \
       or len(test) < TRAIN_FIT_LEN + TEST_TRUTH_LEN:
        return None

    # ---- Val tuning on train trajectory ----
    train_fit = train[:TRAIN_FIT_LEN]
    val_truth = train[TRAIN_FIT_LEN:TRAIN_FIT_LEN + TEST_TRUTH_LEN]
    df_train = _make_df(train_fit)
    val_results = []
    for label, fn in _all_configs():
        fc = fn(df_train)
        if fc is None:
            continue
        val_results.append((smape(val_truth, fc[:TEST_TRUTH_LEN]), label, fn))
    if not val_results:
        return None
    val_results.sort(key=lambda x: x[0])
    val_sm, pick, fn = val_results[0]

    # ---- Test eval with val-picked config ----
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
    print(f"=== pyEDM on dysts univariate (pts_per_period=15, periods=12) ===")
    print(f"Configs per system: {sum(1 for _ in _all_configs())} "
          f"(Simplex × {len(SIMPLEX_ES)} + SMap × {len(SMAP_ES)*len(SMAP_THETAS)})")
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
    print(f"\nTotal: {time.time() - t0:.1f}s, scored {len(rows)} systems, "
          f"skipped {len(skipped)}")
    if skipped:
        print(f"  skipped: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    out = os.path.join(HERE, 'results', 'pyedm_dysts.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = ['system', 'pyedm_pick', 'pyedm_val_smape',
              'pyedm_smape', 'pyedm_mse', 'pyedm_mae']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}\n")

    # ---- Side-by-side leaderboard ----
    smapes = np.array([r['pyedm_smape'] for r in rows
                       if np.isfinite(r['pyedm_smape'])])
    print(f"=== pyEDM aggregate ({len(smapes)} systems scored) ===")
    print(f"  median sMAPE: {np.median(smapes):.2f}")
    print(f"  mean sMAPE:   {smapes.mean():.2f}")

    if os.path.exists(BASELINES):
        bl = json.load(open(BASELINES))
        # Released baselines
        agg = []
        bl_models = sorted({m for s, e in bl.items()
                            for m, v in e.items()
                            if isinstance(v, dict) and 'smape' in v})
        for m in bl_models:
            vals = [float(bl[s][m]['smape'])
                    for s in {r['system'] for r in rows}
                    if s in bl and m in bl[s]
                    and isinstance(bl[s][m], dict)
                    and bl[s][m].get('smape') is not None
                    and bl[s][m]['smape'] == bl[s][m]['smape']]
            if vals:
                agg.append((m, float(np.median(vals)), float(np.mean(vals)), len(vals)))
        # Add our methods (read parrot_gdc CSV if present)
        gdc_csv = os.path.join(HERE, 'results', 'parrot_gdc_dysts.csv')
        if os.path.exists(gdc_csv):
            df_pg = pd.read_csv(gdc_csv)
            agg.append(('Parrot', float(df_pg.parrot_smape.median()),
                        float(df_pg.parrot_smape.mean()), len(df_pg)))
            agg.append(('GDC',    float(df_pg.gdc_smape.median()),
                        float(df_pg.gdc_smape.mean()), len(df_pg)))
        agg.append(('pyEDM (val-tuned)', float(np.median(smapes)),
                    float(smapes.mean()), len(smapes)))
        agg.sort(key=lambda x: x[1])
        print(f"\n=== Leaderboard with pyEDM ===")
        print(f"{'rank':>4s}  {'method':>22s}  {'median':>8s}  {'mean':>8s}  {'n':>5s}")
        for i, (m, med, mn, n) in enumerate(agg):
            tag = ' ← ours' if m in ('Parrot', 'GDC', 'pyEDM (val-tuned)') else ''
            print(f"{i+1:>4d}  {m:>22s}  {med:>8.2f}  {mn:>8.2f}  {n:>5d}{tag}")


if __name__ == "__main__":
    main()
