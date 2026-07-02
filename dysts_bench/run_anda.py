"""AnDA (Lguensat et al., 2017) on the dysts univariate forecasting
benchmark. Mirrors run_pyedm.py's val-tuning protocol.

Two sweeps (run sequentially to allow steelman comparison):
  - "full":  regression ∈ {locally_constant, increment, local_linear}
  - "safe":  regression ∈ {locally_constant, increment}            (no local_linear)

Per system:
  val tuning: build catalog from train[:150], 30-step generative rollout
              vs train[150:179]; pick best config (regression, E, k)
  test eval:  build catalog from test[:150], score vs test[150:179]
  sMAPE on first 29 of 30 generated predictions

Outputs:
  results/anda_dysts_full.csv
  results/anda_dysts_safe.csv
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
BASELINES  = os.path.join(DATA_DIR, 'released_baselines.json')

# Protocol constants
TRAIN_FIT_LEN  = 150
TEST_TRUTH_LEN = 29
PRED_LEN       = 30


# ----- AnDA wrappers --------------------------------------------------------
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
    """Takens-embedded (analog, successor) pairs.
    analog[i]    = series[i:i+E]
    successor[i] = series[i+1:i+E+1]
    """
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
    """Autoregressive 30-step AnDA forecast on a 1-D fit series.
    Returns (n_steps,) array of predicted raw values, or None on error."""
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


# ----- Per-system runner ----------------------------------------------------
def run_one_system(system, train_traj, test_traj, configs):
    train = np.asarray(train_traj, dtype=np.float64)
    test = np.asarray(test_traj, dtype=np.float64)
    if len(train) < TRAIN_FIT_LEN + TEST_TRUTH_LEN \
       or len(test) < TRAIN_FIT_LEN + TEST_TRUTH_LEN:
        return None
    train_fit = train[:TRAIN_FIT_LEN]
    val_truth = train[TRAIN_FIT_LEN:TRAIN_FIT_LEN + TEST_TRUTH_LEN]

    val_results = []
    for regression, E, k in configs:
        fc = _forecast_one(train_fit, E=E, k=k, regression=regression)
        if fc is None or not np.all(np.isfinite(fc)):
            continue
        sm = smape(val_truth, fc[:TEST_TRUTH_LEN])
        val_results.append((sm, regression, E, k))
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


def run_sweep(label, configs, train_data, test_data, systems):
    print(f"\n=== AnDA sweep: {label} ({len(configs)} configs / system) ===")
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
            r = run_one_system(sys_name, train_v, test_v, configs)
            if r is None:
                skipped.append(sys_name)
            else:
                rows.append(r)
        except Exception as e:
            skipped.append(f"{sys_name} ({e})")
        if (i + 1) % 25 == 0 or i == len(systems) - 1:
            print(f"  {i+1}/{len(systems)} systems  [{time.time()-t0:.1f}s]")
    print(f"  Total: {time.time() - t0:.1f}s, scored {len(rows)}, skipped {len(skipped)}")
    if skipped:
        print(f"  skipped: {skipped[:6]}{'...' if len(skipped) > 6 else ''}")
    return rows


def main():
    np.random.seed(0)
    print(f"=== AnDA on dysts univariate (steelman: full + safe sweeps) ===")
    train_data = json.load(open(TRAIN_PATH))
    test_data  = json.load(open(TEST_PATH))
    systems = sorted(set(train_data) & set(test_data))
    print(f"Systems: {len(systems)}")

    REGRESSIONS_FULL = ('locally_constant', 'increment', 'local_linear')
    REGRESSIONS_SAFE = ('locally_constant', 'increment')
    ES = (3, 5, 8)
    KS_PER_E = {3: (4, 10, 25, 50), 5: (6, 10, 25, 50), 8: (9, 10, 25, 50)}

    def build_configs(regressions):
        return [(r, E, k) for r in regressions for E in ES for k in KS_PER_E[E]]

    full_configs = build_configs(REGRESSIONS_FULL)
    safe_configs = build_configs(REGRESSIONS_SAFE)

    rows_full = run_sweep('full', full_configs, train_data, test_data, systems)
    rows_safe = run_sweep('safe', safe_configs, train_data, test_data, systems)

    # Save CSVs
    for label, rows in (('full', rows_full), ('safe', rows_safe)):
        out = os.path.join(HERE, 'results', f'anda_dysts_{label}.csv')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['system', 'anda_pick',
                                              'anda_val_smape', 'anda_smape',
                                              'anda_mse', 'anda_mae'])
            w.writeheader(); w.writerows(rows)
        print(f"  wrote {out}")

    # Aggregate side-by-side on common system set
    print(f"\n=== AnDA aggregate: full vs safe ===")
    for label, rows in (('full', rows_full), ('safe', rows_safe)):
        smapes = np.array([r['anda_smape'] for r in rows
                           if np.isfinite(r['anda_smape'])])
        if len(smapes):
            print(f"  {label:>5s}  n={len(smapes):>3d}  median={np.median(smapes):.2f}  "
                  f"mean={smapes.mean():.2f}")

    # Steelman: pick whichever has lower median, report on the 130-system common set
    bl = json.load(open(BASELINES)) if os.path.exists(BASELINES) else None
    df_paths = [os.path.join(HERE, 'results', 'parrot_gdc_dysts.csv'),
                os.path.join(HERE, 'results', 'pyedm_dysts.csv')]
    import pandas as pd
    pg = pd.read_csv(df_paths[0]) if os.path.exists(df_paths[0]) else None
    pe = pd.read_csv(df_paths[1]) if os.path.exists(df_paths[1]) else None

    def common_aggregate(rows, common):
        sm = np.array([r['anda_smape'] for r in rows
                        if r['system'] in common
                        and np.isfinite(r['anda_smape'])])
        return float(np.median(sm)), float(sm.mean()), len(sm)

    if bl is not None and pg is not None and pe is not None:
        common = set(bl.keys()) & set(pg.system) & set(pe.system) & \
                 {r['system'] for r in rows_full if np.isfinite(r['anda_smape'])} & \
                 {r['system'] for r in rows_safe if np.isfinite(r['anda_smape'])}
        print(f"\n=== 130+ system common-set leaderboard ===")
        print(f"common set size: {len(common)}\n")
        full_med, full_mn, _ = common_aggregate(rows_full, common)
        safe_med, safe_mn, _ = common_aggregate(rows_safe, common)
        print(f"  AnDA full   (val-tuned over locally_constant + increment + local_linear):"
              f"  median {full_med:.2f}  mean {full_mn:.2f}")
        print(f"  AnDA safe   (val-tuned over locally_constant + increment only):"
              f"           median {safe_med:.2f}  mean {safe_mn:.2f}")
        # Steelman pick
        if safe_med < full_med:
            print(f"\n→ Steelman AnDA result: SAFE  (median {safe_med:.2f} < full {full_med:.2f})")
        else:
            print(f"\n→ Steelman AnDA result: FULL  (median {full_med:.2f} <= safe {safe_med:.2f})")


if __name__ == "__main__":
    main()
