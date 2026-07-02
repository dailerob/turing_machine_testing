"""GDC + Parrot on the dysts univariate forecasting benchmark.

Protocol (mirrors `dysts_data/benchmarks/classical_benchmarks.py`):
  - 131 chaotic systems, univariate view (first coordinate)
  - For each system:
      train / test trajectories (180 points each, independent ICs)
      val tuning: fit on train[:150], score 30-step forecast vs train[150:180]
      test eval:  fit on test[:150],  score 30-step forecast vs test[150:179]
  - The released baseline JSON's "values" field is test[150:179] (length 29)
    and predictions are length 30; sMAPE/MSE compute against pred[:29] vs truth.

Methods:
  - Parrot: top-K nearest-prefix lookup over the 150-point fit trajectory,
            raw + 1-step-diff variants, K ∈ {1, 5}, L ∈ {2, 4, 8, 16}.
  - GDC-TS: forecast_many_torch (terminal_behavior='absorb', initial_dist=
            'uniform' built in), 12-config grid: recipe ∈ {raw, diff} ×
            sigma_frac ∈ {0.05, 0.10, 0.25} × alpha ∈ {1.0, 0.99}.
  - Both pick best config per system on the train-trajectory val split.

Output: results/parrot_gdc_dysts.csv (per-system sMAPE / MSE / MAE for
        each method) + a side-by-side comparison vs the released JSON.
"""
from __future__ import annotations
import os, sys, json, csv, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'skolr_bench', 'forecast'))

from parrot_torch import forecast_many_parrot, forecast_many_parrot_diff
from gdc_torch import forecast_many_torch

DATA_DIR = os.path.join(HERE, 'data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.json')
TEST_PATH  = os.path.join(DATA_DIR, 'test.json')
BASELINES  = os.path.join(DATA_DIR, 'released_baselines.json')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

# Protocol constants
TRAIN_FIT_LEN  = 150  # 5/6 of 180
TEST_TRUTH_LEN = 29   # length of "values" field in released_baselines.json
PRED_LEN       = 30   # model.predict(30) = len(test_val)

# Variant grids (val-tuned per system)
PARROT_VARIANTS = []
for L in (2, 4, 8, 16):
    for k in (1, 5):
        PARROT_VARIANTS.append(('raw',  L, k))
        PARROT_VARIANTS.append(('diff', L, k))

GDC_CONFIGS = []
for recipe in ('raw', 'diff'):
    for sigma in (0.05, 0.10, 0.25):
        for alpha in (1.0, 0.99):
            GDC_CONFIGS.append((recipe, sigma, alpha))


def smape(truth, pred):
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    denom = (np.abs(truth) + np.abs(pred)) / 2.0
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return float(100.0 * np.mean(np.abs(truth - pred) / denom))


def mse(truth, pred):
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(np.mean((truth - pred) ** 2))


def mae(truth, pred):
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(np.mean(np.abs(truth - pred)))


# ---- Parrot wrapper for single-system forecast --------------------------
def parrot_forecast(state_series, prime, T, mode, L, k):
    """Single-system parrot forecast. state_series is the historical pool;
    prime is the lookback (last L+1 raw points if diff, last L if raw)."""
    state_series = np.asarray(state_series, dtype=np.float64)
    if mode == 'raw':
        if len(prime) < L:
            return np.full(T, prime[-1])
        primes = np.asarray(prime[-L:], dtype=np.float64).reshape(1, L)
        fc = forecast_many_parrot(state_series, primes, T, k=k,
                                   device=DEVICE, dtype=DTYPE)
    else:  # diff
        if len(prime) < L + 1:
            return np.full(T, prime[-1])
        primes_d = np.asarray(prime[-(L + 1):], dtype=np.float64).reshape(1, L + 1)
        fc = forecast_many_parrot_diff(state_series, primes_d, T, k=k,
                                        device=DEVICE, dtype=DTYPE)
    return fc.detach().cpu().numpy().ravel().astype(np.float64)


# ---- GDC wrapper for single-system forecast -----------------------------
def gdc_forecast(state_series, prime, T, recipe, sigma_frac, alpha):
    state_series = np.asarray(state_series, dtype=np.float64)
    prime = np.asarray(prime, dtype=np.float64)
    L = len(prime)
    if recipe == 'diff':
        if L < 2:
            return np.full(T, prime[-1])
        d_state = np.diff(state_series)
        d_prime = np.diff(prime)
        anchor = float(prime[-1])
        sigma = max(float(np.std(d_state)) * sigma_frac, 1e-9)
        beta = max((sigma * np.sqrt(len(d_prime))) ** 2, 1e-9)
        if len(d_state) < 3:
            return np.full(T, anchor)
        fc_d = forecast_many_torch(d_state, beta, alpha, 0.0,
                                    d_prime.reshape(1, -1), T,
                                    device=DEVICE, dtype=DTYPE)
        fc_d = fc_d.detach().cpu().numpy().ravel().astype(np.float64)
        return anchor + np.cumsum(fc_d)
    else:
        if len(state_series) < 3:
            return np.full(T, prime[-1])
        sigma = max(float(np.std(state_series)) * sigma_frac, 1e-9)
        beta = max((sigma * np.sqrt(L)) ** 2, 1e-9)
        fc = forecast_many_torch(state_series, beta, alpha, 0.0,
                                  prime.reshape(1, -1), T,
                                  device=DEVICE, dtype=DTYPE)
        return fc.detach().cpu().numpy().ravel().astype(np.float64)


# ---- Per-system runner ---------------------------------------------------
def run_one_system(system, train_traj, test_traj):
    """Returns dict with parrot/gdc test sMAPE, MSE, MAE + picked configs."""
    train = np.asarray(train_traj, dtype=np.float64)
    test  = np.asarray(test_traj,  dtype=np.float64)
    if len(train) < TRAIN_FIT_LEN + 1 or len(test) < TRAIN_FIT_LEN + TEST_TRUTH_LEN:
        return None

    # Val tuning split on TRAIN trajectory: train[:150] = "fit", train[150:180] = "val"
    train_fit = train[:TRAIN_FIT_LEN]
    train_val = train[TRAIN_FIT_LEN:TRAIN_FIT_LEN + PRED_LEN]
    val_smape_target = train_val[:TEST_TRUTH_LEN]  # match dysts truth length

    # ---- Parrot val sweep ----
    parrot_val = []
    for mode, L, k in PARROT_VARIANTS:
        try:
            fc = parrot_forecast(train_fit, train_fit, T=PRED_LEN,
                                  mode=mode, L=L, k=k)
            sm = smape(val_smape_target, fc[:TEST_TRUTH_LEN])
            parrot_val.append((sm, mode, L, k))
        except Exception:
            continue
    parrot_val.sort(key=lambda x: x[0])
    if not parrot_val:
        return None
    p_val_sm, p_mode, p_L, p_k = parrot_val[0]

    # ---- GDC val sweep ----
    gdc_val = []
    for recipe, sigma, alpha in GDC_CONFIGS:
        try:
            fc = gdc_forecast(train_fit, train_fit[-min(16, len(train_fit)):],
                               T=PRED_LEN, recipe=recipe,
                               sigma_frac=sigma, alpha=alpha)
            sm = smape(val_smape_target, fc[:TEST_TRUTH_LEN])
            gdc_val.append((sm, recipe, sigma, alpha))
        except Exception:
            continue
    gdc_val.sort(key=lambda x: x[0])
    if not gdc_val:
        return None
    g_val_sm, g_recipe, g_sigma, g_alpha = gdc_val[0]

    # ---- Test eval with val-picked configs on the TEST trajectory ----
    test_fit = test[:TRAIN_FIT_LEN]
    test_truth = test[TRAIN_FIT_LEN:TRAIN_FIT_LEN + TEST_TRUTH_LEN]

    p_fc = parrot_forecast(test_fit, test_fit, T=PRED_LEN,
                            mode=p_mode, L=p_L, k=p_k)[:TEST_TRUTH_LEN]
    g_fc = gdc_forecast(test_fit, test_fit[-min(16, len(test_fit)):],
                         T=PRED_LEN, recipe=g_recipe,
                         sigma_frac=g_sigma, alpha=g_alpha)[:TEST_TRUTH_LEN]

    return dict(
        system=system,
        parrot_smape=smape(test_truth, p_fc),
        parrot_mse=mse(test_truth, p_fc), parrot_mae=mae(test_truth, p_fc),
        parrot_pick=f'{p_mode}_L{p_L}_k{p_k}', parrot_val_smape=p_val_sm,
        gdc_smape=smape(test_truth, g_fc),
        gdc_mse=mse(test_truth, g_fc), gdc_mae=mae(test_truth, g_fc),
        gdc_pick=f'{g_recipe}_s{g_sigma}_a{g_alpha}', gdc_val_smape=g_val_sm,
    )


def main():
    print(f"=== dysts univariate (pts_per_period=15, periods=12) ===")
    print(f"Device: {DEVICE}  dtype={DTYPE}")
    train_data = json.load(open(TRAIN_PATH))
    test_data  = json.load(open(TEST_PATH))
    systems = sorted(set(train_data) & set(test_data))
    print(f"Systems: {len(systems)}")
    print(f"Parrot variants: {len(PARROT_VARIANTS)}, GDC configs: {len(GDC_CONFIGS)}")

    rows = []
    t0 = time.time()
    for i, sys_name in enumerate(systems):
        try:
            train_v = train_data[sys_name].get('values') if isinstance(train_data[sys_name], dict) else train_data[sys_name]
            test_v  = test_data[sys_name].get('values')  if isinstance(test_data[sys_name],  dict) else test_data[sys_name]
            r = run_one_system(sys_name, train_v, test_v)
            if r is not None:
                rows.append(r)
        except Exception as e:
            print(f"  [skip] {sys_name}: {e}")
        if (i + 1) % 20 == 0 or i == len(systems) - 1:
            print(f"  {i+1}/{len(systems)} systems  [{time.time()-t0:.1f}s]")
    print(f"Total: {time.time() - t0:.1f}s, {len(rows)} systems scored")

    out = os.path.join(HERE, 'results', 'parrot_gdc_dysts.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = ['system',
              'parrot_smape', 'parrot_mse', 'parrot_mae',
              'parrot_pick', 'parrot_val_smape',
              'gdc_smape', 'gdc_mse', 'gdc_mae',
              'gdc_pick', 'gdc_val_smape']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}\n")

    # ---- Aggregate medians for parrot vs GDC vs released baselines ----
    arr = lambda key: np.array([r[key] for r in rows
                                 if not np.isnan(r[key]) and np.isfinite(r[key])])
    print(f"=== Median sMAPE across {len(rows)} systems ===")
    p = arr('parrot_smape'); g = arr('gdc_smape')
    print(f"  Parrot (val-tuned)   median={np.median(p):.2f}  mean={p.mean():.2f}")
    print(f"  GDC (val-tuned)      median={np.median(g):.2f}  mean={g.mean():.2f}")

    # Compare to released baselines (uses identical test truth since we
    # match dysts protocol).
    if os.path.exists(BASELINES):
        bl = json.load(open(BASELINES))
        # Get list of models in baselines
        bl_models = set()
        for s, e in bl.items():
            for m, v in e.items():
                if isinstance(v, dict) and 'smape' in v:
                    bl_models.add(m)
        bl_models = sorted(bl_models)
        agg = []
        for m in bl_models:
            vals = []
            for r in rows:
                s = r['system']
                if s in bl and m in bl[s] and isinstance(bl[s][m], dict):
                    sv = bl[s][m].get('smape')
                    if sv is not None and sv == sv:
                        vals.append(float(sv))
            if vals:
                agg.append((m, float(np.median(vals)), float(np.mean(vals)), len(vals)))
        # Add our methods
        agg.append(('Parrot', float(np.median(p)), float(p.mean()), len(p)))
        agg.append(('GDC',    float(np.median(g)), float(g.mean()), len(g)))
        agg.sort(key=lambda x: x[1])
        print()
        print(f"=== Median sMAPE leaderboard (lower is better) ===")
        print(f"{'rank':>4s}  {'method':>22s}  {'median':>8s}  {'mean':>8s}  {'n':>5s}")
        for i, (m, med, mn, n) in enumerate(agg):
            tag = ' ← ours' if m in ('Parrot', 'GDC') else ''
            print(f"{i+1:>4d}  {m:>22s}  {med:>8.2f}  {mn:>8.2f}  {n:>5d}{tag}")


if __name__ == "__main__":
    main()
