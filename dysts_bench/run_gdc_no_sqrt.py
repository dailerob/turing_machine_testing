"""dysts univariate forecasting WITHOUT the sqrt(window_len) scaling
on the GDC's emission variance.

Mirrors run_parrot_gdc.py for the GDC half only, with one change:
    beta = sigma ** 2     (no * np.sqrt(L) factor)

To give the val-tuner a fair chance, the sigma_frac grid is expanded
to span the equivalent range that the original grid covers when the
sqrt(L) factor is included. With typical L≈16, the original grid
{0.05, 0.10, 0.25} corresponds to effective sigma {0.20, 0.40, 1.00}
in the no-sqrt formulation. So the expanded grid is:
    sigma_frac ∈ {0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 4.00}.

Outputs: results/gdc_no_sqrt_dysts.csv  (per-system test sMAPE/MSE/MAE
         + picked configs + val sMAPE)
"""
from __future__ import annotations
import os, sys, json, csv, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'skolr_bench', 'forecast'))

from gdc_torch import forecast_many_torch

DATA_DIR = os.path.join(HERE, 'data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.json')
TEST_PATH  = os.path.join(DATA_DIR, 'test.json')
BASELINES  = os.path.join(DATA_DIR, 'released_baselines.json')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

TRAIN_FIT_LEN  = 150
TEST_TRUTH_LEN = 29
PRED_LEN       = 30

# Expanded sigma grid to cover the range that sqrt(L) scaling
# implicitly spans
SIGMAS = (0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 4.00)
ALPHAS = (1.0, 0.99)
RECIPES = ('raw', 'diff')
GDC_CONFIGS = [(r, s, a) for r in RECIPES for s in SIGMAS for a in ALPHAS]


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


def gdc_forecast_no_sqrt(state_series, prime, T, recipe, sigma_frac,
                          alpha):
    """GDC-TS forecast WITHOUT sqrt(L) on the emission variance."""
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
        # ── No sqrt scaling: beta = sigma², not sigma² * len(d_prime)
        beta = max(sigma ** 2, 1e-9)
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
        # ── No sqrt scaling: beta = sigma², not sigma² * L
        beta = max(sigma ** 2, 1e-9)
        fc = forecast_many_torch(state_series, beta, alpha, 0.0,
                                  prime.reshape(1, -1), T,
                                  device=DEVICE, dtype=DTYPE)
        return fc.detach().cpu().numpy().ravel().astype(np.float64)


def run_one_system(system, train_traj, test_traj):
    train = np.asarray(train_traj, dtype=np.float64)
    test  = np.asarray(test_traj,  dtype=np.float64)
    if (len(train) < TRAIN_FIT_LEN + 1
            or len(test) < TRAIN_FIT_LEN + TEST_TRUTH_LEN):
        return None
    train_fit = train[:TRAIN_FIT_LEN]
    train_val = train[TRAIN_FIT_LEN:TRAIN_FIT_LEN + PRED_LEN]
    val_smape_target = train_val[:TEST_TRUTH_LEN]

    gdc_val = []
    for recipe, sigma, alpha in GDC_CONFIGS:
        try:
            fc = gdc_forecast_no_sqrt(
                train_fit, train_fit[-min(16, len(train_fit)):],
                T=PRED_LEN, recipe=recipe, sigma_frac=sigma, alpha=alpha)
            sm = smape(val_smape_target, fc[:TEST_TRUTH_LEN])
            gdc_val.append((sm, recipe, sigma, alpha))
        except Exception:
            continue
    if not gdc_val:
        return None
    gdc_val.sort(key=lambda x: x[0])
    g_val_sm, g_recipe, g_sigma, g_alpha = gdc_val[0]

    test_fit = test[:TRAIN_FIT_LEN]
    test_truth = test[TRAIN_FIT_LEN:TRAIN_FIT_LEN + TEST_TRUTH_LEN]
    g_fc = gdc_forecast_no_sqrt(
        test_fit, test_fit[-min(16, len(test_fit)):],
        T=PRED_LEN, recipe=g_recipe, sigma_frac=g_sigma,
        alpha=g_alpha)[:TEST_TRUTH_LEN]

    return dict(
        system=system,
        gdc_smape=smape(test_truth, g_fc),
        gdc_mse=mse(test_truth, g_fc),
        gdc_mae=mae(test_truth, g_fc),
        gdc_pick=f'{g_recipe}_s{g_sigma}_a{g_alpha}',
        gdc_val_smape=g_val_sm)


def main():
    print(f"=== dysts univariate, GDC NO-SQRT(L) variant ===")
    print(f"Device: {DEVICE}  dtype={DTYPE}")
    print(f"GDC configs (no sqrt(L)): {len(GDC_CONFIGS)} "
          f"(sigma grid: {SIGMAS})\n")
    train_data = json.load(open(TRAIN_PATH))
    test_data  = json.load(open(TEST_PATH))
    systems = sorted(set(train_data) & set(test_data))

    rows = []
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
            if r is not None:
                rows.append(r)
        except Exception as e:
            print(f"  [skip] {sys_name}: {e}")
        if (i + 1) % 20 == 0 or i == len(systems) - 1:
            print(f"  {i+1}/{len(systems)} systems  "
                  f"[{time.time()-t0:.1f}s]", flush=True)
    print(f"Total: {time.time() - t0:.1f}s, {len(rows)} systems scored\n")

    out = os.path.join(HERE, 'results', 'gdc_no_sqrt_dysts.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = ['system', 'gdc_smape', 'gdc_mse', 'gdc_mae',
              'gdc_pick', 'gdc_val_smape']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}\n")

    g_smape = np.array([r['gdc_smape'] for r in rows
                         if np.isfinite(r['gdc_smape'])])
    print(f"=== GDC (no sqrt(L) scaling), median sMAPE on "
          f"{len(g_smape)} systems ===")
    print(f"  median = {np.median(g_smape):.2f}")
    print(f"  mean   = {g_smape.mean():.2f}\n")

    # Compare to original GDC numbers (run_parrot_gdc.py results)
    orig_path = os.path.join(HERE, 'results', 'parrot_gdc_dysts.csv')
    if os.path.exists(orig_path):
        import pandas as pd
        orig = pd.read_csv(orig_path)
        # Restrict to common systems
        common = set(orig.system) & set(r['system'] for r in rows)
        sub_orig = orig[orig.system.isin(common)]
        sub_new = [r for r in rows if r['system'] in common]
        print(f"=== Side-by-side on {len(common)} common systems ===")
        print(f"  GDC orig (sqrt scaling)  median sMAPE = "
              f"{np.median(sub_orig.gdc_smape):.2f}")
        print(f"  GDC no-sqrt              median sMAPE = "
              f"{np.median([r['gdc_smape'] for r in sub_new]):.2f}")

    # Pick distribution
    from collections import Counter
    pick_counts = Counter(r['gdc_pick'].split('_s')[0] + '_s' + r['gdc_pick'].split('_s')[1].split('_')[0]
                          for r in rows)
    print(f"\n  Pick distribution (recipe + sigma_frac):")
    for k, v in sorted(pick_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"    {k:<14} {v} systems")


if __name__ == "__main__":
    main()
