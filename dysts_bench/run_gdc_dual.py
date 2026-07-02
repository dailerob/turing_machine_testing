"""GDC on dysts univariate with the dual-α grid added.

Same protocol as `run_parrot_gdc.py` for the GDC-only sweep:
  - 131 chaotic systems, univariate (first coord), 180 train / 180 test
  - Val tuning on train[:150] -> train[150:180], score sMAPE vs first 29 of
    the 30-step forecast (matches dysts truth length).
  - Test eval: refit on test[:150], score 30-step forecast vs test[150:179].

GDC config grid (extended with dual-α):
  recipe ∈ {raw, diff}, σ ∈ {0.05, 0.10, 0.25}.
  Single-α (α_ctx == α_fc): α ∈ {1.0, 0.99}     -> 12 configs
  Dual-α  (α_fc = 1.0)    : α_ctx ∈ {0.8, 0.9, 0.95, 0.99} -> 24 configs
                                                       Total: 36 configs.

Output: results/gdc_dual_dysts.csv plus a side-by-side comparison to the
existing parrot_gdc_dysts.csv numbers AND to the released_baselines.json
leaderboard.
"""
from __future__ import annotations
import os, sys, json, csv, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'skolr_bench', 'forecast'))

from gdc_torch import forecast_many_torch, forecast_many_torch_dual

DATA_DIR  = os.path.join(HERE, 'data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.json')
TEST_PATH  = os.path.join(DATA_DIR, 'test.json')
BASELINES  = os.path.join(DATA_DIR, 'released_baselines.json')
EXISTING_GDC_CSV = os.path.join(HERE, 'results', 'parrot_gdc_dysts.csv')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE  = torch.float32

TRAIN_FIT_LEN  = 150
TEST_TRUTH_LEN = 29
PRED_LEN       = 30

# Multi-IC val: per system, score each config on K val windows within the
# train trajectory at FIXED fit_len (removes the variable-fit-length
# confound). For each val_start: fit = train[val_start - VAL_FIT_LEN :
# val_start], val target = train[val_start : val_start + TEST_TRUTH_LEN].
# Each val window uses the same 90-point fit window length so the state-
# space size driving GDC's β = (σ √L)² normalization is identical across
# windows. Three non-overlapping val targets span the trajectory.
VAL_FIT_LEN = 90
VAL_STARTS  = [90, 120, 150]

# Single-α: (α_ctx, α_fc) with equal values.
SINGLE_ALPHAS = [(1.0, 1.0), (0.99, 0.99)]
# Dual-α: α_ctx < 1, α_fc = 1.0.
DUAL_ALPHAS   = [(0.8, 1.0), (0.9, 1.0), (0.95, 1.0), (0.99, 1.0)]
ALPHA_CONFIGS = SINGLE_ALPHAS + DUAL_ALPHAS

GDC_CONFIGS = []
for recipe in ('raw', 'diff'):
    for sigma in (0.05, 0.10, 0.25):
        for ac, afc in ALPHA_CONFIGS:
            GDC_CONFIGS.append((recipe, sigma, ac, afc))


def smape(truth, pred):
    truth = np.asarray(truth, dtype=np.float64)
    pred  = np.asarray(pred,  dtype=np.float64)
    denom = (np.abs(truth) + np.abs(pred)) / 2.0
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return float(100.0 * np.mean(np.abs(truth - pred) / denom))


def mse(truth, pred):
    truth = np.asarray(truth, dtype=np.float64)
    pred  = np.asarray(pred,  dtype=np.float64)
    return float(np.mean((truth - pred) ** 2))


def mae(truth, pred):
    truth = np.asarray(truth, dtype=np.float64)
    pred  = np.asarray(pred,  dtype=np.float64)
    return float(np.mean(np.abs(truth - pred)))


def gdc_forecast(state_series, prime, T, recipe, sigma_frac,
                  alpha_ctx, alpha_fc):
    """Mirrors run_parrot_gdc.gdc_forecast but supports dual-α.

    Falls through to forecast_many_torch when α_ctx == α_fc (this is the
    canonical single-α path); otherwise calls forecast_many_torch_dual.
    """
    state_series = np.asarray(state_series, dtype=np.float64)
    prime = np.asarray(prime, dtype=np.float64)
    L = len(prime)
    same_alpha = alpha_ctx == alpha_fc
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
        if same_alpha:
            fc_d = forecast_many_torch(d_state, beta, alpha_ctx, 0.0,
                                        d_prime.reshape(1, -1), T,
                                        device=DEVICE, dtype=DTYPE)
        else:
            fc_d = forecast_many_torch_dual(d_state, beta, alpha_ctx, 0.0,
                                             alpha_fc, 0.0,
                                             d_prime.reshape(1, -1), T,
                                             device=DEVICE, dtype=DTYPE)
        fc_d = fc_d.detach().cpu().numpy().ravel().astype(np.float64)
        return anchor + np.cumsum(fc_d)
    else:  # raw
        if len(state_series) < 3:
            return np.full(T, prime[-1])
        sigma = max(float(np.std(state_series)) * sigma_frac, 1e-9)
        beta = max((sigma * np.sqrt(L)) ** 2, 1e-9)
        if same_alpha:
            fc = forecast_many_torch(state_series, beta, alpha_ctx, 0.0,
                                      prime.reshape(1, -1), T,
                                      device=DEVICE, dtype=DTYPE)
        else:
            fc = forecast_many_torch_dual(state_series, beta, alpha_ctx, 0.0,
                                           alpha_fc, 0.0,
                                           prime.reshape(1, -1), T,
                                           device=DEVICE, dtype=DTYPE)
        return fc.detach().cpu().numpy().ravel().astype(np.float64)


def run_one_system(system, train_traj, test_traj):
    train = np.asarray(train_traj, dtype=np.float64)
    test  = np.asarray(test_traj,  dtype=np.float64)
    if len(train) < TRAIN_FIT_LEN + 1 or len(test) < TRAIN_FIT_LEN + TEST_TRUTH_LEN:
        return None

    # Build (fit, val_target) pairs at fixed fit_len for each val_start.
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

    # For each config, average val sMAPE across the val windows.
    gdc_val = []
    for recipe, sigma, ac, afc in GDC_CONFIGS:
        sms = []
        try:
            for fit_w, target in val_specs:
                fc = gdc_forecast(fit_w, fit_w[-min(16, len(fit_w)):],
                                   T=PRED_LEN, recipe=recipe,
                                   sigma_frac=sigma,
                                   alpha_ctx=ac, alpha_fc=afc)
                sms.append(smape(target, fc[:TEST_TRUTH_LEN]))
            if sms:
                gdc_val.append((float(np.mean(sms)), recipe, sigma, ac, afc))
        except Exception:
            continue
    gdc_val.sort(key=lambda x: x[0])
    if not gdc_val:
        return None
    g_val_sm, g_recipe, g_sigma, g_ac, g_afc = gdc_val[0]

    test_fit = test[:TRAIN_FIT_LEN]
    test_truth = test[TRAIN_FIT_LEN:TRAIN_FIT_LEN + TEST_TRUTH_LEN]
    g_fc = gdc_forecast(test_fit, test_fit[-min(16, len(test_fit)):],
                         T=PRED_LEN, recipe=g_recipe,
                         sigma_frac=g_sigma,
                         alpha_ctx=g_ac, alpha_fc=g_afc)[:TEST_TRUTH_LEN]

    return dict(
        system=system,
        gdc_smape=smape(test_truth, g_fc),
        gdc_mse=mse(test_truth, g_fc),
        gdc_mae=mae(test_truth, g_fc),
        gdc_pick=f'{g_recipe}_s{g_sigma}_ac{g_ac}_afc{g_afc}',
        gdc_val_smape=g_val_sm,
    )


def main():
    print(f"=== dysts univariate (dual-α, multi-IC val) ===")
    print(f"Device: {DEVICE}  dtype={DTYPE}")
    print(f"GDC configs: {len(GDC_CONFIGS)} "
          f"({len(SINGLE_ALPHAS)} single-α + {len(DUAL_ALPHAS)} dual-α "
          f"× 2 recipe × 3 σ)")
    print(f"Val starts: {VAL_STARTS}  fit_len={VAL_FIT_LEN}  "
          f"({len(VAL_STARTS)} windows averaged per config)")
    train_data = json.load(open(TRAIN_PATH))
    test_data  = json.load(open(TEST_PATH))
    systems = sorted(set(train_data) & set(test_data))
    print(f"Systems: {len(systems)}")

    rows = []
    t0 = time.time()
    for i, sys_name in enumerate(systems):
        try:
            train_v = train_data[sys_name].get('values') \
                if isinstance(train_data[sys_name], dict) else train_data[sys_name]
            test_v  = test_data[sys_name].get('values') \
                if isinstance(test_data[sys_name], dict) else test_data[sys_name]
            r = run_one_system(sys_name, train_v, test_v)
            if r is not None:
                rows.append(r)
        except Exception as e:
            print(f"  [skip] {sys_name}: {e}")
        if (i + 1) % 20 == 0 or i == len(systems) - 1:
            print(f"  {i+1}/{len(systems)} systems  [{time.time()-t0:.1f}s]")
    print(f"Total: {time.time() - t0:.1f}s, {len(rows)} systems scored")

    tag = 'dual' if any(ac != afc for ac, afc in ALPHA_CONFIGS) else 'single'
    out = os.path.join(HERE, 'results',
                        f'gdc_{tag}_dysts_multiIC{len(VAL_STARTS)}_fitlen{VAL_FIT_LEN}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = ['system', 'gdc_smape', 'gdc_mse', 'gdc_mae',
              'gdc_pick', 'gdc_val_smape']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}\n")

    arr = lambda key: np.array([r[key] for r in rows
                                 if not np.isnan(r[key]) and np.isfinite(r[key])])
    g = arr('gdc_smape')
    print(f"=== Dual-α GDC across {len(g)} systems ===")
    print(f"  GDC (dual-α grid)  median={np.median(g):.2f}  mean={g.mean():.2f}")

    # Comparison vs the existing single-α run (if CSV present).
    if os.path.exists(EXISTING_GDC_CSV):
        import csv as _csv
        old_by_system = {}
        with open(EXISTING_GDC_CSV) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    old_by_system[row['system']] = float(row['gdc_smape'])
                except (TypeError, ValueError):
                    continue
        common = [r['system'] for r in rows if r['system'] in old_by_system]
        if common:
            old_vals = np.array([old_by_system[s] for s in common])
            new_vals = np.array([next(r['gdc_smape'] for r in rows
                                       if r['system'] == s) for s in common])
            print(f"\n=== On the {len(common)}-system intersection w/ "
                  f"single-α GDC ===")
            print(f"  GDC single-α (old)  median={np.median(old_vals):.2f}  "
                  f"mean={old_vals.mean():.2f}")
            print(f"  GDC dual-α (new)    median={np.median(new_vals):.2f}  "
                  f"mean={new_vals.mean():.2f}")
            wins  = int((new_vals < old_vals).sum())
            losses = int((new_vals > old_vals).sum())
            ties = int((new_vals == old_vals).sum())
            print(f"  per-system: dual-α wins on {wins}, loses on {losses}, "
                  f"ties {ties} / {len(common)}")

    # Leaderboard against released baselines + old parrot/GDC.
    if os.path.exists(BASELINES):
        bl = json.load(open(BASELINES))
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
        agg.append(('GDC (dual-α)', float(np.median(g)), float(g.mean()), len(g)))
        # also show old GDC if available
        if os.path.exists(EXISTING_GDC_CSV):
            import csv as _csv
            with open(EXISTING_GDC_CSV) as f:
                old_sm = []
                for row in _csv.DictReader(f):
                    try:
                        v = float(row['gdc_smape'])
                        if v == v and np.isfinite(v): old_sm.append(v)
                    except (TypeError, ValueError):
                        continue
            if old_sm:
                old_sm = np.array(old_sm)
                agg.append(('GDC (single-α)', float(np.median(old_sm)),
                             float(old_sm.mean()), len(old_sm)))
        agg.sort(key=lambda x: x[1])
        print()
        print(f"=== Median sMAPE leaderboard (lower is better) ===")
        print(f"{'rank':>4s}  {'method':>22s}  {'median':>8s}  {'mean':>8s}  {'n':>5s}")
        for i, (m, med, mn, n) in enumerate(agg):
            tag = ' ← ours' if 'GDC' in m else ''
            print(f"{i+1:>4d}  {m:>22s}  {med:>8.2f}  {mn:>8.2f}  {n:>5d}{tag}")


if __name__ == "__main__":
    main()
