"""Smoke test: AnDA on a small set of dysts systems.

Goal: verify the AnDA analog-forecasting pipeline works with our
val-tuning protocol, get realistic per-system timing, and decide on the
full sweep config.

Protocol (mirrors run_parrot_gdc.py / run_pyedm.py):
  - For each system:
      val tuning: build catalog from train[:150], score 30-step
                  generative rollout vs train[150:179]
      test eval:  build catalog from test[:150],  score vs test[150:179]
  - sMAPE on first 29 of 30 generated predictions
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANDA_PATH = '/home/roberto/turing_machine_testing/external/AnDA'
sys.path.insert(0, ANDA_PATH)

from AnDA_codes.AnDA_analog_forecasting import AnDA_analog_forecasting


# Dummy classes to mirror AnDA's expected attribute access pattern
class Catalog:
    def __init__(self, analogs, successors):
        self.analogs = analogs           # (n_pairs, E)
        self.successors = successors     # (n_pairs, E)


class AF:
    """Config object passed to AnDA_analog_forecasting."""
    def __init__(self, catalog, k, regression, sampling, E):
        self.catalog = catalog
        self.k = k
        self.regression = regression       # 'locally_constant'|'increment'|'local_linear'
        self.sampling = sampling            # 'gaussian'|'multinomial'
        # neighborhood: (E, E) ones means each var sees all others as neighbours (global mode)
        self.neighborhood = np.ones((E, E), dtype=np.int64)


def build_catalog(series, E):
    """Build (analog, successor) pairs from a 1-D series.
    analog[i]    = series[i:i+E]
    successor[i] = series[i+1:i+E+1]   (Takens vector shifted by one step)
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
    return Catalog(analogs, successors)


def anda_forecast(series_fit, E, k, regression, sampling, n_steps=30):
    """Autoregressive 30-step forecast using AnDA on a 1-D fit series.

    Returns array of length n_steps of predicted raw values, or None on error.
    """
    catalog = build_catalog(series_fit, E)
    if catalog is None or catalog.analogs.shape[0] < k:
        return None
    af = AF(catalog, k=k, regression=regression, sampling=sampling, E=E)
    # Initial query: most recent E values of fit series, as a (1, E) row
    cur = np.asarray(series_fit[-E:], dtype=np.float64).reshape(1, E)
    preds = []
    for step in range(n_steps):
        try:
            xf, xf_mean = AnDA_analog_forecasting(cur, af)
        except Exception:
            return None
        # The forecast is the next Takens vector; the new value is its last element
        next_val = float(xf_mean[0, -1])
        preds.append(next_val)
        # Slide window: append next_val, drop oldest
        cur = np.column_stack([cur[:, 1:], np.array([[next_val]])])
    return np.asarray(preds, dtype=np.float64)


def smape(t, p):
    t = np.asarray(t, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    denom = (np.abs(t) + np.abs(p)) / 2.0
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return float(100 * np.mean(np.abs(t - p) / denom))


# ---- Smoke run on 3 systems with full config grid -----------------------
def main():
    train = json.load(open(os.path.join(HERE, 'data', 'train.json')))
    test  = json.load(open(os.path.join(HERE, 'data', 'test.json')))

    test_systems = ('Lorenz', 'Aizawa', 'Rossler')

    REGRESSIONS = ('locally_constant', 'increment', 'local_linear')
    ES = (3, 5, 8)
    SAMPLING = 'gaussian'  # mean is deterministic; sampling needed only for xf
    np.random.seed(0)      # reproducibility for the gaussian draw

    # Build full (regression, E, k) config grid: k ∈ {E+1, 10, 25, 50}
    CONFIGS = []
    for regression in REGRESSIONS:
        for E in ES:
            for k in sorted({E + 1, 10, 25, 50}):
                CONFIGS.append((regression, E, k))

    print(f"=== AnDA smoke test on 3 dysts systems ===")
    print(f"Configs per system: {len(CONFIGS)} (regression × E × k)")
    print()

    rows = []
    total_t0 = time.time()
    for sys_name in test_systems:
        train_v = np.asarray(train[sys_name]['values'], dtype=np.float64)
        test_v  = np.asarray(test[sys_name]['values'],  dtype=np.float64)

        # ---- Val sweep on train trajectory ----
        val_truth = train_v[150:179]
        val_results = []
        t_sys = time.time()
        for regression, E, k in CONFIGS:
            t0 = time.time()
            fc = anda_forecast(train_v[:150], E=E, k=k,
                                regression=regression,
                                sampling=SAMPLING, n_steps=30)
            dt = time.time() - t0
            if fc is None or not np.all(np.isfinite(fc)):
                continue
            sm = smape(val_truth, fc[:29])
            val_results.append((sm, regression, E, k, dt))

        if not val_results:
            print(f"  {sys_name}: no val pick — skipping")
            continue
        val_results.sort(key=lambda x: x[0])
        best = val_results[0]
        val_sm, best_reg, best_E, best_k, _ = best

        # Print top-10 val results for this system
        print(f"\n  {sys_name} top val picks (of {len(val_results)} valid):")
        for sm, reg, E, k, dt in val_results[:10]:
            mark = ' ← pick' if (reg, E, k) == (best_reg, best_E, best_k) else ''
            print(f"    {reg:<18s} E={E} k={k:>2d}  val_sMAPE={sm:>7.2f}{mark}")

        # ---- Test eval with val-picked config ----
        test_truth = test_v[150:179]
        t0 = time.time()
        fc = anda_forecast(test_v[:150], E=best_E, k=best_k,
                            regression=best_reg, sampling=SAMPLING,
                            n_steps=30)
        test_dt = time.time() - t0
        if fc is None:
            print(f"  {sys_name}: TEST FAILED with val-picked config")
            continue
        test_sm = smape(test_truth, fc[:29])
        sys_total = time.time() - t_sys
        print(f"\n  {sys_name}: val pick = {best_reg}/E={best_E}/k={best_k}  "
              f"val_sMAPE={val_sm:.2f}  test_sMAPE={test_sm:.2f}  "
              f"(total system runtime: {sys_total:.2f}s)\n")
        rows.append((sys_name, best_reg, best_E, best_k, val_sm, test_sm, sys_total))

    total_t = time.time() - total_t0
    print(f"\n=== Summary ===")
    print(f"  total wall-clock: {total_t:.1f}s for {len(rows)} systems")
    if rows:
        per_sys = sum(r[5] for r in rows) / len(rows)
        print(f"  avg per system: {per_sys:.2f}s (val + test)")
        n_full = 131
        print(f"  estimated runtime for full {n_full}-system sweep: {per_sys * n_full:.0f}s "
              f"(~{per_sys * n_full / 60:.1f} min)")
    print()
    print(f"Per-system test results:")
    print(f"  {'system':>12s}  {'regression':>18s}  {'E':>2s}  {'k':>3s}  {'val':>7s}  {'test':>7s}")
    for sys_name, reg, E, k, val_sm, test_sm, _ in rows:
        print(f"  {sys_name:>12s}  {reg:>18s}  {E:>2d}  {k:>3d}  {val_sm:>7.2f}  {test_sm:>7.2f}")


if __name__ == "__main__":
    main()
