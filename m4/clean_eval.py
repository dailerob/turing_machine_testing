"""Clean leakage-free evaluation of GDC for any M4 frequency.

For each (series, config):
  val_smape  = sMAPE( predict(train[:-h]), train[-h:] )
  test_smape = sMAPE( predict(train),       test )

Outputs both:
  (1) per-series val-tuned: pick argmin_config val_smape per series,
      report mean test_smape.
  (2) global single-config: pick argmin_config mean(val_smape across series),
      apply that config to test, report mean test_smape.

Only GDC-TS based methods are evaluated. Naive_last is included as a
fallback for series too short to sweep.

Usage:
  python m4/clean_eval.py <freq>
where freq in {Daily, Weekly, Monthly, Quarterly, Yearly}.
"""
from __future__ import annotations
import os
import sys
import csv
import time
import argparse
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import data_loader as dl
from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries  # noqa: E402


def smape(actual, forecast):
    a = np.asarray(actual, dtype=np.float64); f = np.asarray(forecast, dtype=np.float64)
    denom = (np.abs(a) + np.abs(f)) / 2.0
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return 100.0 * np.mean(np.abs(a - f) / denom)


def mase_scale(train, season):
    """In-sample MAE of the seasonal-naive forecast (denominator of MASE).
    Matches M4's mase() in ML_benchmarks.py."""
    train = np.asarray(train, dtype=np.float64)
    if len(train) <= season:
        if len(train) < 2:
            return 1e-12
        return max(float(np.mean(np.abs(np.diff(train)))), 1e-12)
    return max(float(np.mean(np.abs(train[season:] - train[:-season]))), 1e-12)


# M4 official seasonality period for MASE/Naive2 (Yearly/Weekly/Daily=1).
M4_PERIOD = {"Yearly": 1, "Quarterly": 4, "Monthly": 12,
             "Weekly": 1, "Daily": 1, "Hourly": 24}


def mase(actual, forecast, scale):
    a = np.asarray(actual, dtype=np.float64); f = np.asarray(forecast, dtype=np.float64)
    return float(np.mean(np.abs(a - f)) / scale)


def mse(actual, forecast):
    a = np.asarray(actual, dtype=np.float64); f = np.asarray(forecast, dtype=np.float64)
    return float(np.mean((a - f) ** 2))


def nrmse(actual, forecast, scale):
    """RMSE normalized by training-series scale (std), per-series scale-invariant."""
    a = np.asarray(actual, dtype=np.float64); f = np.asarray(forecast, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((a - f) ** 2)))
    return rmse / max(scale, 1e-12)


def naive_last_pred(train, h):
    return np.full(h, train[-1])


def gdc_raw_forecast(train, window_len, sigma_frac, alpha, theta, h):
    """Hourly recipe: GDC-TS on raw values (no differencing, cycle anchors)."""
    n = len(train)
    if n < window_len + h + 1:
        return naive_last_pred(train, h)
    sigma_per_step = float(np.std(train)) * sigma_frac
    sigma_gdc = sigma_per_step * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = train.reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform')
    prime = train[-window_len:].reshape(-1, 1)
    _, sd = gdc.forecast_gdc_style(prime, n_steps=h)
    nt_mask = (~gdc.terminal_mask).astype(float)
    sd_nt = sd * nt_mask[None, :]
    sd_sum = sd_nt.sum(axis=1, keepdims=True)
    safe = np.where(sd_sum > 1e-12, sd_sum, 1.0)
    return ((sd_nt / safe) @ gdc.states)[:, 0]


def gdc_diff_forecast(train, window_len, sigma_frac, alpha, theta, h):
    n = len(train)
    if n < window_len + h + 2:
        return naive_last_pred(train, h)
    d = np.diff(train)
    if len(d) < window_len + h:
        return naive_last_pred(train, h)
    sigma_per_step = float(np.std(d)) * sigma_frac
    sigma_gdc = sigma_per_step * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = d.reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform')
    prime = d[-window_len:].reshape(-1, 1)
    _, sd = gdc.forecast_gdc_style(prime, n_steps=h)
    nt_mask = (~gdc.terminal_mask).astype(float)
    sd_nt = sd * nt_mask[None, :]
    sd_sum = sd_nt.sum(axis=1, keepdims=True)
    safe = np.where(sd_sum > 1e-12, sd_sum, 1.0)
    forecast_d = ((sd_nt / safe) @ gdc.states)[:, 0]
    return train[-1] + np.cumsum(forecast_d)


# Mark each freq's recipe: 'diff' (default) or 'raw' (hourly).
RECIPE_BY_FREQ = {"Hourly": "raw"}  # all others default to diff

# Candidate GDC configs per frequency. All use absorb mode + self_loop.
CONFIGS_BY_FREQ = {
    "Hourly": [
        ("naive_last", None),
        ("gdc_L24_s0.05_a1.0",  dict(window_len=24,  sigma_frac=0.05, alpha=1.0, theta=0.0)),
        ("gdc_L24_s0.10_a1.0",  dict(window_len=24,  sigma_frac=0.10, alpha=1.0, theta=0.0)),
        ("gdc_L48_s0.05_a1.0",  dict(window_len=48,  sigma_frac=0.05, alpha=1.0, theta=0.0)),
        ("gdc_L48_s0.10_a1.0",  dict(window_len=48,  sigma_frac=0.10, alpha=1.0, theta=0.0)),
        ("gdc_L48_s0.20_a1.0",  dict(window_len=48,  sigma_frac=0.20, alpha=1.0, theta=0.0)),
        ("gdc_L72_s0.05_a1.0",  dict(window_len=72,  sigma_frac=0.05, alpha=1.0, theta=0.0)),
        ("gdc_L72_s0.10_a1.0",  dict(window_len=72,  sigma_frac=0.10, alpha=1.0, theta=0.0)),
        ("gdc_L168_s0.05_a1.0", dict(window_len=168, sigma_frac=0.05, alpha=1.0, theta=0.0)),
        ("gdc_L168_s0.10_a1.0", dict(window_len=168, sigma_frac=0.10, alpha=1.0, theta=0.0)),
    ],
    "Daily": [
        ("naive_last", None),
        ("gdc_L7_s0.50_a1.0",  dict(window_len=7,  sigma_frac=0.50, alpha=1.0,  theta=0.0)),
        ("gdc_L14_s0.50_a1.0", dict(window_len=14, sigma_frac=0.50, alpha=1.0,  theta=0.0)),
        ("gdc_L28_s0.50_a1.0", dict(window_len=28, sigma_frac=0.50, alpha=1.0,  theta=0.0)),
        ("gdc_L14_s1.00_a1.0", dict(window_len=14, sigma_frac=1.00, alpha=1.0,  theta=0.0)),
        ("gdc_L7_s1.00_a1.0",  dict(window_len=7,  sigma_frac=1.00, alpha=1.0,  theta=0.0)),
        ("gdc_L14_s0.50_a0.99",dict(window_len=14, sigma_frac=0.50, alpha=0.99, theta=0.0)),
    ],
    "Weekly": [
        ("naive_last", None),
        ("gdc_L13_s0.25_a0.99", dict(window_len=13, sigma_frac=0.25, alpha=0.99, theta=0.0)),
        ("gdc_L26_s0.10_a0.99", dict(window_len=26, sigma_frac=0.10, alpha=0.99, theta=0.0)),
        ("gdc_L26_s0.25_a0.99", dict(window_len=26, sigma_frac=0.25, alpha=0.99, theta=0.0)),
        ("gdc_L26_s0.50_a0.99", dict(window_len=26, sigma_frac=0.50, alpha=0.99, theta=0.0)),
        ("gdc_L52_s0.25_a0.99", dict(window_len=52, sigma_frac=0.25, alpha=0.99, theta=0.0)),
        ("gdc_L26_s0.25_a0.95", dict(window_len=26, sigma_frac=0.25, alpha=0.95, theta=0.0)),
    ],
    "Monthly": [
        ("naive_last", None),
        ("gdc_L6_s0.25_a0.95",  dict(window_len=6,  sigma_frac=0.25, alpha=0.95, theta=0.0)),
        ("gdc_L6_s0.50_a0.95",  dict(window_len=6,  sigma_frac=0.50, alpha=0.95, theta=0.0)),
        ("gdc_L12_s0.25_a0.95", dict(window_len=12, sigma_frac=0.25, alpha=0.95, theta=0.0)),
        ("gdc_L12_s0.50_a0.95", dict(window_len=12, sigma_frac=0.50, alpha=0.95, theta=0.0)),
        ("gdc_L18_s0.25_a0.95", dict(window_len=18, sigma_frac=0.25, alpha=0.95, theta=0.0)),
        ("gdc_L12_s0.25_a0.99", dict(window_len=12, sigma_frac=0.25, alpha=0.99, theta=0.0)),
        ("gdc_L12_s0.25_a0.9",  dict(window_len=12, sigma_frac=0.25, alpha=0.9,  theta=0.0)),
    ],
    "Quarterly": [
        ("naive_last", None),
        ("gdc_L4_s0.50_a0.9",   dict(window_len=4,  sigma_frac=0.50, alpha=0.9,  theta=0.0)),
        ("gdc_L6_s0.50_a0.9",   dict(window_len=6,  sigma_frac=0.50, alpha=0.9,  theta=0.0)),
        ("gdc_L8_s0.25_a0.9",   dict(window_len=8,  sigma_frac=0.25, alpha=0.9,  theta=0.0)),
        ("gdc_L8_s0.50_a0.9",   dict(window_len=8,  sigma_frac=0.50, alpha=0.9,  theta=0.0)),
        ("gdc_L12_s0.25_a0.9",  dict(window_len=12, sigma_frac=0.25, alpha=0.9,  theta=0.0)),
        ("gdc_L12_s0.25_a0.95", dict(window_len=12, sigma_frac=0.25, alpha=0.95, theta=0.0)),
    ],
    "Yearly": [
        ("naive_last", None),
        ("gdc_L3_s0.50_a0.8",  dict(window_len=3, sigma_frac=0.50, alpha=0.8, theta=0.0)),
        ("gdc_L4_s0.50_a0.8",  dict(window_len=4, sigma_frac=0.50, alpha=0.8, theta=0.0)),
        ("gdc_L6_s0.50_a0.8",  dict(window_len=6, sigma_frac=0.50, alpha=0.8, theta=0.0)),
        ("gdc_L8_s0.50_a0.8",  dict(window_len=8, sigma_frac=0.50, alpha=0.8, theta=0.0)),
        ("gdc_L8_s0.25_a0.8",  dict(window_len=8, sigma_frac=0.25, alpha=0.8, theta=0.0)),
        ("gdc_L8_s0.50_a0.9",  dict(window_len=8, sigma_frac=0.50, alpha=0.9, theta=0.0)),
    ],
}


def _predict(cfg, train, h, recipe='diff'):
    if cfg is None:
        return naive_last_pred(train, h)
    if recipe == 'raw':
        return gdc_raw_forecast(train, h=h, **cfg)
    return gdc_diff_forecast(train, h=h, **cfg)


def run_series(args):
    sid, train, test, configs, h, recipe, season = args
    rows = []
    has_val = len(train) >= 2 * h + 4
    if has_val:
        tr_v = train[:-h]; val = train[-h:]
        val_scale = mase_scale(tr_v, season)
    train_scale = float(np.std(np.diff(train))) if len(train) > 1 else 1.0
    test_mase_scale = mase_scale(train, season)
    for name, cfg in configs:
        try:
            test_pred = _predict(cfg, train, h, recipe)
            test_sm = smape(test, test_pred)
            test_ms = mse(test, test_pred)
            test_nr = nrmse(test, test_pred, train_scale)
            test_ma = mase(test, test_pred, test_mase_scale)
        except Exception:
            test_sm = test_ms = test_nr = test_ma = float('nan')
        if has_val:
            try:
                val_pred = _predict(cfg, tr_v, h, recipe)
                val_sm = smape(val, val_pred)
                val_ms = mse(val, val_pred)
                val_nr = nrmse(val, val_pred, train_scale)
                val_ma = mase(val, val_pred, val_scale)
            except Exception:
                val_sm = val_ms = val_nr = val_ma = float('nan')
        else:
            val_sm = val_ms = val_nr = val_ma = float('nan')
        rows.append(dict(sid=sid, model=name,
                         val_smape=val_sm, test_smape=test_sm,
                         val_mse=val_ms, test_mse=test_ms,
                         val_nrmse=val_nr, test_nrmse=test_nr,
                         val_mase=val_ma, test_mase=test_ma))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("freq", choices=list(CONFIGS_BY_FREQ.keys()))
    ap.add_argument("--per-series-tune", action="store_true",
                    help="If set, additionally compute per-series val-tuned scores. "
                         "Skip for Quarterly/Yearly to save time.")
    ap.add_argument("--out-prefix", default="clean_eval",
                    help="Output file prefix in m4/<freq>/")
    args = ap.parse_args()
    freq = args.freq
    h = dl.horizon(freq)
    configs = CONFIGS_BY_FREQ[freq]
    out_dir = os.path.join(HERE, freq.lower())
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{args.out_prefix}.log")
    csv_path = os.path.join(out_dir, f"{args.out_prefix}_results.csv")
    summary_path = os.path.join(out_dir, f"{args.out_prefix}_summary.md")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f: f.write(msg + "\n")

    log(f"=== clean_eval freq={freq} h={h} configs={len(configs)} per_series_tune={args.per_series_tune} ===")
    log(f"Loading {freq} data...")
    train = dl.load_train(freq); test = dl.load_test(freq)
    ids = sorted(train.keys(), key=lambda s: int(s[1:]))
    recipe = RECIPE_BY_FREQ.get(freq, 'diff')
    log(f"Recipe: {recipe}")
    season = M4_PERIOD[freq]
    log(f"Season (for MASE scale, M4 official): {season}")
    tasks = [(sid, train[sid], test[sid], configs, h, recipe, season) for sid in ids]
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    log(f"Sweeping val+test for {len(tasks)} series, {n_workers} workers")

    all_rows = []
    t0 = time.time(); done = 0
    every = max(1, len(tasks) // 50)
    with mp.Pool(processes=n_workers) as pool:
        chunksize = 32 if len(tasks) > 1000 else 4
        for r in pool.imap_unordered(run_series, tasks, chunksize=chunksize):
            all_rows.extend(r); done += 1
            if done % every == 0 or done == len(tasks):
                pct = 100.0 * done / len(tasks)
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(tasks) - done) / rate
                log(f"  {done}/{len(tasks)} ({pct:.1f}%) [{time.time()-t0:.0f}s, "
                    f"~{eta:.0f}s left]")

    log(f"Done in {time.time()-t0:.1f}s, {len(all_rows)} rows")
    fields = ["sid", "model", "val_smape", "test_smape",
              "val_mse", "test_mse", "val_nrmse", "test_nrmse",
              "val_mase", "test_mase"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    log(f"Wrote {csv_path}")

    from collections import defaultdict

    def aggregate(metric, fmt='%.2f%%'):
        val_key = f'val_{metric}'; test_key = f'test_{metric}'
        by_sid = defaultdict(dict)
        for r in all_rows:
            by_sid[r['sid']][r['model']] = (r[val_key], r[test_key])
        per_series_scores = []; pick_counts = defaultdict(int)
        for sid in ids:
            d = by_sid[sid]
            eligible = [(name, v, t) for name, (v, t) in d.items() if not np.isnan(v)]
            if not eligible:
                test_sm = d.get("naive_last", (np.nan, np.nan))[1]
                best = "naive_last (fallback)"
            else:
                name, v, t = min(eligible, key=lambda x: x[1])
                test_sm = t; best = name
            per_series_scores.append(test_sm); pick_counts[best] += 1
        ps = np.array(per_series_scores, dtype=np.float64)
        finite = ps[np.isfinite(ps)]
        ps_mean = finite.mean() if len(finite) else float('nan')
        ps_med = np.median(finite) if len(finite) else float('nan')
        log(f"\n--- ({metric.upper()}) (1) Per-series val-tuned ---")
        log(f"  test mean = {fmt % ps_mean}   median = {fmt % ps_med}   "
            f"n_finite = {len(finite)}/{len(ps)}")
        for n, c in sorted(pick_counts.items(), key=lambda x: -x[1]):
            log(f"    {n:>30s}: {c}")
        global_val = defaultdict(list); global_test = defaultdict(list)
        for r in all_rows:
            if not np.isnan(r[val_key]): global_val[r['model']].append(r[val_key])
            if not np.isnan(r[test_key]): global_test[r['model']].append(r[test_key])
        rank = []
        for name, _ in configs:
            v = np.array(global_val[name]); t = np.array(global_test[name])
            rank.append((v.mean() if len(v) else np.inf, name,
                         t.mean() if len(t) else np.nan,
                         np.median(t) if len(t) else np.nan))
        rank.sort()
        log(f"\n--- ({metric.upper()}) (2) Global single-config picked by mean val ---")
        log(f"{'model':>30s}  {'mean_val':>11s}  {'mean_test':>11s}  {'med_test':>11s}")
        for v, name, tm, tmd in rank:
            log(f"{name:>30s}  {fmt % v:>11s}  {fmt % tm:>11s}  {fmt % tmd:>11s}")
        bgv, bgn, bgtm, bgtmd = rank[0]
        log(f"\n  best by val: {bgn}  -> test mean = {fmt % bgtm}, median = {fmt % bgtmd}")
        return dict(metric=metric, ps_mean=ps_mean, ps_median=ps_med,
                    pick_counts=dict(pick_counts), rank=rank,
                    best_global=dict(name=bgn, val=bgv, test_mean=bgtm, test_median=bgtmd))

    agg_smape = aggregate('smape', fmt='%.2f%%')
    agg_mase = aggregate('mase',  fmt='%.4f')
    agg_mse = aggregate('mse',   fmt='%.4g')
    agg_nrmse = aggregate('nrmse', fmt='%.4f')

    # ---- OWA against published M4 Naive 2 reference ----
    PUB_NAIVE2 = {  # (mean sMAPE %, mean MASE) from M4 supplementary
        "Yearly":    (16.342, 3.974),
        "Quarterly": (11.012, 1.371),
        "Monthly":   (14.427, 1.063),
        "Weekly":    ( 9.161, 2.777),
        "Daily":     ( 3.045, 3.278),
        "Hourly":    (18.383, 2.395),
    }
    n2_sm, n2_ma = PUB_NAIVE2[freq]
    log(f"\n--- (OWA) vs published M4 Naive 2 ({n2_sm:.3f} sMAPE / {n2_ma:.4f} MASE) ---")
    def owa_for(method, smape_mean, mase_mean):
        owa = 0.5 * (smape_mean / n2_sm + mase_mean / n2_ma)
        return owa
    ps_owa = owa_for('per-series', agg_smape['ps_mean'], agg_mase['ps_mean'])
    log(f"  (1) per-series val-tuned: sMAPE={agg_smape['ps_mean']:.3f}, "
        f"MASE={agg_mase['ps_mean']:.4f}, OWA={ps_owa:.4f}")
    # Global pick: use whichever config the val sMAPE picks (matches the table)
    bg_smape = agg_smape['best_global']
    bg_mase = agg_mase['best_global']
    g_owa_smape_pick = owa_for('global-smape-pick',
        bg_smape['test_mean'],
        # use that same config's MASE test mean
        next(rk[2] for rk in agg_mase['rank'] if False) if False else
        # find this config's test MASE from the per-config aggregation
        np.mean([r['test_mase'] for r in all_rows if r['model'] == bg_smape['name'] and not np.isnan(r['test_mase'])]))
    log(f"  (2) global pick by val-sMAPE [{bg_smape['name']}]: "
        f"OWA = {g_owa_smape_pick:.4f}")
    g_owa_mase_pick = owa_for('global-mase-pick',
        np.mean([r['test_smape'] for r in all_rows if r['model'] == bg_mase['name'] and not np.isnan(r['test_smape'])]),
        bg_mase['test_mean'])
    log(f"  (2') global pick by val-MASE [{bg_mase['name']}]: "
        f"OWA = {g_owa_mase_pick:.4f}")
    owa_summary = dict(naive2_smape=n2_sm, naive2_mase=n2_ma,
                       per_series_owa=ps_owa,
                       global_smape_pick=dict(name=bg_smape['name'], owa=g_owa_smape_pick),
                       global_mase_pick=dict(name=bg_mase['name'], owa=g_owa_mase_pick))

    # ---- Write summary ----
    def _fmt(metric, v):
        if metric == 'smape': return f"{v:.2f}%"
        if metric == 'mse':   return f"{v:.4g}"
        return f"{v:.4f}"  # nrmse / mase
    with open(summary_path, "w") as f:
        f.write(f"# Clean (leakage-free) evaluation: {freq}\n\n")
        f.write(f"- Series: {len(ids)}, h = {h}\n")
        f.write(f"- Candidate configs: {len(configs)} (all GDC-TS absorb + naive fallback)\n\n")
        f.write(f"## OWA vs published M4 Naive 2 (sMAPE={n2_sm:.3f}, MASE={n2_ma:.4f})\n\n")
        f.write(f"- **(1) Per-series val-tuned OWA = {ps_owa:.4f}**\n")
        f.write(f"- (2) Global pick by val-sMAPE [{bg_smape['name']}] OWA = {g_owa_smape_pick:.4f}\n")
        f.write(f"- (2') Global pick by val-MASE [{bg_mase['name']}] OWA = {g_owa_mase_pick:.4f}\n\n")
        for agg in (agg_smape, agg_mase, agg_mse, agg_nrmse):
            m = agg['metric']; M = m.upper()
            f.write(f"## ({M}) (1) Per-series val-tuned\n\n")
            f.write(f"- Test mean: **{_fmt(m, agg['ps_mean'])}**\n")
            f.write(f"- Test median: **{_fmt(m, agg['ps_median'])}**\n\n")
            f.write("Picks:\n\n| config | n picks |\n|---|---:|\n")
            for n, c in sorted(agg['pick_counts'].items(), key=lambda x: -x[1]):
                f.write(f"| {n} | {c} |\n")
            f.write(f"\n## ({M}) (2) Global single config picked by mean val\n\n")
            f.write("| config | mean val | mean test | median test |\n|---|---:|---:|---:|\n")
            for v, name, tm, tmd in agg['rank']:
                f.write(f"| {name} | {_fmt(m, v)} | {_fmt(m, tm)} | {_fmt(m, tmd)} |\n")
            bg = agg['best_global']
            f.write(f"\n**Picked by val: `{bg['name']}` -> test mean "
                    f"= {_fmt(m, bg['test_mean'])}, median = {_fmt(m, bg['test_median'])}**\n\n")
    log(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
