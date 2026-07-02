"""Leakage-free PAutomaC eval for GDC.

Earlier (Table 14) the 7-config GDC selection picked the config with the
lowest TEST gap — an oracle selection, because the PAutomaC competition
ships only train + test(+solution), no validation set.

Here we make it leakage-free:
  1. Split each problem's TRAIN sequences into train' (80%) / val (20%).
  2. Fit each of the 7 GDC configs on train'.
  3. Select the config with the lowest *held-out NLL* on val
     (mean negative log-likelihood the model assigns to val sequences;
     no true distribution needed).
  4. Refit the selected config on the FULL train set.
  5. Score on test using the official PAutomaC perplexity (vs solution).

Outputs a per-problem CSV + summary comparing:
  - GDC val-tuned (leakage-free)   [this script]
  - GDC oracle (best test gap)     [recomputed from the same 7 configs]
  - GDC fixed (ac=0.85, af=0.9999) [single config, no selection]
"""
from __future__ import annotations
import os, sys, csv, time, gc
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_loader import load_problem
from scoring import pautomac_score
from torch_adapters import TorchGDCModel, TorchDualGDCModel


def _free(model):
    """Release a fitted model's GPU tensors."""
    for attr in ('sym_t', 'term_t', 'start_t', 'symbol_onehot'):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except Exception:
                pass
    del model


def build_7_configs():
    """The exact 7 GDC configs used for Table 14."""
    return [
        # 2 single-alpha
        TorchGDCModel(alpha=0.95, theta=0.05, beta=0.0,
                      transition_type='self_loop',
                      terminal_behavior='diffuse',
                      initial_dist='sequence_starts',
                      chunk_size=64),
        TorchGDCModel(alpha=0.50, theta=0.005, beta=0.0,
                      transition_type='self_loop',
                      terminal_behavior='diffuse',
                      initial_dist='sequence_starts',
                      chunk_size=64),
        # 5 dual-alpha (alpha_fc = 0.9999 numerical safety floor)
        TorchDualGDCModel(alpha_ctx=0.30, alpha_fc=0.9999,
                          theta_ctx=0.0, theta_fc=0.0, beta=0.0,
                          terminal_behavior='diffuse',
                          initial_dist='sequence_starts',
                          chunk_size=64),
        TorchDualGDCModel(alpha_ctx=0.50, alpha_fc=0.9999,
                          theta_ctx=0.0, theta_fc=0.0, beta=0.0,
                          terminal_behavior='diffuse',
                          initial_dist='sequence_starts',
                          chunk_size=64),
        TorchDualGDCModel(alpha_ctx=0.70, alpha_fc=0.9999,
                          theta_ctx=0.0, theta_fc=0.0, beta=0.0,
                          terminal_behavior='diffuse',
                          initial_dist='sequence_starts',
                          chunk_size=64),
        TorchDualGDCModel(alpha_ctx=0.85, alpha_fc=0.9999,
                          theta_ctx=0.0, theta_fc=0.0, beta=0.0,
                          terminal_behavior='diffuse',
                          initial_dist='sequence_starts',
                          chunk_size=64),
        TorchDualGDCModel(alpha_ctx=0.95, alpha_fc=0.9999,
                          theta_ctx=0.0, theta_fc=0.0, beta=0.0,
                          terminal_behavior='diffuse',
                          initial_dist='sequence_starts',
                          chunk_size=64),
    ]


# Index of the "fixed" config (ac=0.85) in build_7_configs() — config #6.
FIXED_IDX = 5


VAL_FRAC = 0.2  # held-out fraction of TRAIN for config selection (no cap)
FIXED_NAME = 'tgdc2-ac0.85-af0.9999-tc0.0-tf0.0-b0.0-dif'
SWEEP_CSV = os.path.join(HERE, 'results', 'pautomac_dual_alpha_0999_sweep.csv')


def split_train(train, val_frac=VAL_FRAC, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    n_val = max(1, int(round(val_frac * len(train))))
    val_idx = idx[:n_val]; tr_idx = idx[n_val:]
    train_prime = [train[i] for i in tr_idx]
    val = [train[i] for i in val_idx]
    return train_prime, val


def load_sweep_gaps():
    """Pull per-problem per-config test gaps from the Table-14 source CSV."""
    gaps = {}  # problem -> {config_name: gap}
    with open(SWEEP_CSV) as f:
        for r in csv.DictReader(f):
            gaps.setdefault(int(r['problem']), {})[r['model']] = float(r['gap'])
    return gaps


def main():
    out_csv = os.path.join(HERE, 'results', 'pautomac_leakage_free.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    sweep_gaps = load_sweep_gaps()   # full-train test gaps for all 7 configs
    FIELDS = ['problem', 'alphabet_size', 'lf_pick', 'lf_gap',
              'oracle_pick', 'oracle_gap', 'fixed_gap', 'val_nll_min']

    # Resume: load any prior per-problem results and skip those problems.
    rows = []
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                r['problem'] = int(r['problem'])
                r['alphabet_size'] = int(r['alphabet_size'])
                for k in ('lf_gap', 'oracle_gap', 'fixed_gap', 'val_nll_min'):
                    r[k] = float(r[k])
                rows.append(r); done.add(r['problem'])
        print(f"Resuming: {len(done)} problems already done "
              f"({sorted(done)})\n")

    def flush_csv():
        rows_sorted = sorted(rows, key=lambda r: r['problem'])
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader(); w.writerows(rows_sorted)

    t_total = time.time()
    print("=== PAutomaC leakage-free GDC eval (7 configs, held-out NLL val) ===")
    print(f"    (oracle/fixed gaps pulled from {os.path.basename(SWEEP_CSV)};")
    print(f"     leakage-free pick = lowest held-out NLL on {int(VAL_FRAC*100)}% of train,")
    print(f"     its test gap = the selected config's full-train gap)\n")
    for pi in range(1, 49):
        if pi in done:
            continue
        try:
            problem = load_problem(pi)
        except FileNotFoundError:
            continue
        train = problem['train']; A = problem['alphabet_size']
        train_prime, val = split_train(train)
        config_names = [m.name for m in build_7_configs()]

        if pi not in sweep_gaps:
            print(f"P{pi}: no sweep-CSV gaps; skipping"); continue
        csv_gaps = sweep_gaps[pi]

        # --- Pass 1: fit each config on train', measure held-out val NLL ---
        val_nlls = []
        for ci in range(7):
            m = build_7_configs()[ci]
            m.fit(train_prime, A)
            val_lp = m.score_test_set(val)
            val_nlls.append(-float(np.mean(val_lp)))
            _free(m); torch.cuda.empty_cache()
        gc.collect()

        lf_idx = int(np.argmin(val_nlls))           # leakage-free pick

        # Test gaps from the CSV (full-train fits = the leakage-free final step).
        per_config_gap = [csv_gaps[name] for name in config_names]
        oracle_idx = int(np.argmin(per_config_gap))

        lf_gap = per_config_gap[lf_idx]
        oracle_gap = per_config_gap[oracle_idx]
        fixed_gap = csv_gaps[FIXED_NAME]

        rows.append(dict(
            problem=pi, alphabet_size=A,
            lf_pick=config_names[lf_idx], lf_gap=lf_gap,
            oracle_pick=config_names[oracle_idx], oracle_gap=oracle_gap,
            fixed_gap=fixed_gap,
            val_nll_min=val_nlls[lf_idx],
        ))
        flush_csv()   # checkpoint after every problem (resumable)
        print(f"P{pi:>2d} (A={A:>2d}): "
              f"LF-pick={config_names[lf_idx]:<26s} gap={lf_gap:>7.3f}  | "
              f"oracle={oracle_gap:>7.3f} | fixed={fixed_gap:>7.3f}  "
              f"{'(LF==oracle)' if lf_idx==oracle_idx else ''}",
              flush=True)

    flush_csv()
    print(f"\nWrote {out_csv}  ({time.time()-t_total:.1f}s)\n")

    # --- Summary stats on the gap ---
    lf = np.array([r['lf_gap'] for r in rows])
    oracle = np.array([r['oracle_gap'] for r in rows])
    fixed = np.array([r['fixed_gap'] for r in rows])
    n_lf_eq_oracle = sum(1 for r in rows if r['lf_pick'] == r['oracle_pick'])

    print(f"=== Summary over {len(rows)} problems (gap above entropy floor) ===")
    print(f"{'method':<28s}  {'median':>8s}  {'mean':>8s}  {'max':>8s}")
    for name, arr in [('GDC val-tuned (leakage-free)', lf),
                       ('GDC oracle (best test gap)', oracle),
                       ('GDC fixed (ac=0.85, af=0.9999)', fixed)]:
        print(f"{name:<28s}  {np.median(arr):>8.3f}  {arr.mean():>8.3f}  {arr.max():>8.3f}")
    print(f"\nLeakage-free pick matched the oracle pick on "
          f"{n_lf_eq_oracle}/{len(rows)} problems.")
    print(f"Median gap penalty of leakage-free vs oracle: "
          f"{np.median(lf - oracle):.3f}")


if __name__ == "__main__":
    main()
