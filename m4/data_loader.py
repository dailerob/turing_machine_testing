"""Generic M4 data loader.

Each `<Frequency>-train.csv` / `<Frequency>-test.csv` follows the same
shape: row 0 is a header (V1, V2, ..., VN); each subsequent row is one
series with the id in column 0 and float values padded with NA.

Supported frequencies (from the M4 competition):
  Yearly      h=6
  Quarterly   h=8   season=4
  Monthly     h=18  season=12
  Weekly      h=13  season=1
  Daily       h=14  season=7   (weekly cycle)
  Hourly      h=48  season=24

Backwards-compat: `load_train()` / `load_test()` with no arguments
default to Hourly to keep older scripts working.
"""
from __future__ import annotations
import os
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
INFO_CSV = os.path.join(DATA_DIR, "M4-info.csv")

FREQUENCIES = {
    "Yearly":    dict(horizon=6,  seasonality=1,  prefix="Y"),
    "Quarterly": dict(horizon=8,  seasonality=4,  prefix="Q"),
    "Monthly":   dict(horizon=18, seasonality=12, prefix="M"),
    "Weekly":    dict(horizon=13, seasonality=1,  prefix="W"),
    "Daily":     dict(horizon=14, seasonality=7,  prefix="D"),
    "Hourly":    dict(horizon=48, seasonality=24, prefix="H"),
}

H_HORIZON = FREQUENCIES["Hourly"]["horizon"]
H_SEASONALITY = FREQUENCIES["Hourly"]["seasonality"]


def _seq_path(freq, split):
    return os.path.join(DATA_DIR, f"{freq}-{split}.csv")


def _load_seq_file(path):
    """Return dict id -> 1-D float64 array (NaN-trimmed)."""
    out = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            sid = row[0].strip('"')
            vals = []
            for v in row[1:]:
                v = v.strip().strip('"')
                if v == "" or v.upper() == "NA":
                    continue
                vals.append(float(v))
            out[sid] = np.asarray(vals, dtype=np.float64)
    return out


def load_train(freq="Hourly"):
    return _load_seq_file(_seq_path(freq, "train"))


def load_test(freq="Hourly"):
    return _load_seq_file(_seq_path(freq, "test"))


def horizon(freq="Hourly"):
    return FREQUENCIES[freq]["horizon"]


def seasonality(freq="Hourly"):
    return FREQUENCIES[freq]["seasonality"]


def list_ids(freq="Hourly"):
    pref = FREQUENCIES[freq]["prefix"]
    n = sum(1 for _ in open(_seq_path(freq, "train")))
    return [f"{pref}{i}" for i in range(1, n)]


if __name__ == "__main__":
    for freq in ("Hourly", "Daily"):
        try:
            train = load_train(freq); test = load_test(freq)
            tr_lens = [len(v) for v in train.values()]
            te_lens = [len(v) for v in test.values()]
            print(f"\n=== {freq} ===")
            print(f"  series: train={len(train)}, test={len(test)}")
            print(f"  train lens: min={min(tr_lens)}, "
                  f"median={sorted(tr_lens)[len(tr_lens)//2]}, "
                  f"max={max(tr_lens)}, mean={sum(tr_lens)/len(tr_lens):.1f}")
            print(f"  test lens: min={min(te_lens)}, max={max(te_lens)} "
                  f"(expected {horizon(freq)})")
            for sid in list(train.keys())[:3]:
                v = train[sid]
                print(f"    {sid}: len={len(v)}, range=[{v.min():.2f}, "
                      f"{v.max():.2f}], mean={v.mean():.2f}, std={v.std():.2f}")
        except FileNotFoundError as e:
            print(f"\n=== {freq} === missing data: {e}")
