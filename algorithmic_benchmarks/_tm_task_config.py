"""Shared train / val / test configuration for the algorithmic / TM
benchmarks.

Single source of truth for the input ranges per task, so every
method (GDC, CHMM, ALERGIA, Parrot, HPYLM, PPM-D, KN-3gram) sees
exactly the same train/val/test split. Val is drawn from a stretched
range strictly between train and test — informative for length
extrapolation while never overlapping the test set.

Every task entry has:
  module           — the simulator module (or 'binary_adder' / 'dyck1' marker)
  train_range      — input-length / number-range for training
  val_range        — input-length / number-range for validation (stretched)
  test_range       — input-length / number-range for test
  n_train, n_val, n_test, max_steps  — sample sizes & per-tape step cap
  seeds            — (train, val, test)

Use `simulate_train_val_test(task_name, variant)` to get
(train_runs, val_runs, test_runs) under the canonical split.
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

import parity_tm, increment_tm, reverse_tm  # noqa: E402
import shift_left_tm, bit_count_mod3_tm, anbn_tm  # noqa: E402
import palindrome_tm, subtraction_tm  # noqa: E402
from _tm_common import apply_noread_to_runs  # noqa: E402
from binary_alphabet_adder import (  # noqa: E402
    simulate_random_binary_alphabet_adders, BINARY_ALPHABET_ADDER)

# Training-set multiplier: scales n_train (and n_val) for every task.
# Set via env var TM_TRAIN_MULT=N to rerun with a larger training budget
# while keeping the same val_range/test_range/seeds. Output CSVs are
# suffixed with `_<mult>x` when ≠ 1 by the eval scripts; the canonical
# 1× CSVs are preserved.
TRAIN_MULTIPLIER = int(os.environ.get('TM_TRAIN_MULT', '1'))
SUFFIX = '' if TRAIN_MULTIPLIER == 1 else f'_{TRAIN_MULTIPLIER}x'


# --------------------------------------------------------------------
# Task table
# --------------------------------------------------------------------
# Val ranges are wider than they were originally so they remain
# discriminative at higher train multipliers — when val ties at 0
# across many configs the tie-break dominates and selection becomes
# unreliable on OOD test. Widening sometimes overlaps the test range
# (val and test sample independently with different seeds), but val
# is used only for hyperparameter selection so overlap is acceptable.
TM_TASKS = {
    'parity': dict(
        module=parity_tm,
        train_range=(3, 8), val_range=(10, 20), test_range=(16, 32),
        n_train=300, n_test=20, max_steps=200),
    'increment': dict(
        module=increment_tm,
        train_range=(1, 5), val_range=(5, 10), test_range=(8, 12),
        n_train=300, n_test=20, max_steps=200),
    'reverse': dict(
        module=reverse_tm,
        train_range=(3, 6), val_range=(6, 14), test_range=(10, 16),
        n_train=300, n_test=20, max_steps=10000),
    'binary_adder': dict(
        module='binary_adder',
        train_range=(0, 32), val_range=(32, 500), test_range=(0, 1000),
        n_train=200, n_test=10, max_steps=200_000),
    'shift_left': dict(
        module=shift_left_tm,
        train_range=(3, 8), val_range=(10, 20), test_range=(16, 32),
        n_train=300, n_test=20, max_steps=200),
    'bit_count_mod3': dict(
        module=bit_count_mod3_tm,
        train_range=(3, 8), val_range=(10, 20), test_range=(16, 32),
        n_train=300, n_test=20, max_steps=200),
    'anbn': dict(
        module=anbn_tm,
        train_range=(2, 10), val_range=(10, 18), test_range=(12, 24),
        n_train=300, n_test=20, max_steps=10000),
    'palindrome': dict(
        module=palindrome_tm,
        train_range=(3, 8), val_range=(8, 14), test_range=(10, 16),
        n_train=300, n_test=20, max_steps=10000),
    'subtraction': dict(
        module=subtraction_tm,
        train_range=(1, 5), val_range=(5, 9), test_range=(6, 10),
        n_train=300, n_test=20, max_steps=200_000),
}

# Apply the train multiplier and derive n_val = ceil(0.1 * n_train).
for cfg in TM_TASKS.values():
    cfg['n_train'] = cfg['n_train'] * TRAIN_MULTIPLIER
    cfg['n_val'] = max(1, cfg['n_train'] // 10)

SEEDS = dict(train=42, val=7, test=123)
TASK_ORDER = ['parity', 'increment', 'reverse', 'binary_adder',
              'shift_left', 'bit_count_mod3', 'anbn',
              'palindrome', 'subtraction']


# --------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------
def _simulate_binary_adder(cfg, variant):
    """Simulate train, val, test for binary_adder (special path)."""
    nr = (variant == 'noread')
    tr = simulate_random_binary_alphabet_adders(
        n_runs=cfg['n_train'], num_range=cfg['train_range'],
        max_steps=cfg['max_steps'], seed=SEEDS['train'])
    val = simulate_random_binary_alphabet_adders(
        n_runs=cfg['n_val'], num_range=cfg['val_range'],
        max_steps=cfg['max_steps'], seed=SEEDS['val'])
    te = simulate_random_binary_alphabet_adders(
        n_runs=cfg['n_test'], num_range=cfg['test_range'],
        max_steps=cfg['max_steps'], seed=SEEDS['test'])
    if nr:
        merged_se = dict(tr['symbol_encoding'])
        for src in (val['symbol_encoding'], te['symbol_encoding']):
            for k in src:
                if k not in merged_se: merged_se[k] = len(merged_se)
        merged_st = dict(tr['state_encoding'])
        for src in (val['state_encoding'], te['state_encoding']):
            for k in src:
                if k not in merged_st: merged_st[k] = len(merged_st)
        for d in (tr, val, te):
            d['runs'], _ = apply_noread_to_runs(
                d['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)
    return tr['runs'], val['runs'], te['runs']


def simulate_train_val_test(task_name, variant='original'):
    """Return (train_runs, val_runs, test_runs) under the canonical
    train/val/test split. Variant must be 'original' or 'noread'."""
    cfg = TM_TASKS[task_name]
    if cfg['module'] == 'binary_adder':
        return _simulate_binary_adder(cfg, variant)
    if cfg['module'] == 'dyck1':
        raise NotImplementedError("dyck1 has no canonical OOD val_range yet")
    nr = (variant == 'noread')
    module = cfg['module']
    tr = module.simulate(cfg['n_train'], cfg['train_range'],
                          max_steps=cfg['max_steps'],
                          seed=SEEDS['train'], noread=nr)
    val = module.simulate(cfg['n_val'], cfg['val_range'],
                           max_steps=cfg['max_steps'] * 2,
                           seed=SEEDS['val'], noread=nr)
    te = module.simulate(cfg['n_test'], cfg['test_range'],
                          max_steps=cfg['max_steps'] * 4,
                          seed=SEEDS['test'], noread=nr)
    return tr['runs'], val['runs'], te['runs']
