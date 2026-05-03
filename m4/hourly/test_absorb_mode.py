"""Sanity tests for the new GDC-TS terminal_behavior='absorb' option.

Tests:
  1. At alpha=1, theta=0 absorb mode: applying h transitions to a
     uniform initial distribution should leave mass at positions
     [h, n-1] equal to 1/n each (no diffusion smearing).  Total
     surviving mass should be (n-h)/n.
  2. At alpha=1, theta=0 diffuse mode (default): applying transitions
     spreads terminal mass uniformly -- check mass at non-terminal
     states grows due to terminal redistribution.
  3. M4 hourly equivalence: GDC-TS in absorb mode + forecast_gdc_style
     should give the same predictions as v5_gdc_proper.py's manual
     lookahead approach (within numerical precision).
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M4_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(M4_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, M4_ROOT)
sys.path.insert(0, ROOT)

from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries  # noqa: E402


def test_uniform_drain_at_alpha1_theta0():
    """At alpha=1, theta=0, absorb: uniform mass should drain through
    forward shifts; terminal mass should be lost rather than diffused."""
    n = 10
    values = np.arange(n).reshape(-1, 1).astype(float)
    gdc = GenerativeDenseChainTimeSeries(
        values, beta=0.01, alpha=1.0, theta=0.0,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform',
    )
    dist = np.ones(n) / n
    print("Initial uniform dist:", dist)
    print("Initial sum:", dist.sum())
    for h in range(1, 6):
        dist = gdc._transition(dist)
        print(f"After {h} transitions, sum={dist.sum():.4f}, "
              f"position-mass={dist.round(4).tolist()}")
    expected_sum_h5 = (n - 5) / n
    print(f"\nExpected surviving mass after 5 transitions = (n-h)/n = "
          f"{expected_sum_h5:.4f}")
    print(f"Actual surviving mass = {dist.sum():.4f}")
    assert abs(dist.sum() - expected_sum_h5) < 1e-9, \
        f"Mass conservation failed: {dist.sum()} vs {expected_sum_h5}"
    # Mass at positions [5, 9] (= [h, n-1]) should each be 1/n
    for pos in range(5, n):
        assert abs(dist[pos] - 1/n) < 1e-9, \
            f"Position {pos} has mass {dist[pos]}, expected {1/n}"
    # Mass at positions [0, 4] should be 0
    for pos in range(5):
        assert abs(dist[pos]) < 1e-9, \
            f"Position {pos} has mass {dist[pos]}, expected 0"
    print("PASS: uniform drain at alpha=1, theta=0 absorb mode")


def test_diffuse_vs_absorb_difference():
    """At alpha=1, theta=0, diffuse mode: terminal mass redistributes
    to all non-terminals; absorb mode: mass leaks."""
    n = 10
    values = np.arange(n).reshape(-1, 1).astype(float)
    g_d = GenerativeDenseChainTimeSeries(values, beta=0.01, alpha=1.0,
        theta=0.0, transition_type='self_loop',
        terminal_behavior='diffuse', initial_dist='uniform')
    g_a = GenerativeDenseChainTimeSeries(values, beta=0.01, alpha=1.0,
        theta=0.0, transition_type='self_loop',
        terminal_behavior='absorb', initial_dist='uniform')
    # Start with all mass at terminal
    dist = np.zeros(n); dist[n-1] = 1.0
    d_after = g_d._transition(dist)
    a_after = g_a._transition(dist)
    print(f"Diffuse mode (terminal mass redistributed): "
          f"{d_after.round(4).tolist()}, sum={d_after.sum():.4f}")
    print(f"Absorb mode (terminal mass lost): "
          f"{a_after.round(4).tolist()}, sum={a_after.sum():.4f}")
    # Diffuse: each non-terminal should have 1/(n-1) of the terminal mass
    expected_diffuse = np.full(n, 1/(n-1))
    expected_diffuse[n-1] = 0
    assert np.allclose(d_after, expected_diffuse, atol=1e-9), \
        "Diffuse mode: terminal mass not properly redistributed"
    # Absorb: nothing left (mass is purely lost)
    assert abs(a_after.sum()) < 1e-9, \
        f"Absorb mode: should have zero mass, got sum={a_after.sum()}"
    print("PASS: diffuse vs absorb behavior differs as expected")


def test_m4_equivalence():
    """Verify that GDC-TS + absorb + forecast_gdc_style matches
    v5_gdc_proper.py's manual lookahead on M4 hourly."""
    sys.path.insert(0, HERE)
    import data_loader as dl
    from v5_gdc_proper import gdc_proper_forecast
    from v0_basic_gdc import smape, H_HORIZON

    train_d = dl.load_train()
    test_d = dl.load_test()

    print("\n=== M4 hourly: GDC-TS absorb mode vs v5 manual lookahead ===")
    print(f"{'sid':>5s}  {'config':>22s}  "
          f"{'v5 manual':>10s}  {'GDC absorb':>11s}  {'diff':>9s}")
    for sid in ['H1', 'H50', 'H150', 'H300']:
        train = train_d[sid]; test = test_d[sid]
        for L in [24, 48, 168]:
            sigma_per_step = float(np.std(train)) * 0.10
            sigma_gdc = sigma_per_step * np.sqrt(L)
            beta = sigma_gdc ** 2

            # v5 manual lookahead
            pred_v5, _ = gdc_proper_forecast(train, window_len=L,
                                              sigma_frac=0.10, h=H_HORIZON)
            sm_v5 = smape(test, pred_v5)

            # New: GDC-TS in absorb mode + forecast_gdc_style
            states = train.reshape(-1, 1)
            gdc = GenerativeDenseChainTimeSeries(
                states, beta=beta, alpha=1.0, theta=0.0,
                transition_type='self_loop',
                terminal_behavior='absorb',
                initial_dist='uniform',
            )
            prime = train[-L:].reshape(-1, 1)
            forecasts, _ = gdc.forecast_gdc_style(prime, n_steps=H_HORIZON)
            pred_absorb = forecasts[:, 0]
            sm_absorb = smape(test, pred_absorb)
            diff = float(np.max(np.abs(pred_v5 - pred_absorb)))
            print(f"{sid:>5s}  L={L:>3d}, sigma%=0.10  "
                  f"{sm_v5:>9.4f}%  {sm_absorb:>10.4f}%  {diff:>9.2e}")


if __name__ == "__main__":
    test_uniform_drain_at_alpha1_theta0()
    print()
    test_diffuse_vs_absorb_difference()
    print()
    test_m4_equivalence()
