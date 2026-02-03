# -*- coding: utf-8 -*-
"""
Demo script for GenerativeDenseChainTimeSeries: load test data, fit, forecast,
and plot with confidence intervals (matching the style of the earlier GDC example).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries


def _interpolate_quantile(values: np.ndarray, probs: np.ndarray, quantile: float) -> float:
    """Compute an interpolated quantile from a discrete distribution."""
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_probs = probs[sorted_indices]
    cdf = np.cumsum(sorted_probs)
    if quantile <= cdf[0]:
        return float(sorted_values[0])
    if quantile >= cdf[-1]:
        return float(sorted_values[-1])
    idx = np.searchsorted(cdf, quantile)
    if idx == 0:
        return float(sorted_values[0])
    lower_cdf = cdf[idx - 1]
    upper_cdf = cdf[idx]
    lower_val = sorted_values[idx - 1]
    upper_val = sorted_values[idx]
    if upper_cdf == lower_cdf:
        return float(lower_val)
    t = (quantile - lower_cdf) / (upper_cdf - lower_cdf)
    return float(lower_val + t * (upper_val - lower_val))


def calculate_confidence_intervals(
    model: GenerativeDenseChainTimeSeries,
    state_distributions: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple:
    """
    Compute lower/upper bounds and means from forecast state distributions.
    state_distributions shape: (horizon, n_states).
    """
    lower_quantile = (1 - confidence_level) / 2
    upper_quantile = 1 - lower_quantile
    # Univariate: use first (and only) dimension of states
    state_values = model.states[:, 0]
    lower_bounds = []
    upper_bounds = []
    means = []
    for dist in state_distributions:
        total = np.sum(dist)
        if total > 0:
            probs = dist / total
            mean_val = np.dot(probs, state_values)
            means.append(mean_val)
            lower_bounds.append(_interpolate_quantile(state_values, probs, lower_quantile))
            upper_bounds.append(_interpolate_quantile(state_values, probs, upper_quantile))
        else:
            means.append(0.0)
            lower_bounds.append(0.0)
            upper_bounds.append(0.0)
    return lower_bounds, upper_bounds, means


def main():
    # Load test data (same as example)
    test = pd.read_csv("test.csv")
    windspeed = test.values * 5
    # Univariate: use first column, ensure 1D then (T, 1) for model
    if windspeed.ndim > 1:
        windspeed = windspeed[:, 0]
    windspeed = np.asarray(windspeed, dtype=float)

    test_break = 1020
    train_length = 1000
    context_length = 110
    horizon = 100
    num_plots = 1000  # number of rolling forecast plots (like original script)

    alpha = 0.9999
    beta = 1

    for i in range(num_plots):
        test_break += 1

        # Generating sequence = training window as state means (each row is one state)
        generating_seq = windspeed[test_break - train_length : test_break]
        # States must be (n_states, k); univariate -> (train_length, 1)
        states = generating_seq.reshape(-1, 1)
        obs_seq = generating_seq.copy()[-context_length:].reshape(-1, 1)
        future_seq = windspeed[test_break : test_break + horizon]

        # Sequential transition (no wrap, factor (1-alpha)/n). Same zeroing for all transition types.
        # Note: original GDC uses beta as precision (variance=1/beta); GDC-TS uses beta as variance.
        model = GenerativeDenseChainTimeSeries(
            sequences=states,
            beta=beta,
            alpha=alpha,
            transition_type="sequential",
        )

        # GDC-style forecast: forward pass, zero last state, re-zero after each step.
        forecasts, state_distributions = model.forecast_gdc_style(obs_seq, horizon)
        forecasts = forecasts[:, 0]

        lower_bounds, upper_bounds, _ = calculate_confidence_intervals(
            model, state_distributions, confidence_level=0.95
        )
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        forecast_timesteps = np.arange(len(obs_seq), len(obs_seq) + horizon)

        # Plot (same style as example univariate plot)
        plt.figure(figsize=(10, 5))
        plt.plot(
            np.arange(len(obs_seq)),
            obs_seq[:, 0],
            label="Observed",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            forecast_timesteps,
            forecasts,
            label="Forecast",
            linestyle="--",
            marker="o",
            color="red",
            markersize=4,
            linewidth=2,
        )
        plt.plot(
            forecast_timesteps,
            future_seq,
            label="Actual Future",
            color="green",
            linewidth=2,
            alpha=0.7,
        )
        plt.fill_between(
            forecast_timesteps,
            lower_bounds,
            upper_bounds,
            alpha=0.3,
            color="red",
            label="95% Confidence Interval",
        )
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(
            f"Forecast using GenerativeDenseChainTimeSeries (test_break={test_break})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
