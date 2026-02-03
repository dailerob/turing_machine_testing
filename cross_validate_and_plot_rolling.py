# -*- coding: utf-8 -*-
"""
Cross-Validate and Plot Rolling Forecasts

This script:
1. Loads time series datasets from statsmodels
2. Runs cross-validation on the first half of each dataset to optimize parameters
3. Generates rolling forecast plots one timestep at a time on the second half
   (similar to demo_gdc_time_series.py)

Each dataset will produce multiple plots showing the forecast at each timestep.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries


def _difference(seq: np.ndarray, d: int) -> np.ndarray:
    """Apply differencing d times. Returns 1D array."""
    result = np.asarray(seq, dtype=float).ravel()
    for _ in range(d):
        result = np.diff(result)
    return result


def _undifference(diff_forecasts: np.ndarray, last_values: list, d: int) -> np.ndarray:
    """Reverse differencing to get forecasts in original scale."""
    result = np.asarray(diff_forecasts, dtype=float).ravel()
    for i in range(d - 1, -1, -1):
        cumsum = np.cumsum(result)
        result = last_values[i] + cumsum
    return result


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
    state_values: np.ndarray,
    confidence_level: float = 0.95,
    last_values: list = None,
    d: int = 0,
) -> tuple:
    """Compute lower/upper bounds and means from forecast state distributions."""
    lower_quantile = (1 - confidence_level) / 2
    upper_quantile = 1 - lower_quantile
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
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    means = np.array(means)
    if d > 0 and last_values is not None:
        lower_bounds = _undifference(lower_bounds, last_values, d)
        upper_bounds = _undifference(upper_bounds, last_values, d)
        means = _undifference(means, last_values, d)
    return lower_bounds, upper_bounds, means


def load_datasets():
    """Load time series datasets from statsmodels."""
    import statsmodels.api as sm

    datasets = {}

    co2_data = sm.datasets.co2.load_pandas().data
    co2_values = co2_data["co2"].dropna().values
    datasets["co2"] = (co2_values, 1, "Mauna Loa CO2 (monthly, 1958-2001)")

    sunspots_data = sm.datasets.sunspots.load_pandas().data
    sunspots_values = sunspots_data["SUNACTIVITY"].values
    datasets["sunspots"] = (sunspots_values, 0, "Yearly Sunspot Numbers (1700-2008)")

    macro_data = sm.datasets.macrodata.load_pandas().data
    realgdp = macro_data["realgdp"].values
    datasets["macrodata"] = (realgdp, 1, "US Real GDP (quarterly, 1959-2009)")

    nile_data = sm.datasets.nile.load_pandas().data
    nile_values = nile_data["volume"].values
    datasets["nile"] = (nile_values, 0, "Annual Flow of the Nile (1871-1970)")

    elnino_data = sm.datasets.elnino.load_pandas().data
    elnino_values = elnino_data.iloc[:, 1:].mean(axis=1).values
    datasets["elnino"] = (elnino_values, 0, "El Nino SST (yearly average, 1950-2010)")

    try:
        from statsmodels.datasets import interest_inflation
        ii_data = interest_inflation.load_pandas().data
        inflation_values = ii_data.iloc[:, 0].values
        datasets["interest_inflation"] = (inflation_values, 1, "Interest/Inflation Rates")
    except Exception as e:
        print(f"Could not load interest_inflation dataset: {e}")

    return datasets


def calculate_starting_params(data: np.ndarray):
    """Calculate starting alpha, beta, and context_length for GDC-TS."""
    n = len(data)
    alpha = 1.0 - (1.0 / n)

    differences = np.diff(np.asarray(data, dtype=float).ravel())
    mean_abs_diff = np.abs(np.mean(differences))
    if mean_abs_diff == 0:
        mean_abs_diff = 1e-6
    beta = mean_abs_diff

    context_length = min(100, n)
    return alpha, beta, context_length


def generate_param_grid(
    alpha_start,
    beta_start,
    context_start,
    max_context,
    n_alpha=7,
    n_beta=7,
    n_context=5,
):
    """Generate parameter grid using log-spaced multipliers."""
    one_minus_alpha_multipliers = np.logspace(-2, 1, n_alpha)
    beta_multipliers = np.logspace(0, 2, n_beta)
    context_multipliers = np.logspace(-0.6, 0.6, n_context)

    one_minus_alpha_start = 1.0 - alpha_start
    one_minus_alpha_values = one_minus_alpha_start * np.array(one_minus_alpha_multipliers)
    alpha_values = 1.0 - one_minus_alpha_values
    alpha_values = np.clip(alpha_values, 0.5, 0.9999)

    beta_values = beta_start * np.array(beta_multipliers)
    beta_values = np.maximum(beta_values, 1e-6)

    context_values = context_start * np.array(context_multipliers)
    context_values = np.clip(context_values, 10, max_context)
    context_values = np.unique(np.round(context_values).astype(int))

    return alpha_values, beta_values, context_values


def _forecast_gdc_ts(
    model: GenerativeDenseChainTimeSeries,
    obs_seq: np.ndarray,
    horizon: int,
    d: int,
    train_data_original: np.ndarray,
) -> tuple:
    """Run GDC-style forecast and return forecasts in original scale."""
    forecasts, state_distributions = model.forecast_gdc_style(obs_seq, horizon)
    forecasts_diff = np.asarray(forecasts).reshape(-1, model.states.shape[1])[:, 0]

    if d == 0:
        return forecasts_diff, state_distributions, None

    last_values = []
    diff_series = np.asarray(train_data_original, dtype=float).ravel()
    for _ in range(d):
        last_values.append(float(diff_series[-1]))
        diff_series = np.diff(diff_series)
    forecasts = _undifference(forecasts_diff, last_values, d)
    return forecasts, state_distributions, last_values


def cross_validate_fold(
    train_data,
    val_data,
    d: int,
    alpha: float,
    beta: float,
    context_length: int,
    horizon: int,
) -> float:
    """Evaluate a single CV fold. Returns MAE or np.inf on error."""
    try:
        train_data = np.asarray(train_data, dtype=float).ravel()
        val_data = np.asarray(val_data, dtype=float).ravel()

        if d > 0:
            train_diff = _difference(train_data, d)
        else:
            train_diff = train_data

        n_train = len(train_diff)
        actual_context = min(context_length, n_train - 1)

        model_data = train_diff[:-actual_context] if actual_context > 0 else train_diff
        if len(model_data) < 1:
            return np.inf

        states = model_data.reshape(-1, 1)
        obs_seq = train_diff[-actual_context:].reshape(-1, 1)

        actual_horizon = min(horizon, len(val_data))
        if actual_horizon == 0:
            return np.inf

        model = GenerativeDenseChainTimeSeries(
            sequences=states,
            beta=beta,
            alpha=alpha,
            transition_type="sequential",
        )
        forecasts, _, _ = _forecast_gdc_ts(
            model, obs_seq, actual_horizon, d, train_data
        )
        forecasts = np.asarray(forecasts).ravel()[:actual_horizon]
        actual = val_data[:actual_horizon]
        mae = np.mean(np.abs(forecasts - actual))
        return mae
    except Exception:
        return np.inf


def time_series_cross_validation(
    data,
    d: int,
    alpha: float,
    beta: float,
    context_length: int,
    n_folds: int = 5,
    val_size=None,
    horizon: int = 10,
) -> float:
    """Expanding-window time series CV. Returns mean MAE across folds."""
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)

    if val_size is None:
        available = n // 2
        val_size = max(horizon, available // n_folds)
    min_train = max(10, n // 4)
    fold_maes = []

    for fold in range(n_folds):
        val_start = min_train + fold * val_size
        val_end = val_start + val_size
        if val_end > n:
            break
        train_fold = data[:val_start]
        val_fold = data[val_start:val_end]
        if len(train_fold) < min_train or len(val_fold) < horizon:
            continue
        mae = cross_validate_fold(
            train_fold, val_fold, d, alpha, beta, context_length, horizon
        )
        if np.isfinite(mae):
            fold_maes.append(mae)

    if len(fold_maes) == 0:
        return np.inf
    return np.mean(fold_maes)


def optimize_parameters(
    data,
    d: int,
    alpha_start: float,
    beta_start: float,
    context_start: int,
    n_folds: int = 5,
    horizon: int = 10,
    verbose: bool = True,
):
    """Optimize alpha, beta, context_length via CV."""
    data = np.asarray(data, dtype=float).ravel()
    max_context = len(data)
    alpha_values, beta_values, context_values = generate_param_grid(
        alpha_start, beta_start, context_start, max_context,
        n_alpha=7, n_beta=14, n_context=5,
    )
    one_minus_alpha_start = 1.0 - alpha_start
    total_combinations = len(alpha_values) * len(beta_values) * len(context_values)

    if verbose:
        print(f"  Testing {len(alpha_values)} alpha x {len(beta_values)} beta x {len(context_values)} context = {total_combinations} combinations")
        print(f"  Alpha range: [{alpha_values.min():.6f}, {alpha_values.max():.6f}]")
        print(f"  Beta range: [{beta_values.min():.6f}, {beta_values.max():.6f}]")
        print(f"  Context range: [{context_values.min()}, {context_values.max()}]")

    best_mae = np.inf
    best_alpha, best_beta, best_context = alpha_start, beta_start, context_start

    for alpha in alpha_values:
        for beta in beta_values:
            for context in context_values:
                mae = time_series_cross_validation(
                    data, d, alpha, beta, context, n_folds=n_folds, horizon=horizon
                )
                if mae < best_mae:
                    best_mae, best_alpha, best_beta, best_context = mae, alpha, beta, context

    if verbose:
        best_one_minus_alpha_mult = (1.0 - best_alpha) / one_minus_alpha_start
        best_beta_mult = best_beta / beta_start
        best_context_mult = best_context / context_start
        print(f"  Best CV MAE: {best_mae:.4f}")
        print(f"  Best Alpha: {best_alpha:.6f} ((1-a) x{best_one_minus_alpha_mult:.2f} from start)")
        print(f"  Best Beta: {best_beta:.6f} (x{best_beta_mult:.2f} from start)")
        print(f"  Best Context: {best_context} (x{best_context_mult:.2f} from start)")

    return best_alpha, best_beta, best_context, best_mae


def run_rolling_forecast_with_plots(
    name: str,
    data: np.ndarray,
    d: int,
    description: str,
    alpha: float,
    beta: float,
    context_length: int,
    train_end_idx: int,
    horizon: int = 20,
    num_plots: int = None,
):
    """
    Generate rolling forecast plots one timestep at a time.
    
    For each timestep in the second half:
    - Use data up to that point as training/context
    - Forecast 'horizon' steps ahead
    - Plot observed context, forecast, actual future, and confidence intervals
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)
    
    # Determine how many plots to generate
    max_plots = n - train_end_idx - horizon
    if max_plots <= 0:
        print(f"  Not enough data for rolling forecasts. Need at least {horizon} steps after training.")
        return
    
    if num_plots is None:
        num_plots = max_plots
    else:
        num_plots = min(num_plots, max_plots)
    
    print(f"\n  Generating {num_plots} rolling forecast plots...")
    
    for plot_idx in range(num_plots):
        current_idx = train_end_idx + plot_idx
        
        # Data up to current point for training/context
        train_data = data[:current_idx]
        
        # Future data for comparison
        future_end = min(current_idx + horizon, n)
        future_data = data[current_idx:future_end]
        actual_horizon = len(future_data)
        
        if actual_horizon == 0:
            break
        
        # Apply differencing if needed
        if d > 0:
            train_diff = _difference(train_data, d)
        else:
            train_diff = train_data
        
        n_train = len(train_diff)
        actual_context = min(context_length, n_train - 1)
        
        # Split into model data and context
        model_data = train_diff[:-actual_context] if actual_context > 0 else train_diff
        if len(model_data) < 1:
            continue
        
        states = model_data.reshape(-1, 1)
        obs_seq_diff = train_diff[-actual_context:].reshape(-1, 1)
        
        # Build model and forecast
        model = GenerativeDenseChainTimeSeries(
            sequences=states,
            beta=beta,
            alpha=alpha,
            transition_type="sequential",
        )
        
        forecasts, state_distributions, last_values = _forecast_gdc_ts(
            model, obs_seq_diff, actual_horizon, d, train_data
        )
        forecasts = np.asarray(forecasts).ravel()[:actual_horizon]
        
        # Calculate confidence intervals
        state_values = model.states[:, 0]
        lower_bounds, upper_bounds, _ = calculate_confidence_intervals(
            model, state_distributions, state_values,
            confidence_level=0.95, last_values=last_values, d=d,
        )
        lower_bounds = np.asarray(lower_bounds).ravel()[:actual_horizon]
        upper_bounds = np.asarray(upper_bounds).ravel()[:actual_horizon]
        
        # Get context in original scale for plotting
        if d > 0:
            context_plot = train_data[-actual_context:]
        else:
            context_plot = obs_seq_diff.ravel()
        
        # Create plot
        context_times = np.arange(len(context_plot))
        forecast_times = np.arange(len(context_plot), len(context_plot) + actual_horizon)
        
        plt.figure(figsize=(10, 5))
        plt.plot(
            context_times,
            context_plot,
            label="Observed (Context)",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            forecast_times,
            forecasts,
            label="Forecast",
            linestyle="--",
            marker="o",
            color="red",
            markersize=4,
            linewidth=2,
        )
        plt.plot(
            forecast_times,
            future_data,
            label="Actual Future",
            color="green",
            linewidth=2,
            alpha=0.7,
        )
        plt.fill_between(
            forecast_times,
            lower_bounds,
            upper_bounds,
            alpha=0.3,
            color="red",
            label="95% Confidence Interval",
        )
        
        # Calculate MAE for this forecast
        mae = np.mean(np.abs(forecasts - future_data))
        
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(
            f"{name}: Rolling Forecast (idx={current_idx}, MAE={mae:.3f})\n"
            f"α={alpha:.4f}, β={beta:.4f}, ctx={actual_context}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def process_dataset(
    name: str,
    data: np.ndarray,
    d: int,
    description: str,
    horizon: int = 20,
    n_cv_folds: int = 5,
    num_rolling_plots: int = 10,
):
    """
    Process a single dataset:
    1. Cross-validate on first half to find optimal parameters
    2. Generate rolling forecast plots on second half
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)
    
    # Use first half for cross-validation
    train_end_idx = n // 2
    train_data = data[:train_end_idx]
    
    print(f"\n{'='*70}")
    print(f"Dataset: {name}")
    print(f"Description: {description}")
    print(f"{'='*70}")
    print(f"Total samples: {n}")
    print(f"First half (CV training): {train_end_idx} samples")
    print(f"Second half (rolling forecasts): {n - train_end_idx} samples")
    print(f"Differencing order (d): {d}")
    
    # Calculate starting parameters from first half
    alpha_start, beta_start, context_start = calculate_starting_params(train_data)
    print(f"\nStarting parameters:")
    print(f"  Alpha: {alpha_start:.6f}")
    print(f"  Beta: {beta_start:.6f}")
    print(f"  Context Length: {context_start}")
    
    # Cross-validate on first half
    print(f"\nOptimizing via {n_cv_folds}-fold cross-validation on first half...")
    best_alpha, best_beta, best_context, cv_mae = optimize_parameters(
        train_data, d, alpha_start, beta_start, context_start,
        n_folds=n_cv_folds, horizon=min(horizon, 10),
    )
    
    # Generate rolling forecast plots on second half
    run_rolling_forecast_with_plots(
        name=name,
        data=data,
        d=d,
        description=description,
        alpha=best_alpha,
        beta=best_beta,
        context_length=best_context,
        train_end_idx=train_end_idx,
        horizon=horizon,
        num_plots=num_rolling_plots,
    )
    
    return {
        "name": name,
        "description": description,
        "n_total": n,
        "n_train": train_end_idx,
        "d": d,
        "alpha_opt": best_alpha,
        "beta_opt": best_beta,
        "context_opt": best_context,
        "cv_mae": cv_mae,
    }


def main():
    """Run cross-validated optimization and rolling forecast plots on all datasets."""
    print("=" * 70)
    print("CROSS-VALIDATE AND PLOT ROLLING FORECASTS")
    print("=" * 70)
    print("\nThis script:")
    print("1. Loads time series datasets from statsmodels")
    print("2. Cross-validates on the FIRST HALF of each dataset")
    print("3. Generates rolling forecast plots on the SECOND HALF")
    print("\nLoading datasets...")
    
    datasets = load_datasets()
    print(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    
    # Configuration
    horizon = 30  # Forecast horizon
    n_cv_folds = 20  # Number of CV folds
    num_rolling_plots = 100  # Number of rolling forecast plots per dataset
    
    all_results = []
    
    for name, (data, d, description) in datasets.items():
        result = process_dataset(
            name=name,
            data=data,
            d=d,
            description=description,
            horizon=horizon,
            n_cv_folds=n_cv_folds,
            num_rolling_plots=num_rolling_plots,
        )
        all_results.append(result)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF OPTIMIZED PARAMETERS")
    print("=" * 70)
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "Dataset": r["name"],
            "d": r["d"],
            "n_train": r["n_train"],
            "alpha": f"{r['alpha_opt']:.4f}",
            "beta": f"{r['beta_opt']:.4f}",
            "context": r["context_opt"],
            "CV_MAE": f"{r['cv_mae']:.4f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    return all_results


if __name__ == "__main__":
    results = main()
