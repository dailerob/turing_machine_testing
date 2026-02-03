# -*- coding: utf-8 -*-
"""
GDC-TS Cross-Validation Testing Script

Uses GenerativeDenseChainTimeSeries with sequential transition (no-wrap, GDC-compatible).
Optimizes alpha and beta via cross-validation for each dataset,
then evaluates on the last 20% of data.

For each dataset:
- Starting alpha = 1 - (1 / num_training_samples)
- Starting beta = mean of |1-step differences| (GDC-TS uses beta as variance)
- Cross-validation tests percent variations around these starting points

Supports differencing order d (like original GDC): when d > 0, the generating
sequence and observations are differenced; forecasts are undifferenced for evaluation.
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
    """Reverse differencing to get forecasts in original scale. last_values[0] = last original, etc."""
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
    """
    Compute lower/upper bounds and means from forecast state distributions.
    If d > 0, bounds and means are undifferenced to original scale.
    """
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
    """
    Load time series datasets from statsmodels.

    Returns:
        dict: Dataset names as keys, values are (data, d, description).
    """
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
    """
    Starting alpha, beta, and context_length for GDC-TS.
    GDC-TS uses beta as variance; we set beta = mean(|differences|) for scale.
    """
    n = len(data)
    alpha = 1.0 - (1.0 / n)

    differences = np.diff(np.asarray(data, dtype=float).ravel())
    mean_abs_diff = np.abs(np.mean(differences))
    if mean_abs_diff == 0:
        mean_abs_diff = 1e-6
    # GDC-TS: beta = variance; use mean(|diff|) as scale (same as 1/precision in original GDC)
    beta = mean_abs_diff

    context_length = min(100, n)
    return alpha, beta, context_length


def generate_param_grid(
    alpha_start,
    beta_start,
    context_start,
    max_context,
    one_minus_alpha_multipliers=None,
    beta_multipliers=None,
    context_multipliers=None,
    n_alpha=5,
    n_beta=7,
    n_context=5,
):
    """Generate parameter grid using log-spaced multipliers."""
    if one_minus_alpha_multipliers is None:
        one_minus_alpha_multipliers = np.logspace(-1, 1, n_alpha)
    if beta_multipliers is None:
        beta_multipliers = np.logspace(-2, 2, n_beta)
    if context_multipliers is None:
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
    """
    Run GDC-style forecast (forward pass, zero last state, re-zero each step).
    Uses the model's transition_type; same zeroing for all types.
    Returns (forecasts, state_distributions, last_values) with forecasts in original scale if d > 0.
    obs_seq shape (T, 1); train_data_original is the original-scale training series for last_values.
    """
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
        states = train_diff.reshape(-1, 1)
        actual_context = min(context_length, n_train)
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
    min_train = max(50, n // 4)
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
    """Optimize alpha, beta, context_length via CV. Returns (best_alpha, best_beta, best_context, best_mae, results_df)."""
    data = np.asarray(data, dtype=float).ravel()
    max_context = len(data)
    alpha_values, beta_values, context_values = generate_param_grid(
        alpha_start, beta_start, context_start, max_context,
        n_alpha=7, n_beta=7, n_context=5,
    )
    one_minus_alpha_start = 1.0 - alpha_start
    total_combinations = len(alpha_values) * len(beta_values) * len(context_values)

    if verbose:
        print(f"  Testing {len(alpha_values)} alpha x {len(beta_values)} beta x {len(context_values)} context = {total_combinations} combinations")
        print(f"  (1-alpha) start: {one_minus_alpha_start:.6f}")
        print(f"  Alpha range: [{alpha_values.min():.6f}, {alpha_values.max():.6f}]")
        print(f"  Beta start: {beta_start:.6f}")
        print(f"  Beta range: [{beta_values.min():.6f}, {beta_values.max():.6f}]")
        print(f"  Context start: {context_start}")
        print(f"  Context range: [{context_values.min()}, {context_values.max()}]")

    results = []
    best_mae = np.inf
    best_alpha, best_beta, best_context = alpha_start, beta_start, context_start

    for alpha in alpha_values:
        for beta in beta_values:
            for context in context_values:
                mae = time_series_cross_validation(
                    data, d, alpha, beta, context, n_folds=n_folds, horizon=horizon
                )
                one_minus_alpha_mult = (1.0 - alpha) / one_minus_alpha_start
                beta_mult = beta / beta_start
                context_mult = context / context_start
                results.append({
                    "alpha": alpha,
                    "beta": beta,
                    "context": context,
                    "one_minus_alpha_mult": one_minus_alpha_mult,
                    "beta_mult": beta_mult,
                    "context_mult": context_mult,
                    "cv_mae": mae,
                })
                if mae < best_mae:
                    best_mae, best_alpha, best_beta, best_context = mae, alpha, beta, context

    results_df = pd.DataFrame(results)
    if verbose:
        best_one_minus_alpha_mult = (1.0 - best_alpha) / one_minus_alpha_start
        best_beta_mult = best_beta / beta_start
        best_context_mult = best_context / context_start
        print(f"  Best CV MAE: {best_mae:.4f}")
        print(f"  Best Alpha: {best_alpha:.6f} ((1-a) x{best_one_minus_alpha_mult:.2f} from start)")
        print(f"  Best Beta: {best_beta:.6f} (x{best_beta_mult:.2f} from start)")
        print(f"  Best Context: {best_context} (x{best_context_mult:.2f} from start)")
    return best_alpha, best_beta, best_context, best_mae, results_df


def evaluate_single(
    train_data,
    test_data,
    d: int,
    alpha: float,
    beta: float,
    context_length: int,
    horizon: int,
) -> dict:
    """Evaluate GDC-TS with given parameters. Returns metrics dict with forecasts, CI, MAE, RMSE, MASE, etc."""
    train_data = np.asarray(train_data, dtype=float).ravel()
    test_data = np.asarray(test_data, dtype=float).ravel()

    if d > 0:
        train_diff = _difference(train_data, d)
    else:
        train_diff = train_data

    n_train = len(train_diff)
    states = train_diff.reshape(-1, 1)
    actual_context = min(context_length, n_train)
    obs_seq = train_diff[-actual_context:].reshape(-1, 1)

    model = GenerativeDenseChainTimeSeries(
        sequences=states,
        beta=beta,
        alpha=alpha,
        transition_type="sequential",
    )
    forecasts, state_distributions, last_values = _forecast_gdc_ts(
        model, obs_seq, horizon, d, train_data
    )
    forecasts = np.asarray(forecasts).ravel()[:horizon]
    actual = test_data[:horizon]

    errors = forecasts - actual
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    naive_forecast = np.full(horizon, train_data[-1])
    naive_mae = np.mean(np.abs(naive_forecast - actual))
    mase = mae / naive_mae if naive_mae > 0 else np.inf

    state_values = model.states[:, 0]
    lower_bounds, upper_bounds, _ = calculate_confidence_intervals(
        model, state_distributions, state_values,
        confidence_level=0.95, last_values=last_values, d=d,
    )
    lower_bounds = np.asarray(lower_bounds).ravel()[:horizon]
    upper_bounds = np.asarray(upper_bounds).ravel()[:horizon]
    in_ci = (actual >= lower_bounds) & (actual <= upper_bounds)
    coverage = np.mean(in_ci) * 100

    # Context in original scale for plotting
    if d > 0:
        context_plot = train_data[-actual_context:]
    else:
        context_plot = obs_seq.ravel()

    print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, MASE: {mase:.4f}, CI Coverage: {coverage:.1f}%, Context: {actual_context}")

    return {
        "mae": mae,
        "rmse": rmse,
        "mase": mase,
        "ci_coverage": coverage,
        "forecasts": forecasts,
        "actual": actual,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "context": context_plot,
    }


def evaluate_with_optimized_params(
    name: str,
    data,
    d: int,
    description: str,
    train_ratio: float = 0.8,
    horizon: int = 20,
    n_cv_folds: int = 5,
) -> dict:
    """Optimize via CV, then evaluate on test set. Returns metrics dict."""
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)
    train_size = int(n * train_ratio)

    print(f"\n{'='*70}")
    print(f"Dataset: {name}")
    print(f"Description: {description}")
    print(f"{'='*70}")
    print(f"Total samples: {n}")
    print(f"Training samples: {train_size}")
    print(f"Test samples: {n - train_size}")
    print(f"Differencing order (d): {d}")

    train_data = data[:train_size]
    test_data = data[train_size : train_size + horizon]
    actual_horizon = min(horizon, len(test_data))
    if actual_horizon == 0:
        print("  Error: No test data available")
        return None

    alpha_start, beta_start, context_start = calculate_starting_params(train_data)
    print(f"\nStarting parameters:")
    print(f"  Alpha: {alpha_start:.6f}")
    print(f"  Beta: {beta_start:.6f}")
    print(f"  Context Length: {context_start}")

    print(f"\nOptimizing via {n_cv_folds}-fold cross-validation...")
    best_alpha, best_beta, best_context, cv_mae, cv_results = optimize_parameters(
        train_data, d, alpha_start, beta_start, context_start,
        n_folds=n_cv_folds, horizon=min(horizon, 10),
    )

    print(f"\n--- Baseline Evaluation (starting params) ---")
    baseline_metrics = evaluate_single(
        train_data, test_data, d, alpha_start, beta_start, context_start, actual_horizon
    )

    print(f"\n--- Optimized Evaluation (CV-tuned params) ---")
    optimized_metrics = evaluate_single(
        train_data, test_data, d, best_alpha, best_beta, best_context, actual_horizon
    )

    improvement = (baseline_metrics["mae"] - optimized_metrics["mae"]) / baseline_metrics["mae"] * 100
    print(f"\n--- Improvement ---")
    print(f"MAE reduction: {improvement:.1f}%")

    metrics = {
        "name": name,
        "description": description,
        "n_train": train_size,
        "n_test": actual_horizon,
        "d": d,
        "alpha_start": alpha_start,
        "beta_start": beta_start,
        "context_start": context_start,
        "alpha_opt": best_alpha,
        "beta_opt": best_beta,
        "context_opt": best_context,
        "cv_mae": cv_mae,
        "baseline_mae": baseline_metrics["mae"],
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_mase": baseline_metrics["mase"],
        "opt_mae": optimized_metrics["mae"],
        "opt_rmse": optimized_metrics["rmse"],
        "opt_mase": optimized_metrics["mase"],
        "opt_ci_coverage": optimized_metrics["ci_coverage"],
        "mae_improvement_pct": improvement,
        "forecasts": optimized_metrics["forecasts"],
        "actual": optimized_metrics["actual"],
        "lower_bounds": optimized_metrics["lower_bounds"],
        "upper_bounds": optimized_metrics["upper_bounds"],
        "context": optimized_metrics["context"],
        "cv_results": cv_results,
    }
    return metrics


def plot_cv_heatmap(cv_results, name, alpha_start, beta_start, context_start):
    """Plot CV results as alpha x beta heatmap (best context per cell)."""
    cv_results = cv_results.copy()
    cv_results["one_minus_alpha_mult_round"] = cv_results["one_minus_alpha_mult"].round(4)
    cv_results["beta_mult_round"] = cv_results["beta_mult"].round(4)
    pivot = cv_results.pivot_table(
        index="one_minus_alpha_mult_round",
        columns="beta_mult_round",
        values="cv_mae",
        aggfunc="min",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"x{x:.2f}" for x in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"x{x:.2f}" for x in pivot.index])
    ax.set_xlabel("Beta Multiplier")
    ax.set_ylabel("(1-Alpha) Multiplier")
    ax.set_title(f"CV MAE (best context): {name}\n(Start: a={alpha_start:.4f}, b={beta_start:.2f}, ctx={context_start})")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("CV MAE")
    best_idx = cv_results["cv_mae"].idxmin()
    best_row = cv_results.loc[best_idx]
    best_one_minus_alpha_mult = best_row["one_minus_alpha_mult_round"]
    best_beta_mult = best_row["beta_mult_round"]
    try:
        row_idx = list(pivot.index).index(best_one_minus_alpha_mult)
        col_idx = list(pivot.columns).index(best_beta_mult)
        ax.scatter([col_idx], [row_idx], marker="*", s=300, c="red", edgecolors="white", linewidths=2)
    except ValueError:
        pass
    plt.tight_layout()
    plt.show()


def plot_forecast_comparison(metrics: dict):
    """Plot forecast with optimized parameters and 95% CI."""
    if metrics is None:
        return
    context = metrics["context"]
    forecasts = metrics["forecasts"]
    actual = metrics["actual"]
    lower_bounds = metrics["lower_bounds"]
    upper_bounds = metrics["upper_bounds"]
    context_times = np.arange(len(context))
    forecast_times = np.arange(len(context), len(context) + len(forecasts))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(context_times, context, label="Observed (Context)", color="blue", linewidth=2)
    ax.plot(forecast_times, actual, label="Actual", color="green", linewidth=2, alpha=0.8)
    ax.plot(forecast_times, forecasts, label="GDC-TS Forecast (Optimized)", color="red", linewidth=2, linestyle="--", marker="o", markersize=4)
    ax.fill_between(forecast_times, lower_bounds, upper_bounds, alpha=0.3, color="red", label="95% CI")
    ax.axvline(x=len(context) - 0.5, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Value")
    ax.set_title(
        f"GDC-TS Forecast (CV-Optimized): {metrics['name']}\n"
        f"MAE={metrics['opt_mae']:.3f}, MASE={metrics['opt_mase']:.3f}, "
        f"Improvement={metrics['mae_improvement_pct']:.1f}%"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_summary_table(all_metrics: list) -> pd.DataFrame:
    """Summary table: baseline vs optimized."""
    rows = []
    for m in all_metrics:
        if m is not None:
            rows.append({
                "Dataset": m["name"],
                "d": m["d"],
                "a_opt": f"{m['alpha_opt']:.4f}",
                "b_opt": f"{m['beta_opt']:.2f}",
                "ctx_start": m["context_start"],
                "ctx_opt": m["context_opt"],
                "MAE_base": f"{m['baseline_mae']:.3f}",
                "MAE_opt": f"{m['opt_mae']:.3f}",
                "MASE_opt": f"{m['opt_mase']:.3f}",
                "Improv%": f"{m['mae_improvement_pct']:.1f}%",
            })
    return pd.DataFrame(rows)


def main():
    """Run cross-validated GDC-TS evaluation on statsmodels datasets."""
    print("=" * 70)
    print("GDC-TS CROSS-VALIDATION PARAMETER OPTIMIZATION")
    print("=" * 70)
    print("\nLoading datasets from statsmodels...")
    datasets = load_datasets()
    print(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    all_metrics = []
    for name, (data, d, description) in datasets.items():
        metrics = evaluate_with_optimized_params(
            name=name,
            data=data,
            d=d,
            description=description,
            train_ratio=0.8,
            horizon=20,
            n_cv_folds=5,
        )
        if metrics is not None:
            all_metrics.append(metrics)
            plot_cv_heatmap(
                metrics["cv_results"],
                name,
                metrics["alpha_start"],
                metrics["beta_start"],
                metrics["context_start"],
            )
            plot_forecast_comparison(metrics)
    print("\n" + "=" * 70)
    print("SUMMARY: BASELINE vs CROSS-VALIDATED OPTIMIZATION")
    print("=" * 70)
    summary_df = create_summary_table(all_metrics)
    print(summary_df.to_string(index=False))
    print("\n--- Overall Performance ---")
    baseline_mases = [m["baseline_mase"] for m in all_metrics if m is not None]
    opt_mases = [m["opt_mase"] for m in all_metrics if m is not None]
    improvements = [m["mae_improvement_pct"] for m in all_metrics if m is not None]
    if baseline_mases:
        print(f"Baseline Avg MASE: {np.mean(baseline_mases):.4f}")
        print(f"Optimized Avg MASE: {np.mean(opt_mases):.4f}")
        print(f"Average MAE Improvement: {np.mean(improvements):.1f}%")
        print(f"Datasets improved: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")
    return all_metrics


if __name__ == "__main__":
    results = main()
