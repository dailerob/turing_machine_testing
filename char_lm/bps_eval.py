"""Bits-per-symbol (BPS) scoring for the Dedieu et al. 2019 char-LM
benchmark (Table 4).

BPS = mean over t in [0..T-1] of -log_2 P(x_t | x_{<t}).

Each model exposes its own predictive interface; this module wraps
them all in a uniform `score_bps(model_kind, model, test_seq, ...)`
function. For the existing baselines (HPYLM, PPM-D, KN-3) we use
their `predict_distribution` API. For Parrot we use a smoothed
neighbour vote. For GDC we maintain a state distribution and at each
step compute the marginal predictive over emissions, then condition.

A small floor 1e-12 is added to every probability before log to keep
BPS finite when a model assigns zero (OOV) probability — affects only
edge cases and matches what the paper's Kneser-Ney smoothing does.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from discrete_hpylm import HPYLMPool         # noqa: E402
from discrete_ppm import PPMPool              # noqa: E402
from discrete_parrot import DiscreteParrotPool  # noqa: E402

PROB_FLOOR = 1e-12


def _bps_from_log2(log2_total, n):
    return float(log2_total / max(n, 1))


def score_bps_predict_distribution(model, test_seq, alphabet_size,
                                    chunk_size=4096, alpha_prior=0.0):
    """For models with predict_distribution(prefix, h=1, alpha_prior=...).

    HPYLM/PPM-D/KN-3 fit this. Walks the test sequence, computing
    predictive distribution from the prefix at each step.

    For computational efficiency we slide a max-depth window only —
    HPYLM/PPM-D/KN-3 only depend on the last `max_depth` chars.
    """
    log2_total = 0.0
    T = len(test_seq)
    # We use the trained model's own context window: pass full prefix;
    # the model will internally truncate to its depth. This is O(T).
    for t in range(T):
        prefix = test_seq[:t]
        probs = model.predict_distribution(
            prefix, h=1, alpha_prior=alpha_prior)
        c = int(test_seq[t])
        p = float(probs[c])
        log2_total += -np.log2(max(p, PROB_FLOOR))
    return _bps_from_log2(log2_total, T)


def score_bps_parrot(pool, test_seq, alphabet_size, K=25, alpha_prior=1.0):
    """Parrot BPS: K-NN over fixed-length windows; vote → smoothed dist."""
    log2_total = 0.0
    T = len(test_seq)
    for t in range(T):
        prefix = test_seq[:t]
        probs = pool.predict_distribution(
            prefix, h=1, K=K, alpha_prior=alpha_prior)
        c = int(test_seq[t])
        log2_total += -np.log2(max(float(probs[c]), PROB_FLOOR))
    return _bps_from_log2(log2_total, T)


def score_bps_gdc_dual_torch(train, test_seq, alphabet_size,
                                alpha_ctx, alpha_fc,
                                theta_ctx=0.0, theta_fc=0.0,
                                beta_fc=0.0, beta_ctx=0.0,
                                alpha_prior=0.001,
                                device=None, dtype=None):
    """GPU torch version of score_bps_gdc_dual. ~100x faster on large
    train/test pairs (mobydick, war-peace, calgary-book1).

    Layout: (V, N) one-hot emission matrix M so marginal[c] = (M @ dist)[c]
    is a single matvec per step. Transition and filter are O(N) per step.
    Total cost: O(V*N*T) — feasible on GPU for N up to ~1M, T up to ~2M.
    """
    import torch
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dtype is None:
        dtype = torch.float64
    V = alphabet_size
    N = int(len(train))
    T = int(len(test_seq))
    train_int = torch.as_tensor(np.asarray(train, dtype=np.int64),
                                  device=device)
    test_int = torch.as_tensor(np.asarray(test_seq, dtype=np.int64),
                                  device=device)
    # (V, N) one-hot emission table
    one_hot = torch.zeros((V, N), dtype=dtype, device=device)
    one_hot.scatter_(0, train_int.unsqueeze(0), 1.0)

    floor = alpha_prior / V if alpha_prior > 0 else 0.0

    # Precompute diffusion coefficients
    diffuse_ctx = (1.0 - alpha_ctx - theta_ctx) / max(N - 2, 1)
    diffuse_fc  = (1.0 - alpha_fc  - theta_fc ) / max(N - 2, 1)

    def transition_inplace(d, alpha, theta, diffuse):
        # Returns NEW tensor; doesn't modify d.
        shifted = torch.zeros_like(d)
        shifted[1:] = d[:N - 1]
        total = d.sum()
        return theta * d + alpha * shifted + diffuse * (total - d - shifted)

    dist = torch.zeros(N, dtype=dtype, device=device); dist[0] = 1.0
    log2_total = torch.zeros((), dtype=dtype, device=device)
    inv_V_fc = beta_fc / V
    inv_V_ctx = beta_ctx / V
    log2 = float(np.log(2.0))

    for t in range(T):
        if t > 0:
            pred_state = transition_inplace(dist, alpha_fc, theta_fc, diffuse_fc)
            dist = transition_inplace(dist, alpha_ctx, theta_ctx, diffuse_ctx)
        else:
            pred_state = dist
        # Marginal predictive: (V, N) @ (N,) = (V,)
        probs = one_hot @ pred_state
        if beta_fc > 0:
            total = pred_state.sum()
            probs = (1.0 - beta_fc) * probs + inv_V_fc * total
        if alpha_prior > 0:
            probs = probs + floor
        s = probs.sum()
        probs = probs / s if s.item() > 0 else torch.full_like(probs, 1.0 / V)
        c_obs = int(test_int[t].item())
        p_obs = float(probs[c_obs].item())
        log2_total = log2_total + (-np.log2(max(p_obs, PROB_FLOOR)))
        # Filter on observation: dist *= (one_hot[c_obs] if sharp else
        # (1-β)·match + β/V); then renormalize
        match = one_hot[c_obs]
        if beta_ctx > 0:
            new_dist = dist * (inv_V_ctx + (1.0 - beta_ctx) * match)
        else:
            new_dist = dist * match
        tot = new_dist.sum()
        if tot.item() > 0:
            dist = new_dist / tot
        else:
            dist = torch.full((N,), 1.0 / N, dtype=dtype, device=device)
    return float(log2_total.item()) / max(T, 1)


def score_bps_gdc_dual(train, test_seq, alphabet_size,
                          alpha_ctx, alpha_fc,
                          theta_ctx=0.0, theta_fc=0.0,
                          beta_fc=0.0, beta_ctx=0.0,
                          alpha_prior=0.001):
    """Dual-α GDC BPS scorer.

    At each test position t:
      - pred_state = transition(dist, alpha_fc, theta_fc) — used to compute
        the predictive distribution scored against test[t]
      - dist = transition(dist, alpha_ctx, theta_ctx) — advances the
        state-tracking distribution for the next step
      - filter dist on test[t] (Bayesian update with sharp emission;
        soft emission if beta_ctx > 0)

    The prediction-time operator (α_fc, θ_fc, β_fc) is decoupled from
    the state-tracking operator (α_ctx, θ_ctx, β_ctx). Setting α_fc=α_ctx
    and β_fc=β_ctx recovers the single-α scorer.

    Empirically on Blake (and likely text generally), the sweet spot is
    α_ctx ≈ 0.5, α_fc = 1.0, θ=0, β=0 — soft state-tracking + deterministic
    forecast.
    """
    N = len(train)
    train_int = np.asarray(train, dtype=np.int64)
    V = alphabet_size
    floor = alpha_prior / V if alpha_prior > 0 else 0.0
    em_masks = [np.where(train_int == c)[0] for c in range(V)]

    def transition(d, alpha, theta):
        if N <= 1:
            return d.copy()
        diffuse = (1.0 - alpha - theta) / max(N - 2, 1)
        shifted = np.zeros(N)
        shifted[1:] = d[:-1]
        total = d.sum()
        return theta * d + alpha * shifted + diffuse * (total - d - shifted)

    dist = np.zeros(N); dist[0] = 1.0
    log2_total = 0.0
    T = len(test_seq)
    for t in range(T):
        if t > 0:
            pred_state = transition(dist, alpha_fc, theta_fc)
            dist = transition(dist, alpha_ctx, theta_ctx)
        else:
            pred_state = dist
        probs = np.bincount(train_int, weights=pred_state,
                              minlength=V).astype(np.float64)
        if beta_fc > 0:
            total = float(pred_state.sum())
            probs = (1.0 - beta_fc) * probs + (beta_fc / V) * total
        if alpha_prior > 0:
            probs = probs + floor
        s = probs.sum()
        probs = probs / s if s > 0 else np.full(V, 1.0 / V)
        c_obs = int(test_seq[t])
        log2_total += -np.log2(max(float(probs[c_obs]), PROB_FLOOR))
        if beta_ctx > 0:
            new_dist = dist * (beta_ctx / V)
            m = em_masks[c_obs]
            if m.size:
                new_dist[m] += dist[m] * (1.0 - beta_ctx)
            tot = new_dist.sum()
            dist = new_dist / tot if tot > 0 else np.full(N, 1.0 / N)
        else:
            m = em_masks[c_obs]
            if m.size:
                matched = dist[m]
                tot = matched.sum()
                if tot > 0:
                    nd = np.zeros(N); nd[m] = matched / tot; dist = nd
                else:
                    dist = np.full(N, 1.0 / N)
            else:
                dist = np.full(N, 1.0 / N)
    return _bps_from_log2(log2_total, T)


def score_bps_gdc(gdc, test_seq, alphabet_size, alpha_prior=0.0):
    """GDC BPS: forward filtering with per-step predictive likelihood.

    Supports both beta=0 (delta emission) and beta>0 (Gaussian-style
    emission noise: P(obs|state) = (1-beta) if match else beta/V,
    matching the existing GDC `forward_pass` semantics).

    `alpha_prior > 0` adds a Laplace floor of `alpha_prior/V` to the
    final predictive probabilities (after the model's own smoothing).
    """
    n = gdc.n_states
    alpha = gdc.alpha
    theta = gdc.theta
    gamma = getattr(gdc, 'gamma', 0.0)
    beta = float(getattr(gdc, 'beta', 0.0))
    transition_type = gdc.transition_type
    V = alphabet_size
    floor = alpha_prior / V if alpha_prior > 0 else 0.0

    # Precompute state → emission table and per-emission masks
    state_emission = np.empty(n, dtype=np.int64)
    for key, idxs in gdc._state_to_indices.items():
        state_emission[idxs] = int(key[0])
    em_masks = [np.asarray(gdc._state_to_indices.get((c,), []),
                            dtype=np.int64)
                for c in range(V)]

    dist = gdc._get_initial_distribution()
    log2_total = 0.0
    T = len(test_seq)
    for t in range(T):
        if t > 0:
            dist = gdc._transition(dist, alpha, theta, gamma, transition_type)
        # Predictive over emissions:
        # P(x|history) = sum_i dist[i] * P(x|emit=state_emission[i])
        # = (1-beta) * marginal[x] + beta/V (when beta > 0)
        marginal = np.bincount(state_emission, weights=dist, minlength=V)
        if beta > 0:
            total_dist = float(dist.sum())
            probs = (1.0 - beta) * marginal + (beta / V) * total_dist
        else:
            probs = marginal.copy()
        if alpha_prior > 0:
            probs = probs + floor
        s = probs.sum()
        if s > 0:
            probs /= s
        else:
            probs = np.full(V, 1.0 / V)
        c_obs = int(test_seq[t])
        log2_total += -np.log2(max(float(probs[c_obs]), PROB_FLOOR))
        # Filter step: posterior over states given observation
        if beta > 0:
            # P(state|obs) ∝ dist[i] * ((1-beta) if emit==obs else 0 + beta/V)
            new_dist = dist * (beta / V)
            matching = em_masks[c_obs]
            if matching.size:
                new_dist[matching] += dist[matching] * (1.0 - beta)
            tot = new_dist.sum()
            dist = new_dist / tot if tot > 0 else np.full(n, 1.0 / n)
        else:
            matching = em_masks[c_obs]
            if matching.size:
                matched = dist[matching]
                tot = matched.sum()
                if tot > 0:
                    new_dist = np.zeros(n)
                    new_dist[matching] = matched / tot
                    dist = new_dist
                else:
                    dist = np.full(n, 1.0 / n)
            else:
                dist = np.full(n, 1.0 / n)
    return _bps_from_log2(log2_total, T)
