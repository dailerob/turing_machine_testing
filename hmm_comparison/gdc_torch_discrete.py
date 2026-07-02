"""PyTorch / GPU batched kernel for the *discrete* GDC, mirroring the
numpy `GenerativeDenseChain` for HMM-style scalar-emission prediction.

API:
    horizon_emission_many(
        symbol_of_state: (N,) int  -- the symbol associated with each state
        terminal_mask:   (N,) bool -- terminal positions (last of each seq)
        start_mask:      (N,) bool -- start positions (first of each seq)
        primes:          (B, L) int -- prefix observations per batch entry
        horizons:        list[int] -- horizons to evaluate
        nA:              int       -- alphabet size
        alpha, theta, beta: float
        transition_type: 'self_loop' (only one used by the HMM eval)
        terminal_behavior: 'diffuse' | 'absorb'
        initial_dist:    'uniform' | 'sequence_starts'
        device, dtype
    ) -> (B, len(horizons), nA) tensor of symbol-marginal predictions.

This mirrors the per-prefix loop:
    forward_pass(prefix) -> state_dist
    forecast(state_dist, h) -> state_dist_at_h
    marginalise to symbols -> nA vector
but batches over prefixes (B) and snapshots all requested horizons in one
forecast roll-out (length = max(horizons)).

Numerics: log-domain forward pass for stability with sharp emissions,
linear-domain transitions and forecast (state distributions sum to 1
and stay well-conditioned post-emission).
"""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np
import torch


def _self_loop_transition_batched(dist, alpha, theta, beta_nt, beta_t,
                                   non_terminal_mask, terminal_mask,
                                   last_nt_idx_per_seq, behavior):
    """Self-loop transition on (B, N) state distribution.

    Mirrors GenerativeDenseChain._transition_self_loop:
      out = self_loop + sequential + nt_diffusion + (t_diffusion if diffuse)
    where:
      self_loop  = theta * dist  (applies to ALL states)
      sequential = alpha * shifted   (shifted[i] = non_terminal_dist[i-1])
      nt_diffusion[i] = beta_nt * (sum_nt - nt[i] - shifted[i])
                        - beta_nt * non_terminal_dist[last_nt_idx]  if i==0 ... per seq
      t_diffusion[i]  = beta_t * (sum_t - terminal_dist[i])

    For multi-sequence training, the original numpy uses `last_nt_idx =
    np.where(~terminal_mask)[0][-1]` (only the GLOBAL last non-terminal),
    not per-sequence last-nt indices.  We reproduce that exactly: only
    the global last_nt's contribution is removed from state 0.

    beta_nt = (1 - alpha - theta) / (n - 2)  in 'diffuse' mode (diffusion
        spreads over n-1 non-terminals minus self minus shifted).  In
        'absorb' mode the original code uses the same beta_nt — diffusion
        among non-terminals stays the same, but t_diffusion is dropped
        and terminal mass leaks out.
    beta_t  = (1 - theta) / (n - 1)  for the diffuse-mode terminal share.
    """
    B, N = dist.shape
    nt = dist * non_terminal_mask                  # (B, N), zeros at terminals
    t  = dist * terminal_mask                      # (B, N), zeros elsewhere
    sum_nt = nt.sum(dim=1, keepdim=True)           # (B, 1)
    sum_t  = t.sum(dim=1, keepdim=True)            # (B, 1)
    shifted = torch.zeros_like(dist)
    shifted[:, 1:] = nt[:, :N - 1]                 # no wrap

    self_loop = theta * dist
    sequential = alpha * shifted
    nt_diffusion = beta_nt * (sum_nt - nt - shifted)
    # subtract the global-last-nt contribution from state 0
    last_nt_val = nt[:, last_nt_idx_per_seq]       # (B,)
    nt_diffusion[:, 0] = nt_diffusion[:, 0] - beta_nt * last_nt_val

    out = self_loop + sequential + nt_diffusion
    if behavior == 'diffuse':
        # terminal-mass diffusion to non-terminals
        t_diffusion = beta_t * (sum_t - t)
        out = out + t_diffusion
    return out


@torch.no_grad()
def horizon_emission_many(
    symbol_of_state,           # (N,) int — symbol for each state
    terminal_mask,             # (N,) bool
    start_mask,                # (N,) bool
    primes,                    # (B, L) int
    horizons,                  # list[int], all >= 1
    nA: int,
    alpha: float,
    theta: float,
    beta: float,
    transition_type: str = 'self_loop',
    terminal_behavior: str = 'diffuse',
    initial_dist: str = 'sequence_starts',
    beta_scaling: str = 'none',  # 'none' | 'linear' | 'sqrt' | 'asymptotic'
    alpha_forecast: float = None,  # if set, use during post-prefix forecast
    theta_forecast: float = None,  # if set, use during post-prefix forecast
    device: str = 'cuda',
    dtype: torch.dtype = torch.float64,
):
    """Batched forward+forecast, returning (B, n_horizons, nA) predictions."""
    if transition_type != 'self_loop':
        raise NotImplementedError("only self_loop is implemented in torch kernel")
    if terminal_behavior not in ('diffuse', 'absorb'):
        raise ValueError(terminal_behavior)
    if initial_dist not in ('uniform', 'sequence_starts'):
        raise ValueError(initial_dist)

    sym = torch.as_tensor(np.asarray(symbol_of_state, dtype=np.int64),
                          device=device)
    term = torch.as_tensor(np.asarray(terminal_mask, dtype=bool),
                           device=device)
    start = torch.as_tensor(np.asarray(start_mask, dtype=bool),
                            device=device)
    primes_t = torch.as_tensor(np.asarray(primes, dtype=np.int64),
                               device=device)
    N = sym.shape[0]
    B, L = primes_t.shape
    h_max = max(horizons)
    horizons_sorted = sorted(set(horizons))

    # Edge case: trivial state space
    if N <= 2:
        # Fall back to uniform predictions to mirror numpy behavior.
        out = torch.full((B, len(horizons), nA), 1.0 / nA,
                          dtype=dtype, device=device)
        return out

    non_terminal_mask = (~term).to(dtype)
    terminal_mask_f  = term.to(dtype)
    last_nt_idx = int(torch.where(~term)[0][-1].item())

    # Diffusion coefficients (match numpy code in self_loop branch)
    beta_nt = (1.0 - alpha - theta) / (N - 2)
    beta_t  = (1.0 - theta) / (N - 1)

    # Initial distribution.
    if initial_dist == 'uniform':
        d0 = torch.full((N,), 1.0 / N, dtype=dtype, device=device)
    else:
        starts_f = start.to(dtype)
        d0 = starts_f / starts_f.sum()
    dist = d0.unsqueeze(0).expand(B, N).contiguous()  # (B, N)

    # Vocabulary (number of distinct symbols actually present).
    V = int(sym.max().item()) + 1                # symbols indexed 0..V-1
    # Build state-to-symbol-onehot once (N, nA) so we can both:
    #   (a) compute exact-match indicator per (B, N) given a (B,) obs
    #   (b) marginalize state -> symbol distribution
    sym_clipped = torch.clamp(sym, max=nA - 1)   # symbols >= nA shouldn't happen
    symbol_onehot = torch.zeros((N, nA), dtype=dtype, device=device)
    symbol_onehot.scatter_(1, sym_clipped.unsqueeze(1), 1.0)

    # ---- Forward pass over L observations ----
    # Match numpy convention: noisy emission uses V (vocab) not nA;
    # vocab is "number of distinct observed symbols" but in HMM eval
    # symbols are 0..nA-1, V == number of distinct symbols present.
    V_distinct = int(torch.unique(sym).numel())
    if V_distinct < 1:
        V_distinct = 1
    # L-scaling of beta (discrete analog of σ·√L kernel-bandwidth trick).
    # Computed once per call; the prefix-length-scaled β is applied to
    # every emission step in the prefix, but NOT to the post-prefix
    # forecast roll-out (which keeps the original β).
    if beta_scaling == 'linear':
        beta_eff = min(beta * L, 1.0)
    elif beta_scaling == 'sqrt':
        beta_eff = min(beta * float(math.sqrt(L)), 1.0)
    elif beta_scaling == 'asymptotic':
        # 1 - (1-β)^L: probability that at least one Bernoulli(β) trial
        # fires in L steps. Approaches 1 smoothly without saturating.
        beta_eff = 1.0 - (1.0 - beta) ** L
    elif beta_scaling == 'none':
        beta_eff = beta
    else:
        raise ValueError(f"beta_scaling must be 'none', 'linear', "
                         f"'sqrt', or 'asymptotic'; got {beta_scaling!r}")
    inv_V_beta = beta_eff / V_distinct

    for t_step in range(L):
        if t_step > 0:
            dist = _self_loop_transition_batched(
                dist, alpha, theta, beta_nt, beta_t,
                non_terminal_mask, terminal_mask_f, last_nt_idx,
                terminal_behavior)
        obs = primes_t[:, t_step]                          # (B,) int
        # match indicator: sym[i] == obs[b] -> (B, N)
        match_ind = (sym.unsqueeze(0) == obs.unsqueeze(1)).to(dtype)
        if beta_eff == 0.0:
            # Deterministic emission: zero non-matching states then renormalize.
            new_dist = dist * match_ind
            ssum = new_dist.sum(dim=1, keepdim=True)
            # rows where ssum=0 -> reset to uniform
            zero_rows = (ssum.squeeze(1) == 0)
            new_dist = torch.where(ssum > 0, new_dist / ssum,
                                    torch.full_like(new_dist, 1.0 / N))
            dist = new_dist
        else:
            # P(obs|state) = (1-beta_eff)*match + beta_eff/V_distinct
            emission = (1.0 - beta_eff) * match_ind + inv_V_beta  # (B, N)
            unnorm = dist * emission
            ssum = unnorm.sum(dim=1, keepdim=True)
            new_dist = torch.where(ssum > 0, unnorm / ssum,
                                    torch.full_like(unnorm, 1.0 / N))
            dist = new_dist

    # ---- Forecast to all requested horizons (collecting symbol margins) ----
    # If forecast (α, θ) overrides are provided, recompute diffusion coeffs
    # for the forecast roll-out (e.g., α_forecast=1, θ_forecast=0 → pure
    # deterministic advance with no diffuse smoothing).
    a_fc = alpha if alpha_forecast is None else alpha_forecast
    t_fc = theta if theta_forecast is None else theta_forecast
    beta_nt_fc = (1.0 - a_fc - t_fc) / (N - 2)
    beta_t_fc  = (1.0 - t_fc) / (N - 1)
    out = torch.empty((B, len(horizons), nA), dtype=dtype, device=device)
    cur = dist
    next_h_idx = 0
    for h in range(1, h_max + 1):
        cur = _self_loop_transition_batched(
            cur, a_fc, t_fc, beta_nt_fc, beta_t_fc,
            non_terminal_mask, terminal_mask_f, last_nt_idx,
            terminal_behavior)
        if h in horizons_sorted:
            # Marginalize state-dist to symbols: (B, N) @ (N, nA) -> (B, nA)
            sym_dist = cur @ symbol_onehot
            # In absorb mode the state distribution may not sum to 1.
            # Renormalize the symbol marginal to match the numpy code.
            sym_sum = sym_dist.sum(dim=1, keepdim=True)
            sym_dist = torch.where(sym_sum > 0, sym_dist / sym_sum,
                                    torch.full_like(sym_dist, 1.0 / nA))
            # Place at every output position whose horizon matches h.
            for j, hj in enumerate(horizons):
                if hj == h:
                    out[:, j, :] = sym_dist
    return out
