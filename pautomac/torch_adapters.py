"""GPU-batched PAutomaC scoring for GDC and Parrot.

Both classes expose the `fit / score_test_set` interface that
run_eval_parallel.py uses (via fit + score_test_set rather than the
per-sequence log_prob). Each does a stepped forward pass over the
test set, batched in chunks on GPU.

GDC: maintains a running per-test-sequence posterior over the chain
positions, advancing one observation at a time. Per-step cost
O(chunk_B × N) for the transition kernel (N = chain length).

Parrot: at each step, computes Hamming distance from the current
prefix's L-suffix to all training-corpus L-windows, takes top-K
nearest, averages their continuations. Per-step cost O(chunk_B × M × L)
for the Hamming distance (M = number of valid training L-windows).
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'hmm_comparison'))

from generative_dense_chain import GenerativeDenseChain
from gdc_torch_discrete import _self_loop_transition_batched

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

LOG_EPS = -700.0  # natural-log floor


def _append_end(seqs, end_token):
    return [np.concatenate([np.asarray(s, dtype=np.int64), [end_token]])
            for s in seqs]


def _pad_seqs(seqs, pad=-1):
    """Stack variable-length int sequences into (B, T_max) padded matrix
    with -1 padding marker."""
    B = len(seqs); T_max = max((len(s) for s in seqs), default=0)
    mat = np.full((B, T_max), pad, dtype=np.int64)
    for i, s in enumerate(seqs):
        mat[i, :len(s)] = s
    return mat


# --------------------------------------------------------------------
# GDC torch
# --------------------------------------------------------------------
@torch.no_grad()
def _gdc_score_chunk(
    sym_t, term_t, start_t, last_nt_idx, N,
    seq_chunk,                    # (B, T_max) int with -1 padding
    nA, alpha, theta, beta,
    terminal_behavior, initial_dist,
    device=DEVICE, dtype=DTYPE):
    """Score a chunk of test sequences. Returns (B,) np.float64 log probs."""
    B, T_max = seq_chunk.shape
    seq_t = torch.as_tensor(seq_chunk, device=device)

    non_terminal_mask = (~term_t).to(dtype)
    terminal_mask_f  = term_t.to(dtype)
    beta_nt = (1.0 - alpha - theta) / max(N - 2, 1)
    beta_t  = (1.0 - theta) / max(N - 1, 1)

    if initial_dist == 'uniform':
        d0 = torch.full((N,), 1.0 / N, dtype=dtype, device=device)
    else:
        starts_f = start_t.to(dtype)
        d0 = starts_f / starts_f.sum()
    dist = d0.unsqueeze(0).expand(B, N).contiguous()

    V_distinct = max(int(torch.unique(sym_t).numel()), 1)
    inv_V_beta = beta / V_distinct
    sym_clipped = torch.clamp(sym_t, max=nA - 1)
    symbol_onehot = torch.zeros((N, nA), dtype=dtype, device=device)
    symbol_onehot.scatter_(1, sym_clipped.unsqueeze(1), 1.0)

    log_probs = torch.zeros(B, dtype=dtype, device=device)

    for t in range(T_max):
        if t > 0:
            dist = _self_loop_transition_batched(
                dist, alpha, theta, beta_nt, beta_t,
                non_terminal_mask, terminal_mask_f, last_nt_idx,
                terminal_behavior)
        # symbol marginal under current dist
        sym_dist = dist @ symbol_onehot           # (B, nA)
        sym_sum  = sym_dist.sum(dim=1, keepdim=True)
        sym_dist = torch.where(sym_sum > 0, sym_dist / sym_sum,
                                torch.full_like(sym_dist, 1.0 / nA))

        obs = seq_t[:, t]                          # (B,)
        active = obs >= 0
        obs_safe = torch.clamp(obs, min=0)
        p = sym_dist.gather(1, obs_safe.unsqueeze(1)).squeeze(1)  # (B,)
        log_p = torch.log(torch.clamp(p, min=np.exp(LOG_EPS)))
        log_probs = log_probs + active.to(dtype) * log_p

        # Filter dist on observed obs (active rows only)
        match_ind = (sym_t.unsqueeze(0) == obs.unsqueeze(1)).to(dtype)
        if beta == 0.0:
            new_dist = dist * match_ind
            ssum = new_dist.sum(dim=1, keepdim=True)
            new_dist = torch.where(ssum > 0, new_dist / ssum,
                                    torch.full_like(new_dist, 1.0 / N))
        else:
            emission = (1.0 - beta) * match_ind + inv_V_beta
            unnorm = dist * emission
            ssum = unnorm.sum(dim=1, keepdim=True)
            new_dist = torch.where(ssum > 0, unnorm / ssum,
                                    torch.full_like(unnorm, 1.0 / N))
        active_f = active.to(dtype).unsqueeze(1)
        dist = active_f * new_dist + (1 - active_f) * dist

    return log_probs.detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def _dual_gdc_score_chunk(
    sym_t, term_t, start_t, last_nt_idx, N,
    seq_chunk,                    # (B, T_max) int with -1 padding
    nA, alpha_ctx, theta_ctx, alpha_fc, theta_fc, beta,
    terminal_behavior, initial_dist,
    device=DEVICE, dtype=DTYPE):
    """Dual-α PAutomaC scorer.

    Two transitions per step: α_fc applied for the predictive marginal,
    α_ctx applied for the state-tracking advance. Then filter dist on
    the observation. Setting α_fc==α_ctx and θ_fc==θ_ctx recovers the
    single-α scorer exactly.
    """
    B, T_max = seq_chunk.shape
    seq_t = torch.as_tensor(seq_chunk, device=device)

    non_terminal_mask = (~term_t).to(dtype)
    terminal_mask_f  = term_t.to(dtype)
    beta_nt_ctx = (1.0 - alpha_ctx - theta_ctx) / max(N - 2, 1)
    beta_t_ctx  = (1.0 - theta_ctx) / max(N - 1, 1)
    beta_nt_fc  = (1.0 - alpha_fc  - theta_fc ) / max(N - 2, 1)
    beta_t_fc   = (1.0 - theta_fc )            / max(N - 1, 1)

    if initial_dist == 'uniform':
        d0 = torch.full((N,), 1.0 / N, dtype=dtype, device=device)
    else:
        starts_f = start_t.to(dtype)
        d0 = starts_f / starts_f.sum()
    dist = d0.unsqueeze(0).expand(B, N).contiguous()

    V_distinct = max(int(torch.unique(sym_t).numel()), 1)
    inv_V_beta = beta / V_distinct
    sym_clipped = torch.clamp(sym_t, max=nA - 1)
    symbol_onehot = torch.zeros((N, nA), dtype=dtype, device=device)
    symbol_onehot.scatter_(1, sym_clipped.unsqueeze(1), 1.0)

    log_probs = torch.zeros(B, dtype=dtype, device=device)

    for t in range(T_max):
        if t > 0:
            pred_state = _self_loop_transition_batched(
                dist, alpha_fc, theta_fc, beta_nt_fc, beta_t_fc,
                non_terminal_mask, terminal_mask_f, last_nt_idx,
                terminal_behavior)
            dist = _self_loop_transition_batched(
                dist, alpha_ctx, theta_ctx, beta_nt_ctx, beta_t_ctx,
                non_terminal_mask, terminal_mask_f, last_nt_idx,
                terminal_behavior)
        else:
            pred_state = dist
        sym_dist = pred_state @ symbol_onehot         # (B, nA)
        sym_sum  = sym_dist.sum(dim=1, keepdim=True)
        sym_dist = torch.where(sym_sum > 0, sym_dist / sym_sum,
                                torch.full_like(sym_dist, 1.0 / nA))

        obs = seq_t[:, t]
        active = obs >= 0
        obs_safe = torch.clamp(obs, min=0)
        p = sym_dist.gather(1, obs_safe.unsqueeze(1)).squeeze(1)
        log_p = torch.log(torch.clamp(p, min=np.exp(LOG_EPS)))
        log_probs = log_probs + active.to(dtype) * log_p

        match_ind = (sym_t.unsqueeze(0) == obs.unsqueeze(1)).to(dtype)
        if beta == 0.0:
            new_dist = dist * match_ind
            ssum = new_dist.sum(dim=1, keepdim=True)
            new_dist = torch.where(ssum > 0, new_dist / ssum,
                                    torch.full_like(new_dist, 1.0 / N))
        else:
            emission = (1.0 - beta) * match_ind + inv_V_beta
            unnorm = dist * emission
            ssum = unnorm.sum(dim=1, keepdim=True)
            new_dist = torch.where(ssum > 0, unnorm / ssum,
                                    torch.full_like(unnorm, 1.0 / N))
        active_f = active.to(dtype).unsqueeze(1)
        dist = active_f * new_dist + (1 - active_f) * dist

    return log_probs.detach().cpu().numpy().astype(np.float64)


class TorchDualGDCModel:
    """Torch GPU GDC scorer with decoupled α_ctx / α_fc (PAutomaC).

    α_ctx, θ_ctx govern the state-tracking transition; α_fc, θ_fc govern
    the prediction-time transition applied before each emission scored.
    α_ctx == α_fc and θ_ctx == θ_fc recovers TorchGDCModel.
    """

    def __init__(self, alpha_ctx=0.5, alpha_fc=1.0,
                 theta_ctx=0.0, theta_fc=0.0, beta=0.0,
                 transition_type='self_loop',
                 initial_dist='sequence_starts',
                 terminal_behavior='diffuse',
                 chunk_size=256,
                 device=DEVICE, dtype=DTYPE):
        self.alpha_ctx = alpha_ctx; self.alpha_fc = alpha_fc
        self.theta_ctx = theta_ctx; self.theta_fc = theta_fc
        self.beta = beta
        self.transition_type = transition_type
        self.initial_dist = initial_dist
        self.terminal_behavior = terminal_behavior
        self.chunk_size = chunk_size
        self.device = device; self.dtype = dtype
        self.name = (f'tgdc2-ac{alpha_ctx}-af{alpha_fc}-'
                     f'tc{theta_ctx}-tf{theta_fc}-'
                     f'b{beta}-{terminal_behavior[:3]}')

    def fit(self, train_seqs, alphabet_size):
        # Reuses TorchGDCModel.fit logic by building a GDC with α_ctx
        # (just for chain construction; α is not used in fit).
        self.A = alphabet_size + 1
        self.end_token = alphabet_size
        seqs = _append_end(train_seqs, self.end_token)
        col_seqs = [s.reshape(-1, 1).astype(np.int64) for s in seqs
                    if len(s) > 0]
        gdc = GenerativeDenseChain(
            col_seqs, alpha=self.alpha_ctx, theta=self.theta_ctx,
            gamma=0.0, beta=self.beta,
            transition_type=self.transition_type,
            initial_dist=self.initial_dist,
            terminal_behavior=self.terminal_behavior)
        self.sym_t = torch.as_tensor(
            gdc.states[:, 0].astype(np.int64), device=self.device)
        self.term_t = torch.as_tensor(
            gdc.terminal_mask, device=self.device, dtype=torch.bool)
        self.start_t = torch.as_tensor(
            gdc.start_mask, device=self.device, dtype=torch.bool)
        self.N = int(self.sym_t.shape[0])
        nt_indices = torch.where(~self.term_t)[0]
        self.last_nt_idx = int(nt_indices[-1].item()) if len(nt_indices) > 0 else 0

    def score_test_set(self, test_seqs):
        seqs = _append_end(test_seqs, self.end_token)
        out = np.empty(len(seqs), dtype=np.float64)
        for start in range(0, len(seqs), self.chunk_size):
            chunk = seqs[start:start + self.chunk_size]
            mat = _pad_seqs(chunk)
            log_probs = _dual_gdc_score_chunk(
                self.sym_t, self.term_t, self.start_t, self.last_nt_idx,
                self.N, mat, self.A,
                self.alpha_ctx, self.theta_ctx,
                self.alpha_fc, self.theta_fc, self.beta,
                self.terminal_behavior, self.initial_dist,
                self.device, self.dtype)
            out[start:start + len(chunk)] = log_probs
        return out


class TorchGDCModel:
    """Torch GPU GDC scorer for PAutomaC."""

    def __init__(self, alpha=0.95, theta=0.05, beta=0.0,
                 transition_type='self_loop',
                 initial_dist='sequence_starts',
                 terminal_behavior='diffuse',
                 chunk_size=256,
                 device=DEVICE, dtype=DTYPE):
        self.alpha = alpha; self.theta = theta; self.beta = beta
        self.transition_type = transition_type
        self.initial_dist = initial_dist
        self.terminal_behavior = terminal_behavior
        self.chunk_size = chunk_size
        self.device = device; self.dtype = dtype
        self.name = (f'tgdc-a{alpha}-t{theta}-b{beta}-'
                     f'{terminal_behavior[:3]}')

    def fit(self, train_seqs, alphabet_size):
        self.A = alphabet_size + 1
        self.end_token = alphabet_size
        seqs = _append_end(train_seqs, self.end_token)
        col_seqs = [s.reshape(-1, 1).astype(np.int64) for s in seqs
                    if len(s) > 0]
        gdc = GenerativeDenseChain(
            col_seqs, alpha=self.alpha, theta=self.theta, gamma=0.0,
            beta=self.beta, transition_type=self.transition_type,
            initial_dist=self.initial_dist,
            terminal_behavior=self.terminal_behavior)
        self.sym_t = torch.as_tensor(
            gdc.states[:, 0].astype(np.int64), device=self.device)
        self.term_t = torch.as_tensor(
            gdc.terminal_mask, device=self.device, dtype=torch.bool)
        self.start_t = torch.as_tensor(
            gdc.start_mask, device=self.device, dtype=torch.bool)
        self.N = int(self.sym_t.shape[0])
        nt_indices = torch.where(~self.term_t)[0]
        self.last_nt_idx = int(nt_indices[-1].item()) if len(nt_indices) > 0 else 0

    def score_test_set(self, test_seqs):
        seqs = _append_end(test_seqs, self.end_token)
        out = np.empty(len(seqs), dtype=np.float64)
        for start in range(0, len(seqs), self.chunk_size):
            chunk = seqs[start:start + self.chunk_size]
            mat = _pad_seqs(chunk)
            log_probs = _gdc_score_chunk(
                self.sym_t, self.term_t, self.start_t, self.last_nt_idx,
                self.N, mat, self.A, self.alpha, self.theta, self.beta,
                self.terminal_behavior, self.initial_dist,
                self.device, self.dtype)
            out[start:start + len(chunk)] = log_probs
        return out


# --------------------------------------------------------------------
# Parrot torch
# --------------------------------------------------------------------
class TorchParrotModel:
    """Torch GPU Parrot (top-K Hamming nearest-prefix) scorer for PAutomaC."""

    def __init__(self, L=2, K=5, alpha_prior=1.0, chunk_size=256,
                 device=DEVICE, dtype=DTYPE):
        self.L = L; self.K = K; self.alpha_prior = alpha_prior
        self.chunk_size = chunk_size
        self.device = device; self.dtype = dtype
        self.name = f'tparrot-L{L}-K{K}-a{alpha_prior}'

    def fit(self, train_seqs, alphabet_size):
        self.A = alphabet_size + 1
        self.end_token = alphabet_size
        seqs = _append_end(train_seqs, self.end_token)
        # Build per-sequence windows (no cross-sequence spans), matching
        # the numpy DiscreteParrotPool convention.
        Ws = []
        next_continuations = []
        concat_for_marginal = []
        for s in seqs:
            s = np.asarray(s, dtype=np.int64).ravel()
            concat_for_marginal.append(s)
            if len(s) < self.L + 1:
                continue
            n_w = len(s) - self.L
            starts = np.arange(n_w)
            W = s[starts[:, None] + np.arange(self.L)[None, :]]
            cont = s[self.L:self.L + n_w]
            Ws.append(W); next_continuations.append(cont)
        concat = np.concatenate(concat_for_marginal) if concat_for_marginal \
                 else np.empty(0, dtype=np.int64)
        if Ws:
            windows = np.concatenate(Ws, axis=0)
            continuations = np.concatenate(next_continuations, axis=0)
            self.M = windows.shape[0]
            self.windows_t = torch.as_tensor(windows, device=self.device)
            self.continuations_t = torch.as_tensor(continuations,
                                                    device=self.device)
        else:
            self.M = 0
            self.windows_t = None
            self.continuations_t = None
        # Marginal: numpy uses fixed Laplace=1.0 for fallback marginal.
        counts = np.zeros(self.A, dtype=np.float64)
        if len(concat) > 0:
            for v in concat:
                counts[int(v)] += 1.0
        smoothed_marginal = counts + 1.0
        self.marginal_t = torch.as_tensor(
            smoothed_marginal / smoothed_marginal.sum(),
            device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _score_chunk(self, seq_chunk):
        B, T_max = seq_chunk.shape
        seq_t = torch.as_tensor(seq_chunk, device=self.device)
        log_probs = torch.zeros(B, dtype=self.dtype, device=self.device)
        K_eff = min(self.K, max(self.M, 1))

        for t in range(T_max):
            if t < self.L or self.M < 1:
                # Cold start: use marginal
                pred = self.marginal_t.unsqueeze(0).expand(B, self.A)
            else:
                # Build query (B, L) from last L positions of prefix
                query = seq_t[:, t - self.L:t]
                # Hamming distance: (B, M, L) → (B, M)
                # int8 to save memory for large M
                diff = (query.unsqueeze(1)
                        != self.windows_t.unsqueeze(0)).to(torch.int8)
                dist_mat = diff.sum(dim=2)
                # Top-K smallest distances per row
                _, topk_idx = torch.topk(dist_mat, K_eff,
                                          dim=1, largest=False)
                # Look up continuations and accumulate per-symbol counts
                topk_cont = self.continuations_t[topk_idx]   # (B, K_eff)
                counts = torch.zeros(B, self.A, dtype=self.dtype,
                                     device=self.device)
                ones = torch.ones_like(topk_cont, dtype=self.dtype)
                counts.scatter_add_(1, topk_cont, ones)
                counts = counts + self.alpha_prior
                pred = counts / counts.sum(dim=1, keepdim=True)

            obs = seq_t[:, t]
            active = obs >= 0
            obs_safe = torch.clamp(obs, min=0)
            p = pred.gather(1, obs_safe.unsqueeze(1)).squeeze(1)
            log_p = torch.log(torch.clamp(p, min=np.exp(LOG_EPS)))
            log_probs = log_probs + active.to(self.dtype) * log_p
        return log_probs.detach().cpu().numpy().astype(np.float64)

    def score_test_set(self, test_seqs):
        seqs = _append_end(test_seqs, self.end_token)
        out = np.empty(len(seqs), dtype=np.float64)
        for start in range(0, len(seqs), self.chunk_size):
            chunk = seqs[start:start + self.chunk_size]
            mat = _pad_seqs(chunk)
            out[start:start + len(chunk)] = self._score_chunk(mat)
        return out


__all__ = ['TorchGDCModel', 'TorchParrotModel']
