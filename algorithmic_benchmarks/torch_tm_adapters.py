"""GPU-batched scorers for the TM benchmark eval pattern.

The TM eval predicts the next reduced-alphabet tuple at each position
of each test tape, restricted to candidates whose tuple[0] (read symbol)
matches the actual next read. This module provides two scorers
(`TorchTMParrot`, `TorchTMGDC`) that batch all positions in a single
test tape into one GPU call instead of looping per-position in Python.

Each scorer exposes:
    fit(train_seqs_in_reduced_alphabet, alphabet_size, ...)
    score_tape(tape_ids, actual_next_reads, by_read) -> np.ndarray of
        argmax tuple ids, one per position predicted.

Verified against numpy DiscreteParrotPool / GenerativeDenseChain on
a handful of test tapes (max-abs argmax disagreement = 0 for GDC and
small-K parrot in non-tied cases; ties broken by torch's topk vs
numpy's argpartition, but these only affect predictions in cases
where multiple tuples share the same max).
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'hmm_comparison'))

from generative_dense_chain import GenerativeDenseChain
from gdc_torch_discrete import _self_loop_transition_batched

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32


# --------------------------------------------------------------------
# Parrot torch for TM
# --------------------------------------------------------------------
class TorchTMParrot:
    """Batched Hamming top-K nearest-prefix lookup, masked-argmax per
    position. Mirrors DiscreteParrotPool.predict_argmax with mask."""

    def __init__(self, L=4, K=5, alpha_prior=1.0, device=DEVICE,
                 dtype=DTYPE):
        self.L = L; self.K = K; self.alpha_prior = alpha_prior
        self.device = device; self.dtype = dtype

    def fit(self, train_seqs, alphabet_size):
        """train_seqs: list of int sequences (reduced-alphabet tuple ids)."""
        self.A = int(alphabet_size)
        Ws, conts = [], []
        concat = []
        for s in train_seqs:
            s = np.asarray(s, dtype=np.int64).ravel()
            concat.append(s)
            if len(s) < self.L + 1:
                continue
            n_w = len(s) - self.L
            starts = np.arange(n_w)
            W = s[starts[:, None] + np.arange(self.L)[None, :]]
            cont = s[self.L:self.L + n_w]
            Ws.append(W); conts.append(cont)
        concat_arr = (np.concatenate(concat) if concat
                      else np.empty(0, dtype=np.int64))
        if Ws:
            windows = np.concatenate(Ws, axis=0)
            continuations = np.concatenate(conts, axis=0)
            self.M = int(windows.shape[0])
            self.windows_t = torch.as_tensor(windows, device=self.device)
            self.continuations_t = torch.as_tensor(continuations,
                                                    device=self.device)
        else:
            self.M = 0
            self.windows_t = None
            self.continuations_t = None
        # Marginal: numpy parrot uses fixed Laplace=1.0 for fallback marginal
        counts = np.zeros(self.A, dtype=np.float64)
        for v in concat_arr:
            if 0 <= int(v) < self.A:
                counts[int(v)] += 1.0
        smoothed = counts + 1.0
        self.marginal_np = smoothed / smoothed.sum()

    @torch.no_grad()
    def score_tape(self, tape_ids, actual_next_reads, by_read):
        """tape_ids: (T,) int — reduced-alphabet trace.
        actual_next_reads: (T-1,) int — actual next read symbol per position.
        by_read: dict {int read symbol -> list of valid tuple ids}.
        Returns (T-1,) int argmax tuple ids."""
        tape_ids = np.asarray(tape_ids, dtype=np.int64)
        actual = np.asarray(actual_next_reads, dtype=np.int64)
        T = len(tape_ids)
        if T < 2:
            return np.empty(0, dtype=np.int64)
        n_pred = T - 1
        L = self.L
        preds = np.full(n_pred, -1, dtype=np.int64)

        # Cold-start positions (prefix shorter than L) → marginal fallback
        cold_n = min(L - 1, n_pred)
        for t in range(cold_n):
            cands = by_read.get(int(actual[t]), [])
            if not cands:
                preds[t] = 0; continue
            cand_probs = self.marginal_np[cands]
            preds[t] = int(cands[int(np.argmax(cand_probs))])

        # Hot positions (prefix length >= L)
        hot_positions = list(range(L - 1, n_pred))
        if not hot_positions or self.M == 0:
            # Use marginal fallback for everything else too
            for t in range(cold_n, n_pred):
                cands = by_read.get(int(actual[t]), [])
                if not cands:
                    preds[t] = 0; continue
                cand_probs = self.marginal_np[cands]
                preds[t] = int(cands[int(np.argmax(cand_probs))])
            return preds

        # Build query windows: for position t (0-indexed prediction),
        # query = tape_ids[t+1-L : t+1]
        queries = np.stack([tape_ids[t + 1 - L:t + 1]
                            for t in hot_positions])  # (N, L)
        queries_t = torch.as_tensor(queries, device=self.device)

        # Hamming distance to all training windows
        diff = (queries_t.unsqueeze(1)
                != self.windows_t.unsqueeze(0)).to(torch.int8)
        dist_mat = diff.sum(dim=2)  # (N, M)
        K_eff = min(self.K, self.M)
        _, topk_idx = torch.topk(dist_mat, K_eff, dim=1, largest=False)
        topk_cont = self.continuations_t[topk_idx]  # (N, K)
        counts = torch.zeros(len(hot_positions), self.A,
                             device=self.device, dtype=self.dtype)
        ones = torch.ones_like(topk_cont, dtype=self.dtype)
        counts.scatter_add_(1, topk_cont, ones)
        smoothed = counts + self.alpha_prior   # (N, A)
        smoothed_np = smoothed.cpu().numpy()

        for i, t in enumerate(hot_positions):
            cands = by_read.get(int(actual[t]), [])
            if not cands:
                preds[t] = 0; continue
            cand_probs = smoothed_np[i, cands]
            preds[t] = int(cands[int(np.argmax(cand_probs))])
        return preds

    @torch.no_grad()
    def score_tapes_batched(self, tape_ids_list, actual_next_reads_list,
                             by_read, K=None):
        """Stack all (tape, position) queries from all tapes into one
        Hamming-distance computation, then distribute back."""
        if K is None:
            K = self.K
        if self.M == 0:
            # Fallback: return marginal-based predictions per tape
            return [self.score_tape(t, a, by_read)
                    for t, a in zip(tape_ids_list, actual_next_reads_list)]
        L = self.L
        # Flatten queries: each tape contributes positions L-1..n_pred-1
        # as "hot" queries; positions 0..L-2 use marginal fallback.
        all_queries = []
        all_pred_idx = []     # (tape_idx, position_in_tape) tuples
        all_actuals = []
        per_tape_n_pred = []
        for ti, (tape_ids, actuals) in enumerate(
                zip(tape_ids_list, actual_next_reads_list)):
            tape_ids = np.asarray(tape_ids, dtype=np.int64)
            actuals = np.asarray(actuals, dtype=np.int64)
            T = len(tape_ids)
            n_pred = max(T - 1, 0)
            per_tape_n_pred.append(n_pred)
            for t in range(L - 1, n_pred):
                all_queries.append(tape_ids[t + 1 - L:t + 1])
                all_pred_idx.append((ti, t))
                all_actuals.append(int(actuals[t]))

        # Allocate per-tape preds
        preds_per_tape = [np.full(n, -1, dtype=np.int64)
                          for n in per_tape_n_pred]

        # Cold-start (prefix < L) per-tape: marginal fallback
        for ti, (tape_ids, actuals) in enumerate(
                zip(tape_ids_list, actual_next_reads_list)):
            actuals = np.asarray(actuals, dtype=np.int64)
            n_pred = per_tape_n_pred[ti]
            cold_n = min(L - 1, n_pred)
            for t in range(cold_n):
                cands = by_read.get(int(actuals[t]), [])
                if not cands:
                    preds_per_tape[ti][t] = 0
                else:
                    preds_per_tape[ti][t] = int(
                        cands[int(np.argmax(self.marginal_np[cands]))])

        if not all_queries:
            return preds_per_tape

        # Hot queries: batched Hamming + top-K lookup
        queries_arr = np.stack(all_queries)  # (N_total, L)
        N_total = queries_arr.shape[0]
        queries_t = torch.as_tensor(queries_arr, device=self.device)

        # Hamming distance: (N_total, M, L) → (N_total, M)
        # Chunk if memory is tight
        chunk_size = 512
        smoothed_chunks = []
        for start in range(0, N_total, chunk_size):
            end = min(start + chunk_size, N_total)
            qchunk = queries_t[start:end]
            diff = (qchunk.unsqueeze(1)
                    != self.windows_t.unsqueeze(0)).to(torch.int8)
            dist_mat = diff.sum(dim=2)
            K_eff = min(K, self.M)
            _, topk_idx = torch.topk(dist_mat, K_eff,
                                      dim=1, largest=False)
            topk_cont = self.continuations_t[topk_idx]
            counts = torch.zeros(end - start, self.A,
                                 device=self.device, dtype=self.dtype)
            ones = torch.ones_like(topk_cont, dtype=self.dtype)
            counts.scatter_add_(1, topk_cont, ones)
            smoothed_chunks.append((counts + self.alpha_prior).cpu().numpy())
        smoothed_all = np.concatenate(smoothed_chunks, axis=0)

        # Apply masks per (tape, position)
        for i, (ti, t) in enumerate(all_pred_idx):
            cands = by_read.get(int(all_actuals[i]), [])
            if not cands:
                preds_per_tape[ti][t] = 0
            else:
                cand_probs = smoothed_all[i, cands]
                preds_per_tape[ti][t] = int(
                    cands[int(np.argmax(cand_probs))])
        return preds_per_tape


# --------------------------------------------------------------------
# GDC torch for TM
# --------------------------------------------------------------------
class TorchTMGDC:
    """Batched GDC forward pass on a single tape, with per-position
    masked argmax. Mirrors GenerativeDenseChain.forward_pass + forecast
    + symbol marginalization."""

    def __init__(self, alpha=0.95, theta=0.05, beta=0.0,
                 transition_type='self_loop',
                 initial_dist='sequence_starts',
                 terminal_behavior='diffuse',
                 alpha_fc=None, theta_fc=None,
                 device=DEVICE, dtype=DTYPE):
        self.alpha = alpha; self.theta = theta; self.beta = beta
        self.transition_type = transition_type
        self.initial_dist = initial_dist
        self.terminal_behavior = terminal_behavior
        # Dual-alpha: the per-position PREDICTION transition may use a
        # different (alpha_fc, theta_fc) than the carried context advance.
        # None reuses (alpha, theta) -> original single-alpha behavior.
        self.alpha_fc = alpha_fc; self.theta_fc = theta_fc
        self.device = device; self.dtype = dtype

    def fit(self, train_seqs, alphabet_size):
        self.A = int(alphabet_size)
        col_seqs = [np.asarray(s, dtype=np.int64).reshape(-1, 1)
                    for s in train_seqs if len(s) > 0]
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
        self.last_nt_idx = (int(nt_indices[-1].item())
                            if len(nt_indices) > 0 else 0)
        # Symbol-onehot for marginalization
        sym_clipped = torch.clamp(self.sym_t, max=self.A - 1)
        self.symbol_onehot = torch.zeros((self.N, self.A),
                                          dtype=self.dtype, device=self.device)
        self.symbol_onehot.scatter_(1, sym_clipped.unsqueeze(1), 1.0)
        # Diffusion coefficients
        self.beta_nt = ((1.0 - self.alpha - self.theta)
                        / max(self.N - 2, 1))
        self.beta_t  = (1.0 - self.theta) / max(self.N - 1, 1)
        # Forecast (prediction-step) diffusion coefficients for dual-alpha.
        a_fc = self.alpha if self.alpha_fc is None else self.alpha_fc
        t_fc = self.theta if self.theta_fc is None else self.theta_fc
        self.alpha_fc_eff = a_fc; self.theta_fc_eff = t_fc
        self.beta_nt_fc = (1.0 - a_fc - t_fc) / max(self.N - 2, 1)
        self.beta_t_fc  = (1.0 - t_fc) / max(self.N - 1, 1)
        self._is_dual = (a_fc != self.alpha) or (t_fc != self.theta)

    @torch.no_grad()
    def score_tapes_batched(self, tape_ids_list, actual_next_reads_list,
                             by_read):
        """Batched version: process B test tapes in lockstep on GPU.
        Pads tapes to the same length and masks inactive positions."""
        B = len(tape_ids_list)
        if B == 0:
            return [np.empty(0, dtype=np.int64) for _ in range(B)]
        tapes = [np.asarray(t, dtype=np.int64) for t in tape_ids_list]
        actuals = [np.asarray(a, dtype=np.int64)
                   for a in actual_next_reads_list]
        Ts = np.array([len(t) for t in tapes])
        T_max = int(Ts.max())

        # Pad tapes with -1; pad actuals to T-1 with -1
        tape_mat = np.full((B, T_max), -1, dtype=np.int64)
        actual_mat = np.full((B, T_max - 1 if T_max > 0 else 0), -1,
                             dtype=np.int64)
        for i, (t, a) in enumerate(zip(tapes, actuals)):
            tape_mat[i, :len(t)] = t
            actual_mat[i, :len(a)] = a
        tape_t = torch.as_tensor(tape_mat, device=self.device)

        # Initial distribution (replicated across batch)
        if self.initial_dist == 'uniform':
            d0 = torch.full((self.N,), 1.0 / self.N,
                            dtype=self.dtype, device=self.device)
        else:
            starts_f = self.start_t.to(self.dtype)
            d0 = starts_f / starts_f.sum()
        dist = d0.unsqueeze(0).expand(B, self.N).contiguous()

        non_terminal_mask = (~self.term_t).to(self.dtype)
        terminal_mask_f  = self.term_t.to(self.dtype)
        V_distinct = max(int(torch.unique(self.sym_t).numel()), 1)
        inv_V_beta = self.beta / V_distinct

        # We'll record argmax predictions per (batch, position).
        # Allocate (B, T_max-1) int matrix with -1 padding.
        if T_max > 0:
            preds_mat = np.full((B, T_max - 1), -1, dtype=np.int64)
        else:
            preds_mat = np.zeros((B, 0), dtype=np.int64)

        for t in range(T_max):
            if t > 0:
                prev = dist
                # Carried (context) advance with (alpha, theta).
                dist = _self_loop_transition_batched(
                    prev, self.alpha, self.theta, self.beta_nt,
                    self.beta_t, non_terminal_mask, terminal_mask_f,
                    self.last_nt_idx, self.terminal_behavior)
                # Prediction transition: dual-alpha uses (alpha_fc, theta_fc)
                # from the SAME pre-advance state; single-alpha reuses `dist`.
                if self._is_dual:
                    pred_dist = _self_loop_transition_batched(
                        prev, self.alpha_fc_eff, self.theta_fc_eff,
                        self.beta_nt_fc, self.beta_t_fc, non_terminal_mask,
                        terminal_mask_f, self.last_nt_idx,
                        self.terminal_behavior)
                else:
                    pred_dist = dist
                # Predict tuple at position t (= prediction t-1).
                # Match numpy greedy_sample's argmax-over-chain-positions
                # behaviour: pick position with highest probability
                # among those matching the candidate read, output its
                # tuple-id (MAP over positions, not Bayesian sum).
                # GPU: per-batch max-pool dist over positions sharing
                # the same tuple-id.
                sym_max = torch.full((B, self.A), -1.0,
                                     dtype=self.dtype,
                                     device=self.device)
                sym_idx = self.sym_t.unsqueeze(0).expand(B, self.N)
                sym_max = sym_max.scatter_reduce(
                    1, sym_idx, pred_dist, reduce='amax',
                    include_self=False)
                sym_max_np = sym_max.cpu().numpy()
                pred_idx = t - 1
                for b in range(B):
                    if pred_idx >= Ts[b] - 1:
                        continue
                    cands = by_read.get(int(actual_mat[b, pred_idx]), [])
                    if not cands:
                        preds_mat[b, pred_idx] = 0
                    else:
                        cand_probs = sym_max_np[b, cands]
                        preds_mat[b, pred_idx] = int(
                            cands[int(np.argmax(cand_probs))])

            # Filter on tape[t] for each batch element where t < Ts[b]
            obs = tape_t[:, t]  # (B,) int (-1 for inactive)
            active = (obs >= 0)
            obs_safe = torch.clamp(obs, min=0)
            match_ind = (self.sym_t.unsqueeze(0)
                         == obs_safe.unsqueeze(1)).to(self.dtype)
            if self.beta == 0.0:
                new_dist = dist * match_ind
                ssum = new_dist.sum(dim=1, keepdim=True)
                new_dist = torch.where(
                    ssum > 0, new_dist / ssum,
                    torch.full_like(new_dist, 1.0 / self.N))
            else:
                emission = (1.0 - self.beta) * match_ind + inv_V_beta
                unnorm = dist * emission
                ssum = unnorm.sum(dim=1, keepdim=True)
                new_dist = torch.where(
                    ssum > 0, unnorm / ssum,
                    torch.full_like(unnorm, 1.0 / self.N))
            active_f = active.to(self.dtype).unsqueeze(1)
            dist = active_f * new_dist + (1 - active_f) * dist

        # Unpack results back to per-tape lists
        return [preds_mat[i, :Ts[i] - 1].copy() for i in range(B)]

    @torch.no_grad()
    def score_tape(self, tape_ids, actual_next_reads, by_read):
        tape_ids = np.asarray(tape_ids, dtype=np.int64)
        actual = np.asarray(actual_next_reads, dtype=np.int64)
        T = len(tape_ids)
        if T < 2:
            return np.empty(0, dtype=np.int64)
        n_pred = T - 1

        # Initial distribution
        if self.initial_dist == 'uniform':
            dist = torch.full((self.N,), 1.0 / self.N,
                              dtype=self.dtype, device=self.device)
        else:
            starts_f = self.start_t.to(self.dtype)
            dist = starts_f / starts_f.sum()

        non_terminal_mask = (~self.term_t).to(self.dtype)
        terminal_mask_f  = self.term_t.to(self.dtype)
        V_distinct = max(int(torch.unique(self.sym_t).numel()), 1)
        inv_V_beta = self.beta / V_distinct

        # Collect per-position symbol predictions
        preds = np.full(n_pred, -1, dtype=np.int64)

        # We need at every position t (predicting token t+1) the
        # symbol distribution = (T @ filtered_dist_at_t) marginalized.
        # The forward pass:
        #   for each obs in tape:
        #     if step > 0: dist = transition(dist)
        #     dist = filter(dist, obs)
        # At each prediction step (after observing obs[t]), we need
        # one more transition + marginalize to get P(obs[t+1] | history).

        # We use batch dimension B=1 (single tape) here for simplicity;
        # can later expand to multi-tape batching if needed.
        dist_b = dist.unsqueeze(0)   # (1, N)

        for t in range(T):
            if t > 0:
                dist_b = _self_loop_transition_batched(
                    dist_b, self.alpha, self.theta, self.beta_nt,
                    self.beta_t, non_terminal_mask, terminal_mask_f,
                    self.last_nt_idx, self.terminal_behavior)
            # At this point, dist_b is the prior at step t (before
            # observing obs[t]). For prediction at step t-1's question
            # "what comes after obs[0..t-1]?", the answer's symbol
            # marginal is already captured by this prior.
            #
            # Predicting obs[t] from history obs[0..t-1] means: get
            # symbol marginal of dist_b BEFORE filtering on obs[t].
            # That corresponds to prediction t-1 (the (t-1)-th prediction).
            if t > 0:
                sym_dist = dist_b @ self.symbol_onehot   # (1, A)
                sym_sum  = sym_dist.sum(dim=1, keepdim=True)
                sym_dist = torch.where(sym_sum > 0,
                                        sym_dist / sym_sum,
                                        torch.full_like(sym_dist,
                                                        1.0 / self.A))
                sym_dist_np = sym_dist.cpu().numpy().squeeze(0)
                pred_idx = t - 1
                cands = by_read.get(int(actual[pred_idx]), [])
                if not cands:
                    preds[pred_idx] = 0
                else:
                    cand_probs = sym_dist_np[cands]
                    preds[pred_idx] = int(
                        cands[int(np.argmax(cand_probs))])

            # Filter on obs[t]
            obs = int(tape_ids[t])
            match_ind = (self.sym_t == obs).to(self.dtype).unsqueeze(0)
            if self.beta == 0.0:
                new_dist = dist_b * match_ind
                ssum = new_dist.sum(dim=1, keepdim=True)
                dist_b = torch.where(
                    ssum > 0, new_dist / ssum,
                    torch.full_like(new_dist, 1.0 / self.N))
            else:
                emission = (1.0 - self.beta) * match_ind + inv_V_beta
                unnorm = dist_b * emission
                ssum = unnorm.sum(dim=1, keepdim=True)
                dist_b = torch.where(
                    ssum > 0, unnorm / ssum,
                    torch.full_like(unnorm, 1.0 / self.N))
        return preds


__all__ = ['TorchTMParrot', 'TorchTMGDC']
