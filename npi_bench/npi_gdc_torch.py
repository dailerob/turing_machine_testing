"""Torch kernel for GDC partial-match forward pass on the NPI 6-col chain.

Mirrors `IncrementalGDC` in npi_eval.py but:
  - chain held on GPU as a single (N, k) int64 tensor;
  - posterior is a (B, N) float tensor batched across test pairs;
  - per-column match counting via broadcast compare (no Python dict lookups);
  - transition reuses `_self_loop_transition_batched` from
    hmm_comparison/gdc_torch_discrete.py;
  - argmax-over-marginal-action is computed via scatter_add into a
    (B, n_tuple_cells) joint marginal then argmax.

Numerically matches the numpy version within fp32/fp64 rounding.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'hmm_comparison'))

from gdc_torch_discrete import _self_loop_transition_batched              # noqa: E402


class NpiGDCTorch:
    """Discrete GDC with partial_match on a multi-column chain, on GPU.

    Designed for the NPI 6-col rows (4 obs + 2 action), but the k-arity is
    generic; you set `predict_cols` per `step_predict()` call.

    Usage
    -----
        gdc = NpiGDCTorch(train_traces, alpha=0.99, theta=0.01, beta=0.01,
                          n_action_types=8, n_arg_values=11,
                          device='cuda', dtype=torch.float32)
        gdc.reset(batch_size=B)
        for row_b in init_rows_batched:           # shape (B, k)
            gdc.step_full(row_b)
        for step in range(max_steps):
            obs_b = gather_obs_from_simulators()  # shape (B, k) with -1 for predict cols
            pred_at, pred_arg = gdc.step_predict(obs_b, predict_cols=(4, 5))
            # apply per-batch to simulators, etc.
    """

    def __init__(self, train_traces, alpha: float, theta: float, beta: float,
                  n_action_types: int = 8, n_arg_values: int = 11,
                  predict_cols=(4, 5),
                  terminal_behavior: str = 'absorb',
                  initial_dist: str = 'sequence_starts',
                  device: str = 'cuda',
                  dtype=torch.float64):
        # ---- Build (N, k) chain ----
        all_rows = np.vstack([np.asarray(t, dtype=np.int64) for t in train_traces])
        self.N, self.k = all_rows.shape
        # Start / terminal masks (one True per training trace).
        start_mask = np.zeros(self.N, dtype=bool)
        terminal_mask = np.zeros(self.N, dtype=bool)
        cumsum = 0
        for tr in train_traces:
            start_mask[cumsum] = True
            cumsum += len(tr)
            terminal_mask[cumsum - 1] = True

        self.device = device
        self.dtype = dtype
        self.states_t = torch.as_tensor(all_rows, dtype=torch.int64, device=device)
        self.start_t = torch.as_tensor(start_mask, device=device)
        self.term_t = torch.as_tensor(terminal_mask, device=device)
        self.non_terminal_mask_f = (~self.term_t).to(dtype)
        self.terminal_mask_f = self.term_t.to(dtype)
        nt_indices = torch.where(~self.term_t)[0]
        self.last_nt_idx = int(nt_indices[-1].item()) if len(nt_indices) > 0 else 0

        # Transition / emission constants.
        self.alpha = float(alpha); self.theta = float(theta)
        self.beta = float(beta)
        self.terminal_behavior = terminal_behavior
        self.initial_dist = initial_dist
        self.beta_nt = (1.0 - self.alpha - self.theta) / max(self.N - 2, 1)
        self.beta_t = (1.0 - self.theta) / max(self.N - 1, 1)
        self._beta_over_n = self.beta / self.N

        # For action prediction via scatter_add: (action_type, arg) → flat ID.
        self.predict_cols = tuple(predict_cols)
        self.n_action_types = int(n_action_types)
        self.n_arg_values = int(n_arg_values)
        self.n_joint = self.n_action_types * self.n_arg_values
        c_at, c_arg = self.predict_cols
        # tuple_id = action_type * n_arg + arg
        self._tuple_ids = (
            self.states_t[:, c_at] * self.n_arg_values + self.states_t[:, c_arg]
        ).to(torch.int64)  # (N,)

        # Per-call state
        self.dist = None
        self.is_first = True

    # -----------------------------------------------------------------
    def reset(self, batch_size: int):
        if self.initial_dist == 'sequence_starts':
            init = self.start_t.to(self.dtype)
            init = init / init.sum()
        else:
            init = torch.full((self.N,), 1.0 / self.N,
                              dtype=self.dtype, device=self.device)
        self.dist = init.unsqueeze(0).expand(batch_size, self.N).contiguous()
        self.is_first = True

    # -----------------------------------------------------------------
    def _transition(self):
        self.dist = _self_loop_transition_batched(
            self.dist, self.alpha, self.theta, self.beta_nt, self.beta_t,
            self.non_terminal_mask_f, self.terminal_mask_f,
            self.last_nt_idx, self.terminal_behavior)

    def _match_counts(self, obs_batch: torch.Tensor, cols) -> torch.Tensor:
        """Count, per (batch, chain position), how many of the given cols
        match obs_batch (with -1 treated as masked, contributing 0)."""
        B = obs_batch.shape[0]
        mc = torch.zeros(B, self.N, dtype=self.dtype, device=self.device)
        for c in cols:
            obs_c = obs_batch[:, c]                                    # (B,)
            valid = (obs_c >= 0)                                       # (B,) bool
            # Broadcast compare: states[:, c] (N,) vs obs_c (B,).
            matches = (self.states_t[:, c].unsqueeze(0)
                       == obs_c.unsqueeze(1))                          # (B, N)
            matches = matches & valid.unsqueeze(1)
            mc = mc + matches.to(self.dtype)
        return mc

    def _emission(self, mc: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.beta) * (mc / self.k) + self._beta_over_n

    def _apply_emission(self, dist_prior: torch.Tensor,
                         emission: torch.Tensor) -> torch.Tensor:
        u = dist_prior * emission
        total = u.sum(dim=1, keepdim=True)
        safe_total = torch.where(total > 0, total, torch.ones_like(total))
        normalized = u / safe_total
        uniform = torch.full_like(dist_prior, 1.0 / self.N)
        return torch.where(total > 0, normalized, uniform)

    # -----------------------------------------------------------------
    @torch.no_grad()
    def step_full(self, obs_batch: torch.Tensor):
        """Process a fully-specified row per batch element (B, k)."""
        if not self.is_first:
            self._transition()
        mc = self._match_counts(obs_batch, range(self.k))
        em = self._emission(mc)
        self.dist = self._apply_emission(self.dist, em)
        self.is_first = False

    @torch.no_grad()
    def step_predict(self, obs_batch: torch.Tensor, predict_cols=None):
        """Process a partially-specified row, predict the masked cols by
        argmax over the marginal joint over those cols, then commit.

        Parameters
        ----------
        obs_batch : (B, k) int64 tensor with -1 in predict_cols.
        predict_cols : optional override of self.predict_cols.

        Returns
        -------
        (pred_at, pred_arg) : tuple of (B,) int64 tensors.
        """
        if predict_cols is None:
            predict_cols = self.predict_cols
        else:
            predict_cols = tuple(predict_cols)
        c_at, c_arg = predict_cols
        observed_cols = [c for c in range(self.k) if c not in predict_cols]
        B = obs_batch.shape[0]

        if not self.is_first:
            self._transition()
        dist_prior = self.dist  # (B, N)

        # --- Partial emission (observed cols only) ---
        mc_partial = self._match_counts(obs_batch, observed_cols)
        em_partial = self._emission(mc_partial)
        dist_partial = self._apply_emission(dist_prior, em_partial)

        # --- Argmax over marginal of predict_cols joint ---
        # Bin posterior into (action_type, arg) cells via scatter_add.
        if predict_cols == self.predict_cols:
            tuple_ids = self._tuple_ids
        else:
            tuple_ids = (self.states_t[:, c_at] * self.n_arg_values
                         + self.states_t[:, c_arg]).to(torch.int64)
        joint = torch.zeros(B, self.n_joint,
                             dtype=self.dtype, device=self.device)
        idx_expand = tuple_ids.unsqueeze(0).expand(B, self.N)
        joint.scatter_add_(1, idx_expand, dist_partial)
        best_ids = joint.argmax(dim=1)  # (B,)
        pred_at = (best_ids // self.n_arg_values).to(torch.int64)
        pred_arg = (best_ids % self.n_arg_values).to(torch.int64)

        # --- Commit prediction: fold the action cols into match counts. ---
        # Build a "full-obs" tensor with predicted action values.
        full_obs = obs_batch.clone()
        full_obs[:, c_at] = pred_at
        full_obs[:, c_arg] = pred_arg
        mc_action = self._match_counts(full_obs, list(predict_cols))
        mc_full = mc_partial + mc_action
        em_full = self._emission(mc_full)
        self.dist = self._apply_emission(dist_prior, em_full)
        self.is_first = False

        return pred_at, pred_arg


# ----------------------------------------------------------------------------
# Sanity check: compare torch kernel output to numpy IncrementalGDC.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    from generative_dense_chain import GenerativeDenseChain
    from npi_program import (generate_trace, BLANK, AT_HALT,
        AT_INIT, AT_INIT_A, AT_INIT_B, INIT_BEGIN, INIT_A_END, INIT_B_END, INIT_END)
    from npi_eval import IncrementalGDC, _Simulator, _make_init_rows, forecast_one

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}")

    rng = np.random.default_rng(0)
    # 60 training pairs, 1-3 digits
    pairs = []
    for _ in range(60):
        da = int(rng.integers(1, 4)); db = int(rng.integers(1, 4))
        pairs.append((int(rng.integers(10**(da-1), 10**da)),
                      int(rng.integers(10**(db-1), 10**db))))
    train_traces = [generate_trace(a, b) for a, b in pairs]

    # Reference: numpy GDC
    gdc_np = GenerativeDenseChain(train_traces, alpha=0.95, theta=0.05,
                                    beta=0.05,
                                    transition_type='self_loop',
                                    initial_dist='sequence_starts',
                                    terminal_behavior='absorb',
                                    partial_match=True)
    # Torch GDC
    gdc_t = NpiGDCTorch(train_traces, alpha=0.95, theta=0.05, beta=0.05,
                         device=device, dtype=torch.float64)
    print(f"Chain rows: {gdc_t.N}, k={gdc_t.k}")

    test_pairs = [(34, 7), (123, 456), (99, 1), (5, 8)]

    print(f"\n=== Per-pair sanity check (numpy vs torch) ===")
    for (a, b) in test_pairs:
        # Numpy
        res_np = forecast_one(gdc_np, a, b, max_steps=200)

        # Torch single-pair
        gdc_t.reset(batch_size=1)
        prefix = _make_init_rows(a, b)
        for row in prefix:
            row_t = torch.as_tensor(row, dtype=torch.int64,
                                     device=device).unsqueeze(0)
            gdc_t.step_full(row_t)
        sim = _Simulator(a, b, n_cols_extra=4 + max(len(str(a)), len(str(b))))
        preds_torch = []
        for _ in range(200):
            obs = sim.current_obs()
            obs_t = torch.tensor(
                [[obs[0], obs[1], obs[2], obs[3], -1, -1]],
                dtype=torch.int64, device=device)
            pat, parg = gdc_t.step_predict(obs_t, predict_cols=(4, 5))
            pa = int(pat.item()); pg = int(parg.item())
            preds_torch.append((pa, pg))
            sim.apply(pa, pg)
            if pa == AT_HALT: break
        # Compare prediction sequences
        match = (res_np['predicted_actions'] == preds_torch)
        print(f"  {a:>4} + {b:>4}:  numpy preds len {len(res_np['predicted_actions'])}, "
              f"torch preds len {len(preds_torch)}, "
              f"identical={match}")
        if not match:
            # show first divergence
            for i in range(min(len(preds_torch), len(res_np['predicted_actions']))):
                if res_np['predicted_actions'][i] != preds_torch[i]:
                    print(f"    first diff at step {i}: "
                          f"numpy {res_np['predicted_actions'][i]} vs "
                          f"torch {preds_torch[i]}")
                    break
