"""Soft-emission GDC wrapper.

Standard GDC: each state s emits its stored vector `states[s, :]`
deterministically.  Sampling/prediction uses hard match against the
conditional and reads off the best-matching state's stored values.

Soft-emission GDC: introduce a per-position noise probability `eta`.
For each state s and each position i:

    P(emit_i = a | state s) = (1 - eta) * 1{states[s, i] == a}
                            + eta / V_i

where V_i = number of distinct values at position i in the training
states (i.e. `max(states[:, i]) + 1`).  With eta=0 this reduces to
the deterministic-emission GDC.

Conditioning and prediction both use this soft emission model.
At each test step:

  1. Apply soft conditioning given the observed positions in the
     `conditional` vector:
         log_posterior[s] = log state_dist[s]
                          + sum_{i in cond} log P(emit_i = c_i | s)
  2. For each non-conditional position j, compute the marginal
         P(emit_j = a | history, conditional) =
             sum_s posterior[s] * P(emit_j = a | s)
     and take its argmax.

Vectorised; per-step cost is O(n_states * (k_constraints + sum V_j))
which for the TM tasks is dominated by the constraint sum.
"""

from __future__ import annotations
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from generative_dense_chain import GenerativeDenseChain  # noqa: E402


class SoftEmissionGDC:
    """Wraps a fitted GenerativeDenseChain and adds soft per-position
    emissions with parameter `eta` ∈ [0, 1].

    eta=0 reproduces the standard hard-emission GDC.
    eta=1 emits uniformly regardless of state (degenerate).
    """

    def __init__(self, gdc: GenerativeDenseChain, eta: float = 0.1):
        self.gdc = gdc
        self.eta = float(eta)
        self.states = gdc.states
        self.n = gdc.n_states
        self.k = self.states.shape[1]
        # Per-position alphabet sizes (max value + 1)
        self.V = [int(self.states[:, i].max()) + 1 for i in range(self.k)]
        self._eta_over_V = [self.eta / V_i for V_i in self.V]
        # Cache the unique tuples seen in training, indexed by their
        # values at each position. Used to enumerate candidate emissions.
        self.unique_tuples = np.unique(self.states, axis=0)  # (T, k)

    # ----------------------------------------------------------------
    # Soft posterior and prediction
    # ----------------------------------------------------------------
    def _soft_log_lik(self, conditional):
        """Per-state log-likelihood of observing the constrained
        positions given the (soft) emission model.

        `conditional`: (k,) float array with NaN for don't-care.
        Returns: (n,) log-likelihood vector.
        """
        cond = np.asarray(conditional, dtype=float)
        cond_mask = ~np.isnan(cond)
        ll = np.zeros(self.n, dtype=np.float64)
        for i in range(self.k):
            if not cond_mask[i]:
                continue
            target = int(cond[i])
            V_i = self.V[i]
            # P(emit_i=target | s) = (1-eta) if states[s,i]==target else 0,
            # plus a constant eta/V_i added to all (uniform smoothing)
            match = (self.states[:, i] == target).astype(np.float64)
            p_emit = (1.0 - self.eta) * match + self.eta / V_i
            ll += np.log(np.maximum(p_emit, 1e-300))
        return ll

    def predict(self, state_dist, conditional):
        """Predict the most-likely full tuple under the soft-emission
        model, restricted to tuples seen in training that match the
        conditional values.  Returns shape (k,) with the conditional
        values copied through and non-conditional positions set to the
        argmax-tuple's values.

        At eta=0 this matches the original GDC's `greedy_sample` (joint
        tuple-mass argmax over states matching the conditional).
        """
        cond = np.asarray(conditional, dtype=float)
        cond_mask = ~np.isnan(cond)

        # Soft posterior over states given the conditional positions
        log_post = np.log(np.maximum(state_dist, 1e-300)) + \
            self._soft_log_lik(conditional)
        log_post -= log_post.max()
        posterior = np.exp(log_post)
        z = posterior.sum()
        if z > 0:
            posterior /= z
        else:
            posterior = np.full(self.n, 1.0 / self.n)

        # Candidate tuples = unique training tuples that match the
        # conditional positions (or all tuples if no conditional).
        if cond_mask.any():
            cond_vals = cond[cond_mask].astype(np.int64)
            ut = self.unique_tuples
            match = np.all(ut[:, cond_mask] == cond_vals[None, :], axis=1)
            candidates = ut[match]
            if len(candidates) == 0:
                # Fall back: ignore the conditional
                candidates = ut
        else:
            candidates = self.unique_tuples

        # For each candidate tuple t, score
        #   P(emit t | history) = sum_s posterior[s]
        #                         * prod_i [(1-eta)·1{states[s,i]==t_i} + eta/V_i]
        # We compute per_state product across positions, then dot with posterior.
        best_score = -np.inf
        best_tuple = candidates[0]
        for t in candidates:
            per_state = np.ones(self.n, dtype=np.float64)
            for i in range(self.k):
                ind = (self.states[:, i] == int(t[i])).astype(np.float64)
                p_emit_i = (1.0 - self.eta) * ind + self.eta / self.V[i]
                per_state *= p_emit_i
            score = float(np.dot(posterior, per_state))
            if score > best_score:
                best_score = score
                best_tuple = t

        return best_tuple.astype(float)


# ---------------------------------------------------------------------
# Standalone smoke test: compare to standard GDC on a tiny example
# ---------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    seqs = [np.array([[0, 0, 1], [1, 1, 1], [2, 2, 0]], dtype=np.int64)
            for _ in range(50)]
    gdc = GenerativeDenseChain(seqs, alpha=0.95, theta=0.05, gamma=0.0,
                               beta=0.0, transition_type='self_loop',
                               initial_dist='sequence_starts')
    print(f"states: {gdc.states[:5]}")
    print(f"n_states={gdc.n_states}, k={gdc.states.shape[1]}")

    soft = SoftEmissionGDC(gdc, eta=0.0)  # eta=0 should match standard GDC

    # Forward through one observation, then predict next given a read=1
    obs = np.array([[0, 0, 1]], dtype=np.int64)
    _, hist = gdc.forward_pass(obs, return_history=True)
    forecast = gdc.forecast(hist[-1], n_steps=1)
    cond = np.array([1.0, np.nan, np.nan])
    hard = gdc.greedy_sample(forecast, conditional=cond)
    soft0 = soft.predict(forecast, cond)
    print(f"\nhard pred:  {hard}")
    print(f"soft pred (eta=0): {soft0}")
    soft01 = SoftEmissionGDC(gdc, eta=0.1).predict(forecast, cond)
    soft05 = SoftEmissionGDC(gdc, eta=0.5).predict(forecast, cond)
    print(f"soft pred (eta=0.1): {soft01}")
    print(f"soft pred (eta=0.5): {soft05}")
