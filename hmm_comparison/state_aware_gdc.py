"""GDC variant with emission-aware diffuse on the discrete benchmark.

Standard GDC self_loop transition:
    dist_new = θ·dist + α·shift(dist) + (1−α−θ)·uniform(N)

The 'uniform(N)' diffuse smears mass across all N=80,000 training
positions, ignoring which emission each position represents. The
trace on prefix 89 of sparse_large showed this is benign at low α
(many positions in the same hidden state survive the post-emission
filter and contribute well-averaged emission samples) but fatal at
high α (only 4 specific positions advance, so the next-symbol
marginal is a 4-sample empirical estimate with high variance).

This module implements:

    dist_new = θ·dist + α·shift(dist) + (1−α−θ)·emission_marginal_diffuse(dist)

where the emission_marginal_diffuse re-distributes mass within each
emission group:

    For each symbol s:
        q_s = sum of dist over positions emitting s  (current emission marginal)
        count_s = # training positions emitting s
    diffuse[i] = q_{e_i} / count_{e_i}

This preserves the dist's emission marginal across the diffuse step
(diffuse[i] sums to 1, just like the uniform), but distributes group
totals according to the current posterior's emission marginal rather
than the global training-emission frequency.

For predicting the next symbol from a posterior heavily concentrated
on hidden state s with emission distribution E[s, :], this should
help: emission_marginal_diffuse keeps mass distributed across
many state-s positions (instead of advancing only the few specific
positions that survived the high-α filter) so the marginal next-symbol
prediction averages over the empirical state-s emission samples.
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from generative_dense_chain import GenerativeDenseChain


class EmissionAwareGDC(GenerativeDenseChain):
    """GDC with state-aware (emission-marginal-preserving) diffuse.

    Instead of the standard self_loop's uniform-N diffuse, this class
    uses an emission-conditioned diffuse: mass is re-distributed
    uniformly within each emission group, with group totals equal to
    the current posterior's emission marginal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Precompute, for each training position, the indices that
        # share its emission. Stored as a flat (sorted) array per
        # symbol for vectorised indexing.
        self.symbol_of = self.states[:, 0].astype(np.int64)
        max_sym = int(self.symbol_of.max()) + 1
        self.symbol_indices = [
            np.where(self.symbol_of == s)[0] for s in range(max_sym)]
        self.symbol_count = np.array(
            [len(ix) for ix in self.symbol_indices], dtype=np.int64)
        self.n_sym = max_sym

    def _emission_aware_diffuse(self, dist):
        """Return a diffuse vector that sums to 1, distributed
        uniformly within each emission group with group totals equal
        to the current posterior's emission marginal."""
        out = np.zeros_like(dist)
        for s in range(self.n_sym):
            ix = self.symbol_indices[s]
            if len(ix) == 0:
                continue
            q_s = float(dist[ix].sum())
            if q_s > 0:
                out[ix] = q_s / len(ix)
        return out

    def _transition(self, dist, alpha=None, theta=None, gamma=None,
                    transition_type=None):
        if alpha is None: alpha = self.alpha
        if theta is None: theta = self.theta
        if transition_type is None: transition_type = self.transition_type

        n = self.n_states
        if n == 1:
            return dist.copy()

        if transition_type != 'self_loop':
            # Fall back to parent class for non-self_loop variants
            return super()._transition(dist, alpha, theta, gamma,
                                       transition_type)

        # Build emission-aware self_loop transition
        # next-state shift (advance): roll forward
        advance = np.zeros_like(dist)
        advance[1:] = dist[:-1]
        # terminal positions: their advance mass is treated per
        # terminal_behavior (matches the parent self_loop logic)
        terminal_advance_mass = float(dist[self.terminal_mask].sum())

        if self.terminal_behavior == 'absorb':
            # Terminal mass leaks out — don't redistribute
            pass
        else:
            # Terminal mass diffuses uniformly over non-terminal states
            # (matching the parent's diffuse mode)
            non_term_count = int((~self.terminal_mask).sum())
            if non_term_count > 0:
                redist_mask = ~self.terminal_mask
                advance[redist_mask] += (terminal_advance_mass
                                         / non_term_count)

        # Emission-aware diffuse component
        diffuse = self._emission_aware_diffuse(dist)
        diffuse_mass = 1.0 - alpha - theta

        return theta * dist + alpha * advance + diffuse_mass * diffuse


def make_emission_aware_gdc_forecaster(nA, alpha=0.8, theta=0.001,
                                       beta=0.1,
                                       transition_type='self_loop',
                                       initial_dist='uniform',
                                       terminal_behavior='absorb'):
    """Factory: returns a thin forecaster wrapping EmissionAwareGDC,
    matching the GDCForecaster interface used by `evaluation.py`."""

    class _Forecaster:
        def __init__(self):
            self.nA = nA; self.gdc = None; self._symbol_of_state = None
        def fit(self, sequences):
            seq_arrays = [np.asarray(s, dtype=np.int64).reshape(-1, 1)
                          for s in sequences]
            self.gdc = EmissionAwareGDC(
                seq_arrays, alpha=alpha, theta=theta, gamma=0.0,
                beta=beta, transition_type=transition_type,
                initial_dist=initial_dist,
                terminal_behavior=terminal_behavior)
            self._symbol_of_state = self.gdc.states[:, 0].astype(np.int64)
            return self
        def horizon_emission(self, prefix_obs, h):
            obs = np.asarray(prefix_obs, dtype=np.int64).reshape(-1, 1)
            final = self.gdc.forward_pass(obs, return_history=False)
            fc = self.gdc.forecast(final, n_steps=h)
            out = np.zeros(self.nA)
            np.add.at(out, self._symbol_of_state, fc)
            return out / out.sum() if out.sum() > 0 else \
                   np.full(self.nA, 1.0/self.nA)

    return _Forecaster()


# ---- Quick sanity test ---------------------------------------------
if __name__ == "__main__":
    # Simple invariant: emission_aware_diffuse(uniform_dist) == uniform.
    seqs = [np.array([0, 1, 2, 3, 0, 1, 2, 3]).reshape(-1, 1)]
    g = EmissionAwareGDC(seqs, alpha=0.5, theta=0.1, beta=0.0,
                         transition_type='self_loop',
                         initial_dist='uniform',
                         terminal_behavior='absorb')
    n = g.n_states
    uniform = np.full(n, 1.0/n)
    diff = g._emission_aware_diffuse(uniform)
    print(f"diffuse(uniform) sum = {diff.sum():.6f}, "
          f"max-min = {diff.max() - diff.min():.6f}")
    assert abs(diff.sum() - 1.0) < 1e-9, "diffuse must sum to 1"
    print("OK: emission_aware_diffuse preserves a uniform input.")

    # Concentrated on one emission group: diffuse should redistribute
    # within that emission only.
    one_hot = np.zeros(n); one_hot[0] = 1.0  # all mass at position 0 (emit=0)
    diff = g._emission_aware_diffuse(one_hot)
    same_emit = g.symbol_indices[0]
    print(f"diffuse(one_hot[0]) sum = {diff.sum():.6f}, "
          f"mass on emit-0 group = {diff[same_emit].sum():.6f}, "
          f"mass elsewhere = {diff.sum() - diff[same_emit].sum():.6f}")
    assert abs(diff[same_emit].sum() - 1.0) < 1e-9
    print("OK: emission-aware diffuse keeps mass within an emission group.")
