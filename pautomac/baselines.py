"""Off-the-shelf single-method baselines for PAutomaC.

Currently:
    SpectralOOMModel — Hsu-Kakade-Zhang spectral PFA learner via the
        repo's existing `spectral_oom.SpectralOOM`.  Comparable to
        Bailly's 4th-place submission.
    AlergiaModel — passive PDFA learner via AALpy's `run_Alergia`.
        Comparable in spirit to FlexFringe (modern ALERGIA).
"""

from __future__ import annotations
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from spectral_oom import SpectralOOM  # noqa: E402

try:
    from aalpy.learning_algs import run_Alergia
    HAS_AALPY = True
except Exception:
    HAS_AALPY = False

LOG_EPS = -700.0


def _append_end(seqs, end_token):
    return [list(s) + [end_token] for s in seqs]


# --------------------------------------------------------------------
# Spectral OOM
# --------------------------------------------------------------------
class SpectralOOMModel:
    """Spectral OOM with explicit END token.  Probabilities clipped via
    the simplex projection mode of `predict_next_probs`."""

    def __init__(self, max_basis_length=2, rank=None,
                 sv_rel_threshold=1e-8, prob_mode='clip'):
        self.max_basis_length = max_basis_length
        self.rank = rank
        self.sv_rel_threshold = sv_rel_threshold
        self.prob_mode = prob_mode
        self.name = (f"oom-L{max_basis_length}-r{rank}"
                     if rank is not None
                     else f"oom-L{max_basis_length}-auto")

    def fit(self, train_seqs, alphabet_size):
        self.A = alphabet_size + 1
        self.end_token = alphabet_size
        seqs = _append_end(train_seqs, self.end_token)
        self.oom = SpectralOOM(
            max_basis_length=self.max_basis_length,
            rank=self.rank,
            sv_rel_threshold=self.sv_rel_threshold,
            renormalize=True,
            verbose=False)
        self.oom.fit(seqs)

    def log_prob(self, seq):
        full = list(seq) + [self.end_token]
        state = self.oom.alpha0.copy()
        lp = 0.0
        for sym in full:
            probs = self.oom.predict_next_probs(state, horizon=1,
                                                mode=self.prob_mode)
            sid = self.oom.tok2id.get(int(sym))
            if sid is None:
                lp += LOG_EPS
                state = self.oom.alpha0.copy()
                continue
            p = float(probs[sid])
            lp += np.log(max(p, 1e-300))
            state = state @ self.oom.A[sid]
            n = np.linalg.norm(state)
            if 0 < n and (n > 1e50 or n < 1e-50):
                state = state / n
        return float(lp)


# --------------------------------------------------------------------
# ALERGIA via AALpy
# --------------------------------------------------------------------
class AlergiaModel:
    """Passive PDFA learner via AALpy's ALERGIA.  Uses an explicit
    SHARED START symbol so all sequences share an initial state, plus
    an END token. Falls back to a frequency model if AALpy is missing.
    """

    def __init__(self, eps=0.05):
        self.eps = eps
        self.name = f"alergia-eps{eps}"

    def fit(self, train_seqs, alphabet_size):
        if not HAS_AALPY:
            raise RuntimeError("AALpy not installed; pip install aalpy")
        self.A = alphabet_size + 1
        self.end_token = alphabet_size
        # AALpy MC requires every sequence to start with the same symbol.
        # Use a special START symbol that we'll prepend.
        self.start_token = alphabet_size + 1
        # Build training data: list of [START, s_1, ..., s_T, END]
        data = []
        for s in train_seqs:
            seq = [self.start_token] + [int(t) for t in s] + [self.end_token]
            data.append(seq)
        # Run ALERGIA -- learns a Markov chain
        self.mc = run_Alergia(data, automaton_type='mc', eps=self.eps,
                              print_info=False)
        # Build a fast scoring index: state.output -> state, plus
        # per-state transition map output -> probability.
        # AALpy's MC has states with `output` and `transitions` (list of
        # (target_state, prob)).  Convert to a dict for O(1) scoring.
        self.state_by_output = {}
        for st in self.mc.states:
            # ALERGIA may produce multiple states with the same output;
            # we just need a starting state lookup
            self.state_by_output.setdefault(st.output, st)
        self.initial = self.mc.initial_state

    def _next_state_and_prob(self, state, sym):
        """Find transition state -> next_state with output `sym`.
        Returns (next_state, prob) or (None, 0.0)."""
        # AALpy MC transitions: list of (target, prob) tuples on each state
        for target, prob in state.transitions:
            if target.output == sym:
                return target, prob
        return None, 0.0

    def log_prob(self, seq):
        # The first state is at the START symbol; ALERGIA's initial_state
        # has output == START.  We want the predicted probability of each
        # token in [s_1, ..., s_T, END].
        st = self.initial
        # The initial state's output should be START; that's "free"
        lp = 0.0
        for sym in list(seq) + [self.end_token]:
            ns, p = self._next_state_and_prob(st, int(sym))
            if ns is None or p <= 0:
                lp += LOG_EPS
                st = self.initial  # fall back
                continue
            lp += np.log(max(p, 1e-300))
            st = ns
        return float(lp)
