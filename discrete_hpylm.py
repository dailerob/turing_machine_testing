"""Hierarchical Pitman-Yor Language Model (HPYLM) for discrete sequences.

A minimal, standalone implementation of the n-gram HPYLM (Teh 2006;
Wood et al. 2009 "A Stochastic Memoizer for Sequence Data"), aka
fixed-depth Sequence Memoizer. Used as a Bayesian-nonparametric
n-gram baseline alongside CHMM / ALERGIA / Parrot on our HMM and
Turing-machine benchmarks.

API (mirrors `discrete_parrot.DiscreteParrotPool`):

    pool = HPYLMPool(corpora, alphabet_size, max_depth=4, ...)
        # corpora : list of 1-D int arrays
    p = pool.predict_distribution(prefix, h=1, ...)
        # prefix : 1-D int array; returns (alphabet_size,) probability vector

Implementation notes:
  - Trie of contexts (root → unigram → bigram → ...), with counts and
    Pitman-Yor table assignments per (context, symbol).
  - Single-pass incremental seating with a fixed seed (no Gibbs).
    For each new training token, follow the standard Chinese restaurant
    franchise: with prob ∝ (c - d·t) per existing table, sit at it; else
    open a new table and recursively add a "customer" at the parent
    context (the franchise's coupling). This is the standard "add"
    operation from Teh 2006 §4.
  - At prediction time the recursive PY formula gives:
        P(s | u) = (c_{us} - d·t_{us}) / (c_u + α)  +
                   (α + d·t_u) / (c_u + α) · P(s | parent(u))
    with the empty-context base case returning a uniform distribution
    over the alphabet (or a hyperprior we'll keep at uniform).
  - Hyperparameters (d, α) are shared across depths (single-pair
    HPYLM); a depth-tied version would split them per depth. Single
    pair is the common practical choice.
  - Multi-step prediction (h>1) iterates the recursive PY at each
    step, treating the predicted token as the most recent context.
"""
from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np


class _Node:
    __slots__ = ('parent', 'children', 'c', 't')

    def __init__(self, alphabet_size: int, parent: Optional['_Node'] = None):
        self.parent = parent
        self.children: dict = {}
        self.c = np.zeros(alphabet_size, dtype=np.int64)  # customers per symbol
        self.t = np.zeros(alphabet_size, dtype=np.int64)  # tables per symbol

    @property
    def c_total(self) -> int:
        return int(self.c.sum())

    @property
    def t_total(self) -> int:
        return int(self.t.sum())


class HPYLMPool:
    """Hierarchical Pitman-Yor Language Model with fixed maximum context depth.

    Parameters
    ----------
    corpora : list of 1-D int arrays
        Training sequences (each an arbitrary-length 1-D int array).
    alphabet_size : int
        Size of the symbol alphabet. Tokens must be in [0, alphabet_size).
    max_depth : int
        Cap on the n-gram context length. Setting `max_depth=k` yields
        a (k+1)-gram model (the predicted symbol plus k context tokens).
    discount : float
        Pitman-Yor discount d ∈ (0, 1). 0.5 is a common default.
    concentration : float
        Pitman-Yor concentration α > -d. 1.0 is a common default.
    seed : int
        RNG seed for incremental seating.
    """

    def __init__(self, corpora: Sequence[np.ndarray], alphabet_size: int,
                 max_depth: int = 4, discount: float = 0.5,
                 concentration: float = 1.0, seed: int = 0):
        self.A = int(alphabet_size)
        self.D = int(max_depth)
        self.d = float(discount)
        self.alpha = float(concentration)
        self._rng = np.random.default_rng(seed)
        self.root = _Node(self.A)
        self._fit(corpora)

    # -- fitting ---------------------------------------------------------
    def _walk(self, ctx: tuple, create: bool = False) -> Optional[_Node]:
        """Walk to the node for context `ctx` (left = oldest, right = most recent).
        If create=True, create missing nodes; else return None at the deepest
        existing prefix.
        """
        node = self.root
        for sym in ctx:
            if sym not in node.children:
                if not create:
                    return node
                node.children[sym] = _Node(self.A, parent=node)
            node = node.children[sym]
        return node

    def _add_to_node(self, node: _Node, sym: int) -> None:
        """Add one customer for `sym` at `node`; possibly open a new table
        and recursively propagate to the parent."""
        if node.c[sym] == 0:
            # First customer for this symbol → must open new table
            node.c[sym] = 1
            node.t[sym] = 1
            if node.parent is not None:
                self._add_to_node(node.parent, sym)
            return
        # Existing customers: stochastic CRP seating
        # P(open new table) = (α + d * t_total) / (c_total + α + (existing weight))
        # Standard formula (Teh 2006): probability of new table is
        #   (α + d·t) / (c + α)
        # where c, t are TOTAL customers/tables at this restaurant.
        # Existing-table probability is then 1 - new-table prob.
        existing = node.c[sym] - self.d * node.t[sym]
        new_table = self.alpha + self.d * node.t_total
        total_weight = existing + new_table
        if total_weight <= 0:
            # numerical fallback: open new table
            node.c[sym] += 1
            node.t[sym] += 1
            if node.parent is not None:
                self._add_to_node(node.parent, sym)
            return
        if self._rng.random() < new_table / total_weight:
            # open a new table for this symbol; propagate to parent
            node.c[sym] += 1
            node.t[sym] += 1
            if node.parent is not None:
                self._add_to_node(node.parent, sym)
        else:
            # sit at an existing table; just increment customer count
            node.c[sym] += 1

    def _fit(self, corpora: Sequence[np.ndarray]) -> None:
        for seq in corpora:
            seq = np.asarray(seq, dtype=np.int64).ravel()
            for t in range(len(seq)):
                ctx_full = seq[max(0, t - self.D):t]
                sym = int(seq[t])
                if sym < 0 or sym >= self.A:
                    continue  # skip OOV
                # Add customer at the deepest available context, propagating up
                node = self._walk(tuple(int(x) for x in ctx_full), create=True)
                self._add_to_node(node, sym)

    # -- prediction ------------------------------------------------------
    def _predictive(self, node: _Node) -> np.ndarray:
        """Compute the predictive distribution at `node` recursively.
        Returns (alphabet_size,) probability vector summing to 1."""
        if node is None or node is self.root:
            # Walk all the way up: empty-context base case is uniform over alphabet.
            return np.full(self.A, 1.0 / self.A)
        c_total = node.c_total
        t_total = node.t_total
        if c_total == 0:
            # No customers here yet — fall back to parent
            return self._predictive(node.parent)
        # Pitman-Yor recursive predictive
        denom = c_total + self.alpha
        # First term: per-symbol direct contribution
        first = (node.c.astype(np.float64) - self.d * node.t.astype(np.float64)) / denom
        first = np.clip(first, 0.0, None)
        # Second term: backoff weight × parent predictive
        backoff_weight = (self.alpha + self.d * t_total) / denom
        parent_dist = self._predictive(node.parent)
        return first + backoff_weight * parent_dist

    def predict_distribution(self, prefix: np.ndarray, h: int = 1,
                              alpha_prior: float = 0.0) -> np.ndarray:
        """Return P(x_{t+h} | prefix) as a length-`alphabet_size` vector.

        For h>1, marginalize over the predicted intermediate tokens
        (deterministic-mode: take argmax at each intermediate step and
        condition on that). For our HMM h=1 metric this is just the
        single-step recursive PY predictive.
        """
        prefix = np.asarray(prefix, dtype=np.int64).ravel()
        ctx_full = tuple(int(x) for x in prefix[-self.D:])
        node = self._walk(ctx_full, create=False)
        if h == 1:
            p = self._predictive(node)
            if alpha_prior > 0:
                p = (p + alpha_prior) / (p.sum() + self.A * alpha_prior)
            return p
        # h>1: marginalize step by step (greedy / argmax for simplicity)
        cur_prefix = list(int(x) for x in prefix)
        for step in range(h - 1):
            ctx = tuple(cur_prefix[-self.D:])
            node = self._walk(ctx, create=False)
            p = self._predictive(node)
            cur_prefix.append(int(np.argmax(p)))
        ctx = tuple(cur_prefix[-self.D:])
        node = self._walk(ctx, create=False)
        p = self._predictive(node)
        if alpha_prior > 0:
            p = (p + alpha_prior) / (p.sum() + self.A * alpha_prior)
        return p

    def predict_argmax(self, prefix: np.ndarray, h: int = 1,
                       mask: Optional[np.ndarray] = None,
                       alpha_prior: float = 0.0) -> int:
        p = self.predict_distribution(prefix, h=h, alpha_prior=alpha_prior)
        if mask is not None:
            p = p.copy()
            p[~mask] = -1.0
        return int(np.argmax(p))


__all__ = ['HPYLMPool']


if __name__ == "__main__":
    # Self-test: train on a periodic sequence; predict the next symbol.
    rng = np.random.default_rng(0)
    period = 5
    seq = np.array([i % period for i in range(500)], dtype=np.int64)
    pool = HPYLMPool([seq], alphabet_size=period, max_depth=4)
    # After context (k-1, k-2, ...), the next symbol must be k mod period.
    # Test on a few contexts.
    for end in (10, 20, 100, 200):
        ctx = seq[:end]
        p = pool.predict_distribution(ctx, h=1)
        truth = int(seq[end])
        argmax = int(np.argmax(p))
        ok = '✓' if argmax == truth else '✗'
        print(f"  {ok} prefix len {end}: argmax={argmax} truth={truth}  p={p.round(3)}")
