"""PPM-D (Howard 1993) for discrete sequence prediction.

A minimal fixed-depth Prediction by Partial Matching, variant D, with
interpolated backoff. Used as a classical-compression baseline alongside
CHMM / ALERGIA / Parrot / HPYLM on our HMM and Turing-machine benchmarks.

API mirrors `discrete_parrot.DiscreteParrotPool` and `discrete_hpylm.HPYLMPool`:

    pool = PPMPool(corpora, alphabet_size, max_depth=4, discount=0.5)
        # corpora : list of 1-D int arrays
    p = pool.predict_distribution(prefix, h=1, ...)
        # prefix : 1-D int array; returns (alphabet_size,) probability vector

Implementation notes:
  - Trie of contexts (root → unigram → bigram → ...), with symbol counts
    per node. No table counts (PPM is non-Bayesian; the discount and
    escape probabilities are heuristic, not derived from a CRP prior).
  - At prediction time, the recursive PPM-D formula gives:
        P(s | u) = max(c_us - d, 0) / c_u  +  (d · q_u / c_u) · P(s | parent(u))
    where q_u is the number of distinct symbols seen at context u.
    This is the "interpolated" (blending) form used in modern PPM-D
    references (e.g. Howard 1993, Witten et al. 1999); the alternative
    "exclusion" form falls back only for unseen symbols.
  - The empty-context base case returns a uniform distribution over
    the alphabet (or with a Laplace prior, see `alpha_prior`).
  - Discount d is conventionally 0.5 (the "D" in PPM-D), but exposed
    as a hyperparameter for val-tuning.
"""
from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np


class _Node:
    __slots__ = ('parent', 'children', 'c')

    def __init__(self, alphabet_size: int, parent: Optional['_Node'] = None):
        self.parent = parent
        self.children: dict = {}
        self.c = np.zeros(alphabet_size, dtype=np.int64)  # customers per symbol

    @property
    def c_total(self) -> int:
        return int(self.c.sum())

    @property
    def q_total(self) -> int:
        """Number of distinct symbols with c > 0 at this node."""
        return int((self.c > 0).sum())


class PPMPool:
    """Fixed-depth PPM-D model over discrete sequences.

    Parameters
    ----------
    corpora : list of 1-D int arrays
        Training sequences.
    alphabet_size : int
        Size of the symbol alphabet. Tokens must be in [0, alphabet_size).
    max_depth : int
        Cap on the n-gram context length.
    discount : float
        Absolute-discount value d ∈ [0, 1). 0.5 is the canonical PPM-D.
    """

    def __init__(self, corpora: Sequence[np.ndarray], alphabet_size: int,
                 max_depth: int = 4, discount: float = 0.5):
        self.A = int(alphabet_size)
        self.D = int(max_depth)
        self.d = float(discount)
        self.root = _Node(self.A)
        self._fit(corpora)

    # -- fitting (pure counts; no sampling) -----------------------------
    def _walk(self, ctx: tuple, create: bool = False) -> Optional[_Node]:
        node = self.root
        for sym in ctx:
            if sym not in node.children:
                if not create:
                    return node
                node.children[sym] = _Node(self.A, parent=node)
            node = node.children[sym]
        return node

    def _fit(self, corpora: Sequence[np.ndarray]) -> None:
        for seq in corpora:
            seq = np.asarray(seq, dtype=np.int64).ravel()
            for t in range(len(seq)):
                ctx_full = seq[max(0, t - self.D):t]
                sym = int(seq[t])
                if sym < 0 or sym >= self.A:
                    continue
                # PPM updates counts at every level from the deepest context
                # to the unigram. (Note: this is the "update-exclusion" form.
                # Many PPM variants only update the deepest seen level; we
                # update all levels for simplicity, matching what "trie of
                # n-gram counts" naturally produces.)
                ctx = tuple(int(x) for x in ctx_full)
                for k in range(len(ctx) + 1):
                    node = self._walk(ctx[k:], create=True)
                    node.c[sym] += 1

    # -- prediction -----------------------------------------------------
    def _predictive(self, node: _Node) -> np.ndarray:
        """Recursive PPM-D interpolated predictive at `node`."""
        if node is None or node is self.root:
            # Empty-context base case: uniform over alphabet.
            return np.full(self.A, 1.0 / self.A)
        c_total = node.c_total
        q_total = node.q_total
        if c_total == 0:
            return self._predictive(node.parent)
        # Direct count contribution (with absolute discount)
        first = np.maximum(node.c.astype(np.float64) - self.d, 0.0) / c_total
        # Escape weight × parent
        escape = (self.d * q_total) / c_total
        parent_dist = self._predictive(node.parent)
        return first + escape * parent_dist

    def predict_distribution(self, prefix: np.ndarray, h: int = 1,
                              alpha_prior: float = 0.0) -> np.ndarray:
        prefix = np.asarray(prefix, dtype=np.int64).ravel()
        ctx_full = tuple(int(x) for x in prefix[-self.D:])
        node = self._walk(ctx_full, create=False)
        if h == 1:
            p = self._predictive(node)
            if alpha_prior > 0:
                p = (p + alpha_prior) / (p.sum() + self.A * alpha_prior)
            return p
        # h>1: greedy roll-out
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


__all__ = ['PPMPool']


if __name__ == "__main__":
    # Self-test: train on a periodic sequence; predict the next symbol.
    period = 5
    seq = np.array([i % period for i in range(500)], dtype=np.int64)
    pool = PPMPool([seq], alphabet_size=period, max_depth=4, discount=0.5)
    for end in (10, 20, 100, 200):
        ctx = seq[:end]
        p = pool.predict_distribution(ctx, h=1)
        truth = int(seq[end])
        argmax = int(np.argmax(p))
        ok = '✓' if argmax == truth else '✗'
        print(f"  {ok} prefix len {end}: argmax={argmax} truth={truth}  p={p.round(3)}")
