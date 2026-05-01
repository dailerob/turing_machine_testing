"""Sequence-model wrappers for PAutomaC scoring.

Each model exposes:
    fit(train_seqs, alphabet_size)
    log_prob(seq)  -> natural log of P(seq + END | model)

Where `train_seqs` is a list of int64 arrays (variable length) and
`seq` is a single int64 array.

For sequence-end modelling we append an explicit END token equal to
`alphabet_size` to every training sequence and to every scored
sequence.  The model then operates on alphabet of size `A + 1`.
"""

from __future__ import annotations
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "chmm_tests", "naturecomm_cscg"))

from generative_dense_chain import GenerativeDenseChain  # noqa: E402
from chmm_actions import CHMM, forward as chmm_forward  # noqa: E402

LOG_EPS = -700.0  # natural log of ~5e-305 — used for clipping zero probabilities


def append_end(seqs, end_token):
    return [np.concatenate([s, [end_token]]).astype(np.int64) for s in seqs]


# --------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------
class UniformModel:
    name = 'uniform'

    def fit(self, train_seqs, alphabet_size):
        self.A = alphabet_size + 1   # include END
        self.log_p = -np.log(self.A)

    def log_prob(self, seq):
        T = len(seq) + 1   # +1 for END
        return T * self.log_p


class UnigramModel:
    name = 'unigram'

    def fit(self, train_seqs, alphabet_size, smooth=1.0):
        self.A = alphabet_size + 1
        counts = np.full(self.A, smooth, dtype=np.float64)
        for s in train_seqs:
            for tok in s:
                counts[int(tok)] += 1.0
            counts[alphabet_size] += 1.0  # END
        self.log_p = np.log(counts / counts.sum())
        self.end_token = alphabet_size

    def log_prob(self, seq):
        if len(seq) == 0:
            return self.log_p[self.end_token]
        return float(np.sum(self.log_p[seq.astype(np.int64)])
                     + self.log_p[self.end_token])


class BigramModel:
    """Add-one-smoothed bigram with explicit START + END."""
    name = 'bigram'

    def fit(self, train_seqs, alphabet_size, smooth=1.0):
        # Symbols 0..A-1 plus END=A.  START is implicit (only conditions).
        self.A = alphabet_size
        self.end_token = alphabet_size
        V = alphabet_size + 1
        # Counts include start->first and last->END
        bg = np.full((V + 1, V), smooth, dtype=np.float64)  # row V = START
        START = V
        for s in train_seqs:
            prev = START
            for tok in s:
                bg[prev, int(tok)] += 1
                prev = int(tok)
            bg[prev, self.end_token] += 1
        self.log_p = np.log(bg / bg.sum(axis=1, keepdims=True))

    def log_prob(self, seq):
        START = self.A + 1   # last row index
        prev = START
        lp = 0.0
        for tok in seq:
            lp += self.log_p[prev, int(tok)]
            prev = int(tok)
        lp += self.log_p[prev, self.end_token]
        return float(lp)


class KneserNey3gramModel:
    """Interpolated Kneser-Ney 3-gram with explicit START / END.

    Uses a single discount D (default 0.75).  At inference,
        P(w | h2, h1) = max(c(h2,h1,w) - D, 0) / c(h2,h1)
                       + lambda(h2,h1) * P_KN(w | h1)
    where the lower-order P_KN(w | h1) is a continuation probability.
    """
    name = 'kn3'

    def __init__(self, discount=0.75):
        self.D = discount

    def fit(self, train_seqs, alphabet_size):
        self.A = alphabet_size
        self.end_token = alphabet_size
        # We use two distinct sentinel tokens for START so that the
        # 3-gram (START, START, w) is well-defined for the first symbol
        # of a sequence.
        self.START = alphabet_size + 1
        self.V = alphabet_size + 1   # tokens: 0..A-1 and END=A
        from collections import defaultdict
        c3 = defaultdict(lambda: defaultdict(int))   # c3[(h2,h1)][w]
        c2 = defaultdict(lambda: defaultdict(int))   # c2[h1][w]
        cont1 = defaultdict(set)                     # cont1[w]: distinct h1 preceding w
        cont2 = defaultdict(set)                     # cont2[(h2,h1)]: distinct w following
        for s in train_seqs:
            tokens = [self.START, self.START] + list(int(t) for t in s) + [self.end_token]
            for t in range(2, len(tokens)):
                w = tokens[t]; h1 = tokens[t - 1]; h2 = tokens[t - 2]
                c3[(h2, h1)][w] += 1
                c2[h1][w] += 1
                cont1[w].add(h1)
                cont2[(h2, h1)].add(w)
        self.c3 = c3; self.c2 = c2
        self.cont1 = cont1; self.cont2 = cont2
        # Total counts for c2[h1] sums and c3[(h2,h1)] sums
        self.c3_sum = {k: sum(v.values()) for k, v in c3.items()}
        self.c2_sum = {k: sum(v.values()) for k, v in c2.items()}
        # Continuation-base unigram: P_cont(w) = #{h: w followed h} / #{(h,w) pairs}
        n_bigram_types = sum(len(s) for s in cont1.values())
        if n_bigram_types == 0:
            n_bigram_types = 1
        self.p_cont1 = np.full(self.V + 2, 1e-12)  # extra slot for START
        for w in range(self.V):
            self.p_cont1[w] = len(cont1.get(w, set())) / n_bigram_types
        # Renormalise (excluding START)
        s = self.p_cont1[: self.V].sum()
        if s > 0:
            self.p_cont1[: self.V] /= s

    def _prob_kn1(self, h1, w):
        """Lower-order: P_KN(w | h1) using continuation."""
        D = self.D
        cnt2 = self.c2.get(h1)
        if cnt2 is None or self.c2_sum.get(h1, 0) == 0:
            return float(self.p_cont1[w])
        total = self.c2_sum[h1]
        c = cnt2.get(w, 0)
        n_unique = len(cnt2)
        first = max(c - D, 0) / total
        lam = (D * n_unique) / total
        return first + lam * float(self.p_cont1[w])

    def _prob_kn2(self, h2, h1, w):
        D = self.D
        cnt3 = self.c3.get((h2, h1))
        if cnt3 is None or self.c3_sum.get((h2, h1), 0) == 0:
            return self._prob_kn1(h1, w)
        total = self.c3_sum[(h2, h1)]
        c = cnt3.get(w, 0)
        n_unique = len(cnt3)
        first = max(c - D, 0) / total
        lam = (D * n_unique) / total
        return first + lam * self._prob_kn1(h1, w)

    def log_prob(self, seq):
        toks = [self.START, self.START] + list(int(t) for t in seq) + [self.end_token]
        lp = 0.0
        for t in range(2, len(toks)):
            p = self._prob_kn2(toks[t - 2], toks[t - 1], toks[t])
            lp += np.log(max(p, 1e-300))
        return float(lp)


# --------------------------------------------------------------------
# CHMM
# --------------------------------------------------------------------
class CHMMModel:
    def __init__(self, K=4, n_em_iters=50, pseudocount=1e-3, seed=0):
        self.K = K; self.n_em = n_em_iters; self.pseudo = pseudocount; self.seed = seed
        self.name = f'chmm-K{K}'

    def fit(self, train_seqs, alphabet_size):
        self.A = alphabet_size + 1
        self.end_token = alphabet_size
        seqs = append_end(train_seqs, self.end_token)
        x_train = np.concatenate(seqs).astype(np.int64)
        a_train = np.zeros_like(x_train)
        n_clones = np.full(self.A, self.K, dtype=np.int64)
        self.model = CHMM(n_clones=n_clones, x=x_train, a=a_train,
                          pseudocount=self.pseudo, seed=self.seed)
        self.model.learn_em_T(x_train, a_train, n_iter=self.n_em,
                              term_early=True)

    def log_prob(self, seq):
        x = np.concatenate([seq, [self.end_token]]).astype(np.int64)
        a = np.zeros_like(x)
        # bps returns -log2 P(x_t | x_<t) per timestep
        bps_arr = np.asarray(self.model.bps(x, a), dtype=np.float64)
        log2_p = -float(bps_arr.sum())
        return log2_p * np.log(2)


# --------------------------------------------------------------------
# GDC
# --------------------------------------------------------------------
class GDCModel:
    def __init__(self, alpha=0.95, theta=0.05, gamma=0.0, beta=0.0,
                 transition_type='self_loop',
                 initial_dist='sequence_starts'):
        self.alpha = alpha; self.theta = theta; self.gamma = gamma
        self.beta = beta; self.transition_type = transition_type
        self.initial_dist = initial_dist
        self.name = (f'gdc-a{alpha}-t{theta}-{transition_type}'
                     .replace('self_loop_two_step', '2step')
                     .replace('self_loop', '1step'))

    def fit(self, train_seqs, alphabet_size):
        self.A = alphabet_size + 1
        self.end_token = alphabet_size
        seqs = append_end(train_seqs, self.end_token)
        # GDC expects column-shaped sequences; keep (T, 1) per seq
        col_seqs = [s.reshape(-1, 1).astype(np.int64) for s in seqs
                    if len(s) > 0]
        self.gdc = GenerativeDenseChain(
            col_seqs, alpha=self.alpha, theta=self.theta,
            gamma=self.gamma, beta=self.beta,
            transition_type=self.transition_type,
            initial_dist=self.initial_dist)
        # Cache per-state emission for fast next-symbol marginal
        self.emit = self.gdc.states[:, 0].astype(np.int64)
        self.A_total = int(self.emit.max()) + 1
        if self.A_total < self.A:
            self.A_total = self.A
        # Group state indices by emitted symbol
        self.idx_by_emit = [np.where(self.emit == a)[0]
                            for a in range(self.A_total)]
        self.init_dist = self.gdc._get_initial_distribution(
            self.initial_dist).copy()

    def log_prob(self, seq):
        full = np.concatenate([seq, [self.end_token]]).astype(np.int64)
        dist = self.init_dist.copy()
        log_p = 0.0
        for t, sym in enumerate(full):
            if t > 0:
                # apply transition
                dist = self.gdc.forecast(dist, n_steps=1)
            sym = int(sym)
            if sym >= len(self.idx_by_emit):
                # symbol unseen by GDC's emit set -> tiny prob
                log_p += LOG_EPS
                # No states match: re-initialise to uniform (matches forward_pass)
                dist = np.ones_like(dist) / len(dist)
                continue
            idx = self.idx_by_emit[sym]
            if len(idx) == 0:
                log_p += LOG_EPS
                dist = np.ones_like(dist) / len(dist)
                continue
            q = float(dist[idx].sum())  # P(sym | history)
            if q <= 0:
                # All states emitting this symbol have zero mass; fall
                # back to a smoothing constant.
                log_p += LOG_EPS
                dist = np.ones_like(dist) / len(dist)
                continue
            log_p += float(np.log(q))
            new_dist = np.zeros_like(dist)
            new_dist[idx] = dist[idx] / q
            dist = new_dist
        return log_p
