"""Smoke evaluation: GDC partial-match forecasting on NPI addition traces.

Pipeline (mirrors the existing TM eval pattern, but with 6-col rows):
  1. Generate training traces from (a, b) pairs at training-length operands.
  2. Build GDC with partial_match=True over the concatenated trace chain.
  3. For each test pair:
     - Build the init-block prefix (fully specified rows).
     - Initialize a simulator (4-row tape + 4 pointers).
     - Step loop: feed [obs_p1..p4, -1, -1] for the current step; do GDC
       forward pass; predict (action_type, arg) by argmax over marginal of
       the action cols; apply the predicted action to the simulator;
       repeat until HALT or step cap.
     - Decode row 4 as the output integer.
  4. Score exact match and per-step action accuracy.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from generative_dense_chain import GenerativeDenseChain                       # noqa: E402
from npi_program import (generate_trace, BLANK,                               # noqa: E402
    AT_HALT, AT_RETURN, AT_CALL, AT_MOVE, AT_WRITE, AT_INIT, AT_INIT_A, AT_INIT_B,
    SUB_ADD, SUB_ADD1, SUB_CARRY, SUB_LSHIFT,
    MOVE_p1_L, MOVE_p2_L, MOVE_p3_L, MOVE_p4_L, MOVE_p3_R,
    WRITE_p3_1, INIT_BEGIN, INIT_A_END, INIT_B_END, INIT_END)


# ---------------------------------------------------------------------------
# Incremental partial-match forward pass
# ---------------------------------------------------------------------------
class IncrementalGDC:
    """Step-by-step forward pass over a multi-column chain with
    partial_match emissions. Avoids the O(T²) cost of repeatedly calling
    GenerativeDenseChain.forward_pass on a growing observation list.

    Usage:
        igdc = IncrementalGDC(gdc)
        igdc.reset()
        for obs in init_rows:                      # all 6 cols specified
            igdc.step_full(obs)
        for _ in range(max_steps):
            pred_at, pred_arg = igdc.step_predict(obs_4)
            ...
    """

    def __init__(self, gdc: GenerativeDenseChain):
        assert gdc.partial_match, "IncrementalGDC requires partial_match=True"
        self.gdc = gdc
        self.alpha = gdc.alpha; self.theta = gdc.theta
        self.gamma = gdc.gamma; self.beta = gdc.beta
        self.transition_type = gdc.transition_type
        self.n = gdc.n_states; self.k = gdc.k
        self._pos_idx = gdc._position_value_to_indices
        self._beta_over_n = self.beta / self.n
        self.dist = None
        self.is_first = True

    def reset(self):
        self.dist = self.gdc._get_initial_distribution().copy()
        self.is_first = True

    def _match_counts(self, obs_subset_cols, obs_vec):
        """Count, per chain position, how many of the given cols match obs_vec.
        obs_subset_cols : iterable of col indices to consider.
        obs_vec         : length-k array (only cols in obs_subset_cols read).
        Returns: float ndarray of length n_states.
        """
        mc = np.zeros(self.n)
        for pos in obs_subset_cols:
            v = obs_vec[pos]
            if v == -1: continue
            for idx in self._pos_idx[pos].get(v, []):
                mc[idx] += 1.0
        return mc

    def _emission(self, match_counts):
        return (1.0 - self.beta) * (match_counts / self.k) + self._beta_over_n

    def _apply_emission(self, dist_prior, emission):
        u = dist_prior * emission
        total = u.sum()
        if total > 0:
            return u / total
        # Degenerate fallback: reset to uniform.
        return np.ones(self.n) / self.n

    def step_full(self, obs_vec):
        """Process a fully-specified row (all k cols known)."""
        if not self.is_first:
            self.dist = self.gdc._transition(
                self.dist, self.alpha, self.theta, self.gamma,
                self.transition_type)
        mc = self._match_counts(range(self.k), obs_vec)
        em = self._emission(mc)
        self.dist = self._apply_emission(self.dist, em)
        self.is_first = False

    def step_predict(self, obs_vec, predict_cols=(4, 5)):
        """Process a partially-specified row, predict the masked cols by
        argmax over the marginal of those cols, then commit the prediction.

        Returns
        -------
        tuple of ints — predicted values at the predict_cols (in order).
        """
        # 1. Transition (if not first step).
        if not self.is_first:
            dist_prior = self.gdc._transition(
                self.dist, self.alpha, self.theta, self.gamma,
                self.transition_type)
        else:
            dist_prior = self.dist
        # 2. Partial emission (only the observed cols).
        observed_cols = [c for c in range(self.k) if c not in predict_cols]
        mc_partial = self._match_counts(observed_cols, obs_vec)
        em_partial = self._emission(mc_partial)
        dist_partial = self._apply_emission(dist_prior, em_partial)
        # 3. Predict masked cols via argmax over marginal.
        mask = np.zeros(self.k, dtype=bool); mask[list(predict_cols)] = True
        pred = self.gdc.greedy_sample(dist_partial, mask=mask)
        pred_vals = tuple(int(pred[c]) for c in predict_cols)
        # 4. Commit the prediction: extend the emission with the predicted
        #    cols' match counts on top of the partial counts (reuse mc_partial).
        mc_full = mc_partial.copy()
        for c, v in zip(predict_cols, pred_vals):
            for idx in self._pos_idx[c].get(v, []):
                mc_full[idx] += 1.0
        em_full = self._emission(mc_full)
        self.dist = self._apply_emission(dist_prior, em_full)
        self.is_first = False
        return pred_vals


# ---------------------------------------------------------------------------
# Simulator (executes predicted actions and tracks the 4-row environment)
# ---------------------------------------------------------------------------
class _Simulator:
    """Mirrors `_NPIRunner`'s environment but accepts arbitrary (action_type,
    arg) pairs at each step instead of being driven by the program."""

    def __init__(self, a: int, b: int, n_cols_extra: int = 4):
        max_in = max(len(str(a)), len(str(b)))
        self.n_cols = max_in + n_cols_extra
        self.row1 = self._lsb(a, self.n_cols)
        self.row2 = self._lsb(b, self.n_cols)
        self.row3 = [0] * self.n_cols
        self.row4 = [None] * self.n_cols
        self.p = [None, 0, 0, 0, 0]

    @staticmethod
    def _lsb(n, w):
        out = []
        for _ in range(w):
            out.append(n % 10); n //= 10
        return out

    def _val(self, p_idx):
        col = self.p[p_idx]
        if not (0 <= col < self.n_cols):
            return BLANK
        row = (None, self.row1, self.row2, self.row3, self.row4)[p_idx]
        v = row[col]
        return BLANK if v is None else v

    def current_obs(self):
        return (self._val(1), self._val(2), self._val(3), self._val(4))

    def apply(self, at: int, arg: int):
        """Apply a predicted action to the environment.  Silently ignores
        invalid actions (e.g., INIT_* during forecasting) and out-of-range
        moves."""
        if at == AT_MOVE:
            if arg == MOVE_p1_L: self.p[1] += 1
            elif arg == MOVE_p2_L: self.p[2] += 1
            elif arg == MOVE_p3_L: self.p[3] += 1
            elif arg == MOVE_p4_L: self.p[4] += 1
            elif arg == MOVE_p3_R: self.p[3] -= 1
        elif at == AT_WRITE:
            if arg == WRITE_p3_1:
                if 0 <= self.p[3] < self.n_cols:
                    self.row3[self.p[3]] = 1
            elif 1 <= arg <= 10:
                v = arg - 1
                if 0 <= self.p[4] < self.n_cols:
                    self.row4[self.p[4]] = v
        # CALL / RETURN / HALT / INIT* don't update environment.

    def decode_output(self):
        cells = {i: v for i, v in enumerate(self.row4) if v is not None}
        if not cells:
            return 0
        max_col = max(cells.keys())
        digits = ''.join(str(cells.get(i, 0)) for i in range(max_col, -1, -1)).lstrip('0')
        return int(digits) if digits else 0


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
def _make_init_rows(a: int, b: int) -> np.ndarray:
    """Build the init-block prefix (fully specified 6-tuples) for a, b."""
    rows = []
    def emit(at, arg):
        rows.append((BLANK, BLANK, BLANK, BLANK, at, arg))
    emit(AT_INIT, INIT_BEGIN)
    for d in str(a):
        emit(AT_INIT_A, int(d))
    emit(AT_INIT, INIT_A_END)
    for d in str(b):
        emit(AT_INIT_B, int(d))
    emit(AT_INIT, INIT_B_END)
    emit(AT_INIT, INIT_END)
    return np.array(rows, dtype=np.int64)


def forecast_one(gdc: GenerativeDenseChain, a: int, b: int,
                  max_steps: int = 300,
                  train_obs_set: 'set | None' = None):
    """Forecast the trace for (a, b) via incremental GDC partial-match
    prediction. O(T·N) total (T steps, N chain positions).

    If `train_obs_set` is provided, also records per-step whether each test
    observation (4-tuple of pointer values) was seen in any training row.
    """
    prefix_rows = _make_init_rows(a, b)
    gt_trace = generate_trace(a, b)
    n_init = prefix_rows.shape[0]
    gt_post_init = gt_trace[n_init:]   # ground-truth action rows after init

    sim = _Simulator(a, b, n_cols_extra=4 + max(len(str(a)), len(str(b))))

    igdc = IncrementalGDC(gdc); igdc.reset()
    # Consume init prefix (all 6 cols specified per row).
    for row in prefix_rows:
        igdc.step_full(row)

    predicted_actions = []
    test_obs_seen = []   # one entry per forecast step: 1 if obs in training, else 0
    halted = False
    for step in range(max_steps):
        obs = sim.current_obs()
        if train_obs_set is not None:
            test_obs_seen.append(int(obs in train_obs_set))
        obs_vec = np.array(
            [obs[0], obs[1], obs[2], obs[3], -1, -1], dtype=np.int64)
        pred_at, pred_arg = igdc.step_predict(obs_vec, predict_cols=(4, 5))
        predicted_actions.append((pred_at, pred_arg))
        sim.apply(pred_at, pred_arg)
        if pred_at == AT_HALT:
            halted = True
            break

    gt_actions = [(int(r[4]), int(r[5])) for r in gt_post_init]
    return dict(
        predicted_output=sim.decode_output(),
        predicted_actions=predicted_actions,
        gt_actions=gt_actions,
        n_steps=len(predicted_actions),
        halted=halted,
        test_obs_seen=test_obs_seen,
    )


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------
def run_smoke():
    rng = np.random.default_rng(42)

    # Train on 1-3 digit operands (mirrors NPI's training-length distribution).
    n_train = 1200
    train_pairs = []
    for _ in range(n_train):
        digs_a = int(rng.integers(1, 4))   # 1, 2, or 3 digits
        digs_b = int(rng.integers(1, 4))
        a = int(rng.integers(10**(digs_a-1), 10**digs_a))
        b = int(rng.integers(10**(digs_b-1), 10**digs_b))
        train_pairs.append((a, b))
    print(f"Training pairs: {n_train}")

    # Build training traces and concatenate.
    train_traces = [generate_trace(a, b) for (a, b) in train_pairs]
    total_rows = sum(t.shape[0] for t in train_traces)
    print(f"Trace lengths: min={min(t.shape[0] for t in train_traces)}, "
          f"max={max(t.shape[0] for t in train_traces)}, "
          f"mean={total_rows / n_train:.1f}")
    print(f"Total chain rows: {total_rows}")

    # Build GDC with partial_match.
    t0 = time.time()
    gdc = GenerativeDenseChain(
        train_traces, alpha=0.95, theta=0.05, beta=0.05,
        transition_type='self_loop',
        initial_dist='sequence_starts',
        terminal_behavior='absorb',
        partial_match=True,
    )
    print(f"GDC fit: {gdc.n_states} chain positions, k={gdc.k}  "
          f"({time.time()-t0:.2f}s)")

    # Build the set of distinct 4-tape observation tuples that ever appear
    # in training. Excludes the init prefix's all-BLANK observations.
    train_obs_set = set()
    for tr in train_traces:
        for row in tr:
            op1, op2, op3, op4 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            if not (op1 == BLANK and op2 == BLANK and op3 == BLANK and op4 == BLANK):
                train_obs_set.add((op1, op2, op3, op4))
    print(f"Distinct training 4-tape obs tuples: {len(train_obs_set)}")

    # Eval pairs: 25 test pairs per length bucket. train-len-{1,2,3} are
    # in-distribution; len-{4,5,7,10} are length-OOD.
    def n_digit_pair(lo, hi, n):
        return [(int(rng.integers(lo, hi)), int(rng.integers(lo, hi)))
                for _ in range(n)]
    eval_buckets = [
        ('train-len-1', n_digit_pair(0, 10, 25)),
        ('train-len-2', n_digit_pair(10, 100, 25)),
        ('train-len-3', n_digit_pair(100, 1000, 25)),
        ('len-4 OOD',   n_digit_pair(1000, 10000, 25)),
        ('len-5 OOD',   n_digit_pair(10000, 100000, 25)),
        ('len-7 OOD',   n_digit_pair(10**6, 10**7, 25)),
        ('len-10 OOD',  n_digit_pair(10**9, 10**10, 25)),
    ]

    for bucket_name, pairs in eval_buckets:
        t0 = time.time()
        n_correct = 0
        action_correct = 0; action_total = 0
        obs_in_train = 0; obs_total = 0
        for (a, b) in pairs:
            a, b = int(a), int(b)
            res = forecast_one(gdc, a, b, max_steps=400,
                                train_obs_set=train_obs_set)
            ok = (res['predicted_output'] == a + b)
            n_correct += int(ok)
            L = min(len(res['predicted_actions']), len(res['gt_actions']))
            for i in range(L):
                if res['predicted_actions'][i] == res['gt_actions'][i]:
                    action_correct += 1
            action_total += L
            obs_in_train += sum(res['test_obs_seen'])
            obs_total += len(res['test_obs_seen'])
        rate = 100.0 * n_correct / len(pairs)
        act_rate = (100.0 * action_correct / action_total
                    if action_total else 0.0)
        cov_rate = (100.0 * obs_in_train / obs_total
                    if obs_total else 0.0)
        print(f"  [{bucket_name:>13s}]  exact={n_correct:>2d}/{len(pairs)} ({rate:>5.1f}%)  "
              f"action_acc={act_rate:>5.1f}%  obs_in_train={cov_rate:>5.1f}% "
              f"({obs_in_train}/{obs_total})  ({time.time()-t0:.1f}s)")

    # Show one detailed forecast for diagnosis (3-digit test).
    a, b = 234, 567
    res = forecast_one(gdc, a, b)
    print(f"\nDetailed forecast {a}+{b}={a+b}:")
    print(f"  predicted_output = {res['predicted_output']}")
    print(f"  predicted vs gt actions (first 30):")
    for i in range(min(30, len(res['predicted_actions']), len(res['gt_actions']))):
        p = res['predicted_actions'][i]; g = res['gt_actions'][i]
        mark = '✓' if p == g else '✗'
        print(f"    step {i:>3}  pred=({p[0]},{p[1]:>2})  gt=({g[0]},{g[1]:>2})  {mark}")


if __name__ == "__main__":
    run_smoke()
