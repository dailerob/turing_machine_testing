"""Flat LSTM baseline for NPI addition traces.

Mirrors NPI's interface but without compositional program structure:
  - At each step t, the model gets (history-of-rows-via-LSTM) + (current obs)
  - and predicts (action_type, arg) of the current row.
At inference time, the LSTM runs step-by-step driven by a Python simulator,
exactly like the GDC eval.

This is the same baseline NPI's paper compares against (the "flat seq2seq
LSTM" in Figures 5/6).
"""
from __future__ import annotations
import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from npi_program import (generate_trace, BLANK,                                 # noqa: E402
    AT_HALT, AT_INIT, AT_INIT_A, AT_INIT_B,
    MOVE_p1_L, MOVE_p2_L, MOVE_p3_L, MOVE_p4_L, MOVE_p3_R,
    WRITE_p3_1,
    INIT_BEGIN, INIT_A_END, INIT_B_END, INIT_END)
from npi_eval import _Simulator, _make_init_rows                                # noqa: E402


# Embedding-table SIZES. We use 11 (= max obs value + 1, since BLANK=10) for
# every obs column so the same index space works across all 4 — op3 only ever
# takes values in {0, 1, BLANK} but the larger table is harmless.
N_OBS1 = 11
N_OBS2 = 11
N_OBS3 = 11
N_OBS4 = 11
N_ACTION_TYPES = 8
N_ARG = 11


class FlatLSTM(nn.Module):
    """2-layer LSTM with per-column embeddings; predicts (at, arg).

    Architecture (matches NPI's hidden size; ~1M params total):
      - Per-column embedding of the input row (6 cols).
      - LSTM(emb_in, hidden, n_layers).
      - Heads: (lstm_out ‖ current_obs_emb) → action_type / arg logits.
    """

    def __init__(self, emb_dim: int = 32, hidden: int = 256, n_layers: int = 2,
                  dropout: float = 0.0):
        super().__init__()
        # Per-column embedding tables.
        self.emb_o1 = nn.Embedding(N_OBS1 + 1, emb_dim, padding_idx=N_OBS1)
        self.emb_o2 = nn.Embedding(N_OBS2 + 1, emb_dim, padding_idx=N_OBS2)
        self.emb_o3 = nn.Embedding(N_OBS3 + 1, emb_dim, padding_idx=N_OBS3)
        self.emb_o4 = nn.Embedding(N_OBS4 + 1, emb_dim, padding_idx=N_OBS4)
        self.emb_at = nn.Embedding(N_ACTION_TYPES + 1, emb_dim,
                                    padding_idx=N_ACTION_TYPES)
        self.emb_ag = nn.Embedding(N_ARG + 1, emb_dim, padding_idx=N_ARG)
        # Note: padding_idx = max class id (we'll add 1 to all training inputs
        # if we ever need to use it; for now we won't actually pad).

        in_dim = 6 * emb_dim
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=n_layers,
                             batch_first=True, dropout=dropout)
        # Heads condition on (lstm_out, current-obs-embedding).
        head_in = hidden + 4 * emb_dim
        self.head_at = nn.Linear(head_in, N_ACTION_TYPES)
        self.head_ag = nn.Linear(head_in, N_ARG)

    def embed_row(self, row: torch.Tensor) -> torch.Tensor:
        """row: (..., 6) int64 → (..., 6*emb_dim)."""
        e = [self.emb_o1(row[..., 0]), self.emb_o2(row[..., 1]),
             self.emb_o3(row[..., 2]), self.emb_o4(row[..., 3]),
             self.emb_at(row[..., 4]), self.emb_ag(row[..., 5])]
        return torch.cat(e, dim=-1)

    def embed_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (..., 4) int64 → (..., 4*emb_dim)."""
        e = [self.emb_o1(obs[..., 0]), self.emb_o2(obs[..., 1]),
             self.emb_o3(obs[..., 2]), self.emb_o4(obs[..., 3])]
        return torch.cat(e, dim=-1)

    def forward(self, prefix_rows: torch.Tensor, obs_now: torch.Tensor,
                  state=None):
        """Run one step (inference path) or many (training path).

        Parameters
        ----------
        prefix_rows : (B, T, 6) int64
            Trace rows up to (but not including) the row we predict.
            For step 0, use a length-0 sequence or a single BOS row.
        obs_now : (B, 4) int64
            Observation of the row to predict (the 4 obs cols).
        state : (h, c) tuple to continue from a previous LSTM state, or None.

        Returns
        -------
        (logits_at, logits_arg, new_state) all shaped (B, n_classes) / lstm-state.
        """
        if prefix_rows.shape[1] == 0:
            # No history: zero LSTM init. We just take initial hidden state.
            B = prefix_rows.shape[0]
            h = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size,
                             device=prefix_rows.device,
                             dtype=next(self.parameters()).dtype)
            c = torch.zeros_like(h)
            lstm_out = h[-1].unsqueeze(1)   # (B, 1, hidden) — use last layer
            new_state = (h, c)
        else:
            emb = self.embed_row(prefix_rows)         # (B, T, 6*emb)
            if state is None:
                lstm_out, new_state = self.lstm(emb)
            else:
                lstm_out, new_state = self.lstm(emb, state)
        # Use the last timestep's output.
        last = lstm_out[:, -1]                        # (B, hidden)
        ctx = torch.cat([last, self.embed_obs(obs_now)], dim=-1)
        return self.head_at(ctx), self.head_ag(ctx), new_state

    def forward_full_trace(self, rows: torch.Tensor):
        """Process a full trace in parallel (training path).

        For target row t, the LSTM sees rows 0..t-1 (shifted by one).
        Predict (action_type, arg) of row t conditioned on obs of row t.

        Parameters
        ----------
        rows : (B, T, 6)  full trace per example.

        Returns
        -------
        logits_at, logits_arg : (B, T, n_classes) each.
        """
        B, T, _ = rows.shape
        # LSTM input: at step t (0-indexed), feed row t-1.
        # For step 0, feed a "BOS" row of all zeros.
        bos = torch.zeros((B, 1, 6), dtype=rows.dtype, device=rows.device)
        # Use padding indices so the embedding lookup is well-defined.
        bos[..., 0] = N_OBS1; bos[..., 1] = N_OBS2; bos[..., 2] = N_OBS3
        bos[..., 3] = N_OBS4; bos[..., 4] = N_ACTION_TYPES; bos[..., 5] = N_ARG
        lstm_inp = torch.cat([bos, rows[:, :-1]], dim=1)  # (B, T, 6)
        emb = self.embed_row(lstm_inp)                    # (B, T, 6*emb)
        lstm_out, _ = self.lstm(emb)                      # (B, T, hidden)
        obs_emb = self.embed_obs(rows[..., :4])           # (B, T, 4*emb)
        ctx = torch.cat([lstm_out, obs_emb], dim=-1)
        return self.head_at(ctx), self.head_ag(ctx)


# ---------------------------------------------------------------------------
# Data: build padded trace batches
# ---------------------------------------------------------------------------
def pad_traces(traces, pad_row=None):
    """traces : list of (T_i, 6) np.int64 arrays.
    Returns: (rows, lengths) where rows is (B, T_max, 6) tensor and
    lengths is (B,) int64.
    """
    B = len(traces)
    T_max = max(t.shape[0] for t in traces)
    if pad_row is None:
        # use padding indices in each column
        pad_row = np.array([N_OBS1, N_OBS2, N_OBS3, N_OBS4,
                             N_ACTION_TYPES, N_ARG], dtype=np.int64)
    out = np.tile(pad_row, (B, T_max, 1))
    lengths = np.zeros(B, dtype=np.int64)
    for i, t in enumerate(traces):
        out[i, :t.shape[0]] = t
        lengths[i] = t.shape[0]
    return out, lengths


# ---------------------------------------------------------------------------
# Inference: drive the LSTM step-by-step through a simulator
# ---------------------------------------------------------------------------
@torch.no_grad()
def lstm_forecast_one(model: FlatLSTM, a: int, b: int, device: str,
                       max_steps: int = 600):
    """Run the LSTM on (a, b) like our GDC eval driver does."""
    model.eval()
    prefix_rows = _make_init_rows(a, b).astype(np.int64)   # (T_init, 6)
    sim = _Simulator(a, b, n_cols_extra=4 + max(len(str(a)), len(str(b))))

    # Build a growing prefix-rows list. Start with init rows (fully known).
    prefix = torch.as_tensor(prefix_rows, dtype=torch.int64, device=device).unsqueeze(0)  # (1, T_init, 6)

    predicted_actions = []
    halted = False
    for step in range(max_steps):
        obs = sim.current_obs()
        obs_now = torch.tensor(
            [[obs[0], obs[1], obs[2], obs[3]]],
            dtype=torch.int64, device=device)
        logits_at, logits_ag, _ = model(prefix, obs_now)
        pred_at = int(logits_at.argmax(dim=-1).item())
        pred_ag = int(logits_ag.argmax(dim=-1).item())
        predicted_actions.append((pred_at, pred_ag))
        sim.apply(pred_at, pred_ag)
        if pred_at == AT_HALT:
            halted = True
            break
        # Append the full row (obs + chosen action) to the prefix.
        new_row = torch.tensor(
            [[[obs[0], obs[1], obs[2], obs[3], pred_at, pred_ag]]],
            dtype=torch.int64, device=device)
        prefix = torch.cat([prefix, new_row], dim=1)
    return dict(predicted_output=sim.decode_output(),
                predicted_actions=predicted_actions,
                halted=halted)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, train_traces, device, n_epochs: int = 40,
          batch_size: int = 64, lr: float = 1e-3, log_every: int = 5):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(train_traces)
    losses = []
    t_total = time.time()
    for epoch in range(n_epochs):
        model.train()
        order = np.random.permutation(n)
        ep_loss = 0.0; n_batches = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch_traces = [train_traces[i] for i in idx]
            rows_np, lengths = pad_traces(batch_traces)
            rows = torch.as_tensor(rows_np, device=device)
            lengths_t = torch.as_tensor(lengths, device=device)
            B, T, _ = rows.shape
            logits_at, logits_ag = model.forward_full_trace(rows)
            tgt_at = rows[..., 4]; tgt_ag = rows[..., 5]
            # Replace padded positions in the targets with -100 so
            # cross_entropy ignores them (they're outside the class range).
            mask = (torch.arange(T, device=device)[None]
                    < lengths_t[:, None])
            tgt_at_m = torch.where(mask, tgt_at, torch.full_like(tgt_at, -100))
            tgt_ag_m = torch.where(mask, tgt_ag, torch.full_like(tgt_ag, -100))
            l_at = F.cross_entropy(
                logits_at.reshape(-1, N_ACTION_TYPES),
                tgt_at_m.reshape(-1), ignore_index=-100, reduction='sum')
            l_ag = F.cross_entropy(
                logits_ag.reshape(-1, N_ARG),
                tgt_ag_m.reshape(-1), ignore_index=-100, reduction='sum')
            denom = mask.sum().clamp(min=1)
            loss = (l_at + l_ag) / denom
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1
        ep_loss /= max(n_batches, 1)
        losses.append(ep_loss)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"  epoch {epoch+1:>3d}/{n_epochs}  loss={ep_loss:.4f}  "
                  f"[{time.time()-t_total:.1f}s elapsed]")
    print(f"Training: {time.time()-t_total:.1f}s")
    return losses


# ---------------------------------------------------------------------------
# Main: build, train, eval
# ---------------------------------------------------------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}")

    # Same training set as our GDC sweeps: 10000 random 1-3 digit pairs, seed 42.
    rng = np.random.default_rng(42)
    n_train = 10000
    train_pairs = []
    for _ in range(n_train):
        da = int(rng.integers(1, 4)); db = int(rng.integers(1, 4))
        train_pairs.append((int(rng.integers(10**(da-1), 10**da)),
                             int(rng.integers(10**(db-1), 10**db))))
    train_traces = [generate_trace(a, b) for (a, b) in train_pairs]
    print(f"n_train={n_train}, mean trace_len="
          f"{np.mean([t.shape[0] for t in train_traces]):.1f}")

    model = FlatLSTM(emb_dim=32, hidden=256, n_layers=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    train(model, train_traces, device, n_epochs=40, batch_size=64,
          lr=1e-3, log_every=5)

    # Eval (same buckets as GDC sweep).
    import random as _r
    _r.seed(43)
    def py_digit_pair(n_digits, n):
        return [(_r.randint(10**(n_digits-1), 10**n_digits - 1),
                 _r.randint(10**(n_digits-1), 10**n_digits - 1))
                for _ in range(n)]
    eval_buckets = [
        ('len-1 (in)',  py_digit_pair(1, 25)),
        ('len-2 (in)',  py_digit_pair(2, 25)),
        ('len-3 (in)',  py_digit_pair(3, 25)),
        ('len-4 OOD',   py_digit_pair(4, 25)),
        ('len-5 OOD',   py_digit_pair(5, 25)),
        ('len-7 OOD',   py_digit_pair(7, 25)),
        ('len-10 OOD',  py_digit_pair(10, 25)),
        ('len-15 OOD',  py_digit_pair(15, 25)),
        ('len-20 OOD',  py_digit_pair(20, 25)),
    ]

    print(f"\n=== LSTM eval, same buckets as GDC ===")
    for bucket_name, pairs in eval_buckets:
        t0 = time.time()
        n_correct = 0; action_correct = 0; action_total = 0
        for a, b in pairs:
            res = lstm_forecast_one(model, a, b, device, max_steps=600)
            ok = (res['predicted_output'] == a + b)
            n_correct += int(ok)
            gt = generate_trace(a, b)
            gt_post_init = gt[len(_make_init_rows(a, b)):]
            gt_actions = [(int(r[4]), int(r[5])) for r in gt_post_init]
            L = min(len(res['predicted_actions']), len(gt_actions))
            for i in range(L):
                if res['predicted_actions'][i] == gt_actions[i]:
                    action_correct += 1
            action_total += L
        rate = 100.0 * n_correct / len(pairs)
        act_rate = (100.0 * action_correct / action_total
                    if action_total else 0.0)
        print(f"  [{bucket_name:>13s}]  exact={n_correct:>2d}/25 ({rate:>5.1f}%)  "
              f"action={act_rate:>5.1f}%   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
