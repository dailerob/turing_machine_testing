"""Flat LSTM baseline for TM-trace prediction (mirrors TorchTMGDC's interface).

For each training tape, the trace is encoded as a 1D sequence of reduced
(read, write, dir) tuple-IDs. The LSTM is trained autoregressively: given
tokens 0..t-1, predict token t. At eval time:

    1. Read context = test tape tokens 0..t-1 → LSTM hidden state.
    2. Get the LSTM's next-tuple distribution.
    3. Constrain to tuple-IDs whose `read` matches the actual read at t
       (which the simulator/ground-truth trace provides).
    4. Argmax over those candidates → predicted tuple-ID.

Same `score_tapes_batched(tape_ids_list, actual_next_reads_list, by_read)`
interface so we can drop this into existing benchmark code or compare
side-by-side to TorchTMGDC.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64,
                  hidden: int = 256, n_layers: int = 2,
                  dropout: float = 0.0):
        super().__init__()
        # Embedding includes a PAD slot at index vocab_size.
        self.vocab_size = vocab_size
        self.pad_idx = vocab_size
        self.emb = nn.Embedding(vocab_size + 1, emb_dim,
                                  padding_idx=self.pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=n_layers,
                             batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor, state=None):
        emb = self.emb(x)
        if state is None:
            out, state = self.lstm(emb)
        else:
            out, state = self.lstm(emb, state)
        return self.head(out), state


class TorchTMLSTM:
    """LSTM analogue of TorchTMGDC with the same .fit/.score_tapes_batched API."""

    def __init__(self, hidden: int = 256, n_layers: int = 2,
                  emb_dim: int = 64, lr: float = 1e-3,
                  n_epochs: int = 30, batch_size: int = 32,
                  device: str = 'cuda', dtype=torch.float32):
        self.hidden = hidden; self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.lr = lr; self.n_epochs = n_epochs; self.batch_size = batch_size
        self.device = device; self.dtype = dtype

    # -----------------------------------------------------------------
    def fit(self, train_seqs, alphabet_size: int, verbose: bool = True):
        """train_seqs: list of 1D int64 arrays of tuple-IDs.
        alphabet_size: number of distinct tuple-IDs (= |reduced_alphabet|).
        """
        self.A = int(alphabet_size)
        self.N = self.A
        self.model = _LSTMSeqModel(self.A, self.emb_dim, self.hidden,
                                    self.n_layers).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        seqs = [s for s in train_seqs if len(s) >= 2]
        if not seqs:
            return
        # Convert to tensors (per-seq).
        seq_tensors = [torch.as_tensor(s, dtype=torch.int64,
                                         device=self.device) for s in seqs]
        lengths = [len(s) for s in seqs]
        T_max = max(lengths)
        # Pre-pad into a single matrix once (since alphabet is small).
        n = len(seq_tensors)
        pad = self.model.pad_idx
        data = torch.full((n, T_max), pad, dtype=torch.int64,
                           device=self.device)
        for i, s in enumerate(seq_tensors):
            data[i, :lengths[i]] = s
        lens_t = torch.as_tensor(lengths, device=self.device)
        # Train.
        t0 = time.time()
        for ep in range(self.n_epochs):
            self.model.train()
            order = torch.randperm(n, device=self.device)
            ep_loss = 0.0; n_batches = 0
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                batch = data[idx]                         # (B, T_max)
                lens_b = lens_t[idx]                      # (B,)
                B, T = batch.shape
                # Input = batch[:, :-1]; target = batch[:, 1:]
                inp = batch[:, :-1]
                tgt = batch[:, 1:].clone()
                # Mask out positions beyond seq length (predicting next from
                # within length: valid for positions where input pos < len-1).
                pos = torch.arange(T - 1, device=self.device)[None]
                valid = pos < (lens_b[:, None] - 1)
                tgt = torch.where(valid, tgt, torch.full_like(tgt, -100))
                logits, _ = self.model(inp)               # (B, T-1, A)
                loss = F.cross_entropy(
                    logits.reshape(-1, self.A),
                    tgt.reshape(-1), ignore_index=-100, reduction='mean')
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); n_batches += 1
            if verbose and ((ep + 1) % max(self.n_epochs // 8, 1) == 0
                             or ep == 0):
                print(f"    epoch {ep+1:>3d}/{self.n_epochs}  "
                      f"loss={ep_loss/max(n_batches,1):.4f}  "
                      f"[{time.time()-t0:.1f}s]")
        if verbose:
            print(f"    LSTM trained in {time.time()-t0:.1f}s "
                  f"({sum(p.numel() for p in self.model.parameters())/1e3:.1f}k params)")

    # -----------------------------------------------------------------
    @torch.no_grad()
    def score_tapes_batched(self, tape_ids_list, actual_next_reads_list,
                              by_read):
        """For each tape, predict next-token tuple-IDs constrained by
        `by_read[next_read]` at each step. Returns a list of per-tape
        prediction arrays (length T-1 each).
        """
        self.model.eval()
        B = len(tape_ids_list)
        if B == 0:
            return []
        tapes_np = [np.asarray(t, dtype=np.int64) for t in tape_ids_list]
        actuals_np = [np.asarray(a, dtype=np.int64)
                       for a in actual_next_reads_list]
        Ts = [len(t) for t in tapes_np]
        T_max = max(Ts) if Ts else 0
        if T_max == 0:
            return [np.empty(0, dtype=np.int64) for _ in tapes_np]

        # Pre-pad inputs (with pad_idx). We'll run the LSTM in one batched
        # pass since lengths are similar enough.
        pad = self.model.pad_idx
        inp = np.full((B, T_max), pad, dtype=np.int64)
        for i, t in enumerate(tapes_np):
            inp[i, :len(t)] = t
        inp_t = torch.as_tensor(inp, device=self.device)
        logits, _ = self.model(inp_t)   # (B, T_max, A)

        # logits[b, t] predicts token at position t+1 from history 0..t.
        preds = [np.full(max(Ts[b] - 1, 0), -1, dtype=np.int64) for b in range(B)]
        # For each (b, t) where we need a prediction (t < Ts[b] - 1):
        # next_read = actuals_np[b][t]
        # candidates = by_read[next_read]
        # pred = candidates[argmax(logits[b, t, candidates])]
        logits_np = logits.cpu().numpy()
        for b in range(B):
            T_b = Ts[b]
            for t in range(T_b - 1):
                read = int(actuals_np[b][t])
                cands = by_read.get(read, [])
                if not cands:
                    preds[b][t] = 0
                    continue
                cand_arr = np.asarray(cands, dtype=np.int64)
                cand_logits = logits_np[b, t, cand_arr]
                preds[b][t] = int(cand_arr[int(np.argmax(cand_logits))])
        return preds
