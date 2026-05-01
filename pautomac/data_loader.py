"""PAutomaC file loaders."""

from __future__ import annotations
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "PAutomaC-competition_sets")


def _load_seq_file(path):
    """Read a .train or .test file.  Returns (sequences, alphabet_size).
    sequences is a list of int64 numpy arrays (possibly empty)."""
    with open(path) as f:
        header = f.readline().strip().split()
        n_seqs = int(header[0]); alphabet = int(header[1])
        seqs = []
        for _ in range(n_seqs):
            parts = f.readline().strip().split()
            length = int(parts[0])
            if length == 0:
                seqs.append(np.zeros(0, dtype=np.int64))
            else:
                seqs.append(np.asarray([int(x) for x in parts[1:1 + length]],
                                       dtype=np.int64))
    return seqs, alphabet


def load_problem(i, data_dir=DATA_DIR):
    """Return dict with train, test, true_probs, alphabet_size."""
    train, A = _load_seq_file(os.path.join(data_dir, f"{i}.pautomac.train"))
    test, A2 = _load_seq_file(os.path.join(data_dir, f"{i}.pautomac.test"))
    assert A == A2, f"Alphabet mismatch for problem {i}: {A} vs {A2}"
    sol_path = os.path.join(data_dir, f"{i}.pautomac_solution.txt")
    with open(sol_path) as f:
        n = int(f.readline().strip())
        true_probs = np.asarray(
            [float(f.readline().strip()) for _ in range(n)],
            dtype=np.float64)
    assert len(test) == len(true_probs), (
        f"Test/solution length mismatch for problem {i}: "
        f"{len(test)} vs {len(true_probs)}")
    return {'index': i, 'train': train, 'test': test,
            'true_probs': true_probs, 'alphabet_size': A}


def summary(problem):
    train = problem['train']; test = problem['test']
    train_lens = np.asarray([len(s) for s in train])
    test_lens = np.asarray([len(s) for s in test])
    return {
        'problem': problem['index'],
        'alphabet_size': problem['alphabet_size'],
        'n_train': len(train),
        'n_test': len(test),
        'train_total_tokens': int(train_lens.sum()),
        'test_total_tokens': int(test_lens.sum()),
        'train_len_mean': float(train_lens.mean()),
        'train_len_max': int(train_lens.max()) if len(train_lens) else 0,
        'test_len_mean': float(test_lens.mean()),
        'test_len_max': int(test_lens.max()) if len(test_lens) else 0,
    }


if __name__ == "__main__":
    import json
    for i in [1, 2, 3, 7, 25, 48]:
        try:
            p = load_problem(i)
            print(json.dumps(summary(p), indent=2))
        except FileNotFoundError as e:
            print(f"problem {i}: missing file ({e})")
