"""Replication of the Dedieu et al. 2019 (arXiv:1905.00507) Table 4
character-level language modeling protocol.

Section 4.2 of the paper specifies (verbatim):
  "For each data set, we first transform all numbers to their character
   forms and replace all words which appear only once in the training
   or only in the test set into the word rare. We then remove all
   characters other than the 26 letters and space. The first 90% of
   our data is used as training set, and the remaining 10% as the
   test set. In addition, for computational reason, we limit our
   training set size to 750,000: this enforces us to reduce the
   training set ratio on some data sets."

This module loads the eight datasets from their public sources and
applies that preprocessing pipeline. Output train/test sizes are
checked against the paper's Table 4 numbers and any drift is reported.

Datasets:
  - blake-poems, carroll-alice, shakespeare-hamlet, shakespeare-macbeth,
    milton-paradise, melville-mobydick : NLTK gutenberg corpus
  - war-peace : Karpathy's char-rnn repo
  - calgary-book1 : Calgary corpus, UCI mirror
"""
from __future__ import annotations
import os, re, urllib.request, hashlib
from collections import Counter
import numpy as np
import nltk
from nltk.corpus import gutenberg
from num2words import num2words

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Paper's reported Table 4 sizes (for sanity-checking our preprocessing)
# --------------------------------------------------------------------
PAPER_SIZES = {
    'blake-poems':         (29912,   3300),
    'carroll-alice':       (118931, 13063),
    'shakespeare-hamlet':  (130101, 14332),
    'shakespeare-macbeth': (79646,   8824),
    'milton-paradise':     (382942, 42297),
    'melville-mobydick':   (750000, 387864),
    'war-peace':           (750000, 2237883),
    'calgary-book1':       (638677,  7116),
}

NLTK_FILES = {
    'blake-poems':         'blake-poems.txt',
    'carroll-alice':       'carroll-alice.txt',
    'shakespeare-hamlet':  'shakespeare-hamlet.txt',
    'shakespeare-macbeth': 'shakespeare-macbeth.txt',
    'milton-paradise':     'milton-paradise.txt',
    'melville-mobydick':   'melville-moby_dick.txt',
}

# Karpathy's char-rnn never published a warpeace dataset; Project
# Gutenberg's "War and Peace" #2600 is the standard source the paper
# is approximating. After our preprocessing the size lands near the
# paper's 750k+2.24M ~= 3M-char total.
WAR_PEACE_URL = 'https://www.gutenberg.org/files/2600/2600-0.txt'
CALGARY_URL = 'https://corpus.canterbury.ac.nz/resources/calgary.tar.gz'

ALPHABET = list('abcdefghijklmnopqrstuvwxyz ')
CHAR_TO_ID = {c: i for i, c in enumerate(ALPHABET)}
ID_TO_CHAR = {i: c for i, c in enumerate(ALPHABET)}
ALPHABET_SIZE = len(ALPHABET)  # 27


# --------------------------------------------------------------------
# Raw text fetch
# --------------------------------------------------------------------
def _ensure_nltk():
    try:
        gutenberg.fileids()
    except LookupError:
        nltk.download('gutenberg', quiet=True)


def _fetch(url, fname):
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        return path
    print(f"  Downloading {url} → {path}")
    urllib.request.urlretrieve(url, path)
    return path


def get_raw_text(name):
    """Return raw text for a named dataset."""
    _ensure_nltk()
    if name in NLTK_FILES:
        return gutenberg.raw(NLTK_FILES[name])
    if name == 'war-peace':
        path = _fetch(WAR_PEACE_URL, 'warpeace_2600.txt')
        return open(path, encoding='utf-8-sig').read()
    if name == 'calgary-book1':
        archive = os.path.join(DATA_DIR, 'calgary.tar.gz')
        if not os.path.exists(archive):
            print(f"  Downloading {CALGARY_URL} → {archive}")
            urllib.request.urlretrieve(CALGARY_URL, archive)
        import tarfile
        book1_path = os.path.join(DATA_DIR, 'book1')
        if not os.path.exists(book1_path):
            with tarfile.open(archive) as tf:
                tf.extract('book1', DATA_DIR)
        return open(book1_path, encoding='latin-1').read()
    raise KeyError(name)


# --------------------------------------------------------------------
# Preprocessing pipeline
# --------------------------------------------------------------------
_NUMBER_RE = re.compile(r'\d+')


def _numbers_to_words(text):
    """Replace each contiguous run of digits with its English word form."""
    def sub(m):
        try:
            return ' ' + num2words(int(m.group(0))) + ' '
        except Exception:
            return ' '
    return _NUMBER_RE.sub(sub, text)


def _strip_to_alphabet(text):
    """Lowercase, replace any char not in {a-z, space} with space, collapse runs."""
    text = text.lower()
    text = re.sub(r'[^a-z]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _replace_rare_words(train_str, test_str):
    """Replace words that are rare with the literal token 'rare'.

    Paper: 'replace all words which appear only once in the training
    or only in the test set into the word rare'. We interpret this as:
    (a) words with training count == 1 (hapax legomena in train), and
    (b) words appearing in test but with training count == 0 (OOV).
    The union becomes the rare-set; all occurrences are replaced.
    """
    train_tokens = train_str.split(' ')
    test_tokens = test_str.split(' ')
    train_count = Counter(t for t in train_tokens if t)
    test_words = set(t for t in test_tokens if t)
    rare_words = {w for w, c in train_count.items() if c == 1}
    rare_words |= {w for w in test_words if train_count.get(w, 0) == 0}
    train_out = ' '.join('rare' if t in rare_words else t for t in train_tokens)
    test_out = ' '.join('rare' if t in rare_words else t for t in test_tokens)
    return train_out, test_out, len(rare_words)


def preprocess(name, verbose=True):
    """Apply the paper's preprocessing pipeline; return (train_str, test_str).

    Returns
    -------
    train_str, test_str : str
        Final character sequences over alphabet {a-z, space}.
    info : dict
        Lengths, hashes, OOV count, paper-comparison.
    """
    raw = get_raw_text(name)
    if verbose:
        print(f"[{name}] raw chars: {len(raw):,}")

    # 1. Numbers → words
    text = _numbers_to_words(raw)
    # 2. Strip to alphabet
    text = _strip_to_alphabet(text)
    if verbose:
        print(f"[{name}] post-strip chars: {len(text):,}")

    # 3. 90/10 split with 750k train cap
    n = len(text)
    cap = 750_000
    train_target = min(int(0.9 * n), cap)
    # If 90/10 produces train > cap, the train is capped at 750k and
    # test is everything after — which gives the very large mobydick/
    # war-peace test sizes in the paper.
    if int(0.9 * n) > cap:
        # Cap dominates: train = first 750k, test = remainder
        train_str = text[:cap]
        test_str = text[cap:]
    else:
        # Standard 90/10 on the cleaned text
        cut = int(0.9 * n)
        train_str = text[:cut]
        test_str = text[cut:]

    # 4. Rare-word replacement
    train_str, test_str, n_rare = _replace_rare_words(train_str, test_str)

    info = dict(
        name=name,
        n_train=len(train_str), n_test=len(test_str),
        n_rare_words=n_rare,
        paper_train=PAPER_SIZES[name][0], paper_test=PAPER_SIZES[name][1],
        alphabet_size=ALPHABET_SIZE,
    )
    if verbose:
        pt, ptst = PAPER_SIZES[name]
        dt = len(train_str) - pt; dtst = len(test_str) - ptst
        print(f"[{name}] train: {len(train_str):,} (paper {pt:,}, Δ={dt:+d}); "
              f"test: {len(test_str):,} (paper {ptst:,}, Δ={dtst:+d}); "
              f"rare-words: {n_rare}")
    return train_str, test_str, info


def encode(text):
    """Encode a char sequence to int IDs over the 27-symbol alphabet."""
    return np.array([CHAR_TO_ID[c] for c in text], dtype=np.int64)


def load(name, verbose=True):
    """Return (train_ids, test_ids, info) for a dataset."""
    train_str, test_str, info = preprocess(name, verbose=verbose)
    return encode(train_str), encode(test_str), info


if __name__ == '__main__':
    print(f"\nAlphabet size: {ALPHABET_SIZE} (a-z + space)")
    print(f"Paper Table 4 sizes vs our pipeline:\n")
    print(f"{'dataset':<22} {'paper_train':>12} {'paper_test':>12} "
          f"{'our_train':>12} {'our_test':>12} {'Δtrain':>8} {'Δtest':>8}")
    print('-' * 90)
    for name in PAPER_SIZES:
        try:
            tr, te, info = load(name, verbose=False)
        except Exception as e:
            print(f"{name:<22} skipped ({e!r})")
            continue
        pt, ptst = PAPER_SIZES[name]
        print(f"{name:<22} {pt:>12,} {ptst:>12,} {len(tr):>12,} {len(te):>12,} "
              f"{len(tr)-pt:>+8d} {len(te)-ptst:>+8d}")
