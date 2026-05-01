"""Download and extract the PAutomaC competition archive."""

from __future__ import annotations
import os
import sys
import tarfile
import urllib.request

URL = "https://grammarlearning.org/files/2020/05/PAutomaC-competition_sets.tar.gz"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
ARCHIVE = os.path.join(DATA_DIR, "pautomac.tar.gz")
EXTRACTED = os.path.join(DATA_DIR, "PAutomaC-competition_sets")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isfile(ARCHIVE):
        print(f"Downloading {URL} ...", flush=True)
        urllib.request.urlretrieve(URL, ARCHIVE)
        print(f"  saved to {ARCHIVE}", flush=True)
    else:
        print(f"Archive already present: {ARCHIVE}", flush=True)
    if not os.path.isdir(EXTRACTED):
        print(f"Extracting to {EXTRACTED} ...", flush=True)
        with tarfile.open(ARCHIVE) as tf:
            tf.extractall(DATA_DIR)
    else:
        print(f"Already extracted: {EXTRACTED}", flush=True)
    files = os.listdir(EXTRACTED)
    print(f"  {len(files)} files in {EXTRACTED}", flush=True)


if __name__ == "__main__":
    main()
