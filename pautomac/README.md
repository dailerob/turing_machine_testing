# PAutomaC evaluation

Scaffold for evaluating GDC and CHMM on the **PAutomaC 2012**
benchmark suite (Verwer, Eyraud, de la Higuera; JMLR 2014).

PAutomaC = "Probabilistic Automaton Competition." 48 synthetic
problems generated from PFAs / PNFAs / HMMs of varying size,
density, and emission ambiguity. Each problem:

* a **train set** of i.i.d. sequences from the target machine
* a **test set** of (held-out) sequences from the same machine
* a **solution** giving the *true* probability of each test sequence
  under the target machine

The PAutomaC metric ([§4 of the paper][1]) is

    Score(M) = 2^(- Σ_t  pT(t) · log2 pM(t))

where `pT(t)` is the true probability (from the solution file,
normalised over the test set) and `pM(t)` is the model's predicted
probability of `t` (also normalised over the test set). **Lower is
better; the minimum (= the perplexity floor) is `2^H(pT)`**.

[1]: https://proceedings.mlr.press/v21/verwer12a.html

## Folder layout

```
pautomac/
├── README.md              — this file
├── PAUTOMAC_FORMAT.md     — file-format spec
├── data/
│   ├── pautomac.tar.gz    — original archive (12 MB)
│   └── PAutomaC-competition_sets/
│       ├── 1.pautomac.train, 1.pautomac.test, ...
│       └── ... (48 problems)
├── download.py            — fetch + extract the tarball
├── data_loader.py         — load .train / .test / .solution
├── scoring.py             — official PAutomaC scoring + aux baselines
├── models.py              — model wrappers (Uniform, Unigram, CHMM, GDC)
├── run_eval.py            — main runner across (problem, model)
└── results/               — generated CSVs and figures
```

## Quick start

```bash
# 1. Get the data (already done in this worktree).
python pautomac/download.py

# 2. Run the full sweep (uniform / unigram / CHMM K=2,4,8 / GDC).
python pautomac/run_eval.py --problems 1,2,3,4,5

# 3. Or specify a single problem.
python pautomac/run_eval.py --problems 1
```

## What's implemented

* PAutomaC official scoring (perplexity, entropy floor, lift over
  uniform / unigram).
* **Uniform** baseline: constant 1/|alphabet|.
* **Unigram** baseline: maximum-likelihood symbol distribution from
  training data.
* **CHMM** scorer (Dedieu et al, via upstream `chmm_actions`) with
  end-of-sequence token.
* **GDC** scorer with end-of-sequence token, computed via
  forward-marginal next-symbol distributions.

## Data citation

> Sicco Verwer, Rémi Eyraud, Colin de la Higuera. *Results of the
> PAutomaC Probabilistic Automaton Learning Competition.* JMLR
> Workshop and Conference Proceedings 21:243-248, ICGI 2012.

The data archive is hosted at
[grammarlearning.org/pautomac](https://grammarlearning.org/pautomac/).
Use is for academic research only.
