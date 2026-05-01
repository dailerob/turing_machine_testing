# PAutomaC file format and scoring

## File format

Each problem `i ∈ {1, ..., 48}` ships four files:

* `i.pautomac.train` — training sequences sampled from the target.
* `i.pautomac.test` — test sequences (also from the target).
* `i.pautomac_solution.txt` — true probabilities of the test sequences.
* `i.pautomac_model.txt` — full description of the target machine
  (states, transitions, emissions). Not used for evaluation but
  available for analysis.

### `.train` and `.test`

Plain text, space-separated.

```
<n_sequences> <alphabet_size>
<length> <s_1> <s_2> ... <s_length>
<length> <s_1> <s_2> ... <s_length>
...
```

* **Header**: number of sequences and alphabet size on a single line.
* **Body**: one line per sequence. The first integer is the
  sequence's length; the remaining `length` integers are the
  symbols. Length 0 = empty sequence (allowed; means "the machine
  halts immediately").
* Symbols are integers in `{0, 1, ..., alphabet_size − 1}`.
* No trailing markers; the sequence is implicitly terminated.

Example (`1.pautomac.train`):

```
20000 8
12 5 4 1 1 5 3 4 7 4 7 5 0
26 4 4 7 4 4 4 7 4 4 7 1 2 4 2 4 5 3 4 5 3 4 1 4 5 1 3
...
```

→ 20,000 training sequences, alphabet size 8, first sequence has
length 12 with symbols `(5, 4, 1, 1, 5, 3, 4, 7, 4, 7, 5, 0)`.

### `.pautomac_solution.txt`

```
<n_sequences>
<true_prob_of_test_seq_1>
<true_prob_of_test_seq_2>
...
```

* First line: integer count (matches `.test`'s sequence count).
* Each remaining line: a float, the true probability of the
  corresponding test sequence under the target machine.
* These probabilities are *not* normalised across the test set —
  they are the absolute probabilities under the target. The scoring
  metric divides by their sum (see below), so for evaluation we
  always normalise.

## Sequence likelihood under a model

For evaluation we need `pM(t)` for each test sequence `t`. For an
auto-regressive sequence model:

```
pM(t) = pM(s_1) · pM(s_2 | s_1) · ... · pM(s_T | s_{<T}) · pM(END | s_{1..T})
```

The final `pM(END | history)` term is the model's probability that
the sequence stops at length `T`. In a PFA this is built-in. For
GDC and CHMM we approximate it by:

* appending an explicit `END` token (= `alphabet_size`, i.e. one
  index past the original alphabet) to every training sequence;
* training the model on the augmented alphabet of size
  `alphabet_size + 1`;
* at scoring time, for a test sequence `t = (s_1, ..., s_T)` we
  prepend nothing, append `END`, and compute the joint probability
  via one-step predictives.

For the **uniform** baseline we report two sub-variants:

* **uniform-no-end**: `pM(t) = (1/A)^T` (ignores the geometric
  length distribution).
* **uniform-with-end**: `pM(t) = (1/(A+1))^T · (1/(A+1))` —
  treats `END` as just another symbol.

Both score similarly; we report the latter for fair comparison.

## Official PAutomaC scoring

Given the per-test-sequence true probabilities `pT(t)` and model
probabilities `pM(t)`, the official scoring metric is

```
Score(M) = 2^( - Σ_t  pT_norm(t) · log2 pM_norm(t) )
```

where:

* `pT_norm(t) = pT(t) / Σ_t' pT(t')`
* `pM_norm(t) = pM(t) / Σ_t' pM(t')`

both normalised over the test set.

* **Lower is better** (it's a perplexity).
* The *minimum* (perplexity floor) is `2^H(pT_norm)` — achieved
  iff `pM_norm = pT_norm` exactly.
* The *uniform* score is `|test set|`.

We report:

* `score` — Score(M) above.
* `entropy_floor` — `2^H(pT_norm)`, the best achievable.
* `gap` — `score − entropy_floor`, in perplexity units.
* `lift` — `(uniform_score − score) / (uniform_score − entropy_floor)`,
  fraction of the way from uniform-baseline to optimal. 0 = no
  better than uniform; 1 = optimal.

`lift` is the cleanest single-number summary across problems with
different alphabets and different test set entropies.
