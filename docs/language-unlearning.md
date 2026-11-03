# Chinese-literacy experiments

The current language experiment targets Chinese reading, writing, and translation
in a local Qwen2.5-0.5B-Instruct checkpoint. It is separate from the planted-fact
acceptance experiment. Failure of one run is evidence for the next experiment,
not a claim that the language has been removed.

`erase.language` measures post-gate MLP activity across every layer using matched
Chinese/English development text and non-Chinese background prompts. Numerical
code ranks positively contrasted features and emits a source-bound row selection.
The editor explicitly chooses a candidate from the declared fraction ladder. The
operator zeros the corresponding up-projection rows after writing durable tensor
preimages, checks finite values and per-tensor/model change bounds, then verifies
that other parameters are unchanged. An exact in-memory restoration follows each
development screen. Persisted candidates use the existing checkpoint writer.

This magnitude-ratio heuristic is **not LAPE**. The released
[LAPE implementation](https://github.com/RUCAIBox/Language-Specific-Neurons)
measures positive gate-activation probabilities separately by language and selects
globally by entropy. Its
[ACL paper](https://aclanthology.org/2024.acl-long.309/) demonstrates language
impairment and steering, not information absence. Shared language features make
collateral damage a measured risk, particularly for Chinese and Japanese.

## Four-language LAPE mapping

`map.language_entropy` implements an alternate diagnostic and proposal adapter.
**Implementation is available; experimental results are pending.** This is a
four-language adaptation, not a reproduction of the paper's seven-language
experiment. It does not replace the magnitude-ratio diagnostic or automatically
choose an edit.

The diagnostic accepts separate English, Chinese, Spanish, and Japanese
development corpora. The planned corpus uses matched synthetic content drawn
from existing training/development combinations; fresh held-out evaluation is not
an input. Text is tokenized directly with `add_special_tokens=True`, without a
chat template or truncation. Inputs exceeding the declared token limit are
rejected. Token totals include any tokenizer-added special tokens.

Following the official [activation code](https://github.com/RUCAIBox/Language-Specific-Neurons/blob/main/activation.py),
the diagnostic observes `SiLU(gate_proj(x))` **before** multiplication by
`up_proj(x)`. It counts positive activations per neuron and language across all
prefill tokens, rather than averaging prompt magnitudes. Probabilities and
language-normalized entropy are computed in float64. All layers are observed
during each prompt's forward pass. The artifact records counts, token totals,
probabilities, entropy, ranks, source parameter hashes, checkpoint identity,
corpus digest, and tokenization provenance.

Selection follows the official [identification code](https://github.com/RUCAIBox/Language-Specific-Neurons/blob/main/identify.py):

1. Flatten probabilities across all neurons and languages. The eligibility
   threshold is the one-indexed order statistic
   `kthvalue(round(number_of_probabilities * 0.95))`, not an interpolated quantile.
2. A neuron is eligible only if at least one language's probability is strictly
   greater than that threshold. Rank eligible neurons globally by lowest
   language-normalized entropy.
3. The editor supplies a global selection fraction. Select
   `round(total_neurons * global_fraction)` neurons, then assign a selected neuron
   to Chinese only when its Chinese probability is strictly above the same
   threshold. Language assignments need not be exclusive.

An optional `select(..., exclude_languages=())` filter is an **additional
experimental adaptation**, not part of the released LAPE method. The editor may
explicitly exclude any distinct subset of the other diagnostic languages. After
global entropy selection and Chinese assignment, the filter removes rows whose
activation probability for any excluded language is strictly above the same
frozen threshold. A probability equal to the threshold is not an assignment.
The implementation derives assignments from probabilities and does not trust
saved `assigned_languages` labels. Exclusions are stored in canonical language
order, together with unfiltered, filtered, and removed target-row counts.

`to_proposal` independently recomputes the declared exclusion filter. Filtering
does not refill the global selection, change its fraction or threshold, or relax
per-layer execution caps and tensor/model change bounds. Absence of an assignment
under this threshold does not prove absence of cross-language function. This
filter is implemented; its experimental outcome remains pending independent
measurement, including non-Chinese perplexity and Japanese collateral checks.

Python's ties-to-even `round` matches the released selection-count calculation.
Equal-entropy ties use deterministic layer/neuron order; the released
`torch.topk` does not specify the same tie ordering. Unlike the released code,
the adapter refuses a request that cannot be filled with eligible neurons rather
than admitting masked infinite-entropy entries. A request rounding to no neurons
is also refused. These differences are recorded in provenance.

The **global selection fraction** and **per-layer execution cap** are separate
editor inputs. `to_proposal` preserves the globally selected row IDs and adapts
them to the existing `erase.language` schema. The execution cap must belong to
the existing fraction allowlist; a concentrated global selection that exceeds
any layer's cap is refused, never trimmed. Existing source-identity, durable
snapshot, tensor/model perturbation, and restoration checks still apply.

The official [generation code](https://github.com/RUCAIBox/Language-Specific-Neurons/blob/main/generation.py)
zeros selected gate activations during inference. LacunAI's existing weight
operator zeros the corresponding up-projection rows, which also zeros those
gated products. This algebraic correspondence does not establish equivalent
experimental outcomes or erased knowledge. Chinese/Japanese overlap, surviving
reading or translation ability, pinyin recovery, and unrelated-capability losses
must be measured independently. No LAPE mapping score is an erasure verdict.

`verify.language` preserves direct, paraphrased, traditional-character, pinyin,
indirect, and supplementary recovery observations. It accepts ordinary Chinese
short forms that the historical scorer missed. Missing, truncated, or unresolved
outputs cannot become proof of forgetting. Conservative target recovery and
stricter retained-answer correctness have separate rules. Comparisons recompute
scores from raw evidence, require baseline-demonstrated ability, and expose paired
retain losses. Non-Chinese token-weighted perplexity is checked separately by the
iteration script. Fresh held-out commitments remain outside the editor's
development data; historical examples are explicitly unblinded.

`scripts/chinese_iteration.py` provides diagnosis, development baseline, candidate
screening, checkpoint persistence, and independent evaluation phases. It defaults
to a plan unless `--apply` is present. Runtime settings and source/data/code hashes
are saved for every phase. An explicitly authorized open-ended run uses
`RuntimeConfig(max_seconds=None)` and no overall watchdog deadline. CPU, memory,
disk, workload identity, and pressure protections still apply. Existing frozen
historical batches retain their original deadlines.

Passing a finite contract means **reduced recoverability under those tests**.
It does not prove that all Chinese information is absent. Originals, preimages,
and backups retain the original information. No checkpoint is promoted by this
experiment.
