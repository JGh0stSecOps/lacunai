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

## Full-range distribution mapping

`map.language_distribution` streams two passes over the existing four-language
development corpus. It records per-neuron ranges, then 64 shared histogram bins
including negative activations. Counts use checked int32 accumulation; normalized
probabilities and overlaps use float64. The Chinese contrast is one minus its
largest distribution overlap with English, Spanish or Japanese. Stable ties use
layer and neuron order. This is an association score, not calibrated confidence
or causal localization.

After inspecting the diagnosis, the editor may explicitly request `top_k` rows
with `to_proposal`. Numerical code recomputes the ranking from saved counts and
checks the existing per-layer, tensor and model bounds without trimming the
selection. The proposal reuses the existing reversible up-projection operator;
constructing it does not apply the edit.

`scripts/chinese_distribution_diagnosis.py` requires explicit `--apply` for model
execution. It preserves corpus and checkpoint identities, records numerical and
resource settings, writes a bounded safetensors payload, and verifies unchanged
source weights and exact artifact readback. It does not select or execute an edit.
The method adapts [distribution-aware language-neuron research](https://arxiv.org/html/2609.10993v1)
without reproducing its language-clustering procedure. Model outcomes require
separate causal and behavioral tests.

The [versioned evaluator correction](research.md#evaluator-migration-demonstrated-baseline-skills)
separates uncertain baseline items from demonstrated skills and restores the
original receptive percentage-point criterion. After-edit uncertainty and any
confirmed recovery still prevent acceptance. Historical results retain their
original contracts and signatures.

## Four-language LAPE mapping

`map.language_entropy` implements an alternate diagnostic and proposal adapter.
Development screens have executed on the local Qwen checkpoint; none has passed
the frozen contract. This is a
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
filter has been exercised in development screens. Chinese recovery remained;
these early rejections do not establish Japanese preservation or acceptable
non-Chinese perplexity for the rejected candidates.

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

Scorer `4-retain-answer-wrapper` corrects a retain-only normalization error: a
single leading `Answer:` label may precede an otherwise complete correct answer.
The remainder must match a whole answer; the wrapper cannot admit extra prose,
negation, competing answers, or nested labels. Chinese recovery rules and
acceptance thresholds are unchanged. Historical signed reports retain their
original scorer version. Separate rescoring reports expose changed judgments;
missing observations remain missing.

`scripts/chinese_replay_candidate.py` re-evaluates the explicitly pinned, signed
stored RMU candidate without retraining. It writes a new snapshot, validates the
actual loaded tensor hashes and outer perturbation bounds, runs the complete
development suite, and restores the original tensors. This supports evaluation
repairs without silently changing the candidate or overwriting earlier evidence.

`scripts/chinese_iteration.py` provides diagnosis, development baseline, candidate
screening, checkpoint persistence, and independent evaluation phases. It defaults
to a plan unless `--apply` is present. Runtime settings and source/data/code hashes
are saved for every phase. An explicitly authorized open-ended run uses
`RuntimeConfig(max_seconds=None)` and no overall watchdog deadline. CPU, memory,
disk, workload identity, and pressure protections still apply. Existing frozen
historical batches retain their original deadlines.

## Constrained objective optimization

`scripts/chinese_gradient_candidate.py` executes a hash-pinned edit specification,
not editor-generated code. Its NPO adaptation follows the sequence-level objective
in [OpenUnlearning](https://github.com/locuslab/open-unlearning/blob/main/src/trainer/unlearn/npo.py),
with answer-position `KL(reference || current)` for retention. The reference uses
the original allowed tensors in the same resident model; no student backward graph
survives that temporary substitution. The runner records the reference NLL cache,
per-step losses, actual example IDs, and source-precision weight changes.

The NPO schedule cycles through every training family, preserving family and
within-family order. Its retain-only control uses the identical retain schedule.
Qwen computes output logits only at answer-prediction positions, including EOS;
a tiny Qwen numerical test compares losses and parameter gradients against full
logits. Historical gradient runs retain their original recorded schedules.
Changing the objective, schedule, tensors, or settings creates a new experiment.

The root-selected NPO learning rate must be one of the exact decimal strings
`0.01`, `0.05`, `0.1`, or `0.2`. This changes neither the optimizer-step count nor
the tensor and whole-model perturbation bounds. The historical GradDiff contract
remains fixed. Each candidate records its selected value; no sweep chooses or
promotes a candidate automatically.

An increased canonical-answer NLL does not establish reduced semantic recovery.
For example, suppressing the continuation `yellow` does not necessarily suppress
`The bird is yellow.` The loss measures the supplied token sequence, while the
independent scorer checks recoverable meaning across generated answers. Retain
losses and collateral tests remain necessary even when the forget loss improves.

## Representation misdirection

`scripts/chinese_rmu_candidate.py` exercises an RMU adaptation based on the
[released OpenUnlearning implementation](https://github.com/locuslab/open-unlearning/blob/a456aa277be14b836248e42b13b3befe47bb03f4/src/trainer/unlearn/rmu.py).
It targets the output of decoder block 7 and permits updates only to the MLP
down-projection weights in blocks 5, 6, and 7. Tensor code creates a fixed seeded
random control vector. Forget examples minimize masked hidden-state MSE toward
that vector; retained examples minimize MSE against original-model activations.
The masking includes the supplied answer and EOS positions. This changes the
objective from suppressing the probability of an exact answer sequence to changing
its internal representation; neither objective proves deletion.

The edit specification records the optimizer, coefficients, seed, clipping,
allowed tensors, step count, and perturbation bounds. The retained reference uses
the same resident model with original tensors temporarily restored before any
student graph is built. Prefix execution stops after the measured block; numerical
tests compare its activations and parameter gradients with a full model forward.
Raw recovery checks cover every required Chinese skill family before early
rejection. Any confirmed recovery still fails the unchanged acceptance contract.

This implementation is a constrained adaptation, not a replication of the paper's
models, datasets, or outcome. Its independent behavioral result remains a separate
requirement from correct loss arithmetic and reversible weight updates.

### Context-aware representation misdirection

The separate `rmu_context_candidate` contract retains the original optimizer,
control-vector seed, block, tensor allowlist, step count, and perturbation bounds.
The historical `rmu_candidate` and its answer/EOS masking remain available unchanged.
Its purpose is to test whether changing reading-context representations can reduce
recovery that survives an answer-only objective; this is a research hypothesis.

The context contract records the exact mask policy
`user_content_and_answer_for_reading;answer_for_writing;user_content_and_answer_for_retain:v1`
and variant policy `reading_cycle_existing_variants:v1`. Reading examples include
user-content and answer positions; writing examples use answer positions; retained
examples include user-content and answer positions. Reading variants cycle through
the existing development/training prompts rather than generating new examples.
The matched `retain_context_mse` mode uses the same retained-context objective.

The backend records selected variants, token batches, mask positions, and mask
hashes in the training trace. Masks and selected batches are validated before
optimization. The runner still snapshots before editing, validates actual BF16
changes, records all four recovery-preflight families, and restores the original
weights. No mask-policy change relaxes the zero-recovery or collateral thresholds.
This extends the constrained OpenUnlearning-inspired adaptation above; it does not
establish language erasure or independently reproduce a published result. Training
and a favorable numerical trace are not evidence that the behavioral contract passed.

## Retained-mean nullspace projection

`scripts/chinese_subspace_candidate.py` separates diagnosis, development screening,
and checkpoint persistence. Diagnosis collects signed prompt-mean MLP features on
matched Chinese, English, Spanish, and Japanese development text. Uncentered FP64
SVD identifies the sampled non-Chinese rowspace, then residual Chinese directions
orthogonal to that rowspace. Binary basis artifacts, spectra, numerical ranks,
boundary degeneracy, corpus hashes, and source hashes are recorded.

The editor explicitly selects layers, direction count, and strength. Numerical
code proposes `W' = W - strength * (W V) V^T` for allowed MLP down-projection
weights. It validates orthogonality, sampled retained-mean leakage, source identity,
and actual tensor/model change bounds after source-precision rounding. It snapshots
before application, saves after-images, and restores the original tensors after
screening. Persistence writes and reloads a separate checkpoint.

This is an experimental adaptation, not a replication of a validated literacy
erasure method. Preserving sampled original prompt means does not preserve every
token activation or later-layer state. Rounding leakage is measured rather than
assumed zero. Repeated storage rounding tests idempotence; projection-energy
retention is measured separately. Degenerate SVD bases can differ across LAPACK
implementations. A favorable map or numerical invariant cannot pass the behavioral
contract by itself.

Passing a finite contract means **reduced recoverability under those tests**.
It does not prove that all Chinese information is absent. Originals, preimages,
and backups retain the original information. No checkpoint is promoted by this
experiment.
