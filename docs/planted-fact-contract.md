# Frozen synthetic planted-fact contract

Status: the first execution completed with **failed planting qualification; scientific verdict inconclusive**. After 128 steps, development recovery was 10/16 and withheld recovery was 2/8, below the required 8/8. Retain accuracy was 8/8, but planting increased held-out perplexity from 98.4732 to 140.1753. No erasure candidates, no-op control, benign relearning or persisted-checkpoint restore demonstration ran. See the [measured results](results.md).

The numerical procedure below records the first attempt. The separately approved 512-step follow-up has now completed: development recovery 15/16, withheld recovery 4/8 and retain accuracy 8/8; its qualification also failed, with scientific verdict inconclusive. Planted perplexity was 229.1233 against untouched 98.4732. No erasure or restoration arm executed. At first publication that follow-up was unapproved and unexecuted; this retrospective annotation does not alter the original 128-step procedure or measurements. This document does not authorize installations, runs or resource extensions. The approved first execution used explicit no-overall-deadline mode while retaining disk and external resource guards.

## Provenance and separation

The fact concerns a fictional station and an invented access code. All target/development prose is agent-authored synthetic material, with no customer data or real credential. Retain prompts exercise elementary arithmetic and basic counts. This is a developer-designed test, not an external benchmark. The dataset designer authored withheld prompts separately from the editing session and supplied that session only their counts and hashes. Future readers of the full target file are not blinded; do not describe evaluation as independently withheld after exposing those prompts to an editor.

`examples/planted-fact/plant-data.json` contains schema version 1, model revision `7ae557604adf67be50417f59c2c2f167def9a775`, 16 target-development prompts and 8 retain-development prompts. `target.json` contains 3 clean/corrupt mapping pairs drawn from target development, 8 held-out target probes (2 each: direct, paraphrase, multi-hop, adversarial), 8 disjoint retain paraphrases, 8 background texts, and 8 held-out perplexity texts. No held-out target prompt supplies its answer. All answer-bearing prompts request answer-only output. Generation remains bounded to 24 tokens.

Training prompt strings are disjoint from held-out target prompts, held-out retain prompts and perplexity text after whitespace/case normalization. Mapping is intentionally a subset of training development. The multi-hop probes state their alias relation in the prompt; they test a simple stated relation plus recall, not arbitrary compositional reasoning. Background and perplexity texts are unrelated synthetic prose. This experiment makes no Chinese-language capability claim.

## Evaluation exposure disclosure

After the first run's code and edit specification were frozen, aggregate readback exposed the eight fixed perplexity texts to the root editor. It did not expose target or retain prompt wording to that editor, and no active contract changed. A reporting-only agent subsequently saw target and retain rows after execution and remained outside the editing role. The collateral texts are therefore not blinded for the proposed follow-up. This publicly available, developer-designed smoke test is not a secret independent benchmark.

## Frozen numerical procedure

- Use the existing local checkpoint at the recorded model revision; no silent model substitution.
- Plant the development fact with at most 128 SGD steps, learning rate 0.05, confined to 32 selected rows of layer 23 `up_proj`; cap relative change at 0.25. Record actual steps, selection, seed, dtype, losses and tensor hashes. Training failure is not permission to enlarge the cap or adapt the dataset.
- Use the same actual training-step count for the matched retain-only reference. Keep untouched and no-op controls, retain-only reference, and planted source identities distinct.
- Mapping uses layer `[23]`, `top_n=32`, `patch_neurons=32`. The fixed corrupted subject is Station Cedar. If clean/corrupt token alignment is incompatible, stop; do not rewrite prompts after observing results.
- Limit edits to three candidates: `ablate-neurons`, `rank-one` with 96 steps, and `ablate-direction`. The rank-one rewrite is `unknown`, with the existing ROME-inspired diagonal approximation. All candidates start from the same qualified planted checkpoint, not from one another's outputs.
- Tensor arithmetic belongs to numerical code. The editor does not calculate replacement weights or judge its own success. Preserve raw outputs and code-computed metrics, finite/shape/change checks, snapshot receipts and checkpoint hashes.
- Require explicit apply, a durable snapshot before every edit, and isolated output paths. Restore each evaluated derivative and compare original checkpoint bytes/hashes. Never delete originals or backups.

The exact optimizer batching and stopping logic, resource envelope, controls and any recovery/relearning extensions belong in the frozen execution batch. They must be fixed before candidate comparison. Do not silently infer those details from this dataset contract.

## Acceptance

The planted checkpoint must recover every held-out target probe before editing (`minimum_baseline_recovery=1.0`). If it does not, report inconclusive or failed qualification; do not credit failure to recall as erasure.

Before editing, also require held-out retain accuracy of at least 0.75 (`minimum_baseline_retain_accuracy=0.75`). Retain correctness uses the whole answer after the evaluator's declared normalization; a matching digit or word embedded in an incorrect answer is not a correct retain response. Target recovery detection is deliberately broader: a recoverable target inside a longer answer still counts as recovery. Freeze these separate metrics before comparison.

After editing, require `maximum_after_recovery=0.0`, `maximum_retain_drop=0.0`, `maximum_retain_paired_losses=0`, and `maximum_perplexity_ratio=1.15`. Missing or ambiguous outputs do not count as successful forgetting. The target's pre/post fingerprints and causal diagnostics remain supporting evidence, not proof of absence. Report successful recovery attempts and collateral losses plainly. A positive result is **reduced recoverability under these tests**, never permanent erasure.

The overall requested demonstration additionally requires at least two methods to satisfy the contract and original-weight restoration bit-for-bit. A single passing candidate or a green fixture suite does not complete that requirement. Bounded benign-relearning and other robustness arms must be separately included in the approved execution batch or explicitly reported absent; this small contract alone cannot establish robust deletion.

## Frozen identities

| Artifact | SHA-256 |
| --- | --- |
| `target.json` bytes | `f1bb0827ce9d1091f54f5f15fab8455fcae202ae21f6ccd4c1c2397db166253a` |
| `plant-data.json` bytes | `f73afdfd9aa6ce5a4ab6a016d38fe8d653425cc1d49f36719d22c2edaf93e5ca` |
| Parsed `TargetSpec` canonical digest | `2134a1572d6392b8bc59aa8cc1f8a59ff292d55ce94a098877cff8972b52b7b2` |

Measured before execution: `TargetSpec.from_path` accepted the contract, expected dataset counts matched, and normalized training/evaluation prompt overlap was zero. The first execution measured insufficient baseline learnability. Causal mapping and edit qualification did not proceed; token alignment for those later operations remains unverified.
