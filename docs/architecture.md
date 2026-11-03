# LacunAI experimental architecture

LacunAI isolates research into model-information recoverability from NeurAIlyzer's
time-based agent-state hygiene. Researchers need bounded, reproducible edits;
operators need resource isolation and preserved originals; reviewers need raw
evidence that is independent of the editor's assertions. The package contains no
state-wipe, snapshot-restore, MCP or production-model deployment adapter.

The research hypothesis is that an LLM can identify useful constrained edits from
diagnostic evidence. Behavioral change does not establish information deletion.
Reports distinguish failed, inconclusive and passed experimental contracts, and
describe any positive finding as reduced recoverability under the stated tests.

## Components and boundaries

| Component | Responsibility |
| --- | --- |
| `experimental.spec` | Admit the exact frozen contract and protected output paths. |
| `experimental.data` | Construct synthetic development, withheld and retain examples. |
| `experimental.tensors` | Diagnose activations, execute allowed tensor edits, train bounded baselines and validate checkpoints. |
| `validation` | Validate externally supplied frozen JSON through the same strict edit parser, without loading models. |
| `experimental.editor` | Parse declarative specifications; preserve the historical local-editor adapter separately. |
| `experimental.evaluation` | Score raw responses and compute comparisons against frozen criteria. |
| `experimental.runner` | Orchestrate controls, candidates and benign relearning; independently verify evidence. |
| `experimental.resources` | Observe host workloads and pressure; abort within the approved resource envelope. |
| `experimental.artifacts`, `experimental.prepare` | Bound transfers and writes, hash artifacts, preserve originals and report progress. |
| `experimental.batch`, `experimental.exercise` | Preserve the original default-off batch and its separately frozen engineering supplement. |

An editor supplies meaning and diagnostic identifiers. It does not supply numerical
replacement weights or executable code. Tensor code owns arithmetic, fixed strength
presets, shape and finite-value checks, source identity, change bounds and checkpoint
round trips. PyTorch and Transformers are imported only inside runtime functions.
Scalar reporting uses code-computed results, including Decimal where appropriate.

The execution sequence is contract admission, untouched-model qualification,
diagnosis, constrained proposal, numerical validation, independent behavioral
evaluation and explicit reporting. The original checkpoint remains immutable.
Checkpoint derivatives are source-hash-bound; no-op controls must preserve output
tokens and first-step logits. Baseline and retain-only controls, retained raw
outputs, recovery probes and bounded benign relearning limit what a result can claim.
Promotion is disabled. Originals and backups still contain the original information.

## Current-session supplied specifications

`lacunai validate-spec` requires explicit opt-in, a contract, development diagnostics,
the specification file and its independently frozen SHA-256. It reuses `parse_spec`
and the exact tensor/operation/strength allowlists. A valid response means only that
the schema and input hashes passed. It grants no execution approval, performs no
numerical edit and establishes no checkpoint validity or unlearning result.

This route does not invoke a local or hosted editor. The session supplying the
specification must separately disclose its model identity when known, data
recipients, costs and evidence exposure. Evaluation must remain independent of the
editor. Historical evaluation seen by an editor must be labeled unblinded.

## Historical experiment versus future runs

The package was separated from the experimental modules initially developed beside
NeurAIlyzer. The namespace changed to `lacunai`; the numerical and evaluation
algorithms were retained. Path protection was extracted without importing the
state-hygiene configuration or adapters. The `neurailyzer-pilot-20261102` artifact
directory name, `neurailyzer-base-plus-delta-v1` checkpoint wire format, frozen
contract digests and `NEURAILYZER_ARTIFACT_BUDGET_ROOT` environment name remain as
historical identities. A newly staged source bundle must name the LacunAI source
path; it cannot masquerade as the earlier frozen source bundle.

The historical local-editor batch is exposed as `lacunai experimental run`, with
`scripts/experimental_batch.py` and `scripts/experimental_exercise.py` preserving
its execution protocols. Their existing admission checks are specific to the
frozen pilot. They are not general authorization for another dataset, model,
provider, machine or budget. Future experiments need a newly reviewed, explicitly
approved contract and corresponding admission support.

The running corrected pilot uses a separately frozen private session-spec executor,
not a new LacunAI runtime entrypoint. It retains its original source namespace and
artifacts while running. It records the earlier incorrect local-editor attempt as
historical provenance and uses the current session's frozen specification. The
original baseline qualification failed; subsequent engineering evidence is
permanently inconclusive for that original efficacy contract. A fresh post-freeze
audit is separate from historical examples already seen by the editor.

This repository port does not establish that the LacunAI namespace has run a live
model experiment. Local parser, orchestration and guard tests provide software
evidence only. Optional real-tensor tests report skips when the ML dependencies
are absent. Completed controls, a nonzero checkpoint, all required comparisons and
relearning arms, and independent artifact verification are necessary before an
engineering exercise can be called complete. No success is inferred from progress.

## Verification and approval boundaries

Tests exercise the default-off CLI, corrupt inputs, strict edit grammar, changed
checkpoint sources, bounded writes, pressure aborts, raw-result recomputation and
guard-removal controls. Numerical tests use small real tensors when the optional
runtime is available. Production checkpoints, private customer data, automatic
promotion and deletion of originals are outside this architecture.

Private approval receipts, model files, raw audit material and operational host
details belong outside the source repository. Builds and staged runs must identify
their source revision and hashes. Source review and test evidence do not replace
receiver-side artifact verification or separately approved model execution.
