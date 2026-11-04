# LacunAI experimental architecture

LacunAI isolates research into model-information recoverability from NeurAIlyzer's
time-based agent-state hygiene. Researchers need bounded, reproducible edits;
operators need resource isolation and preserved originals; reviewers need raw
evidence that is independent of the editor's assertions. Optional state and serving adapters are isolated from the core; no serving process is started automatically.

The research hypothesis is that an LLM can identify useful constrained edits from
diagnostic evidence. Behavioral change does not establish information deletion.
Reports distinguish failed, inconclusive and passed experimental contracts, and
describe any positive finding as reduced recoverability under the stated tests.

## ADR: one model-side service, optional state bridge

Status: implemented software interfaces and an executed single-fact experiment. Two methods passed that experiment's narrow contract; a fresh supplement was inconclusive. The [results](results.md) record both outcomes and the observed limitations.

Decision: keep NeurAIlyzer responsible for time-based state hygiene and put model-side work in LacunAI. CLI and MCP call one `core.service.Service`; neither owns a second numerical pipeline. NeurAIlyzer has no reverse dependency. `lacunai[state]` is an explicit optional dependency and does not make state clearing automatic.

| Package | Responsibility |
| --- | --- |
| `core` | Frozen target/runtime contracts, local HF loading, checkpoint identity, run records and orchestration. |
| `map` | Sparse prefill fingerprints, background IDF, causal activation patching, persistent query/export index. |
| `erase` | Allowlisted neuron/direction/rank-one operators with bounded changes. |
| `verify` | Independent probes and collateral comparison; immutable signed reports and raw evidence. |
| `restore` | Mandatory touched-tensor snapshots, isolated derivative writes and byte-checked restoration. |
| `adapters` | Explicit guarded local state clearing; signed-checkpoint vLLM launch preparation only. |
| `cli`, `mcp_server` | Shared service invocation, opt-in writes, structured results and visible progress. |

The built-in loader supports local safetensors Qwen2/Llama gated MLP checkpoints, CPU float32. It does not resolve hosted model names or download missing artifacts. Seeds, thread limits and time budgets are explicit. Map/locate/verify run inference when requested; erase/forget/restore default to plans and require `apply` for weight or state writes. Explicit in-place operation is supported only through the same snapshot path. A plan may write a run record and signed report, but never loads the model or changes weights.

A frozen target separates mapping development prompts, withheld recall probes and retain prompts. Fingerprints alone are correlational; activation patching supplies bounded causal evidence. `rank-one` uses a ROME-inspired diagonal approximation, not the full ROME/MEMIT algorithms. None of these methods proves knowledge is absent. The default CLI is not an autonomous editor agent: supplied target specifications and numerical methods are explicit. The historical constrained session-editor workflow remains available below.

Ed25519 report signatures cover report bytes and frozen supporting artifacts. Verification requires a separately pinned public key. Mutable operational metadata is frozen into signed evidence before later restoration events. Signing keys live outside source checkouts. A signature authenticates integrity against that key, not scientific validity.

The state bridge accepts only explicit configured local session/sandbox scopes, checks keep lists and best-effort liveness, rejects model/workspace overlap and refuses retention pruning. State deletion is whole-store scoped and cannot selectively remove the targeted fact. The vLLM hook checks a signed passed contract and exact checkpoint inventory and emits a loopback-only offline command; it never deploys or reloads a model. Live serving remains unverified.

Consequences: model numerical dependencies and the optional state/MCP dependencies can be installed separately. Core help remains usable without Torch. Test fixtures establish transport, arithmetic and restoration behavior. The completed single-fact experiment separately exercised baseline learning, two methods, independent reload, collateral checks and exact restoration. Surviving partial answers, malformed outputs, planting damage and the inconclusive fresh supplement prevent interpreting that result as information deletion.

## Historical components and boundaries

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
evaluation and explicit reporting. The historical protocol keeps the original checkpoint immutable.
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

The corrected pilot uses a separately frozen private session-spec executor,
not a new LacunAI runtime entrypoint. Its execution retains the original source
namespace and artifacts. It records the earlier incorrect local-editor attempt as
historical provenance and uses the current session's frozen specification. The
original baseline qualification failed; subsequent engineering evidence is
permanently inconclusive for that original efficacy contract. A fresh post-freeze
audit is separate from historical examples already seen by the editor.

The original port alone supplied software evidence. Subsequent LacunAI model
runs are recorded separately in the [experiment log](results.md). Optional
real-tensor tests still report skips when ML dependencies are absent. A new
experiment needs its own controls, comparisons, relearning arms and independent
artifact verification; earlier results do not establish its outcome.

## Connected-fact extension

The [connected-fact contract](connected-fact-contract.md) adds a versioned target
graph, explicit fact identities, typed reconstruction rules, per-fact evaluation
and full-answer likelihood against a matched retain-only reference. Schema-one
target serialization and historical experiment contracts remain unchanged.
Qualification prompts and final evaluation prompts have separate roles. The
pilot must qualify planting against the untouched model before attempting an
edit, and the final evaluation cannot select or tune candidates.

Checkpoint materialization reuses the existing touched-tensor transaction and
restore implementation. Sparse after-images identify each new control and
candidate; the shared scratch checkpoint is restored after each nested operation
and to its entry bytes at finalization. This storage arrangement requires an
explicit scratch declaration and enough capacity for the writer's temporary
full shard, even though durable snapshots contain only touched tensors.

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
