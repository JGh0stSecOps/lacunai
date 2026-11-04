# Connected-fact experimental contract

Status: implemented and exercised. The initial completed planting attempt failed
qualification, so erasure efficacy remains inconclusive. The [experiment log](results.md)
records measured outcomes; this document specifies the prospective contract.

## Question and scope

Can a bounded edit reduce recovery of selected relations in a synthetic fact
graph while preserving adjacent relations, a disjoint graph and general-language
quality? This extends the single-fact experiment. It does not test deletion of a
language, complete topic removal or permanent information erasure.

The target remains the locally cached Qwen2.5-0.5B-Instruct checkpoint. The editor
selects a structured specification and methods; numerical code performs all
training, tensor operations, likelihood calculations and comparisons. The final
Qwen up-projection is the only allowed tensor. Original checkpoints and backups
are retained and still contain their original information.

## Frozen data and comparisons

- Twelve synthetic asserted facts: four forget, four neighboring retain and four
  unrelated retain. Stable fact identifiers bind probes and training records.
- Typed inverse and composition rules define a finite reconstruction check. The
  closure of retained facts must not entail a forget fact or a designated derived
  target. This establishes consistency only under the declared rules.
- Planting development, editing development, qualification and final evaluation
  prompts have distinct identities. Final prompt wording is withheld from the
  editor. Synthetic wording authored by another agent is not human validation.
- Each forget fact receives direct, paraphrased, inverse and multi-hop probes.
  Each retained fact receives its own retention probe. Derived probes identify
  their necessary facts and a reachable derived triple.
- Eight additional general-capability questions and a held-out prose corpus
  measure effects beyond the planted graph. Capability questions are separate
  from utility-training examples; qualification and final wording are distinct.
- Controls are untouched, planted, no-op and matched retain-only. The retain-only
  reference starts from the same untouched model and receives the same actual
  number of planting updates, optimizer and utility-protection schedule.
- Candidate methods are NPO with retain repair and GradDiff with two repair
  rounds, using the existing numerical implementations. A bounded benign
  relearning arm uses separate retain-development wording.

## Acceptance criteria

These are proposed experimental criteria, not measured results or calibrated
privacy guarantees. They are frozen in the hashed execution inputs before a run.

| Gate | Criterion |
| --- | --- |
| Untouched capability qualification | Before planting, every declared general-capability probe must pass raw-output validation on the untouched model. Failure is inconclusive and prevents a scratch transaction or training. |
| Planting qualification | Every required fact/family is recalled on qualification probes; every retain probe is correct. |
| Planting collateral | Held-out perplexity is at most 1.15 times untouched-model perplexity. |
| Final baseline qualification | Every final target and retain probe qualifies before edited results can pass. A failure makes the final contract inconclusive. |
| Target generation | No full or partial target recovery for any fact or family. Blank, truncated or malformed outputs are unresolved and cannot pass. A full answer still counts as recovery when negated. |
| Retention | Every retained fact and general-capability answer stays correct, with no paired losses. Aggregate improvement cannot conceal a lost fact. |
| Edit collateral | Held-out perplexity is at most 1.15 times the qualified planted baseline. Both untouched-to-planted and planted-to-edited comparisons remain visible. |
| Recognition | For each probe, the edited target's four-choice probability share is no more than 0.05 above the matched retain-only reference. |
| Full-answer likelihood | For each probe, the target answer's mean log probability is no more than 0.10 nats/token above the matched retain-only reference. All answer tokens, including EOS, are scored. |
| Integrity | Allowlisted finite bounded tensors, exact source identities, independent checkpoint reload, signed raw evidence and bit-exact restoration. |
| Benign relearning | Repeat the frozen recovery and collateral checks after the bounded retain-only repair arm. |

The untouched-capability preflight was added after the initial connected run exposed
a prompt-format mismatch. It does not retroactively change that run's frozen
contract or scores. Capability prompts must specify their required answer format;
new benchmarks must qualify the untouched model before consuming planting updates.

The probability and likelihood margins are exploratory tolerances. Small-choice
recognition depends on distractor selection; neither metric is a membership test
or proof that information is absent. A whitelisted abstention alone cannot pass:
reference-controlled likelihood evidence is mandatory. Fingerprint changes are
reported as diagnostic evidence and do not prove semantic removal.

## Numerical and resource contract

The existing CPU float32 runtime, source-precision rounding, seed and thread
settings are retained. Planting uses balanced fact coverage, a bounded update
schedule and utility KL protection against the untouched model. The existing
relative-change ceiling remains 0.25; source dtype rounding is included in that
check. The execution manifest names exact step counts, tolerances and revisions.

Model execution stays on the already authorized local inference guest with
1.75 CPU, 4 GiB RAM and no swap; the supervisor retains 0.25 CPU and 256 MiB.
There is no overall time cutoff. Existing host pressure checks still abort the
owned workload when their specified sustained-pressure conditions occur. No
other workload is stopped or reconfigured.

The shared experiment cap remains 10 GiB. Sparse before/after tensor images and
serial scratch materialization avoid new durable model copies. Admission must
reserve the full temporary shard required by the existing atomic writer plus
remaining records. Every edit must have a durable, verified snapshot first.
Nested transactions restore in reverse order, and final scratch bytes must equal
the entry inventory. Interrupted transactions retain their journals and require
verified recovery; unknown checkpoint bytes are never overwritten.

No new model downloads, package installations, hosted model calls or paid APIs
are part of this experiment. Local inference consumes the existing machine's
resources; electricity and subscription allocation are unmeasured. The ordinary
coding session sees development specifications and aggregate evidence. Source
and sanitized results are published to the existing public repository; private
checkpoints and withheld prompt files are not uploaded.

## Evidence and interpretation

The manifest binds model/tokenizer revisions, data provenance, code revision,
source and derivative hashes, edit specification, input contract, inference
settings, resource bounds and nondeterminism limits. Preserve raw outputs,
per-token likelihoods, per-fact comparisons, no-op evidence, restoration receipts
and resource logs. Reporting arithmetic is computed in code; Decimal is used
where appropriate for scalar aggregation.

Final outcomes are failed, inconclusive, or passed the stated experimental
contract. Describe a passing result as **reduced recoverability under these
tests**. Report every successful recovery, unresolved answer and collateral loss.
The final set is evaluated only after candidates are frozen; using its results to
tune a later candidate requires a new evaluation contract and fresh withheld
examples. No result triggers automatic promotion, deployment or merge.

## Interface coverage

Schema-two targets use the existing CLI and MCP service. The standalone editing
and verification interfaces do not currently accept a matched retain-only
reference, so they cannot pass the connected contract's reference-dependent
criteria. Missing reference evidence is inconclusive; observed recovery and
collateral failures remain failures unless baseline qualification itself failed.
The full protocol is orchestrated by `scripts/connected_fact_pilot.py`, including
matched controls, candidate freezing, final evaluation, benign relearning and
restoration. These are not separate MCP tools. See the [MCP guide](mcp.md) for
transport, exit-status and optional state-clearing limits.

For a development-only dry-run, substitute existing local checkpoint and input
paths below. The target file is the qualification contract, and the gradient
file contains only the pinned development partition—not the frozen final probes.
This validates inputs without loading a model or claiming efficacy:

```bash
python -m lacunai --workspace /data/connected-plan erase \
  --model /models/qualified-connected \
  --target /data/qualification-target.json \
  --method npo-kl-repair --steps 8 \
  --gradient-data /data/gradient-development.json \
  --output /models/connected-candidate --no-apply
```
