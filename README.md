<picture>
  <source media="(prefers-color-scheme: dark)" srcset="brand/lacunai-stacked-tagline-dark.svg">
  <img src="brand/lacunai-stacked-tagline-light.svg" alt="LacunAI — For agents learning to forget." width="520">
</picture>

# LacunAI

Experimental, auditable weight and memory editing for language models.

Use LacunAI from the command line or as an [MCP server for agents](docs/mcp.md).
Both interfaces expose the same mapping, editing, verification and restoration service.

> **Experimental research software. Reliable information deletion has not been demonstrated.**
> An edit that changes an answer can leave the underlying information recoverable.
> Tests, signatures, and exact restoration validate specific behavior or artifact
> integrity; they do not prove that a model has forgotten information permanently.

An editing model interprets the target and diagnostic evidence, then proposes a constrained JSON specification. Numerical code applies the edit. Independent evaluation measures recoverability and collateral damage. Weight-space programming is a research hypothesis; an incorrect answer or refusal does not demonstrate information deletion.

## Current experiment: an existing pretrained fact

The next experiment targets **“Paris is the capital of France”** in the original
Qwen2.5-0.5B-Instruct checkpoint. It performs no planting or pre-erasure fine-tuning.
The original must demonstrate recall before either weight-editing method runs.
Untouched and no-op controls, withheld recovery probes, collateral checks, benign
relearning and exact restoration use the existing transactional pipeline.
**This experiment has not yet produced a model result.** See its
[pretrained-fact contract](docs/pretrained-fact-contract.md).

## Previous experiment: connected facts

The latest five-fact pilot **failed planting qualification after 512 updates; erasure efficacy remains inconclusive**. It exercised utility KL plus `0.25 × mean utility-answer NLL`, but recovered only 12/16 target answers and 5/8 retained-fact answers, while capability correctness fell from 8/8 untouched to 4/8. Direct and inverse target recall reached 4/4 each; paraphrased recall was 3/4 and multi-hop recall 1/4. Neither unlearning method ran. Original checkpoint files remained unchanged; the working checkpoint was restored exactly.

The [connected contract](docs/connected-fact-contract.md) tests neighboring and unrelated facts, inverse and multi-hop recovery, capabilities, and full-answer likelihood against a matched retain-only reference. Capability probes were selected for untouched-model competence; this is preservation testing, not unbiased broad capability accuracy.

The [experiment log](docs/results.md) preserves the earlier twelve-fact and five-fact failures alongside the [utility-preservation result](research/results/connected-five-fact-utility-2026-11-04.json). The latest objective was exercised under unchanged acceptance thresholds. Comparing these individual runs does not establish improvement; software tests and artifact signatures are not efficacy evidence.

## Earlier experiment: one fact

**Two methods passed the stated experimental contract on one planted fact.**
Qwen2.5-0.5B-Instruct first learned the fact with 64/64 development and 8/8
evaluation recall. NPO with retain-only repair and GradDiff with two repair rounds
each reduced measured full-fact recovery to 0/8 while retaining 8/8 unrelated
answers. Both passed independent checkpoint reload, bounded benign relearning,
and bit-exact weight restoration with matching restored generations.

These are adaptively selected results on a reused small evaluation set. A fresh
multilingual supplement found no confirmed full-fact recovery, but was
**inconclusive**: the baseline recovered only 9/16 target answers and answered
1/8 retain probes correctly, failing qualification; one NPO response was ambiguous. See all outcomes in the [measured results](docs/results.md).

**Partial numeric prefixes survived; some target answers became malformed.**
The measured activations changed but remained active. Planting itself raised
held-out perplexity from 98.47 to 142.75; edit collateral compares against that
planted baseline. This is **reduced recoverability under these tests**, not
proof of information deletion or general quality preservation.

1. Plant the fact and verify that the model can recall it before editing.
2. Map the relevant activations and apply bounded edits to isolated checkpoints.
3. Compare direct and paraphrased recovery against untouched, no-op and established
   unlearning controls, while measuring unrelated capabilities.
4. Restore the original weights and verify their bytes and hashes.

See the [single-fact experiment contract](docs/planted-fact-contract.md) for the
frozen acceptance criteria. At least two editing methods must pass; a failed
answer or a green software test does not complete the experiment.

## Published results

The [experiment log](docs/results.md) and its machine-readable records report
what worked, what failed, successful recovery attempts, collateral damage and
missing measurements. Software checks, development screens and independent model
evaluations are identified separately. New results are appended; failed results
are not replaced with later successes.

## Map, erase, verify, restore

The shared service supports local Hugging Face safetensors checkpoints with Qwen2/Llama gated MLPs. The built-in runtime is currently **CPU float32**. No hosted inference or automatic model download is used. CLI and MCP share the same execution and snapshot code.

- **Map / locate:** sparse activation fingerprints with background IDF weighting, then activation patching to rank causal effects. The persistent index exports JSON and per-layer heatmaps. These are localization measurements, not a unique physical address for a fact.
- **Erase:** neuron ablation, direction projection, a bounded rank-one rewrite, the retained historical column operator, and experimental `npo-kl` / `grad-diff-kl` proposals. `rank-one` is a ROME-inspired diagonal approximation, not a full ROME/MEMIT implementation. The gradient methods require separate development data and are restricted to the final Qwen up-projection matrix.
- **Verify:** frozen direct/paraphrased and optional multi-hop/adversarial probes, paired retain checks, held-out perplexity, and before/after fingerprints. Raw evidence is retained.
- **Restore:** snapshots preserve the original bytes of touched tensors and checkpoint metadata; restoration validates checkpoint hashes. No subtraction of rounded floating-point deltas is used to reconstruct originals.

## Boundaries

- Weight writes are dry-run by default; `--apply` is required. `map`, `locate`, and `verify` execute local inference when invoked. Historical batch execution still requires its explicit enablement and approved contract.
- The editor supplies a structured specification, never executable code or invented replacement weights.
- Tensor operations are allowlisted and checked for shape, dtype, finiteness, bounded change, source identity and checkpoint reload integrity.
- Outputs default to new paths. `--in-place` requires explicit selection and `--apply`; snapshots are mandatory before edits. Promotion is manual.
- Development examples, evaluation examples, raw outputs and code-computed results have distinct roles.
- Outcomes are **failed**, **inconclusive**, or **passed the stated experimental contract**. Passing concerns reduced recoverability under the specified tests, not permanent erasure.

## Use

```sh
python -m pip install -e .
lacunai --help
lacunai validate-spec --help
# Optional local model runtime; installation alone does not run a model:
python -m pip install -e '.[model]'
```

Provide an existing local checkpoint and a frozen target JSON matching `core.spec.TargetSpec`: mapping prompts and corrupted controls, disjoint withheld direct/paraphrased probes, retain probes, background corpus, held-out perplexity texts, and acceptance thresholds. The source checkpoint must demonstrate target recall before an applied edit proceeds. Freeze the target before comparing candidates.

```sh
# These trace commands perform local inference:
lacunai --workspace /data/lacunai map --model /models/source --target /data/target.json
lacunai --workspace /data/lacunai locate --model /models/source --target /data/target.json
# Plan only (no model loading or weight writes):
lacunai --workspace /data/lacunai erase --model /models/source --target /data/target.json \
  --method ablate-neurons --output /models/candidate
# Execute only within an approved resource envelope:
lacunai --workspace /data/lacunai erase --model /models/source --target /data/target.json \
  --method ablate-direction --output /models/candidate-direction --apply \
  --max-seconds 3600 --threads 2 --seed 42
lacunai --workspace /data/lacunai verify RUN_ID
lacunai --workspace /data/lacunai restore RUN_ID           # plan
lacunai --workspace /data/lacunai restore RUN_ID --apply   # restore snapshot
lacunai --workspace /data/lacunai runs
```

Commands emit JSON results to stdout and progress to stderr. Applied edits and verification return nonzero for failed or inconclusive scientific contracts. Signed JSON/Markdown reports attest artifact integrity against a separately trusted public key; signatures do not prove information absence or scientific correctness. Retained originals and backups still contain original information.

The experimental gradient methods require `--gradient-data /data/development.json` on both `erase` and `forget`, including dry-runs. This file contains `schema_version: 1`, the exact checkpoint `source_fingerprint`, `forget` and `retain_development` arrays of `{prompt, answer}` objects, and a provenance string. Evaluation probes and held-out perplexity texts cannot be used as training examples. Applied runs freeze this data, verify its checkpoint binding and use the existing snapshot, independent verification and restoration flow. These bounded OpenUnlearning-inspired adaptations are not evidence of successful unlearning; consult the recorded experiment results.

### MCP and serving

```sh
python -m pip install -e '.[mcp]'
lacunai --workspace /data/lacunai mcp serve
```

MCP exposes `lacunai_map`, `lacunai_locate`, `lacunai_erase`, `lacunai_verify`, `lacunai_restore`, `lacunai_runs`, and `lacunai_forget` over stdio with the same defaults. `adapters.vllm.prepare_launch` validates a signed passed report and the exact output checkpoint, then returns a loopback-only offline serving command. It never starts or reloads a server; live vLLM serving is unverified.

`validate-spec` validates supplied JSON against its expected hash, diagnostics and contract without loading a model. See [architecture and execution boundaries](docs/architecture.md) for operators, run records and resource controls.

## Identity

[SVG assets and usage](brand/README.md) · [Identity specimen](brand/specimen.html)

The name takes inspiration from Lacuna in *Eternal Sunshine of the Spotless Mind*. The supplied memory-clinic helmet is redrawn as vector geometry: ordered electrodes transition into an open mesh, with silver and ivory on navy and orange AI lettering. The wordmark uses original outlined serif capitals.

**For agents learning to forget.** The line describes a research ambition; experimental results determine what the system actually achieves.

See the [research basis and limits](docs/research.md), including [fast-weight programmers and their connection to LacunAI](docs/fast-weight-programmers.md).

## Development

```sh
python -m pip install -e '.[dev,mcp]'
python -m pytest
python -m ruff check .
python -m mypy src
```

Tests that require an optional tensor runtime must report a skip when it is unavailable. Unit fixtures do not constitute a completed model experiment.

The `checks` workflow defines a lightweight source/test job and a CPU tensor job. The tensor job uses synthetic fixtures and offline Hugging Face settings, not downloaded pretrained models or the real pilot. Workflow configuration is not evidence of a green hosted run.

Apache-2.0. See [LICENSE](LICENSE) and [source provenance](docs/source-provenance.md).
