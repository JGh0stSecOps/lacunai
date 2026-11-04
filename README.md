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

## Current experiment: one fact

The latest real pilot learned **one planted synthetic fact** in Qwen2.5-0.5B-Instruct:
64/64 development and 8/8 withheld recall, with 8/8 retain accuracy. All three
erasure methods then **failed**: neuron ablation retained 8/8 target answers,
rank-one retained 5/8, and direction ablation retained 8/8. After bounded benign
relearning, recovery was 7/8, 7/8 and 8/8 respectively.

Planting also raised held-out perplexity from 98.47 to 142.75. Signed restoration
receipts report exact weights and matching generations. Independent file rehashing
confirmed the final restored checkpoint; earlier intermediate restores have signed
runtime receipts. The earlier 128-step and 512-step attempts remain
recorded as inconclusive qualification failures. Reliable removal and the
requested two-method demonstration remain unproven.

The latest eight-step gradient follow-up also **failed**. NPO reduced target
recovery to 0/8 but lost one of eight retained answers. Retain-only relearning
recovered that answer while target recovery stayed at zero; this repaired state
was not saved and independently reloaded in that run. GradDiff produced ambiguous
target responses and collateral losses. Both methods restored the original weights
and generations. Explicit repair variants are implemented; their efficacy is
unmeasured. The separately frozen multilingual supplement has not run. See the
[measured results](docs/results.md) for controls, prior attempts and limitations.

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
