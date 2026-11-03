<picture>
  <source media="(prefers-color-scheme: dark)" srcset="brand/lacunai-stacked-tagline-dark.svg">
  <img src="brand/lacunai-stacked-tagline-light.svg" alt="LacunAI — For agents learning to forget." width="520">
</picture>

# LacunAI

Experimental, auditable weight and memory editing for language models.

An editing model interprets the target and diagnostic evidence, then proposes a constrained JSON specification. Numerical code applies the edit. Independent evaluation measures recoverability and collateral damage. Weight-space programming is a research hypothesis; an incorrect answer or refusal does not demonstrate information deletion.

NeurAIlyzer remains a time-based agent-state management tool. LacunAI owns this separate line of weight-editing research.

## Map, erase, verify, restore

The shared service supports local Hugging Face safetensors checkpoints with Qwen2/Llama gated MLPs. The built-in runtime is currently **CPU float32**. No hosted inference or automatic model download is used. CLI and MCP share the same execution and snapshot code.

- **Map / locate:** sparse activation fingerprints with background IDF weighting, then activation patching to rank causal effects. The persistent index exports JSON and per-layer heatmaps. These are localization measurements, not a unique physical address for a fact.
- **Erase:** neuron ablation, direction projection, a bounded rank-one rewrite, and the retained historical column operator. `rank-one` is a ROME-inspired diagonal approximation, not a full ROME/MEMIT implementation.
- **Verify:** frozen direct/paraphrased and optional multi-hop/adversarial probes, paired retain checks, held-out perplexity, and before/after fingerprints. Raw evidence is retained.
- **Restore:** snapshots preserve the original bytes of touched tensors and checkpoint metadata; restoration validates checkpoint hashes. No subtraction of rounded floating-point deltas is used to reconstruct originals.

The new planted-fact experiment demonstrating two editing methods and exact restoration is **pending**. Software tests and the historical pilot do not satisfy that acceptance criterion.

The separate [Chinese-literacy experiment](docs/language-unlearning.md) measures
all-layer language activity, screens explicitly selected neuron edits, and checks
Chinese recovery alongside non-Chinese collateral damage. Its outcome requires
actual model evidence; the presence of the operator is not evidence of forgetting.

## Boundaries

- Weight writes are dry-run by default; `--apply` is required. `map`, `locate`, and `verify` execute local inference when invoked. Historical batch execution still requires its explicit enablement and approved contract.
- The editor supplies a structured specification, never executable code or invented replacement weights.
- Tensor operations are allowlisted and checked for shape, dtype, finiteness, bounded change, source identity and checkpoint reload integrity.
- Outputs default to new paths. `--in-place` requires explicit selection and `--apply`; snapshots are mandatory before edits. Promotion is manual.
- Development examples, evaluation examples, raw outputs and code-computed results have distinct roles.
- Outcomes are **failed**, **inconclusive**, or **passed the stated experimental contract**. Passing concerns reduced recoverability under the specified tests, not permanent erasure.

## First pilot

The historical first real experiment uses Qwen2.5-0.5B-Instruct on local CPU hardware. The editing decisions come from the current Codex session. It compares an untouched checkpoint, a no-op, GradDiff, retain-only fine-tuning and a frozen session-authored edit, with bounded benign-relearning tests.

That batch uses immutable copies captured before the LacunAI project split. Its historical paths and wire-format identifiers remain unchanged to preserve evidence. This repository is the standalone source port; its renamed package was not substituted into the frozen worker.

The original broad qualification contract was not satisfied, so this pilot is ineligible for an overall pass. Its per-skill outcomes must be assessed from saved raw outputs and independent verification. No checkpoint, raw run dataset or private execution receipt is included in this repository.

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

### Optional integrations

```sh
python -m pip install -e '.[mcp]'
lacunai --workspace /data/lacunai mcp serve
```

MCP exposes `lacunai_map`, `lacunai_locate`, `lacunai_erase`, `lacunai_verify`, `lacunai_restore`, `lacunai_runs`, and `lacunai_forget` over stdio with the same defaults. `adapters.vllm.prepare_launch` validates a signed passed report and the exact output checkpoint, then returns a loopback-only offline serving command. It never starts or reloads a server; live vLLM serving is unverified.

`lacunai[state]` installs NeurAIlyzer from a pinned Git commit; see [migration and installation provenance](docs/migration.md). NeurAIlyzer does not depend on LacunAI. `forget` uses the weight pipeline and can also clear state only when supplied an explicit `--state-config` and local `--state-scope session` / `--state-scope sandbox`. An applied state wipe follows successful weight verification, preserves NeurAIlyzer's liveness and keep-list checks, and requires a snapshot. Retention that would delete older backups is refused. Remote/provider stores are excluded.

**State clearing is store scoped:** all non-kept data in the selected configured roots is wiped, not just the semantic fact targeted in weights. Without a state configuration, `forget` performs only the weight operation. Keep model/checkpoint/workspace paths outside state roots.

`validate-spec` validates supplied JSON against its expected hash, diagnostics and contract without loading a model. The `experimental` command group preserves the original frozen pilot protocols; see [architecture and execution boundaries](docs/architecture.md). The [historical Linux CPU dependency lock](requirements/pilot-linux-cpu.lock) records that pilot's environment. Future runs require their own reviewed bounds.

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
