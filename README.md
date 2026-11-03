<picture>
  <source media="(prefers-color-scheme: dark)" srcset="brand/lacunai-horizontal-dark.svg">
  <img src="brand/lacunai-horizontal-light.svg" alt="LacunAI" width="520">
</picture>

# LacunAI

Experimental, auditable weight and memory editing for language models.

An editing model interprets the target and diagnostic evidence, then proposes a constrained JSON specification. Numerical code applies the edit. Independent evaluation measures recoverability and collateral damage. Weight-space programming is a research hypothesis; an incorrect answer or refusal does not demonstrate information deletion.

NeurAIlyzer remains a time-based agent-state management tool. LacunAI owns this separate line of weight-editing research.

## Boundaries

- Experimental execution is disabled by default and requires explicit enablement and an approved contract.
- The editor supplies a structured specification, never executable code or invented replacement weights.
- Tensor operations are allowlisted and checked for shape, dtype, finiteness, bounded change, source identity and checkpoint reload integrity.
- Original checkpoints remain intact. Derivatives are isolated. Promotion is manual.
- Development examples, evaluation examples, raw outputs and code-computed results have distinct roles.
- Outcomes are **failed**, **inconclusive**, or **passed the stated experimental contract**. Passing concerns reduced recoverability under the specified tests, not permanent erasure.

## First pilot

The first real experiment uses Qwen2.5-0.5B-Instruct on local CPU hardware. The editing decisions come from the current Codex session. It compares an untouched checkpoint, a no-op, GradDiff, retain-only fine-tuning and a frozen session-authored edit, with bounded benign-relearning tests.

That batch uses immutable copies captured before the LacunAI project split. Its historical paths and wire-format identifiers remain unchanged to preserve evidence. This repository is the standalone source port; its renamed package was not substituted into the frozen worker.

The original broad qualification contract was not satisfied, so this pilot is ineligible for an overall pass. Its per-skill outcomes must be assessed from saved raw outputs and independent verification. No checkpoint, raw run dataset or private execution receipt is included in this repository.

## Use

```sh
python -m pip install -e .
lacunai --help
lacunai validate-spec --help
```

Installing the base package does not install an ML runtime or authorize model execution. `validate-spec` validates a supplied specification against its expected hash, diagnostics and contract; it does not run an editor, load a checkpoint or perform tensor validation.

The `experimental` command group preserves the original frozen pilot adapters. They are bounded historical protocols, not a general-purpose hosted-model service. See [architecture and execution boundaries](docs/architecture.md).

The [historical Linux CPU dependency lock](requirements/pilot-linux-cpu.lock) records the runtime used by the pilot. It is specific to that environment; installation and future model runs require their own reviewed execution contract.

## Identity

[SVG assets and usage](brand/README.md) · [Identity specimen](brand/specimen.html)

The name and clinic-stationery direction take inspiration from Lacuna in *Eternal Sunshine of the Spotless Mind*. The LacunAI mark and lettering are original vector artwork.

## Development

```sh
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
python -m mypy src
```

Tests that require an optional tensor runtime must report a skip when it is unavailable. Unit fixtures do not constitute a completed model experiment.

Pull requests run the `checks` workflow without the optional ML runtime. Initial local validation is separate from hosted CI; repository creation does not dispatch a model experiment.

Apache-2.0. See [LICENSE](LICENSE) and [source provenance](docs/source-provenance.md).
