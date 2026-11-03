# NeurAIlyzer → LacunAI migration

NeurAIlyzer retains time-based state hygiene: session history, sandbox files and its existing state/provider integrations. LacunAI owns model-side localization, editing, verification and restoration. LacunAI may optionally call NeurAIlyzer; NeurAIlyzer must run without LacunAI installed.

The original weight-editing additions were **uncommitted** in the NeurAIlyzer source worktree, so there was no weight-feature commit history to extract with `git filter-repo` or subtree splitting. The preserved LacunAI implementation commits retain the migrated source history from the initial port onward. [Source provenance](source-provenance.md) records the source base and archive digest. Historical wire identifiers and frozen private experiment artifacts remain unchanged; they identify the actual code that ran.

The companion NeurAIlyzer change replaces the experimental entrypoint with a thin deprecation message for one release. Use `lacunai experimental ...` for historical protocols and the new `map`, `locate`, `erase`, `verify`, `restore`, `runs`, and `forget` commands for the shared service. The removal/shim change is a separate review; this document does not claim it has merged.

## Optional state dependency

The PyPI project lookup returned no published NeurAIlyzer distribution during integration. Until a release is published, `lacunai[state]` uses this immutable public Git source:

```text
neurailyzer @ git+https://github.com/JGh0stSecOps/neurailyzer.git@7e1a92f8fda837968c32f63b217754a3aa16eb70
```

This stable development revision provides the guarded state API; it predates the companion deprecation shim. Installation contacts GitHub and the configured Python package index for dependencies. No model or user state is uploaded by installation. The `[model]` extra uses the same pinned tensor dependencies as `[experimental]`; `[mcp]` selects MCP SDK major version 2. Base installation includes cryptography for signed reports and does not install Torch or NeurAIlyzer.

An explicit state config is required before `forget` can wipe state. Only local session and sandbox stores are accepted, mandatory snapshots and keep lists are preserved, and active stores or snapshot-pruning conditions are refused. State scope is the entire selected store, not a semantic subset. Originals and backups continue to contain the original information.

## Evidence boundaries

Run records identify source checkpoint hashes, frozen target contracts and runtime settings. Reports retain raw evaluation outputs and sign artifact integrity; they cannot prove permanent erasure. The new real planted-fact acceptance experiment remains pending. Local tests, historical CPU pilot evidence, repository creation and CI configuration are distinct from a completed new experiment or a green hosted CI run.
