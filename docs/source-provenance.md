# Source provenance

LacunAI was separated from an uncommitted experimental worktree of NeurAIlyzer after the project scope changed. NeurAIlyzer retains time-based agent-state management; its state, wipe, restore and MCP implementations are not part of LacunAI.

The source worktree was based on commit `1681ec4467c6e2b967db7d3b0239e884e1205dcb`. The experimental additions were outside that commit and were preserved in a private, hash-verified archive before migration. The archive SHA-256 is `a9d81d10c44b3551edeec43e7bb5b79445df685eba3b7549633a0d4993b90af0`.

The ongoing pilot's source bundle, contracts, model revisions, datasets, checkpoint hashes and execution receipts remain immutable. Historical identifiers such as `neurailyzer-pilot-20261102` and `neurailyzer-base-plus-delta-v1` are retained where changing them would invalidate a frozen contract or provenance relationship. They are historical protocol identifiers, not a shared product scope.

The standalone namespace port uses `lacunai`. A passing local test of this port is not a claim that the running experiment executed the renamed package. Run evidence names the exact frozen source bytes that actually ran.

No original checkpoint or execution artifact was deleted or overwritten for this migration. Private operational material and raw run artifacts remain outside the repository.
