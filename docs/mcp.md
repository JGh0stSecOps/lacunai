# LacunAI over MCP

LacunAI exposes its local model-editing service over **stdio MCP**. The CLI and MCP tools use the same `core.service.Service` implementation, including checkpoint validation, mandatory snapshots, evaluation and restoration. This is a local subprocess server; it does not call hosted model APIs.

## Install and start

From a checkout, install the transport into your chosen Python environment:

```sh
python -m pip install -e '.[mcp]'
python -m lacunai --workspace /absolute/path/lacunai-runs mcp serve
```

The equivalent installed command is `lacunai --workspace /absolute/path/lacunai-runs mcp serve`. Use an absolute workspace path and the Python executable from the environment where LacunAI is installed. The MCP client starts and owns the subprocess; do not also start a second server for the same connection.

The MCP extra does **not** install the model runtime. For model operations, install `.[mcp,model]` and supply an existing local supported checkpoint and target specification. Installation alone does not download a model or run inference. The built-in model runtime is currently CPU float32 for supported Qwen2/Llama architectures.

## Client configuration

For clients using an `mcpServers` JSON configuration, replace the paths below:

```json
{
  "mcpServers": {
    "lacunai": {
      "command": "/absolute/path/venv/bin/python",
      "args": [
        "-m", "lacunai",
        "--workspace", "/absolute/path/lacunai-runs",
        "mcp", "serve"
      ]
    }
  }
}
```

These are configuration examples, not a claim that a particular client installation has been configured or tested. The Python environment must contain LacunAI; a checkout's working directory alone is not an installation.

## Tools and execution boundaries

| Tool | Behavior |
| --- | --- |
| `lacunai_map` | Run local inference to build a target fingerprint and map. |
| `lacunai_locate` | Run local inference and activation patching to localize a target. |
| `lacunai_erase` | Plan an edit; `apply=true` is required for weight writes. |
| `lacunai_verify` | Evaluate a recorded run with local inference. |
| `lacunai_restore` | Plan restoration; `apply=true` is required to restore checkpoint bytes. |
| `lacunai_runs` | List recorded runs and maps without model inference. |
| `lacunai_forget` | Use the weight pipeline, optionally followed by configured state clearing. |

`apply` defaults to false for erase, forget and restore. `in_place` defaults to false for editing. Map, locate and verify perform inference when called; they are not dry-run operations. Model commands accept runtime limits, seeds and thread counts through their tool schemas.

`lacunai_erase` and `lacunai_forget` accept `gradient_data`, an absolute JSON-file path, for experimental methods `npo-kl` and `grad-diff-kl`. The CLI equivalent is `--gradient-data`. It is required even for a plan and rejected for other methods. The file binds development-only forget/retain examples to the exact source checkpoint fingerprint; withheld evaluation examples are rejected. Both methods use the same snapshot and verification service, with fixed strength and change bounds and at most 128 numerical updates. A successful dry-run validates inputs without loading a model or establishing scientific efficacy.

Scientific verdicts and operational errors are returned as structured data. Clients must inspect `verdict`, `status` and `error`; successful MCP transport does not mean the scientific experiment passed. A missing restore run, for example, returns `status: failed` and an error rather than creating a run or changing weights.

Optional state clearing requires the `state` extra and an explicit state configuration. It is limited to configured local session/sandbox stores, preserves guarded snapshots and keep lists, and follows successful weight verification. Clearing is **store scoped**, not selective deletion of a semantic fact. See the [optional state integration](migration.md) for installation provenance and boundaries.

Protocol messages use stdout; progress and diagnostics use stderr. Avoid wrappers that merge stderr into stdout, which would corrupt the MCP stream.

## Verification and current limits

`tests/test_mcp_stdio.py` starts `python -m lacunai ... mcp serve` through the installed MCP SDK client. It initializes the real protocol, checks tool discovery and apply defaults, calls the production service to list an empty workspace, checks a missing-run restoration failure, and confirms the connection still handles the next request. It uses no mocked service, downloaded checkpoint or model inference. This transport smoke test does not establish editing efficacy or deployment into a specific agent client.

LacunAI currently provides stdio only. The vLLM adapter prepares a validated serving command but does not start or reload a server; live vLLM serving remains unverified. Signed reports establish artifact integrity, not permanent information erasure. Originals and backups retain their original information.
