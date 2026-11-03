"""Real stdio protocol against the production service; no model inference."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="MCP extra is not installed")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


def test_real_stdio_discovery_runs_and_missing_restore(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "lacunai", "--workspace", str(workspace), "mcp", "serve"],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": str(root / "src"),
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )

    async def exercise() -> None:
        with (tmp_path / "server.stderr.log").open("w") as errors:
            async with (
                stdio_client(server, errlog=errors) as (reader, writer),
                ClientSession(reader, writer, read_timeout_seconds=20) as client,
            ):
                initialized = await client.initialize()
                assert initialized.server_info.name == "LacunAI"
                response = await client.list_tools()
                tools = {tool.name: tool for tool in response.tools}
                assert set(tools) == {
                    f"lacunai_{name}"
                    for name in (
                        "map",
                        "locate",
                        "erase",
                        "verify",
                        "restore",
                        "runs",
                        "forget",
                    )
                }
                for name in ("erase", "forget", "restore"):
                    schema = tools[f"lacunai_{name}"].input_schema
                    assert schema["properties"]["apply"]["default"] is False
                    assert "apply" not in schema.get("required", [])
                result = await client.call_tool("lacunai_runs", {})
                assert not result.is_error
                assert result.structured_content == {"runs": [], "maps": []}
                result = await client.call_tool("lacunai_restore", {"run_id": "missing-run"})
                assert not result.is_error  # service failure is structured tool data
                failure = result.structured_content
                assert failure["status"] == "failed"
                assert failure["error"] and failure["error_type"] in {
                    "ValueError",
                    "FileNotFoundError",
                }
                # A failed request must not poison the stream or manufacture a run.
                again = await client.call_tool("lacunai_runs", {})
                assert again.structured_content == {"runs": [], "maps": []}

    asyncio.run(exercise())
    stderr = (tmp_path / "server.stderr.log").read_text()
    assert "workspace_operation" in stderr
    assert not any(p.suffix == ".safetensors" for p in workspace.rglob("*"))
    # The protocol client successfully parsed stdout throughout; progress stays on stderr.
    assert any(
        json.loads(line).get("state") == "failed"
        for line in stderr.splitlines()
        if line.startswith("{")
    )
