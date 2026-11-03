"""Transport fixtures test software contracts, not model unlearning."""

import asyncio
import json
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from lacunai.cli import app
from lacunai.mcp_server import build_server


@pytest.fixture
def service(monkeypatch):
    from lacunai.core import service as module

    calls = []

    class FixtureService:
        def __init__(self, workspace):
            self.workspace = workspace

        def __getattr__(self, name):
            def call(**kwargs):
                calls.append((name, kwargs))
                return {
                    "dry_run": not kwargs.get("apply", False),
                    "verdict": "inconclusive",
                    "operation": name,
                }

            return call

    monkeypatch.setattr(module, "Service", FixtureService)
    return calls


def test_cli_apply_and_failure_exit(service, tmp_path):
    runner = CliRunner()
    args = [
        "--workspace",
        str(tmp_path),
        "erase",
        "--model",
        "/model",
        "--target",
        "/target",
        "--method",
        "ablate-neurons",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["dry_run"]
    assert service[-1][1]["apply"] is False
    assert "active" in result.stderr
    result = runner.invoke(app, [*args, "--apply"])
    assert result.exit_code == 1
    assert service[-1][1]["apply"] is True
    assert json.loads(result.stdout)["verdict"] == "inconclusive"


def test_cli_restore_and_verify(service):
    runner = CliRunner()
    assert runner.invoke(app, ["restore", "run"]).exit_code == 0
    assert service[-1][1]["apply"] is False
    assert runner.invoke(app, ["restore", "run", "--apply"]).exit_code == 0
    assert service[-1][1]["apply"] is True
    assert runner.invoke(app, ["verify", "run"]).exit_code == 1


def test_invalid_runtime_json(service):
    result = CliRunner().invoke(app, ["map", "--model", "/m", "--target", "/t", "--threads", "0"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error_type"] == "ValueError"
    assert not service


def test_help_no_ml_import():
    code = (
        "from lacunai.cli import app; from typer.testing import CliRunner; import sys; "
        "r=CliRunner().invoke(app,['--help']); assert r.exit_code==0; "
        "assert not ({'torch','transformers','neurailyzer','mcp'} & sys.modules.keys())"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_mcp_shared_service(service, tmp_path):
    server = build_server(tmp_path)

    async def check():
        tools = await server.list_tools()
        assert {t.name for t in tools} == {
            f"lacunai_{n}"
            for n in ("map", "locate", "erase", "verify", "restore", "runs", "forget")
        }
        await server.call_tool(
            "lacunai_erase", {"model": "/m", "target": "/t", "method": "ablate-neurons"}
        )
        assert service[-1][0] == "erase"
        assert service[-1][1]["apply"] is False
        await server.call_tool("lacunai_restore", {"run_id": "r", "apply": True})
        assert service[-1][1]["apply"] is True

    asyncio.run(check())


def test_state_refuses_unknown_scope_and_missing_extra(tmp_path, monkeypatch):
    from lacunai.adapters import neurailyzer as bridge

    config = tmp_path / "state.toml"
    config.write_text("")
    for scopes in ([], ["all"], ["remote"], ["rag"], ["session", "session"]):
        with pytest.raises(ValueError):
            bridge.preflight(config, scopes, [])

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(bridge, "import_module", missing)
    with pytest.raises(RuntimeError, match="optional"):
        bridge.preflight(config, ["session"], [])


def test_vllm_signed_checkpoint(tmp_path):
    from lacunai.adapters.vllm import prepare_launch
    from lacunai.core.model import checkpoint_inventory
    from lacunai.core.spec import digest
    from lacunai.verify.reports import write_signed_report

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"fixture-not-a-model")
    run = tmp_path / "run"
    run.mkdir()
    keys = tmp_path / "keys"
    record = {
        "run_id": "fixture",
        "target_id": "fixture",
        "status": "complete",
        "verdict": "passed_the_stated_experimental_contract",
        "output": str(checkpoint),
        "output_fingerprint": digest(checkpoint_inventory(checkpoint, 1024)),
    }
    write_signed_report(run, record, keys)
    result = prepare_launch(run, keys / "public.key", checkpoint)
    assert result["argv"][1] == "serve"
    assert result["started"] is False
    with pytest.raises(ValueError, match="path"):
        prepare_launch(run, keys / "public.key", tmp_path / "other")
    (checkpoint / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_launch(run, keys / "public.key", checkpoint)
    (run / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        prepare_launch(run, keys / "public.key", checkpoint)


def test_real_synthetic_state_snapshot_keep_and_apply(tmp_path):
    pytest.importorskip("neurailyzer")
    from lacunai.adapters.neurailyzer import preflight, wipe_state

    state = tmp_path / "state"
    state.mkdir()
    forgotten = state / "synthetic.txt"
    forgotten.write_text("synthetic fact")
    kept = state / "kept.txt"
    kept.write_text("retained")
    config = tmp_path / "state.toml"
    config.write_text(
        f'[targets.session]\npaths = ["{state}"]\n[keep]\npaths = ["{kept}"]\n'
        f'[snapshots]\ndir = "{tmp_path / "snapshots"}"\n'
    )
    with pytest.raises(ValueError, match="overlaps"):
        preflight(config, ["session"], [state / "checkpoint"])
    plan = wipe_state(config, ["session"], [])
    assert plan["dry_run"] and forgotten.exists()
    result = wipe_state(config, ["session"], [], apply=True)
    assert result["performed"] and result["snapshot_id"]
    assert not forgotten.exists() and kept.read_text() == "retained"
    assert result["verified"] == {"session": True}


def test_state_retention_refuses_before_wipe(tmp_path):
    pytest.importorskip("neurailyzer")
    from neurailyzer.snapshots import SnapshotStore

    from lacunai.adapters.neurailyzer import wipe_state

    state = tmp_path / "state"
    state.mkdir()
    fact = state / "synthetic"
    fact.write_text("retain original")
    store = SnapshotStore(tmp_path / "snapshots")
    original = store.take({"session": (state,)}, "original")
    config = tmp_path / "state.toml"
    config.write_text(
        f'[targets.session]\npaths=["{state}"]\n[snapshots]\n'
        f'dir="{tmp_path / "snapshots"}"\nretention=1\n'
    )
    with pytest.raises(RuntimeError, match="retention"):
        wipe_state(config, ["session"], [], apply=True)
    assert fact.read_text() == "retain original"
    assert [s.id for s in store.list()] == [original.id]
