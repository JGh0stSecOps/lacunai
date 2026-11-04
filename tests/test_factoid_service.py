"""Production routing and software-only durable transaction checks, not ML efficacy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lacunai.cli import app
from lacunai.core import service as module
from lacunai.core.service import Service
from lacunai.core.spec import digest
from lacunai.erase import methods
from lacunai.verify.reports import verify_report
from tests.test_service_lifecycle import lifecycle as lifecycle_fixture
from tests.test_service_lifecycle import target_file
from tests.test_verification_records import spec

lifecycle = lifecycle_fixture


def data_file(tmp_path, fingerprint="a" * 64):
    data = {
        "schema_version": 1,
        "source_fingerprint": fingerprint,
        "forget": [{"prompt": "development target only", "answer": "drazel"}],
        "retain_development": [
            {"prompt": f"development retain only {i}", "answer": "oak"} for i in range(32)
        ],
        "provenance": "synthetic software routing fixture; no model evidence",
    }
    path = tmp_path / "gradient.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize(
    "method",
    [
        "npo-kl",
        "grad-diff-kl",
        "npo-kl-repair",
        "grad-diff-kl-repair",
        "undial-kl",
        "undial-kl-repair",
        "grad-diff-kl-repair2",
    ],
)
def test_required_even_dryrun_and_wrong_method_refused(tmp_path, monkeypatch, method):
    service = Service(tmp_path / "workspace")
    target = target_file(tmp_path)
    monkeypatch.setattr(module, "HFRuntime", lambda *a: pytest.fail("loaded model"))
    with pytest.raises(ValueError, match="required"):
        service.erase(tmp_path / "model", target, method)
    with pytest.raises(ValueError, match="regular file"):
        service.erase(tmp_path / "model", target, method, gradient_data=tmp_path / "absent")
    with pytest.raises(ValueError, match="only accepted"):
        service.erase(
            tmp_path / "model", target, "ablate-neurons", gradient_data=data_file(tmp_path)
        )
    assert service.store.list() == []


@pytest.mark.parametrize("split", ["probes", "retain", "perplexity_texts"])
def test_evaluation_overlap_refused_before_runtime(tmp_path, monkeypatch, split):
    service = Service(tmp_path / "workspace")
    target = target_file(tmp_path)
    path = data_file(tmp_path)
    data = json.loads(path.read_text())
    values = getattr(spec(), split)
    prompt = values[0] if split == "perplexity_texts" else values[0].prompt
    data["forget"][0]["prompt"] = "  " + prompt.upper() + "  "
    path.write_text(json.dumps(data))
    monkeypatch.setattr(module, "HFRuntime", lambda *a: pytest.fail("loaded model"))
    with pytest.raises(ValueError, match="overlaps"):
        service.erase(tmp_path / "model", target, "npo-kl", gradient_data=path, apply=True)
    assert service.store.list() == []


def test_wrong_source_rejected_before_map_or_snapshot(lifecycle, tmp_path, monkeypatch):
    service, model, target = lifecycle
    monkeypatch.setattr(service, "_mapping", lambda *a, **k: pytest.fail("mapping before binding"))
    monkeypatch.setattr(
        module, "prepare_snapshot", lambda *a, **k: pytest.fail("snapshot before binding")
    )
    before = module.checkpoint_inventory(model, 1024**2)
    with pytest.raises(ValueError, match="source fingerprint"):
        service.erase(model, target, "npo-kl", apply=True, gradient_data=data_file(tmp_path))
    assert module.checkpoint_inventory(model, 1024**2) == before
    assert service.store.list()[0]["status"] == "failed"


def test_frozen_data_delivered_signed_and_restorable(lifecycle, tmp_path, monkeypatch):
    service, model, target = lifecycle
    path = data_file(tmp_path, digest(module.checkpoint_inventory(model, 1024**2)))
    expected = json.loads(path.read_text())
    original_apply = methods.apply_method
    original_runtime = module.HFRuntime

    def runtime(*args):
        # Caller input changes after it was frozen; execution must use the frozen copy.
        path.write_text("{}")
        return original_runtime(*args)

    monkeypatch.setattr(module, "HFRuntime", runtime)
    seen = []

    def apply(*args, **kwargs):
        seen.append(kwargs["development_data"])
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(methods, "apply_method", apply)
    result = service.erase(model, target, "npo-kl", apply=True, gradient_data=path)
    directory = service.store.path(result["run_id"])
    assert seen == [expected]
    assert json.loads((directory / "gradient-data.json").read_text()) == expected
    assert result["metadata"]["gradient_data"]["content_sha256"] == digest(expected)
    verify_report(directory, service.workspace / "signing/public.key")
    restored = service.restore(result["run_id"], apply=True)
    assert restored["status"] == "complete"


@pytest.mark.parametrize(
    "method",
    [
        "npo-kl",
        "grad-diff-kl",
        "npo-kl-repair",
        "grad-diff-kl-repair",
        "undial-kl",
        "undial-kl-repair",
        "grad-diff-kl-repair2",
    ],
)
@pytest.mark.parametrize("operation", ["erase", "forget"])
def test_cli_production_dryrun_routes_data(tmp_path, monkeypatch, operation, method):
    monkeypatch.setattr(module, "HFRuntime", lambda *a: pytest.fail("dryrun loaded model"))
    target = target_file(tmp_path)
    data = data_file(tmp_path)
    args = [
        "--workspace",
        str(tmp_path / "workspace"),
        operation,
        "--model",
        str(tmp_path / "model"),
        "--target",
        str(target),
        "--method",
        method,
        "--gradient-data",
        str(data),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    record = Service(tmp_path / "workspace").store.list()[0]
    assert record["dry_run"] and record["metadata"]["gradient_data"]["content_sha256"]


@pytest.mark.parametrize(
    "method",
    [
        "npo-kl",
        "npo-kl-repair",
        "grad-diff-kl-repair",
        "undial-kl",
        "undial-kl-repair",
        "grad-diff-kl-repair2",
    ],
)
def test_real_mcp_stdio_schema_and_dryrun(tmp_path, method):
    pytest.importorskip("mcp")
    import asyncio
    import os
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    target = target_file(tmp_path)
    data = data_file(tmp_path)
    root = Path(__file__).resolve().parents[1]
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "lacunai", "--workspace", str(tmp_path / "workspace"), "mcp", "serve"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src"), "HF_HUB_OFFLINE": "1"},
    )

    async def exercise():
        with (tmp_path / "stderr.log").open("w") as errors:
            async with (
                stdio_client(server, errlog=errors) as (reader, writer),
                ClientSession(reader, writer, read_timeout_seconds=20) as client,
            ):
                await client.initialize()
                schemas = {t.name: t.input_schema for t in (await client.list_tools()).tools}
                for operation in ("erase", "forget"):
                    props = schemas["lacunai_" + operation]["properties"]
                    assert props["gradient_data"]["default"] is None
                    assert props["apply"]["default"] is False
                    result = await client.call_tool(
                        "lacunai_" + operation,
                        {
                            "model": str(tmp_path / "nonexistent-model"),
                            "target": str(target),
                            "method": method,
                            "gradient_data": str(data),
                        },
                    )
                    assert not result.is_error
                    assert result.structured_content["dry_run"] is True
                    assert result.structured_content["metadata"]["gradient_data"]["content_sha256"]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "settings",
    [
        {"strength": 0.5},
        {"strength": True},
        {"max_relative_change": 0.3},
        {"steps": 129},
        {"steps": True},
    ],
)
def test_gradient_settings_refused_before_runtime(tmp_path, monkeypatch, settings):
    service = Service(tmp_path / "workspace")
    target = target_file(tmp_path)
    monkeypatch.setattr(module, "HFRuntime", lambda *a: pytest.fail("loaded model"))
    with pytest.raises(ValueError, match="settings"):
        service.erase(
            tmp_path / "model", target, "npo-kl", gradient_data=data_file(tmp_path), **settings
        )
    assert service.store.list() == []


@pytest.mark.parametrize("method", ["npo-kl-repair", "grad-diff-kl-repair", "undial-kl-repair"])
@pytest.mark.parametrize("steps", [121, 128, True])
def test_repair_total_bound_before_runtime(tmp_path, monkeypatch, method, steps):
    monkeypatch.setattr(module, "HFRuntime", lambda *a: pytest.fail("invalid steps loaded model"))
    service = Service(tmp_path / "workspace")
    with pytest.raises(ValueError, match="1..120"):
        service.erase(
            tmp_path / "model",
            target_file(tmp_path),
            method,
            gradient_data=data_file(tmp_path),
            steps=steps,
            apply=True,
        )


@pytest.mark.parametrize("method", ["npo-kl-repair", "grad-diff-kl-repair", "undial-kl-repair"])
def test_repair_retain_count_refused_before_runtime(tmp_path, monkeypatch, method):
    monkeypatch.setattr(module, "HFRuntime", lambda *a: pytest.fail("invalid data loaded model"))
    path = data_file(tmp_path)
    data = json.loads(path.read_text())
    data["retain_development"] = data["retain_development"][:1]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="32 retain"):
        Service(tmp_path / "workspace").erase(
            tmp_path / "model", target_file(tmp_path), method, gradient_data=path, apply=True
        )


@pytest.mark.parametrize("steps", [113, 128, True])
def test_repair2_total_bound_before_runtime(tmp_path, monkeypatch, steps):
    monkeypatch.setattr(module, "HFRuntime", lambda *a: pytest.fail("invalid steps loaded model"))
    with pytest.raises(ValueError, match="1..112"):
        Service(tmp_path / "workspace").erase(
            tmp_path / "model",
            target_file(tmp_path),
            "grad-diff-kl-repair2",
            gradient_data=data_file(tmp_path),
            steps=steps,
            apply=True,
        )
