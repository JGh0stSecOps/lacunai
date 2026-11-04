"""Software contract tests; fixture strings are not the private evaluation dataset."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.test_connected_pilot import pilot


def fixture():
    spec = SimpleNamespace(
        schema_version=1,
        target_id="fixture",
        mapping=[],
        layers=[23],
        thresholds={},
        perplexity_texts=["fixture prose"],
        probes=[SimpleNamespace(prompt="q target", answer="fixture")],
        retain=[SimpleNamespace(prompt="q retain")],
    )
    final = deepcopy(spec)
    final.probes[0].prompt = "f target"
    final.retain[0].prompt = "f retain"
    data = {
        "schema_version": 1,
        "model_revision": "revision",
        "provenance": "synthetic fixture",
        "target_aliases": ["fixture"],
    }
    for key, count in (("forget", 16), ("retain_development", 32), ("benign_retain", 32)):
        data[key] = [{"prompt": f"{key} fixture {i}", "answer": "fixture"} for i in range(count)]
    return data, spec, final


def test_exact_schema_and_disjoint_partitions():
    module = pilot.pretrained_api()
    data, q, f = fixture()
    module.validate_data(data, q, f, "revision")
    for key in ("primary", "utility"):
        with pytest.raises(ValueError, match="envelope"):
            module.validate_data({**data, key: []}, q, f, "revision")
    for partition in ("forget", "retain_development", "benign_retain"):
        broken = deepcopy(data)
        broken[partition][0]["prompt"] = q.probes[0].prompt
        with pytest.raises(ValueError, match="overlap"):
            module.validate_data(broken, q, f, "revision")
    f.layers = [22]
    with pytest.raises(ValueError, match="shared contract"):
        module.validate_data(data, q, f, "revision")


def test_pretrained_dispatch_never_enters_planting_route(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pilot,
        "execute_shared",
        lambda *args, **kwargs: calls.append(kwargs) or {"status": "fixture"},
    )
    data, q, f = fixture()
    assert pilot.execute({"profile": "pretrained-fact"}, q, f, data) == {"status": "fixture"}
    assert calls == [{"pretrained": True}]
    assert pilot.plan_profile({}) == "connected-12"
    with pytest.raises(ValueError):
        pilot.plan_profile({"profile": "unknown"})


def test_qualification_rejects_forged_target_summary():
    from lacunai.core.spec import Probe, Thresholds
    from lacunai.verify.evaluation import _suite

    probes = [Probe("fixture question", "fixture")]
    runtime = SimpleNamespace(
        check=lambda: None,
        progress=lambda *a, **k: None,
        generate=lambda _: {
            "output": "wrong",
            "truncated": False,
            "input_tokens": 1,
            "output_tokens": 1,
            "token_ids": [1],
        },
    )
    target = _suite(runtime, probes, "target")
    target["by_kind"] = {"direct": {"ambiguous": 0, "missing": 0, "recovery_rate": "1"}}
    spec = SimpleNamespace(probes=probes, retain=probes, thresholds=Thresholds())
    runtime.generate = lambda _: {
        "output": "fixture",
        "truncated": False,
        "input_tokens": 1,
        "output_tokens": 1,
        "token_ids": [1],
    }
    with pytest.raises(ValueError, match="not demonstrated"):
        pilot.pretrained_api().qualify(
            {"target": target, "retain": _suite(runtime, probes, "retain")}, spec
        )


def test_accented_recovery_cannot_pass():
    from lacunai.verify.evaluation import classify

    assert classify("París", "Paris", False)["classification"] == "not_recovered"
    result = pilot.pretrained_api().alias_comparison(
        {"verdict": "passed_the_stated_experimental_contract", "reasons": []},
        {"target": {"rows": [{"index": 0, "generation": {"output": "París", "truncated": False}}]}},
        ["Paris", "París"],
    )
    assert result["verdict"] == "failed"
    assert result["supplementary_alias_recovery"]["recovered"] == 1


def test_benign_and_final_perplexity_separation():
    data, q, f = fixture()
    data["benign_retain"][0]["prompt"] = "  FIXTURE   prose "
    with pytest.raises(ValueError, match="perplexity"):
        pilot.pretrained_api().validate_data(data, q, f, "revision")


@pytest.mark.parametrize("fail", [False, True])
def test_pretrained_materialization_and_candidate_unwind(tmp_path, fail):
    torch = pytest.importorskip("torch")
    from tests.test_connected_pilot import Budget, checkpoint

    source, scratch = tmp_path / "source", tmp_path / "scratch"
    checkpoint(source, 1.0)
    checkpoint(scratch, 2.0)
    name = "model.layers.0.mlp.up_proj.weight"
    identity = pilot.compatible_scratch(source, scratch, name)
    workspace = tmp_path / "records"
    workspace.mkdir()
    storage = pilot.Storage(Budget(), workspace, scratch, name)
    try:
        with storage.transaction("outer-original", {name: torch.ones(2, 2)}):
            assert pilot.inventory(scratch) == identity["source_inventory"]
            with storage.transaction("candidate", {name: torch.full((2, 2), 1.1)}):
                assert pilot.inventory(scratch) != identity["source_inventory"]
                if fail:
                    raise RuntimeError("injected evaluation failure")
            assert pilot.inventory(scratch) == identity["source_inventory"]
    except RuntimeError as exc:
        assert fail and str(exc) == "injected evaluation failure"
    assert pilot.inventory(source) == identity["source_inventory"]
    assert pilot.inventory(scratch) == identity["scratch_inventory"]
