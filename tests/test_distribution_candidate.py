"""Software and real-tensor contracts; fixtures do not establish unlearning efficacy."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from lacunai.experimental import data

spec = importlib.util.spec_from_file_location(
    "distribution_candidate",
    Path(__file__).parents[1] / "scripts/chinese_distribution_candidate.py",
)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
SPEC = dict(schema_version=1, method="distribution_row_ablation", top_k=1, execution_fraction=0.01)


@pytest.mark.parametrize(
    "key,value",
    [
        ("top_k", True),
        ("top_k", 0),
        ("top_k", -1),
        ("top_k", 1.5),
        ("top_k", "1"),
        ("execution_fraction", True),
        ("execution_fraction", 0.02),
        ("execution_fraction", float("nan")),
        ("schema_version", True),
        ("extra", "code"),
    ],
)
def test_strict_root_spec(key, value):
    runner.validate_spec(SPEC)
    with pytest.raises(ValueError, match="root-selected"):
        runner.validate_spec({**SPEC, key: value})


def records(items):
    return [
        dict(
            item_id=item["id"],
            family=item["family"],
            variant=n,
            prompt=prompt,
            output=item["answer"],
            input_tokens=1,
            output_tokens=1,
            token_ids=[1],
            first_step_logits_sha256="a" * 64,
            truncated=False,
        )
        for item in items
        for n, prompt in enumerate(item["prompts"])
    ]


def fixture(tmp_path):
    primary = [
        data._primary_item("dev_primary", family, 0, (4, 0, 4), (0, 1, 0))
        for family in runner.evaluation.CONTRACT["required_primary_families"]
    ]
    retain = [data._retain_item("dev_retain", "english", 0, (0, 0, 0))]
    dataset = dict(dev_primary=primary, dev_retain=retain)
    contract = copy.deepcopy(runner.evaluation.CONTRACT)
    contract.pop("primary_baseline_unknowns")
    contract.update(
        name="chinese-literacy-zero-recovery-v2", minimum_receptive_accuracy_drop="0.50"
    )
    before = dict(
        development=True,
        contract=contract,
        scorer_version="4",
        primary=dict(spec=primary, raw=records(primary)),
        retain=dict(spec=retain, raw=records(retain)),
        dataset_sha256=runner.digest(dict(primary=primary, retain=retain)),
        inference_sha256="b" * 64,
        perplexity={
            "perplexity": "3.5",
            "tokens": 10,
            "corpus_split": "dev_retain",
            "corpus_sha256": runner.digest(
                [item["prompts"][0] + "\n" + item["answer"] for item in retain]
            ),
        },
    )
    args = [
        "screen",
        "--root",
        str(tmp_path / "runs"),
        "--source",
        str(tmp_path / "source"),
        "--source-fingerprint",
        "a" * 64,
        "--name",
        "screen-01",
    ]
    for key, value in [
        ("diagnosis", {"checkpoint_fingerprint": "a" * 64}),
        ("edit-spec", SPEC),
        ("dataset", dataset),
        ("before", before),
        ("histograms", None),
    ]:
        path = tmp_path / key / "input.json"
        path.parent.mkdir()
        path.write_bytes(json.dumps(value).encode() if value is not None else b"payload")
        args += [
            "--" + key,
            str(path),
            "--" + key + "-sha256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
        ]
    (tmp_path / "before" / "manifest.json").write_text(
        json.dumps({"checkpoint_fingerprint": "a" * 64})
    )
    return args, dataset, before


def test_versioned_raw_rescore_preserves_historical_input(tmp_path):
    _, dataset, before = fixture(tmp_path)
    original = copy.deepcopy(before)
    updated = runner.rescore_before(before, dataset)
    assert before == original
    assert updated["contract"] == runner.evaluation.CONTRACT
    assert updated["scorer_version"] == runner.evaluation.SCORER_VERSION
    assert updated["perplexity"] == before["perplexity"]
    assert updated["primary"]["raw"] == before["primary"]["raw"]
    before["dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        runner.rescore_before(before, dataset)


def test_dryrun_does_not_load_or_write(tmp_path, monkeypatch, capsys):
    args, _, _ = fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner", *args])
    monkeypatch.setattr(runner, "HFRuntime", lambda *a: pytest.fail("model loaded"))
    assert runner.main() == 0
    assert not (tmp_path / "runs").exists()
    assert "plan_only" in capsys.readouterr().err


def test_bad_pin_emits_terminal_failure_before_load(tmp_path, monkeypatch, capsys):
    args, _, _ = fixture(tmp_path)
    args[args.index("--histograms-sha256") + 1] = "0" * 64
    monkeypatch.setattr(sys, "argv", ["runner", *args])
    with pytest.raises(ValueError, match="pin"):
        runner.main()
    output = capsys.readouterr().err
    assert "active" in output and "failed" in output
    assert not (tmp_path / "runs").exists()


def test_existing_phase_never_overwritten(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path)
    directory = tmp_path / "runs" / "screen-01"
    directory.mkdir(parents=True)
    marker = directory / "status.json"
    marker.write_text("original")
    monkeypatch.setattr(sys, "argv", ["runner", *args, "--apply"])
    with pytest.raises(FileExistsError):
        runner.main()
    assert marker.read_text() == "original"


def test_real_snapshot_apply_reference_and_restore(tmp_path):
    pytest.importorskip("torch")
    from test_language_edit import numerical_diagnosis, numerical_runtime

    runtime = numerical_runtime(tmp_path)
    proposal = runner.operator.propose(runtime, numerical_diagnosis(runtime), 0.01)
    reference = runner.reference_without_copies(runtime, proposal)
    names = runner.operator.touched_tensors(runtime, proposal)
    snapshot = runner.prepare_snapshot(runtime.path, tmp_path / "snapshot", names)
    state = runner.operator.apply(runtime, proposal, snapshot)
    assert any(
        runner.tensors.tensor_hash(runtime.model.state_dict()[n]) != reference["hashes"][n]
        for n in names
    )
    assert runner.operator.restore(runtime, state)["restored_bit_exact"]
    assert {
        n: runner.tensors.tensor_hash(v) for n, v in runtime.model.state_dict().items()
    } == reference["hashes"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("perplexity", "Infinity"),
        ("perplexity", "NaN"),
        ("perplexity", "0"),
        ("perplexity", "-1"),
        ("tokens", True),
        ("tokens", 0),
        ("corpus_sha256", "0" * 64),
        ("corpus_split", "heldout_retain"),
    ],
)
def test_invalid_baseline_perplexity_rejected(tmp_path, field, value):
    _, dataset, before = fixture(tmp_path)
    before["perplexity"][field] = value
    with pytest.raises(ValueError, match="perplexity"):
        runner.rescore_before(before, dataset)


def test_snapshot_aggregate_reservation_precedes_writer(tmp_path, monkeypatch):
    from types import SimpleNamespace

    root = tmp_path / "root"
    root.mkdir()
    (root / "existing").write_bytes(b"x" * 20)
    tensor = SimpleNamespace(numel=lambda: 100, element_size=lambda: 4)
    runtime = SimpleNamespace(
        path=tmp_path / "source", model=SimpleNamespace(state_dict=lambda: {"weight": tensor})
    )
    watch = SimpleNamespace(
        root=root,
        artifact_budget_root=root,
        limits={"disk_max_bytes": 4 * 1024**2 + 410},
        check=lambda: None,
    )
    calls = []
    monkeypatch.setattr(runner, "prepare_snapshot", lambda *a, **kw: calls.append(kw) or {})
    with pytest.raises(ValueError, match="budget"):
        runner.budgeted_snapshot(runtime, root / "run", ["weight"], watch)
    assert calls == []
    watch.limits["disk_max_bytes"] += 100
    runner.budgeted_snapshot(runtime, root / "run", ["weight"], watch)
    assert calls == [{"max_bytes": watch.limits["disk_max_bytes"] - 20}]


@pytest.mark.parametrize("bad_inference", [False, True])
def test_main_screen_failure_restores_and_checks_original_files(
    tmp_path, monkeypatch, bad_inference
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    args, _, _ = fixture(tmp_path)
    inventory = {"config.json": "a" * 64}
    fingerprint = runner.digest(inventory)
    args[args.index("--source-fingerprint") + 1] = fingerprint
    for key in ("diagnosis",):
        path = Path(args[args.index("--" + key) + 1])
        path.write_text(json.dumps({"checkpoint_fingerprint": fingerprint}))
        args[args.index("--" + key + "-sha256") + 1] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "before" / "manifest.json").write_text(
        json.dumps({"checkpoint_fingerprint": fingerprint})
    )
    contract = {
        "runtime": {},
        "resources": {"target_prompt_tokens": 128, "target_output_tokens": 64},
    }
    path = Path(args[args.index("--before") + 1])
    before = json.loads(path.read_text())
    before["inference_sha256"] = "0" * 64 if bad_inference else runner.digest(contract)
    path.write_text(json.dumps(before))
    args[args.index("--before-sha256") + 1] = hashlib.sha256(path.read_bytes()).hexdigest()
    events = []
    value = SimpleNamespace(shape=(1, 1), dtype="float32", current="original")
    runtime = SimpleNamespace(
        path=tmp_path / "source",
        inventory=inventory,
        contract=contract,
        model=SimpleNamespace(state_dict=lambda: {"weight": value}),
        check=lambda: None,
    )
    watch = SimpleNamespace(check=lambda: None)
    monkeypatch.setattr(sys, "argv", ["runner", *args, "--apply"])
    monkeypatch.setattr(runner, "host_sample", lambda *a: {})
    monkeypatch.setattr(runner, "Watchdog", lambda *a: nullcontext(watch))
    monkeypatch.setattr(
        runner, "checkpoint_inventory", lambda *a: events.append("files") or inventory
    )
    monkeypatch.setattr(runner, "HFRuntime", lambda *a: runtime)
    monkeypatch.setattr(runner.tensors, "tensor_hash", lambda v: v.current)
    monkeypatch.setattr(
        runner,
        "import_module",
        lambda name: SimpleNamespace(
            load_file=lambda *a, **kw: {},
            to_proposal=lambda *a, **kw: {"source": {"norm_squared": 1}},
        ),
    )
    monkeypatch.setattr(runner.operator, "touched_tensors", lambda *a: ["weight"])
    monkeypatch.setattr(runner, "budgeted_snapshot", lambda *a: {})
    monkeypatch.setattr(runner.iteration, "bind_snapshot", lambda *a: None)

    def apply(*a):
        value.current = "edited"
        return {"before_images": {}, "diagnostics": {}}

    def restore(*a):
        events.append("restore")
        value.current = "original"
        return {"restored_bit_exact": True}

    monkeypatch.setattr(runner.operator, "apply", apply)
    monkeypatch.setattr(runner.operator, "restore", restore)
    monkeypatch.setattr(runner.gradient, "save_after_image", lambda *a: {})

    def fail(*a):
        raise RuntimeError("injected evaluation failure")

    monkeypatch.setattr(runner.rmu, "screen", fail)
    if bad_inference:
        with pytest.raises(ValueError, match="inference identity"):
            runner.main()
        assert events == ["files"]
        assert value.current == "original"
        return
    with pytest.raises(RuntimeError, match="injected evaluation"):
        runner.main()
    assert events == ["files", "restore", "files"]
    directory = tmp_path / "runs" / "screen-01"
    assert json.loads((directory / "restore.json").read_text())["canonical_source_hashes_verified"]
    assert json.loads((directory / "status.json").read_text())["state"] == "failed"
    assert value.current == "original"
