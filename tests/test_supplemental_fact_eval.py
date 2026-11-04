import importlib.util
from pathlib import Path

import pytest


def module():
    p = Path(__file__).resolve().parents[1] / "scripts/supplemental_fact_eval.py"
    s = importlib.util.spec_from_file_location("supplement", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def rows():
    return [
        {
            "id": str(i),
            "split": "target" if i < 16 else "retain",
            "language": "en" if i < 8 else "es" if i < 12 else "zh",
            "kind": "pretrained_numeric_fact" if i < 20 else "neighboring_task_capability",
            "classification": "recovered",
        }
        for i in range(24)
    ]


def test_actual_pairing_missing_baseline_and_recovery_failure():
    m = module()
    before = rows()
    after = rows()
    for x in after[:16]:
        x["classification"] = "not_recovered"
    assert m.aggregate(before, after)["verdict"] == "passed the stated experimental contract"
    before[0]["classification"] = "not_recovered"
    assert m.aggregate(before, after)["verdict"] == "inconclusive"
    after[1].update(classification="ambiguous", observed_target_string=True)
    assert m.aggregate(before, after)["verdict"] == "failed"
    assert m.aggregate(before, after)["target_recoveries"] == 1


def test_retain_loss_and_ambiguous_outputs_never_pass():
    m = module()
    before = rows()
    after = rows()
    for x in after[:16]:
        x["classification"] = "not_recovered"
    after[16]["classification"] = "missing"
    assert m.aggregate(before, after)["paired_retain_losses"] == ["16"]
    after[16]["classification"] = "recovered"
    after[0]["classification"] = "ambiguous"
    assert m.aggregate(before, after)["verdict"] == "inconclusive"
    after.pop()
    with pytest.raises(ValueError, match="identities"):
        m.aggregate(before, after)


def test_receipt_binding_and_eight_tensor_guard():
    m = module()
    desc = {"shape": [1], "dtype": "BF16", "before_sha256": "a" * 64, "length": 2}
    snapshot = {
        "before_files": {"model.safetensors": {"sha256": "b" * 64}},
        "tensors": {"x": desc},
        "snapshot_id": "s",
    }
    row = {"shape": [1], "dtype": "BF16", "before_raw_sha256": "a" * 64, "bytes": 2}
    after = {
        "tensors": {"x": row},
        "snapshot_id": "s",
        "artifact_sha256": "c" * 64,
        "artifact": "after.safetensors",
        "artifact_bytes": 64,
    }
    blob = {"path": "/tmp/after.safetensors", "sha256": "c" * 64}
    inventory = {"model.safetensors": "b" * 64}
    lineage(snapshot, after)
    m.validate_receipts(snapshot, after, inventory, blob)
    with pytest.raises(ValueError, match="before inventory"):
        m.validate_receipts(snapshot, after, {}, blob)
    after["tensors"]["x"]["dtype"] = "F32"
    with pytest.raises(ValueError, match="descriptor"):
        m.validate_receipts(snapshot, after, inventory, blob)
    after["tensors"]["x"]["dtype"] = "BF16"
    snapshot["tensors"] = {str(i): desc for i in range(9)}
    after["tensors"] = {str(i): row for i in range(9)}
    with pytest.raises(ValueError, match="inventory"):
        m.validate_receipts(snapshot, after, inventory, blob)


def test_pinned_mismatch_and_symlink_refuse(tmp_path):
    m = module()
    p = tmp_path / "fixture.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        m.pinned({"path": str(p), "sha256": "0" * 64})
    q = tmp_path / "link"
    q.symlink_to(p)
    with pytest.raises(ValueError, match="redirected"):
        m.safe(q)


def tiny_afterimage(tmp_path):
    """Offline actual Qwen forward, BF16 storage, and the production transaction."""
    import hashlib
    import json
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    safe = pytest.importorskip("safetensors.torch")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = (
        transformers.Qwen2ForCausalLM(
            transformers.Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                tie_word_embeddings=False,
                attention_dropout=0.0,
            )
        )
        .eval()
        .to(torch.bfloat16)
        .float()
    )
    model.requires_grad_(False)
    name = "lm_head.weight"
    original = model.state_dict()[name].to(torch.bfloat16)
    altered = original.clone()
    altered[7, :] += 1

    def raw(v):
        return v.contiguous().view(torch.uint8).numpy().tobytes()

    blob = tmp_path / "after.safetensors"
    safe.save_file({name: altered}, str(blob))
    inventory = {"model.safetensors": "a" * 64}
    runtime = SimpleNamespace(model=model, storage_dtypes={name: "BF16"}, inventory=inventory)
    desc = {
        "shape": list(original.shape),
        "dtype": "BF16",
        "before_sha256": hashlib.sha256(raw(original)).hexdigest(),
        "length": len(raw(original)),
    }
    snapshot = {
        "before_files": {"model.safetensors": {"sha256": "a" * 64}},
        "tensors": {name: desc},
        "snapshot_id": "tiny",
    }
    receipt = {
        "tensors": {
            name: {
                "shape": desc["shape"],
                "dtype": "BF16",
                "bytes": desc["length"],
                "before_raw_sha256": desc["before_sha256"],
                "after_raw_sha256": hashlib.sha256(raw(altered)).hexdigest(),
            }
        },
        "snapshot_id": "tiny",
        "artifact": blob.name,
        "artifact_bytes": blob.stat().st_size,
        "artifact_sha256": hashlib.sha256(blob.read_bytes()).hexdigest(),
    }
    lineage(snapshot, receipt)
    output = tmp_path / "evidence"
    output.mkdir()
    ids = torch.tensor([[1, 2, 3, 4]])

    def logits():
        with torch.no_grad():
            return model(input_ids=ids, use_cache=False).logits.clone()

    return module(), runtime, snapshot, receipt, blob, output, logits, torch, json


@pytest.mark.parametrize("generation_error", [False, True])
def test_real_tiny_qwen_afterimage_changes_logits_and_restores(tmp_path, generation_error, capsys):
    m, runtime, snapshot, receipt, blob, out, logits, torch, json = tiny_afterimage(tmp_path)
    canonical = m.state_hashes(runtime)
    before = logits()
    seen = []

    def operation():
        after = logits()
        seen.append(after)
        assert not torch.equal(before, after)
        if generation_error:
            raise RuntimeError("injected generation failure")
        return {"measured_logit_change": True}

    if generation_error:
        with pytest.raises(RuntimeError, match="generation failure"):
            m.candidate_transaction(
                runtime, snapshot, receipt, blob, out, "rank-one", canonical, operation
            )
    else:
        assert m.candidate_transaction(
            runtime, snapshot, receipt, blob, out, "rank-one", canonical, operation
        )["measured_logit_change"]
    assert len(seen) == 1
    assert m.state_hashes(runtime) == canonical
    assert torch.equal(logits(), before)
    assert json.loads((out / "rank-one-restore.json").read_text())["in_memory_bit_exact"]


@pytest.mark.parametrize("corruption", ["blob_hash", "before_hash", "shape"])
def test_real_afterimage_corruption_rejected_without_weight_drift(tmp_path, corruption):
    m, runtime, snapshot, receipt, blob, out, logits, torch, _ = tiny_afterimage(tmp_path)
    canonical = m.state_hashes(runtime)
    before = logits()
    name = "lm_head.weight"
    if corruption == "blob_hash":
        receipt["artifact_sha256"] = "0" * 64
    elif corruption == "before_hash":
        receipt["tensors"][name]["before_raw_sha256"] = "0" * 64
        snapshot["tensors"][name]["before_sha256"] = "0" * 64
    else:
        receipt["tensors"][name]["shape"] = [1, 512]
        snapshot["tensors"][name]["shape"] = [1, 512]
    with pytest.raises(ValueError):
        m.candidate_transaction(
            runtime,
            snapshot,
            receipt,
            blob,
            out,
            "rank-one",
            canonical,
            lambda: pytest.fail("invalid candidate evaluated"),
        )
    assert m.state_hashes(runtime) == canonical
    assert torch.equal(logits(), before)


def test_cli_failure_reports_terminal_state_without_model(tmp_path):
    import os
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts/supplemental_fact_eval.py"
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    env = dict(os.environ, PYTHONPATH=str(script.parent.parent / "src"))
    process = subprocess.run(
        [sys.executable, str(script), "--plan", str(plan), "--plan-sha256", "0" * 64, "--apply"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert process.returncode == 1
    import json

    progress = [json.loads(line) for line in process.stdout.splitlines()]
    assert progress[0]["phase"] == "preparing"
    assert progress[-1]["phase"] == "failed"
    assert not any(item["phase"] == "complete" for item in progress)


def lineage(snapshot, after):
    snapshot.update(
        status="restored",
        run_dir="/fixture/runs/run-1",
        after_files={"model.safetensors": {"sha256": "d" * 64}},
    )
    after.update(
        snapshot_directory=snapshot["run_dir"], checkpoint_after_files=snapshot["after_files"]
    )
    for name, desc in snapshot["tensors"].items():
        desc.update(file="model.safetensors", blob="before/fixture.bin")
        after["tensors"][name].update(source_file=desc["file"], before_image_blob=desc["blob"])


@pytest.mark.parametrize("mutation", ["duplicate", "invalid", "language", "split", "kind"])
def test_strict_paired_rows_refuse_false_green(mutation):
    m = module()
    before = rows()
    after = rows()
    for row in after[:16]:
        row["classification"] = "not_recovered"
    if mutation == "duplicate":
        after.append(dict(after[0]))
    elif mutation == "invalid":
        after[0]["classification"] = "invented"
    else:
        after[0][mutation] = "changed"
    with pytest.raises(ValueError):
        m.aggregate(before, after)


def test_candidate_summary_binding_refuses_other_method_evidence():
    m = module()
    snapshot = {"run_dir": "/fixture/runs/run-1"}
    after = {"snapshot_id": "s", "artifact_sha256": "a" * 64}
    item = {"method": "rank-one"}
    summary = {
        "candidates": [{"method": "rank-one", "run_id": "run-1", "afterimages": dict(after)}]
    }
    m.bind_candidate(item, snapshot, after, summary)
    summary["candidates"][0]["afterimages"]["artifact_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="bound"):
        m.bind_candidate(item, snapshot, after, summary)


def test_signed_terminal_artifacts_detect_tampering(tmp_path):
    import json

    from lacunai.verify.reports import verify_report

    m = module()
    output = tmp_path / "output"
    output.mkdir()
    keys = tmp_path / "keys"
    status = {"state": "failed", "verdict": "inconclusive", "error": "fixture exact failure"}
    m.write(output, "status.json", status)
    m.sign_terminal(output, keys, status)
    assert verify_report(output, keys / "public.key")["verified"]
    assert (
        json.loads((output / "report.json").read_text())["scientific_summary"]["error"]
        == "fixture exact failure"
    )
    (output / "status.json").write_text("{}")
    with pytest.raises(ValueError):
        verify_report(output, keys / "public.key")


def test_transaction_prevalidation_records_exact_error_and_ends_progress(tmp_path, capsys):
    from types import SimpleNamespace

    m = module()
    # The production boundary records failures even before optional Torch import succeeds.
    with pytest.raises(Exception) as error:
        m.candidate_transaction(
            SimpleNamespace(), {}, {}, tmp_path / "missing", tmp_path, "rank-one", {}, lambda: None
        )
    import json

    record = json.loads((tmp_path / "rank-one-error.json").read_text())
    assert record["error"] == str(error.value)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["phase"] == "candidate_transaction" and events[-1]["state"] == "failed"


def test_after_checkpoint_binding_is_mandatory():
    m = module()
    desc = {"shape": [1], "dtype": "BF16", "before_sha256": "a" * 64, "length": 2}
    snapshot = {
        "before_files": {"model.safetensors": {"sha256": "b" * 64}},
        "tensors": {"x": desc},
        "snapshot_id": "s",
    }
    after = {
        "tensors": {
            "x": {"shape": [1], "dtype": "BF16", "before_raw_sha256": "a" * 64, "bytes": 2}
        },
        "snapshot_id": "s",
        "artifact_sha256": "c" * 64,
        "artifact": "after.safetensors",
        "artifact_bytes": 64,
    }
    lineage(snapshot, after)
    after["checkpoint_after_files"] = {"model.safetensors": {"sha256": "e" * 64}}
    with pytest.raises(ValueError, match="after checkpoint inventory"):
        m.validate_receipts(
            snapshot,
            after,
            {"model.safetensors": "b" * 64},
            {"path": "/tmp/after.safetensors", "sha256": "c" * 64},
        )


@pytest.mark.parametrize(
    "methods", [("ablate-neurons", "rank-one", "ablate-direction"), ("npo-kl", "grad-diff-kl")]
)
def test_terminal_cohort_preserves_legacy_and_gradient_methods(methods):
    m = module()
    terminal = {"candidates": [{"method": method} for method in methods]}
    assert m.expected_methods(terminal) == set(methods)


@pytest.mark.parametrize("method", ["npo-kl", "grad-diff-kl"])
def test_gradient_receipt_and_method_summary_binding(method):
    m = module()
    assert method in m.METHODS
    inventory = {"model.safetensors": "b" * 64}
    name = "model.layers.23.mlp.up_proj.weight"
    snapshot = {
        "before_files": {"model.safetensors": {"sha256": "b" * 64}},
        "tensors": {name: {"shape": [1], "dtype": "BF16", "before_sha256": "a" * 64, "length": 2}},
        "snapshot_id": "gradient-snapshot",
    }
    after = {
        "tensors": {
            name: {"shape": [1], "dtype": "BF16", "before_raw_sha256": "a" * 64, "bytes": 2}
        },
        "snapshot_id": "gradient-snapshot",
        "artifact_sha256": "c" * 64,
        "artifact": "after.safetensors",
        "artifact_bytes": 64,
    }
    lineage(snapshot, after)
    m.validate_receipts(
        snapshot, after, inventory, {"path": "/tmp/after.safetensors", "sha256": "c" * 64}
    )
    other = "grad-diff-kl" if method == "npo-kl" else "npo-kl"
    summary = {
        "candidates": [
            {"method": method, "run_id": "run-1", "afterimages": after},
            {"method": other, "run_id": "other-run", "afterimages": after},
        ]
    }
    m.bind_candidate({"method": method}, snapshot, after, summary)
    with pytest.raises(ValueError, match="bound"):
        m.bind_candidate({"method": other}, snapshot, after, summary)


@pytest.mark.parametrize(
    "methods", [[], ["npo-kl", "npo-kl"], ["unknown"], ["npo-kl"], ["rank-one"]]
)
def test_invalid_terminal_cohort_refused(methods):
    with pytest.raises(ValueError, match="cohort"):
        module().expected_methods({"candidates": [{"method": x} for x in methods]})


@pytest.mark.parametrize(
    "methods", [("ablate-neurons", "rank-one", "ablate-direction"), ("npo-kl", "grad-diff-kl")]
)
def test_complete_and_partial_cohort_verdict(methods):
    m = module()
    expected = m.expected_methods({"candidates": [{"method": x} for x in methods]})
    passed = "passed the stated experimental contract"
    assert m.cohort_verdict([passed] * len(methods), []) == passed
    absent = sorted(expected - set(methods[:-1]))
    assert m.cohort_verdict([passed] * (len(methods) - 1), absent) == "inconclusive"
