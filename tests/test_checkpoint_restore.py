"""Exercise exact disk undo, including values float arithmetic cannot reconstruct."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest

from lacunai.restore import checkpoints as cp


def weight_file(path: Path, tensors: dict[str, bytes], *, dtype: str = "F32") -> bytes:
    header = {"__metadata__": {"format": "pt", "note": "preserve padding and metadata"}}
    offset = 0
    width = {"F32": 4, "F16": 2}[dtype]
    for name, data in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": [len(data) // width],
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    payload = struct.pack("<Q", len(raw)) + raw + b"".join(tensors.values())
    path.write_bytes(payload)
    return payload


def source(tmp_path: Path, *, shards: bool = False) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen2"}')
    (root / "tokenizer.json").write_text('{"kind":"fixture"}')
    (root / "README.md").write_text("Model card")
    (root / "LICENSE").write_text("Apache-2.0")
    # Negative zero, subnormal, and adjacent representable values must survive byte-for-byte.
    data = struct.pack("<IIII", 0x80000000, 0x00000001, 0x3F800001, 0xBF800001)
    if shards:
        weight_file(root / "model-00001-of-00002.safetensors", {"a": data})
        weight_file(root / "model-00002-of-00002.safetensors", {"b": struct.pack("<ff", 5, 7)})
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors",
                    }
                }
            )
        )
    else:
        weight_file(root / "model.safetensors", {"a": data, "b": struct.pack("<ff", 5, 7)})
    return root


def raw_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only substitute the optional tensor conversion seam; actual parser, IO, hashes,
    # compression, transaction journal and restoration remain production code.
    def convert(value: bytes, row: dict) -> bytes:
        if len(value) != row["length"]:
            raise cp.SnapshotError("replacement byte count differs")
        return value

    monkeypatch.setattr(cp, "_replacement_bytes", convert)


def tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def edited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, shards: bool = False):
    raw_converter(monkeypatch)
    src = source(tmp_path, shards=shards)
    run = tmp_path / "run"
    selection = ["a", "b"] if shards else ["a"]
    snap = cp.prepare_snapshot(src, run, selection)
    replacements = {"a": struct.pack("<ffff", 0, 2, 3, 4)}
    if shards:
        replacements["b"] = struct.pack("<ff", 8, 9)
    out = tmp_path / "edited"
    receipt = cp.write_edited_checkpoint(src, out, replacements, snap)
    return src, run, out, receipt


def test_exact_restore_and_dry_run_preserve_headers_tokens_and_source(tmp_path, monkeypatch):
    src, run, out, receipt = edited(tmp_path, monkeypatch)
    before = tree(src)
    assert receipt["status"] == "edited"
    assert tree(out) != before
    assert set(receipt["tensors"]) == {"a"}
    blob = next((run / "before").iterdir())
    assert len(zlib.decompress(blob.read_bytes())) == 16
    assert len(zlib.decompress(blob.read_bytes())) < len(before["model.safetensors"])
    after = tree(out)
    dry = cp.restore_snapshot(run)
    assert dry["status"] == "planned" and not dry["apply"]
    assert tree(out) == after
    restored = cp.restore_snapshot(run, apply=True)
    assert restored["restored_bit_exact"] is True
    assert tree(out) == tree(src) == before
    assert blob.exists()


def test_shards_are_restored_exactly(tmp_path, monkeypatch):
    src, run, out, _ = edited(tmp_path, monkeypatch, shards=True)
    cp.restore_snapshot(run, apply=True)
    assert tree(out) == tree(src)


def test_explicit_in_place_requires_snapshot_and_restores_bytes(tmp_path, monkeypatch):
    raw_converter(monkeypatch)
    src = source(tmp_path)
    before = tree(src)
    run = tmp_path / "run"
    snap = cp.prepare_snapshot(src, run, ["a"])
    with pytest.raises(cp.SnapshotError, match="in-place"):
        cp.write_edited_checkpoint(src, src, {"a": bytes(16)}, snap)
    assert tree(src) == before
    cp.write_edited_checkpoint(src, src, {"a": bytes(16)}, snap, in_place=True)
    assert tree(src) != before
    cp.restore_snapshot(run, apply=True)
    assert tree(src) == before


@pytest.mark.parametrize("tamper", ["blob", "missing", "receipt"])
def test_snapshot_failure_precedes_any_checkpoint_output(tmp_path, monkeypatch, tamper):
    raw_converter(monkeypatch)
    src = source(tmp_path)
    before = tree(src)
    run = tmp_path / "run"
    snap = cp.prepare_snapshot(src, run, ["a"])
    if tamper == "blob":
        next((run / "before").iterdir()).write_bytes(b"corrupt")
    elif tamper == "missing":
        (run / "snapshot.json").rename(run / "missing.json")
    else:
        snap["source"] = str(tmp_path / "different")
    out = tmp_path / "edited"
    with pytest.raises((cp.SnapshotError, FileNotFoundError)):
        cp.write_edited_checkpoint(src, out, {"a": bytes(16)}, snap)
    assert not out.exists()
    assert tree(src) == before


def test_unwritable_snapshot_journal_precedes_checkpoint_mutation(tmp_path, monkeypatch):
    raw_converter(monkeypatch)
    src = source(tmp_path)
    run = tmp_path / "run"
    snap = cp.prepare_snapshot(src, run, ["a"])
    before = tree(src)

    def fail(*args, **kwargs):
        raise PermissionError("journal unwritable")

    monkeypatch.setattr(cp, "_record", fail)
    with pytest.raises(PermissionError, match="journal unwritable"):
        cp.write_edited_checkpoint(src, src, {"a": bytes(16)}, snap, in_place=True)
    assert tree(src) == before


@pytest.mark.parametrize("tamper", ["blob", "after", "extra", "link"])
def test_restore_preflights_all_inputs_before_any_mutation(tmp_path, monkeypatch, tamper):
    _, run, out, _ = edited(tmp_path, monkeypatch, shards=True)
    if tamper == "blob":
        blobs = sorted((run / "before").iterdir())
        blobs[-1].write_bytes(b"corrupt")
    elif tamper == "after":
        p = out / "model-00002-of-00002.safetensors"
        p.write_bytes(p.read_bytes()[:-1] + b"X")
    elif tamper == "extra":
        (out / "unrecorded.json").write_text("{}")
    else:
        p = out / "tokenizer.json"
        p.rename(out / "saved.json")
        p.symlink_to(out / "saved.json")
    before = tree(out)
    with pytest.raises((cp.SnapshotError, RuntimeError)):
        cp.restore_snapshot(run, apply=True)
    assert tree(out) == before


@pytest.mark.parametrize(
    "bad", ["overlap", "shape", "index", "python", "pickle", "symlink", "extra_shard"]
)
def test_reject_malformed_or_unsafe_checkpoint_before_snapshot(tmp_path, bad):
    src = source(tmp_path, shards=bad == "index")
    if bad in {"overlap", "shape"}:
        p = src / "model.safetensors"
        data = p.read_bytes()
        length = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + length])
        if bad == "overlap":
            header["b"]["data_offsets"] = [0, 8]
        else:
            header["a"]["shape"] = [999]
        raw = json.dumps(header).encode()
        p.write_bytes(struct.pack("<Q", len(raw)) + raw + data[8 + length :])
    elif bad == "index":
        p = src / "model.safetensors.index.json"
        index = json.loads(p.read_text())
        index["weight_map"]["a"] = "../outside.safetensors"
        p.write_text(json.dumps(index))
    elif bad == "python":
        (src / "modeling_custom.py").write_text("raise RuntimeError('must not execute')")
    elif bad == "pickle":
        (src / "pytorch_model.bin").write_bytes(b"pickle")
    elif bad == "symlink":
        (src / "linked.json").symlink_to(src / "config.json")
    else:
        weight_file(src / "extra.safetensors", {"c": bytes(4)})
    with pytest.raises((cp.SnapshotError, RuntimeError)):
        cp.prepare_snapshot(src, tmp_path / "run", ["a"])
    assert not (tmp_path / "run/snapshot.json").exists()


def test_source_change_after_snapshot_refuses_write(tmp_path, monkeypatch):
    raw_converter(monkeypatch)
    src = source(tmp_path)
    snap = cp.prepare_snapshot(src, tmp_path / "run", ["a"])
    (src / "config.json").write_text("{}")
    out = tmp_path / "edited"
    with pytest.raises(cp.SnapshotError, match="source checkpoint changed"):
        cp.write_edited_checkpoint(src, out, {"a": bytes(16)}, snap)
    assert not out.exists()


def test_budget_refusal_and_no_overwrite(tmp_path):
    src = source(tmp_path)
    with pytest.raises(cp.SnapshotError, match="budget"):
        cp.prepare_snapshot(src, tmp_path / "too-small", ["a"], max_bytes=1)
    cp.prepare_snapshot(src, tmp_path / "run", ["a"])
    before = tree(tmp_path / "run")
    with pytest.raises(cp.SnapshotError, match="already exists"):
        cp.prepare_snapshot(src, tmp_path / "run", ["a"])
    assert tree(tmp_path / "run") == before


def test_real_tensor_conversion_checks_shape_finiteness_and_storage_dtype(tmp_path):
    torch = pytest.importorskip("torch")
    descriptor = {"dtype": "F16", "shape": [2], "length": 4}
    raw = cp._replacement_bytes(torch.tensor([1.0, -0.0]), descriptor)
    assert raw == struct.pack("<HH", 0x3C00, 0x8000)
    for value in (torch.ones(3), torch.tensor([float("nan"), 0]), torch.tensor([1e30, 0])):
        with pytest.raises(cp.SnapshotError):
            cp._replacement_bytes(value, descriptor)


def test_interrupted_multishard_commit_restores_committed_file_only(tmp_path, monkeypatch):
    raw_converter(monkeypatch)
    src = source(tmp_path, shards=True)
    before = tree(src)
    run, out = tmp_path / "run", tmp_path / "edited"
    snap = cp.prepare_snapshot(src, run, ["a", "b"])
    real_replace = cp.os.replace

    def interrupt(src_path, dst_path):
        if str(dst_path).endswith("model-00002-of-00002.safetensors"):
            raise OSError("simulated second-shard interruption")
        return real_replace(src_path, dst_path)

    monkeypatch.setattr(cp.os, "replace", interrupt)
    with pytest.raises(OSError, match="second-shard"):
        cp.write_edited_checkpoint(src, out, {"a": bytes(16), "b": bytes(8)}, snap)
    monkeypatch.setattr(cp.os, "replace", real_replace)
    result = cp.restore_snapshot(run, apply=True)
    assert result["restored_bit_exact"]
    for name, data in before.items():
        assert (out / name).read_bytes() == data
    assert tree(src) == before
    assert list(out.glob("*.partial"))  # Interrupted staging evidence is retained, never deleted.


def test_interrupted_restore_resumes_without_touching_unrecognized_bytes(tmp_path, monkeypatch):
    src, run, out, _ = edited(tmp_path, monkeypatch, shards=True)
    real_replace = cp.os.replace

    def interrupt(src_path, dst_path):
        if str(dst_path).endswith("model-00002-of-00002.safetensors"):
            raise OSError("simulated restore interruption")
        return real_replace(src_path, dst_path)

    monkeypatch.setattr(cp.os, "replace", interrupt)
    with pytest.raises(OSError, match="restore interruption"):
        cp.restore_snapshot(run, apply=True)
    monkeypatch.setattr(cp.os, "replace", real_replace)
    assert json.loads((run / "snapshot.json").read_text())["status"] == "restoring"
    assert cp.restore_snapshot(run, apply=True)["restored_bit_exact"]
    assert tree(out) == tree(src)
    assert cp.restore_snapshot(run, apply=True)["files"] == []


def test_snapshot_preflight_happens_before_replacement_conversion(tmp_path, monkeypatch):
    src = source(tmp_path)
    run = tmp_path / "run"
    snap = cp.prepare_snapshot(src, run, ["a"])
    next((run / "before").iterdir()).write_bytes(b"broken")
    converted = []

    def conversion_must_not_happen(*args):
        converted.append(True)
        return bytes(16)

    monkeypatch.setattr(cp, "_replacement_bytes", conversion_must_not_happen)
    with pytest.raises(cp.SnapshotError, match="snapshot blob"):
        cp.write_edited_checkpoint(src, tmp_path / "edited", {"a": bytes(16)}, snap)
    assert converted == []


def test_progress_is_visible_before_write_and_failure_is_terminal(tmp_path, monkeypatch, capsys):
    src = source(tmp_path)
    run = tmp_path / "run"
    snap = cp.prepare_snapshot(src, run, ["a"])
    capsys.readouterr()

    def conversion(*args):
        lines = capsys.readouterr().err.splitlines()
        assert json.loads(lines[0]) == {"phase": "checkpoint_write", "status": "active"}
        raise ValueError("forced numerical rejection")

    monkeypatch.setattr(cp, "_replacement_bytes", conversion)
    with pytest.raises(ValueError, match="numerical rejection"):
        cp.write_edited_checkpoint(src, tmp_path / "edited", {"a": bytes(16)}, snap)
    assert json.loads(capsys.readouterr().err.splitlines()[-1])["status"] == "failed"
    assert not (tmp_path / "edited").exists()
