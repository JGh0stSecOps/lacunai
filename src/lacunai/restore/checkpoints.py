"""Durable touched-tensor before-images and bit-exact safetensors restoration.

Snapshots contain compressed original bytes for selected tensors, not model copies
or floating-point subtraction deltas. A snapshot is not an authenticity signature;
the run/report layer signs provenance separately.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
import sys
import time
import uuid
import zlib
from importlib import import_module
from pathlib import Path
from typing import Any

from lacunai.experimental.artifacts import atomic_json, secure_path, sha256_file

DEFAULT_MAX_BYTES = 8 * 1024**3
_MAX_HEADER = 16 * 1024**2
_FORMAT = "lacunai-tensor-before-images-v1"
_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}
_FLOATS = {"F16": "float16", "BF16": "bfloat16", "F32": "float32", "F64": "float64"}


class SnapshotError(ValueError):
    """A checkpoint or undo receipt failed a preflight invariant."""


def _emit(phase: str, status: str = "active", **fields: Any) -> None:
    print(json.dumps({"phase": phase, "status": status, **fields}), file=sys.stderr, flush=True)


def _hash(path: Path) -> str:
    return sha256_file(path, lambda event: print(json.dumps(event), file=sys.stderr, flush=True))


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _root(value: Path, *, existing: bool = True) -> Path:
    if "://" in str(value):
        raise SnapshotError("only local checkpoint paths are supported")
    path = value.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)) or path.resolve() != path:
        raise SnapshotError("redirected paths are refused")
    if existing and not path.is_dir():
        raise SnapshotError("checkpoint/run directory is absent")
    return path


def _regular(path: Path) -> None:
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise SnapshotError("checkpoint input must be a regular unlinked file")


def _json(path: Path) -> dict[str, Any]:
    _regular(path)
    if path.stat().st_size > _MAX_HEADER:
        raise SnapshotError("JSON metadata exceeds limit")
    value = json.loads(path.read_text(), object_pairs_hook=_unique)
    if not isinstance(value, dict):
        raise SnapshotError("JSON metadata must be an object")
    return value


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotError("duplicate JSON key")
        result[key] = value
    return result


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _record(run_dir: Path, value: dict[str, Any]) -> None:
    atomic_json(secure_path(run_dir, "snapshot.json"), value)
    _sync_directory(run_dir)
    if _json(run_dir / "snapshot.json") != value:
        raise SnapshotError("snapshot receipt readback differs")


def _inventory(root: Path, ignored: set[str] | None = None) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SnapshotError("checkpoint tree contains a link")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if ignored and relative in ignored:
            continue
        _regular(path)
        if path.suffix.lower() not in {
            ".safetensors",
            ".json",
            ".txt",
            ".md",
            ".model",
            ".tiktoken",
        } and path.name not in {
            "LICENSE",
            "LICENSE-APACHE",
            "LICENSE-MIT",
            ".gitattributes",
            "NOTICE",
        }:
            raise SnapshotError(f"unsupported checkpoint file: {relative}")
        records[relative] = {"size": path.stat().st_size, "sha256": _hash(path)}
    if not records or "config.json" not in records:
        raise SnapshotError("a local Hugging Face config.json is required")
    return records


def _layout(path: Path) -> dict[str, dict[str, Any]]:
    _regular(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise SnapshotError("truncated safetensors prefix")
        size = struct.unpack("<Q", prefix)[0]
        if size < 2 or size > _MAX_HEADER or size + 8 > path.stat().st_size:
            raise SnapshotError("invalid safetensors header length")
        header = json.loads(stream.read(size), object_pairs_hook=_unique)
    if not isinstance(header, dict):
        raise SnapshotError("safetensors header must be an object")
    payload_size = path.stat().st_size - 8 - size
    result = {}
    intervals = []
    for name, value in header.items():
        if name == "__metadata__":
            if not isinstance(value, dict) or any(not isinstance(v, str) for v in value.values()):
                raise SnapshotError("invalid safetensors metadata")
            continue
        if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
            raise SnapshotError("invalid tensor descriptor")
        dtype, shape, offsets = value["dtype"], value["shape"], value["data_offsets"]
        if (
            dtype not in _SIZES
            or not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)
        ):
            raise SnapshotError("invalid tensor dtype or dimensions")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
        ):
            raise SnapshotError("invalid tensor offsets")
        start, end = offsets
        if (
            start < 0
            or end < start
            or end > payload_size
            or end - start != math.prod(shape) * _SIZES[dtype]
        ):
            raise SnapshotError("tensor payload shape/offset mismatch")
        intervals.append((start, end))
        result[name] = {
            "dtype": dtype,
            "shape": shape,
            "offset": 8 + size + start,
            "length": end - start,
        }
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise SnapshotError("safetensors payload has overlapping ranges or holes")
        cursor = end
    if cursor != payload_size:
        raise SnapshotError("unindexed safetensors payload")
    return result


def _tensors(root: Path, inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = sorted(name for name in inventory if name.endswith(".safetensors"))
    if not files:
        raise SnapshotError("safetensors weights are required; pickle weights are refused")
    indexes = [name for name in inventory if name.endswith(".safetensors.index.json")]
    if len(indexes) > 1 or (len(files) > 1 and len(indexes) != 1):
        raise SnapshotError("multiple weight files require one safetensors shard index")
    result = {}
    for filename in files:
        for name, descriptor in _layout(secure_path(root, filename)).items():
            if name in result:
                raise SnapshotError("duplicate tensor across shards")
            result[name] = {"file": filename, **descriptor}
    if indexes:
        mapping = _json(secure_path(root, indexes[0])).get("weight_map")
        if not isinstance(mapping, dict) or set(mapping) != set(result):
            raise SnapshotError("shard index does not exactly cover tensors")
        for name, filename in mapping.items():
            if not isinstance(filename, str):
                raise SnapshotError("invalid shard path")
            secure_path(root, filename)
            if filename != result[name]["file"]:
                raise SnapshotError("shard index differs from stored tensor")
    return result


def _read_slice(path: Path, descriptor: dict[str, Any]) -> bytes:
    with path.open("rb") as stream:
        stream.seek(descriptor["offset"])
        data = stream.read(descriptor["length"])
    if len(data) != descriptor["length"]:
        raise SnapshotError("tensor bytes truncated")
    return data


def _blob(run_dir: Path, row: dict[str, Any], max_bytes: int) -> bytes:
    if type(row.get("length")) is not int or not 0 <= row["length"] <= max_bytes:
        raise SnapshotError("invalid before-image length")
    path = secure_path(run_dir, row["blob"])
    _regular(path)
    if path.stat().st_size != row["compressed_size"] or path.stat().st_size > max_bytes:
        raise SnapshotError("snapshot blob length differs")
    compressed = path.read_bytes()
    if _digest(compressed) != row["compressed_sha256"]:
        raise SnapshotError("snapshot blob hash differs")
    decoder = zlib.decompressobj()
    data = decoder.decompress(compressed, row["length"] + 1)
    if (
        not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
        or len(data) != row["length"]
        or _digest(data) != row["before_sha256"]
    ):
        raise SnapshotError("snapshot before-image differs")
    return data


def prepare_snapshot(
    source: Path, run_dir: Path, tensor_names: list[str], max_bytes: int = DEFAULT_MAX_BYTES
) -> dict[str, Any]:
    """Persist and read back exact selected tensor bytes before numerical edits."""
    _emit("snapshot_prepare")
    try:
        source, run_dir = _root(source), _root(run_dir, existing=False)
        if source == run_dir or source.is_relative_to(run_dir) or run_dir.is_relative_to(source):
            raise SnapshotError("snapshot and checkpoint trees must be separate")
        if (
            type(max_bytes) is not int
            or max_bytes <= 0
            or not tensor_names
            or len(set(tensor_names)) != len(tensor_names)
        ):
            raise SnapshotError("positive budget and unique nonempty tensor selection required")
        inventory = _inventory(source)
        layout = _tensors(source, inventory)
        if any(name not in layout or layout[name]["dtype"] not in _FLOATS for name in tensor_names):
            raise SnapshotError("snapshot selects an absent or unsupported non-floating tensor")
        size = sum(layout[name]["length"] for name in tensor_names)
        checkpoint_bytes = sum(row["size"] for row in inventory.values())
        staged_bytes = sum(
            inventory[name]["size"] for name in {layout[n]["file"] for n in tensor_names}
        )
        if size + checkpoint_bytes + staged_bytes > max_bytes:
            raise SnapshotError("snapshot and checkpoint exceed write budget")
        run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if (run_dir / "snapshot.json").exists():
            raise SnapshotError("snapshot already exists")
        blobs = secure_path(run_dir, "before")
        blobs.mkdir(mode=0o700)
        records = {}
        for index, name in enumerate(tensor_names, 1):
            row = layout[name]
            data = _read_slice(secure_path(source, row["file"]), row)
            compressed = zlib.compress(data)
            filename = f"before/{_digest(name.encode())}.z"
            path = secure_path(run_dir, filename)
            fd = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(compressed)
                stream.flush()
                os.fsync(stream.fileno())
            record = {
                **row,
                "blob": filename,
                "before_sha256": _digest(data),
                "compressed_sha256": _digest(compressed),
                "compressed_size": len(compressed),
            }
            if _blob(run_dir, record, max_bytes) != data:
                raise SnapshotError("snapshot verification failed")
            records[name] = record
            _emit("snapshot_prepare", completed=index, total=len(tensor_names), unit="tensors")
        _sync_directory(blobs)
        if _inventory(source) != inventory:
            raise SnapshotError("source changed while preparing snapshot")
        value = {
            "format": _FORMAT,
            "snapshot_id": uuid.uuid4().hex,
            "run_dir": str(run_dir),
            "source": str(source),
            "status": "prepared",
            "max_bytes": max_bytes,
            "before_files": inventory,
            "tensors": records,
        }
        _record(run_dir, value)
        _emit("snapshot_prepare", "complete", snapshot_id=value["snapshot_id"], tensor_bytes=size)
        return value
    except Exception as exc:
        _emit("snapshot_prepare", "failed", error=str(exc))
        raise


def _validated(
    run_dir: Path, supplied: dict[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, bytes]]:
    run_dir = _root(run_dir)
    value = _json(secure_path(run_dir, "snapshot.json"))
    if supplied is not None and value != supplied:
        raise SnapshotError("supplied snapshot differs from durable receipt")
    if (
        value.get("format") != _FORMAT
        or value.get("run_dir") != str(run_dir)
        or not value.get("tensors")
    ):
        raise SnapshotError("invalid snapshot receipt")
    if type(value.get("max_bytes")) is not int or value["max_bytes"] <= 0:
        raise SnapshotError("invalid snapshot budget")
    lengths = [row.get("length") for row in value["tensors"].values()]
    if any(type(n) is not int or n < 0 for n in lengths) or sum(lengths) > value["max_bytes"]:
        raise SnapshotError("snapshot decompressed bytes exceed budget")
    data = {name: _blob(run_dir, row, value["max_bytes"]) for name, row in value["tensors"].items()}
    if sum(len(item) for item in data.values()) > value["max_bytes"]:
        raise SnapshotError("snapshot decompressed bytes exceed budget")
    return value, data


def _replacement_bytes(tensor: Any, descriptor: dict[str, Any]) -> bytes:
    torch = import_module("torch")
    if not isinstance(tensor, torch.Tensor) or list(tensor.shape) != descriptor["shape"]:
        raise SnapshotError("replacement is not a tensor with the recorded shape")
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise SnapshotError("replacement must contain finite floating-point values")
    converted = tensor.detach().cpu().to(getattr(torch, _FLOATS[descriptor["dtype"]])).contiguous()
    if not bool(torch.isfinite(converted).all()):
        raise SnapshotError("replacement overflows the source dtype")
    data = bytes(converted.view(torch.uint8).numpy().tobytes())
    if len(data) != descriptor["length"]:
        raise SnapshotError("replacement byte count differs")
    return data


def _copy(source: Path, target: Path) -> None:
    _regular(source)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    total = source.stat().st_size
    _emit("checkpoint_copy", completed=0, total=total, unit="bytes", file=target.name)
    completed = 0
    last = time.monotonic()
    with source.open("rb") as src, target.open("xb") as dst:
        for block in iter(lambda: src.read(1 << 20), b""):
            dst.write(block)
            completed += len(block)
            if time.monotonic() - last >= 5:
                _emit(
                    "checkpoint_copy",
                    completed=completed,
                    total=total,
                    unit="bytes",
                    file=target.name,
                )
                last = time.monotonic()
        dst.flush()
        os.fsync(dst.fileno())
    if completed != total:
        raise SnapshotError("source size changed during copy")
    _sync_directory(target.parent)
    _emit(
        "checkpoint_copy",
        "complete",
        completed=completed,
        total=total,
        unit="bytes",
        file=target.name,
    )


def write_edited_checkpoint(
    source: Path,
    output: Path,
    replacements: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    in_place: bool = False,
) -> dict[str, Any]:
    """Write raw tensor payloads only after durable snapshot and source validation."""
    _emit("checkpoint_write")
    try:
        source, output = _root(source), _root(output, existing=False)
        run_dir = _root(Path(snapshot["run_dir"]))
        record, originals = _validated(run_dir, snapshot)
        if record["status"] != "prepared" or record["source"] != str(source):
            raise SnapshotError("snapshot is not a prepared receipt for this source")
        if not replacements or set(replacements) != set(record["tensors"]):
            raise SnapshotError("replacement set differs from snapshotted tensors")
        if output == source and not in_place or in_place and output != source:
            raise SnapshotError("in-place requires explicit opt-in and identical source/output")
        if output != source and (
            output.exists() or output.is_relative_to(source) or source.is_relative_to(output)
        ):
            raise SnapshotError("new output must be absent and separate from source")
        if output == run_dir or output.is_relative_to(run_dir) or run_dir.is_relative_to(output):
            raise SnapshotError("output overlaps snapshot storage")
        if _inventory(source) != record["before_files"]:
            raise SnapshotError("source checkpoint changed since snapshot")
        layout = _tensors(source, record["before_files"])
        replacements_raw = {}
        for name, row in record["tensors"].items():
            if layout.get(name) != {
                k: row[k] for k in ("file", "dtype", "shape", "offset", "length")
            }:
                raise SnapshotError("snapshot tensor descriptor differs from source header")
            if _read_slice(secure_path(source, row["file"]), row) != originals[name]:
                raise SnapshotError("snapshot before-image differs from source tensor")
            replacements_raw[name] = _replacement_bytes(replacements[name], row)
        # Prove journal replacement works before a checkpoint path is created or changed.
        record = {**record, "output": str(output), "status": "writing", "in_place": in_place}
        _record(run_dir, record)
        if not in_place:
            output.mkdir(mode=0o700, parents=True)
            for filename in record["before_files"]:
                _copy(secure_path(source, filename), secure_path(output, filename))
        elif any(
            secure_path(output, row["file"]).stat().st_nlink != 1
            for row in record["tensors"].values()
        ):
            raise SnapshotError("in-place writes to multiply linked files are refused")
        by_file: dict[str, list[str]] = {}
        for name, row in record["tensors"].items():
            by_file.setdefault(row["file"], []).append(name)
        staged = {}
        after = dict(record["before_files"])
        for filename, names in by_file.items():
            destination = secure_path(output, filename)
            temporary = destination.with_name(
                f".{destination.name}.{record['snapshot_id']}.partial"
            )
            _copy(destination, temporary)
            with temporary.open("r+b") as stream:
                for name in names:
                    stream.seek(record["tensors"][name]["offset"])
                    stream.write(replacements_raw[name])
                stream.flush()
                os.fsync(stream.fileno())
            for name in names:
                if _read_slice(temporary, record["tensors"][name]) != replacements_raw[name]:
                    raise SnapshotError("edited tensor readback differs")
            staged[filename] = temporary.relative_to(output).as_posix()
            after[filename] = {"size": temporary.stat().st_size, "sha256": _hash(temporary)}
        record = {**record, "after_files": after, "staged_files": staged}
        _record(run_dir, record)
        # Every after-image and journal has been verified before atomic file replacement.
        for filename, staged_name in staged.items():
            destination = secure_path(output, filename)
            if _hash(destination) != record["before_files"][filename]["sha256"]:
                raise SnapshotError("checkpoint changed before atomic replacement")
            os.replace(secure_path(output, staged_name), destination)
            _sync_directory(destination.parent)
        if _inventory(output) != after:
            raise SnapshotError("edited checkpoint full-file readback differs")
        record = {**record, "status": "edited"}
        _record(run_dir, record)
        _emit("checkpoint_write", "complete", output=str(output), snapshot_id=record["snapshot_id"])
        return record
    except Exception as exc:
        _emit("checkpoint_write", "failed", error=str(exc))
        raise


def restore_snapshot(run_dir: Path, *, apply: bool = False) -> dict[str, Any]:
    """Restore only recognized after-images; unchanged files in interrupted commits are safe."""
    _emit("checkpoint_restore")
    try:
        record, originals = _validated(_root(run_dir))
        if record.get("status") not in {
            "edited",
            "writing",
            "restoring",
            "restored",
        } or not record.get("after_files"):
            raise SnapshotError("snapshot has no verified after-image transaction")
        output = _root(Path(record["output"]))
        stages = set(record.get("staged_files", {}).values()) | set(
            record.get("restore_staged_files", {}).values()
        )
        current = _inventory(output, stages)
        if set(current) != set(record["after_files"]):
            raise SnapshotError("checkpoint contains missing or unrecorded files")
        layout = _tensors(output, current)
        to_restore = set()
        for filename, actual in current.items():
            expected = record["after_files"][filename]
            if actual == expected:
                if expected != record["before_files"][filename]:
                    to_restore.add(filename)
            elif (
                record["status"] in {"writing", "restoring", "restored"}
                and actual == record["before_files"][filename]
            ):
                continue
            else:
                raise SnapshotError("checkpoint differs from its recorded after-image")
        for name, row in record["tensors"].items():
            if layout.get(name) != {
                k: row[k] for k in ("file", "dtype", "shape", "offset", "length")
            }:
                raise SnapshotError("snapshot offsets differ from current tensor header")
        for filename, temporary in record.get("staged_files", {}).items():
            path = secure_path(output, temporary)
            if path.exists() and (
                _hash(path) != record["after_files"][filename]["sha256"]
                or path.stat().st_size != record["after_files"][filename]["size"]
            ):
                raise SnapshotError("staged transaction file differs")
        for filename, staged_name in record.get("restore_staged_files", {}).items():
            path = secure_path(output, staged_name)
            if path.exists() and (
                _hash(path) != record["before_files"][filename]["sha256"]
                or path.stat().st_size != record["before_files"][filename]["size"]
            ):
                raise SnapshotError("staged restore file differs")
        result = {
            "snapshot_id": record["snapshot_id"],
            "output": str(output),
            "apply": apply,
            "status": "planned",
            "files": sorted(to_restore),
            "format": _FORMAT,
        }
        if not apply:
            _emit("checkpoint_restore", "complete", dry_run=True, files=len(to_restore))
            return result
        # Stage complete restored files before changing any destination. A journal makes
        # a partially committed multi-shard restore resumable without accepting unknown bytes.
        restore_stages = dict(record.get("restore_staged_files", {}))
        for filename in sorted(to_restore):
            destination = secure_path(output, filename)
            if (
                destination.stat().st_nlink != 1
                or _hash(destination) != record["after_files"][filename]["sha256"]
            ):
                raise SnapshotError("checkpoint changed after restore preflight")
            if (
                filename in restore_stages
                and secure_path(output, restore_stages[filename]).exists()
            ):
                continue
            temporary = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.restore-partial"
            )
            _copy(destination, temporary)
            with temporary.open("r+b") as stream:
                for name, row in record["tensors"].items():
                    if row["file"] == filename:
                        stream.seek(row["offset"])
                        stream.write(originals[name])
                stream.flush()
                os.fsync(stream.fileno())
            if _hash(temporary) != record["before_files"][filename]["sha256"]:
                raise SnapshotError("restored whole-file hash differs; destination unchanged")
            restore_stages[filename] = temporary.relative_to(output).as_posix()
        record = {**record, "status": "restoring", "restore_staged_files": restore_stages}
        _record(_root(run_dir), record)
        stages |= set(restore_stages.values())
        for filename in sorted(to_restore):
            destination = secure_path(output, filename)
            if _hash(destination) != record["after_files"][filename]["sha256"]:
                raise SnapshotError("checkpoint changed before restore commit")
            os.replace(secure_path(output, restore_stages[filename]), destination)
            _sync_directory(destination.parent)
        if _inventory(output, stages) != record["before_files"]:
            raise SnapshotError("restored checkpoint does not match original files")
        _record(_root(run_dir), {**record, "status": "restored", "restored_bit_exact": True})
        result.update(
            status="restored", restored_bit_exact=True, before_files=record["before_files"]
        )
        _emit("checkpoint_restore", "complete", restored_bit_exact=True)
        return result
    except Exception as exc:
        _emit("checkpoint_restore", "failed", error=str(exc))
        raise
