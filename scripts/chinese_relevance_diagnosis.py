"""Pinned training-only answer-loss relevance diagnosis; never edits weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import chinese_distribution_diagnosis as distribution  # noqa: E402
import chinese_iteration as iteration  # noqa: E402
from lacunai.core.model import HFRuntime, active_phase, checkpoint_inventory, progress  # noqa: E402
from lacunai.core.spec import RuntimeConfig, digest  # noqa: E402
from lacunai.experimental.artifacts import atomic_json  # noqa: E402
from lacunai.experimental.resources import Watchdog, host_sample  # noqa: E402

VERSION = "answer-loss-relevance-diagnosis-v1"


def pinned_corpus(path: Path, pin: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{64}", pin):
        raise ValueError("invalid corpus SHA256")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("linked corpus refused")
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("corpus exceeds bounded input size")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError("corpus bytes differ from pinned SHA256")
    value: dict[str, Any] = json.loads(raw)
    fields = {
        "schema_version",
        "forget_train",
        "retain_train",
        "source_dataset_sha256",
        "provenance",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("training corpus schema differs; no evaluation splits accepted")
    for split, count in (("forget_train", 64), ("retain_train", 80)):
        rows = value[split]
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError("training corpus count differs")
        ids = set()
        for item in rows:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("id"), str)
                or not item["id"].startswith(split + ":")
                or item["id"] in ids
                or not isinstance(item.get("family"), str)
                or not item["family"]
                or not isinstance(item.get("answer"), str)
                or not item["answer"]
                or not isinstance(item.get("prompts"), list)
                or not item["prompts"]
                or any(not isinstance(x, str) or not x.strip() for x in item["prompts"])
            ):
                raise ValueError("invalid or duplicated training item")
            ids.add(item["id"])
    if not isinstance(value["source_dataset_sha256"], str) or not re.fullmatch(
        r"[a-f0-9]{64}", value["source_dataset_sha256"]
    ):
        raise ValueError("corpus provenance hash missing")
    if not isinstance(value["provenance"], str) or not value["provenance"].strip():
        raise ValueError("corpus provenance missing")
    return value


canonical_hashes = distribution.canonical_hashes


def persist(
    directory: Path, metadata: dict[str, Any], payload: dict[str, Any], watch: Any
) -> dict[str, Any]:
    from lacunai.experimental.resources import tree_bytes
    from lacunai.experimental.tensors import file_hash, validate_write_budget

    metadata_path, payload_path = (
        directory / "diagnosis.json",
        directory / "sensitivities.safetensors",
    )
    if metadata_path.exists() or payload_path.exists():
        raise ValueError("diagnosis output already exists")
    if not payload:
        raise ValueError("empty sensitivity payload")
    # Safetensors data plus conservative per-entry JSON/header and atomic JSON
    # allowance. Reserve against the same aggregate tree as the watchdog.
    payload_bytes = sum(v.numel() * v.element_size() for v in payload.values())
    header_bytes = sum(
        len(k.encode("utf-8")) + len(str(list(v.shape))) + 256 for k, v in payload.items()
    )
    metadata_bytes = len(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
    additional = payload_bytes + header_bytes + 2 * metadata_bytes + 1024**2
    roots = {watch.root, watch.artifact_budget_root or watch.root}
    for root in roots:
        validate_write_budget(tree_bytes(root), additional, watch.limits["disk_max_bytes"])
    torch = import_module("torch")
    storage = import_module("safetensors.torch")
    mapper = import_module("lacunai.map.language_relevance")
    mapper.validate_payload(metadata, payload)
    for value in payload.values():
        if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise ValueError("invalid sensitivity payload tensor")
    watch.check()
    with active_phase("relevance_persistence"):
        storage.save_file(payload, str(payload_path))
        atomic_json(metadata_path, metadata)
    with active_phase("relevance_artifact_readback"):
        reloaded = storage.load_file(str(payload_path), device="cpu")
        mapper.validate_payload(metadata, reloaded)
        if set(reloaded) != set(payload) or any(
            reloaded[k].dtype != v.dtype
            or reloaded[k].shape != v.shape
            or not torch.equal(reloaded[k], v)
            for k, v in payload.items()
        ):
            raise ValueError("sensitivity roundtrip differs")
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("metadata roundtrip differs")
        watch.check()
        return {p.name: file_hash(p) for p in (metadata_path, payload_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "corpus"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-fingerprint", "corpus-sha256", "name"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    progress("relevance_diagnosis", state="active")
    directory = None
    status = {
        "phase": "relevance-diagnose",
        "state": "running",
        "version": VERSION,
        "weights_edited": False,
        "absence_proven": False,
    }
    try:
        if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.name) is None:
            raise ValueError("invalid run name")
        if not re.fullmatch(r"[a-f0-9]{64}", args.source_fingerprint):
            raise ValueError("invalid source fingerprint")
        corpus = pinned_corpus(args.corpus, args.corpus_sha256)
        if not args.apply:
            progress("relevance_diagnosis", state="plan_only", launch=False, overall_deadline=None)
            return 0
        root = args.root.absolute()
        source = args.source.absolute()
        if any(p.is_symlink() for p in (root, *root.parents)):
            raise ValueError("linked workspace refused")
        if root == source or root in source.parents or source in root.parents:
            raise ValueError("workspace and source overlap")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        candidate = root / args.name
        candidate.mkdir(mode=0o700, exist_ok=False)
        directory = candidate
        atomic_json(directory / "status.json", status)
        config = RuntimeConfig(max_seconds=None, max_new_tokens=64, seed=42, torch_threads=2)
        if not (root / "host-before.json").exists():
            atomic_json(root / "host-before.json", host_sample(root))
        with Watchdog(root, {"resources": iteration.LIMITS}, None, args.name) as watch:
            with active_phase("relevance_source_load"):
                before = checkpoint_inventory(source, config.max_model_bytes)
                if digest(before) != args.source_fingerprint:
                    raise ValueError("source differs from pinned checkpoint")
                runtime: Any = HFRuntime(source, config)
                if runtime.inventory != before:
                    raise ValueError("source changed during loading")
            runtime_check = runtime.check

            def checked_runtime() -> None:
                runtime_check()
                watch.check()

            runtime.check = checked_runtime
            code_root = Path(__file__).resolve().parents[1]
            code = [
                Path(__file__),
                Path(iteration.__file__),
                Path(distribution.__file__),
                *sorted((code_root / "src").rglob("*.py")),
            ]
            atomic_json(
                directory / "manifest.json",
                {
                    "version": VERSION,
                    "operation": "answer_loss_relevance_diagnosis",
                    "checkpoint_fingerprint": args.source_fingerprint,
                    "source_inventory": before,
                    "corpus_bytes_sha256": args.corpus_sha256,
                    "corpus": corpus,
                    "runtime": asdict(config),
                    "resources": dict(iteration.LIMITS),
                    "overall_deadline": None,
                    "code_hashes": {
                        str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in code
                    },
                    "weights_edited": False,
                    "claim_limit": (
                        "Training answer-loss mask sensitivity only; "
                        "no removal or causal efficacy claim"
                    ),
                },
            )
            original = canonical_hashes(runtime, watch)
            try:
                with active_phase("relevance_answer_loss_diagnosis"):
                    mapper = import_module("lacunai.map.language_relevance")
                    metadata, payload = mapper.diagnose(
                        runtime, {key: corpus[key] for key in ("forget_train", "retain_train")}
                    )
                status["artifacts_sha256"] = persist(directory, metadata, payload, watch)
            finally:
                with active_phase("relevance_source_unchanged_verification"):
                    # An expired pressure guard must not skip read-only integrity evidence.
                    runtime.check = lambda: None
                    after = canonical_hashes(runtime, SimpleNamespace(check=lambda: None))
                    after_files = checkpoint_inventory(source, config.max_model_bytes)
                    unchanged = original == after and before == after_files
                    atomic_json(
                        directory / "source-verification.json",
                        {
                            "canonical_before": original,
                            "canonical_after": after,
                            "files_before": before,
                            "files_after": after_files,
                            "unchanged": unchanged,
                        },
                    )
                    if not unchanged:
                        raise ValueError("diagnosis modified original tensors or checkpoint")
            watch.check()
            status.update(state="complete", source_unchanged=True)
            atomic_json(directory / "status.json", status)
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        if directory is not None:
            atomic_json(directory / "status.json", status)
        progress("relevance_diagnosis", state="failed", error=status["error"])
        raise
    progress("relevance_diagnosis", state="complete", source_unchanged=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
