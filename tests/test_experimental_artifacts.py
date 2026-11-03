# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Real file/byte tests for the experimental boundary, not model experiments."""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest

from lacunai.experimental.artifacts import (
    ArtifactError,
    Budget,
    atomic_json,
    secure_path,
    sha256_file,
    tree_bytes,
    verify_originals,
)
from lacunai.experimental.prepare import (
    CheckedRedirect,
    Downloader,
    extract_uv,
    parse_lock,
    run_command,
    select_wheel,
    validate_url,
    wheel_score,
)


def budget(root: Path, events: list[dict[str, Any]], limit: int = 10000) -> Budget:
    return Budget(root, 100 << 20, limit, int(time.time()) + 120, events.append)


class Response(io.BytesIO):
    def __init__(self, body: bytes, events: list[dict[str, Any]], length: int | None = None):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body) if length is None else length)}
        self.events = events

    def read(self, size: int = -1) -> bytes:
        assert any(
            e.get("phase") == "transfer" and e.get("status") == "active" for e in self.events
        ), "progress must precede first response-body read"
        return super().read(size)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "nested/../../escape"])
def test_path_traversal_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(ArtifactError, match="relative path"):
        secure_path(tmp_path, path)


def test_symlink_and_original_mutation_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactError, match="symlink"):
        secure_path(tmp_path, "redirect/weights")
    original = tmp_path / "models/target/model.safetensors"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"original")
    original.chmod(0o444)
    records = [
        {
            "path": "models/target/model.safetensors",
            "size": 8,
            "sha256": hashlib.sha256(b"original").hexdigest(),
        }
    ]
    assert verify_originals(tmp_path, records)[0]["verified"]
    original.chmod(0o600)
    original.write_bytes(b"changed!")
    with pytest.raises(ArtifactError, match="changed"):
        verify_originals(tmp_path, records)


def test_network_budget_is_reserved_and_persisted_before_download(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    current = budget(tmp_path, events, 10)
    current.reserve_network(9)
    current.received(2)
    restored = budget(tmp_path, events, 10)
    assert restored.network_reserved_bytes == 9
    assert restored.network_received_bytes == 2
    with pytest.raises(ArtifactError, match="network"):
        restored.reserve_network(2)
    assert json.loads((tmp_path / "receipts/network.json").read_text())["reserved_body_bytes"] == 9


def test_disk_budget_counts_partial_files(tmp_path: Path) -> None:
    (tmp_path / "weights.partial").write_bytes(b"x" * 50)
    current = Budget(tmp_path, (1 << 20) + 100, 100, int(time.time()) + 120)
    assert tree_bytes(tmp_path) == 50
    with pytest.raises(ArtifactError, match="disk"):
        current.check(51)


def test_transfer_verified_only_after_actual_bytes_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[dict[str, Any]] = []
    body = b"real bytes for a synthetic checkpoint"
    client = Downloader(budget(tmp_path, events))
    monkeypatch.setattr(client, "_open", lambda *_args: Response(body, events))
    receipt = client.file(
        "https://huggingface.co/example",
        "models/target/test.bin",
        len(body),
        hashlib.sha256(body).hexdigest(),
        readonly=True,
    )
    assert (tmp_path / receipt["path"]).read_bytes() == body
    assert events[0]["status"] == "active"
    assert events[-1] == {"phase": "verification", "label": "test.bin", "status": "complete"}
    assert client.budget.network_received_bytes == len(body)
    assert not ((tmp_path / receipt["path"]).stat().st_mode & 0o222)


def test_hash_mismatch_preserves_partial_and_does_not_promote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[dict[str, Any]] = []
    client = Downloader(budget(tmp_path, events))
    monkeypatch.setattr(client, "_open", lambda *_args: Response(b"wrong", events))
    with pytest.raises(ArtifactError, match="hash mismatch"):
        client.file("https://huggingface.co/example", "models/target/test.bin", 5, "0" * 64)
    assert not (tmp_path / "models/target/test.bin").exists()
    assert (tmp_path / "models/target/test.bin.partial").read_bytes() == b"wrong"
    assert events[-1]["status"] == "failed"


def test_incorrect_size_fails_and_ends_active_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[dict[str, Any]] = []
    client = Downloader(budget(tmp_path, events))
    monkeypatch.setattr(client, "_open", lambda *_args: Response(b"too long", events))
    with pytest.raises(ArtifactError, match="size|length"):
        client.file("https://huggingface.co/example", "bad", 2, None)
    assert not (tmp_path / "bad").exists()
    assert events[-1]["status"] == "failed"


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/a",
        "https://huggingface.co.evil/a",
        "https://user:password@huggingface.co/a",
        "https://127.0.0.1/a",
        "file:///etc/passwd",
    ],
)
def test_unapproved_sources_and_redirects_rejected(url: str) -> None:
    with pytest.raises(ArtifactError, match="HTTPS"):
        validate_url(url)
    with pytest.raises(ArtifactError, match="HTTPS"):
        CheckedRedirect().redirect_request(None, None, 302, "redirect", {}, url)


def test_lock_and_wheel_selection_do_not_install_unapproved_distribution() -> None:
    allowed = "a" * 64
    entries = parse_lock(f"sample==1.2 \\\n    --hash=sha256:{allowed}\n")
    metadata = {
        "urls": [
            {
                "filename": "sample-1.2-cp313-cp313-manylinux_2_17_x86_64.whl",
                "size": 10,
                "digests": {"sha256": "b" * 64},
            },
            {"filename": "sample-1.2-py3-none-any.whl", "size": 10, "digests": {"sha256": allowed}},
            {
                "filename": "sample-1.2-cp313-cp313-win_amd64.whl",
                "size": 10,
                "digests": {"sha256": allowed},
            },
        ]
    }
    assert select_wheel(metadata, entries[0])["filename"] == "sample-1.2-py3-none-any.whl"
    assert wheel_score("tokenizers-0.21.4-cp39-abi3-manylinux_2_17_x86_64.whl") is not None
    with pytest.raises(ArtifactError, match="unhashed"):
        parse_lock("sample>=1.2\n")


def test_uv_extraction_ignores_other_members_and_refuses_replacement(tmp_path: Path) -> None:
    wheel = tmp_path / "uv.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("uv/uv", b"binary")
        archive.writestr("../../escape", b"must not extract")
    extracted = extract_uv(wheel, tmp_path, budget(tmp_path, []))
    assert extracted.read_bytes() == b"binary"
    assert not (tmp_path.parent / "escape").exists()
    with pytest.raises(ArtifactError, match="already exists"):
        extract_uv(wheel, tmp_path, budget(tmp_path, []))


def test_atomic_record_refuses_symlink_without_changing_target(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.write_text("preserve")
    (tmp_path / "record").symlink_to(original)
    with pytest.raises(ArtifactError, match="redirected"):
        atomic_json(tmp_path / "record", {"overwrite": True})
    assert original.read_text() == "preserve"


def test_installer_output_reaches_progress_before_child_completion(tmp_path: Path) -> None:
    import sys

    events: list[dict[str, Any]] = []
    run_command(
        [sys.executable, "-u", "-c", "print('actual subprocess output', flush=True)"],
        tmp_path,
        budget(tmp_path, events),
    )
    statuses = [event["status"] for event in events]
    assert statuses[0] == "active"
    assert statuses[-1] == "complete"
    assert statuses.index("output") < statuses.index("complete")


def test_deadline_refuses_work_before_network(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="deadline"):
        Budget(tmp_path, 100 << 20, 100, int(time.time()) - 1)


def test_streaming_hash_matches_independent_sha256(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "payload"
    path.write_bytes(bytes(range(256)) * 8000)
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_progress_is_flushed_while_child_is_still_running() -> None:
    import subprocess
    import sys

    script = (
        "from lacunai.experimental.artifacts import emit; import time; "
        "emit({'phase': 'working', 'status': 'active'}); time.sleep(1)"
    )
    started = time.monotonic()
    process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    assert process.stdout is not None
    try:
        event = json.loads(process.stdout.readline())
        assert event["status"] == "active"
        assert time.monotonic() - started < 0.8, "progress buffered until worker completion"
        assert process.poll() is None, "flushed progress must arrive before process completion"
        assert process.wait(timeout=5) == 0
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
