# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Counted public downloads and an offline, experiment-local Python environment."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import selectors
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactError,
    Budget,
    Progress,
    atomic_json,
    emit,
    private_directory,
    secure_path,
    sha256_file,
    verify_originals,
)

PUBLIC_HOSTS = frozenset(
    {
        "pypi.org",
        "files.pythonhosted.org",
        "download.pytorch.org",
        "download-r2.pytorch.org",
        "huggingface.co",
    }
)
PUBLIC_SUFFIXES = (".huggingface.co", ".hf.co")
METADATA_MAX_BYTES = 4 << 20


def validate_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or not (host in PUBLIC_HOSTS or host.endswith(PUBLIC_SUFFIXES))
    ):
        raise ArtifactError("download URL is outside approved public HTTPS sources")


class CheckedRedirect(urllib.request.HTTPRedirectHandler):
    """Validate every redirected request before opening its connection."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Downloader:
    """A bounded downloader with no retry and no credentials or ambient proxies."""

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), CheckedRedirect()
        )

    def _open(self, url: str, method: str = "GET") -> Any:
        validate_url(url)
        request = urllib.request.Request(  # noqa: S310 — HTTPS allowlist checked above.
            urllib.parse.urldefrag(url)[0],
            method=method,
            headers={"User-Agent": "LacunAI-research-pilot/1", "Accept-Encoding": "identity"},
        )
        return self.opener.open(request, timeout=20)

    def head_size(self, url: str) -> int:
        self.budget.check()
        emit({"phase": "metadata", "label": "wheel size", "status": "active"}, self.budget.progress)
        with self._open(url, "HEAD") as response:
            size = response.headers.get("Content-Length")
            if not size or not size.isdigit() or int(size) <= 0:
                raise ArtifactError("download source did not declare a positive content length")
            return int(size)

    def _stream(
        self, url: str, output: Any, maximum: int, label: str, expected_size: int | None = None
    ) -> tuple[int, str]:
        self.budget.reserve_network(maximum)
        emit(
            {
                "phase": "transfer",
                "label": label,
                "completed": 0,
                "total": expected_size,
                "unit": "bytes",
                "status": "active",
            },
            self.budget.progress,
        )
        completed = 0
        digest = hashlib.sha256()
        last = time.monotonic()
        try:
            with self._open(url) as response:
                length = response.headers.get("Content-Length")
                if length and (not length.isdigit() or int(length) > maximum):
                    raise ArtifactError("declared response exceeds admitted transfer size")
                if expected_size is not None and length and int(length) != expected_size:
                    raise ArtifactError("declared response length differs from pinned size")
                while completed < maximum:
                    self.budget.check()
                    chunk = response.read(min(1 << 20, maximum - completed))
                    if not chunk:
                        break
                    self.budget.received(len(chunk))
                    completed += len(chunk)
                    output.write(chunk)
                    digest.update(chunk)
                    if time.monotonic() - last >= 5:
                        emit(
                            {
                                "phase": "transfer",
                                "label": label,
                                "completed": completed,
                                "total": expected_size,
                                "unit": "bytes",
                                "status": "active",
                            },
                            self.budget.progress,
                        )
                        last = time.monotonic()
            if completed >= maximum:
                raise ArtifactError("response reached its admitted maximum body size")
            if expected_size is not None and completed != expected_size:
                raise ArtifactError("downloaded length differs from pinned size")
        except Exception:
            emit(
                {
                    "phase": "transfer",
                    "label": label,
                    "completed": completed,
                    "total": expected_size,
                    "unit": "bytes",
                    "status": "failed",
                },
                self.budget.progress,
            )
            raise
        emit(
            {
                "phase": "transfer",
                "label": label,
                "completed": completed,
                "total": expected_size,
                "unit": "bytes",
                "status": "copied_pending_hash",
            },
            self.budget.progress,
        )
        return completed, digest.hexdigest()

    def metadata(self, url: str) -> dict[str, Any]:
        output = io.BytesIO()
        self._stream(url, output, METADATA_MAX_BYTES, "package metadata")
        value = json.loads(output.getvalue())
        if not isinstance(value, dict):
            raise ArtifactError("package metadata is not a JSON object")
        return value

    def file(
        self,
        url: str,
        relative: str | Path,
        size: int,
        expected_hash: str | None,
        *,
        readonly: bool = False,
    ) -> dict[str, Any]:
        path = secure_path(self.budget.root, relative)
        if path.exists():
            if expected_hash is None:
                raise ArtifactError("existing artifact has no approved hash for reuse")
            digest = sha256_file(path, self.budget.progress)
            if digest != expected_hash or path.stat().st_size != size:
                raise ArtifactError("existing artifact differs from approved bytes")
            if readonly and path.stat().st_mode & 0o222:
                raise ArtifactError("existing original model file is writable")
            return {
                "path": path.relative_to(self.budget.root).as_posix(),
                "size": size,
                "sha256": digest,
                "source": urllib.parse.urldefrag(url)[0],
                "reused": True,
            }
        partial = secure_path(self.budget.root, str(relative) + ".partial")
        if partial.exists():
            raise ArtifactError("prior partial download retained; automatic retry is disabled")
        self.budget.check(size + 1)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(partial, flags, 0o600)
        with os.fdopen(fd, "wb") as output:
            _, stream_hash = self._stream(url, output, size + 1, path.name, size)
            output.flush()
            os.fsync(output.fileno())
        digest = sha256_file(partial, self.budget.progress)
        if digest != stream_hash or (expected_hash is not None and digest != expected_hash):
            emit(
                {"phase": "verification", "label": path.name, "status": "failed"},
                self.budget.progress,
            )
            raise ArtifactError("downloaded artifact hash mismatch; partial preserved")
        if path.exists() or path.is_symlink():
            raise ArtifactError("destination appeared during download")
        # link is atomic no-replace; this inode is then made immutable to the pilot.
        # A partial name is deliberately preserved rather than auto-deleted.
        os.link(partial, path, follow_symlinks=False)
        path.chmod(0o444 if readonly else 0o600)
        emit(
            {"phase": "verification", "label": path.name, "status": "complete"},
            self.budget.progress,
        )
        return {
            "path": path.relative_to(self.budget.root).as_posix(),
            "size": size,
            "sha256": digest,
            "source": urllib.parse.urldefrag(url)[0],
            "reused": False,
        }


def parse_lock(text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for joined in text.replace("\\\n", " ").splitlines():
        line = joined.strip()
        if not line or line.startswith("#"):
            continue
        hashes = re.findall(r"--hash=sha256:([0-9a-f]{64})", line)
        body = re.sub(r"\s+--hash=sha256:[0-9a-f]{64}", "", line).strip()
        pinned = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", body)
        direct = re.fullmatch(r"([A-Za-z0-9_.-]+) @ (https://\S+)", body)
        if not hashes or (pinned is None and direct is None):
            raise ArtifactError("unsupported or unhashed requirement in approved lock")
        if pinned:
            entries.append({"name": pinned[1], "version": pinned[2], "hashes": hashes})
        elif direct:
            entries.append({"name": direct[1], "url": direct[2], "hashes": hashes})
    if not entries or len({entry["name"] for entry in entries}) != len(entries):
        raise ArtifactError("empty lock or duplicate requirement name")
    return entries


def wheel_score(filename: str) -> tuple[int, int, str] | None:
    """Select CPython 3.13 x86_64 glibc wheels; no host-dependent resolver imports."""
    if not filename.endswith(".whl"):
        return None
    pieces = filename[:-4].rsplit("-", 3)
    if len(pieces) != 4:
        return None
    _, python_tags, abi_tags, platforms = pieces
    platforms_list = platforms.split(".")
    supported_platform = any(
        tag == "any"
        or tag
        in ("manylinux1_x86_64", "manylinux2010_x86_64", "manylinux2014_x86_64", "linux_x86_64")
        or (
            re.fullmatch(r"manylinux_2_\d+_x86_64", tag) is not None
            and int(tag.split("_")[2]) <= 41
        )
        for tag in platforms_list
    )
    if not supported_platform:
        return None
    score = -1
    for py in python_tags.split("."):
        for abi in abi_tags.split("."):
            if py == "cp313" and abi in ("cp313", "abi3", "none"):
                score = max(score, 100)
            elif py.startswith("cp3") and py[3:].isdigit() and abi == "abi3":
                minor = int(py[3:])
                if 2 <= minor <= 13:
                    score = max(score, 50 + minor)
            elif py in ("py3", "py313") and abi == "none":
                score = max(score, 10)
    if score < 0:
        return None
    return score, int("any" not in platforms_list), filename


def select_wheel(metadata: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    candidates: list[tuple[tuple[int, int, str], dict[str, Any]]] = []
    for wheel in metadata.get("urls", []):
        score = wheel_score(str(wheel.get("filename", "")))
        if (
            score is not None
            and wheel.get("digests", {}).get("sha256") in entry["hashes"]
            and not wheel.get("yanked")
            and isinstance(wheel.get("size"), int)
        ):
            candidates.append((score, wheel))
    if not candidates:
        raise ArtifactError(f"no approved binary wheel for {entry['name']} on CPython 3.13 Linux")
    return max(candidates, key=lambda item: item[0])[1]


def extract_uv(wheel: Path, root: Path, budget: Budget) -> Path:
    destination = secure_path(root, "tools/uv")
    if destination.exists():
        raise ArtifactError("uv executable already exists; automatic replacement disabled")
    with zipfile.ZipFile(wheel) as archive:
        candidates = [
            item
            for item in archive.infolist()
            if item.filename == "uv/uv" or re.fullmatch(r"uv-[^/]+\.data/scripts/uv", item.filename)
        ]
        if len(candidates) != 1:
            raise ArtifactError("bootstrap wheel has no unique approved uv executable member")
        member = candidates[0]
        if member.file_size <= 0 or member.file_size > 100 << 20:
            raise ArtifactError("bootstrap executable size is outside its bound")
        budget.check(member.file_size)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with archive.open(member) as source, destination.open("xb") as output:
            copied = 0
            while chunk := source.read(1 << 20):
                copied += len(chunk)
                if copied > member.file_size:
                    raise ArtifactError("bootstrap member exceeds declared size")
                output.write(chunk)
        if copied != member.file_size:
            raise ArtifactError("incomplete bootstrap executable")
    destination.chmod(0o700)
    return destination


def run_command(command: list[str], root: Path, budget: Budget) -> None:
    """Forward child output and emit a heartbeat while a quiet installer is active."""
    env = os.environ.copy()
    for env_key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "UV_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
    ):
        env.pop(env_key, None)
    env.update(
        {
            "UV_CACHE_DIR": str(private_directory(root, "cache/uv")),
            "TMPDIR": str(private_directory(root, "tmp")),
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_OFFLINE": "1",
            "UV_NO_PROGRESS": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    emit(
        {"phase": "installation", "label": Path(command[0]).name, "status": "active"},
        budget.progress,
    )
    process = subprocess.Popen(  # noqa: S603 — only approved local executable/arguments.
        command,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if process.stdout is None:
        raise ArtifactError("installer output pipe was not created")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    start = time.monotonic()
    last = start
    try:
        while selector.get_map():
            budget.check()
            for key, _ in selector.select(timeout=1):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                else:
                    emit(
                        {
                            "phase": "installation",
                            "status": "output",
                            "text": chunk.decode("utf-8", errors="replace"),
                        },
                        budget.progress,
                    )
            if time.monotonic() - last >= 5:
                emit(
                    {
                        "phase": "installation",
                        "status": "active",
                        "elapsed_seconds": int(time.monotonic() - start),
                    },
                    budget.progress,
                )
                last = time.monotonic()
        status = process.wait(timeout=10)
        if status:
            raise ArtifactError(f"isolated installer exited {status}")
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        emit({"phase": "installation", "status": "failed"}, budget.progress)
        raise
    finally:
        selector.close()
        process.stdout.close()
    emit({"phase": "installation", "status": "complete"}, budget.progress)


def prepare(
    root: Path, contract: dict[str, Any], deadline_epoch: int, progress: Progress | None = None
) -> dict[str, Any]:
    """Prepare the approved target and editor without executing either model."""
    resources = contract["resources"]
    deadline = min(deadline_epoch, int(time.time()) + int(resources["prepare_seconds"]))
    budget = Budget(
        root,
        int(resources["disk_max_bytes"]),
        int(resources["download_max_bytes"]),
        deadline,
        progress,
    )
    emit({"phase": "preparation", "status": "active"}, progress)
    client = Downloader(budget)
    bootstrap = contract["bootstrap"]
    bootstrap_receipt = client.file(
        bootstrap["url"], "downloads/uv.whl", int(bootstrap["bytes"]), str(bootstrap["sha256"])
    )
    executable = extract_uv(root / bootstrap_receipt["path"], root, budget)
    requirements = secure_path(root, "requirements.lock")
    entries = parse_lock(requirements.read_text())
    wheels: list[dict[str, Any]] = []
    offline_lines: list[str] = []
    for index, entry in enumerate(entries):
        emit(
            {
                "phase": "dependency_selection",
                "completed": index,
                "total": len(entries),
                "unit": "packages",
                "label": entry["name"],
                "status": "active",
            },
            progress,
        )
        if "url" in entry:
            url = entry["url"]
            filename = urllib.parse.unquote(Path(urllib.parse.urlsplit(url).path).name)
            if wheel_score(filename) is None:
                raise ArtifactError("direct wheel does not match the approved Python platform")
            size = client.head_size(url)
            fragment = urllib.parse.parse_qs(urllib.parse.urlsplit(url).fragment)
            digest = fragment.get("sha256", [None])[0]
            if digest not in entry["hashes"]:
                raise ArtifactError("direct wheel hash is absent from the lock")
        else:
            metadata = client.metadata(
                f"https://pypi.org/pypi/{entry['name']}/{entry['version']}/json"
            )
            wheel = select_wheel(metadata, entry)
            filename, url, size = wheel["filename"], wheel["url"], wheel["size"]
            digest = wheel["digests"]["sha256"]
        receipt = client.file(url, Path("wheels") / filename, size, digest)
        wheels.append(receipt)
        offline_lines.append(
            f"{entry['name']} @ {(root / receipt['path']).as_uri()} --hash=sha256:{digest}"
        )
    offline_lock = secure_path(root, "requirements.offline.lock")
    if offline_lock.exists():
        raise ArtifactError("offline lock already exists")
    offline_lock.write_text("\n".join(offline_lines) + "\n")
    offline_lock.chmod(0o600)
    # Reserve wheel expansion and one installation staging copy before launching uv.
    expanded = 0
    for receipt in wheels:
        with zipfile.ZipFile(root / receipt["path"]) as archive:
            expanded += sum(info.file_size for info in archive.infolist())
    budget.check(expanded * 2)
    if (root / "venv").exists():
        raise ArtifactError("isolated environment already exists; replacement disabled")
    run_command(
        [
            str(executable),
            "--no-config",
            "venv",
            "--offline",
            "--python",
            "/usr/bin/python3",
            str(root / "venv"),
        ],
        root,
        budget,
    )
    run_command(
        [
            str(executable),
            "--no-config",
            "pip",
            "sync",
            "--offline",
            "--no-index",
            "--no-cache",
            "--python",
            str(root / "venv/bin/python"),
            "--require-hashes",
            "--only-binary",
            ":all:",
            str(offline_lock),
        ],
        root,
        budget,
    )
    originals: list[dict[str, Any]] = []
    for role in ("target", "editor"):
        model = contract[role]
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ArtifactError("model revision must be an exact commit")
        for entry in model["files"]:
            filename = entry["path"]
            url = (
                f"https://huggingface.co/{model['id']}/resolve/"
                f"{model['revision']}/{urllib.parse.quote(filename, safe='/')}"
            )
            record = client.file(
                url,
                Path("models") / role / filename,
                entry["size"],
                entry.get("sha256"),
                readonly=True,
            )
            record["role"] = role
            record["revision"] = model["revision"]
            originals.append(record)
            atomic_json(secure_path(root, "receipts/originals.json"), originals)
    verification = verify_originals(root, originals, progress)
    receipt = {
        "status": "prepared_not_model_executed",
        "bootstrap": bootstrap_receipt,
        "wheels": wheels,
        "originals": originals,
        "verification": verification,
        "network_reserved_body_bytes": budget.network_reserved_bytes,
        "network_received_body_bytes": budget.network_received_bytes,
        "requirements_sha256": sha256_file(requirements, progress),
        "offline_requirements_sha256": sha256_file(offline_lock, progress),
    }
    atomic_json(secure_path(root, "receipts/preparation.json"), receipt)
    emit({"phase": "preparation", "status": "complete"}, progress)
    return receipt
