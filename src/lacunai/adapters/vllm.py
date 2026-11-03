"""Prepare an offline serving command; never start or reload a server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def prepare_launch(
    run_dir: Path, trusted_public_key: Path, checkpoint: Path, port: int = 8000
) -> dict[str, Any]:
    from lacunai.core.model import checkpoint_inventory
    from lacunai.core.spec import bounded_file, digest
    from lacunai.verify.reports import verify_report

    if isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    verification = verify_report(run_dir, trusted_public_key)
    record = json.loads(bounded_file(run_dir / "report.json", 32 * 1024**2))
    if (
        record.get("status") != "complete"
        or record.get("verdict") != "passed_the_stated_experimental_contract"
        or record.get("dry_run")
    ):
        raise ValueError("serving requires a signed passed experimental contract")
    checkpoint = checkpoint.absolute()
    if checkpoint != Path(record.get("output", "")).absolute():
        raise ValueError("checkpoint path differs from signed output")
    inventory = checkpoint_inventory(checkpoint, 4 * 1024**3)
    if not record.get("output_fingerprint") or digest(inventory) != record["output_fingerprint"]:
        raise ValueError("checkpoint fingerprint differs from signed output")
    return {
        "argv": [
            "vllm",
            "serve",
            str(checkpoint),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--load-format",
            "safetensors",
        ],
        "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
        "version_requirement": (
            "vLLM with the documented serve CLI and safetensors load format; "
            "operator must verify installed version compatibility"
        ),
        "live_serving_verified": False,
        "started": False,
        "report_verification": verification,
    }
