"""Bounded, source-pinned development inputs for experimental gradient edits."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from lacunai.core.spec import TargetSpec, bounded_file
from lacunai.erase.factoid_gradient import validate_inputs


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def validate_record(data: Any, source_fingerprint: str | None = None) -> dict[str, Any]:
    fields = {"schema_version", "source_fingerprint", "forget", "retain_development", "provenance"}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError("factoid development schema fields differ")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported factoid development schema")
    pin = data["source_fingerprint"]
    if not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{64}", pin) is None:
        raise ValueError("invalid source fingerprint")
    if source_fingerprint is not None and pin != source_fingerprint:
        raise ValueError("development source fingerprint differs")
    provenance = data["provenance"]
    if not isinstance(provenance, str) or not provenance.strip() or len(provenance) > 8192:
        raise ValueError("development provenance must be bounded nonempty text")
    validate_inputs("npo-kl", 1, data["forget"], data["retain_development"])
    prompts = [
        normalized(row["prompt"]) for key in ("forget", "retain_development") for row in data[key]
    ]
    if len(set(prompts)) != len(prompts):
        raise ValueError("duplicate or cross-split development prompt")
    return data


def load_factoid_data(
    path: Path, spec: TargetSpec, source_fingerprint: str | None = None
) -> dict[str, Any]:
    data = validate_record(json.loads(bounded_file(path)), source_fingerprint)
    withheld = {normalized(row.prompt) for row in (*spec.probes, *spec.retain)}
    withheld.update(normalized(row.prompt) for row in getattr(spec, "capability_probes", []))
    withheld.update(normalized(value) for value in spec.perplexity_texts)
    if any(
        normalized(row["prompt"]) in withheld
        for key in ("forget", "retain_development")
        for row in data[key]
    ):
        raise ValueError("development prompt overlaps frozen evaluation")
    return data
