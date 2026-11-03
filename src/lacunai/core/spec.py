"""Versioned target contracts; evaluation is frozen before any edit."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def bounded_file(path: Path, limit: int = 8 * 1024 * 1024) -> bytes:
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("input must be an unredirected regular file")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("input exceeds byte limit")
    return data


def text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 8192:
        raise ValueError(f"{name} must be nonempty bounded text")
    return value


@dataclass(frozen=True)
class Probe:
    prompt: str
    answer: str
    kind: str = "direct"

    def __post_init__(self) -> None:
        text(self.prompt, "prompt")
        text(self.answer, "answer")
        if self.kind not in {"direct", "paraphrase", "multi_hop", "adversarial", "retain"}:
            raise ValueError("unknown probe kind")


@dataclass(frozen=True)
class MappingProbe:
    prompt: str
    corrupt_prompt: str
    answer: str

    def __post_init__(self) -> None:
        for name in ("prompt", "corrupt_prompt", "answer"):
            text(getattr(self, name), name)
        if self.prompt == self.corrupt_prompt:
            raise ValueError("corruption must differ from the clean prompt")


@dataclass(frozen=True)
class Thresholds:
    minimum_baseline_recovery: float = 1.0
    minimum_baseline_retain_accuracy: float = 0.75
    maximum_after_recovery: float = 0.0
    maximum_retain_drop: float = 0.0
    maximum_perplexity_ratio: float = 1.15
    maximum_retain_paired_losses: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"invalid threshold: {name}")
            if name == "maximum_retain_paired_losses":
                if type(value) is not int or not 0 <= value <= 256:
                    raise ValueError("paired-loss limit must be a bounded integer")
            elif name == "maximum_perplexity_ratio":
                if not 1 <= value <= 2:
                    raise ValueError("perplexity ratio must be between one and two")
            elif not 0 <= value <= 1:
                raise ValueError(f"threshold outside [0,1]: {name}")


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    description: str
    mapping: list[MappingProbe]
    probes: list[Probe]
    retain: list[Probe]
    background: list[str]
    perplexity_texts: list[str]
    layers: list[int] = field(default_factory=list)
    top_n: int = 8
    patch_neurons: int = 8
    rewrite: str = "unknown"
    thresholds: Thresholds = field(default_factory=Thresholds)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", self.target_id):
            raise ValueError("target_id must be a bounded safe identifier")
        text(self.description, "description")
        text(self.rewrite, "rewrite")
        if self.schema_version != 1:
            raise ValueError("unsupported target schema")
        for name in ("mapping", "probes", "retain", "background", "perplexity_texts"):
            value = getattr(self, name)
            if not isinstance(value, list) or not 1 <= len(value) <= 256:
                raise ValueError(f"{name} must contain between one and 256 records")
        for value in self.background + self.perplexity_texts:
            text(value, "corpus text")
        if not {"direct", "paraphrase"} <= {p.kind for p in self.probes}:
            raise ValueError("withheld evaluation needs direct and paraphrased probes")
        sets = [
            {" ".join(p.prompt.split()).casefold() for p in self.mapping},
            {" ".join(p.prompt.split()).casefold() for p in self.probes},
            {" ".join(p.prompt.split()).casefold() for p in self.retain},
        ]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("mapping, withheld and retain prompts must be disjoint")
        if any(type(i) is not int or not 0 <= i < 256 for i in self.layers):
            raise ValueError("invalid layer identifier")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("duplicate layers")
        for value in (self.top_n, self.patch_neurons):
            if type(value) is not int or not 1 <= value <= 128:
                raise ValueError("neuron count must be within [1,128]")

    @property
    def digest(self) -> str:
        return digest(asdict(self))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TargetSpec:
        value = dict(value)
        for name, klass in (("mapping", MappingProbe), ("probes", Probe), ("retain", Probe)):
            value[name] = [klass(**item) for item in value[name]]
        value["thresholds"] = Thresholds(**value.get("thresholds", {}))
        return cls(**value)

    @classmethod
    def from_path(cls, path: Path) -> TargetSpec:
        return cls.from_dict(json.loads(bounded_file(path)))


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    torch_threads: int = 2
    max_prompt_tokens: int = 256
    max_new_tokens: int = 24
    max_seconds: int = 3600
    max_model_bytes: int = 4 * 1024**3
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        bounds = {
            "seed": (0, 2**32 - 1),
            "torch_threads": (1, 8),
            "max_prompt_tokens": (16, 1024),
            "max_new_tokens": (1, 128),
            "max_seconds": (1, 21600),
            "max_model_bytes": (1, 16 * 1024**3),
        }
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"invalid runtime bound: {key}")
        if self.device != "cpu" or self.dtype != "float32":
            raise ValueError("this runtime supports declared CPU float32 execution only")
