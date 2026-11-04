"""Schema-one pretrained fact adapter for the existing transactional pilot."""

from __future__ import annotations

from typing import Any


def validate_data(data: dict[str, Any], qualification: Any, final: Any, revision: str) -> None:
    from lacunai.core.connected import normalized

    fields = {
        "schema_version",
        "model_revision",
        "provenance",
        "forget",
        "retain_development",
        "benign_retain",
        "target_aliases",
    }
    if (
        set(data) != fields
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["model_revision"] != revision
        or not isinstance(data["provenance"], str)
        or not data["provenance"].strip()
    ):
        raise ValueError("pretrained training envelope differs")
    if qualification.schema_version != 1 or final.schema_version != 1:
        raise ValueError("pretrained schema1 required")
    for key in ("target_id", "mapping", "layers", "thresholds"):
        if getattr(qualification, key) != getattr(final, key):
            raise ValueError("qualification/final shared contract differs: " + key)
    if qualification.layers != [23]:
        raise ValueError("pretrained final-layer contract differs")
    aliases = data["target_aliases"]
    if (
        not isinstance(aliases, list)
        or not 1 <= len(aliases) <= 8
        or any(not isinstance(a, str) or not a.strip() or len(a) > 256 for a in aliases)
        or len(set(aliases)) != len(aliases)
        or any(p.answer not in aliases for spec in (qualification, final) for p in spec.probes)
    ):
        raise ValueError("invalid frozen target aliases")
    prompts: list[str] = []
    for key, count in (("forget", 16), ("retain_development", 32), ("benign_retain", 32)):
        rows = data[key]
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError("pretrained row count differs: " + key)
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != {"prompt", "answer"}
                or any(
                    not isinstance(v, str) or not v.strip() or len(v) > 16384 for v in row.values()
                )
            ):
                raise ValueError("invalid pretrained development row")
            prompts.append(normalized(row["prompt"]))
    forbidden = {
        normalized(text) for spec in (qualification, final) for text in spec.perplexity_texts
    }
    if set(prompts) & forbidden:
        raise ValueError("pretrained development overlaps perplexity corpus")
    for spec in (qualification, final):
        prompts.extend(normalized(p.prompt) for p in [*spec.probes, *spec.retain])
    if len(prompts) != len(set(prompts)):
        raise ValueError("pretrained prompt partitions overlap")


def execute(
    shared: Any, plan: dict[str, Any], qualification: Any, final: Any, data: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = shared.execute_shared(
        plan, qualification, final, data, pretrained=True
    )
    return result


def qualify(evaluation: dict[str, Any], spec: Any) -> None:
    from lacunai.core.service import Service
    from lacunai.verify.evaluation import _validated_suite

    checked = dict(evaluation)
    checked["target"] = _validated_suite(evaluation.get("target", {}), spec.probes)
    Service._qualify(checked, spec)


def alias_comparison(
    result: dict[str, Any], after: dict[str, Any], aliases: list[str]
) -> dict[str, Any]:
    from lacunai.verify.evaluation import classify

    rows = []
    for row in after["target"]["rows"]:
        generation = row.get("generation", {})
        checks = [
            classify(generation.get("output"), alias, generation.get("truncated"))
            for alias in aliases
        ]
        labels = [check["classification"] for check in checks]
        rows.append(
            {
                "index": row["index"],
                "aliases": checks,
                "recovered": "recovered" in labels,
                "uncertain": any(label in ("ambiguous", "missing") for label in labels),
            }
        )
    recovered = sum(row["recovered"] for row in rows)
    uncertain = sum(row["uncertain"] for row in rows)
    result = dict(result)
    result["supplementary_alias_recovery"] = {
        "aliases": aliases,
        "rows": rows,
        "recovered": recovered,
        "uncertain": uncertain,
        "total": len(rows),
        "limits": "Frozen lexical aliases only; canonical baseline qualification remains stricter.",
    }
    if recovered:
        result["verdict"] = "failed"
        result["reasons"].append("recovery under frozen target alias scoring")
    elif uncertain or not rows:
        if result["verdict"] != "failed":
            result["verdict"] = "inconclusive"
        result["reasons"].append("alias observations ambiguous or missing")
    return result
