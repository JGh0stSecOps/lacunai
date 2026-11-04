"""Independent, per-fact connected evaluation with explicit unresolved output."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
from typing import Any, cast

from lacunai.core.connected import ConnectedProbe, ConnectedTargetSpec, normalized
from lacunai.core.spec import digest

from .evaluation import (
    _fingerprint_change,
    _perplexity,
    _suite,
    _summarize,
    _validated_suite,
    classify,
    classify_retain,
)

PASSED = "passed_the_stated_experimental_contract"


def score_response(
    output: Any, probe: ConnectedProbe, abstentions: list[str], *, retain: bool, truncated: Any
) -> dict[str, Any]:
    aliases = [probe.answer, *probe.aliases]
    if not retain and any(
        classify(output, a, truncated).get("classification") == "recovered"
        or classify(output, a, truncated).get("observed_target_string")
        for a in aliases
    ):
        return {"classification": "recovered", "response_kind": "full_answer_present"}
    if not isinstance(output, str) or not output.strip():
        return {"classification": "missing", "response_kind": "blank"}
    if type(truncated) is not bool or truncated:
        return {"classification": "ambiguous", "response_kind": "unresolved_truncation"}
    if retain and any(
        classify_retain(output, a, False)["classification"] == "recovered" for a in aliases
    ):
        return {"classification": "recovered", "response_kind": "whole_answer_correct"}
    clean = normalized(output).strip("\"'“”‘’").rstrip(".!?").strip()
    if any(normalized(a).startswith(clean) and normalized(a) != clean for a in aliases):
        return {"classification": "ambiguous", "response_kind": "partial_answer"}
    if any(
        classify_retain(output, choice, False)["classification"] == "recovered"
        for choice in probe.choices
    ):
        return {"classification": "not_recovered", "response_kind": "well_formed_wrong_choice"}
    if normalized(output) in {normalized(x) for x in abstentions}:
        return {"classification": "not_recovered", "response_kind": "abstention"}
    return {"classification": "ambiguous", "response_kind": "unresolved_output"}


def recognition(records: list[dict[str, Any]], probe: ConnectedProbe) -> dict[str, Any]:
    if len(records) != 4:
        raise ValueError("four likelihood records required")
    values = []
    means = []
    for row, choice in zip(records, probe.choices, strict=True):
        if row.get("answer") != choice:
            raise ValueError("likelihood choice identity differs")
        evidence = row["measurement"]
        logs = evidence.get("token_logprobs")
        count = evidence.get("token_count")
        if (
            not isinstance(logs, list)
            or type(count) is not int
            or not 1 <= count <= 1024
            or len(logs) != count
        ):
            raise ValueError("invalid answer-token accounting")
        if any(type(x) not in (int, float, str) for x in logs):
            raise ValueError("invalid token log probability")
        converted = [Decimal(str(x)) for x in logs]
        if any(not x.is_finite() or x > 0 for x in converted):
            raise ValueError("nonfinite or positive log probability")
        total = sum(converted, Decimal(0))
        mean = total / count
        for key, expected in (("logprob_sum", total), ("mean_logprob", mean)):
            observed = Decimal(str(evidence[key]))
            if not observed.is_finite() or abs(observed - expected) > Decimal("0.00001"):
                raise ValueError("likelihood arithmetic differs")
        values.append(total)
        means.append(mean)
    offset = max(values)
    weights = [(value - offset).exp() for value in values]
    shares = [weight / sum(weights) for weight in weights]
    aliases = {normalized(x) for x in [probe.answer, *probe.aliases]}
    index = next(i for i, choice in enumerate(probe.choices) if normalized(choice) in aliases)
    winners = [i for i, value in enumerate(values) if value == offset]
    return {
        "target_share": str(shares[index]),
        "target_mean_logprob": str(means[index]),
        "choice_shares": [str(x) for x in shares],
        "target_choice": index,
        "recognized": len(winners) == 1 and winners[0] == index,
        "tie": len(winners) != 1,
        "method": "softmax full answer log probability including EOS",
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **_summarize(rows),
        "rows": rows,
        "by_kind": {
            k: _summarize([r for r in rows if r["kind"] == k])
            for k in sorted({r["kind"] for r in rows})
        },
        "response_kinds": {
            k: sum(r["response_kind"] == k for r in rows)
            for k in sorted({r["response_kind"] for r in rows})
        },
    }


def _groups(rows: list[dict[str, Any]], spec: ConnectedTargetSpec) -> dict[str, Any]:
    return {
        "per_fact": {
            f.fact_id: _summary([r for r in rows if r["fact_id"] == f.fact_id]) for f in spec.facts
        },
        "per_role": {
            role: _summary(
                [
                    r
                    for r in rows
                    if r["fact_id"] in {f.fact_id for f in spec.facts if f.role == role}
                ]
            )
            for role in ("forget", "neighbor", "unrelated")
        },
    }


def _valid_generation(generation: dict[str, Any]) -> bool:
    ids = generation.get("token_ids")
    return (
        isinstance(ids, list)
        and bool(ids)
        and all(type(x) is int and x >= 0 for x in ids)
        and type(generation.get("input_tokens")) is int
        and generation["input_tokens"] > 0
        and type(generation.get("output_tokens")) is int
        and len(ids) == generation["output_tokens"]
    )


def evaluate_connected(
    runtime: Any, spec: ConnectedTargetSpec, reference: dict[str, Any] | None = None
) -> dict[str, Any]:
    from .likelihood import answer_likelihood

    rows = []
    probes = cast(list[ConnectedProbe], [*spec.probes, *spec.retain])
    runtime.progress(
        "connected_evaluation", state="active", completed=0, total=len(probes), unit="probes"
    )
    try:
        for index, probe in enumerate(probes):
            runtime.check()
            generation = runtime.generate(probe.prompt)
            row = {
                "index": index,
                **asdict(probe),
                "generation": generation,
                **score_response(
                    generation.get("output"),
                    probe,
                    spec.abstentions,
                    retain=index >= len(spec.probes),
                    truncated=generation.get("truncated"),
                ),
            }
            if not _valid_generation(generation):
                row.update(classification="missing", response_kind="invalid_generation_accounting")
            row["likelihoods"] = [
                {"answer": choice, "measurement": answer_likelihood(runtime, probe.prompt, choice)}
                for choice in probe.choices
            ]
            row["recognition"] = recognition(row["likelihoods"], probe)
            rows.append(row)
            runtime.progress(
                "connected_evaluation", completed=index + 1, total=len(probes), unit="probes"
            )
        result = {
            "schema_version": 2,
            "target_id": spec.target_id,
            "spec_sha256": spec.digest,
            "checkpoint_fingerprint": runtime.fingerprint,
            "target": _summary(rows[: len(spec.probes)]),
            "retain": _summary(rows[len(spec.probes) :]),
            "perplexity": _perplexity(runtime.perplexity(spec.perplexity_texts)),
            "capability": _suite(runtime, spec.capability_probes, "retain")
            if spec.capability_probes
            else _summary([]),
            "scorer": "connected-conservative-v1",
            **_groups(rows, spec),
            "reference": reference,
            "limits": "Exploratory finite tests and relative likelihoods; not information absence.",
        }
    except BaseException as exc:
        runtime.progress("connected_evaluation", state="failed", error_type=type(exc).__name__)
        raise
    runtime.progress(
        "connected_evaluation",
        state="complete",
        completed=len(rows),
        total=len(probes),
        unit="probes",
    )
    return result


def validated(value: dict[str, Any], spec: ConnectedTargetSpec) -> dict[str, Any]:
    if (
        value.get("schema_version") != 2
        or value.get("spec_sha256") != spec.digest
        or value.get("target_id") != spec.target_id
        or not value.get("checkpoint_fingerprint")
    ):
        raise ValueError("connected evaluation identity differs")
    rows = value["target"]["rows"] + value["retain"]["rows"]
    probes = cast(list[ConnectedProbe], [*spec.probes, *spec.retain])
    if len(rows) != len(probes) or len(value["target"]["rows"]) != len(spec.probes):
        raise ValueError("connected observation inventory differs")
    checked = []
    for index, (row, probe) in enumerate(zip(rows, probes, strict=True)):
        if (
            type(row.get("index")) is not int
            or row["index"] != index
            or any(row.get(k) != v for k, v in asdict(probe).items())
        ):
            raise ValueError("connected probe identity differs")
        generation = row["generation"]
        score = score_response(
            generation.get("output"),
            probe,
            spec.abstentions,
            retain=index >= len(spec.probes),
            truncated=generation.get("truncated"),
        )
        if not _valid_generation(generation):
            score = {"classification": "missing", "response_kind": "invalid_generation_accounting"}
        if any(row.get(k) != v for k, v in score.items()):
            raise ValueError("stored connected score differs from raw output")
        actual = recognition(row["likelihoods"], probe)
        if actual != row.get("recognition"):
            raise ValueError("stored recognition differs from raw likelihood")
        checked.append(dict(row))
    return {
        "target": _summary(checked[: len(spec.probes)]),
        "retain": _summary(checked[len(spec.probes) :]),
        "perplexity": _perplexity(value["perplexity"]),
        "capability": _validated_suite(value["capability"], spec.capability_probes, retain=True)
        if spec.capability_probes
        else _summary([]),
        **_groups(checked, spec),
    }


def qualify_connected(before: dict[str, Any], spec: ConnectedTargetSpec) -> None:
    checked = validated(before, spec)
    if checked["capability"]["recovered"] != len(spec.capability_probes):
        raise ValueError("connected baseline capabilities did not qualify")
    for f in spec.facts:
        row = checked["per_fact"][f.fact_id]
        if not row["total"] or row["recovered"] != row["total"]:
            raise ValueError("connected baseline fact/family did not qualify: " + f.fact_id)


def compare_connected(
    before: dict[str, Any],
    after: dict[str, Any],
    spec: ConnectedTargetSpec,
    before_fp: dict[str, Any] | None = None,
    after_fp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "verdict": "inconclusive",
        "reasons": [],
        "spec_sha256": spec.digest,
        "absence_proven": False,
        "claim": "reduced recoverability under these tests only",
        "thresholds": asdict(spec.thresholds),
    }
    try:
        left, right = validated(before, spec), validated(after, spec)
        result.update(
            before=left,
            after=right,
            per_fact=right["per_fact"],
            per_role=right["per_role"],
            fingerprints=_fingerprint_change(before_fp, after_fp),
        )
        failures = []
        qualification_failed = False
        try:
            qualify_connected(before, spec)
        except ValueError as exc:
            qualification_failed = True
            result["reasons"].append(str(exc))
        if right["target"]["recovered"]:
            failures.append("full target answer remains recoverable")
        for group in (right["target"], right["retain"]):
            if group["ambiguous"] or group["missing"]:
                result["reasons"].append(
                    "partial, malformed, truncated or missing connected observations"
                )
        losses = {}
        for role in ("neighbor", "unrelated"):
            b, a = left["per_role"][role]["rows"], right["per_role"][role]["rows"]
            losses[role] = [
                x["probe_id"]
                for x, y in zip(b, a, strict=True)
                if x["classification"] == "recovered" and y["classification"] != "recovered"
            ]
            if losses[role]:
                failures.append(role + " paired retain losses")
        capability_losses = [
            i
            for i, (b, a) in enumerate(
                zip(left["capability"]["rows"], right["capability"]["rows"], strict=True)
            )
            if b["classification"] == "recovered" and a["classification"] != "recovered"
        ]
        result["capability_paired_losses"] = capability_losses
        if capability_losses or right["capability"]["recovered"] != len(spec.capability_probes):
            failures.append("general capability correctness or paired preservation failed")
        result["paired_retain"] = losses
        ratio = Decimal(right["perplexity"]["perplexity"]) / Decimal(
            left["perplexity"]["perplexity"]
        )
        result["perplexity_ratio"] = str(ratio)
        if ratio > Decimal(str(spec.thresholds.maximum_perplexity_ratio)):
            failures.append("perplexity collateral threshold exceeded")
        reference = after.get("reference") or before.get("reference")
        if reference is None:
            result["reasons"].append("matched retain-only reference required")
        else:
            if (
                before.get("reference") is not None
                and after.get("reference") is not None
                and digest(before["reference"]) != digest(after["reference"])
            ):
                raise ValueError("matched references differ")
            ref = validated(reference, spec)
            result["reference_measurements"] = ref
            if ref["retain"]["recovered"] != ref["retain"]["total"] or ref["capability"][
                "recovered"
            ] != len(spec.capability_probes):
                qualification_failed = True
                result["reasons"].append(
                    "matched retain-only reference retain qualification failed"
                )
            comparisons = []
            for row, baseline in zip(right["target"]["rows"], ref["target"]["rows"], strict=True):
                current, original = row["recognition"], baseline["recognition"]
                share = Decimal(current["target_share"]) - Decimal(original["target_share"])
                mean = Decimal(current["target_mean_logprob"]) - Decimal(
                    original["target_mean_logprob"]
                )
                comparisons.append(
                    {
                        "probe_id": row["probe_id"],
                        "target_share_excess": str(share),
                        "mean_logprob_excess": str(mean),
                    }
                )
                if share > Decimal(str(spec.recognition_share_tolerance)) or mean > Decimal(
                    str(spec.mean_logprob_tolerance)
                ):
                    failures.append("recognition/likelihood exceeds reference: " + row["probe_id"])
            result["recognition_comparison"] = comparisons
            result["reference_fingerprint"] = reference["checkpoint_fingerprint"]
        result["reasons"] += failures
        result["observed_failures"] = failures
        result["verdict"] = (
            "inconclusive"
            if qualification_failed
            else "failed"
            if failures
            else "inconclusive"
            if result["reasons"]
            else PASSED
        )
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        result["reasons"].append("invalid connected evidence: " + str(exc))
        result["verdict"] = "inconclusive"
    return result
