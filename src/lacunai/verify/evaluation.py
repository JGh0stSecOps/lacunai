"""Deterministic lexical recovery and collateral checks with explicit unknowns."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict
from decimal import Decimal
from typing import Any

from lacunai.core.spec import Probe, TargetSpec

_NUMBER = r"[+\-−]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[+\-]?\d+)?"


def _normal(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def classify(output: Any, answer: str, truncated: Any = False) -> dict[str, Any]:
    """Presence means lexical recovery, including quoted/negated target strings."""
    if not isinstance(output, str) or not output.strip():
        return {"classification": "missing", "reason": "no_nonempty_output"}
    normalized, expected = _normal(output), _normal(answer)
    number = _NUMBER
    if re.fullmatch(number, expected):
        # Consume complete numeric tokens: signs, grouped thousands and exponents
        # cannot leave a misleading matching substring behind. Decimal equality also
        # recognizes equivalent representations rather than mistaking 3.0 for erasure.
        expected_number = Decimal(expected.replace(",", "").replace("−", "-").replace(" ", ""))
        matched = any(
            Decimal(token.group().replace(",", "").replace("−", "-").replace(" ", ""))
            == expected_number
            for token in re.finditer(r"(?<![\w.,+\-−])" + number + r"(?!\w|[.,]\d)", normalized)
        )
    else:
        matched = re.search(r"(?<!\w)" + re.escape(expected) + r"(?!\w)", normalized) is not None
    if type(truncated) is not bool or truncated:
        return {
            "classification": "ambiguous",
            "reason": "truncated_or_unknown_completion",
            "observed_target_string": matched,
        }
    return {
        "classification": "recovered" if matched else "not_recovered",
        "reason": "bounded_target_string_present" if matched else "target_string_not_observed",
    }


def classify_retain(output: Any, answer: str, truncated: Any = False) -> dict[str, Any]:
    """Whole-answer correctness, separate from conservative target-string recovery."""
    observation = classify(output, answer, truncated)
    if observation["classification"] in {"missing", "ambiguous"}:
        return observation

    def clean(value: str) -> str:
        return _normal(value).strip("\"'“”‘’").rstrip(".!?").strip("\"'“”‘’").strip()

    candidate, expected = clean(output), clean(answer)
    if re.fullmatch(_NUMBER, candidate) and re.fullmatch(_NUMBER, expected):

        def decimal(value: str) -> Decimal:
            return Decimal(value.replace(",", "").replace("−", "-").replace(" ", ""))

        correct = decimal(candidate) == decimal(expected)
    else:
        correct = candidate == expected
    return {
        "classification": "recovered" if correct else "not_recovered",
        "reason": "whole_answer_correct" if correct else "whole_answer_incorrect",
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        name: sum(row["classification"] == name for row in rows)
        for name in ("recovered", "not_recovered", "ambiguous", "missing")
    }
    return {
        **counts,
        "correct": counts["recovered"],
        "total": len(rows),
        "recovery_rate": str(Decimal(counts["recovered"]) / Decimal(len(rows))) if rows else None,
    }


def _suite(runtime: Any, probes: list[Probe], split: str) -> dict[str, Any]:
    if not probes:
        raise ValueError("empty evaluation suite refused")
    rows = []
    runtime.progress("evaluation", split=split, completed=0, total=len(probes), unit="probes")
    for index, probe in enumerate(probes):
        runtime.check()
        generation = runtime.generate(probe.prompt)
        row = {"index": index, **asdict(probe), "generation": generation}
        scorer = classify_retain if split == "retain" else classify
        row.update(scorer(generation.get("output"), probe.answer, generation.get("truncated")))
        ids = generation.get("token_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(i) is not int or i < 0 for i in ids)
            or type(generation.get("input_tokens")) is not int
            or generation["input_tokens"] <= 0
            or type(generation.get("output_tokens")) is not int
            or len(ids) != generation["output_tokens"]
        ):
            row.update(classification="missing", reason="invalid_generation_accounting")
        rows.append(row)
        runtime.progress(
            "evaluation", split=split, completed=index + 1, total=len(probes), unit="probes"
        )
    return {
        **_summarize(rows),
        "metric": "whole_answer_correctness" if split == "retain" else "target_string_recovery",
        "rows": rows,
        "by_kind": {
            kind: _summarize([r for r in rows if r["kind"] == kind])
            for kind in sorted({p.kind for p in probes})
        },
    }


def _perplexity(value: dict[str, Any]) -> dict[str, Any]:
    rows = value.get("examples", [])
    if not rows:
        raise ValueError("perplexity has no per-text evidence")
    loss, count = Decimal(0), 0
    for row in rows:
        if type(row.get("tokens")) is not int or row["tokens"] <= 0:
            raise ValueError("invalid perplexity token count")
        nll = Decimal(str(row["nll_sum"]))
        if not nll.is_finite() or nll < 0:
            raise ValueError("invalid perplexity loss")
        loss += nll
        count += row["tokens"]
    mean = loss / Decimal(count)
    if mean > 80:
        raise ValueError("perplexity overflow")
    # Tensor runtime supplies NLL; scalar reporting recomputes the weighted total.
    return {
        "nll_sum": str(loss),
        "tokens": count,
        "mean_nll": str(mean),
        "perplexity": str(mean.exp()),
        "examples": rows,
        "method": "exp(sum(token NLL)/sum(valid tokens)); independent bounded texts",
    }


def evaluate(runtime: Any, spec: TargetSpec) -> dict[str, Any]:
    if spec.schema_version == 2:
        from .connected import evaluate_connected

        return evaluate_connected(runtime, spec)  # type: ignore[arg-type]
    target = _suite(runtime, spec.probes, "target")
    retain = _suite(runtime, spec.retain, "retain")
    ppl = _perplexity(runtime.perplexity(spec.perplexity_texts))
    return {
        "schema_version": 1,
        "target_id": spec.target_id,
        "spec_sha256": spec.digest,
        "checkpoint_fingerprint": runtime.fingerprint,
        "target": target,
        "retain": retain,
        "perplexity": ppl,
        "scorer": "NFKC/casefold target recovery and whole-answer retain correctness v2",
        "limits": "Lexical recoverability only; not semantic equivalence or information absence.",
    }


def _validated_suite(
    value: dict[str, Any], probes: list[Probe], *, retain: bool = False
) -> dict[str, Any]:
    rows = value.get("rows", [])
    if len(rows) != len(probes):
        raise ValueError("missing evaluation observations")
    seen: set[int] = set()
    checked = []
    for row, probe in zip(rows, probes, strict=True):
        if any(row.get(key) != expected for key, expected in asdict(probe).items()):
            raise ValueError("evaluation prompt/answer differs from frozen specification")
        index = row.get("index")
        if type(index) is not int or index in seen or index != len(seen):
            raise ValueError("duplicate or reordered evaluation observation")
        seen.add(index)
        generation = row.get("generation", {})
        scorer = classify_retain if retain else classify
        calculated = scorer(generation.get("output"), probe.answer, generation.get("truncated"))
        ids = generation.get("token_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(i) is not int or i < 0 for i in ids)
            or type(generation.get("input_tokens")) is not int
            or generation["input_tokens"] <= 0
            or type(generation.get("output_tokens")) is not int
            or len(ids) != generation["output_tokens"]
        ):
            calculated = {"classification": "missing", "reason": "invalid_generation_accounting"}
        if any(row.get(key) != val for key, val in calculated.items()):
            raise ValueError("stored classifications differ from raw output")
        checked.append(row)
    return {
        **_summarize(checked),
        "metric": "whole_answer_correctness" if retain else "target_string_recovery",
        "rows": checked,
        "by_kind": {
            kind: _summarize([r for r in checked if r["kind"] == kind])
            for kind in sorted({p.kind for p in probes})
        },
    }


def _fingerprint_change(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> dict[str, Any]:
    if before is None or after is None:
        return {"status": "unavailable", "absence_proven": False}
    result: dict[str, Any] = {
        "status": "observed",
        "layers": {},
        "absence_proven": False,
        "interpretation": "Activation association only; readout edits need not silence features.",
    }
    for layer in sorted(set(before.get("layers", {})) | set(after.get("layers", {}))):
        left, right = before.get("layers", {}).get(layer), after.get("layers", {}).get(layer)
        if left is None or right is None:
            result["layers"][layer] = {"status": "missing"}
            continue
        x, y = left.get("target_mean"), right.get("target_mean")
        if not isinstance(x, list) or not isinstance(y, list) or len(x) != len(y):
            result["layers"][layer] = {"status": "missing_magnitudes"}
            continue
        if any(not math.isfinite(float(v)) for v in x + y):
            raise ValueError("nonfinite fingerprint magnitude")
        active_left = {i for i, value in enumerate(x) if abs(float(value)) > 1e-8}
        active_right = {i for i, value in enumerate(y) if abs(float(value)) > 1e-8}
        union = active_left | active_right
        result["layers"][layer] = {
            "before_active": len(active_left),
            "after_active": len(active_right),
            "active_overlap": str(Decimal(len(active_left & active_right)) / Decimal(len(union)))
            if union
            else None,
            "mean_absolute_before": str(
                sum((Decimal(str(abs(float(v)))) for v in x), Decimal(0)) / Decimal(len(x))
            )
            if x
            else None,
            "mean_absolute_after": str(
                sum((Decimal(str(abs(float(v)))) for v in y), Decimal(0)) / Decimal(len(y))
            )
            if y
            else None,
            "zero_tolerance": "1e-8",
        }
    return result


def compare(
    before: dict[str, Any],
    after: dict[str, Any],
    spec: TargetSpec,
    before_fp: dict[str, Any] | None = None,
    after_fp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if spec.schema_version == 2:
        from .connected import compare_connected

        return compare_connected(before, after, spec, before_fp, after_fp)  # type: ignore[arg-type]
    result: dict[str, Any] = {
        "verdict": "inconclusive",
        "reasons": [],
        "spec_sha256": spec.digest,
        "thresholds": asdict(spec.thresholds),
        "absence_proven": False,
        "claim": "reduced recoverability under these tests, if the contract passes",
    }
    try:
        if any(
            v.get("spec_sha256") != spec.digest
            or v.get("target_id") != spec.target_id
            or not v.get("checkpoint_fingerprint")
            for v in (before, after)
        ):
            raise ValueError("evaluation identity differs from frozen target")
        left = {
            "target": _validated_suite(before["target"], spec.probes),
            "retain": _validated_suite(before["retain"], spec.retain, retain=True),
        }
        right = {
            "target": _validated_suite(after["target"], spec.probes),
            "retain": _validated_suite(after["retain"], spec.retain, retain=True),
        }
        bp, ap = _perplexity(before["perplexity"]), _perplexity(after["perplexity"])
        for value in (bp, ap):
            if [r.get("text") for r in value["examples"]] != spec.perplexity_texts:
                raise ValueError("perplexity corpus differs from frozen target")
        if bp["tokens"] != ap["tokens"]:
            raise ValueError("perplexity tokenization differs")
        result.update(before=left, after=right, perplexity_before=bp, perplexity_after=ap)
        result["fingerprints"] = _fingerprint_change(before_fp, after_fp)
        result["paired_retain"] = {
            "losses": [
                b["index"]
                for b, a in zip(left["retain"]["rows"], right["retain"]["rows"], strict=True)
                if b["classification"] == "recovered" and a["classification"] == "not_recovered"
            ],
            "gains": [
                b["index"]
                for b, a in zip(left["retain"]["rows"], right["retain"]["rows"], strict=True)
                if b["classification"] == "not_recovered" and a["classification"] == "recovered"
            ],
        }
        ppl_ratio = Decimal(ap["perplexity"]) / Decimal(bp["perplexity"])
        retain_drop = Decimal(left["retain"]["recovery_rate"]) - Decimal(
            right["retain"]["recovery_rate"]
        )
        result.update(perplexity_ratio=str(ppl_ratio), retain_accuracy_drop=str(retain_drop))
        if any(s["ambiguous"] or s["missing"] for group in (left, right) for s in group.values()):
            result["reasons"].append("ambiguous, truncated or missing observations")
        for kind, row in left["target"]["by_kind"].items():
            if Decimal(row["recovery_rate"]) < Decimal(
                str(spec.thresholds.minimum_baseline_recovery)
            ):
                result["reasons"].append(f"baseline {kind} recovery did not qualify")
        if Decimal(left["retain"]["recovery_rate"]) < Decimal(
            str(spec.thresholds.minimum_baseline_retain_accuracy)
        ):
            result["reasons"].append("baseline retain answer accuracy did not qualify")
        failures = []
        if ppl_ratio > Decimal(str(spec.thresholds.maximum_perplexity_ratio)):
            failures.append("perplexity collateral threshold exceeded")
        if not any(
            s["ambiguous"] or s["missing"] for s in (left["retain"], right["retain"])
        ) and retain_drop > Decimal(str(spec.thresholds.maximum_retain_drop)):
            failures.append("retain accuracy collateral threshold exceeded")
        if len(result["paired_retain"]["losses"]) > spec.thresholds.maximum_retain_paired_losses:
            failures.append("paired retain loss threshold exceeded")
        recovery_failures = []
        for kind, row in right["target"]["by_kind"].items():
            if Decimal(row["recovery_rate"]) > Decimal(str(spec.thresholds.maximum_after_recovery)):
                recovery_failures.append(f"target remains recoverable through {kind} probes")
        if failures:
            result["reasons"].extend(failures + recovery_failures)
            result["verdict"] = "failed"
        elif recovery_failures:
            if not result["reasons"]:
                result["verdict"] = "failed"
            result["reasons"].extend(recovery_failures)
        elif not result["reasons"]:
            result["verdict"] = "passed_the_stated_experimental_contract"
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        result["reasons"].append(str(exc))
    return result
