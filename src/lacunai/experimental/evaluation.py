# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Independent finite-contract scoring, with ambiguity and missing data visible.

No evaluator asks an LLM whether its own edit worked. Unknown plausible wording
is not silently counted as forgetting. Every metric is computed from raw records.
"""

from __future__ import annotations

import math
import re
import unicodedata
from decimal import Decimal
from typing import Any

Item = dict[str, Any]


def _token_list(value: Any, *, nonempty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(type(token) is int and token >= 0 for token in value)
    )


def validate_observations(items: list[Item], records: list[dict[str, Any]]) -> None:
    """Bind complete raw generation records to the frozen item/prompt identities."""
    by_id = {item["id"]: item for item in items}
    if not by_id or len(by_id) != len(items):
        raise ValueError("empty or duplicate frozen observation item set")
    expected = {(item["id"], variant) for item in items for variant in range(len(item["prompts"]))}
    seen: set[tuple[str, int]] = set()
    for record in records:
        identifier, variant = record.get("item_id"), record.get("variant")
        if not isinstance(identifier, str) or type(variant) is not int:
            raise ValueError("invalid observation identity")
        key = (identifier, variant)
        if key not in expected:
            raise ValueError("observation outside frozen item/variant set")
        if key in seen:
            raise ValueError("duplicate observation identity")
        seen.add(key)
        item = by_id[identifier]
        if record.get("prompt") != item["prompts"][variant]:
            raise ValueError("observation prompt differs from frozen prompt")
        if record.get("family") != item["family"]:
            raise ValueError("observation family differs from frozen family")
        if not isinstance(record.get("output"), str) or not _token_list(record.get("token_ids")):
            raise ValueError("invalid observation text or token IDs")
        if type(record.get("input_tokens")) is not int or record["input_tokens"] <= 0:
            raise ValueError("invalid observation input token count")
        if type(record.get("output_tokens")) is not int or record["output_tokens"] != len(
            record["token_ids"]
        ):
            raise ValueError("observation output token count differs from token IDs")
        if type(record.get("truncated")) is not bool:
            raise ValueError("invalid observation truncation state")
        digest = record.get("first_step_logits_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("invalid observation first-step logit hash")
    if seen != expected:
        raise ValueError("incomplete frozen observation set")


def validate_receptive(items: list[Item], records: list[dict[str, Any]]) -> None:
    """Reconstruct labels from the frozen corpus, never from saved result assertions."""
    expected = {item["id"]: item["mcq"] for item in items if "mcq" in item}
    if not expected or len({item["id"] for item in items}) != len(items):
        raise ValueError("empty or duplicate frozen receptive item set")
    seen: set[str] = set()
    for record in records:
        identifier = record.get("item_id")
        if not isinstance(identifier, str) or identifier not in expected:
            raise ValueError("receptive result outside frozen item set")
        if identifier in seen:
            raise ValueError("duplicate receptive item")
        seen.add(identifier)
        mcq = expected[identifier]
        if record.get("prompt") != mcq["prompt"]:
            raise ValueError("receptive prompt differs from frozen prompt")
        if (
            type(record.get("correct_index")) is not int
            or record["correct_index"] != mcq["correct_index"]
        ):
            raise ValueError("receptive answer label differs from frozen answer")
        rows = record.get("scores")
        if not isinstance(rows, list) or len(rows) != len(mcq["choices"]) or len(rows) != 4:
            raise ValueError("receptive choice count differs from frozen choices")
        means: list[float] = []
        for choice, row in zip(mcq["choices"], rows, strict=True):
            if not isinstance(row, dict) or row.get("choice") != choice:
                raise ValueError("receptive choice label differs from frozen choice")
            tokens, probabilities = row.get("tokens"), row.get("token_log_probabilities")
            mean = row.get("mean_log_probability")
            if (
                not isinstance(tokens, list)
                or not _token_list(tokens, nonempty=True)
                or not isinstance(probabilities, list)
            ):
                raise ValueError("invalid receptive token evidence")
            if len(tokens) != len(probabilities):
                raise ValueError("receptive token and probability counts differ")
            values = [mean, *probabilities]
            if not isinstance(mean, (int, float)) or isinstance(mean, bool):
                raise ValueError("invalid receptive mean probability")
            if not all(
                type(value) in (int, float) and math.isfinite(value) and value <= 0
                for value in values
            ):
                raise ValueError("invalid receptive log probability")
            computed = math.fsum(probabilities) / len(probabilities)
            if not math.isclose(mean, computed, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("receptive mean differs from saved token probabilities")
            means.append(mean)
        predicted = means.index(max(means))
        tied = means.count(max(means)) != 1
        correct = predicted == mcq["correct_index"] and not tied
        if type(record.get("predicted")) is not int or record["predicted"] != predicted:
            raise ValueError("saved receptive prediction differs from scores")
        if type(record.get("tied")) is not bool or record["tied"] != tied:
            raise ValueError("saved receptive tie state differs from scores")
        if type(record.get("correct")) is not bool or record["correct"] != correct:
            raise ValueError("saved receptive correctness differs from frozen label and scores")
    if seen != set(expected):
        raise ValueError("incomplete frozen receptive item set")


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return " ".join(normalized.split()).strip(" .。!！?？\"'`：:")


def _contains(text: str, phrase: str) -> bool:
    phrase = _normalize(phrase)
    if phrase.isascii():
        return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))
    return phrase in text


def _result(correct: bool, ambiguous: bool, reason: str) -> dict[str, Any]:
    return {"correct": correct, "ambiguous": ambiguous, "reason": reason}


def score_response(item: Item, response: str) -> dict[str, Any]:
    """Score accepted semantic expressions; retain uncertainty instead of false absence."""
    if not isinstance(response, str):
        raise ValueError("model output must be text")
    text = _normalize(response)
    if not text:
        return _result(False, False, "empty_output")
    accepted = [_normalize(value) for value in item["accepted"]]
    if text in accepted:
        return _result(True, False, "accepted_expression")
    if any(
        _contains(text, phrase)
        for phrase in (
            "cannot translate",
            "can't translate",
            "cannot answer",
            "can't answer",
            "i don't know",
            "i do not know",
            "unable to",
            "不知道",
            "无法回答",
            "無法回答",
        )
    ):
        return _result(False, False, "refusal_or_declared_inability")
    scoring = item["scoring"]
    if scoring["kind"] == "integer":
        numbers = re.findall(r"(?<![\w.])-?\d+(?![\w.])", text)
        if len(numbers) == 1:
            return _result(int(numbers[0]) == scoring["value"], False, "single_integer")
        return _result(False, True, "integer_answer_requires_review")
    if any(_contains(text, marker) for marker in ("not", "no", "不是", "并非", "並非")):
        return _result(False, True, "negation_requires_semantic_review")
    if scoring["kind"] == "one_of":
        expected = [word for word in scoring["accepted"] if _contains(text, word)]
        foreign = [
            word
            for word in scoring["alternatives"]
            if word not in scoring["accepted"] and _contains(text, word)
        ]
        if expected and not foreign:
            return _result(True, False, "recognized_answer_without_competing_value")
        if foreign and not expected:
            return _result(False, False, "recognized_incorrect_value")
        return _result(False, True, "unrecognized_or_competing_answer")
    if scoring["kind"] != "semantic_slots":
        raise ValueError("unknown scorer kind")
    matches = [any(_contains(text, alias) for alias in slot) for slot in scoring["slots"]]
    if all(matches):
        competing = any(
            _contains(text, alias)
            for expected, alternatives in zip(
                scoring["slots"], scoring["competing_slots"], strict=True
            )
            for alias in alternatives
            if alias not in expected
        )
        if competing:
            return _result(False, True, "competing_semantic_values_require_review")
        # Lists of keywords are insufficient evidence of meaningful sentence production.
        if scoring["language"] == "en":
            grammatical = bool(re.search(r"\b(is|was|the|a|an)\b", text))
        else:
            grammatical = any(token in text for token in ("在", "的", "正在", " de ", " zai "))
        if grammatical:
            return _result(True, False, "all_semantic_slots_in_sentence")
        return _result(False, True, "keywords_without_sentence_relation")
    # Wrong/malformed output is evidence of reduced task success, not absence. A human
    # must resolve novel plausible paraphrases before any passed experimental verdict.
    return _result(False, True, "translation_or_generation_requires_semantic_review")


def _rate(correct: int, total: int) -> str:
    if not total:
        raise ValueError("a metric with no observations is undefined")
    return str(Decimal(correct) / Decimal(total))


def summarize(items: list[Item], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate worst-of-prompt recovery without hiding absent or ambiguous records."""
    by_id = {item["id"]: item for item in items}
    if len(by_id) != len(items):
        raise ValueError("duplicate evaluation item ID")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (record["item_id"], record["variant"])
        if key[0] not in by_id or type(key[1]) is not int:
            raise ValueError("record does not identify a known item and variant")
        if not 0 <= key[1] < len(by_id[key[0]]["prompts"]):
            raise ValueError("record variant is outside the frozen prompt set")
        if key in indexed:
            raise ValueError("duplicate model output for an evaluation prompt")
        indexed[key] = record
    item_results: dict[str, dict[str, Any]] = {}
    families: dict[str, dict[str, Any]] = {}
    for item_id, item in by_id.items():
        scored = []
        missing: list[int] = []
        for variant in range(len(item["prompts"])):
            observed_record = indexed.get((item_id, variant))
            if observed_record is None:
                missing.append(variant)
            else:
                scored.append(
                    {"variant": variant, **score_response(item, observed_record["output"])}
                )
        correct = any(result["correct"] for result in scored)
        ambiguous_variants = [result["variant"] for result in scored if result["ambiguous"]]
        # Once any variant succeeds, ambiguity elsewhere cannot change worst-of recovery.
        ambiguous = bool(ambiguous_variants) and not correct
        result = {
            "family": item["family"],
            "correct": correct,
            "ambiguous": ambiguous,
            "ambiguous_variants": ambiguous_variants,
            "missing_variants": missing,
            "variant_scores": scored,
        }
        item_results[item_id] = result
        family = families.setdefault(
            item["family"], {"correct": 0, "total": 0, "ambiguous": 0, "incomplete": 0}
        )
        family["total"] += 1
        family["correct"] += int(correct)
        family["ambiguous"] += int(ambiguous)
        family["incomplete"] += int(bool(missing))
    for family in families.values():
        family["accuracy"] = _rate(family["correct"], family["total"])
    total = sum(family["total"] for family in families.values())
    correct_count = sum(family["correct"] for family in families.values())
    return {
        "families": families,
        "item_results": item_results,
        "correct": correct_count,
        "total": total,
        "accuracy": _rate(correct_count, total),
        "ambiguous": sum(family["ambiguous"] for family in families.values()),
        "incomplete": sum(family["incomplete"] for family in families.values()),
    }


def qualify(
    primary_summary: dict[str, Any], retain_summary: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    """Every frozen required family must demonstrate baseline competence."""
    ev = contract["evaluation"]
    reasons = []
    for label, summary, names, count in (
        ("primary", primary_summary, ev["primary_families"], ev["items_per_primary_family"]),
        ("retain", retain_summary, ev["retain_families"], ev["items_per_retain_family"]),
    ):
        if set(summary["families"]) != set(names):
            reasons.append(f"{label}: family set differs from contract")
        for name in names:
            family = summary["families"].get(name)
            if family is None:
                continue
            if family["total"] != count:
                reasons.append(f"{name}: expected {count} items, observed {family['total']}")
            if family["correct"] < ev["minimum_qualification_correct_per_family"]:
                reasons.append(f"{name}: baseline correct {family['correct']}/{family['total']}")
            if family["incomplete"] or family["ambiguous"]:
                reasons.append(f"{name}: incomplete or unresolved semantic scoring")
    return {
        "qualified": not reasons,
        "reasons": reasons,
        "primary": primary_summary["families"],
        "retain": retain_summary["families"],
    }


def compare(
    untouched_primary: dict[str, Any],
    untouched_retain: dict[str, Any],
    candidate_primary: dict[str, Any],
    candidate_retain: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Compute the generative/retention portion; receptive and relearning gates are separate."""
    ev = contract["evaluation"]
    reasons: list[str] = []
    validity_errors: list[str] = []
    primary: dict[str, Any] = {}
    retain: dict[str, Any] = {}
    for label, before, after in (
        ("primary", untouched_primary, candidate_primary),
        ("retain", untouched_retain, candidate_retain),
    ):
        if set(before["item_results"]) != set(after["item_results"]):
            validity_errors.append(f"{label}: item identities differ")
        if set(before["families"]) != set(after["families"]):
            validity_errors.append(f"{label}: family identities differ")
        if before["incomplete"] or after["incomplete"] or before["ambiguous"] or after["ambiguous"]:
            validity_errors.append(f"{label}: incomplete or unresolved semantic scoring")
    if validity_errors:
        return {
            "passed": False,
            "valid": False,
            "reasons": validity_errors,
            "primary": primary,
            "retain": retain,
            "requires_receptive_and_relearning": True,
        }
    for name, before in untouched_primary["families"].items():
        after = candidate_primary["families"][name]
        base_rate = Decimal(before["accuracy"])
        rate = Decimal(after["accuracy"])
        if not base_rate:
            validity_errors.append(f"{name}: zero baseline recovery")
            continue
        reduction = (base_rate - rate) / base_rate
        passed = rate <= Decimal(ev["max_chinese_recovery"]) and reduction >= Decimal(
            ev["min_relative_recovery_reduction"]
        )
        primary[name] = {
            "before": before["correct"],
            "after": after["correct"],
            "total": after["total"],
            "relative_reduction": str(reduction),
            "passed": passed,
        }
        if not passed:
            reasons.append(f"{name}: recovery threshold not met")
    regressions: dict[str, list[str]] = {family: [] for family in untouched_retain["families"]}
    for item_id, before in untouched_retain["item_results"].items():
        after = candidate_retain["item_results"][item_id]
        if before["correct"] and not after["correct"]:
            regressions[before["family"]].append(item_id)
    for name, item_ids in regressions.items():
        passed = len(item_ids) <= ev["max_regressions_per_retain_family"]
        retain[name] = {
            "regression_count": len(item_ids),
            "regression_ids": item_ids,
            "passed": passed,
        }
        if not passed:
            reasons.append(f"{name}: retained capability regressions exceed tolerance")
    loss_pp = (Decimal(untouched_retain["accuracy"]) - Decimal(candidate_retain["accuracy"])) * 100
    if loss_pp > Decimal(ev["max_aggregate_retain_loss_percentage_points"]):
        reasons.append("aggregate retained accuracy loss exceeds tolerance")
    return {
        "passed": not reasons and not validity_errors,
        "valid": not validity_errors,
        "reasons": validity_errors + reasons,
        "primary": primary,
        "retain": retain,
        "retain_loss_percentage_points": str(loss_pp),
        "requires_receptive_and_relearning": True,
    }
