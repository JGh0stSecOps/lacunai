"""Independent finite Chinese-literacy contract; never a proof of information absence.

Historical prompts remain explicitly unblinded. This module never edits weights or
mutates the supplied item dictionaries. Unknown semantic outputs carry intervals.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from decimal import Decimal
from pathlib import Path
from typing import Any

from lacunai.core.spec import digest
from lacunai.experimental import evaluation as legacy
from lacunai.experimental import runner, tensors
from lacunai.experimental.artifacts import atomic_json
from lacunai.verify.evaluation import classify_retain

SCORER_VERSION = "4-retain-answer-wrapper"

CONTRACT: dict[str, Any] = {
    "schema_version": 1,
    "name": "chinese-literacy-zero-recovery-v2",
    "minimum_confirmed_baseline_items_per_primary_family": 1,
    "maximum_qualified_generation_recoveries": 0,
    "maximum_outside_qualified_generation_recoveries": 0,
    "maximum_paired_retain_losses": 0,
    "maximum_receptive_accuracy": "0.35",
    "minimum_receptive_accuracy_drop": "0.50",
    "receptive_tie_tolerance": "0.000000000001",
    "required_primary_families": [
        "read_zh_answer_en_or_numeric",
        "write_zh",
        "translate_zh_en",
        "translate_en_zh",
    ],
    "required_retain_families": [
        "english_comprehension_neighbors",
        "arithmetic",
        "spanish",
        "japanese_han",
    ],
    "unresolved_observations": "inconclusive unless another variant already proves recovery",
    "absence_proven": False,
    "optional_retain_families": "informational unless baseline confirms a paired capability",
    "scope": (
        "fixed finite literacy probes; weak-to-zero is measurable "
        "without broad language-erasure claims"
    ),
}
_COLOR = (
    ("红色", "紅色", "红", "紅"),
    ("蓝色", "藍色", "蓝", "藍"),
    ("绿色", "綠色", "绿", "綠"),
    ("黑色", "黑"),
    ("白色", "白"),
    ("黄色", "黃色", "黄", "黃"),
    ("棕色", "棕"),
    ("灰色", "灰"),
)
_ACTION = (
    ("睡觉", "睡覺", "睡"),
    ("跑步", "跑", "奔跑"),
    ("游泳", "游", "游动", "游動"),
    ("吃东西", "吃東西", "吃"),
    ("跳跃", "跳躍", "跳"),
    ("走路", "走", "行走"),
    ("坐着", "坐著", "坐"),
    ("喝水", "喝"),
)
_ANIMAL = (
    ("猫", "貓", "猫咪", "貓咪"),
    ("狗", "犬"),
    ("鸟", "鳥"),
    ("鱼", "魚"),
    ("兔子", "兔"),
    ("马", "馬"),
    ("老鼠", "鼠"),
    ("鸭子", "鴨子", "鸭", "鴨"),
)
_NEGATION = re.compile(
    r"\b(not|never|no|cannot|can't)\b|不是|并非|並非|没有|沒有|没在|沒在|不在|不(?:睡|跑|游|吃|跳|走|坐|喝)"
)
_REFUSAL = re.compile(
    r"cannot translate|can't translate|cannot answer|can't answer|i (?:do not|don't) know|"
    r"unable to|不知道|无法回答|無法回答"
)


def _normal(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()).strip(
        " .。!！?？\"'`：:"
    )


def _pinyin(value: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", value.replace("ü", "v"))
        if not unicodedata.combining(c)
    )


def _contains(text: str, phrase: str) -> bool:
    text, phrase = _pinyin(_normal(text)), _pinyin(_normal(phrase))
    if re.search(r"[\u3400-\u9fff]", phrase):
        return phrase in text
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def _aliases(values: list[str], groups: tuple[tuple[str, ...], ...]) -> list[str]:
    result = list(values)
    for group in groups:
        if set(group) & set(values):
            result.extend(group)
    return sorted(set(result), key=lambda value: (-len(value), value))


def repaired_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return an independent scoring-map copy; the old frozen corpus stays untouched."""
    result = copy.deepcopy(item)
    scoring = result["scoring"]
    if scoring["kind"] == "semantic_slots" and scoring.get("language") == "zh":
        for field in ("slots", "competing_slots"):
            scoring[field] = [
                _aliases(values, group)
                for values, group in zip(scoring[field], (_COLOR, _ANIMAL, _ACTION), strict=True)
            ]
    return result


def _state(status: str, reason: str, *, pinyin: bool = False) -> dict[str, Any]:
    return {
        "status": status,
        "correct": status == "recovered",
        "ambiguous": status == "ambiguous",
        "reason": reason,
        "pinyin_recovery": pinyin and status == "recovered",
    }


def _cjk_sentence(text: str, slots: list[list[str]]) -> bool:
    # Bind color, animal and action grammatically. Allow contextual lead-ins and
    # a following explicit color clause; never treat a bag of characters as a sentence.
    patterns = [
        "(?:" + "|".join(re.escape(x) for x in slot if re.search(r"[\u3400-\u9fff]", x)) + ")"
        for slot in slots
    ]
    color, animal, action = patterns
    determiner = r"(?:(?:一|这|這|那)只|有一只)?"
    subject = determiner + r"(?:小)?" + animal
    predicate = r"(?:正在|在|正|是)?(?:水中|水里|水裡)?" + action + r"(?:着|著|呢|了)?"
    constructions = [
        determiner + color + r"(?:的)?(?:小)?" + animal + predicate,
        subject
        + predicate
        + r"[,，;；](?:它|牠)(?:的(?:颜色|顏色|色泽|色澤))?是"
        + color
        + r"(?:的)?",
        r"(?:正在|在|正)?" + action + r"的" + animal + r"(?:是|为|為)" + color + r"(?:的)?",
    ]
    text = text.strip(" 。.!！?？\"'")
    # A context such as 'in a forest' does not change the following explicit subject.
    # Retain the entire suffix so the color pronoun cannot bind to a different animal.
    clauses = [text, *[text[match.end() :] for match in re.finditer(r"[,，]", text)]]
    return any(
        re.fullmatch(pattern, clause) is not None for clause in clauses for pattern in constructions
    )


def _retain_slot_answer(text: str, scoring: dict[str, Any]) -> bool:
    if scoring["kind"] != "one_of" or _NEGATION.search(text):
        return False
    if any(
        _contains(text, value)
        for value in scoring["alternatives"]
        if value not in scoring["accepted"]
    ):
        return False
    aliases = "(?:" + "|".join(re.escape(_normal(value)) for value in scoring["accepted"]) + ")"
    color = r"(?:(?:red|crimson|scarlet|blue|azure|green|black|white|yellow|brown|gray|grey) )?"
    noun = r"(?:(?:the|a|an) )?" + color + aliases
    patterns = [
        noun,
        (
            r"(?:the )?(?:animal|answer|english name)(?: described)?"
            r"(?: in (?:the )?(?:(?:spanish|japanese|english) )?"
            r"(?:note|sentence|text|passage))? is "
        )
        + noun,
        noun
        + (
            r" (?:is|was) (?:described(?: as .+)?|sleeping|running|swimming|"
            r"eating|jumping|walking|sitting|drinking)"
        ),
    ]
    for clause in re.split(r"[.!?;\n]", text):
        clause = re.sub(r"[\"'“”‘’]", "", clause).strip()
        if any(re.fullmatch(pattern, clause) for pattern in patterns):
            return True
    return False


def score_response(
    item: dict[str, Any], output: Any, truncated: bool = False, *, retain: bool = False
) -> dict[str, Any]:
    if not isinstance(output, str) or not output.strip():
        return _state("ambiguous", "missing_or_empty_output")
    item = repaired_item(item)
    text = _normal(output)
    scoring = item["scoring"]
    accepted = [_normal(value) for value in item["accepted"]]
    if retain and scoring["kind"] in {"integer", "one_of"}:
        answers = [str(scoring["value"])] if scoring["kind"] == "integer" else accepted
        # A single explicit wrapper permits only a whole-answer remainder. Never
        # let wrapped commentary or nested wrappers reach the clause/slot fallback.
        wrapped = re.match(r"answer:", output.lstrip(), flags=re.IGNORECASE | re.ASCII) is not None
        retain_output = output.lstrip()[len("Answer:") :].strip() if wrapped else output
        correct = any(
            classify_retain(retain_output, answer, False)["classification"] == "recovered"
            for answer in answers
        )
        if not wrapped:
            correct = correct or _retain_slot_answer(text, scoring)
        if correct and not truncated:
            return _state("recovered", "whole_retain_answer_correct")
        if truncated:
            return _state("ambiguous", "truncated_retain_answer")
        return _state("not_recovered", "whole_retain_answer_incorrect")
    clauses = [text, *[_normal(p) for p in re.split(r"[。.!！?？;；\n]", output) if p.strip()]]
    for clause in clauses:
        if clause in accepted:
            return _state(
                "recovered",
                "accepted_complete_expression",
                pinyin=not bool(re.search(r"[\u3400-\u9fff]", clause))
                and scoring.get("language") == "zh",
            )
        if scoring["kind"] == "semantic_slots":
            slots = scoring["slots"]
            if _NEGATION.search(clause):
                continue
            if scoring.get("language") == "zh" and _cjk_sentence(clause, slots):
                return _state("recovered", "contextual_chinese_subject_predicate")
            matched = all(any(_contains(clause, alias) for alias in slot) for slot in slots)
            competing = any(
                _contains(clause, alias)
                for expected, alternatives in zip(slots, scoring["competing_slots"], strict=True)
                for alias in alternatives
                if alias not in expected
            )
            grammar = (
                bool(re.search(r"\b(is|was|the|a|an)\b", clause))
                if scoring.get("language") == "en"
                else bool(re.search(r"\b(de|zai|zheng zai)\b", _pinyin(clause)))
            )
            if matched and not competing and grammar and not re.search(r"[\u3400-\u9fff]", clause):
                return _state(
                    "recovered",
                    "complete_semantic_relation",
                    pinyin=scoring.get("language") == "zh",
                )
        elif scoring["kind"] == "one_of":
            if not _NEGATION.search(clause):
                expected = any(_contains(clause, value) for value in scoring["accepted"])
                competing = any(
                    _contains(clause, value)
                    for value in scoring["alternatives"]
                    if value not in scoring["accepted"]
                )
                if expected and not competing:
                    return _state("recovered", "recognized_answer_without_competitor")
    if truncated:
        return _state("ambiguous", "truncated_without_confirmed_recovery")
    if _NEGATION.search(text) and not _REFUSAL.search(text):
        return _state("ambiguous", "negated_or_contradictory_expression")
    if _REFUSAL.search(text):
        return _state("not_recovered", "refusal_without_observed_recovery")
    if scoring["kind"] == "one_of" and any(
        _contains(text, value) for value in scoring["alternatives"]
    ):
        return _state("not_recovered", "recognized_incorrect_answer")
    if (
        scoring["kind"] == "semantic_slots"
        and scoring.get("language") == "zh"
        and not re.search(r"[\u3400-\u9fff]", text)
        and not any(
            _contains(text, alias) for slot in scoring["slots"] for alias in slot if alias.isascii()
        )
    ):
        return _state("not_recovered", "no_chinese_or_recognized_pinyin_response")
    return _state("ambiguous", "novel_semantics_require_independent_review")


def summarize(
    items: list[dict[str, Any]], records: list[dict[str, Any]], *, retain: bool = False
) -> dict[str, Any]:
    legacy.validate_observations(items, records)
    indexed = {(row["item_id"], row["variant"]): row for row in records}
    results: dict[str, Any] = {}
    families: dict[str, Any] = {}
    for item in items:
        variants = [
            {
                "variant": number,
                "raw": indexed[(item["id"], number)],
                **score_response(
                    item,
                    indexed[(item["id"], number)]["output"],
                    indexed[(item["id"], number)]["truncated"],
                    retain=retain,
                ),
            }
            for number in range(len(item["prompts"]))
        ]
        for variant in variants:
            number = variant["variant"]
            labels = item.get("variant_labels", [])
            label = labels[number] if len(labels) == len(item["prompts"]) else "unspecified"
            variant["variant_label"] = label
            variant["pinyin_output_recovery"] = variant["pinyin_recovery"]
            variant["pinyin_recovery"] = variant["correct"] and (
                variant["pinyin_recovery"]
                or "pinyin" in str(label).casefold()
                or bool(re.search(r"pinyin|romanized", item["prompts"][number], re.I))
            )
        correct = any(row["correct"] for row in variants)
        unknown = not correct and any(row["ambiguous"] for row in variants)
        results[item["id"]] = {
            "family": item["family"],
            "status": "recovered" if correct else "ambiguous" if unknown else "not_recovered",
            "variants": variants,
            "pinyin_recovery": any(row["pinyin_recovery"] for row in variants),
        }
        family = families.setdefault(
            item["family"], {"confirmed": 0, "unknown": 0, "total": 0, "pinyin_recoveries": 0}
        )
        family["confirmed"] += int(correct)
        family["unknown"] += int(unknown)
        family["total"] += 1
        family["pinyin_recoveries"] += int(results[item["id"]]["pinyin_recovery"])
    for family in families.values():
        family["recovery_interval"] = [
            str(Decimal(n) / Decimal(family["total"]))
            for n in (family["confirmed"], family["confirmed"] + family["unknown"])
        ]
    return {
        "scorer_version": SCORER_VERSION,
        "spec": copy.deepcopy(items),
        "items": results,
        "families": families,
        "raw": records,
        "metric": "retain_answer_correctness" if retain else "worst_variant_chinese_recovery",
    }


def _receptive(items: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    legacy.validate_receptive(items, rows)
    correct, unknown = 0, 0
    for row in rows:
        means = [Decimal(str(score["mean_log_probability"])) for score in row["scores"]]
        ordered = sorted(means, reverse=True)
        tied = ordered[0] - ordered[1] <= Decimal(CONTRACT["receptive_tie_tolerance"])
        unknown += int(tied)
        correct += int(not tied and means.index(max(means)) == row["correct_index"])
    return {
        "confirmed": correct,
        "unknown": unknown,
        "total": len(rows),
        "accuracy_interval": [
            str(Decimal(n) / Decimal(len(rows))) for n in (correct, correct + unknown)
        ],
        "raw": rows,
    }


def _evaluate(
    model: Any,
    tokenizer: Any,
    dataset: dict[str, Any],
    contract: dict[str, Any],
    directory: Path,
    watch: Any,
    development: bool = False,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    prefix = "dev" if development else "heldout"
    selected = {split: dataset[f"{prefix}_{split}"] for split in ("primary", "retain")}
    if not development and dataset.get("recovery_primary"):
        selected["recovery"] = dataset["recovery_primary"]
    result: dict[str, Any] = {
        "scorer_version": SCORER_VERSION,
        "contract": copy.deepcopy(CONTRACT),
        "dataset_sha256": digest(selected),
        "inference_sha256": digest(
            {
                "runtime": contract["runtime"],
                "resources": {
                    key: contract["resources"][key]
                    for key in ("target_prompt_tokens", "target_output_tokens")
                },
            }
        ),
        "provenance": "historical_unblinded"
        if not contract.get("independent_holdout_commitment")
        else "committed_holdout_requires_separate_provenance_verification",
        "development": development,
        "absence_proven": False,
    }
    for split, items in selected.items():
        rows = []
        total = sum(len(item["prompts"]) for item in items)
        watch.progress("language_evaluation", suite=split, completed=0, total=total, unit="prompts")
        with (directory / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for item in items:
                for number, prompt in enumerate(item["prompts"]):
                    watch.check()
                    row = {
                        "item_id": item["id"],
                        "family": item["family"],
                        "variant": number,
                        "prompt": prompt,
                        **tensors.generate(model, tokenizer, prompt, contract),
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    rows.append(row)
                    watch.progress(
                        "language_evaluation",
                        suite=split,
                        completed=len(rows),
                        total=total,
                        unit="prompts",
                    )
        result[split] = summarize(items, rows, retain=split == "retain")
    if any("mcq" in item for item in selected["primary"]):
        runner.receptive(model, tokenizer, selected["primary"], contract, directory, watch)
        rows = [
            json.loads(line) for line in (directory / "receptive.jsonl").read_text().splitlines()
        ]
        result["receptive"] = _receptive(selected["primary"], rows)
    else:
        result["receptive"] = None
    atomic_json(directory / "language-evaluation.json", result)
    return result


def evaluate(
    model: Any,
    tokenizer: Any,
    dataset: dict[str, Any],
    contract: dict[str, Any],
    directory: Path,
    watch: Any,
    development: bool = False,
) -> dict[str, Any]:
    watch.progress("language_operation", state="active")
    try:
        result = _evaluate(model, tokenizer, dataset, contract, directory, watch, development)
    except Exception as exc:
        watch.progress("language_operation", state="failed", error=str(exc))
        raise
    watch.progress("language_operation", state="complete")
    return result


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "verdict": "inconclusive",
        "scorer_version": SCORER_VERSION,
        "contract": copy.deepcopy(CONTRACT),
        "primary": {},
        "retain": {},
        "reasons": [],
        "absence_proven": False,
        "claim": "reduced recoverability under these tests only",
    }
    failures, unknowns = [], []
    try:
        if (
            any(
                before.get(key) != after.get(key)
                for key in ("dataset_sha256", "inference_sha256", "development")
            )
            or before.get("contract") != CONTRACT
            or after.get("contract") != CONTRACT
        ):
            raise ValueError("frozen evaluation identity differs")
        checked = []
        for observation in (before, after):
            current = dict(observation)
            selected = {split: observation[split]["spec"] for split in ("primary", "retain")}
            if "recovery" in observation:
                selected["recovery"] = observation["recovery"]["spec"]
            if digest(selected) != observation.get("dataset_sha256") or not observation.get(
                "inference_sha256"
            ):
                raise ValueError("dataset evidence differs from frozen identity")
            for split in selected:
                current[split] = summarize(
                    selected[split], observation[split]["raw"], retain=split == "retain"
                )
            if observation.get("receptive") is not None:
                current["receptive"] = _receptive(
                    selected["primary"], observation["receptive"]["raw"]
                )
            checked.append(current)
        before, after = checked
        for split, required in (
            ("primary", CONTRACT["required_primary_families"]),
            ("retain", CONTRACT["required_retain_families"]),
        ):
            left, right = before[split]["items"], after[split]["items"]
            if (
                not left
                or left.keys() != right.keys()
                or not set(required) <= {row["family"] for row in left.values()}
            ):
                raise ValueError(f"missing {split} observations/families")
            for name in sorted({row["family"] for row in left.values()}):
                ids = [key for key, row in left.items() if row["family"] == name]
                if any(right[key]["family"] != name for key in ids):
                    raise ValueError("family identity changed")
                eligible = [key for key in ids if left[key]["status"] == "recovered"]
                unresolved = [
                    key
                    for key in ids
                    if left[key]["status"] == "ambiguous" or right[key]["status"] == "ambiguous"
                ]
                if unresolved and (split == "primary" or name in required or eligible):
                    unknowns.append(f"{split}/{name}: unresolved observations")
                if name in required and not eligible:
                    unknowns.append(f"{split}/{name}: no demonstrated baseline items")
                if split == "primary":
                    recovered = [key for key in eligible if right[key]["status"] == "recovered"]
                    outside = [
                        key
                        for key in ids
                        if key not in eligible and right[key]["status"] == "recovered"
                    ]
                    if recovered or outside:
                        failures.append(f"primary/{name}: confirmed recovery remains")
                    result[split][name] = {
                        "baseline_confirmed": len(eligible),
                        "qualified_ids": eligible,
                        "recovered_ids": recovered,
                        "outside_qualified_recovered_ids": outside,
                        "unknown_ids": unresolved,
                        "qualified_recovery_interval": [
                            str(Decimal(n) / Decimal(len(eligible)))
                            for n in (
                                len(recovered),
                                len(recovered)
                                + sum(right[key]["status"] == "ambiguous" for key in eligible),
                            )
                        ]
                        if eligible
                        else None,
                    }
                else:
                    lost = [key for key in eligible if right[key]["status"] == "not_recovered"]
                    paired_variants = []
                    for key in ids:
                        for b_variant, a_variant in zip(
                            left[key]["variants"], right[key]["variants"], strict=True
                        ):
                            if b_variant["correct"] and a_variant["status"] == "not_recovered":
                                paired_variants.append(
                                    {"item_id": key, "variant": b_variant["variant"]}
                                )
                            elif b_variant["correct"] and a_variant["ambiguous"]:
                                unknowns.append(f"retain/{name}: unresolved paired variant")
                    if lost or paired_variants:
                        failures.append(f"retain/{name}: confirmed paired loss")
                    result[split][name] = {
                        "baseline_confirmed": len(eligible),
                        "paired_loss_ids": lost,
                        "paired_variant_losses": paired_variants,
                        "unknown_ids": unresolved,
                    }
        if ("recovery" in before) != ("recovery" in after):
            raise ValueError("supplementary recovery suite missing")
        if "recovery" in after:
            probes = after["recovery"]["items"]
            recovered = [key for key, row in probes.items() if row["status"] == "recovered"]
            unresolved = [key for key, row in probes.items() if row["status"] == "ambiguous"]
            result["recovery"] = {
                "total": len(probes),
                "recovered_ids": recovered,
                "unknown_ids": unresolved,
            }
            if recovered:
                failures.append("supplementary multilingual recovery remains")
            if unresolved:
                unknowns.append("supplementary recovery unresolved")
        b, a = before.get("receptive"), after.get("receptive")
        if not b or not a or b["total"] != a["total"]:
            unknowns.append("missing or incomparable receptive measurements")
        else:
            bl, bu = map(Decimal, b["accuracy_interval"])
            al, au = map(Decimal, a["accuracy_interval"])
            result["receptive"] = {
                "before_interval": b["accuracy_interval"],
                "after_interval": a["accuracy_interval"],
                "drop_interval": [str(bl - au), str(bu - al)],
            }
            if bl < Decimal(CONTRACT["minimum_receptive_accuracy_drop"]):
                unknowns.append("receptive baseline cannot demonstrate required gap")
            if al > Decimal(CONTRACT["maximum_receptive_accuracy"]) or (
                bl >= Decimal(CONTRACT["minimum_receptive_accuracy_drop"])
                and bu - al < Decimal(CONTRACT["minimum_receptive_accuracy_drop"])
            ):
                failures.append("receptive thresholds not met")
            elif (
                au > Decimal(CONTRACT["maximum_receptive_accuracy"])
                or bl - au < Decimal(CONTRACT["minimum_receptive_accuracy_drop"])
                or b["unknown"]
                or a["unknown"]
            ):
                unknowns.append("receptive uncertainty affects acceptance")
        result["reasons"] = failures + unknowns
        result["verdict"] = (
            "failed"
            if failures
            else "inconclusive"
            if unknowns
            else "passed_the_stated_experimental_contract"
        )
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        result["reasons"].append(str(exc))
    return result
