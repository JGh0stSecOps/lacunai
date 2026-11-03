# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Finite experimental data/scoring checks; these do not certify model unlearning."""

from __future__ import annotations

from copy import deepcopy

import pytest

from lacunai.experimental import evaluation
from lacunai.experimental.data import (
    PRIMARY,
    RETAIN,
    build_dataset,
    dataset_sha256,
    validate_dataset,
)
from lacunai.experimental.evaluation import compare, qualify, score_response, summarize


@pytest.fixture
def contract():
    return {
        "runtime": {"seed": 20261102},
        "evaluation": {
            "primary_families": list(PRIMARY),
            "retain_families": list(RETAIN),
            "items_per_primary_family": 16,
            "items_per_retain_family": 16,
            "prompt_variants_per_item": 4,
            "minimum_qualification_correct_per_family": 12,
            "max_chinese_recovery": "0.25",
            "min_relative_recovery_reduction": "0.50",
            "max_regressions_per_retain_family": 1,
            "max_aggregate_retain_loss_percentage_points": "5",
        },
    }


def records(items, output=None):
    return [
        {
            "item_id": item["id"],
            "variant": variant,
            "output": item["answer"] if output is None else output,
        }
        for item in items
        for variant in range(len(item["prompts"]))
    ]


def test_dataset_is_deterministic_with_actual_variation(contract):
    first = build_dataset(contract)
    assert dataset_sha256(first) == dataset_sha256(build_dataset(contract))
    changed = deepcopy(contract)
    changed["runtime"]["seed"] += 1
    assert dataset_sha256(first) != dataset_sha256(build_dataset(changed))
    assert all("diagnostic_zh" in item and "diagnostic_en" in item for item in first["dev_primary"])
    assert all("diagnostic_zh" not in item for item in first["heldout_primary"])
    reading = [item for item in first["heldout_primary"] if "receptive" in item]
    assert [item["receptive"]["correct_index"] for item in reading] == list(range(4)) * 4
    assert all(
        item["answer"] == item["receptive"]["choices"][item["receptive"]["correct_index"]]
        for item in reading
    )


def test_heldout_combination_cannot_be_injected_into_development(contract):
    dataset = build_dataset(contract)
    dataset["dev_primary"][0]["combination"] = dataset["heldout_primary"][0]["combination"]
    with pytest.raises(ValueError, match="heldout lexical combination"):
        validate_dataset(dataset, contract)


def test_distractor_facts_also_cannot_leak_across_splits(contract):
    dataset = build_dataset(contract)
    dataset["forget_train"][0]["context_combinations"].append(
        dataset["heldout_primary"][0]["combination"]
    )
    with pytest.raises(ValueError, match="heldout lexical combination"):
        validate_dataset(dataset, contract)


def test_no_duplicate_ids_and_no_missing_withheld_family(contract):
    dataset = build_dataset(contract)
    dataset["heldout_primary"][1]["id"] = dataset["heldout_primary"][0]["id"]
    with pytest.raises(ValueError, match="duplicate item ID"):
        validate_dataset(dataset, contract)
    dataset = build_dataset(contract)
    dataset["heldout_retain"].pop()
    with pytest.raises(ValueError, match="family counts"):
        validate_dataset(dataset, contract)


def test_relearning_rejects_chinese_instruction(contract):
    dataset = build_dataset(contract)
    dataset["relearn"][0]["answer"] = "Translate Chinese into English."
    with pytest.raises(ValueError, match="excluded language"):
        validate_dataset(dataset, contract)


def test_semantic_synonyms_and_pinyin_count_as_recovery(contract):
    dataset = build_dataset(contract)
    english = next(
        item for item in dataset["heldout_primary"] if item["family"] == "translate_zh_en"
    )
    words = [slot[-1] for slot in english["scoring"]["slots"]]
    assert score_response(english, f"The {words[0]} {words[1]} is {words[2]}.")["correct"]
    chinese = next(item for item in dataset["heldout_primary"] if item["family"] == "write_zh")
    assert score_response(chinese, chinese["accepted"][-1])["correct"]
    assert not score_response(chinese, english["answer"])["correct"]


def test_unknown_paraphrase_does_not_manufacture_forgetting(contract):
    item = build_dataset(contract)["heldout_primary"][0]
    result = score_response(item, "Its shade resembles a ripe strawberry.")
    assert result["ambiguous"]
    assert not result["correct"]


def test_conflicting_answer_and_keyword_lists_are_not_scored_correct(contract):
    item = next(x for x in build_dataset(contract)["heldout_primary"] if x["family"] == "write_zh")
    words = [slot[0] for slot in item["scoring"]["slots"]]
    assert score_response(item, "、".join(words))["ambiguous"]
    other = next(
        x for x in item["scoring"]["competing_slots"][0] if x not in item["scoring"]["slots"][0]
    )
    result = score_response(item, item["answer"] + other)
    assert result["ambiguous"] and not result["correct"]


def test_missing_or_duplicate_outputs_cannot_pass_qualification(contract):
    dataset = build_dataset(contract)
    primary, retain = dataset["heldout_primary"], dataset["heldout_retain"]
    raw = records(primary)
    summary = summarize(primary, raw[:-1])
    assert summary["incomplete"] == 1
    assert not qualify(summary, summarize(retain, records(retain)), contract)["qualified"]
    with pytest.raises(ValueError, match="duplicate model output"):
        summarize(primary, raw + [raw[0]])


def test_worst_variant_success_survives_other_variant_ambiguity(contract):
    item = build_dataset(contract)["heldout_primary"][0]
    raw = records([item], "unrecognized answer")
    raw[2]["output"] = item["answer"]
    summary = summarize([item], raw)
    result = summary["item_results"][item["id"]]
    assert result["correct"] and not result["ambiguous"]
    assert len(result["ambiguous_variants"]) == 3


def test_retention_regressions_cannot_be_hidden_by_aggregate_gain(contract):
    dataset = build_dataset(contract)
    primary, retain = dataset["heldout_primary"], dataset["heldout_retain"]
    baseline_primary = summarize(primary, records(primary))
    baseline_retain = summarize(retain, records(retain))
    after = records(retain)
    after[0]["output"] = "I cannot answer."
    after[1]["output"] = "I cannot answer."
    result = compare(
        baseline_primary,
        baseline_retain,
        summarize(primary, records(primary, "I cannot answer.")),
        summarize(retain, after),
        contract,
    )
    assert result["valid"] and not result["passed"]
    assert result["retain"]["english_comprehension_neighbors"]["regression_count"] == 2
    assert result["retain_loss_percentage_points"] == "2.500"


def test_comparison_is_not_a_complete_unlearning_verdict(contract):
    dataset = build_dataset(contract)
    primary, retain = dataset["heldout_primary"], dataset["heldout_retain"]
    before_p = summarize(primary, records(primary))
    before_r = summarize(retain, records(retain))
    assert qualify(before_p, before_r, contract)["qualified"]
    result = compare(
        before_p,
        before_r,
        summarize(primary, records(primary, "I cannot answer.")),
        before_r,
        contract,
    )
    assert result["passed"] and result["requires_receptive_and_relearning"]


def observed_records(items):
    return [
        {
            "item_id": item["id"],
            "family": item["family"],
            "variant": variant,
            "prompt": prompt,
            "output": item["answer"],
            "token_ids": [11],
            "input_tokens": 12,
            "output_tokens": 1,
            "truncated": False,
            "first_step_logits_sha256": "a" * 64,
        }
        for item in items
        for variant, prompt in enumerate(item["prompts"])
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt", "A different question"),
        ("family", "other"),
        ("item_id", "not-frozen"),
        ("variant", True),
        ("token_ids", [True]),
        ("output_tokens", 2),
        ("output", None),
        ("truncated", "false"),
    ],
)
def test_saved_observations_reject_tampered_identity_and_payload(contract, field, value):
    items = build_dataset(contract)["heldout_primary"][:1]
    raw = observed_records(items)
    evaluation.validate_observations(items, raw)
    raw[0][field] = value
    with pytest.raises(ValueError):
        evaluation.validate_observations(items, raw)


def test_saved_observations_require_every_frozen_variant_once(contract):
    items = build_dataset(contract)["heldout_primary"][:1]
    raw = observed_records(items)
    with pytest.raises(ValueError, match="incomplete"):
        evaluation.validate_observations(items, raw[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.validate_observations(items, raw + [raw[0]])


def receptive_records(items):
    result = []
    for item in items:
        if "mcq" not in item:
            continue
        mcq = item["mcq"]
        scores = [
            {
                "choice": choice,
                "mean_log_probability": -float(index + 1),
                "token_log_probabilities": [-float(index + 1)],
                "tokens": [index + 1],
            }
            for index, choice in enumerate(mcq["choices"])
        ]
        result.append(
            {
                "item_id": item["id"],
                "prompt": mcq["prompt"],
                "correct_index": mcq["correct_index"],
                "scores": scores,
                "predicted": 0,
                "correct": mcq["correct_index"] == 0,
                "tied": False,
            }
        )
    return result


@pytest.mark.parametrize("tamper", ["label", "prompt", "choice", "nan", "duplicate", "missing"])
def test_receptive_readback_does_not_trust_saved_answer_labels(contract, tamper):
    items = build_dataset(contract)["heldout_primary"]
    raw = receptive_records(items)
    evaluation.validate_receptive(items, raw)
    if tamper == "label":
        raw[0]["correct_index"] = (raw[0]["correct_index"] + 1) % 4
    elif tamper == "prompt":
        raw[0]["prompt"] = "Altered question"
    elif tamper == "choice":
        raw[0]["scores"][0]["choice"] = "different answer"
    elif tamper == "nan":
        raw[0]["scores"][0]["mean_log_probability"] = float("nan")
    elif tamper == "duplicate":
        raw.append(raw[0])
    else:
        raw.pop()
    with pytest.raises(ValueError):
        evaluation.validate_receptive(items, raw)
