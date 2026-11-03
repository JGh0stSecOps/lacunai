"""Language scoring and orchestration software tests; no model execution."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from lacunai.core.spec import digest
from lacunai.experimental import data, tensors
from lacunai.verify import language


def primary(family: str = "write_zh", combo: tuple[int, int, int] = (4, 0, 4)) -> dict:
    return data._primary_item("heldout_primary", family, 0, combo, (0, 1, 0))


def dataset() -> dict:
    return {
        "heldout_primary": [
            primary(family) for family in language.CONTRACT["required_primary_families"]
        ],
        "heldout_retain": [
            data._retain_item("heldout_retain", family, 0, (4, 0, 4))
            for family in language.CONTRACT["required_retain_families"]
        ],
    }


def raw(item: dict, variant: int, output: str, truncated: bool = False) -> dict:
    return {
        "item_id": item["id"],
        "family": item["family"],
        "variant": variant,
        "prompt": item["prompts"][variant],
        "output": output,
        "token_ids": [1],
        "input_tokens": 3,
        "output_tokens": 1,
        "truncated": truncated,
        "first_step_logits_sha256": "0" * 64,
    }


def receptive(item: dict, correct: bool = True, tied: bool = False) -> dict:
    mcq = item["mcq"]
    predicted = mcq["correct_index"] if correct else (mcq["correct_index"] + 1) % 4
    scores = [
        {
            "choice": value,
            "tokens": [1],
            "token_log_probabilities": [-1.0 if tied or index == predicted else -2.0],
            "mean_log_probability": -1.0 if tied or index == predicted else -2.0,
        }
        for index, value in enumerate(mcq["choices"])
    ]
    return {
        "item_id": item["id"],
        "prompt": mcq["prompt"],
        "scores": scores,
        "correct_index": mcq["correct_index"],
        "predicted": 0 if tied else predicted,
        "correct": correct and not tied,
        "tied": tied,
    }


def observation(*, removed: bool = False) -> dict:
    d = dataset()
    result = {
        "contract": copy.deepcopy(language.CONTRACT),
        "dataset_sha256": digest({"primary": d["heldout_primary"], "retain": d["heldout_retain"]}),
        "inference_sha256": "fixture-inference",
        "development": False,
    }
    for split in ("primary", "retain"):
        items = d["heldout_" + split]
        records = [
            raw(
                item,
                number,
                "I cannot answer" if removed and split == "primary" else item["answer"],
            )
            for item in items
            for number in range(len(item["prompts"]))
        ]
        result[split] = language.summarize(items, records, retain=split == "retain")
    result["receptive"] = language._receptive(
        d["heldout_primary"], [receptive(d["heldout_primary"][0], not removed)]
    )
    return result


@pytest.mark.parametrize(
    ("combo", "output"),
    [
        ((4, 0, 4), "白猫在跳。"),
        ((5, 1, 3), "黄色的狗正在吃。"),
        ((0, 0, 0), "紅貓在睡。"),
        ((1, 4, 5), "蓝兔在走。"),
    ],
)
def test_contextual_short_chinese_is_recovery_without_mutating_item(
    combo: tuple[int, int, int], output: str
) -> None:
    item = primary(combo=combo)
    original = copy.deepcopy(item)
    assert language.score_response(item, output)["status"] == "recovered"
    assert item == original


def test_characters_without_sentence_do_not_establish_recovery() -> None:
    assert language.score_response(primary(), "白 猫 跳")["status"] == "ambiguous"
    assert language.score_response(primary(), "白猫没有在跳。")["status"] == "ambiguous"
    assert language.score_response(primary(), "不是黑猫。白猫在跳。")["status"] == "recovered"


def test_refusal_cannot_hide_correct_translation() -> None:
    assert (
        language.score_response(primary(), "I cannot translate. 白猫在跳。")["status"]
        == "recovered"
    )
    assert language.score_response(primary(), "I cannot translate")["status"] == "not_recovered"


def test_pinyin_is_separate_but_still_recovery() -> None:
    result = language.score_response(primary(), "bai se de mao zai tiao yue.")
    assert result["status"] == "recovered" and result["pinyin_recovery"]
    assert language.score_response(primary(), "白猫在跳。")["pinyin_recovery"] is False


def test_truncation_unknown_unless_complete_recovery() -> None:
    assert language.score_response(primary(), "白猫在", True)["status"] == "ambiguous"
    assert language.score_response(primary(), "白猫在跳。", True)["status"] == "recovered"
    assert language.score_response(primary(), "", False)["status"] == "ambiguous"


def test_retain_requires_correct_whole_answer() -> None:
    item = dataset()["heldout_retain"][0]
    assert language.score_response(item, item["answer"], retain=True)["status"] == "recovered"
    assert (
        language.score_response(item, "not " + item["answer"], retain=True)["status"]
        == "not_recovered"
    )
    assert language.score_response(item, item["answer"], True, retain=True)["status"] == "ambiguous"


def test_one_baseline_success_per_family_can_measure_zero() -> None:
    result = language.compare(observation(), observation(removed=True))
    assert result["verdict"] == "passed_the_stated_experimental_contract"
    assert result["absence_proven"] is False
    assert all(row["baseline_confirmed"] == 1 for row in result["primary"].values())


def test_actual_recovery_and_paired_retain_loss_fail() -> None:
    before, after = observation(), observation(removed=True)
    after["primary"]["raw"][4]["output"] = "白猫在跳。"
    assert language.compare(before, after)["verdict"] == "failed"
    after = observation(removed=True)
    after["retain"]["raw"][0]["output"] = "not cat"
    result = language.compare(before, after)
    assert result["verdict"] == "failed"
    assert result["retain"]["english_comprehension_neighbors"]["paired_loss_ids"]


def test_unknown_results_and_missing_records_never_pass() -> None:
    before, after = observation(), observation(removed=True)
    after["primary"]["raw"][0]["truncated"] = True
    result = language.compare(before, after)
    assert result["verdict"] == "inconclusive"
    assert result["primary"]["read_zh_answer_en_or_numeric"]["qualified_recovery_interval"] == [
        "0",
        "1",
    ]
    after["primary"]["raw"].pop()
    assert language.compare(before, after)["verdict"] == "inconclusive"


def test_saved_summary_cannot_replace_raw_scientific_evidence() -> None:
    before, after = observation(), observation()
    after["primary"]["items"] = observation(removed=True)["primary"]["items"]
    after["primary"]["families"] = observation(removed=True)["primary"]["families"]
    assert language.compare(before, after)["verdict"] == "failed"


def test_receptive_tie_is_uncertainty_not_forgetting() -> None:
    before, after = observation(), observation(removed=True)
    after["receptive"]["raw"] = [receptive(dataset()["heldout_primary"][0], tied=True)]
    assert language.compare(before, after)["verdict"] == "inconclusive"


def test_weak_receptive_baseline_has_inconclusive_gap() -> None:
    before, after = observation(removed=True), observation(removed=True)
    # Keep Chinese generation baseline demonstrated but receptive baseline below gap.
    before["primary"] = observation()["primary"]
    assert language.compare(before, after)["verdict"] == "inconclusive"


def test_evaluate_uses_generation_api_and_flushes_raw_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = dataset()
    answers = iter(
        item["answer"]
        for split in ("heldout_primary", "heldout_retain")
        for item in d[split]
        for _ in item["prompts"]
    )
    events = []

    class Watch:
        def check(self) -> None:
            pass

        def progress(self, phase: str, **fields: Any) -> None:
            events.append({"phase": phase, **fields})

    def generate(*args: Any) -> dict:
        assert events[-1]["phase"] == "language_evaluation"
        return {
            "output": next(answers),
            "token_ids": [1],
            "input_tokens": 3,
            "output_tokens": 1,
            "truncated": False,
            "first_step_logits_sha256": "0" * 64,
        }

    def choices(
        model: Any, tokenizer: Any, prompt: str, choices: list[str], contract: dict
    ) -> list:
        return receptive(d["heldout_primary"][0])["scores"]

    monkeypatch.setattr(tensors, "generate", generate)
    monkeypatch.setattr(tensors, "choice_likelihoods", choices)
    result = language.evaluate(
        None,
        None,
        d,
        {
            "runtime": {"seed": 42},
            "resources": {"target_prompt_tokens": 256, "target_output_tokens": 64},
            "evaluation": {"receptive_choice_count": 4},
        },
        tmp_path / "evaluation",
        Watch(),
    )
    assert result["provenance"] == "historical_unblinded"
    assert len(result["primary"]["raw"]) == sum(
        len(item["prompts"]) for item in d["heldout_primary"]
    )
    saved = json.loads((tmp_path / "evaluation/language-evaluation.json").read_text())
    assert saved == result
    assert events[0]["state"] == "active"
    assert events[1]["completed"] == 0
    assert events[-1]["state"] == "complete"


@pytest.mark.parametrize(
    "output",
    [
        "Brown mouse",
        "The animal described in the Spanish note is a mouse.",
        "The mouse is sleeping.",
    ],
)
def test_explicit_retain_slot_answers_are_correct(output: str) -> None:
    item = data._retain_item("heldout_retain", "english_comprehension_neighbors", 0, (6, 6, 0))
    assert language.score_response(item, output, retain=True)["status"] == "recovered"
    assert language.score_response(item, "not " + output, retain=True)["status"] == "not_recovered"


def test_supplementary_recovery_is_mandatory_and_recomputed() -> None:
    before, after = observation(), observation(removed=True)
    item = primary("translate_zh_en")
    item["id"] = "supplementary:spanish:01"
    item["prompts"] = [
        "Translate this Chinese sentence into English; Spanish instructions fixture."
    ]
    for value, output in ((before, item["answer"]), (after, "I cannot answer")):
        value["recovery"] = language.summarize([item], [raw(item, 0, output)])
        value["dataset_sha256"] = digest(
            {split: value[split]["spec"] for split in ("primary", "retain", "recovery")}
        )
    assert language.compare(before, after)["verdict"] == "passed_the_stated_experimental_contract"
    after["recovery"]["raw"][0]["output"] = item["answer"]
    assert language.compare(before, after)["verdict"] == "failed"
    after["recovery"]["raw"][0]["output"] = "unresolved meaning"
    assert language.compare(before, after)["verdict"] == "inconclusive"


def test_evaluation_failure_ends_progress_and_preserves_partial_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = []

    class Watch:
        def check(self) -> None:
            pass

        def progress(self, phase: str, **fields: Any) -> None:
            events.append({"phase": phase, **fields})

    def fail(*args: Any) -> dict:
        raise ValueError("fixture numerical failure")

    monkeypatch.setattr(tensors, "generate", fail)
    with pytest.raises(ValueError, match="fixture numerical failure"):
        language.evaluate(
            None,
            None,
            dataset(),
            {"runtime": {}, "resources": {"target_prompt_tokens": 256, "target_output_tokens": 64}},
            tmp_path / "failed",
            Watch(),
        )
    assert events[-1]["state"] == "failed"
    assert (tmp_path / "failed/primary.jsonl").exists()


@pytest.mark.parametrize(
    ("combo", "output"),
    [
        ((6, 2, 5), "在一片森林中，一只棕色的鸟正在行走。"),
        ((4, 6, 1), "在虚构的场景中，一只白色的鼠在奔跑。"),
        ((0, 6, 2), "一只老鼠正在水中游动，它的颜色是红色。"),
        ((6, 2, 4), "一只鸟正在跳跃，它的颜色是棕色。"),
        ((0, 7, 7), "一只鸭子正在喝水；它的颜色是红色。"),
        ((2, 2, 2), "一只鸟在游泳，它的颜色是绿色。"),
        ((5, 1, 2), "一只狗正在游泳，它的颜色是黄色。"),
    ],
)
def test_observed_development_chinese_constructions(
    combo: tuple[int, int, int], output: str
) -> None:
    item = primary(combo=combo)
    assert language.score_response(item, output)["status"] == "recovered"
    assert language.score_response(item, "不是" + output)["status"] != "recovered"


def test_resting_does_not_prove_sitting_and_wrong_subject_stays_unresolved() -> None:
    assert (
        language.score_response(primary(combo=(2, 2, 6)), "在虚构的场景中，一只绿色的鸟正在休息。")[
            "status"
        ]
        == "ambiguous"
    )
    assert (
        language.score_response(primary(combo=(5, 1, 2)), "一只豚子正在水中游动，它的颜色是黄色。")[
            "status"
        ]
        != "recovered"
    )
    assert (
        language.score_response(
            primary(combo=(0, 6, 2)), "一只老鼠正在水中游动，一只猫的颜色是红色。"
        )["status"]
        != "recovered"
    )


def test_sentence_retain_slot_is_correct_without_accepting_wrong_animals() -> None:
    bird = data._retain_item("dev_retain", "spanish", 1, (0, 2, 0))
    assert (
        language.score_response(
            bird, "The animal described in the sentence is a bird.", retain=True
        )["status"]
        == "recovered"
    )
    assert (
        language.score_response(
            bird, "The animal described in the sentence is a cat.", retain=True
        )["status"]
        == "not_recovered"
    )
    mouse = data._retain_item("dev_retain", "spanish", 0, (0, 6, 0))
    assert language.score_response(mouse, "Rat.", retain=True)["status"] == "not_recovered"
