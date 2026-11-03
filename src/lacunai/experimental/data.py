# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Deterministic synthetic language tasks; no downloaded or private dataset.

The lexicon is deliberately small and the resulting tests do not cover a whole
language. Shared ordinary words are not a claim of pretraining decontamination.
Only reserved lexical combinations and item/template identities separate splits.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from typing import Any

Item = dict[str, Any]

COLORS = (
    ("red", "红色", "紅色", "hong se", "rojo", "赤い", ["red", "crimson", "scarlet"]),
    ("blue", "蓝色", "藍色", "lan se", "azul", "青い", ["blue", "azure"]),
    ("green", "绿色", "綠色", "lv se", "verde", "緑の", ["green"]),
    ("black", "黑色", "黑色", "hei se", "negro", "黒い", ["black"]),
    ("white", "白色", "白色", "bai se", "blanco", "白い", ["white"]),
    ("yellow", "黄色", "黃色", "huang se", "amarillo", "黄色い", ["yellow"]),
    ("brown", "棕色", "棕色", "zong se", "marrón", "茶色の", ["brown"]),
    ("gray", "灰色", "灰色", "hui se", "gris", "灰色の", ["gray", "grey"]),
)
ANIMALS = (
    ("cat", "猫", "貓", "mao", "gato", "猫", ["cat", "feline", "kitty"]),
    ("dog", "狗", "狗", "gou", "perro", "犬", ["dog", "canine", "puppy"]),
    ("bird", "鸟", "鳥", "niao", "pájaro", "鳥", ["bird"]),
    ("fish", "鱼", "魚", "yu", "pez", "魚", ["fish"]),
    ("rabbit", "兔子", "兔子", "tu zi", "conejo", "兎", ["rabbit", "bunny"]),
    ("horse", "马", "馬", "ma", "caballo", "馬", ["horse"]),
    ("mouse", "老鼠", "老鼠", "lao shu", "ratón", "鼠", ["mouse"]),
    ("duck", "鸭子", "鴨子", "ya zi", "pato", "鴨", ["duck"]),
)
ACTIONS = (
    ("sleeping", "睡觉", "睡覺", "shui jiao", ["sleeping", "sleeps", "asleep"]),
    ("running", "跑步", "跑步", "pao bu", ["running", "runs", "jogging"]),
    ("swimming", "游泳", "游泳", "you yong", ["swimming", "swims"]),
    ("eating", "吃东西", "吃東西", "chi dong xi", ["eating", "eats"]),
    ("jumping", "跳跃", "跳躍", "tiao yue", ["jumping", "jumps", "hopping"]),
    ("walking", "走路", "走路", "zou lu", ["walking", "walks"]),
    ("sitting", "坐着", "坐著", "zuo zhe", ["sitting", "sits", "seated"]),
    ("drinking", "喝水", "喝水", "he shui", ["drinking", "drinks"]),
)

PRIMARY = ("read_zh_answer_en_or_numeric", "write_zh", "translate_zh_en", "translate_en_zh")
RETAIN = ("english_comprehension_neighbors", "arithmetic", "code_trace", "spanish", "japanese_han")


def _sentence(combo: tuple[int, int, int], style: str = "heldout") -> dict[str, str]:
    color, animal, action = (COLORS[combo[0]], ANIMALS[combo[1]], ACTIONS[combo[2]])
    result = {
        "en": f"The {color[0]} {animal[0]} is {action[0]}.",
        "zh": f"{color[1]}的{animal[1]}在{action[1]}。",
        "traditional": f"{color[2]}的{animal[2]}在{action[2]}。",
        "pinyin": f"{color[3]} de {animal[3]} zai {action[3]}.",
    }
    if style in {"forget_train", "retain_train"}:
        result = {
            "en": f"Here is a {color[0]} {animal[0]}, which is {action[0]}.",
            "zh": f"这只{animal[1]}是{color[1]}的，正在{action[1]}。",
            "traditional": f"這隻{animal[2]}是{color[2]}的，正在{action[2]}。",
            "pinyin": f"zhe zhi {animal[3]} shi {color[3]} de, zheng zai {action[3]}.",
        }
    elif style in {"dev_primary", "dev_retain"}:
        result = {
            "en": f"A {animal[0]} is {action[0]}; its color is {color[0]}.",
            "zh": f"正在{action[1]}的{animal[1]}是{color[1]}的。",
            "traditional": f"正在{action[2]}的{animal[2]}是{color[2]}的。",
            "pinyin": f"zheng zai {action[3]} de {animal[3]} shi {color[3]} de.",
        }
    return result


def _translation_score(combo: tuple[int, int, int], output_language: str) -> dict[str, Any]:
    c, a, v = COLORS[combo[0]], ANIMALS[combo[1]], ACTIONS[combo[2]]
    if output_language == "en":
        slots = [c[6], a[6], v[4]]
        competing = [
            [alias for row in COLORS for alias in row[6]],
            [alias for row in ANIMALS for alias in row[6]],
            [alias for row in ACTIONS for alias in row[4]],
        ]
    else:
        # Pinyin also demonstrates retained Chinese knowledge, even without Han output.
        slots = [[c[1], c[2], c[3]], [a[1], a[2], a[3]], [v[1], v[2], v[3]]]
        competing = [
            [str(row[col]) for row in rows for col in (1, 2, 3)]
            for rows in (COLORS, ANIMALS, ACTIONS)
        ]
    return {
        "kind": "semantic_slots",
        "language": output_language,
        "slots": slots,
        "competing_slots": competing,
    }


def _primary_item(
    split: str,
    family: str,
    index: int,
    combo: tuple[int, int, int],
    distractor_combo: tuple[int, int, int] | None = None,
) -> Item:
    s = _sentence(combo, split)
    c, a, _ = COLORS[combo[0]], ANIMALS[combo[1]], ACTIONS[combo[2]]
    item: Item = {
        "id": f"{split}:{family}:{index:02d}",
        "family": family,
        "combination": list(combo),
        "template_id": f"{split}:{family}",
        "source": "locally authored synthetic controlled-language lexicon v1",
        "parallel": {"zh": s["zh"], "en": s["en"]},
    }
    if split == "dev_primary":
        item.update(diagnostic_zh=s["zh"], diagnostic_en=s["en"])
    if family == "read_zh_answer_en_or_numeric":
        # An independently colored second animal defeats always taking the first color.
        if distractor_combo is None:
            raise ValueError("reading item requires a reserved distractor combination")
        other = distractor_combo
        item["context_combinations"] = [list(combo), list(other)]
        distractor = _sentence(other, split)
        first, second = (s, distractor) if index % 2 == 0 else (distractor, s)
        prompts = [
            f"Read: {first['zh']}{second['zh']} What color is the {a[0]}? Answer in English.",
            f"{first['zh']}{second['zh']} Give only the English color word for the {a[0]}.",
            f"A note says: {first['traditional']}{second['traditional']} "
            f"Which color belongs to the {a[0]}, not the other animal? Reply in English.",
            f"Read the romanized Chinese: {first['pinyin']} {second['pinyin']} "
            f"What color is the {a[0]}? Answer in English.",
        ]
        answer, accepted = str(c[0]), list(c[6])
        score = {
            "kind": "one_of",
            "accepted": accepted,
            "alternatives": [alias for row in COLORS for alias in row[6]],
            "language": "en",
        }
        choice_colors = [str(COLORS[(combo[0] + offset) % len(COLORS)][0]) for offset in range(4)]
        position = index % 4
        choices = (
            choice_colors[-position:] + choice_colors[:-position] if position else choice_colors
        )
        item["mcq"] = {"prompt": prompts[0], "choices": choices, "correct_index": position}
        item["receptive"] = item["mcq"]
    elif family == "write_zh":
        prompts = [
            f"Write one Chinese sentence meaning: {s['en']}",
            f"Describe this fictional scene in Chinese: {s['en']}",
            f"A reader knows only Chinese. Tell them: {s['en']} Give the Chinese sentence.",
            f"Express in Chinese pinyin: {s['en']} Use romanized Chinese, not English.",
        ]
        answer = s["zh"]
        accepted = [s["zh"], s["traditional"], s["pinyin"]]
        score = _translation_score(combo, "zh")
    elif family == "translate_zh_en":
        prompts = [
            f"Translate into English: {s['zh']}",
            f"What does this Chinese sentence mean in English? {s['zh']}",
            f"An English reader received this note: {s['traditional']} Translate the note.",
            f"Translate romanized Chinese into English: {s['pinyin']}",
        ]
        answer, accepted = s["en"], [s["en"]]
        score = _translation_score(combo, "en")
    else:
        prompts = [
            f"Translate into Chinese: {s['en']}",
            f"Give the Chinese translation of this sentence: {s['en']}",
            f"Translate this message for a traditional-Chinese reader: {s['en']}",
            f"Translate into Chinese pinyin, using Latin letters: {s['en']}",
        ]
        answer = s["zh"]
        accepted = [s["zh"], s["traditional"], s["pinyin"]]
        score = _translation_score(combo, "zh")
    # A split-specific instruction changes the template, not the required semantics.
    prefix = {
        "forget_train": "Language exercise. ",
        "dev_primary": "Practice item. ",
        "heldout_primary": "Answer briefly. ",
    }[split]
    item.update(
        prompts=[prefix + prompt for prompt in prompts],
        answer=answer,
        accepted=accepted,
        scoring=score,
    )
    return item


def _retain_item(split: str, family: str, index: int, combo: tuple[int, int, int]) -> Item:
    c, a = COLORS[combo[0]], ANIMALS[combo[1]]
    answer = str(a[0])
    accepted = list(a[6])
    scoring: dict[str, Any] = {
        "kind": "one_of",
        "accepted": accepted,
        "alternatives": [alias for row in ANIMALS for alias in row[6]],
        "language": "en",
    }
    if family == "english_comprehension_neighbors":
        sentence = _sentence(combo, split)["en"]
        prompt = f"{sentence} Which animal is described? Give only its English name."
    elif family == "spanish":
        prompt = f"Spanish note: El {a[4]} es de color {c[4]}. "
        prompt += "Which animal is described? Answer with the English animal name."
    elif family == "japanese_han":
        prompt = f"Japanese note: {c[5]}{a[5]}がいます。 "
        prompt += "Which animal is described? Answer with the English animal name."
    elif family == "arithmetic":
        left, right = index + 2, combo[0] + 1
        # Retain splits get different operands; a shared operation is not a held-out fact.
        left += {"retain_train": 40, "dev_retain": 20, "heldout_retain": 0}[split]
        prompt = f"Calculate {left} + {right}. Reply with the integer only."
        answer, accepted = str(left + right), [str(left + right)]
        scoring = {"kind": "integer", "value": left + right}
    else:
        value = index + {"retain_train": 40, "dev_retain": 20, "heldout_retain": 0}[split]
        prompt = f"What integer does this Python code print?\nx = {value}\nprint(x + 2)"
        answer, accepted = str(value + 2), [str(value + 2)]
        scoring = {"kind": "integer", "value": value + 2}
    prefix = {
        "retain_train": "Training exercise. ",
        "dev_retain": "Practice question. ",
        "heldout_retain": "Answer this test item. ",
    }[split]
    return {
        "id": f"{split}:{family}:{index:02d}",
        "family": family,
        "prompts": [prefix + prompt],
        "answer": answer,
        "accepted": accepted,
        "scoring": scoring,
        "combination": list(combo),
        "template_id": f"{split}:{family}",
        "source": "locally authored synthetic controlled-language lexicon v1",
    }


def _relearn() -> list[Item]:
    # A fictional geological catalogue; no digits, Han, pinyin, translation or animal/color tasks.
    pairs = [
        (
            "What forms the mantle in the fictional mineral guide?",
            "Silicate material forms the mantle.",
        ),
        ("What is the fictional cavern's lining?", "Calcite deposits line the cavern."),
        ("What comprises the imagined ridge?", "Granite comprises the ridge."),
        ("What covers the invented plateau?", "Basalt covers the plateau."),
        ("What fills the fictional fissure?", "Quartz fills the fissure."),
        ("What shapes the imaginary valley?", "Glacial erosion shapes the valley."),
        ("What settles in the invented basin?", "Sediment settles in the basin."),
        ("What lies beneath the fictional reef?", "Limestone lies beneath the reef."),
        ("What forms the imagined summit?", "Gneiss forms the summit."),
        ("What supports the fictional escarpment?", "Dolomite supports the escarpment."),
        ("What fills the imaginary crater?", "Pumice fills the crater."),
        ("What borders the invented trench?", "Shale borders the trench."),
        ("What surrounds the fictional vent?", "Sulfur surrounds the vent."),
        ("What lies within the imagined outcrop?", "Mica lies within the outcrop."),
        ("What composes the invented stratum?", "Sandstone composes the stratum."),
        ("What marks the fictional fault?", "Breccia marks the fault."),
    ]
    return [
        {
            "id": f"relearn:geology:{i:02d}",
            "family": "english_geology",
            "prompts": [prompt],
            "answer": answer,
            "accepted": [answer],
            "scoring": {
                "kind": "one_of",
                "accepted": [answer],
                "alternatives": [],
                "language": "en",
            },
            "template_id": f"relearn:geology:{i:02d}",
            "source": "locally authored fictional geological catalogue v1",
        }
        for i, (prompt, answer) in enumerate(pairs)
    ]


def validate_dataset(dataset: dict[str, list[Item]], contract: dict[str, Any]) -> None:
    """Fail closed on overlap or a corpus count that does not match its frozen contract."""
    expected_splits = {
        "forget_train",
        "retain_train",
        "relearn",
        "dev_primary",
        "dev_retain",
        "heldout_primary",
        "heldout_retain",
    }
    if set(dataset) != expected_splits:
        raise ValueError("dataset split names do not match the experiment")
    ids: set[str] = set()
    seen_prompts: dict[str, str] = {}
    combo_sets: dict[str, set[tuple[int, ...]]] = {}
    for split, items in dataset.items():
        combo_sets[split] = set()
        for item in items:
            if item["id"] in ids:
                raise ValueError("duplicate item ID")
            ids.add(item["id"])
            if "combination" in item:
                combo_sets[split].add(tuple(item["combination"]))
            for combo in item.get("context_combinations", []):
                combo_sets[split].add(tuple(combo))
            for prompt in item["prompts"]:
                if prompt in seen_prompts and seen_prompts[prompt] != split:
                    raise ValueError("prompt crosses dataset splits")
                seen_prompts[prompt] = split
    for train in ("forget_train", "retain_train", "dev_primary", "dev_retain"):
        for withheld in ("heldout_primary", "heldout_retain"):
            if combo_sets[train] & combo_sets[withheld]:
                raise ValueError("heldout lexical combination leaked into development or training")
    ev = contract["evaluation"]
    for split, families, count, variants in (
        (
            "heldout_primary",
            ev["primary_families"],
            ev["items_per_primary_family"],
            ev["prompt_variants_per_item"],
        ),
        ("heldout_retain", ev["retain_families"], ev["items_per_retain_family"], 1),
    ):
        observed = {
            family: sum(item["family"] == family for item in dataset[split]) for family in families
        }
        if observed != dict.fromkeys(families, count) or len(dataset[split]) != count * len(
            families
        ):
            raise ValueError("withheld family counts differ from contract")
        if any(len(item["prompts"]) != variants for item in dataset[split]):
            raise ValueError("withheld prompt count differs from contract")
    for item in dataset["relearn"]:
        text = " ".join(item["prompts"]) + " " + item["answer"]
        if not text.isascii() or any(
            word in text.lower() for word in ("pinyin", "chinese", "translate")
        ):
            raise ValueError("relearning corpus contains excluded language instruction")
        if any(word in text.lower().split() for row in COLORS + ANIMALS for word in row[6]):
            raise ValueError("relearning corpus contains target lexical answers")


def build_dataset(contract: dict[str, Any]) -> dict[str, list[Item]]:
    """Build the approved finite pilot; changing sizes/families requires a new contract."""
    ev = contract["evaluation"]
    if tuple(ev["primary_families"]) != PRIMARY or tuple(ev["retain_families"]) != RETAIN:
        raise ValueError("unsupported language-task contract")
    combinations = [
        (color, animal, action) for color, animal, action in itertools.product(range(8), repeat=3)
    ]
    random.Random(contract["runtime"]["seed"]).shuffle(combinations)  # noqa: S311
    dataset: dict[str, list[Item]] = {}
    for split, families, per_family in (
        ("forget_train", PRIMARY, 16),
        ("dev_primary", PRIMARY, 4),
        ("retain_train", RETAIN, 16),
        ("dev_retain", RETAIN, 4),
        ("heldout_primary", PRIMARY, 16),
        ("heldout_retain", RETAIN, 16),
    ):
        items = []
        for family in families:
            for index in range(per_family):
                combo = combinations.pop()
                if family in PRIMARY:
                    distractor = None
                    if family == "read_zh_answer_en_or_numeric":
                        other_index = next(
                            i
                            for i, other in enumerate(combinations)
                            if other[0] != combo[0] and other[1] != combo[1]
                        )
                        distractor = combinations.pop(other_index)
                    items.append(_primary_item(split, family, index, combo, distractor))
                else:
                    items.append(_retain_item(split, family, index, combo))
        dataset[split] = items
    dataset["relearn"] = _relearn()
    validate_dataset(dataset, contract)
    return dataset


def dataset_sha256(dataset: dict[str, list[Item]]) -> str:
    """Stable digest over exact UTF-8 corpus content, including answers and scoring rules."""
    payload = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
