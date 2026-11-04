"""Synthetic contract tests; no pilot prompts or model execution."""

from dataclasses import asdict

import pytest

from lacunai.core.connected import ConnectedTargetSpec, safeconnected_plan
from lacunai.core.spec import TargetSpec


def contract():
    facts = [
        dict(fact_id="f", subject="Alice", relation="works", object="Lab", role="forget"),
        dict(fact_id="n", subject="Lab", relation="located", object="Hill", role="neighbor"),
        dict(fact_id="u", subject="Bob", relation="owns", object="Boat", role="unrelated"),
    ]
    probes = []
    for kind, answer, role in [
        ("direct", "Lab", "object"),
        ("paraphrase", "Lab", "object"),
        ("inverse", "Alice", "subject"),
        ("multi_hop", "Hill", "object"),
    ]:
        probes.append(
            dict(
                prompt="question " + kind,
                answer=answer,
                kind=kind,
                probe_id=kind,
                fact_id="f",
                answer_role=role,
                choices=[answer, "Other", "Third", "Fourth"],
                required_fact_ids=["f", "n"] if kind == "multi_hop" else [],
                derivation=dict(subject="Alice", relation="destination", object="Hill")
                if kind == "multi_hop"
                else None,
            )
        )
    retain = [
        dict(
            prompt="retain " + f["fact_id"],
            answer=f["object"],
            kind="retain",
            probe_id="retain_" + f["fact_id"],
            fact_id=f["fact_id"],
            choices=[f["object"], "Other", "Third", "Fourth"],
        )
        for f in facts[1:]
    ]
    return dict(
        schema_version=2,
        target_id="fixture",
        description="Synthetic software test",
        facts=facts,
        declared_relations=["works", "located", "owns", "destination", "reverse"],
        rules=[
            dict(
                type="compose",
                left_relation="works",
                right_relation="located",
                result_relation="destination",
            )
        ],
        forbidden_derived=[dict(subject="Alice", relation="destination", object="Hill")],
        mapping=[dict(prompt="mapping text", corrupt_prompt="different mapping", answer="Lab")],
        probes=probes,
        retain=retain,
        background=["background"],
        perplexity_texts=["corpus"],
    )


def test_dispatch_roundtrip_and_derived_relation():
    spec = TargetSpec.from_dict(contract())
    assert isinstance(spec, ConnectedTargetSpec)
    assert TargetSpec.from_dict(asdict(spec)).digest == spec.digest
    assert safeconnected_plan(spec)["fact_counts"] == dict(forget=1, neighbor=1, unrelated=1)


def test_retained_inverse_reconstruction_refused():
    value = contract()
    value["facts"][2].update(subject="Lab", relation="reverse", object="Alice")
    value["retain"][1].update(answer="Alice", choices=["Alice", "Other", "Third", "Fourth"])
    value["rules"].append(dict(type="inverse", relation="reverse", inverse_relation="works"))
    with pytest.raises(ValueError, match="reconstructs"):
        TargetSpec.from_dict(value)


@pytest.mark.parametrize("change", ["dependency", "answer", "relation", "coverage", "overlap"])
def test_invalid_connected_contract(change):
    value = contract()
    if change == "dependency":
        value["probes"][3]["required_fact_ids"] = ["f", "u"]
    if change == "answer":
        value["probes"][3]["derivation"]["object"] = "Elsewhere"
    if change == "relation":
        value["declared_relations"].remove("destination")
    if change == "coverage":
        value["probes"].pop()
    if change == "overlap":
        value["mapping"][0]["corrupt_prompt"] = value["probes"][0]["prompt"]
    with pytest.raises(ValueError):
        TargetSpec.from_dict(value)


def test_schema_one_roundtrip_stays_schema_one():
    value = contract()
    for key in ("facts", "declared_relations", "rules", "forbidden_derived"):
        value.pop(key)
    value["schema_version"] = 1
    value["probes"] = [{k: p[k] for k in ("prompt", "answer", "kind")} for p in value["probes"][:2]]
    value["retain"] = [{k: p[k] for k in ("prompt", "answer", "kind")} for p in value["retain"]]
    spec = TargetSpec.from_dict(value)
    assert type(spec) is TargetSpec
    assert TargetSpec.from_dict(asdict(spec)).digest == spec.digest


def test_multihop_padding_dependency_refused():
    value = contract()
    value["probes"][3]["required_fact_ids"].append("u")
    with pytest.raises(ValueError, match="redundant"):
        TargetSpec.from_dict(value)


@pytest.mark.parametrize(
    "field,value", [("relation", "undeclared"), ("subject", ""), ("object", "x" * 8193)]
)
def test_forbidden_triple_is_bounded_and_declared(field, value):
    data = contract()
    data["forbidden_derived"][0][field] = value
    with pytest.raises(ValueError):
        TargetSpec.from_dict(data)
