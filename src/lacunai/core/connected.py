"""Frozen connected-fact contracts; finite typed rules, never executable editor code."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, cast

from .spec import MappingProbe, Probe, TargetSpec, Thresholds, text


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def identifier(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", value):
        raise ValueError("invalid connected identifier")


@dataclass(frozen=True)
class Fact:
    fact_id: str
    subject: str
    relation: str
    object: str
    role: str

    def __post_init__(self) -> None:
        identifier(self.fact_id)
        for key in ("subject", "relation", "object"):
            text(getattr(self, key), key)
        if self.role not in {"forget", "neighbor", "unrelated"}:
            raise ValueError("unknown fact role")

    @property
    def triple(self) -> tuple[str, str, str]:
        return tuple(normalized(x) for x in (self.subject, self.relation, self.object))  # type: ignore[return-value]


@dataclass(frozen=True)
class ConnectedProbe(Probe):
    probe_id: str = ""
    fact_id: str = ""
    aliases: list[str] = field(default_factory=list)
    choices: list[str] = field(default_factory=list)
    answer_role: str = "object"
    required_fact_ids: list[str] = field(default_factory=list)
    derivation: dict[str, str] | None = None

    def __post_init__(self) -> None:
        identifier(self.probe_id)
        identifier(self.fact_id)
        if not isinstance(self.required_fact_ids, list) or len(self.required_fact_ids) > 128:
            raise ValueError("invalid required fact IDs")
        for fact_id in self.required_fact_ids:
            identifier(fact_id)
        if len(set(self.required_fact_ids)) != len(self.required_fact_ids):
            raise ValueError("duplicate required fact IDs")
        text(self.prompt, "prompt")
        text(self.answer, "answer")
        if self.kind not in {
            "direct",
            "paraphrase",
            "inverse",
            "multi_hop",
            "adversarial",
            "retain",
        }:
            raise ValueError("unknown connected probe kind")
        if self.answer_role not in {"subject", "object"}:
            raise ValueError("invalid answer role")
        if self.kind == "inverse" and self.answer_role != "subject":
            raise ValueError("inverse probe must answer subject")
        if not isinstance(self.aliases, list) or len(self.aliases) > 16:
            raise ValueError("invalid aliases")
        for value in self.aliases:
            text(value, "alias")
        accepted = [normalized(x) for x in [self.answer, *self.aliases]]
        if len(set(accepted)) != len(accepted):
            raise ValueError("duplicate answer aliases")
        if not isinstance(self.choices, list) or len(self.choices) != 4:
            raise ValueError("four choices required")
        for value in self.choices:
            text(value, "choice")
        if len({normalized(x) for x in self.choices}) != 4:
            raise ValueError("duplicate choices")
        if sum(normalized(x) in accepted for x in self.choices) != 1:
            raise ValueError("exactly one answer choice required")


def retained_closure(
    facts: list[Fact],
    rules: list[dict[str, str]],
    declared_relations: list[str],
    *,
    include_forget: bool = False,
) -> set[tuple[str, str, str]]:
    if not isinstance(rules, list) or len(rules) > 64:
        raise ValueError("invalid reconstruction rule count")
    relations = {normalized(r) for r in declared_relations}
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("rule must be a typed object")
        fields = (
            {"type", "relation", "inverse_relation"}
            if rule.get("type") == "inverse"
            else {"type", "left_relation", "right_relation", "result_relation"}
            if rule.get("type") == "compose"
            else set()
        )
        if not fields or set(rule) != fields:
            raise ValueError("only typed inverse/compose rules supported")
        for key in fields - {"type"}:
            text(rule[key], "rule relation")
            if normalized(rule[key]) not in relations:
                raise ValueError("rule refers to undeclared relation")
    closure = {f.triple for f in facts if include_forget or f.role != "forget"}
    while True:
        added = set(closure)
        for raw in rules:
            rule = {k: normalized(v) for k, v in raw.items()}
            if rule["type"] == "inverse":
                added.update(
                    (o, rule["inverse_relation"], s) for s, r, o in closure if r == rule["relation"]
                )
            else:
                right: dict[str, list[str]] = {}
                for s, r, o in closure:
                    if r == rule["right_relation"]:
                        right.setdefault(s, []).append(o)
                added.update(
                    (s, rule["result_relation"], z)
                    for s, r, o in closure
                    if r == rule["left_relation"]
                    for z in right.get(o, [])
                )
        if len(added) > 4096:
            raise ValueError("reconstruction closure exceeds finite budget")
        if added == closure:
            return closure
        closure = added


@dataclass(frozen=True)
class ConnectedTargetSpec(TargetSpec):
    schema_version: int = 2
    facts: list[Fact] = field(default_factory=list)
    declared_relations: list[str] = field(default_factory=list)
    capability_probes: list[Probe] = field(default_factory=list)
    rules: list[dict[str, str]] = field(default_factory=list)
    forbidden_derived: list[dict[str, str]] = field(default_factory=list)
    required_kinds: list[str] = field(
        default_factory=lambda: ["direct", "paraphrase", "inverse", "multi_hop"]
    )
    abstentions: list[str] = field(default_factory=lambda: ["I do not know."])
    recognition_share_tolerance: float = 0.05
    mean_logprob_tolerance: float = 0.10

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("connected schema version must be 2")
        # Reuse common v1 bounds without changing v1 serialization or validation.
        common = {k: getattr(self, k) for k in TargetSpec.__dataclass_fields__}
        common["schema_version"] = 1
        common["probes"] = [
            Probe(p.prompt, p.answer, p.kind if p.kind not in {"inverse"} else "paraphrase")
            for p in self.probes
        ]
        common["retain"] = [Probe(p.prompt, p.answer, "retain") for p in self.retain]
        TargetSpec(**common)
        if (
            not isinstance(self.facts, list)
            or not 3 <= len(self.facts) <= 128
            or any(not isinstance(f, Fact) for f in self.facts)
        ):
            raise ValueError("invalid connected facts")
        if (
            not isinstance(self.declared_relations, list)
            or not 1 <= len(self.declared_relations) <= 128
        ):
            raise ValueError("declared relations required")
        for relation in self.declared_relations:
            text(relation, "declared relation")
        relations = {normalized(r) for r in self.declared_relations}
        if len(relations) != len(self.declared_relations) or any(
            normalized(f.relation) not in relations for f in self.facts
        ):
            raise ValueError("duplicate or undeclared fact relation")
        ids = {f.fact_id: f for f in self.facts}
        if len(ids) != len(self.facts) or len({f.triple for f in self.facts}) != len(self.facts):
            raise ValueError("duplicate fact identity or triple")
        if {f.role for f in self.facts} != {"forget", "neighbor", "unrelated"}:
            raise ValueError("all three fact roles required")
        if (
            not isinstance(self.required_kinds, list)
            or len(set(self.required_kinds)) != len(self.required_kinds)
            or not {"direct", "paraphrase"} <= set(self.required_kinds)
            or not set(self.required_kinds)
            <= {"direct", "paraphrase", "inverse", "multi_hop", "adversarial"}
        ):
            raise ValueError("invalid required probe families")
        probes = cast(list[ConnectedProbe], [*self.probes, *self.retain])
        if any(not isinstance(p, ConnectedProbe) for p in probes):
            raise ValueError("connected probes required")
        if len({p.probe_id for p in probes}) != len(probes) or len(
            {normalized(p.prompt) for p in probes}
        ) != len(probes):
            raise ValueError("duplicate probe ID or prompt")
        if (
            not isinstance(self.capability_probes, list)
            or len(self.capability_probes) > 64
            or any(type(p) is not Probe for p in self.capability_probes)
        ):
            raise ValueError("invalid capability probes")
        capability_prompts = {normalized(p.prompt) for p in self.capability_probes}
        if len(capability_prompts) != len(self.capability_probes) or capability_prompts & {
            normalized(p.prompt) for p in probes
        }:
            raise ValueError("capability prompts overlap")
        hidden = {normalized(p.prompt) for p in probes} | capability_prompts
        development = [x for p in self.mapping for x in (p.prompt, p.corrupt_prompt)]
        if hidden & {normalized(x) for x in development + self.background + self.perplexity_texts}:
            raise ValueError("connected evaluation/development prompts overlap")
        for p in probes:
            if p.fact_id not in ids or any(i not in ids for i in p.required_fact_ids):
                raise ValueError("unknown probe fact dependency")
            if p.kind == "multi_hop":
                if (
                    len(p.required_fact_ids) < 2
                    or p.fact_id not in p.required_fact_ids
                    or not isinstance(p.derivation, dict)
                    or set(p.derivation) != {"subject", "relation", "object"}
                ):
                    raise ValueError("multi-hop derivation and dependencies required")
                derived = tuple(
                    normalized(text(p.derivation[k], k)) for k in ("subject", "relation", "object")
                )
                available = retained_closure(
                    [ids[i] for i in p.required_fact_ids],
                    self.rules,
                    self.declared_relations,
                    include_forget=True,
                )
                if derived not in available or normalized(p.answer) != normalized(
                    p.derivation[p.answer_role]
                ):
                    raise ValueError("multi-hop answer is not derived from required facts")
                for removed in p.required_fact_ids:
                    without = retained_closure(
                        [ids[i] for i in p.required_fact_ids if i != removed],
                        self.rules,
                        self.declared_relations,
                        include_forget=True,
                    )
                    if derived in without:
                        raise ValueError("redundant multi-hop fact dependency")
            elif (
                p.derivation is not None
                or p.required_fact_ids not in ([], [p.fact_id])
                or normalized(p.answer) != normalized(getattr(ids[p.fact_id], p.answer_role))
            ):
                raise ValueError("probe answer/fact binding differs")
            if p in self.retain and any(ids[i].role == "forget" for i in p.required_fact_ids):
                raise ValueError("retain depends on forget fact")
            if (p in self.probes) != (ids[p.fact_id].role == "forget") or (
                p in self.retain and p.kind != "retain"
            ):
                raise ValueError("probe fact role differs")
        for f in self.facts:
            kinds = {p.kind for p in probes if p.fact_id == f.fact_id}
            if not (set(self.required_kinds) if f.role == "forget" else {"retain"}) <= kinds:
                raise ValueError("per-fact family coverage missing")
        if not isinstance(self.abstentions, list) or not 1 <= len(self.abstentions) <= 16:
            raise ValueError("invalid abstention whitelist")
        for value in self.abstentions:
            text(value, "abstention")
        if (
            type(self.recognition_share_tolerance) not in (int, float)
            or self.recognition_share_tolerance != 0.05
            or type(self.mean_logprob_tolerance) not in (int, float)
            or self.mean_logprob_tolerance != 0.10
        ):
            raise ValueError("exploratory recognition tolerances differ")
        closure = retained_closure(self.facts, self.rules, self.declared_relations)
        if not isinstance(self.forbidden_derived, list) or len(self.forbidden_derived) > 128:
            raise ValueError("invalid forbidden triples")
        forbidden = {f.triple for f in self.facts if f.role == "forget"}
        for row in self.forbidden_derived:
            if not isinstance(row, dict) or set(row) != {"subject", "relation", "object"}:
                raise ValueError("invalid forbidden triple")
            triple = tuple(normalized(text(row[k], k)) for k in ("subject", "relation", "object"))
            if triple[1] not in relations:
                raise ValueError("undeclared forbidden relation")
            forbidden.add((triple[0], triple[1], triple[2]))
        if closure & forbidden:
            raise ValueError("retained graph reconstructs forbidden knowledge")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ConnectedTargetSpec:
        value = dict(value)
        value["facts"] = [Fact(**x) for x in value["facts"]]
        value["capability_probes"] = [Probe(**x) for x in value.get("capability_probes", [])]
        value["mapping"] = [MappingProbe(**x) for x in value["mapping"]]
        for key in ("probes", "retain"):
            value[key] = [ConnectedProbe(**x) for x in value[key]]
        value["thresholds"] = Thresholds(**value.get("thresholds", {}))
        return cls(**value)


def safeconnected_plan(spec: ConnectedTargetSpec) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "spec_sha256": spec.digest,
        "fact_counts": {
            role: sum(f.role == role for f in spec.facts)
            for role in ("forget", "neighbor", "unrelated")
        },
        "probe_counts": {
            "target": len(spec.probes),
            "retain": len(spec.retain),
            "capability": len(spec.capability_probes),
        },
        "required_kinds": list(spec.required_kinds),
        "rules": len(spec.rules),
        "reference_required": True,
        "absence_proven": False,
    }
