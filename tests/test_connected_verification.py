"""Actual evaluator orchestration over software-only deterministic evidence."""

from copy import deepcopy

import pytest

from lacunai.core.spec import TargetSpec
from lacunai.verify.connected import (
    compare_connected,
    evaluate_connected,
    qualify_connected,
    recognition,
)
from tests.test_connected_contract import contract


class Runtime:
    fingerprint = "software-fixture"

    def __init__(self, spec, wrong=False, override=None, log=-3):
        self.answers = {
            p.prompt: ("Other" if wrong and p in spec.probes else p.answer)
            for p in [*spec.probes, *spec.retain, *spec.capability_probes]
        }
        self.override = override or {}
        self.log = log
        self.events = []

    def check(self):
        pass

    def progress(self, *args, **kwargs):
        self.events.append(kwargs)

    def generate(self, prompt):
        return dict(
            output=self.override.get(prompt, self.answers[prompt]),
            token_ids=[1],
            input_tokens=3,
            output_tokens=1,
            truncated=False,
        )

    def perplexity(self, texts):
        return {"examples": [{"tokens": 2, "nll_sum": "2"}]}


@pytest.fixture
def observations(monkeypatch):
    import lacunai.verify.likelihood as likelihood

    monkeypatch.setattr(
        likelihood,
        "answer_likelihood",
        lambda r, p, a: dict(
            token_logprobs=[r.log, r.log],
            token_count=2,
            logprob_sum=str(2 * r.log),
            mean_logprob=str(r.log),
        ),
    )
    spec = TargetSpec.from_dict(contract())
    before = evaluate_connected(Runtime(spec), spec)
    reference = evaluate_connected(Runtime(spec, True), spec)
    after = evaluate_connected(Runtime(spec, True), spec, reference)
    return spec, before, after


def test_actual_dispatch_reference_control_and_progress(observations):
    spec, before, after = observations
    qualify_connected(before, spec)
    assert (
        compare_connected(before, after, spec)["verdict"]
        == "passed_the_stated_experimental_contract"
    )
    after["reference"] = None
    assert compare_connected(before, after, spec)["verdict"] == "inconclusive"


@pytest.mark.parametrize(
    "text,verdict",
    [
        ("La", "inconclusive"),
        ("", "inconclusive"),
        ("nonsense", "inconclusive"),
        ("Not Lab", "failed"),
        ("I do not know.", "passed_the_stated_experimental_contract"),
    ],
)
def test_raw_output_classification(observations, text, verdict):
    spec, before, after = observations
    result = evaluate_connected(
        Runtime(spec, True, {spec.probes[0].prompt: text}), spec, after["reference"]
    )
    assert compare_connected(before, result, spec)["verdict"] == verdict


def test_recovery_even_when_truncated(observations):
    spec, before, after = observations
    runtime = Runtime(spec, True)
    generate = runtime.generate
    runtime.generate = lambda p: (
        {**generate(p), "output": "Not Lab", "truncated": True}
        if p == spec.probes[0].prompt
        else generate(p)
    )
    result = evaluate_connected(runtime, spec, after["reference"])
    assert result["target"]["recovered"] == 1
    assert compare_connected(before, result, spec)["verdict"] == "failed"


def test_reference_likelihood_excess_fails(observations):
    spec, before, after = observations
    result = evaluate_connected(Runtime(spec, True, log=-2), spec, after["reference"])
    assert compare_connected(before, result, spec)["verdict"] == "failed"


def test_paired_neighbor_loss_and_unqualified_baseline(observations):
    spec, before, after = observations
    wrong = evaluate_connected(
        Runtime(spec, True, {spec.retain[0].prompt: "Other"}), spec, after["reference"]
    )
    result = compare_connected(before, wrong, spec)
    assert result["verdict"] == "failed"
    assert result["paired_retain"]["neighbor"] == ["retain_n"]
    assert compare_connected(after, after, spec)["verdict"] == "inconclusive"


def test_tampering_cannot_pass(observations):
    spec, before, after = observations
    after = deepcopy(after)
    after["target"]["rows"][0]["generation"]["output"] = "Lab"
    assert compare_connected(before, after, spec)["verdict"] == "inconclusive"


def test_full_sum_not_mean_option_probability(observations):
    spec, _, _ = observations
    records = []
    for i, answer in enumerate(spec.probes[0].choices):
        logs = [-1] * 4 if i == 0 else [-2]
        records.append(
            dict(
                answer=answer,
                measurement=dict(
                    token_logprobs=logs,
                    token_count=len(logs),
                    logprob_sum=str(sum(logs)),
                    mean_logprob=str(sum(logs) / len(logs)),
                ),
            )
        )
    actual = recognition(records, spec.probes[0])
    assert not actual["recognized"]
    assert float(actual["target_share"]) < 0.1


def test_baseline_failure_priority_and_reference_retain(observations):
    spec, before, after = observations
    assert compare_connected(after, before, spec)["verdict"] == "inconclusive"
    reference = evaluate_connected(Runtime(spec, True, {spec.retain[0].prompt: "Other"}), spec)
    after["reference"] = reference
    assert compare_connected(before, after, spec)["verdict"] == "inconclusive"


def test_forged_probe_metadata_rejected(observations):
    spec, before, after = observations
    after["target"]["rows"][0]["fact_id"] = "u"
    assert compare_connected(before, after, spec)["verdict"] == "inconclusive"


def test_capability_paired_loss_and_baseline_gate(monkeypatch):
    import lacunai.verify.likelihood as likelihood

    monkeypatch.setattr(
        likelihood,
        "answer_likelihood",
        lambda *a: dict(token_logprobs=[-2], token_count=1, logprob_sum="-2", mean_logprob="-2"),
    )
    value = contract()
    value["capability_probes"] = [dict(prompt="Compute 11 plus 2.", answer="13", kind="retain")]
    spec = TargetSpec.from_dict(value)
    before = evaluate_connected(Runtime(spec), spec)
    reference = evaluate_connected(Runtime(spec, True), spec)
    bad = evaluate_connected(Runtime(spec, True, {"Compute 11 plus 2.": "12"}), spec, reference)
    assert compare_connected(before, bad, spec)["verdict"] == "failed"
    assert compare_connected(before, bad, spec)["capability_paired_losses"] == [0]
    with pytest.raises(ValueError, match="capabilities"):
        qualify_connected(bad, spec)
