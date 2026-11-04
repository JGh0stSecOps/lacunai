"""Connected contracts must reach the shared service's per-fact admission gate."""

import json

import pytest

from lacunai.core.service import Service
from lacunai.core.spec import TargetSpec
from lacunai.verify.connected import evaluate_connected
from tests.test_connected_contract import contract
from tests.test_connected_verification import Runtime
from tests.test_factoid_methods import data


def test_connected_service_qualifies_each_fact(monkeypatch):
    from lacunai.verify import likelihood

    monkeypatch.setattr(
        likelihood,
        "answer_likelihood",
        lambda *args: {
            "token_logprobs": [-2, -2],
            "token_count": 2,
            "logprob_sum": "-4",
            "mean_logprob": "-2",
        },
    )
    spec = TargetSpec.from_dict(contract())
    before = evaluate_connected(Runtime(spec), spec)
    Service._qualify(before, spec)
    for probe in [spec.probes[0], spec.retain[0], spec.retain[1]]:
        bad = evaluate_connected(Runtime(spec, override={probe.prompt: "Other"}), spec)
        with pytest.raises(ValueError, match="baseline fact/family did not qualify"):
            Service._qualify(bad, spec)


def test_connected_gradient_data_cannot_train_capability_evaluation(tmp_path):
    from lacunai.erase.factoid_data import load_factoid_data

    raw_spec = contract()
    raw_spec["capability_probes"] = [
        {"prompt": "Independent capability question", "answer": "answer", "kind": "retain"}
    ]
    spec = TargetSpec.from_dict(raw_spec)
    value = data()
    value["forget"][0]["prompt"] = " Independent  CAPABILITY question "
    path = tmp_path / "development.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="overlaps frozen evaluation"):
        load_factoid_data(path, spec)
