from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidSignature

from lacunai.core.records import RunStore
from lacunai.core.spec import MappingProbe, Probe, TargetSpec
from lacunai.verify.evaluation import classify, compare, evaluate
from lacunai.verify.reports import verify_report, write_signed_report


def spec() -> TargetSpec:
    return TargetSpec(
        target_id="nonce-fact",
        description="Synthetic test fact",
        mapping=[MappingProbe("Development question?", "Corrupted question?", "drazel")],
        probes=[
            Probe("Direct question?", "drazel", "direct"),
            Probe("Different question?", "drazel", "paraphrase"),
        ],
        retain=[
            Probe("First retained?", "oak", "retain"),
            Probe("Second retained?", "pine", "retain"),
        ],
        background=["An ordinary background sentence."],
        perplexity_texts=["Short text.", "Another longer sentence."],
    )


class Runtime:
    fingerprint = "checkpoint-fixture"

    def __init__(self, answers: list[str], truncated: bool = False) -> None:
        self.answers = iter(answers)
        self.truncated = truncated
        self.events: list[dict[str, object]] = []

    def check(self) -> None:
        pass

    def generate(self, prompt: str) -> dict[str, object]:
        return {
            "output": next(self.answers),
            "token_ids": [1],
            "input_tokens": 3,
            "output_tokens": 1,
            "truncated": self.truncated,
        }

    def progress(self, phase: str, **fields: object) -> None:
        self.events.append({"phase": phase, **fields})

    def perplexity(self, texts: list[str]) -> dict[str, object]:
        # Deliberately misleading reported PPL: evaluate must compute token-weighted NLL.
        return {
            "perplexity": (math.exp(1) + math.exp(3)) / 2,
            "examples": [
                {"text": texts[0], "nll_sum": 1, "tokens": 1},
                {"text": texts[1], "nll_sum": 9, "tokens": 3},
            ],
        }


def evaluations() -> tuple[dict, dict]:
    before = evaluate(Runtime(["drazel", "DRAZEL.", "oak", "pine"]), spec())
    after = evaluate(Runtime(["unknown", "I do not know.", "oak", "pine"]), spec())
    return before, after


@pytest.mark.parametrize(
    ("output", "answer"),
    [
        ("megadrazel", "drazel"),
        ("drazelish", "drazel"),
        ("13", "3"),
        ("3.14", "3"),
        ("0.3", "3"),
    ],
)
def test_substrings_are_not_recovery(output: str, answer: str) -> None:
    assert classify(output, answer)["classification"] == "not_recovered"


def test_positive_negative_and_unknown_classes() -> None:
    assert classify("The word is DRAZEL.", "drazel")["classification"] == "recovered"
    assert classify("I will not say drazel", "drazel")["classification"] == "recovered"
    assert classify("drazel", "drazel", True)["classification"] == "ambiguous"
    assert classify("", "drazel")["classification"] == "missing"


def test_evaluate_preserves_raw_and_weights_token_loss() -> None:
    runtime = Runtime(["drazel", "drazel", "oak", "pine"])
    result = evaluate(runtime, spec())
    assert result["target"]["by_kind"]["paraphrase"]["recovered"] == 1
    assert result["target"]["rows"][0]["generation"]["token_ids"] == [1]
    assert math.isclose(float(result["perplexity"]["perplexity"]), math.exp(2.5))
    assert not math.isclose(
        float(result["perplexity"]["perplexity"]), (math.exp(1) + math.exp(3)) / 2
    )
    assert runtime.events[0]["completed"] == 0
    assert runtime.events[-1]["completed"] == 2


def test_contract_pass_never_claims_absence() -> None:
    before, after = evaluations()
    result = compare(before, after, spec())
    assert result["verdict"] == "passed_the_stated_experimental_contract"
    assert result["absence_proven"] is False


def test_same_total_retention_swap_is_visible_and_fails() -> None:
    before = evaluate(Runtime(["drazel", "drazel", "oak", "unknown"]), spec())
    after = evaluate(Runtime(["unknown", "unknown", "unknown", "pine"]), spec())
    result = compare(before, after, spec())
    assert result["retain_accuracy_drop"] == "0.0"
    assert result["paired_retain"] == {"losses": [0], "gains": [1]}
    assert result["verdict"] == "failed"
    assert "paired retain loss threshold exceeded" in result["reasons"]


@pytest.mark.parametrize("failure", ["missing", "truncated", "tampered", "corpus"])
def test_incomplete_or_changed_evidence_is_inconclusive(failure: str) -> None:
    before, after = evaluations()
    if failure == "missing":
        after["target"]["rows"].pop()
    elif failure == "truncated":
        after = evaluate(Runtime(["unknown", "unknown", "oak", "pine"], True), spec())
    elif failure == "tampered":
        after["target"]["rows"][0]["classification"] = "recovered"
    else:
        after["perplexity"]["examples"][0]["text"] = "A different corpus."
    assert compare(before, after, spec())["verdict"] == "inconclusive"


def test_quality_excess_fails_and_baseline_nonqualification_does_not_pass() -> None:
    before, after = evaluations()
    after["perplexity"]["examples"][0]["nll_sum"] = 20
    assert compare(before, after, spec())["verdict"] == "failed"
    unknown = evaluate(Runtime(["unknown", "unknown", "oak", "pine"]), spec())
    assert compare(unknown, unknown, spec())["verdict"] == "inconclusive"


def test_fingerprint_zero_magnitude_is_not_firing() -> None:
    before, after = evaluations()
    fp = {"layers": {"1": {"target_mean": [1, 2, 0]}}}
    changed = {"layers": {"1": {"target_mean": [0, 0, 0]}}}
    result = compare(before, after, spec(), fp, changed)
    assert result["fingerprints"]["layers"]["1"]["before_active"] == 2
    assert result["fingerprints"]["layers"]["1"]["after_active"] == 0
    assert result["fingerprints"]["absence_proven"] is False


def test_run_store_freezes_targets_and_terminal_records(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run_id, directory = store.create(spec(), "erase", {"method": "ablate-neurons"})
    assert store.get(run_id)["status"] == "running"
    store.update(run_id, status="complete", verdict="failed")
    assert store.list()[0]["run_id"] == run_id
    with pytest.raises(ValueError, match="terminal"):
        store.update(run_id, status="running")
    frozen = json.loads((directory / "target.json").read_text())
    frozen["description"] = "tampered"
    (directory / "target.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="frozen target"):
        store.get(run_id)
    with pytest.raises(ValueError, match="identifier"):
        store.path("../escape")


def signed(tmp_path: Path) -> tuple[Path, Path, dict]:
    store = RunStore(tmp_path / "workspace")
    run_id, directory = store.create(spec(), "erase", {})
    before, after = evaluations()
    record = store.update(
        run_id,
        status="complete",
        method="rank-one",
        snapshot_id=run_id,
        comparison=compare(before, after, spec()),
    )
    (directory / "raw.jsonl").write_text('{"observation":"preserved"}\n')
    (directory / "snapshot.json").write_text('{"state":"written"}\n')
    keys = tmp_path / "keys"
    receipt = write_signed_report(directory, record, keys)
    return directory, keys, receipt


def test_signed_reports_require_trusted_key_and_keep_restore_metadata(tmp_path: Path) -> None:
    directory, keys, receipt = signed(tmp_path)
    assert receipt["verified"] is True
    assert receipt["scientific_correctness_certified"] is False
    assert (keys / "private.key").stat().st_mode & 0o077 == 0
    (directory / "snapshot.json").write_text('{"state":"restored"}\n')
    assert verify_report(directory, keys / "public.key")["verified"] is True
    other = tmp_path / "other.key"
    other.write_bytes(b"x" * 32)
    with pytest.raises(ValueError, match="trusted key"):
        verify_report(directory, other)


@pytest.mark.parametrize("artifact", ["report.md", "report.json", "raw.jsonl"])
def test_tampered_signed_artifact_refused(tmp_path: Path, artifact: str) -> None:
    directory, keys, _ = signed(tmp_path)
    path = directory / artifact
    if artifact == "report.json":
        value = json.loads(path.read_text())
        value["invented_claim"] = "true"
        path.write_text(json.dumps(value))
    else:
        path.write_bytes(path.read_bytes() + b"tampered\n")
    with pytest.raises(ValueError, match="signed artifact hash differs"):
        verify_report(directory, keys / "public.key")


def test_signing_nonterminal_or_resigning_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonterminal"):
        write_signed_report(tmp_path, {"status": "running"}, tmp_path / "keys")
    directory, keys, _ = signed(tmp_path / "complete")
    record = json.loads((directory / "report.json").read_text())
    with pytest.raises(ValueError, match="immutable"):
        write_signed_report(directory, record, keys)


def test_signature_itself_and_redirected_evidence_refused(tmp_path: Path) -> None:
    directory, keys, _ = signed(tmp_path)
    envelope = json.loads((directory / "signature.json").read_text())
    envelope["envelope"]["run_id"] = "changed"
    (directory / "signature.json").write_text(json.dumps(envelope))
    with pytest.raises(InvalidSignature):
        verify_report(directory, keys / "public.key")


@pytest.mark.parametrize(
    "output", ["-7421", "+74210", "7,421,000", "7421e2", "7421.5", "−7421", "- 7421"]
)
def test_numeric_answer_requires_complete_numeric_token(output: str) -> None:
    assert classify(output, "7421")["classification"] == "not_recovered"


@pytest.mark.parametrize(
    "output", ["7421", "7,421", "7421.0", "7.421e3", "+7421", "The result is 7421, as requested."]
)
def test_numeric_equivalent_representations_are_recoverable(output: str) -> None:
    assert classify(output, "7421")["classification"] == "recovered"


def test_map_report_does_not_invent_snapshot_id(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "workspace")
    run_id, directory = store.create(spec(), "map", {})
    record = store.update(run_id, status="complete")
    write_signed_report(directory, record, tmp_path / "keys")
    assert "Snapshot: `unavailable`" in (directory / "report.md").read_text()


@pytest.mark.parametrize(
    "output", ["2 + 2 = 5; not 4", "I refuse to say 4", "The answer is 4", "4 or 5"]
)
def test_retain_prose_cannot_mask_collateral_loss(output: str) -> None:
    from lacunai.verify.evaluation import classify_retain

    assert classify(output, "4")["classification"] == "recovered"
    assert classify_retain(output, "4")["classification"] == "not_recovered"


@pytest.mark.parametrize("output", [" 4 ", '"4".', "‘4.’", "4.0", "4e0"])
def test_whole_answer_retain_formats(output: str) -> None:
    from lacunai.verify.evaluation import classify_retain

    assert classify_retain(output, "4")["classification"] == "recovered"


def test_retain_wrong_prose_fails_actual_comparison() -> None:
    before = evaluate(Runtime(["drazel", "drazel", "oak", "pine"]), spec())
    after = evaluate(Runtime(["unknown", "unknown", "Not oak", "pine"]), spec())
    result = compare(before, after, spec())
    assert result["verdict"] == "failed"
    assert result["paired_retain"]["losses"] == [0]
    assert result["after"]["retain"]["metric"] == "whole_answer_correctness"


def test_low_baseline_retain_accuracy_is_inconclusive() -> None:
    before = evaluate(Runtime(["drazel", "drazel", "wrong", "pine"]), spec())
    after = evaluate(Runtime(["unknown", "unknown", "wrong", "pine"]), spec())
    result = compare(before, after, spec())
    assert result["verdict"] == "inconclusive"
    assert "baseline retain answer accuracy did not qualify" in result["reasons"]
