"""Screen one root-selected diagnostic row ablation; no automatic candidate search."""

from __future__ import annotations

import argparse
import copy
import math
import re
import sys
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from importlib import import_module
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import chinese_baselines as baseline  # noqa: E402
import chinese_gradient_candidate as gradient  # noqa: E402
import chinese_iteration as iteration  # noqa: E402
import chinese_rmu_candidate as rmu  # noqa: E402
from lacunai.core.model import HFRuntime, active_phase, checkpoint_inventory, progress  # noqa: E402
from lacunai.core.spec import RuntimeConfig, digest  # noqa: E402
from lacunai.erase import language as operator  # noqa: E402
from lacunai.experimental import tensors  # noqa: E402
from lacunai.experimental.artifacts import atomic_json  # noqa: E402
from lacunai.experimental.resources import Watchdog, host_sample, tree_bytes  # noqa: E402
from lacunai.restore.checkpoints import prepare_snapshot  # noqa: E402
from lacunai.verify import language as evaluation  # noqa: E402

BACKENDS = {
    "distribution_row_ablation": "lacunai.map.language_distribution",
    "relevance_row_ablation": "lacunai.map.language_relevance",
}
PAYLOAD_KINDS = {
    "distribution_row_ablation": "histograms",
    "relevance_row_ablation": "sensitivities",
}


def validate_diagnosis(spec: dict[str, Any], diagnosis: dict[str, Any]) -> Any:
    validate_spec(spec)
    mapper = import_module(BACKENDS[spec["method"]])
    if (
        type(diagnosis.get("schema_version")) is not int
        or diagnosis.get("schema_version") != mapper.VERSION
        or diagnosis.get("provenance") != mapper.PROVENANCE
    ):
        raise ValueError("diagnosis schema/provenance does not match selected backend")
    return mapper


def validate_spec(spec: dict[str, Any]) -> None:
    if (
        set(spec) != {"schema_version", "method", "top_k", "execution_fraction"}
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or type(spec["method"]) is not str
        or spec["method"] not in BACKENDS
        or type(spec["top_k"]) is not int
        or spec["top_k"] < 1
        or type(spec["execution_fraction"]) not in (int, float)
        or not math.isfinite(spec["execution_fraction"])
        or spec["execution_fraction"] not in operator.FRACTIONS
    ):
        raise ValueError("invalid root-selected row edit specification")


def pinned_payload(path: Path, pin: str) -> None:
    if not re.fullmatch("[0-9a-f]{64}", pin) or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("invalid histogram path or hash pin")
    if not path.is_file() or path.stat().st_size > 512 * 1024**2:
        raise ValueError("histogram artifact missing or oversized")
    if tensors.file_hash(path) != pin:
        raise ValueError("histogram bytes differ from pin")


def rescore_before(before: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    legacy = copy.deepcopy(evaluation.CONTRACT)
    legacy.pop("primary_baseline_unknowns", None)
    legacy.update(name="chinese-literacy-zero-recovery-v2", minimum_receptive_accuracy_drop="0.50")
    if not before.get("development") or before.get("contract") not in (legacy, evaluation.CONTRACT):
        raise ValueError("baseline contract is not a supported versioned development contract")
    ppl = before.get("perplexity", {})
    try:
        value = Decimal(str(ppl.get("perplexity")))
    except InvalidOperation as error:
        raise ValueError("invalid baseline perplexity") from error
    texts = [item["prompts"][0] + "\n" + item["answer"] for item in dataset["dev_retain"]]
    if (
        not value.is_finite()
        or value <= 0
        or type(ppl.get("tokens")) is not int
        or ppl["tokens"] <= 0
        or ppl.get("corpus_split") != "dev_retain"
        or ppl.get("corpus_sha256") != digest(texts)
    ):
        raise ValueError(
            "baseline perplexity must be finite positive with exact development corpus"
        )
    updated = copy.deepcopy(before)
    for suite, key in (("primary", "dev_primary"), ("retain", "dev_retain")):
        if before[suite]["spec"] != dataset[key]:
            raise ValueError("baseline development data differs")
        updated[suite] = evaluation.summarize(
            dataset[key], before[suite]["raw"], retain=suite == "retain"
        )
    selected = {suite: updated[suite]["spec"] for suite in ("primary", "retain")}
    if "recovery" in before:
        updated["recovery"] = evaluation.summarize(
            before["recovery"]["spec"], before["recovery"]["raw"]
        )
        selected["recovery"] = updated["recovery"]["spec"]
    if digest(selected) != before.get("dataset_sha256") or not before.get("inference_sha256"):
        raise ValueError("baseline raw dataset identity differs")
    updated.update(
        contract=copy.deepcopy(evaluation.CONTRACT), scorer_version=evaluation.SCORER_VERSION
    )
    return updated


def reference_without_copies(runtime: Any, proposal: dict[str, Any]) -> dict[str, Any]:
    state = runtime.model.state_dict()
    hashes = {n: tensors.tensor_hash(v) for n, v in state.items()}
    return {
        "hashes": hashes,
        "shapes": {n: list(v.shape) for n, v in state.items()},
        "dtypes": {n: str(v.dtype) for n, v in state.items()},
        "original_tensors": {},
        "model_norm_squared": proposal["source"]["norm_squared"],
        "fingerprint": tensors.source_fingerprint(hashes),
    }


def budgeted_snapshot(
    runtime: Any, directory: Path, names: list[str], watch: Any
) -> dict[str, Any]:
    # FP32 runtime bytes conservatively bound source BF16/F32 bytes; zlib's
    # worst-case expansion is below this allowance, including receipt metadata.
    state = runtime.model.state_dict()
    size = sum(state[name].numel() * state[name].element_size() for name in names)
    additional = size + size // 100 + 4 * 1024**2
    remaining = []
    for root in {watch.root, watch.artifact_budget_root or watch.root}:
        used = tree_bytes(root)
        tensors.validate_write_budget(used, additional, watch.limits["disk_max_bytes"])
        remaining.append(watch.limits["disk_max_bytes"] - used)
    result = prepare_snapshot(runtime.path, directory, names, max_bytes=min(remaining))
    watch.check()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["screen"])
    for name in ("root", "source", "diagnosis", "edit-spec", "dataset", "before"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--histograms", "--payload", dest="histograms", type=Path, required=True)
    parser.add_argument(
        "--histograms-sha256", "--payload-sha256", dest="histograms_sha256", required=True
    )
    for name in (
        "source-fingerprint",
        "diagnosis-sha256",
        "edit-spec-sha256",
        "dataset-sha256",
        "before-sha256",
        "name",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    progress("row_screen", state="active")
    directory = None
    status: dict[str, Any] = {
        "phase": "screen",
        "method": None,
        "state": "running",
        "verdict": "inconclusive",
    }
    try:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.name) or not re.fullmatch(
            "[a-f0-9]{64}", args.source_fingerprint
        ):
            raise ValueError("invalid run name or source fingerprint")
        spec = baseline.pinned_json(args.edit_spec, args.edit_spec_sha256)
        validate_spec(spec)
        status.update(
            method=spec["method"],
            mapper=BACKENDS[spec["method"]],
            payload_kind=PAYLOAD_KINDS[spec["method"]],
        )
        diagnosis = baseline.pinned_json(args.diagnosis, args.diagnosis_sha256)
        mapper = validate_diagnosis(spec, diagnosis)
        if diagnosis.get("checkpoint_fingerprint") != args.source_fingerprint:
            raise ValueError("diagnosis source fingerprint differs")
        pinned_payload(args.histograms, args.histograms_sha256)
        dataset = baseline.pinned_json(args.dataset, args.dataset_sha256)
        if set(dataset) != {"dev_primary", "dev_retain"} or any(not dataset[k] for k in dataset):
            raise ValueError("only nonempty development evaluation inputs accepted")
        original_before = baseline.pinned_json(args.before, args.before_sha256)
        before_manifest = baseline.verify_before_source(args.before, args.source_fingerprint)
        before = rescore_before(original_before, dataset)
        if not args.apply:
            progress("row_screen", state="plan_only", launch=False)
            return 0
        root, source = args.root.absolute(), args.source.absolute()
        if (
            any(p.is_symlink() for p in (root, *root.parents))
            or root == source
            or root in source.parents
            or source in root.parents
        ):
            raise ValueError("workspace is redirected or overlaps source")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        candidate = root / args.name
        candidate.mkdir(mode=0o700, exist_ok=False)
        directory = candidate
        atomic_json(directory / "status.json", status)
        config = RuntimeConfig(max_seconds=None, max_new_tokens=64, seed=42, torch_threads=2)
        if not (root / "host-before.json").exists():
            atomic_json(root / "host-before.json", host_sample(root))
        with Watchdog(root, {"resources": iteration.LIMITS}, None, args.name) as watch:
            with active_phase("row_candidate_load"):
                inventory = checkpoint_inventory(source, config.max_model_bytes)
                if digest(inventory) != args.source_fingerprint:
                    raise ValueError("original source fingerprint differs")
                runtime: Any = HFRuntime(source, config)
                if runtime.inventory != inventory:
                    raise ValueError("original source changed during load")
            inference = digest(
                {
                    "runtime": runtime.contract["runtime"],
                    "resources": {
                        key: runtime.contract["resources"][key]
                        for key in ("target_prompt_tokens", "target_output_tokens")
                    },
                }
            )
            if before["inference_sha256"] != inference:
                raise ValueError("baseline inference identity differs from candidate runtime")
            runtime.contract["evaluation"] = {"receptive_choice_count": 4}
            check = runtime.check

            def guarded_check() -> None:
                check()
                watch.check()

            runtime.check = guarded_check
            code_root = Path(__file__).resolve().parents[1]
            files = [
                *sorted((code_root / "src").rglob("*.py")),
                Path(__file__),
                *[Path(str(m.__file__)) for m in (baseline, gradient, iteration, rmu)],
            ]
            manifest = {
                **baseline.common_manifest_fields(
                    args.source_fingerprint,
                    {str(p.relative_to(code_root)): tensors.file_hash(p) for p in files},
                    {},
                ),
                "entrypoint": "scripts/chinese_distribution_candidate.py",
                "phase": "screen",
                "method": spec["method"],
                "mapper": BACKENDS[spec["method"]],
                "payload_kind": PAYLOAD_KINDS[spec["method"]],
                "objective_spec": spec,
                "source_inventory": inventory,
                "runtime": asdict(config),
                "resources": dict(iteration.LIMITS),
                "overall_deadline": None,
                "input_sha256": {
                    k: getattr(args, k + "_sha256")
                    for k in ("diagnosis", "histograms", "edit_spec", "dataset", "before")
                },
                "before_manifest": before_manifest,
                "before_original_contract": original_before["contract"],
                "before_original_scorer": original_before.get("scorer_version"),
                "evaluation_contract": copy.deepcopy(evaluation.CONTRACT),
                "scorer_version": evaluation.SCORER_VERSION,
                "max_perplexity_ratio": "1.15",
                "baseline_rescore": {
                    "version": 1,
                    "source_sha256": args.before_sha256,
                    "output": "before-rescored.json",
                    "operation": "raw_evidence_rescore_without_generation",
                },
                "recovery_protocol": rmu.RECOVERY_PROTOCOL,
                "screen_helper": "chinese_rmu_candidate.screen",
                "training_performed": False,
            }
            atomic_json(directory / "manifest.json", manifest)
            atomic_json(directory / "before-rescored.json", before)
            atomic_json(directory / "edit-spec.json", spec)
            with active_phase("row_proposal"):
                pinned_payload(args.histograms, args.histograms_sha256)
                payload = import_module("safetensors.torch").load_file(
                    str(args.histograms), device="cpu"
                )
                proposal = mapper.to_proposal(
                    runtime,
                    diagnosis,
                    payload,
                    top_k=spec["top_k"],
                    execution_fraction=spec["execution_fraction"],
                )
                del payload
                atomic_json(directory / "proposal.json", proposal)
                names = operator.touched_tensors(runtime, proposal)
            state = None
            with active_phase("row_durable_snapshot"):
                snapshot = budgeted_snapshot(runtime, directory, names, watch)
                iteration.bind_snapshot(runtime, snapshot)
                reference = reference_without_copies(runtime, proposal)
            try:
                with active_phase("row_apply"):
                    state = operator.apply(runtime, proposal, snapshot)
                reference["original_tensors"] = state["before_images"]
                shapes = {tuple(reference["shapes"][n]) for n in names}
                if len(shapes) != 1:
                    raise ValueError("afterimage requires uniform touched tensor shapes")
                contract = {
                    "target": {"revision": baseline.MODEL_REVISION},
                    "resources": dict(iteration.LIMITS),
                    "edits": {
                        "allowed_tensors": names,
                        "expected_shape": list(next(iter(shapes))),
                        "max_tensor_relative_frobenius": operator.MAX_TENSOR_RELATIVE,
                        "max_model_relative_frobenius": operator.MAX_MODEL_RELATIVE,
                    },
                }
                with active_phase("row_afterimage"):
                    status["delta_manifest"] = gradient.save_after_image(
                        runtime, reference, contract, directory, root
                    )
                    atomic_json(directory / "operator.json", state["diagnostics"])
                status.update(rmu.screen(runtime, dataset, before, directory, watch))
            finally:
                try:
                    with active_phase("row_exact_restore"):
                        receipt = operator.restore(runtime, state) if state is not None else {}
                        actual = {
                            n: tensors.tensor_hash(v) for n, v in runtime.model.state_dict().items()
                        }
                        if actual != reference["hashes"]:
                            raise ValueError("canonical source tensors differ after restoration")
                        receipt.update(
                            restored_bit_exact=True, canonical_source_hashes_verified=True
                        )
                        atomic_json(directory / "restore.json", receipt)
                finally:
                    if checkpoint_inventory(source, config.max_model_bytes) != inventory:
                        raise ValueError("original checkpoint files changed")
            status.update(state="complete", source_files_unchanged=True)
            atomic_json(directory / "status.json", status)
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        if directory is not None:
            atomic_json(directory / "status.json", status)
        progress("row_screen", state="failed", error=status["error"])
        raise
    progress("row_screen", state="complete", verdict=status["verdict"])
    return 0 if status["verdict"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
