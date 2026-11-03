"""Local Chinese-language diagnosis and explicit candidate execution; no overall deadline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append("/usr/lib/python3/dist-packages")

from lacunai.core.model import HFRuntime, active_phase, checkpoint_inventory  # noqa: E402
from lacunai.core.spec import RuntimeConfig, bounded_file, digest  # noqa: E402
from lacunai.erase import language as operator  # noqa: E402
from lacunai.experimental.artifacts import atomic_json  # noqa: E402
from lacunai.experimental.resources import Watchdog, host_sample  # noqa: E402
from lacunai.restore.checkpoints import prepare_snapshot, write_edited_checkpoint  # noqa: E402
from lacunai.verify import language as evaluation  # noqa: E402

LIMITS = {
    "disk_max_bytes": 10 * 1024**3,
    "min_host_available_bytes": 16 * 1024**3,
    "min_root_free_bytes": 8 * 1024**3,
    "min_bulk_free_bytes": 100 * 1024**3,
    "max_load1": 40,
    "memory_psi_some_avg10": 1.0,
    "io_psi_full_avg10": 5.0,
    "sampling_interval_seconds": 5,
    "pressure_consecutive_samples": 3,
}


def score(
    runtime: HFRuntime, dataset: dict[str, Any], path: Path, watch: Watchdog, development: bool
) -> dict[str, Any]:
    with active_phase("chinese_evaluation"):
        result = evaluation.evaluate(
            runtime.model,
            runtime.tokenizer,
            dataset,
            runtime.contract,
            path,
            watch,
            development=development,
        )
    # A fixed non-Chinese development corpus, never candidate-generated content.
    texts = [item["prompts"][0] + "\n" + item["answer"] for item in dataset["dev_retain"]]
    with active_phase("retain_perplexity"):
        result["perplexity"] = runtime.perplexity(texts)
    atomic_json(path / "result.json", result)
    return result


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result = evaluation.compare(before, after)
    if before["perplexity"]["tokens"] != after["perplexity"]["tokens"]:
        raise ValueError("perplexity tokenization changed")
    ratio = Decimal(str(after["perplexity"]["perplexity"])) / Decimal(
        str(before["perplexity"]["perplexity"])
    )
    result["perplexity_ratio"] = str(ratio)
    result["maximum_perplexity_ratio"] = "1.15"
    if ratio > Decimal("1.15"):
        result["verdict"] = "failed"
        result.setdefault("reasons", []).append("non-Chinese perplexity exceeds frozen threshold")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["diagnose", "screen", "persist", "baseline", "dev-baseline", "verify"]
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--fraction", type=float, choices=operator.FRACTIONS)
    parser.add_argument("--name", required=True)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        print(json.dumps({"state": "plan_only", "phase": args.phase}), flush=True)
        return 0
    if not args.name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("invalid run name")
    root = args.root.absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("linked workspace refused")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = root / args.name
    directory.mkdir(mode=0o700, exist_ok=False)
    if not (root / "host-before.json").exists():
        atomic_json(root / "host-before.json", host_sample(root))
    status: dict[str, Any] = {"phase": args.phase, "state": "running", "verdict": "inconclusive"}
    atomic_json(directory / "status.json", status)
    config = RuntimeConfig(max_seconds=None, max_new_tokens=64, seed=42, torch_threads=2)
    dataset = json.loads(bounded_file(args.dataset, 32 * 1024**2))
    if "dev_retain" not in dataset:
        development = json.loads(bounded_file(root / "development-data.json", 32 * 1024**2))
        dataset["dev_retain"] = development["dev_retain"]
    try:
        with Watchdog(root, {"resources": LIMITS}, None, args.name) as watch:
            with active_phase("load_original"):
                runtime = HFRuntime(args.source, config)
            runtime.contract["evaluation"] = {"receptive_choice_count": 4}
            code_root = Path(__file__).resolve().parents[1]
            atomic_json(
                directory / "manifest.json",
                {
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                    "checkpoint_fingerprint": runtime.fingerprint,
                    "source_inventory": runtime.inventory,
                    "dataset_sha256": digest(dataset),
                    "runtime": asdict(config),
                    "resources": LIMITS,
                    "evaluation_contract": evaluation.CONTRACT,
                    "editor": "current Codex conversation",
                    "fraction": args.fraction,
                    "overall_deadline": None,
                    "code_hashes": {
                        str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(__file__), *sorted((code_root / "src").rglob("*.py"))]
                    },
                    "claim_limit": "finite-test recoverability; no proof of information absence",
                },
            )
            if args.phase == "diagnose":
                pairs = [
                    {key: item[key] for key in ("diagnostic_zh", "diagnostic_en")}
                    for item in dataset["dev_primary"]
                ]
                background = [
                    item["prompts"][0]
                    for item in dataset["dev_retain"]
                    if item["family"] in {"spanish", "japanese_han"}
                ]
                with active_phase("all_layer_language_diagnosis"):
                    diagnosis = operator.diagnose(runtime, pairs, background)
                atomic_json(directory / "diagnosis.json", diagnosis)
                if not args.diagnose_only:
                    result = score(runtime, dataset, directory / "baseline", watch, True)
                    status["result"] = result
            elif args.phase in {"baseline", "dev-baseline", "verify"}:
                result = score(
                    runtime, dataset, directory / "evaluation", watch, args.phase == "dev-baseline"
                )
                status["result"] = result
                if args.before:
                    before = json.loads(bounded_file(args.before, 32 * 1024**2))
                    status["comparison"] = compare(before, result)
                    status["verdict"] = status["comparison"]["verdict"]
            else:
                if args.diagnosis is None or args.fraction is None:
                    raise ValueError("candidate needs a frozen diagnosis and allowed fraction")
                diagnosis = json.loads(bounded_file(args.diagnosis, 64 * 1024**2))
                proposal = operator.propose(runtime, diagnosis, args.fraction)
                atomic_json(directory / "edit-spec.json", proposal)
                if checkpoint_inventory(args.source, config.max_model_bytes) != runtime.inventory:
                    raise ValueError("original checkpoint changed before snapshot")
                names = operator.touched_tensors(runtime, proposal)
                with active_phase("durable_snapshot"):
                    snapshot = prepare_snapshot(args.source, directory, names)
                state = operator.apply(runtime, proposal, snapshot)
                try:
                    atomic_json(directory / "edit-diagnostics.json", state["diagnostics"])
                    if args.phase == "screen":
                        result = score(runtime, dataset, directory / "evaluation", watch, True)
                        status["result"] = result
                        if args.before:
                            before = json.loads(bounded_file(args.before, 32 * 1024**2))
                            status["comparison"] = compare(before, result)
                            status["verdict"] = status["comparison"]["verdict"]
                    else:
                        replacements = {
                            name: dict(runtime.model.named_parameters())[name].detach()
                            for name in names
                        }
                        output = root / (args.name + "-checkpoint")
                        with active_phase("checkpoint_write"):
                            receipt = write_edited_checkpoint(
                                args.source, output, replacements, snapshot
                            )
                        status["checkpoint"] = str(output)
                        status["snapshot_id"] = receipt["snapshot_id"]
                finally:
                    with active_phase("exact_in_memory_restore"):
                        restored = operator.restore(runtime, state)
                    atomic_json(directory / "restore.json", restored)
                    del state
                    gc.collect()
            status["state"] = "complete"
            atomic_json(directory / "status.json", status)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(directory / "status.json", status)
        raise
    print(json.dumps({key: status[key] for key in ("phase", "state", "verdict")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
