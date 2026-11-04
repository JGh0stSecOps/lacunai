# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Bounded CPU tensor operations; no model-generated Python is executed.

ML imports are deliberately lazy. An ordinary LacunAI invocation does not
load a model runtime. Every derivative refers to immutable source bytes and
contains only complete changed tensors, never an in-place source update.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any, cast

Progress = Callable[[dict[str, Any]], None]
Check = Callable[[], None]
_CHUNK = 262144
_INTEROP_CONFIGURED = False


class TensorError(ValueError):
    """A numerical, provenance or specification invariant failed."""


def _emit(progress: Progress | None, **event: Any) -> None:
    if progress:
        progress(event)


def configure_runtime(contract: dict[str, Any]) -> None:
    """Apply the declared CPU settings before any tensor/model computation."""
    np = import_module("numpy")
    torch = import_module("torch")

    global _INTEROP_CONFIGURED
    runtime = contract["runtime"]
    if runtime["device"] != "cpu" or runtime["dtype"] != "float32":
        raise TensorError("only the approved CPU float32 runtime is supported")
    random.seed(runtime["seed"])
    np.random.seed(runtime["seed"])
    torch.manual_seed(runtime["seed"])
    torch.set_num_threads(runtime["torch_threads"])
    if not _INTEROP_CONFIGURED:
        torch.set_num_interop_threads(runtime["interop_threads"])
        _INTEROP_CONFIGURED = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor: Any) -> str:
    """Hash dtype, dimensions and canonical CPU bytes with bounded copies."""
    torch = import_module("torch")

    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise TensorError("tensor hash requires a CPU float32 tensor")
    digest = hashlib.sha256()
    digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
    flat = tensor.detach().reshape(-1)
    for offset in range(0, flat.numel(), _CHUNK):
        block = flat[offset : offset + _CHUNK].contiguous()
        digest.update(block.numpy().tobytes())
    return digest.hexdigest()


def _square_norm(tensor: Any) -> float:
    torch = import_module("torch")

    flat = tensor.detach().reshape(-1)
    total = torch.zeros((), dtype=torch.float64)
    for offset in range(0, flat.numel(), _CHUNK):
        block = flat[offset : offset + _CHUNK]
        if not bool(torch.isfinite(block).all()):
            raise TensorError("non-finite tensor value")
        total += block.to(torch.float64).square().sum()
    return float(total)


def source_fingerprint(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def _verify_source_files(path: Path, contract: dict[str, Any], role: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise TensorError("model source must be a real directory")
    for record in contract[role]["files"]:
        source = path / record["path"]
        if source.is_symlink() or not source.is_file():
            raise TensorError(f"missing or linked source file: {record['path']}")
        if source.stat().st_size != record["size"]:
            raise TensorError(f"source size mismatch: {record['path']}")
        if record.get("sha256") and file_hash(source) != record["sha256"]:
            raise TensorError(f"source hash mismatch: {record['path']}")


def load_target(
    model_path: str | Path, contract: dict[str, Any], role: str = "target"
) -> tuple[Any, Any]:
    """Load only local, pinned safetensors using a built-in model architecture."""
    torch = import_module("torch")
    transformers = import_module("transformers")
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer

    if role not in {"target", "editor"}:
        raise TensorError("unknown model role")
    path = Path(model_path)
    _verify_source_files(path, contract, role)
    configure_runtime(contract)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model, loading = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    missing = set(loading.get("missing_keys", []))
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    if tied and model.get_input_embeddings().weight is model.get_output_embeddings().weight:
        missing.discard("lm_head.weight")
    if missing or any(loading.get(k) for k in ("unexpected_keys", "mismatched_keys", "error_msgs")):
        raise TensorError(f"checkpoint did not load strictly: {loading}")
    if tokenizer.eos_token_id is None:
        raise TensorError("pinned tokenizer has no EOS token")
    model.to("cpu")
    model.eval()
    if role == "target":
        state = model.state_dict()
        for name in contract["edits"]["allowed_tensors"]:
            if name not in state or list(state[name].shape) != contract["edits"]["expected_shape"]:
                raise TensorError(f"target tensor shape mismatch: {name}")
    return model, tokenizer


def _prompt_ids(tokenizer: Any, prompt: str, cap: int) -> Any:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if encoded.shape[-1] > cap:
        raise TensorError(f"prompt exceeds declared cap: {encoded.shape[-1]} > {cap}")
    return encoded


def generate(
    model: Any, tokenizer: Any, prompt: str, contract: dict[str, Any], role: str = "target"
) -> dict[str, Any]:
    torch = import_module("torch")

    if role not in {"target", "editor"}:
        raise TensorError("unknown generation role")
    resources = contract["resources"]
    input_cap = resources["editor_input_tokens" if role == "editor" else "target_prompt_tokens"]
    cap = resources["editor_output_tokens" if role == "editor" else "target_output_tokens"]
    ids = _prompt_ids(tokenizer, prompt, input_cap)
    generation_config = import_module("transformers").GenerationConfig(
        do_sample=False,
        max_new_tokens=cap,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    model.eval()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            generation_config=generation_config,
        )
    generated = outputs.sequences[0, ids.shape[-1] :].tolist()
    if not generated or any(not bool(torch.isfinite(score).all()) for score in outputs.scores):
        raise TensorError("generation returned no tokens or non-finite scores")
    output = tokenizer.decode(generated, skip_special_tokens=True)
    if role == "editor" and len(output.encode()) > resources["editor_output_bytes"]:
        raise TensorError("editor output exceeds byte cap")
    return {
        "output": output,
        "input_tokens": ids.shape[-1],
        "output_tokens": len(generated),
        "token_ids": generated,
        "first_step_logits_sha256": tensor_hash(outputs.scores[0].to(torch.float32)),
        "truncated": len(generated) == cap and generated[-1] != tokenizer.eos_token_id,
    }


def choice_likelihoods(
    model: Any, tokenizer: Any, prompt: str, choices: list[str], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    """Mean answer-token log likelihood, without asking an LLM to score."""
    torch = import_module("torch")

    if len(choices) != contract["evaluation"]["receptive_choice_count"]:
        raise TensorError("wrong receptive choice count")
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    if len(prefix_ids) > contract["resources"]["target_prompt_tokens"]:
        raise TensorError("receptive prompt exceeds cap")
    scores = []
    model.eval()
    for choice in choices:
        ids = tokenizer(prefix + choice, add_special_tokens=False)["input_ids"]
        if ids[: len(prefix_ids)] != prefix_ids or len(ids) <= len(prefix_ids):
            raise TensorError("choice tokenization changed prompt boundary")
        if len(ids) - len(prefix_ids) > contract["resources"]["target_output_tokens"]:
            raise TensorError("receptive answer exceeds cap")
        tensor = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            logits = model(input_ids=tensor, attention_mask=torch.ones_like(tensor)).logits
            answer_logits = logits[0, len(prefix_ids) - 1 : -1].to(torch.float64)
            targets = tensor[0, len(prefix_ids) :]
            selected = torch.log_softmax(answer_logits, -1).gather(1, targets[:, None])[:, 0]
        scores.append(
            {
                "choice": choice,
                "mean_log_probability": float(selected.mean()),
                "token_log_probabilities": selected.tolist(),
                "tokens": targets.tolist(),
            }
        )
    return scores


def capture_reference(model: Any, contract: dict[str, Any]) -> dict[str, Any]:
    allowed = set(contract["edits"]["allowed_tensors"])
    state = model.state_dict()
    if not allowed.issubset(state):
        raise TensorError("missing allowed source tensors")
    hashes, shapes, dtypes, originals = {}, {}, {}, {}
    for name, tensor in state.items():
        _square_norm(tensor)
        hashes[name] = tensor_hash(tensor)
        shapes[name] = list(tensor.shape)
        dtypes[name] = str(tensor.dtype)
        if name in allowed:
            if shapes[name] != contract["edits"]["expected_shape"]:
                raise TensorError(f"source shape mismatch: {name}")
            originals[name] = tensor.detach().clone()
    total = math.fsum(_square_norm(p) for p in model.parameters())
    if not math.isfinite(total) or total <= 0:
        raise TensorError("invalid source model norm")
    return {
        "hashes": hashes,
        "shapes": shapes,
        "dtypes": dtypes,
        "original_tensors": originals,
        "model_norm_squared": total,
        "fingerprint": source_fingerprint(hashes),
    }


def _relative_changes(
    tensors: dict[str, Any], reference: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    per_tensor = {}
    squares = []
    for name, value in tensors.items():
        original = reference["original_tensors"][name]
        if (
            list(value.shape) != reference["shapes"][name]
            or str(value.dtype) != reference["dtypes"][name]
        ):
            raise TensorError(f"tensor shape/dtype changed: {name}")
        value_squared = _square_norm(value)
        if value_squared == 0:
            raise TensorError(f"all-zero tensor refused: {name}")
        delta_squared = _square_norm(value - original)
        original_squared = _square_norm(original)
        if original_squared == 0:
            raise TensorError(f"zero source tensor cannot use relative bound: {name}")
        relative = math.sqrt(delta_squared / original_squared)
        if relative > float(contract["edits"]["max_tensor_relative_frobenius"]) + 1e-9:
            raise TensorError(f"per-tensor delta bound exceeded: {name}")
        squares.append(delta_squared)
        per_tensor[name] = {
            "relative_frobenius": relative,
            "delta_norm_squared": delta_squared,
            "source_sha256": reference["hashes"][name],
            "sha256": tensor_hash(value),
        }
    whole = math.sqrt(math.fsum(squares) / reference["model_norm_squared"])
    if whole > float(contract["edits"]["max_model_relative_frobenius"]) + 1e-9:
        raise TensorError("whole-model delta bound exceeded")
    return {"tensors": per_tensor, "model_relative_frobenius": whole}


def validate_state(
    model: Any, reference: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    state = model.state_dict()
    if set(state) != set(reference["hashes"]):
        raise TensorError("checkpoint key set changed")
    allowed = set(contract["edits"]["allowed_tensors"])
    changed = {}
    for name, tensor in state.items():
        if (
            list(tensor.shape) != reference["shapes"][name]
            or str(tensor.dtype) != reference["dtypes"][name]
        ):
            raise TensorError(f"checkpoint shape/dtype changed: {name}")
        actual = tensor_hash(tensor)
        if actual != reference["hashes"][name]:
            if name not in allowed:
                raise TensorError(f"unapproved tensor changed: {name}")
            changed[name] = tensor
    result = _relative_changes(changed, reference, contract)
    result["changed_tensor_count"] = len(changed)
    result["source_fingerprint"] = reference["fingerprint"]
    return result


def validate_edit_spec(
    spec: Any, diagnostics: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(spec, dict) or set(spec) != {
        "operation",
        "source_fingerprint",
        "edits",
        "rationale",
    }:
        raise TensorError("edit specification has missing or unknown fields")
    if spec["operation"] != contract["edits"]["operation"]:
        raise TensorError("operation is not allowed")
    if spec["source_fingerprint"] != diagnostics["source_fingerprint"]:
        raise TensorError("edit specification source fingerprint mismatch")
    if not isinstance(spec["rationale"], str) or len(spec["rationale"]) > 2000:
        raise TensorError("invalid rationale")
    edits = spec["edits"]
    if not isinstance(edits, list) or not 1 <= len(edits) <= contract["edits"]["max_tensors"]:
        raise TensorError("invalid edited tensor count")
    used_tensors = set()
    choices = {x["id"]: x for x in diagnostics["columns"]}
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"tensor", "column_ids", "strength"}:
            raise TensorError("unknown tensor edit fields")
        name = edit["tensor"]
        if (
            not isinstance(name, str)
            or name not in contract["edits"]["allowed_tensors"]
            or name in used_tensors
        ):
            raise TensorError("unknown or duplicate target tensor")
        used_tensors.add(name)
        if (
            not isinstance(edit["strength"], str)
            or edit["strength"] not in contract["edits"]["strength_presets"]
        ):
            raise TensorError("unknown strength preset")
        ids = edit["column_ids"]
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= contract["edits"]["max_columns_per_tensor"]
        ):
            raise TensorError("invalid column count")
        if any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
            raise TensorError("invalid or duplicate diagnostic column IDs")
        for identifier in ids:
            if identifier not in choices or choices[identifier]["tensor"] != name:
                raise TensorError("column ID does not belong to approved diagnostic tensor")
            column = choices[identifier]["column"]
            if type(column) is not int or not 0 <= column < contract["edits"]["expected_shape"][1]:
                raise TensorError("diagnostic column is out of range")
    return spec


def apply_edit(
    model: Any,
    spec: dict[str, Any],
    diagnostics: dict[str, Any],
    contract: dict[str, Any],
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch = import_module("torch")

    reference = reference or capture_reference(model, contract)
    if diagnostics["source_fingerprint"] != reference["fingerprint"]:
        raise TensorError("diagnostic source fingerprint mismatch")
    if validate_state(model, reference, contract)["changed_tensor_count"]:
        raise TensorError("candidate must begin from the untouched checkpoint")
    validate_edit_spec(spec, diagnostics, contract)
    state = model.state_dict()
    choices = {x["id"]: x for x in diagnostics["columns"]}
    candidates = {}
    for edit in spec["edits"]:
        name = edit["tensor"]
        candidate = state[name].detach().clone()
        ids = [choices[x]["column"] for x in edit["column_ids"]]
        candidate[:, ids] *= 1.0 - float(contract["edits"]["strength_presets"][edit["strength"]])
        candidates[name] = candidate
    _relative_changes(candidates, reference, contract)
    try:
        with torch.no_grad():
            for name, candidate in candidates.items():
                state[name].copy_(candidate)
        return validate_state(model, reference, contract)
    except Exception:
        with torch.no_grad():
            for name in candidates:
                state[name].copy_(reference["original_tensors"][name])
        raise


def restore_reference(model: Any, reference: dict[str, Any], contract: dict[str, Any]) -> None:
    torch = import_module("torch")

    with torch.no_grad():
        state = model.state_dict()
        for name, value in reference["original_tensors"].items():
            state[name].copy_(value)
    if validate_state(model, reference, contract)["changed_tensor_count"] != 0:
        raise TensorError("restored checkpoint is not identical to source")


def save_delta(
    model: Any,
    reference: dict[str, Any],
    target_dir: str | Path,
    contract: dict[str, Any],
    budget_root: Path | None = None,
) -> dict[str, Any]:
    safetensors = import_module("safetensors.torch")
    load, save = safetensors.load, safetensors.save

    target = Path(target_dir)
    if target.exists() or target.is_symlink():
        raise TensorError("checkpoint destination already exists")
    diff = validate_state(model, reference, contract)
    changed = {name: model.state_dict()[name].detach().contiguous() for name in diff["tensors"]}
    blob = save(changed)
    reconstructed = load(blob)
    if set(reconstructed) != set(changed) or any(
        tensor_hash(reconstructed[n]) != tensor_hash(v) for n, v in changed.items()
    ):
        raise TensorError("safetensors round trip changed tensors")
    manifest = {
        "format": "neurailyzer-base-plus-delta-v1",
        "source_fingerprint": reference["fingerprint"],
        "target_revision": contract["target"]["revision"],
        "source_hashes": reference["hashes"],
        "diff": diff,
        "delta_sha256": hashlib.sha256(blob).hexdigest(),
    }
    encoded_manifest = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False).encode()
    if "resources" in contract and "disk_max_bytes" in contract["resources"]:
        from .artifacts import secure_path, tree_bytes

        if budget_root is None:
            raise TensorError("checkpoint requires an explicit artifact budget root")
        budget_root = budget_root.absolute()
        target = target.absolute()
        if not target.is_relative_to(budget_root):
            raise TensorError("checkpoint escapes the artifact budget root")
        secure_path(budget_root, target.relative_to(budget_root))
        # Reserve a MiB for concurrent heartbeat/receipt writes; no data pruning.
        validate_write_budget(
            tree_bytes(budget_root),
            len(blob) + len(encoded_manifest) + 1024 * 1024,
            contract["resources"]["disk_max_bytes"],
        )
    target.mkdir(mode=0o700, parents=False)
    tensor_path = target / "changed.safetensors"
    with tensor_path.open("xb") as stream:
        stream.write(blob)
    tensor_path.chmod(0o600)
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as stream:
        stream.write(encoded_manifest)
    manifest_path.chmod(0o600)
    return manifest


def validate_write_budget(used: int, additional: int, limit: int) -> None:
    """Admit the complete serialized write before creating its first file."""
    if min(used, additional, limit) < 0 or used + additional > limit:
        raise TensorError("checkpoint write exceeds aggregate artifact budget")


def load_delta(
    model: Any, path: str | Path, reference: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    torch = import_module("torch")
    load_file = import_module("safetensors.torch").load_file

    source = Path(path)
    if source.is_symlink() or any(
        (source / name).is_symlink() for name in ("manifest.json", "changed.safetensors")
    ):
        raise TensorError("linked checkpoint refused")
    manifest = cast(dict[str, Any], json.loads((source / "manifest.json").read_text()))
    if manifest.get("format") != "neurailyzer-base-plus-delta-v1":
        raise TensorError("unknown derivative checkpoint format")
    if (
        manifest["source_fingerprint"] != reference["fingerprint"]
        or manifest["source_hashes"] != reference["hashes"]
    ):
        raise TensorError("derivative source hash mismatch")
    if manifest["target_revision"] != contract["target"]["revision"]:
        raise TensorError("derivative target revision mismatch")
    if file_hash(source / "changed.safetensors") != manifest["delta_sha256"]:
        raise TensorError("derivative file hash mismatch")
    changed = load_file(source / "changed.safetensors", device="cpu")
    if not set(changed).issubset(contract["edits"]["allowed_tensors"]):
        raise TensorError("unapproved derivative tensor")
    if set(changed) != set(manifest["diff"]["tensors"]):
        raise TensorError("derivative key set mismatch")
    _relative_changes(changed, reference, contract)
    for name, value in changed.items():
        if tensor_hash(value) != manifest["diff"]["tensors"][name]["sha256"]:
            raise TensorError("derivative tensor hash mismatch")
    restore_reference(model, reference, contract)
    try:
        with torch.no_grad():
            state = model.state_dict()
            for name, value in changed.items():
                state[name].copy_(value)
        actual = validate_state(model, reference, contract)
        if actual != manifest["diff"]:
            raise TensorError("loaded derivative does not match recorded numerical diff")
    except Exception:
        restore_reference(model, reference, contract)
        raise
    return manifest


def diagnose(
    model: Any,
    tokenizer: Any,
    dev_items: list[dict[str, Any]],
    contract: dict[str, Any],
    progress_callback: Progress | None = None,
    check_callback: Check | None = None,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank contrastive activation columns; this is exploratory, not a probe of erasure.

    Items require explicit diagnostic_zh and diagnostic_en texts with matched
    meaning. No test-set text or model-generated translation is accepted here.
    """
    torch = import_module("torch")

    reference = reference or capture_reference(model, contract)
    allowed = contract["edits"]["allowed_tensors"]
    sums = {
        lang: {
            name: torch.zeros(contract["edits"]["expected_shape"][1], dtype=torch.float64)
            for name in allowed
        }
        for lang in ("zh", "en")
    }
    pairs = [item for item in dev_items if "diagnostic_zh" in item and "diagnostic_en" in item]
    if not pairs:
        raise TensorError("no explicit paired development diagnostics")
    model.eval()
    for i, item in enumerate(pairs):
        if check_callback:
            check_callback()
        for lang in ("zh", "en"):
            handles = []
            for name in allowed:
                module = model.get_submodule(name.removesuffix(".weight"))

                def hook(_module: Any, inputs: Any, name: str = name, lang: str = lang) -> None:
                    activation = inputs[0].detach().to(torch.float64).abs().mean(dim=(0, 1))
                    if not bool(torch.isfinite(activation).all()):
                        raise TensorError("non-finite diagnostic activation")
                    sums[lang][name] += activation

                handles.append(module.register_forward_pre_hook(hook))
            try:
                ids = _prompt_ids(
                    tokenizer,
                    item[f"diagnostic_{lang}"],
                    contract["resources"]["target_prompt_tokens"],
                )
                with torch.inference_mode():
                    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
        _emit(progress_callback, phase="diagnosis", completed=i + 1, total=len(pairs), unit="pairs")
    columns = []
    for name in allowed:
        zh, en = sums["zh"][name] / len(pairs), sums["en"][name] / len(pairs)
        score = (zh - en) / (zh + en + 1e-12)
        # Stable sort makes tied diagnostic values deterministic.
        selected = torch.argsort(score, descending=True, stable=True)[:16].tolist()
        layer = name.split(".")[2]
        for column in selected:
            columns.append(
                {
                    "id": f"l{layer}c{column}",
                    "tensor": name,
                    "column": column,
                    "score": float(score[column]),
                    "zh_mean_abs": float(zh[column]),
                    "en_mean_abs": float(en[column]),
                }
            )
    return {
        "source_fingerprint": reference["fingerprint"],
        "columns": columns,
        "pair_count": len(pairs),
        "normalization": "(mean_abs_zh-mean_abs_en)/(mean_abs_zh+mean_abs_en+1e-12)",
        "limits": "heuristic activation association; not causal localization or absence evidence",
    }


def grad_diff_objective(
    retain_loss: Any, forget_loss: Any, alpha: float = 1.0, gamma: float = 1.0
) -> Any:
    return alpha * retain_loss - gamma * forget_loss


def _answer_loss(model: Any, tokenizer: Any, item: dict[str, Any], cap: int) -> Any:
    torch = import_module("torch")

    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": item["prompts"][0]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full = tokenizer(prefix + item["answer"], add_special_tokens=False)["input_ids"]
    if full[: len(prefix_ids)] != prefix_ids or len(full) <= len(prefix_ids):
        raise TensorError("training answer boundary changed under tokenization")
    full.append(tokenizer.eos_token_id)
    if len(full) > cap:
        raise TensorError(f"training example exceeds token cap: {len(full)} > {cap}")
    ids = torch.tensor([full], dtype=torch.long)
    labels = ids.clone()
    labels[:, : len(prefix_ids)] = -100
    return model(
        input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False
    ).loss


def _project(model: Any, reference: dict[str, Any], contract: dict[str, Any]) -> None:
    torch = import_module("torch")

    state = model.state_dict()
    with torch.no_grad():
        squares = []
        for name, original in reference["original_tensors"].items():
            value = state[name]
            delta = value - original
            delta_squared = _square_norm(delta)
            limit = float(contract["edits"]["max_tensor_relative_frobenius"]) * math.sqrt(
                _square_norm(original)
            )
            norm = math.sqrt(delta_squared)
            if norm > limit:
                delta *= (limit / norm) * (1.0 - 1e-6)
                value.copy_(original + delta)
            squares.append(_square_norm(value - original))
        norm = math.sqrt(math.fsum(squares))
        limit = float(contract["edits"]["max_model_relative_frobenius"]) * math.sqrt(
            reference["model_norm_squared"]
        )
        if norm > limit:
            for name, original in reference["original_tensors"].items():
                state[name].copy_(
                    original + (state[name] - original) * (limit / norm) * (1.0 - 1e-6)
                )


def train(
    model: Any,
    tokenizer: Any,
    forget_items: list[dict[str, Any]],
    retain_items: list[dict[str, Any]],
    contract: dict[str, Any],
    mode: str,
    progress_callback: Progress | None = None,
    check_callback: Check | None = None,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parameter-restricted GradDiff, matched retain control or benign relearning."""
    torch = import_module("torch")

    if mode not in {"graddiff", "retain_finetune", "relearn"} or not retain_items:
        raise TensorError("invalid training mode or empty retain data")
    if mode == "graddiff" and not forget_items:
        raise TensorError("GradDiff needs forget examples")
    # For relearning, this reference is the original source as well, so the
    # manifest loader can reconstruct the full combined derivative exactly.
    reference = reference or capture_reference(model, contract)
    allowed = set(contract["edits"]["allowed_tensors"])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(parameters) != len(allowed):
        raise TensorError("trainable parameter set does not match tensor allowlist")
    settings = contract["relearning"] if mode == "relearn" else contract["baseline"]
    steps = settings.get("steps_per_arm", settings.get("steps"))
    optimizer = torch.optim.SGD(parameters, lr=float(settings["learning_rate"]), momentum=0.0)
    model.eval()  # gradients remain enabled; no stochastic dropout
    trace = []
    for step in range(steps):
        if check_callback:
            check_callback()
        optimizer.zero_grad(set_to_none=True)
        retain = _answer_loss(
            model,
            tokenizer,
            retain_items[step % len(retain_items)],
            contract["baseline"]["sequence_tokens"],
        )
        if not bool(torch.isfinite(retain)):
            raise TensorError("non-finite retain loss")
        retain_value = float(retain.detach())
        # Backprop separately to avoid retaining both activation graphs.
        (float(contract["baseline"]["alpha"]) * retain).backward()
        forget_value = None
        if mode == "graddiff":
            forget = _answer_loss(
                model,
                tokenizer,
                forget_items[step % len(forget_items)],
                contract["baseline"]["sequence_tokens"],
            )
            if not bool(torch.isfinite(forget)):
                raise TensorError("non-finite forget loss")
            forget_value = float(forget.detach())
            (-float(contract["baseline"]["gamma"]) * forget).backward()
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(contract["baseline"]["gradient_clip"]), error_if_nonfinite=True
        )
        optimizer.step()
        _project(model, reference, contract)
        event = {
            "phase": mode,
            "completed": step + 1,
            "total": steps,
            "unit": "steps",
            "retain_loss": retain_value,
            "forget_loss": forget_value,
            "gradient_norm": float(norm),
        }
        trace.append(event)
        _emit(progress_callback, **event)
    optimizer.zero_grad(set_to_none=True)
    return {"mode": mode, "trace": trace, "diff": validate_state(model, reference, contract)}


def numerical_smoke(contract: dict[str, Any]) -> dict[str, Any]:
    """A real tensor/gradient smoke before loading either downloaded model."""
    torch = import_module("torch")

    configure_runtime(contract)
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    retain = (x * x).sum()
    forget = x.sum()
    loss = grad_diff_objective(retain, forget)
    loss.backward()
    if not torch.equal(x.grad, torch.tensor([1.0, 3.0])) or float(loss.detach()) != 2.0:
        raise TensorError("GradDiff numerical smoke failed")
    return {
        "torch": torch.__version__,
        "loss": float(loss.detach()),
        "gradient": x.grad.tolist(),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def numeric_selftest(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run actual tiny tensors and removal plants without requiring pytest.

    Plant replacements exist only during this single-threaded startup test and
    are restored in finally blocks. No downloaded model or source file changes.
    """
    import copy

    torch = import_module("torch")
    safetensors = import_module("safetensors.torch")
    load, save = safetensors.load, safetensors.save

    if contract is None:
        contract = {
            "runtime": {
                "device": "cpu",
                "dtype": "float32",
                "seed": 20261102,
                "torch_threads": 2,
                "interop_threads": 1,
            }
        }
    smoke = numerical_smoke(contract)
    policy = {
        "edits": {
            "allowed_tensors": ["linear.weight"],
            "expected_shape": [2, 4],
            "operation": "attenuate_diagnostic_columns",
            "max_tensors": 1,
            "max_columns_per_tensor": 1,
            "strength_presets": {"low": "0.01", "high": "0.5"},
            "max_tensor_relative_frobenius": "0.02",
            "max_model_relative_frobenius": "0.01",
        }
    }
    model = torch.nn.Module()
    model.add_module("linear", torch.nn.Linear(4, 2, bias=False))
    model.register_parameter("untouched", torch.nn.Parameter(torch.tensor([4.0])))
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
    reference = capture_reference(model, policy)
    diagnostics = {
        "source_fingerprint": reference["fingerprint"],
        "columns": [{"id": "test-column", "tensor": "linear.weight", "column": 0}],
    }
    spec = {
        "operation": "attenuate_diagnostic_columns",
        "source_fingerprint": reference["fingerprint"],
        "rationale": "synthetic numerical test",
        "edits": [{"tensor": "linear.weight", "column_ids": ["test-column"], "strength": "low"}],
    }
    passed: list[str] = []
    plants: list[dict[str, str]] = []

    def reject(name: str, action: Callable[[], Any]) -> None:
        try:
            action()
        except TensorError:
            passed.append(name)
        else:
            raise AssertionError(f"{name}: expected TensorError was not raised")

    before = model.linear(torch.ones(1, 4)).detach().clone()
    if validate_state(model, reference, policy)["changed_tensor_count"] != 0:
        raise AssertionError("no-op changed tensors")
    restored = load(save({"linear.weight": model.linear.weight.detach()}))
    if not torch.equal(restored["linear.weight"], model.linear.weight):
        raise AssertionError("safetensors did not preserve no-op tensor")
    passed.append("no_op_safetensors_identity")
    if not torch.equal(before, model.linear(torch.ones(1, 4))):
        raise AssertionError("no-op forward differs")
    passed.append("no_op_forward_identity")
    diff = apply_edit(model, spec, diagnostics, policy, reference)
    if (
        diff["changed_tensor_count"] != 1
        or tensor_hash(model.untouched) != reference["hashes"]["untouched"]
    ):
        raise AssertionError("valid edit did not preserve unselected tensor")
    passed.append("bounded_edit_and_unselected_identity")
    restore_reference(model, reference, policy)
    if not torch.equal(before, model.linear(torch.ones(1, 4))):
        raise AssertionError("restored forward differs")
    passed.append("restored_forward_identity")

    invalid = copy.deepcopy(spec)
    invalid["edits"][0]["tensor"] = "untouched"
    reject("tensor_allowlist", lambda: validate_edit_spec(invalid, diagnostics, policy))
    invalid = copy.deepcopy(spec)
    invalid["source_fingerprint"] = "wrong"
    reject("source_fingerprint", lambda: validate_edit_spec(invalid, diagnostics, policy))
    invalid = copy.deepcopy(spec)
    invalid["edits"][0]["column_ids"] = ["invented"]
    reject("diagnostic_allowlist", lambda: validate_edit_spec(invalid, diagnostics, policy))
    reject("nonfinite", lambda: _square_norm(torch.tensor([float("nan")])))
    reject(
        "shape", lambda: _relative_changes({"linear.weight": torch.ones(3, 4)}, reference, policy)
    )
    reject(
        "zero", lambda: _relative_changes({"linear.weight": torch.zeros(2, 4)}, reference, policy)
    )
    unsafe = {"linear.weight": model.linear.weight.detach() * 0.5}
    reject("delta_bound", lambda: _relative_changes(unsafe, reference, policy))
    with torch.no_grad():
        model.untouched.add_(1)
    reject("unselected_tamper", lambda: validate_state(model, reference, policy))
    with torch.no_grad():
        model.untouched.fill_(4)

    # This replaces the production bound validator, not the test assertion.
    real_bound_validator = globals()["_relative_changes"]
    try:
        globals()["_relative_changes"] = lambda *_args: {
            "tensors": {},
            "model_relative_frobenius": 0,
        }
        try:
            reject("delta_bound_removed", lambda: _relative_changes(unsafe, reference, policy))
        except AssertionError as exc:
            plants.append({"guard": "delta_bounds", "red": str(exc)})
        else:
            raise AssertionError("bound removal plant did not turn assertion red")
    finally:
        globals()["_relative_changes"] = real_bound_validator
    reject("delta_bound_restored", lambda: _relative_changes(unsafe, reference, policy))

    real_schema_validator = globals()["validate_edit_spec"]
    try:
        globals()["validate_edit_spec"] = lambda proposed, *_args: proposed
        try:
            reject("schema_removed", lambda: validate_edit_spec(invalid, diagnostics, policy))
        except AssertionError as exc:
            plants.append({"guard": "edit_schema", "red": str(exc)})
        else:
            raise AssertionError("schema removal plant did not turn assertion red")
    finally:
        globals()["validate_edit_spec"] = real_schema_validator
    reject("schema_restored", lambda: validate_edit_spec(invalid, diagnostics, policy))
    reject("write_budget_one_byte_over", lambda: validate_write_budget(4, 7, 10))
    validate_write_budget(4, 6, 10)
    passed.append("write_budget_exact_boundary")
    real_write_validator = globals()["validate_write_budget"]
    try:
        globals()["validate_write_budget"] = lambda *_args: None
        try:
            reject("write_budget_removed", lambda: validate_write_budget(4, 7, 10))
        except AssertionError as exc:
            plants.append({"guard": "write_budget", "red": str(exc)})
        else:
            raise AssertionError("write budget removal did not turn assertion red")
    finally:
        globals()["validate_write_budget"] = real_write_validator
    reject("write_budget_restored", lambda: validate_write_budget(4, 7, 10))
    validate_state(model, reference, policy)
    return {
        "status": "passed",
        "passed_count": len(passed),
        "checks": passed,
        "removal_plants": plants,
        "numerical_smoke": smoke,
    }
