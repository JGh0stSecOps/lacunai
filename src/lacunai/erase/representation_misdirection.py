"""Bounded RMU adaptation: masked block activations, reversible original-model teacher."""

from __future__ import annotations

import hashlib
from importlib import import_module
from typing import Any

from lacunai.erase.gradient_objectives import (
    SCHEDULE_METHOD,
    answer_batch,
    family_schedule,
    original_teacher,
)
from lacunai.experimental import tensors
from lacunai.map.localize import record_digest

ALLOWED = tuple(f"model.layers.{layer}.mlp.down_proj.weight" for layer in (5, 6, 7))
FIXED = {
    "schema_version": 1,
    "method": "rmu_candidate",
    "steps": 60,
    "optimizer": "AdamW",
    "learning_rate": "0.0001",
    "betas": [0.9, 0.999],
    "eps": "0.00000001",
    "weight_decay": "0",
    "alpha": "1.0",
    "gamma": "1.0",
    "gradient_clip": "1.0",
    "steering_coeff": "20",
    "seed": 42,
    "sequence_tokens": 128,
    "hook_module": "model.layers.7",
    "inner_tensor_relative_frobenius": "0.0195",
    "inner_model_relative_frobenius": "0.0024",
    "outer_tensor_relative_frobenius": "0.02",
    "outer_model_relative_frobenius": "0.0025",
}
READING_FAMILIES = ("read_zh_answer_en_or_numeric", "translate_zh_en")
WRITING_FAMILIES = ("write_zh", "translate_en_zh")
CONTEXT_FIXED = {
    **FIXED,
    "method": "rmu_context_candidate",
    "mask_policy": "user_content_and_answer_for_reading;answer_for_writing;"
    "user_content_and_answer_for_retain:v1",
    "variant_policy": "reading_cycle_existing_variants:v1",
}

PROVENANCE = {
    "openunlearning": "https://github.com/locuslab/open-unlearning/blob/"
    "a456aa277be14b836248e42b13b3befe47bb03f4/src/trainer/unlearn/rmu.py",
    "wmdp": "https://github.com/centerforaisafety/wmdp/blob/main/rmu/unlearn.py",
    "adaptation": "Root-selected adaptation, not a paper/config replication. CPU float32 Qwen2; "
    "canonical answer/EOS hidden positions only; one resident "
    "model; original teacher substitution; frozen AdamW/projection bounds and balanced families",
    "mask": "labels != -100 at hidden-state positions, not shifted causal-logit positions",
    "reduction": "microbatch one: mean squared difference across "
    "masked tokens and hidden dimensions",
    "early_stop": "private sentinel from the block7 output hook; later blocks/head not executed",
    "equivalence_tolerance": {"atol": 2e-6, "rtol": 2e-5},
    "limitation": "Representation misdirection is not evidence of knowledge absence.",
}


def validate_spec(spec: dict[str, Any], inner: dict[str, Any]) -> None:
    context = spec.get("method") == "rmu_context_candidate"
    fixed = CONTEXT_FIXED if context else FIXED
    modes = ("rmu_context", "retain_context_mse") if context else ("rmu", "retain_mse")
    if set(spec) != set(fixed) | {"mode", "allowed_tensors"}:
        raise ValueError("RMU specification fields differ from frozen contract")
    for name, expected in fixed.items():
        if type(spec[name]) is not type(expected) or spec[name] != expected:
            raise ValueError(f"RMU frozen specification differs: {name}")
    if any(type(beta) is not float for beta in spec["betas"]):
        raise ValueError("RMU betas must be exact JSON floats")
    if spec["mode"] not in modes or spec["allowed_tensors"] != list(ALLOWED):
        raise ValueError("RMU mode or tensor allowlist differs")
    edits = inner.get("edits", {})
    if (
        edits.get("allowed_tensors") != list(ALLOWED)
        or edits.get("expected_shape") != [896, 4864]
        or str(edits.get("max_tensor_relative_frobenius")) != "0.0195"
        or str(edits.get("max_model_relative_frobenius")) != "0.0024"
    ):
        raise ValueError("RMU projection bounds/shape differ from frozen inner contract")


def context_variants(
    items: list[dict[str, Any]], schedule: dict[str, Any], *, retain: bool
) -> list[int]:
    occurrences: dict[str, int] = {}
    variants = []
    for index in schedule["indices"]:
        item = items[index]
        family, prompts = item["family"], item.get("prompts")
        if (
            not isinstance(prompts, list)
            or not prompts
            or any(not isinstance(p, str) or not p for p in prompts)
        ):
            raise ValueError("context schedule requires nonempty prompt variants")
        if not retain and family not in READING_FAMILIES + WRITING_FAMILIES:
            raise ValueError("context forget family is unsupported")
        occurrence = occurrences.get(family, 0)
        variants.append(
            occurrence % len(prompts) if not retain and family in READING_FAMILIES else 0
        )
        occurrences[family] = occurrence + 1
    return variants


def user_content_offsets(tokenizer: Any, prompt: str, answer: str) -> dict[str, Any]:
    """Validate a literal chat-content span against fast-tokenizer character offsets."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("context masks require fast-tokenizer offset mappings")
    marker = "LACUNAI_CONTENT_BOUNDARY_7f931a"

    def render(content: str) -> str:
        return str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
        )

    layout = render(marker)
    if layout.count(marker) != 1:
        raise ValueError("chat template does not preserve a unique literal user span")
    before, after = layout.split(marker)
    prefix = render(prompt)
    if prefix != before + prompt + after:
        raise ValueError("chat template transforms user content or boundaries")
    start, end = len(before), len(before) + len(prompt)
    encoded = tokenizer(prefix + answer, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) != len(offsets) or not ids:
        raise ValueError("offset/token inventory differs")
    positions = []
    coverage: set[int] = set()
    special = set(tokenizer.all_special_ids)
    for index, pair in enumerate(offsets):
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(type(value) is not int for value in pair)
        ):
            raise ValueError("invalid tokenizer character offsets")
        left, right = pair
        if not 0 <= left <= right <= len(prefix + answer):
            raise ValueError("tokenizer offsets exceed rendered content")
        if left < end and right > start:
            if left < start or right > end or ids[index] in special:
                raise ValueError(
                    "user-content token crosses boundary or contains role/special token"
                )
            positions.append(index)
            coverage.update(range(left, right))
    if not positions or any(
        start + i not in coverage for i, c in enumerate(prompt) if not c.isspace()
    ):
        raise ValueError("user-content offsets do not cover the literal prompt")
    return {
        "input_ids": ids,
        "offset_mapping": [list(pair) for pair in offsets],
        "user_content_span": [start, end],
        "user_content_positions": positions,
    }


def context_batch(
    tokenizer: Any, item: dict[str, Any], variant: int, *, retain: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = import_module("torch")
    if type(variant) is not int or not 0 <= variant < len(item["prompts"]):
        raise ValueError("invalid context prompt variant")
    selected = {**item, "prompts": [item["prompts"][variant]]}
    batch = answer_batch(tokenizer, selected, 128)
    include_user = retain or item["family"] in READING_FAMILIES
    mask = batch["labels"] != -100
    evidence: dict[str, Any] = {}
    if include_user:
        evidence = user_content_offsets(tokenizer, selected["prompts"][0], item["answer"])
        if evidence["input_ids"] != batch["input_ids"][0, :-1].tolist():
            raise ValueError("offset tokenization differs from canonical answer tokenization")
        mask[0, evidence["user_content_positions"]] = True
    elif item["family"] not in WRITING_FAMILIES or variant != 0:
        raise ValueError("context writing must use canonical variant zero")
    batch["activation_mask"] = mask
    ids, labels = batch["input_ids"].tolist(), batch["labels"].tolist()
    positions = torch.nonzero(mask[0], as_tuple=False).flatten().tolist()
    record = {
        "item_id": item["id"],
        "family": item["family"],
        "variant": variant,
        "policy": "user_content_and_answer" if include_user else "answer",
        "positions": positions,
        "mask_sha256": record_digest(mask.tolist()),
        "input_ids": ids,
        "labels": labels,
        "token_batch_sha256": record_digest({"ids": ids, "labels": labels}),
        "user_content_span": evidence.get("user_content_span"),
        "user_content_positions": evidence.get("user_content_positions", []),
        "offset_mapping": evidence.get("offset_mapping", []),
    }
    return batch, record


class _CapturedBlock(Exception):
    """Only a matching local sentinel may end a bounded forward successfully."""


def capture_hidden(model: Any, batch: dict[str, Any]) -> Any:
    """Return block7 output at answer positions; remove our hook on every exit."""
    torch = import_module("torch")
    if getattr(model.config, "model_type", None) != "qwen2":
        raise ValueError("RMU early-stop capture requires Qwen2")
    labels, ids = batch["labels"], batch["input_ids"]
    if labels.ndim != 2 or labels.shape != ids.shape or ids.shape[0] != 1:
        raise ValueError("RMU capture requires aligned microbatch-one labels")
    mask = batch.get("activation_mask", labels != -100)
    if mask.dtype != torch.bool or mask.shape != labels.shape:
        raise ValueError("RMU hidden-state mask shape/dtype differs")
    if not bool(mask.any()):
        raise ValueError("RMU answer hidden-state mask is empty")
    module = dict(model.named_modules()).get(FIXED["hook_module"])
    if module is None:
        raise ValueError("RMU block7 hook module missing")
    captured: list[Any] = []
    sentinel = _CapturedBlock()

    def hook(module: Any, inputs: Any, output: Any) -> None:
        del module, inputs
        value = output[0] if isinstance(output, tuple) else output
        if (
            value.ndim != 3
            or list(value.shape[:2]) != list(ids.shape)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.shape[-1] != model.config.hidden_size
            or captured
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("RMU block output shape/dtype/values differ")
        captured.append(value[mask])
        raise sentinel

    handle = module.register_forward_hook(hook)
    try:
        try:
            model(input_ids=ids, attention_mask=batch["attention_mask"], use_cache=False)
        except _CapturedBlock as exc:
            if exc is not sentinel:
                raise
        if len(captured) != 1:
            raise ValueError("RMU block7 hook did not execute exactly once")
        return captured[0]
    finally:
        handle.remove()
        sentinel.__traceback__ = None


def masked_mse(activations: Any, target: Any) -> Any:
    torch = import_module("torch")
    if (
        activations.ndim != 2
        or activations.numel() == 0
        or target.shape != activations.shape
        or not bool(torch.isfinite(activations).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise ValueError("RMU masked activations/target differ")
    loss = (activations - target.detach()).square().mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite RMU activation MSE")
    return loss


def control_vector() -> tuple[Any, dict[str, Any]]:
    torch = import_module("torch")
    generator = torch.Generator(device="cpu").manual_seed(42)
    raw = torch.rand((1, 1, 896), generator=generator, dtype=torch.float32)
    control = raw / torch.linalg.vector_norm(raw) * 20.0
    norm = float(torch.linalg.vector_norm(control.to(torch.float64)))
    if not bool(torch.isfinite(control).all()) or abs(norm - 20.0) > 1e-5:
        raise ValueError("RMU control-vector norm differs")
    return control, {
        "generator": "torch.Generator(device=cpu)",
        "seed": 42,
        "shape": [1, 1, 896],
        "dtype": "torch.float32",
        "steering_coeff": "20",
        "raw_bytes_sha256": hashlib.sha256(control.numpy().tobytes()).hexdigest(),
        "tensor_sha256": tensors.tensor_hash(control),
        "norm": norm,
        "norm_tolerance": 1e-5,
    }


def train(
    runtime: Any,
    reference: dict[str, Any],
    corpus: dict[str, Any],
    inner: dict[str, Any],
    spec: dict[str, Any],
    watch: Any,
) -> dict[str, Any]:
    watch.progress("rmu_training", state="active", completed=0, total=60, unit="steps")
    try:
        return _train(runtime, reference, corpus, inner, spec, watch)
    except BaseException as exc:
        watch.progress("rmu_training", state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for parameter in runtime.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None


def _train(
    runtime: Any,
    reference: dict[str, Any],
    corpus: dict[str, Any],
    inner: dict[str, Any],
    spec: dict[str, Any],
    watch: Any,
) -> dict[str, Any]:
    validate_spec(spec, inner)
    torch = import_module("torch")
    forget, retain = corpus.get("forget_train"), corpus.get("retain_train")
    if (
        not isinstance(forget, list)
        or len(forget) != 64
        or not isinstance(retain, list)
        or len(retain) != 80
    ):
        raise ValueError("RMU requires frozen forget64 and retain80 records")
    retain_schedule = family_schedule(retain, 60)
    planned_forget = family_schedule(forget, 60)
    if len(retain_schedule["family_order"]) != 5 or len(planned_forget["family_order"]) != 4:
        raise ValueError("RMU requires five retain and four forget families")
    context = spec["method"] == "rmu_context_candidate"
    forget_schedule = planned_forget if spec["mode"] in ("rmu", "rmu_context") else None
    schedule = {"method": SCHEDULE_METHOD, "retain": retain_schedule, "forget": forget_schedule}
    if context:
        retain_schedule["variant_indices"] = context_variants(retain, retain_schedule, retain=True)
        if forget_schedule is not None:
            forget_schedule["variant_indices"] = context_variants(
                forget, forget_schedule, retain=False
            )
    if set(reference["original_tensors"]) != set(ALLOWED):
        raise ValueError("RMU reference tensor allowlist differs")
    watch.progress("rmu_reference_validation", state="active")
    watch.check()
    runtime.check()
    if tensors.validate_state(runtime.model, reference, inner)["changed_tensor_count"]:
        raise ValueError("RMU must begin from untouched reference source")
    parameters = dict(runtime.model.named_parameters())
    if getattr(runtime.model.config, "hidden_size", None) != 896:
        raise ValueError("RMU source hidden width differs from frozen control vector")
    for name, parameter in parameters.items():
        if parameter.requires_grad:
            raise ValueError("RMU expects initially frozen runtime parameters")
        if name in ALLOWED and (
            parameter.dtype != torch.float32
            or parameter.device.type != "cpu"
            or list(parameter.shape) != [896, 4864]
        ):
            raise ValueError("RMU requires CPU float32 allowed tensors with frozen shape")
    runtime.model.eval()
    batches: dict[str, dict[tuple[int, int], Any]] = {}
    qualification: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for split, items in (("forget", forget), ("retain", retain)):
        batches[split], qualification[split] = {}, {}
        selected_schedule = forget_schedule if split == "forget" else retain_schedule
        if context:
            requests = (
                list(
                    dict.fromkeys(
                        zip(
                            selected_schedule["indices"],
                            selected_schedule["variant_indices"],
                            strict=True,
                        )
                    )
                )
                if selected_schedule
                else []
            )
        else:
            requests = [(index, 0) for index in range(len(items))]
        watch.progress(
            "rmu_tokenization", split=split, completed=0, total=len(requests), unit="examples"
        )
        for completed, (index, variant) in enumerate(requests, 1):
            watch.check()
            runtime.check()
            if context:
                batch, evidence = context_batch(
                    runtime.tokenizer, items[index], variant, retain=split == "retain"
                )
                qualification[split][index, variant] = evidence
            else:
                batch = answer_batch(runtime.tokenizer, items[index], 128)
            batches[split][index, variant] = batch
            watch.progress(
                "rmu_tokenization",
                split=split,
                completed=completed,
                total=len(requests),
                unit="examples",
            )
    control, control_record = control_vector()
    watch.progress("rmu_reference_validation", state="complete")
    for name in ALLOWED:
        parameters[name].requires_grad_(True)
    trainable = [parameters[name] for name in ALLOWED]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(spec["learning_rate"]),
        betas=tuple(spec["betas"]),
        eps=float(spec["eps"]),
        weight_decay=float(spec["weight_decay"]),
    )
    trace = []
    for step in range(60):
        watch.check()
        runtime.check()
        optimizer.zero_grad(set_to_none=True)
        retain_index = retain_schedule["indices"][step]
        retain_variant = retain_schedule["variant_indices"][step] if context else 0
        batch = batches["retain"][retain_index, retain_variant]
        watch.progress("rmu_retain_teacher", completed=step, total=60, unit="steps")
        # All prior backward passes finished before swapping any parameter storage.
        with original_teacher(runtime.model, reference):
            teacher = capture_hidden(runtime.model, batch).detach()
        current = capture_hidden(runtime.model, batch)
        retain_loss = masked_mse(current, teacher)
        retain_value = float(retain_loss.detach())
        (float(spec["alpha"]) * retain_loss).backward()
        del current, teacher, retain_loss
        forget_value = None
        if forget_schedule is not None:
            forget_index = forget_schedule["indices"][step]
            forget_variant = forget_schedule["variant_indices"][step] if context else 0
            current = capture_hidden(runtime.model, batches["forget"][forget_index, forget_variant])
            forget_loss = masked_mse(current, control.reshape(1, 896).expand_as(current))
            forget_value = float(forget_loss.detach())
            (float(spec["gamma"]) * forget_loss).backward()
            del current, forget_loss
        for parameter in trainable:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError("RMU missing or nonfinite allowed gradient")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        tensors._project(runtime.model, reference, inner)
        event = {
            "phase": spec["mode"],
            "completed": step + 1,
            "total": 60,
            "unit": "steps",
            "retain_loss": retain_value,
            "retain_mse": retain_value,
            "forget_loss": forget_value,
            "forget_mse": forget_value,
            "gradient_norm": float(norm),
            "schedule_method": SCHEDULE_METHOD,
            "retain_index": retain_index,
            "retain_item_id": retain[retain_index]["id"],
            "retain_family": retain[retain_index]["family"],
            "forget_index": forget_schedule["indices"][step] if forget_schedule else None,
            "forget_item_id": forget_schedule["item_ids"][step] if forget_schedule else None,
            "forget_family": forget_schedule["families"][step] if forget_schedule else None,
        }
        if context:
            event.update(
                retain_variant=retain_variant,
                retain_mask=qualification["retain"][retain_index, retain_variant],
                forget_variant=forget_schedule["variant_indices"][step]
                if forget_schedule
                else None,
                forget_mask=qualification["forget"][
                    forget_schedule["indices"][step], forget_schedule["variant_indices"][step]
                ]
                if forget_schedule
                else None,
            )
        trace.append(event)
        watch.progress(
            spec["mode"], **{key: value for key, value in event.items() if key != "phase"}
        )
    optimizer.zero_grad(set_to_none=True)
    watch.progress("rmu_final_validation", state="active")
    diff = tensors.validate_state(runtime.model, reference, inner)
    watch.progress("rmu_training", state="complete", completed=60, total=60, unit="steps")
    return {
        "mode": spec["mode"],
        "trace": trace,
        "diff": diff,
        "schedule": schedule,
        "spec_sha256": record_digest(spec),
        "control_vector": control_record,
        "source_reference_fingerprint": reference["fingerprint"],
        "provenance": (
            {
                **PROVENANCE,
                "mask": CONTEXT_FIXED["mask_policy"],
                "variant_policy": CONTEXT_FIXED["variant_policy"],
                "adaptation": "Root-selected context-mask adaptation; literal user offsets exclude "
                "role/system tokens; all chosen variants qualified before optimization",
            }
            if context
            else PROVENANCE
        ),
    }
