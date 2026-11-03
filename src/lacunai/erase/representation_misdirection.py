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
    if set(spec) != set(FIXED) | {"mode", "allowed_tensors"}:
        raise ValueError("RMU specification fields differ from frozen contract")
    for name, expected in FIXED.items():
        if type(spec[name]) is not type(expected) or spec[name] != expected:
            raise ValueError(f"RMU frozen specification differs: {name}")
    if any(type(beta) is not float for beta in spec["betas"]):
        raise ValueError("RMU betas must be exact JSON floats")
    if spec["mode"] not in ("rmu", "retain_mse") or spec["allowed_tensors"] != list(ALLOWED):
        raise ValueError("RMU mode or tensor allowlist differs")
    edits = inner.get("edits", {})
    if (
        edits.get("allowed_tensors") != list(ALLOWED)
        or edits.get("expected_shape") != [896, 4864]
        or str(edits.get("max_tensor_relative_frobenius")) != "0.0195"
        or str(edits.get("max_model_relative_frobenius")) != "0.0024"
    ):
        raise ValueError("RMU projection bounds/shape differ from frozen inner contract")


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
    mask = labels != -100
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
    forget_schedule = planned_forget if spec["mode"] == "rmu" else None
    schedule = {"method": SCHEDULE_METHOD, "retain": retain_schedule, "forget": forget_schedule}
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
    batches: dict[str, list[Any]] = {}
    for split, items in (("forget", forget), ("retain", retain)):
        batches[split] = []
        watch.progress(
            "rmu_tokenization", split=split, completed=0, total=len(items), unit="examples"
        )
        for index, item in enumerate(items):
            watch.check()
            runtime.check()
            batches[split].append(answer_batch(runtime.tokenizer, item, 128))
            watch.progress(
                "rmu_tokenization",
                split=split,
                completed=index + 1,
                total=len(items),
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
        batch = batches["retain"][retain_index]
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
            current = capture_hidden(runtime.model, batches["forget"][forget_index])
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
        "provenance": PROVENANCE,
    }
