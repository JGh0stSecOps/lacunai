"""Frozen NPO plus answer-position retain KL; numerical code never chooses candidates."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from typing import Any

from lacunai.experimental import tensors
from lacunai.map.localize import record_digest

ALLOWED = tuple(f"model.layers.{layer}.mlp.down_proj.weight" for layer in range(16, 24))
LEARNING_RATES = ("0.01", "0.05", "0.1", "0.2")
FIXED = {
    "schema_version": 1,
    "method": "npo_kl_candidate",
    "steps": 60,
    "optimizer": "SGD",
    "learning_rate": "0.01",
    "beta": "0.1",
    "alpha": "1.0",
    "gamma": "1.0",
    "gradient_clip": "1.0",
    "sequence_tokens": 128,
    "inner_tensor_relative_frobenius": "0.0195",
    "inner_model_relative_frobenius": "0.0024",
    "outer_tensor_relative_frobenius": "0.02",
    "outer_model_relative_frobenius": "0.0025",
    "retain_objective": "KL(reference||current):mean_answer_tokens",
}
PROVENANCE = {
    "npo": "https://github.com/locuslab/open-unlearning/blob/main/src/trainer/unlearn/npo.py",
    "kernels": "https://github.com/locuslab/open-unlearning/blob/main/src/trainer/utils.py",
    "retain_adaptation": "KL(reference||current), vocabulary sum then mean over causal answer "
    "positions including EOS; prompt positions excluded",
    "teacher": "same resident model temporarily restored to original allowed weights; no graph "
    "exists across teacher weight substitution",
    "head_compute": "Qwen2 logits_to_keep causal answer positions only; "
    "FP32 equivalence atol=2e-6 rtol=2e-5",
    "schedule": "family-round-robin:first-seen-family:within-family-order:v1",
    "npo_reduction": "sequence-SUM answer NLL including EOS, not mean token NLL",
}


SCHEDULE_METHOD = "family-round-robin:first-seen-family:within-family-order:v1"


def family_schedule(items: list[dict[str, Any]], steps: int) -> dict[str, Any]:
    """Cycle families equally, preserving original family and per-family item order."""
    if type(steps) is not int or steps < 1 or not items:
        raise ValueError("schedule requires positive steps and nonempty items")
    groups: dict[str, list[int]] = {}
    identifiers: set[str] = set()
    for index, item in enumerate(items):
        family, identifier = item.get("family"), item.get("id")
        if (
            not isinstance(family, str)
            or not family
            or not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
        ):
            raise ValueError("schedule requires nonempty families and unique item IDs")
        identifiers.add(identifier)
        groups.setdefault(family, []).append(index)
    families = list(groups)
    counts = dict.fromkeys(families, 0)
    indices = []
    for step in range(steps):
        family = families[step % len(families)]
        indices.append(groups[family][counts[family] % len(groups[family])])
        counts[family] += 1
    return {
        "method": SCHEDULE_METHOD,
        "family_order": families,
        "indices": indices,
        "item_ids": [items[index]["id"] for index in indices],
        "families": [items[index]["family"] for index in indices],
        "family_counts": counts,
    }


def validate_spec(spec: dict[str, Any], inner: dict[str, Any]) -> None:
    if set(spec) != set(FIXED) | {"mode", "allowed_tensors"}:
        raise ValueError("NPO specification fields differ from frozen contract")
    learning_rate = spec["learning_rate"]
    if type(learning_rate) is not str or learning_rate not in LEARNING_RATES:
        raise ValueError("NPO learning rate differs from frozen contract enum")
    for name, expected in {**FIXED, "learning_rate": learning_rate}.items():
        if type(spec[name]) is not type(expected) or spec[name] != expected:
            raise ValueError(f"NPO frozen specification differs: {name}")
    if spec["mode"] not in {"npo_kl", "retain_kl"} or spec["allowed_tensors"] != list(ALLOWED):
        raise ValueError("NPO mode or tensor allowlist differs")
    edits = inner.get("edits", {})
    if (
        edits.get("allowed_tensors") != list(ALLOWED)
        or edits.get("expected_shape") != [896, 4864]
        or str(edits.get("max_tensor_relative_frobenius"))
        != FIXED["inner_tensor_relative_frobenius"]
        or str(edits.get("max_model_relative_frobenius")) != FIXED["inner_model_relative_frobenius"]
    ):
        raise ValueError("NPO projection bounds/shape differ from frozen inner contract")


def answer_batch(tokenizer: Any, item: dict[str, Any], cap: int) -> dict[str, Any]:
    """Token construction matches experimental.tensors._answer_loss exactly."""
    torch = import_module("torch")
    if (
        not isinstance(item.get("prompts"), list)
        or not item["prompts"]
        or not isinstance(item["prompts"][0], str)
        or not isinstance(item.get("answer"), str)
        or not item["answer"]
        or type(cap) is not int
        or cap < 1
        or tokenizer.eos_token_id is None
    ):
        raise ValueError("invalid bounded answer example")
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": item["prompts"][0]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full = tokenizer(prefix + item["answer"], add_special_tokens=False)["input_ids"]
    if not prefix_ids or full[: len(prefix_ids)] != prefix_ids or len(full) <= len(prefix_ids):
        raise ValueError("training answer boundary changed under tokenization")
    full.append(tokenizer.eos_token_id)
    if len(full) > cap:
        raise ValueError("training example exceeds token cap")
    ids = torch.tensor([full], dtype=torch.long)
    labels = ids.clone()
    labels[:, : len(prefix_ids)] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def _answer_logits(
    logits: Any, batch: dict[str, Any], *, selected_positions: bool = False
) -> tuple[Any, Any]:
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    if selected_positions:
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != int(mask.sum()):
            raise ValueError("selected answer logits dimensions differ")
        if not bool(mask.any()):
            raise ValueError("answer mask is empty")
        return logits[0].float(), labels[mask]
    if (
        logits.ndim != 3
        or list(logits.shape[:2]) != list(batch["input_ids"].shape)
        or not bool(mask.any())
    ):
        raise ValueError("answer logits/labels dimensions differ or answer mask is empty")
    selected = logits[:, :-1, :][mask].float()
    return selected, labels[mask]


def sequence_nll(logits: Any, batch: dict[str, Any], *, selected_positions: bool = False) -> Any:
    torch = import_module("torch")
    if batch["input_ids"].shape[0] != 1:
        raise ValueError("NPO kernel requires microbatch one")
    selected, targets = _answer_logits(logits, batch, selected_positions=selected_positions)
    loss = torch.nn.functional.cross_entropy(selected, targets, reduction="sum")
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite sequence answer NLL")
    return loss


def npo_loss(current_nll: Any, reference_nll: Any, beta: float) -> Any:
    torch = import_module("torch")
    if beta != 0.1:
        raise ValueError("NPO beta differs from frozen contract")
    reference = torch.as_tensor(
        reference_nll, dtype=current_nll.dtype, device=current_nll.device
    ).detach()
    if reference.shape != current_nll.shape or not bool(torch.isfinite(reference).all()):
        raise ValueError("reference NPO loss shape/value differs")
    result = (-2.0 / beta * torch.nn.functional.logsigmoid(beta * (current_nll - reference))).mean()
    if not bool(torch.isfinite(result)):
        raise ValueError("nonfinite NPO loss")
    return result


def retain_kl(
    student_logits: Any,
    teacher_log_probs: Any,
    batch: dict[str, Any],
    *,
    selected_positions: bool = False,
) -> Any:
    torch = import_module("torch")
    selected, _ = _answer_logits(student_logits, batch, selected_positions=selected_positions)
    current = torch.nn.functional.log_softmax(selected, dim=-1)
    if current.shape != teacher_log_probs.shape or not bool(
        torch.isfinite(teacher_log_probs).all()
    ):
        raise ValueError("retain teacher probabilities differ from answer positions")
    # PyTorch kl_div(log_current, log_reference) computes KL(reference || current).
    loss = (
        torch.nn.functional.kl_div(
            current, teacher_log_probs.detach(), reduction="none", log_target=True
        )
        .sum(-1)
        .mean()
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite retain KL")
    return loss


@contextmanager
def original_teacher(model: Any, reference: dict[str, Any]) -> Iterator[None]:
    """Caller must finish all backward graphs before entering this context."""
    torch = import_module("torch")
    parameters = dict(model.named_parameters())
    originals = reference["original_tensors"]
    if not originals or any(name not in parameters for name in originals):
        raise ValueError("teacher source tensor inventory differs")
    for name, original in originals.items():
        current = parameters[name]
        if (
            list(current.shape) != list(original.shape)
            or current.dtype != original.dtype
            or current.device.type != "cpu"
            or original.device.type != "cpu"
        ):
            raise ValueError("teacher source shape/dtype/device differs")
    # Complete all saves before mutating anything, including failure during allocation.
    saved = {name: parameters[name].detach().clone() for name in originals}
    try:
        with torch.no_grad():
            for name, original in originals.items():
                parameters[name].copy_(original)
            yield
    finally:
        with torch.no_grad():
            for name, current in saved.items():
                parameters[name].copy_(current)
            if any(not torch.equal(parameters[name], current) for name, current in saved.items()):
                raise ValueError("teacher context failed exact current-weight restoration")


def answer_forward(model: Any, batch: dict[str, Any]) -> Any:
    """Qwen2 4.51.3 computes the head only at causal answer prediction positions."""
    torch = import_module("torch")
    if getattr(model.config, "model_type", None) != "qwen2" or batch["input_ids"].shape[0] != 1:
        raise ValueError("selected answer head requires Qwen2 microbatch one")
    positions = torch.nonzero(batch["labels"][0, 1:] != -100, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("selected answer head has no answer positions")
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        logits_to_keep=positions,
    ).logits


def teacher_log_probs(model: Any, reference: dict[str, Any], batch: dict[str, Any]) -> Any:
    torch = import_module("torch")
    with original_teacher(model, reference):
        logits = answer_forward(model, batch)
        selected, _ = _answer_logits(logits, batch, selected_positions=True)
        return torch.nn.functional.log_softmax(selected, dim=-1).detach()


def _check(runtime: Any, watch: Any) -> None:
    watch.check()
    runtime.check()


def train(
    runtime: Any,
    reference: dict[str, Any],
    corpus: dict[str, Any],
    inner: dict[str, Any],
    spec: dict[str, Any],
    watch: Any,
) -> dict[str, Any]:
    watch.progress("npo_kl_training", state="active", completed=0, total=60, unit="steps")
    try:
        return _train(runtime, reference, corpus, inner, spec, watch)
    except BaseException as exc:
        watch.progress("npo_kl_training", state="failed", error=f"{type(exc).__name__}: {exc}")
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
        raise ValueError("NPO requires frozen forget64 and retain80 training records")
    retain_schedule = family_schedule(retain, spec["steps"])
    planned_forget_schedule = family_schedule(
        forget, spec["steps"] if spec["mode"] == "npo_kl" else 1
    )
    if (
        len(retain_schedule["family_order"]) != 5
        or len(planned_forget_schedule["family_order"]) != 4
    ):
        raise ValueError("NPO requires five retain and four forget families")
    # A retain-only control has no forget update schedule, despite reference precomputation.
    forget_schedule = planned_forget_schedule if spec["mode"] == "npo_kl" else None
    schedule = {"method": SCHEDULE_METHOD, "retain": retain_schedule, "forget": forget_schedule}
    if set(reference["original_tensors"]) != set(ALLOWED):
        raise ValueError("NPO reference tensor allowlist differs")
    if tensors.validate_state(runtime.model, reference, inner)["changed_tensor_count"]:
        raise ValueError("NPO must begin from the untouched reference source")
    parameters = dict(runtime.model.named_parameters())
    for name, parameter in parameters.items():
        if parameter.requires_grad:
            raise ValueError("NPO expects initially frozen runtime parameters")
        if name in ALLOWED and (parameter.dtype != torch.float32 or parameter.device.type != "cpu"):
            raise ValueError("NPO requires CPU float32 allowed tensors")
    runtime.model.eval()
    # Cache token batches, not logits: teacher reference NLLs are scalar records only.
    forget_batches = [
        answer_batch(runtime.tokenizer, item, spec["sequence_tokens"]) for item in forget
    ]
    retain_batches = [
        answer_batch(runtime.tokenizer, item, spec["sequence_tokens"]) for item in retain
    ]
    cached = []
    watch.progress(
        "npo_reference_forget", state="active", completed=0, total=len(forget), unit="examples"
    )
    with torch.no_grad():
        for index, batch in enumerate(forget_batches):
            _check(runtime, watch)
            logits = answer_forward(runtime.model, batch)
            value = float(sequence_nll(logits, batch, selected_positions=True))
            cached.append(
                {
                    "index": index,
                    "item_id": forget[index]["id"],
                    "family": forget[index]["family"],
                    "nll_sum": value,
                    "answer_tokens": int((batch["labels"][:, 1:] != -100).sum()),
                    "input_tokens": int(batch["input_ids"].shape[-1]),
                    "token_batch_sha256": record_digest(
                        {"ids": batch["input_ids"].tolist(), "labels": batch["labels"].tolist()}
                    ),
                }
            )
            del logits
            watch.progress(
                "npo_reference_forget",
                completed=index + 1,
                total=len(forget),
                unit="examples",
                reference_nll_sum=value,
            )
    for name in ALLOWED:
        parameters[name].requires_grad_(True)
    trainable = [parameters[name] for name in ALLOWED]
    optimizer = torch.optim.SGD(trainable, lr=float(spec["learning_rate"]), momentum=0.0)
    trace = []
    for step in range(spec["steps"]):
        _check(runtime, watch)
        optimizer.zero_grad(set_to_none=True)
        retain_index = retain_schedule["indices"][step]
        retain_batch = retain_batches[retain_index]
        # No student graph is alive across the temporary teacher copy.
        teacher = teacher_log_probs(runtime.model, reference, retain_batch)
        student = answer_forward(runtime.model, retain_batch)
        kl = retain_kl(student, teacher, retain_batch, selected_positions=True)
        retain_value = float(kl.detach())
        (float(spec["alpha"]) * kl).backward()
        del student, teacher, kl
        forget_nll = None
        forget_objective = None
        if spec["mode"] == "npo_kl":
            if forget_schedule is None:
                raise ValueError("NPO forget schedule is missing")
            index = forget_schedule["indices"][step]
            batch = forget_batches[index]
            logits = answer_forward(runtime.model, batch)
            nll = sequence_nll(logits, batch, selected_positions=True)
            objective = npo_loss(nll, cached[index]["nll_sum"], float(spec["beta"]))
            forget_nll, forget_objective = float(nll.detach()), float(objective.detach())
            (float(spec["gamma"]) * objective).backward()
            del logits, nll, objective
        for parameter in trainable:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError("NPO missing or nonfinite allowed gradient")
        norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(spec["gradient_clip"]), error_if_nonfinite=True
        )
        optimizer.step()
        tensors._project(runtime.model, reference, inner)
        event = {
            "phase": spec["mode"],
            "completed": step + 1,
            "total": spec["steps"],
            "unit": "steps",
            "retain_loss": retain_value,
            "retain_kl": retain_value,
            "forget_loss": forget_objective,
            "forget_nll_sum": forget_nll,
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
    diff = tensors.validate_state(runtime.model, reference, inner)
    watch.progress(
        "npo_kl_training", state="complete", completed=len(trace), total=spec["steps"], unit="steps"
    )
    return {
        "mode": spec["mode"],
        "trace": trace,
        "diff": diff,
        "reference_forget_nll": cached,
        "schedule": schedule,
        "spec_sha256": record_digest(spec),
        "provenance": PROVENANCE,
        "limits": "Constrained objective optimization does not establish knowledge absence. "
        "Source-precision roundtrip and independent collateral tests remain mandatory.",
    }
