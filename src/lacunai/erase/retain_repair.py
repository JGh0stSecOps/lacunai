"""Eight retain-only SGD updates shared with the benign-relearning check.

Updates remain in memory on success. Callers own persistence, evaluation and
restoration after evaluation. On failure this helper restores its starting tensors.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from importlib import import_module
from typing import Any

from lacunai.core.model import active_phase
from lacunai.experimental import tensors

STEPS = 8


def repair(
    runtime: Any,
    names: list[str],
    retain_development: list[dict[str, str]],
    *,
    reference: dict[str, Any] | None = None,
    check: Callable[[], Any] | None = None,
    on_step: Callable[[list[dict[str, Any]]], Any] | None = None,
    bound_tolerance: float = 1e-9,
) -> list[dict[str, Any]]:
    torch = import_module("torch")
    parameters = dict(runtime.model.named_parameters())
    if (
        not names
        or len(set(names)) != len(names)
        or any(n not in parameters for n in names)
        or not retain_development
        or bound_tolerance not in (0, 1e-9)
    ):
        raise ValueError("invalid retain-repair parameter/data/bound contract")
    for row in retain_development:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("prompt", "answer")):
            raise ValueError("invalid retain-repair development row")
    if any(p.grad is not None for p in parameters.values()):
        raise ValueError("retain repair refuses preexisting gradients")
    if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in parameters.values()):
        raise ValueError("retain repair requires CPU float32")
    originals = {n: parameters[n].detach().clone() for n in names}
    bounds = originals if reference is None else reference
    if set(bounds) != set(names):
        raise ValueError("retain-repair bound reference names differ")
    for n in names:
        if (
            bounds[n].shape != parameters[n].shape
            or bounds[n].dtype != torch.float32
            or bounds[n].device.type != "cpu"
            or tensors._square_norm(bounds[n]) <= 0
        ):
            raise ValueError("invalid retain-repair bound reference")
    hashes = {
        n: tensors.tensor_hash(t) for n, t in runtime.model.state_dict().items() if n not in names
    }
    flags = {n: p.requires_grad for n, p in parameters.items()}
    trace: list[dict[str, Any]] = []
    success = False
    runtime.progress("benign_relearning", state="active", completed=0, total=STEPS, unit="steps")
    try:
        for n, p in parameters.items():
            p.requires_grad_(n in names)
        optimizer = torch.optim.SGD([parameters[n] for n in names], lr=0.05)
        for step in range(STEPS):
            if check is not None:
                check()
            item = retain_development[step % len(retain_development)]
            with active_phase("benign_relearning"):
                optimizer.zero_grad(set_to_none=True)
                loss = runtime.loss(item["prompt"], item["answer"])
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite benign-relearning loss")
                loss.backward()
                for n in names:
                    grad = parameters[n].grad
                    if grad is None or not bool(torch.isfinite(grad).all()):
                        raise ValueError("invalid benign-relearning gradient")
                torch.nn.utils.clip_grad_norm_(
                    [parameters[n] for n in names], 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                for n in names:
                    relative = math.sqrt(
                        tensors._square_norm(parameters[n] - bounds[n])
                        / tensors._square_norm(bounds[n])
                    )
                    if not math.isfinite(relative) or relative > 0.25 + bound_tolerance:
                        raise ValueError("benign relearning exceeds relative-norm bound")
                trace.append({"step": step + 1, "loss": float(loss.detach())})
                if on_step is not None:
                    on_step(trace)
            runtime.progress("benign_relearning", completed=step + 1, total=STEPS, unit="steps")
        actual = {
            n: tensors.tensor_hash(t)
            for n, t in runtime.model.state_dict().items()
            if n not in names
        }
        if actual != hashes:
            raise ValueError("retain repair changed non-target state")
        success = True
    except BaseException as exc:
        runtime.progress("benign_relearning", state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        primary = sys.exception()
        try:
            if not success:
                with torch.no_grad():
                    for n in names:
                        parameters[n].copy_(originals[n])
            for n, p in parameters.items():
                p.requires_grad_(flags[n])
                p.grad = None
            if not success and any(not torch.equal(parameters[n], originals[n]) for n in names):
                raise RuntimeError("retain-repair failure restoration differs")
        except BaseException as exc:
            if primary is not None:
                raise BaseExceptionGroup("repair and restoration failed", [primary, exc]) from None
            raise
    return trace
