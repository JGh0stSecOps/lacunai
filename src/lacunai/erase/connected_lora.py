"""Ephemeral rank-eight adapter for the single permitted planting tensor."""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from typing import Any


class Adapter:
    def __init__(self, module: Any, original: Any) -> None:
        torch = import_module("torch")
        self.original = original
        self.enabled = True
        generator = torch.Generator(device="cpu").manual_seed(42)
        self.a = torch.nn.Parameter(
            torch.empty((8, original.shape[1]), dtype=torch.float32, device="cpu")
        )
        self.b = torch.nn.Parameter(
            torch.zeros((original.shape[0], 8), dtype=torch.float32, device="cpu")
        )
        torch.nn.init.kaiming_uniform_(self.a, a=5**0.5, generator=generator)
        self.handle = module.register_forward_hook(self.forward)

    @property
    def parameters(self) -> list[Any]:
        return [self.a, self.b]

    def forward(self, module: Any, inputs: Any, output: Any) -> Any:
        if not self.enabled:
            return output
        functional = import_module("torch.nn.functional")
        return output + functional.linear(functional.linear(inputs[0], self.a), self.b)

    @contextmanager
    def disabled(self) -> Iterator[None]:
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous

    def merged(self) -> Any:
        torch = import_module("torch")
        if any(not bool(torch.isfinite(p).all()) for p in self.parameters):
            raise ValueError("nonfinite adapter factors")
        return self.original + self.b @ self.a

    def project(self) -> dict[str, float]:
        from lacunai.erase.connected_training import project

        torch = import_module("torch")
        merged = self.merged().detach()
        evidence = project(merged, self.original)
        with torch.no_grad():
            self.b.mul_(evidence["projection_scale"])
        # Check the actual factor product after rounding/scaling, not a zero base delta.
        from lacunai.erase.factoid_gradient import _relative

        evidence["post_projection_relative"] = _relative(self.merged().detach(), self.original)
        return evidence

    def close(self) -> None:
        self.handle.remove()
        for parameter in self.parameters:
            parameter.grad = None
