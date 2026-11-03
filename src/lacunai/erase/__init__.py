"""Allowlisted numerical interventions, with provenance and bounded changes."""

from .methods import METHOD_INFO, apply_method, touched_tensors

__all__ = ["METHOD_INFO", "apply_method", "touched_tensors"]
