"""HF runtime using the original pilot's strict loader and numerical kernels."""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any

from lacunai.experimental import tensors

from .spec import RuntimeConfig, bounded_file, digest


def progress(phase: str, **fields: Any) -> None:
    print(json.dumps({"phase": phase, **fields}, allow_nan=False), file=sys.stderr, flush=True)


@contextmanager
def active_phase(phase: str) -> Iterator[None]:
    """Keep silent numerical/library operations visibly active without invented totals."""
    started = time.monotonic()
    stopped = threading.Event()

    def pulse() -> None:
        while not stopped.wait(5):
            progress(phase, state="active", elapsed_seconds=round(time.monotonic() - started, 3))

    progress(phase, state="active", elapsed_seconds=0)
    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        yield
    except BaseException:
        progress(phase, state="failed", elapsed_seconds=round(time.monotonic() - started, 3))
        raise
    else:
        progress(phase, state="complete", elapsed_seconds=round(time.monotonic() - started, 3))
    finally:
        stopped.set()
        thread.join()


def checkpoint_inventory(path: Path, max_bytes: int) -> dict[str, str]:
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_dir():
        raise ValueError("checkpoint must be a local unredirected directory")
    files: dict[str, str] = {}
    total = 0
    candidates = sorted(path.rglob("*"))
    if len(candidates) > 4096:
        raise ValueError("checkpoint has too many entries")
    progress("checkpoint_hashing", completed=0, total=len(candidates), unit="entries")
    for number, item in enumerate(candidates, 1):
        if item.is_symlink():
            raise ValueError("linked checkpoint entry refused")
        if item.is_dir():
            continue
        if not item.is_file() or item.suffix in {".bin", ".pt", ".pth", ".py", ".pkl"}:
            raise ValueError("only local safetensors checkpoints with inert metadata are supported")
        total += item.stat().st_size
        if total > max_bytes:
            raise ValueError("checkpoint exceeds declared byte limit")
        with active_phase("checkpoint_file_hashing"):
            files[str(item.relative_to(path))] = tensors.file_hash(item)
        progress("checkpoint_hashing", completed=number, total=len(candidates), unit="entries")
    if not any(name.endswith(".safetensors") for name in files) or "config.json" not in files:
        raise ValueError("checkpoint requires config.json and safetensors weights")
    return files


class HFRuntime:
    """No hosted APIs, remote code, implicit downloads, or model-generated operations."""

    def __init__(self, path: Path, config: RuntimeConfig | None = None) -> None:
        self.path = path.absolute()
        self.config = config or RuntimeConfig()
        self.started = time.monotonic()
        self.inventory = checkpoint_inventory(self.path, self.config.max_model_bytes)
        self.fingerprint = digest(self.inventory)
        model_config = json.loads(bounded_file(self.path / "config.json"))
        if model_config.get("model_type") not in {"qwen2", "llama"} or model_config.get("auto_map"):
            raise ValueError("only built-in Qwen2 and Llama gated-MLP architectures are supported")
        if model_config.get("quantization_config"):
            raise ValueError("quantized checkpoint formats are not supported")
        self.contract = {
            "target": {
                "files": [
                    {"path": name, "sha256": sha, "size": (self.path / name).stat().st_size}
                    for name, sha in self.inventory.items()
                ]
            },
            "runtime": {
                "device": self.config.device,
                "dtype": self.config.dtype,
                "seed": self.config.seed,
                "torch_threads": self.config.torch_threads,
                "interop_threads": 1,
            },
            "resources": {
                "target_prompt_tokens": self.config.max_prompt_tokens,
                "target_output_tokens": self.config.max_new_tokens,
            },
            "edits": {"allowed_tensors": [], "expected_shape": []},
        }
        self.progress("model_loading", state="active")
        with active_phase("checkpoint_loading"):
            self.model, self.tokenizer = tensors.load_target(self.path, self.contract)
        self.storage_dtypes: dict[str, str] = {}
        for name in self.inventory:
            if name.endswith(".safetensors"):
                with import_module("safetensors").safe_open(
                    self.path / name, framework="pt", device="cpu"
                ) as stored:
                    for key in stored.keys():  # noqa: SIM118 — safe_open is not a mapping.
                        if key in self.storage_dtypes:
                            raise ValueError("duplicate stored tensor name")
                        self.storage_dtypes[key] = stored.get_slice(key).get_dtype()
        self.layer_ids = list(range(len(self.model.model.layers)))
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.check()
        self.progress("model_loading", state="complete", checkpoint=self.fingerprint)

    def progress(self, phase: str, **fields: Any) -> None:
        progress(phase, elapsed_seconds=round(time.monotonic() - self.started, 3), **fields)

    def check(self) -> None:
        if (
            self.config.max_seconds is not None
            and time.monotonic() - self.started > self.config.max_seconds
        ):
            raise TimeoutError("declared model-operation runtime exhausted")

    def mlp(self, layer: int) -> Any:
        if type(layer) is not int or layer not in self.layer_ids:
            raise ValueError("layer is outside this checkpoint")
        return self.model.model.layers[layer].mlp

    def encode(self, prompt: str) -> Any:
        self.check()
        return tensors._prompt_ids(self.tokenizer, prompt, self.config.max_prompt_tokens)

    def generate(self, prompt: str) -> dict[str, Any]:
        self.check()
        with active_phase("generation"):
            result = tensors.generate(self.model, self.tokenizer, prompt, self.contract)
        self.check()
        return result

    def loss(self, prompt: str, answer: str) -> Any:
        self.check()
        return tensors._answer_loss(
            self.model,
            self.tokenizer,
            {"prompts": [prompt], "answer": answer},
            self.config.max_prompt_tokens,
        )

    def next_logprob(self, prompt: str, answer: str) -> float:
        torch = import_module("torch")
        ids = self.encode(prompt)
        prefix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        full = self.tokenizer(prefix + answer, add_special_tokens=False)["input_ids"]
        if full[: ids.shape[-1]] != ids[0].tolist() or len(full) <= ids.shape[-1]:
            raise ValueError("answer tokenization boundary changed")
        with torch.no_grad():
            logits = (
                self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                .logits[0, -1]
                .float()
            )
            value = torch.log_softmax(logits, dim=-1)[full[ids.shape[-1]]]
        if not bool(torch.isfinite(value)):
            raise ValueError("nonfinite answer log probability")
        self.check()
        return float(value)

    def perplexity(self, texts: list[str]) -> dict[str, Any]:
        """Token-weighted NLL; full non-overlapping examples, no averaged perplexities."""
        torch = import_module("torch")
        total_loss, count = 0.0, 0
        rows = []
        for number, value in enumerate(texts, 1):
            self.check()
            ids = self.tokenizer(value, return_tensors="pt", add_special_tokens=True)["input_ids"]
            if not 2 <= ids.shape[-1] <= self.config.max_prompt_tokens:
                raise ValueError("perplexity text outside declared token bounds")
            with torch.no_grad():
                loss = float(
                    self.model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        labels=ids,
                        use_cache=False,
                    ).loss
                )
            if not math.isfinite(loss):
                raise ValueError("nonfinite perplexity loss")
            tokens = ids.shape[-1] - 1
            total_loss += loss * tokens
            count += tokens
            rows.append({"text": value, "nll_sum": loss * tokens, "tokens": tokens})
            self.progress("perplexity", completed=number, total=len(texts), unit="texts")
        if not count or total_loss / count > 80:
            raise ValueError("undefined or overflowing perplexity")
        return {
            "nll_sum": total_loss,
            "tokens": count,
            "mean_nll": total_loss / count,
            "perplexity": math.exp(total_loss / count),
            "examples": rows,
            "method": "token-weighted next-token NLL; each bounded text scored independently",
        }
