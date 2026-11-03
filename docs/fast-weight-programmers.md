# Fast-weight programmers and LacunAI

Sources inspected on 2026-11-03 UTC. This is research context and an unimplemented proposal; it changes neither the frozen pilot nor its acceptance contract.

Schmidhuber's [author overview](https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html), dated 26 March 2021 and marked updated November 2025, points to the original work. Its broad historical priority claims are the author's interpretation. The narrower technical connection below follows the papers, without equating every Transformer with the original architecture.

## A controller learns to write a weight memory

The [1991 technical report FKI-147-91](https://people.idsia.ch/~juergen/FKI-147-91ocr.pdf), dated 26 March 1991, defines two networks. Gradient descent trains the slow network's parameters to produce input-dependent changes to the fast network's weights. Those fast weights retain sequence information, so a feedforward network can answer differently after different histories. The controller learns what to store through errors propagated across the weight-update dynamics; the fast memory is part of the forward computation, not a separately fitted checkpoint.

The [1992 Neural Computation paper, 4(1):131–139](https://sferics.idsia.ch/pub/juergen/fastweights.pdf), also available as [author-hosted HTML](https://people.idsia.ch/~juergen/fastweights/node2.html), describes both one controller output per fast weight and a smaller FROM/TO interface. Products of FROM and TO activations supply weight increments. With a column-vector convention, the additive case is `M_next = M + v kᵀ`; the original formulation also permits differentiable squashing. This is learned control of associative memory: the controller constructs addressing and content patterns. Its temporary-variable-binding experiment remembers a changing parking location. That demonstrates usable temporal storage, not deletion of pretrained knowledge.

## Outer products and delta updates

Schlag, Irie and Schmidhuber's [2021 paper](https://arxiv.org/pdf/2102.11174v3) was first submitted on 22 February; the inspected version is v3, 9 June 2021 ([ICML/PMLR publication](https://proceedings.mlr.press/v139/schlag21a.html)). Sections 3–4 express linear attention as a fast-weight associative memory, with normalization accounted for. A query reads `M h_query`. Purely additive writes can interfere when keys overlap or memory capacity is exceeded.

Their delta rule retrieves `v_old = M h_key`, then writes `M_next = M + beta (v − v_old) h_keyᵀ`. The controller learns the write strength `beta`; the paper specifies key/query normalization. This corrects an association instead of indefinitely accumulating values. Retrieval, translation and language-modeling results do not establish robust unlearning.

The [released synthetic implementation](https://github.com/ischlag/fast-weight-transformers/blob/ebe13ea2e6409da91d12d22175f4f6efe8bf2f0a/synthetic/linearAttention.py), inspected at commit `ebe13ea2e6409da91d12d22175f4f6efe8bf2f0a`, computes these operations with tensors, including normalized keys and sigmoid write strengths. The repository provides synthetic experiment logs under an [MIT license](https://github.com/ischlag/fast-weight-transformers/blob/ebe13ea2e6409da91d12d22175f4f6efe8bf2f0a/LICENSE). This note uses original paraphrases; no implementation was imported.

## Connection and a bounded future hypothesis

The current Codex-directed pilot uses a different mechanism: an external LLM selects diagnostic column identifiers and fixed strength presets in a declarative specification. Existing [tensor code](../src/lacunai/experimental/tensors.py) attenuates allowed columns of pretrained Qwen weights. The editor is not trained end-to-end through this intervention, and the resulting checkpoint change persists across prompts. The shared idea is a controller directing weight changes; neither the architecture nor the learning process is identical.

A future operator could test a rank-one residual update to one allowed matrix:

`delta_W = eta (v_target − W k) kᵀ / (kᵀk + epsilon)`

This is a proposed adaptation, not the paper's exact rule. Tensor code would derive `k` and `v_target` from frozen synthetic development activations with compatible dimensions; the LLM would select evidence identifiers and an approved preset. Code would reject degenerate keys and enforce rank, norm, shape, dtype, finite-value, seed/tolerance, hash and checkpoint-reload requirements.

A new approval would freeze the operator, resource bounds and independent evaluation before execution. Preserve originals, no-op and established controls, withheld recovery probes, collateral tests and bounded benign relearning. Even successful local association suppression would not prove information absent elsewhere. The research claim could only be reduced recoverability under the stated tests; originals and backups retain the original information.
