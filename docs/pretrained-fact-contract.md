# Existing pretrained fact experiment

Status: prepared source; no model result yet. Target: **Paris is the capital of
France**, as already represented in Qwen2.5-0.5B-Instruct. No planting, LoRA or
other pre-erasure fine-tuning is part of this experiment.

The opt-in `pretrained-fact` profile in `scripts/connected_fact_pilot.py` reuses
the existing local model loader, mapping, snapshot transactions, editing service,
evaluation and signing. The schema-one adapter validates separate development,
qualification and final evaluation inputs. Dry-run remains the default.

The original checkpoint must recover every qualification target and answer every
qualification retain question correctly. Its read-only bytes are materialized in
the existing isolated scratch checkpoint, with a snapshot preserving the scratch's
prior contents. Repeated no-op generations must equal the original's generations.

The initial method cohort remains `npo-kl-repair` with eight base updates and
`grad-diff-kl-repair2` with four base updates. These reuse the existing additional
retain-repair schedules. Both require measured positive causal evidence in the
allowed final-layer up projection; the experiment cannot manufacture localization
if that condition fails. Each starts from the same original weights. Candidate
tensor files are frozen before final probes run. The separate benign-relearning
test uses eight retain-only updates in memory.

Acceptance requires all final baseline probes to qualify, no final target
recovery, no paired retain losses, and held-out perplexity at most 1.15 times the
original baseline. Recovery is checked again after benign relearning. Frozen
lexical aliases include both `Paris` and `París`: recovery through either spelling
fails the contract. Ambiguous or missing observations cannot pass. The canonical
baseline scorer remains stricter than this supplemental alias check; an alternate
spelling alone can therefore make baseline qualification inconclusive.

Final probes include direct, paraphrased, indirect and adversarial extraction,
plus French and Spanish questions. Retain probes cover neighboring facts and
unrelated abilities. Prompt wording is held separately from editor development
examples. The first cohort has no adaptive candidate selection from final scores.

There is **no matched retain-only pretraining reference**: the original pretraining
corpus and a full retraining run are unavailable within this experiment. This
limits distributional-unlearning claims. Hidden-state changes and lexical
non-recovery do not establish information absence. The outcome is failed,
inconclusive, or passed the stated experimental contract; any pass means only
reduced recoverability under these tests.

The frozen execution batch declares the exact code, inputs, inference settings,
model/tokenizer revision and resource limits before execution. The existing
resource-pressure protections remain active. Original and backup checkpoints
retain the original information; neither is deleted or uploaded. Every applied
candidate and the scratch materialization must restore bit-for-bit.
