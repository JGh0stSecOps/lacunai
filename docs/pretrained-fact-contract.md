# Existing pretrained fact experiment

Status: calibrated baseline qualified; all-layer diagnosis selected layer 21 after the initial final-layer refusal. No forgetting edit has completed. Target: **Paris is the capital of
France**, as already represented in Qwen2.5-0.5B-Instruct. No planting, LoRA or
other pre-erasure fine-tuning is part of this experiment.

The first run recovered all 8 qualification target answers (4 direct and 4 paraphrased), with 7/8 retain answers accepted. The remaining retain response was a factually correct full sentence naming the Moon, while the frozen expected answer was the single word `Moon`. The whole-answer scorer rejected the wording; this is not evidence of missing knowledge. The scientific erasure outcome is **inconclusive**: neither method, no-op comparison, final held-out evaluation nor benign relearning ran. No weights were edited. Independent readback verified 8 signed evidence files, 73 code files and exact inventories of all 9 original and 9 scratch files.

The second run applied that qualification-only expected-answer calibration; the scorer, thresholds and final held-out probes stayed unchanged. Original, source-materialization and no-op evaluations each scored 8/8 target and 8/8 retain, with matching generations and perplexity. Localization then refused the first attempted method: one valid mapping pair at layer 23 had effect −2.366231918334961 and selected no neurons. This is a first-answer-token intervention result, not evidence that the fact is absent elsewhere. Zero optimized or evaluated candidates resulted; erasure remains inconclusive.

Both materialization snapshots were restored, the journal was empty, and all 9 original and 9 scratch files matched independently checked expected inventories. The audit verified 29 signed files and 73 code files. One of 16 host samples recorded I/O-pressure and paging violations; stable process identities do not establish no interference. This attempt did not reach all-layer diagnosis; its subsequent result is reported separately below. The [first baseline](../research/results/pretrained-fact-baseline-2026-11-04.json) and [localization aggregate](../research/results/pretrained-fact-localization-2026-11-04.json) preserve both outcomes.

The subsequent read-only diagnosis tested all 24 layers and found positive intervention effects in seven. Layer 21 was strongest; three of eight tested neurons had positive effects: 4090, 1507 and 535. Independent verification checked signed mapping artifacts, raw-score arithmetic and unchanged original/cache inventories. This evidence comes from one development prompt pair and does not establish general localization or erasure. See the [all-layer mapping aggregate](../research/results/pretrained-fact-mapping-2026-11-04.json). The earlier final-layer cohort and its numerical contract below remain historical; a later edit specification must explicitly name its measured tensors and methods.

The opt-in `pretrained-fact` profile in `scripts/connected_fact_pilot.py` reuses
the existing local model loader, mapping, snapshot transactions, editing service,
evaluation and signing. The schema-one adapter validates separate development,
qualification and final evaluation inputs. Dry-run remains the default.

The original checkpoint must recover every qualification target and answer every
qualification retain question correctly. Its read-only bytes are materialized in
the existing isolated scratch checkpoint, with a snapshot preserving the scratch's
prior contents. Repeated no-op generations must equal the original's generations.

The prepared `scripts/pretrained_localized_pilot.py` cohort binds the signed all-layer diagnosis to layer 21. It compares `ablate-neurons` on the three selected up-projection rows with `ablate-direction` on the down-projection matrix. Both use strength 1 and a maximum relative tensor change of 0.25. The direction is a target-versus-background mean difference, not an independently proven causal direction. This cohort is implemented but unexecuted; no efficacy result is implied.

The localized runner reuses the existing operators, checkpoint transactions, raw scoring, signing and benign-relearning check. Its finite storage ledger covers five snapshots and two candidate afterimages, plus bounded metadata. Both candidates must be frozen before final probes, and interrupted recovery restores a child transaction before the outer scratch materialization. Dry-run is the default; `--apply` is required for execution or recovery. The execution batch must be separately frozen before use.

The historical initial method cohort used `npo-kl-repair` with eight base updates and
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
