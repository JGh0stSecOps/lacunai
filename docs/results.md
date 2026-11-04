# Experimental results

## Explicit repair — one method passed, two-method contract failed

The saved `npo-kl-repair` candidate passed direct evaluation, an independent reload, and a separate benign-relearning test. Target recovery was 0/8 and retain accuracy 8/8 in each arm, without ambiguous observations or paired retain losses. Its source-precision weights restored bit-for-bit and restored generations matched the planted baseline. This is reduced recoverability under these tests.

| Method | Base / repair updates | Direct and independent target / retain | After benign relearning target / retain | Direct PPL / planted | Overall method result |
| --- | --- | --- | --- | --- | --- |
| npo-kl-repair | 8 / 8 | 0/8 / 8/8 | 0/8 / 8/8 | 0.8377663406272934043903308152 | passed the stated experimental contract |
| grad-diff-kl-repair | 4 / 8 | 0/8 / 6/8 | 0/8 / 8/8 | 0.7500580232620554090938483920 | failed |

GradDiff's saved candidate failed the zero-collateral-loss requirement even though another eight retain-only updates repaired both answers in memory. The two-method cohort therefore failed. Persisting that additional repair is a new candidate, not a reinterpretation of this result.

Independent readback verified 102 signed evidence files and 55 source files at executed revision `7ebda14aa3bc7fe9bc6a8545cfba9de6066eb1c0`. Both methods have exact weight and generation restoration receipts; separate remote hashing verified all nine final scratch files and unchanged original/planted checkpoints. Preserved sparse candidate files matched their signed hashes. One of 61 host samples recorded resource violations, with 1,165 physical paging-out pages in total. Protected-worker identities remained continuous; absence of performance interference is not proven.

See the [repair result JSON](../research/results/planted-fact-repair-2026-11-04.json) for exact scores and evidence hashes. Aggregate feedback from the reused evaluation set guided this adaptation; these are exploratory results. The separately frozen multilingual evaluation remains unrun. The retain-only reference matches original planting, not the unlearning update count. Original checkpoints and backups retain the information, and planting damage remains distinct from edit collateral.

## Eight-step gradients — failed, with a retain-repair signal

Both methods completed eight updates within the unchanged weight-change bound. Baseline and no-op target/retain scores were 8/8; no-op generations matched exactly. Independent checkpoint reload reproduced each direct result.

| Method / arm | Confirmed target recovery | Ambiguous target responses | Retain correct | PPL / planted | Verdict |
| --- | --- | --- | --- | --- | --- |
| NPO direct and independent | 0/8 | 0 | 7/8 | 0.8154452446794253725374924200 | failed |
| NPO after benign relearning | 0/8 | 0 | 8/8 | 0.8375067053543070512217007539 | passed this arm only |
| GradDiff direct and independent | 0/8 | 8 | 4/8 | 0.9544499920945956237895942626 | failed |
| GradDiff after benign relearning | 0/8 | 8 | 6/8 | 0.8271011527266070415595556361 | failed |

NPO's saved candidate failed the zero-collateral-loss requirement. Its subsequent eight retain-only SGD updates repaired the observed loss without recovering the target, but that repaired state existed only in FP32 memory in this experiment. It was not a separately persisted and independently evaluated candidate. GradDiff's ambiguous responses cannot count as successful forgetting. Neither method passed the complete contract.

Both methods have exact weight and generation restoration receipts. Separate remote file hashing confirmed all nine final scratch files equal the planted checkpoint and the original source unchanged; preserved sparse candidate files matched their signed receipt hashes. Local readback verified 102 signed files and 53 source files for revision `19782d1ca97d7c86bcdaee434e471ba0c0402bdb`.

The monitor observed five samples with violations among 63 samples, with 9,406 physical paging-out pages in total. No three-consecutive-sample abort occurred. Worker identities stayed continuous; absence of performance interference is not proven. Runtime limits, model/tokenizer revisions and evaluation thresholds were unchanged.

The [eight-step result JSON](../research/results/planted-fact-eight-step-2026-11-04.json) contains exact scores and evidence hashes. Aggregate feedback from this reused evaluation set now informs subsequent candidates; later results on it are adaptive development evidence. The separately frozen multilingual supplement remains unrun. New explicit repair variants must pass saved-checkpoint evaluation and another benign-relearning test. Original checkpoints and backups retain the information.

## Gradient follow-up — failed at the weight-change bound

The qualified planted checkpoint was reused without further planting. Both gradient methods attempted 128 updates under the fixed 0.25 relative-change bound:

| Method | Completed updates | Rejected update | Edited checkpoint / evaluation |
| --- | --- | --- | --- |
| npo-kl | 18 | 19 | Not produced / not executed |
| grad-diff-kl | 10 | 11 | Not produced / not executed |

Both raised `ValueError: factoid update exceeds relative bound`. The overall contract **failed**; each method's erasure efficacy remains **inconclusive** because no edited checkpoint reached behavioral evaluation. No post-edit independent evaluation or benign-relearning arm ran. Completed counts were derived from collected progress events; the rejected update is the next sequential iteration, not an independently recorded final delta norm.

Fresh baseline and no-op checks both recovered 8/8 target probes with 8/8 retain accuracy and perplexity 142.7528471758874694472894703 over 122 tokens. No-op generations matched exactly. Each transaction restored the original planted state and reproduced its generations; restored perplexity divided by baseline perplexity is exactly 1 using `decimal.Decimal`. Separate final file rehashing confirmed all nine scratch files equal the planted checkpoint, with original source files unchanged. Earlier intermediate restoration is supported by signed runtime receipts and generation comparisons.

Independent readback verified Ed25519 integrity for 42 signed files and 53 code files at revision `a3796abfa6b5abef64def0f52fab59db6f08997e`. It does not certify scientific correctness. The reused retain-only reference was matched to the original 192-step planting, not the new unlearning updates. The two preceding operational attempts remain in the evidence history: one was stopped by physical-paging guards before candidate evaluation; the other failed a mandatory loader-test fixture before target loading. Neither produced a signed scientific aggregate.

See the [gradient bound result JSON](../research/results/planted-fact-gradient-bounds-2026-11-03.json) for exact aggregates and receipt digests. This adds no successful erasure result. The separate multilingual supplement remains unrun; retained originals and backups still contain the fact.

## Gradient follow-up — resource-aborted, inconclusive

The first gradient follow-up reproduced 8/8 target and 8/8 retain baseline scores, then copied an isolated scratch checkpoint. The physical host supervisor stopped its owned model container after three consecutive paging violations; neither NPO nor GradDiff executed. The container reported exit 137 with `OOMKilled=false`. No signed scientific aggregate was produced before interruption.

Terminal readback verified the original checkpoint unchanged and all nine scratch files equal to the planted baseline. The 3,179,224-byte metadata archive has SHA-256 `47beefbb15b864f56627037ffe61d15935460856ae09da5f60c5b7032cdb4c53`; it is transport-verified evidence, not a signed completed experiment. The frozen code was `d43ff05bd0c3ad7de4dfd175036ae3ffaa485095`.

The retry support explicitly reuses only a separately declared scratch checkpoint after complete hash and alias checks. It preserves prior evidence and avoids another full checkpoint copy. Loading requests the existing runtime’s low-memory initialization path; reduced peak memory remains unmeasured. Resource limits and evaluation thresholds are unchanged.

## Contextual planted-fact pilot — failed

Planting qualified in this attempt: the selected 4,864-row profile reached 64/64 development recovery, 8/8 withheld recovery and 8/8 retain accuracy after 192 updates. Saved-checkpoint reload matched the qualification generations. The matched retain-only reference also ran 192 updates and recovered 0/8 target probes with 8/8 retain accuracy; the no-op control matched planted-baseline generations.

**All three erasure methods failed.** Recovery persisted in both direct evaluation and independent reruns:

| Method | Direct recovery | Independent recovery | After benign relearning | Direct PPL / planted | Verdict |
| --- | --- | --- | --- | --- | --- |
| ablate-neurons | 8/8 | 8/8 | 7/8 | 1.1127 | failed |
| rank-one | 5/8 | 5/8 | 7/8 | 0.9310 | failed |
| ablate-direction | 8/8 | 8/8 | 8/8 | 1.1971 | failed |

All direct, independent and benign-relearning retain scores were 8/8. Direction ablation also exceeded the frozen 1.15 perplexity-ratio threshold. Rank-one reduced measured recall to 5/8, but recovery increased to 7/8 after eight bounded retain-only SGD updates. This does not satisfy zero recovery or demonstrate reliable deletion.

Planting itself increased held-out perplexity from 98.4732 to 142.7528, a factor of **1.449661327964025065273278508**. These values and candidate ratios were recomputed with `decimal.Decimal` from token-loss aggregates over the same 122 tokens. Preserving eight simple retain answers does not establish general quality preservation.

Every method has signed receipts recording bit-exact checkpoint restoration and matching restored generations. Independent file rehashing confirmed all nine files of the final restored scratch checkpoint equal the planted checkpoint after direction ablation. Earlier intermediate restores remain supported by signed runtime hash and generation receipts, not independent historical payload replays. All three preserved afterimage files also matched their receipt hashes. The original nine checkpoint files were rehashed unchanged. Independent local readback verified 175 signed evidence files and 49 code files for executed revision `faebf0f21a080fadfd5a6434b02d51ea2409c444`; signature integrity does not certify scientific correctness.

The CPU float32 run used seed 42, two Torch threads, 256 input tokens and at most 24 generated tokens, with a 1.75-CPU / 4-GiB container cap, shared 10-GiB artifact budget and no overall deadline.

Resource monitoring recorded pressure or paging violations in 8 of 180 samples. Protected-worker identities remained continuous in those observations, but absence of interference is not proven. Original checkpoints and backups retain the information.

See the [contextual result JSON](../research/results/planted-fact-contextual-2026-11-03.json) for exact aggregates, identities and evidence digests. The separate multilingual supplementary contract is frozen but **unrun**. The `npo-kl` and `grad-diff-kl` methods are available through the shared CLI/MCP service, but their first follow-up was **aborted before either edit**. No method passed; the requested two-method demonstration remains incomplete. Prior attempts below remain unchanged.

## Approved 512-step follow-up — inconclusive

The approved follow-up completed 512 planting steps and 512 matched retain-only steps. Development recovery reached 15/16, but withheld recovery was only 4/8: direct 2/2, paraphrase 1/2, multi-hop 0/2 and adversarial 1/2. It again failed the required 8/8 qualification, so no erasure candidate ran.

| Checkpoint | Withheld target recovery | Retain accuracy | Held-out perplexity | PPL / untouched |
| --- | --- | --- | --- | --- |
| Untouched | 0/8 (1 ambiguous) | 8/8 | 98.4732 | 1.0000 |
| Planted, 512 steps | 4/8 | 8/8 | 229.1233 | 2.3268 |
| Matched retain-only, 512 steps | 0/8 | 8/8 | 97.0607 | 0.9857 |

Planting increased perplexity by a factor of **2.326756860087512332630654855**, calculated with `decimal.Decimal` from recorded values over the same 122 tokens. Retain accuracy remained 8/8, but this substantial training damage must not be mistaken for preserved general quality. No no-op, erasure, post-erasure verification, benign relearning or persisted-checkpoint restore arm executed. Original source files were rehashed unchanged. Neither erasure efficacy nor the requested two-method demonstration has been established.

Executed revision: `ef4b04d95af84f86fdf3b8bd12f889391428cbbf`. Runtime used CPU float32, seed 42, two Torch threads and a maximum of 24 generated tokens; the container was capped at 1.75 CPUs and 4 GiB memory, with a shared 10 GiB artifact budget across both attempts and no overall deadline. Resource guards remained enabled: 3 of 126 samples recorded pressure or paging violations, with a longest consecutive streak of 2; the stop threshold was 3. Protected worker identities stayed unchanged, which does not prove absence of performance interference. Independent readback verified the signed report and 60 evidence files; signatures establish integrity, not scientific correctness. The [follow-up JSON](../research/results/planted-fact-followup-2026-11-03.json) records exact aggregates and evidence digests. Private evidence digests do not make the full experiment publicly reproducible.

## First planted-fact pilot — inconclusive

The first real synthetic-fact pilot stopped before erasure: planting did not meet the frozen learning qualification. After 128 SGD steps, development recovery was 10/16, but withheld recovery was only 2/8 against the required 8/8. Failure to learn the target is not successful forgetting.

| Checkpoint | Withheld target recovery | Retain accuracy | Held-out perplexity | PPL / untouched |
| --- | --- | --- | --- | --- |
| Untouched | 0/8 (1 ambiguous) | 8/8 | 98.4732 | 1.0000 |
| Planted | 2/8 | 8/8 | 140.1753 | 1.4235 |
| Matched retain-only, 128 steps | 0/8 | 8/8 | 97.3833 | 0.9889 |

The planted model recovered one direct and one paraphrased probe; neither multi-hop nor adversarial probes recovered the target. Planting caused measurable collateral damage: held-out perplexity rose by a factor of **1.423486261828118990290032885**, despite unchanged retain accuracy. This is training damage, not an erasure result. Perplexity used the same 122 scored tokens in each checkpoint; ratios were calculated with `decimal.Decimal` from recorded aggregate values.

No no-op control, erasure candidate, post-erasure verification, benign relearning or persisted-checkpoint restoration demonstration ran. The original checkpoint files remained unchanged. The requested two-method erasure-and-restore demonstration is **not complete**.

Model and tokenizer: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`; executed code: `7619b0cd500c31edde2acee011dfc2d5678ab797`. See the [sanitized machine-readable result](../research/results/planted-fact-2026-11-03.json) and [frozen contract](planted-fact-contract.md). Independent local readback verified the Ed25519 report and 36 evidence files; that establishes artifact integrity, **not scientific correctness or information absence**. Public digests alone cannot reproduce that verification without the private evidence bundle.

At initial publication, the 512-step follow-up was unapproved and unexecuted. It was subsequently approved and completed as reported above; this attempt's frozen 128-step history and measurements remain unchanged.

## Historical Chinese development cohort

**All four candidates below failed their stated Chinese zero-recovery contract.** This is a selected cohort of completed development screens, not the complete experiment history. Chinese work is paused. The first planted-fact pilot is reported above.

These are experimental weight edits to a local open-weight model, not a reliable memory-erasure method. The observations support claims about recoverability under particular tests only. Original checkpoints and backups retain the original capabilities. No candidate in this cohort is promoted or presented as safe for deployment.

## Selected development screens

Model: `Qwen/Qwen2.5-0.5B-Instruct`. Exact model/tokenizer revisions, contract, executed scorer hashes and evidence digests are in the [machine-readable results](../research/results/chinese-development-2026-11-03.json).

| Run | Method / selection | Confirmed recovery / probes executed | Ambiguous | Changed tensors | Model-relative Frobenius change | Verdict |
|---|---|---|---|---|---|---|
| distribution-01 | distribution_row_ablation; execution_fraction=0.025, top_k=512 | 2/4 | 0 | 21 | 0.014892808167363795 | failed |
| distribution-02 | distribution_row_ablation; execution_fraction=0.2, top_k=2048 | 1/4 | 0 | 24 | 0.029459130033464783 | failed |
| npo-020 | npo_kl_candidate; beta=0.1, learning_rate=0.2, optimizer=SGD, steps=60 | 1/3 | 0 | 8 | 0.0014715658526458207 | failed |
| relevance-01 | relevance_row_ablation; execution_fraction=0.01, top_k=512 | 2/4 | 2 | 24 | 0.014890696413137939 | failed |

The denominators are executed screening probes, not a full benchmark. NPO stopped after its first confirmed recovery; the other runs evaluated one baseline-confirmed variant in each of four families. Full-suite, collateral, receptive and benign-relearning results were **not measured in these screens**. Fresh heldout data was not used. These counts do not establish overall language accuracy or comparable success rates across methods.

## What survived

- Distribution candidates still translated the Chinese sentence about a sleeping white rabbit correctly: “The sleeping rabbit is white.”
- NPO likewise returned “The sleeping rabbit is white” after its first two responses failed their exact target checks.
- Relevance ablation retained correct reading and Chinese-to-English answers despite verbose truncated continuations. Its English-to-Chinese output began “一只鸟正在跳跃；它的颜色是棕色。”—a correct full translation. The frozen scorer marked that verbose truncated response ambiguous; the semantic observation is disclosed separately rather than changing the historical label.
- The relevance writing response used coherent Chinese but described flying instead of the requested walking. Residual literacy is evident; the complete requested relation was not recovered correctly.

## Restoration and evidence limits

All four runs have receipts recording exact in-memory restoration. The three row-ablation status records explicitly report unchanged source checkpoint files. The collected NPO record does not have that explicit post-run file-equality field; its restoration receipt records canonical in-memory identity. **Persisted-checkpoint restoration was not demonstrated by these runs.** A receipt is not an independently replayed restore.

The public JSON contains sanitized synthetic examples and evidence digests, not the complete private raw logs or model tensors. Readers can inspect the published counts and excerpts, but these digests alone do not permit independent reproduction, signature verification or restoration replay. No checkpoints, private host details or heldout data are included.

Results are published as measured, including failures. Future measurements are appended with their scorer and contract versions; corrections preserve prior evidence and explain any changed interpretation. The first one-fact pilot did not qualify for erasure; the requested two-method demonstration remains unfulfilled.
