# Experimental results

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

A follow-up proposal allowing up to 512 planting steps is **not approved and has not executed**. It does not change this attempt's frozen 128-step history.

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
