# Experimental results

## Pretrained fact — qualification passed, localization refused

**Scientific outcome: inconclusive for erasure.** After changing only the qualification Moon expected answer to the observed correct sentence, the original model scored **8/8 target and 8/8 retain**. Final held-out probes and the scorer were unchanged. Untouched, source-materialization and no-op generations matched, with identical perplexity of 64.00652063548204055896541934 over 60 tokens.

The first attempted method stopped before an editing update: layer 23's single valid mapping pair produced a causal effect of **−2.366231918334961**, selecting **zero neurons**. This metric measures the first answer-token log probability at the final prefill position. The corrupted-context score was −10.03126049041748 and the patched score −12.397492408752441. A negative intervention result for one layer and pair does not establish that the fact has no representation elsewhere.

There was one failed pre-edit service record, **zero completed candidate edits and zero evaluated candidate methods**. Its stale summary label said running; the audited service failure and absent candidate snapshots/afterimages establish the terminal outcome. Final held-out candidate evaluation and benign relearning were not reached. No forgetting edit ran; the two materialization snapshots were restored and the journal was empty. Independent readback verified 29 signed evidence files, 73 code files, and exact expected inventories of all 9 original and 9 scratch checkpoint files.

Of 16 host samples, one recorded both an I/O-pressure ceiling violation and paging activity; positive paging deltas totaled 12,603 pages. Protected process identities remained stable, which does not prove absence of interference. There is no matched retain-only pretraining reference. The next step is read-only all-layer diagnosis, not a completed experiment. [Public aggregate and exact pins](../research/results/pretrained-fact-localization-2026-11-04.json). Earlier baseline measurements below remain unchanged.

## Pretrained fact baseline — qualification wording mismatch

**Scientific outcome: inconclusive for erasure; no editing method ran.** The original Qwen2.5-0.5B-Instruct recovered the Paris/France fact in all 8 qualification probes: 4/4 direct and 4/4 paraphrased. Retain scoring was 7/8. The rejected response correctly named the Moon in a full sentence, whereas the frozen expected answer was the single word `Moon`. This was a whole-answer formatting mismatch, not an established knowledge failure. The gate stopped execution before scratch materialization, no-op comparison, candidate editing, final held-out evaluation or benign relearning.

Untouched perplexity was 64.00652063548204055896541934 over 60 tokens. No planting or pre-erasure fine-tuning occurred; no matched retain-only pretraining reference was available. Independent verification checked 8 signed files and 73 code files; all 9 original and 9 scratch checkpoint files matched their expected inventories. No snapshot restoration is claimed because no edit transaction ran. The 5 recorded host samples contained no threshold violations or positive paging deltas; stable protected identities do not establish absence of interference.

The next calibration changes only the qualification Moon expected answer to the observed correct sentence. Final held-out probes, thresholds and the scorer remain unchanged; this entry does not claim that rerun occurred. [Public aggregate and exact code/model pins](../research/results/pretrained-fact-baseline-2026-11-04.json). Signatures establish artifact integrity, not information deletion.

## Five-fact utility-preservation pilot — planting failed

**Scientific outcome: inconclusive.** Connected10 exercised the utility-preservation objective for 512 planting updates and 32 qualification screens. The objective added `0.25 × mean utility-answer NLL` to utility KL, sharing the same utility forward pass. The data, model, tensor allowance, update ceiling and acceptance thresholds matched connected09. Planting still failed joint qualification: the final screen scored target **12/16**, retained facts **5/8** and capabilities **4/8**, versus **8/8** untouched capabilities. No unlearning method ran.

| Update | Target | Retain | Capabilities |
| ---: | ---: | ---: | ---: |
| 16 | 8/16 | 3/8 | 8/8 |
| 32 | 5/16 | 4/8 | 8/8 |
| 48 | 10/16 | 5/8 | 8/8 |
| 64 | 11/16 | 6/8 | 7/8 |
| 80 | 10/16 | 6/8 | 7/8 |
| 96 | 11/16 | 6/8 | 8/8 |
| 112 | 11/16 | 5/8 | 6/8 |
| 128 | 11/16 | 6/8 | 7/8 |
| 144 | 10/16 | 5/8 | 6/8 |
| 160 | 12/16 | 6/8 | 8/8 |
| 176 | 12/16 | 5/8 | 6/8 |
| 192 | 12/16 | 6/8 | 6/8 |
| 208 | 12/16 | 5/8 | 5/8 |
| 224 | 10/16 | 5/8 | 6/8 |
| 240 | 10/16 | 5/8 | 6/8 |
| 256 | 10/16 | 5/8 | 4/8 |
| 272 | 11/16 | 5/8 | 5/8 |
| 288 | 10/16 | 5/8 | 3/8 |
| 304 | 11/16 | 5/8 | 6/8 |
| 320 | 11/16 | 5/8 | 3/8 |
| 336 | 10/16 | 5/8 | 5/8 |
| 352 | 11/16 | 6/8 | 4/8 |
| 368 | 12/16 | 6/8 | 5/8 |
| 384 | 12/16 | 5/8 | 3/8 |
| 400 | 13/16 | 5/8 | 5/8 |
| 416 | 12/16 | 5/8 | 3/8 |
| 432 | 11/16 | 5/8 | 2/8 |
| 448 | 12/16 | 5/8 | 5/8 |
| 464 | 12/16 | 5/8 | 2/8 |
| 480 | 12/16 | 5/8 | 5/8 |
| 496 | 12/16 | 5/8 | 2/8 |
| 512 | 12/16 | 5/8 | 4/8 |

At the final screen, direct and inverse target recall were 4/4 each, paraphrased recall was 3/4 and multi-hop recall was 1/4. Four target answers and three retained-fact answers were ambiguous. The untouched perplexity was 30.91693274222041057653646864 over 337 tokens. Planted perplexity was not measured because generation qualification never passed; matched-reference training, no-op comparison, candidate editing, final factual evaluation and benign relearning were not reached.

| Final qualification screen | Connected09: utility KL | Connected10: utility KL + answer NLL |
| --- | ---: | ---: |
| Target recall | 13/16 | 12/16 |
| Retained facts | 6/8 | 5/8 |
| Capabilities | 3/8 | 4/8 |

This is a descriptive comparison of one run per objective, not evidence of improvement or causation. Independent verification checked the coefficient and finite scalar diagnostics for all 512 updates; it did not independently rerun the gradients. Matched-reference use of the objective remains unexercised in this pilot because qualification failed first.

Independent verification checked 45 signed files and 71 executed code files. All nine original and nine working-checkpoint file hashes matched their expected inventories; the snapshot was restored and the journal was empty. The original checkpoint remained read-only. Signatures establish integrity and provenance, not scientific certification or information absence.

The 80 host samples contained one paging-out violation totaling 6846 pages. Protected process identities remained stable, which does not prove absence of interference. Resource limits are pinned supervisor/launcher declarations; the independent aggregate does not certify a fresh measurement of cgroup limits.

See the [complete aggregate result and evidence digests](../research/results/connected-five-fact-utility-2026-11-04.json). Earlier results remain below; retained originals and backups still contain their original information.

## Five-fact pilot — planting failed

**Scientific outcome: inconclusive.** The smaller `connected-5` profile completed 512 planting updates and 32 qualification screens. Its one forget relation, two neighboring relations and two unrelated relations did not jointly qualify. The final screen scored target 13/16, retained facts 6/8 and capabilities 3/8 versus 8/8 capabilities before planting. No unlearning method ran.

| Update | Target | Retain | Capabilities |
| ---: | ---: | ---: | ---: |
| 16 | 6/16 | 2/8 | 8/8 |
| 32 | 5/16 | 3/8 | 8/8 |
| 48 | 10/16 | 6/8 | 8/8 |
| 64 | 11/16 | 6/8 | 8/8 |
| 80 | 8/16 | 7/8 | 8/8 |
| 96 | 9/16 | 7/8 | 6/8 |
| 112 | 11/16 | 7/8 | 5/8 |
| 128 | 10/16 | 6/8 | 7/8 |
| 144 | 11/16 | 7/8 | 6/8 |
| 160 | 10/16 | 5/8 | 6/8 |
| 176 | 11/16 | 6/8 | 7/8 |
| 192 | 11/16 | 6/8 | 5/8 |
| 208 | 12/16 | 6/8 | 6/8 |
| 224 | 14/16 | 6/8 | 7/8 |
| 240 | 13/16 | 6/8 | 6/8 |
| 256 | 12/16 | 6/8 | 6/8 |
| 272 | 14/16 | 6/8 | 6/8 |
| 288 | 12/16 | 6/8 | 6/8 |
| 304 | 11/16 | 6/8 | 7/8 |
| 320 | 11/16 | 6/8 | 7/8 |
| 336 | 11/16 | 5/8 | 6/8 |
| 352 | 11/16 | 6/8 | 5/8 |
| 368 | 13/16 | 5/8 | 6/8 |
| 384 | 13/16 | 5/8 | 6/8 |
| 400 | 12/16 | 6/8 | 5/8 |
| 416 | 13/16 | 5/8 | 6/8 |
| 432 | 11/16 | 7/8 | 4/8 |
| 448 | 13/16 | 6/8 | 6/8 |
| 464 | 12/16 | 6/8 | 6/8 |
| 480 | 13/16 | 6/8 | 6/8 |
| 496 | 13/16 | 6/8 | 4/8 |
| 512 | 13/16 | 6/8 | 3/8 |

At the final screen, direct, paraphrased and inverse recall each reached 4/4, but multi-hop recall was 1/4 with three unresolved answers. Neither joint qualification nor collateral preservation passed. The untouched corpus perplexity remained 30.91693274222041057653646864 over 337 tokens. Planted perplexity was not measured because generation qualification failed. Matched retain-only training, no-op comparison, candidate edits, final factual evaluation and benign relearning were not reached.

Independent verification checked 45 signed files and 71 executed code files, recomputed all screen aggregates, and verified the empty transaction journal and restored snapshot. Receiver hashes matched all nine original and nine working-checkpoint files. The original checkpoint remained read-only.

The 79 host samples contained 5 samples with paging-out violations totaling 48837 pages; none reached the sustained-abort threshold. Protected process identities remained stable; this does not prove absence of interference. An earlier preparation attempt was resource-aborted and is separate from this completed planting result.

At the time of this run, the utility-answer NLL correction was prospective. It was subsequently exercised in connected10, reported above, without qualifying planting. The connected09 measurements and frozen objective remain unchanged.

See the [complete aggregate result and digests](../research/results/connected-five-fact-2026-11-04.json). Signatures establish byte integrity, not information deletion.

## Connected-fact retry with prequalified capabilities — planting failed

**Planting failed qualification after 512 updates; scientific outcome: inconclusive. No unlearning method was tested.** The final screen recovered 11/16 target answers and 8/8 retained-fact answers, but only 2/8 capability answers. The untouched model had passed all eight capability checks: this is collateral loss during planting, before erasure.

| Update | Direct | Paraphrase | Inverse | Multi-hop | Target | Retain | Capabilities |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2/4 | 3/4 | 0/4 | 4/4 | 9/16 | 5/8 | 6/8 |
| 128 | 2/4 | 3/4 | 0/4 | 3/4 | 8/16 | 6/8 | 6/8 |
| 192 | 2/4 | 3/4 | 2/4 | 0/4 | 7/16 | 6/8 | 5/8 |
| 256 | 4/4 | 3/4 | 2/4 | 3/4 | 12/16 | 6/8 | 6/8 |
| 320 | 4/4 | 2/4 | 3/4 | 3/4 | 12/16 | 7/8 | 3/8 |
| 384 | 3/4 | 3/4 | 1/4 | 3/4 | 10/16 | 8/8 | 4/8 |
| 448 | 3/4 | 3/4 | 2/4 | 4/4 | 12/16 | 6/8 | 4/8 |
| 512 | 4/4 | 3/4 | 2/4 | 2/4 | 11/16 | 8/8 | 2/8 |

All screens were recomputed from signed raw evidence. Matched retain-only training, no-op comparison, candidate edits, final factual evaluation and benign relearning were not reached. Planted perplexity was not measured because generation qualification failed; untouched perplexity remained 30.91693274222041057653646864 over the same 337-token corpus.

Independent readback verified 21 signed files and 71 code files. All snapshots were restored, the transaction journal was empty, and receiver hashes matched all nine original and nine scratch checkpoint files. The 75 host samples recorded zero guard violations and zero positive paging-out pages; this does not prove absence of interference. Signatures attest artifact integrity, not deletion or scientific correctness.

See the [retry result JSON](../research/results/connected-fact-qualified-capabilities-2026-11-04.json). Capability preservation is conditioned on prior untouched-model performance, not an unbiased measure of broad capability accuracy.

## Capability bank preparation — failed first bank, qualifying revision

The first fixed bank produced 25/32 correct responses and 11 pairs with both phrasings correct. It **failed** because only two science pairs qualified against the unchanged quota of three. No partial selection was accepted. The other category quotas were met.

A prospective revision replaced only the three ineligible science pairs, preserving the other thirteen pairs, priority order, quotas and strict whole-answer scorer. It produced 28/32 correct responses and 13 fully eligible pairs; the fixed policy selected eight pairs across arithmetic, lexical, basic-science and general-knowledge categories. Both phrasings passed for every selected pair. This **passed preparation**, not unlearning, and does not reinterpret the first bank's failure.

These unsigned screens were independently checked against raw responses and transport inventories. Original and scratch files were unchanged. They evaluated no factual final probes and performed no planting or weight edits. See [failed bank JSON](../research/results/connected-fact-bank-04-2026-11-04.json) and [qualifying bank JSON](../research/results/connected-fact-bank-06-2026-11-04.json).

## Connected-fact capability preflight — failed before planting

The new untouched-model gate completed 16 capability generations and refused to start the planting/unlearning pilot. Qualification capability correctness was **6/8**; the independently phrased final capability subset was **7/8**. No factual final probes were evaluated. Both partitions had zero missing or ambiguous responses.

Explicit one-word or digit-integer instructions were used with the unchanged strict whole-answer scorer. Post-hoc review of the three rejected responses (**DERIVED**) found genuinely incorrect complete short answers, not formatting-only or accepted-equivalence gaps. No score was changed. The unlearning outcome remains **inconclusive**.

Independent readback verified the raw capability scores, frozen code/plan pins, 71 source files, and exact hashes for nine original and nine reused-scratch checkpoint files. This preflight evidence is transport-hash verified but **not Ed25519 signed**; no scientific aggregate exists. Five host samples recorded no guard violations or positive paging-out delta. This does not prove absence of interference.

That preflight failure led to a predetermined bank of 16 capability pairs. Its subsequent failed and qualifying screens are reported above. Selection requires both phrasings to pass exact scoring under fixed category quotas and priority; this is competence-conditioned preservation testing, not unbiased broad capability accuracy. The fact graph and factual evaluation wording remained unchanged.

See the [capability-preflight result JSON](../research/results/connected-fact-preflight-2026-11-04.json) for exact counts, evidence limits and digests.

## Connected-fact pilot — planting qualification inconclusive

The connected benchmark did not reach unlearning. At the 512-update limit, the model recovered 10/16 target probes and answered 7/8 retained-fact probes correctly. Capability correctness was 0/8 under the frozen whole-answer scorer. No candidate edits, matched retain-only training, benign relearning, or final factual evaluation ran.

| Planting update | Direct | Paraphrase | Inverse | Multi-hop | Target total | Retain | Capabilities |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2/4 | 2/4 | 0/4 | 2/4 | 6/16 | 5/8 | 0/8 |
| 128 | 3/4 | 3/4 | 0/4 | 2/4 | 8/16 | 7/8 | 0/8 |
| 192 | 4/4 | 3/4 | 0/4 | 4/4 | 11/16 | 7/8 | 0/8 |
| 256 | 4/4 | 3/4 | 0/4 | 3/4 | 10/16 | 7/8 | 0/8 |
| 320 | 4/4 | 3/4 | 0/4 | 4/4 | 11/16 | 7/8 | 0/8 |
| 384 | 4/4 | 3/4 | 0/4 | 4/4 | 11/16 | 7/8 | 0/8 |
| 448 | 4/4 | 2/4 | 0/4 | 2/4 | 8/16 | 7/8 | 0/8 |
| 512 | 4/4 | 2/4 | 0/4 | 4/4 | 10/16 | 7/8 | 0/8 |

All eight screens were recomputed from signed raw records. The final target screen also had six unresolved outputs; the capability screen had one truncated ambiguous output. A zero in the table is a measured correctness count, not evidence of absent information. Inverse retrieval remained 0/4 throughout.

**Post-hoc semantic review (DERIVED):** at update 192, seven capability outputs were semantically correct but violated the exact answer format; one was a truncated non-answer. At update 512, six were semantically correct but nonconforming, one was a malformed multi-number response, and one was a truncated non-answer. All four final inverse outputs repeated the relation object instead of identifying the subject. These observations do not replace the frozen scores or establish preserved general capabilities.

Untouched perplexity was 30.91693274222041057653646864 over 337 scored tokens. Planted perplexity and its ratio were **not measured**: the generation screen never qualified for that evaluation. This corpus differs from the older single-fact corpus, so their raw perplexities should not be compared as one series.

Independent readback verified 21 signed files, 71 executed source files, one restored outer snapshot, an empty transaction journal, and exact hashes for all nine original and nine reused-scratch checkpoint files. One of 80 host samples exceeded the load threshold; observed paging-out deltas totaled zero pages. Stable workload identities do not establish absence of performance interference.

A prospective correction adds explicit short-answer formatting, requires untouched capability qualification before planting, and balances forward/inverse training within the existing budget. It was not executed in revision `53ffa16b5e5018166ee545f9335a10c058969839`; historical thresholds and outcomes remain unchanged.

See the [connected-fact result JSON](../research/results/connected-fact-2026-11-04.json) for the learning curve, ambiguity counts, restoration scope and evidence digests. Signatures establish provenance, not scientific certification.

## Connected-fact first attempt — resource-guard abort

The earlier attempt was stopped after three consecutive physical-host load violations. Its five numerical/guard self-tests passed, but no terminal scientific summary or signed aggregate was available. Candidate execution remains unknown from those absent records; no efficacy result is claimed. The stopped worker exited with code 137 and was not marked OOM-killed. All nine original and nine scratch checkpoint files matched their pre-run hashes.

The four recorded host samples included three load violations and zero observed paging-out delta. This is **inconclusive, unexercised scientific evidence**, separate from the retry’s planting failure. See the [guard-abort JSON](../research/results/connected-fact-abort-2026-11-04.json).

## Two-method repair follow-up — passed the stated experimental contract

Both saved candidates passed the frozen reused-probe contract, independent reload and a separate benign-relearning test. This is reduced recoverability under these tests; the fresh supplement was subsequently inconclusive, as detailed below.

| Method | Arm | Target recovered | Retain correct | PPL / planted |
|---|---|---:|---:|---:|
| npo-kl-repair | direct | 0/8 | 8/8 | 0.8377663406272934043903308152 |
| npo-kl-repair | independent | 0/8 | 8/8 | 0.8377663406272934043903308152 |
| npo-kl-repair | benign-relearning | 0/8 | 8/8 | 0.8252863336906772021468436989 |
| grad-diff-kl-repair2 | direct | 0/8 | 8/8 | 0.7965371463646040520589890906 |
| grad-diff-kl-repair2 | independent | 0/8 | 8/8 | 0.7965371463646040520589890906 |
| grad-diff-kl-repair2 | benign-relearning | 0/8 | 8/8 | 0.8001409507238711492182649532 |

Adaptive selection used earlier evaluation feedback; reused probes are not an untouched confirmation set.
Planting raised held-out perplexity from 98.47 to 142.75; collateral ratios compare to the damaged planted baseline.
NPO outputs are largely malformed code-like fragments; GradDiff repair2 returns a single digit. Full fact recovery was not observed, but output degradation does not establish information absence.
Original checkpoints and backups retain the information.
Signed receipts establish provenance, not scientific certification. Remote file hashes and signed runtime restoration are distinct evidence; tensor payloads were not recomputed locally.
The fresh multilingual supplement is reported separately below.
Host worker identity continuity does not prove absence of performance interference.

Both methods restored checkpoint bytes and baseline generations exactly. Independent readback verified 102 signed evidence files, 55 source files and 12 loader receipts. Separate remote hashing verified unchanged original/planted files, all nine restored scratch files and both candidate tensor payloads. Three of 63 host samples recorded pressure or paging violations, with 2,508 paging-out pages in total. These observations do not prove absence of performance interference.

See the [two-method result JSON](../research/results/planted-fact-two-method-2026-11-04.json) for exact measurements, post-hoc output-audit counts and evidence hashes. All 4,864 measured final-layer features remained active after each edit. Partial numeric prefixes survived despite zero full-fact recoveries.

## Fresh multilingual supplement — inconclusive

The frozen supplement failed baseline qualification: target recall was 9/16
(English 5/8, Spanish 3/4, Chinese 1/4), and retain accuracy was 1/8.

| Checkpoint | Confirmed full-fact recovery | Ambiguous target responses | Retain correct |
| --- | ---: | ---: | ---: |
| Planted baseline | 9/16 | 0 | 1/8 |
| npo-kl-repair | 0/16 | 1 | 2/8 |
| grad-diff-kl-repair2 | 0/16 | 0 | 2/8 |

Neither candidate lost the baseline's single correct retain answer. Those narrow
paired results do not establish broad retention. Missing baseline knowledge and
the ambiguous NPO response do not count as successful erasure. The original
reused-probe contract passed; this separate supplement remains **inconclusive**.

Post-hoc review covered all 72 outputs. None of the 32 edited-target responses
contained a complete recovery, including spelled-out and Chinese-numeral forms. Edited target outputs were incomplete
numeric prefixes, except for one truncated Spanish introduction without an
answer. That response remains ambiguous under the frozen scorer. These results
show answer degradation, not information absence or a proven erasure mechanism.

Independent readback verified 85 signed evidence files and 56 source files.
Signed canonical-state restoration receipts and unchanged source-file hashes
support this in-memory evaluation; it did not write a new checkpoint or replay
persisted-checkpoint restoration. Two of 16 host samples recorded resource
violations, with 10,576 paging-out pages. Worker identity continuity does not
prove absence of performance interference.

See the [supplement result JSON](../research/results/planted-fact-supplement-2026-11-04.json)
for the frozen contract digest, per-language counts, output-audit categories and
provenance. The same saved candidates were evaluated without additional training.

## Explicit repair — one method passed, two-method contract failed

The saved `npo-kl-repair` candidate passed direct evaluation, an independent reload, and a separate benign-relearning test. Target recovery was 0/8 and retain accuracy 8/8 in each arm, without ambiguous observations or paired retain losses. Its source-precision weights restored bit-for-bit and restored generations matched the planted baseline. This is reduced recoverability under these tests.

| Method | Base / repair updates | Direct and independent target / retain | After benign relearning target / retain | Direct PPL / planted | Overall method result |
| --- | --- | --- | --- | --- | --- |
| npo-kl-repair | 8 / 8 | 0/8 / 8/8 | 0/8 / 8/8 | 0.8377663406272934043903308152 | passed the stated experimental contract |
| grad-diff-kl-repair | 4 / 8 | 0/8 / 6/8 | 0/8 / 8/8 | 0.7500580232620554090938483920 | failed |

GradDiff's saved candidate failed the zero-collateral-loss requirement even though another eight retain-only updates repaired both answers in memory. The two-method cohort therefore failed. Persisting that additional repair is a new candidate, not a reinterpretation of this result.

Independent readback verified 102 signed evidence files and 55 source files at executed revision `7ebda14aa3bc7fe9bc6a8545cfba9de6066eb1c0`. Both methods have exact weight and generation restoration receipts; separate remote hashing verified all nine final scratch files and unchanged original/planted checkpoints. Preserved sparse candidate files matched their signed hashes. One of 61 host samples recorded resource violations, with 1,165 physical paging-out pages in total. Protected-worker identities remained continuous; absence of performance interference is not proven.

See the [repair result JSON](../research/results/planted-fact-repair-2026-11-04.json) for exact scores and evidence hashes. Aggregate feedback from the reused evaluation set guided this adaptation; these are exploratory results. At this stage the separately frozen multilingual evaluation had not run. The retain-only reference matches original planting, not the unlearning update count. Original checkpoints and backups retain the information, and planting damage remains distinct from edit collateral.

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

The [eight-step result JSON](../research/results/planted-fact-eight-step-2026-11-04.json) contains exact scores and evidence hashes. Aggregate feedback from this reused evaluation set now informs subsequent candidates; later results on it are adaptive development evidence. At this stage the separately frozen multilingual supplement had not run. New explicit repair variants must pass saved-checkpoint evaluation and another benign-relearning test. Original checkpoints and backups retain the information.

## Gradient follow-up — failed at the weight-change bound

The qualified planted checkpoint was reused without further planting. Both gradient methods attempted 128 updates under the fixed 0.25 relative-change bound:

| Method | Completed updates | Rejected update | Edited checkpoint / evaluation |
| --- | --- | --- | --- |
| npo-kl | 18 | 19 | Not produced / not executed |
| grad-diff-kl | 10 | 11 | Not produced / not executed |

Both raised `ValueError: factoid update exceeds relative bound`. The overall contract **failed**; each method's erasure efficacy remains **inconclusive** because no edited checkpoint reached behavioral evaluation. No post-edit independent evaluation or benign-relearning arm ran. Completed counts were derived from collected progress events; the rejected update is the next sequential iteration, not an independently recorded final delta norm.

Fresh baseline and no-op checks both recovered 8/8 target probes with 8/8 retain accuracy and perplexity 142.7528471758874694472894703 over 122 tokens. No-op generations matched exactly. Each transaction restored the original planted state and reproduced its generations; restored perplexity divided by baseline perplexity is exactly 1 using `decimal.Decimal`. Separate final file rehashing confirmed all nine scratch files equal the planted checkpoint, with original source files unchanged. Earlier intermediate restoration is supported by signed runtime receipts and generation comparisons.

Independent readback verified Ed25519 integrity for 42 signed files and 53 code files at revision `a3796abfa6b5abef64def0f52fab59db6f08997e`. It does not certify scientific correctness. The reused retain-only reference was matched to the original 192-step planting, not the new unlearning updates. The two preceding operational attempts remain in the evidence history: one was stopped by physical-paging guards before candidate evaluation; the other failed a mandatory loader-test fixture before target loading. Neither produced a signed scientific aggregate.

See the [gradient bound result JSON](../research/results/planted-fact-gradient-bounds-2026-11-03.json) for exact aggregates and receipt digests. This adds no successful erasure result. At this stage the separate multilingual supplement had not run; retained originals and backups still contain the fact.

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

See the [contextual result JSON](../research/results/planted-fact-contextual-2026-11-03.json) for exact aggregates, identities and evidence digests. At this stage the separate multilingual supplementary contract was frozen but **unrun**. The `npo-kl` and `grad-diff-kl` methods are available through the shared CLI/MCP service, but their first follow-up was **aborted before either edit**. No method passed in this contextual cohort. Subsequent gradient and repair cohorts are reported above. Prior attempts below remain unchanged.

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

No no-op control, erasure candidate, post-erasure verification, benign relearning or persisted-checkpoint restoration demonstration ran. The original checkpoint files remained unchanged. This initial attempt did not demonstrate two-method erasure and restoration.

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

Results are published as measured, including failures. Future measurements are appended with their scorer and contract versions; corrections preserve prior evidence and explain any changed interpretation. The first one-fact pilot did not qualify for erasure; subsequent cohorts and their outcomes are reported above.
