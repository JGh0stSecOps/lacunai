# Research basis and limits

LacunAI treats model-directed weight editing as a hypothesis to test. It does not present behavioral suppression as reliable information deletion.

- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) demonstrates behavioral editing, including a reasoning-loop intervention. It motivates an editor inspecting diagnostic evidence and proposing weight-space interventions; it does not establish dependable unlearning.
- [Released experiment artifacts, pinned revision](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f) include transcripts, edit manifests, reconstruction code and evaluations. The complete original agent harness is not supplied. LacunAI does not claim to reproduce that harness.
- [OpenUnlearning](https://arxiv.org/abs/2506.12618v2) and its [pinned implementation](https://github.com/locuslab/open-unlearning/tree/2d98027677f3468d52b6baa3fc4f2b9f2dc08820) provide established comparison methods. The pilot uses the GradDiff NLL objective: minimize retain loss minus forget loss, with declared equal coefficients. This is a comparison baseline, not a deletion certificate.
- [Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning](https://arxiv.org/html/2406.13356v4) motivates testing recovery after bounded, related benign training and checking that the relearning data does not itself contain the withheld targets.
- [Eight Methods to Evaluate Robust Unlearning in LLMs](https://arxiv.org/html/2402.16835v1) and [Prompt Attacks Reveal Superficial Knowledge Removal in Unlearning Methods](https://arxiv.org/html/2506.10236v2) motivate varied recovery attempts and collateral-capability measurements.
- [Distribution-aware Language Neuron Identification](https://arxiv.org/html/2609.10993v1) motivates observing negative as well as positive activations. The experimental histogram mapper uses language-distribution overlap; it does not reproduce the paper's clustering or establish Chinese removal in the pilot model.

The [fast-weight programmer research note](fast-weight-programmers.md) examines the original controller-and-memory architecture, its delta update rule, and a bounded future hypothesis. It distinguishes learned online memory writes from the current externally directed checkpoint edits.

## Pilot evaluation boundaries

Chinese reading, writing and both translation directions are separate skills. Reading permits English or numeric answers so that suppressing Chinese-character output cannot masquerade as lost comprehension. Alternate prompts, mixed-language requests, receptive likelihoods, neighboring facts and non-Chinese tasks provide additional observations.

The frozen scorer leaves outputs outside its accepted expressions ambiguous. Those cases cannot be counted as established forgetting. Raw output examples must accompany apparent score changes; semantically plausible wording changes can fall outside a narrow accepted-expression set.

A true reference pretrained without Chinese is unavailable. Retain-only continuation is a control, not a counterfactual model that never learned Chinese. The initial broad qualification contract was not satisfied; that historical outcome remains unchanged. The accepted weak-to-none scope measures demonstrated baseline skills, requiring at least one confirmed baseline item in every primary family. It does not establish absence of Chinese knowledge.

Historical evaluation outputs were visible to the current-session editor. A separate small audit was generated after the edit specification was committed; it uses unseen combinations of the same templates and lexicon. It is descriptive; historical results remain governed by their original acceptance contracts.

No finite collection of failed extraction attempts establishes information absence. Retained original checkpoints and backups still contain the original capabilities.

### Evaluator migration: demonstrated baseline skills

Scorer `5-qualified-baseline-receptive-gap` and contract `chinese-literacy-zero-recovery-v3` report primary `baseline_unknown_ids` separately from after-edit uncertainty. A baseline-unknown item is not a demonstrated skill and does not invalidate measured loss of other qualified items. Every after-edit ambiguity still blocks acceptance, and any confirmed after-edit recovery fails, including recovery outside the qualified set. Retain ambiguity and paired-loss gates are unchanged.

The receptive gate requires an absolute decrease of 0.25 (25 percentage points), restoring the historical receptive criterion, with maximum accuracy still 0.35. Thus a measured baseline of 0.50 and after score of 0.25 meets this gate; this is not a relative reduction criterion. The preceding v2 contract used a 0.50 absolute decrease. A baseline unable to demonstrate the required gap remains inconclusive.

Existing artifacts, signatures, scorer labels, and historical verdicts are preserved. Applying this correction requires new versioned evaluations or rescoring artifacts with explicit provenance; old results must not be relabeled. A pass means reduced recoverability under the stated finite tests of demonstrated skills, never proof that the entire Chinese language is absent. These changes do not resolve independent reproducibility or control failures.
