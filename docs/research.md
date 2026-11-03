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

### Proposed mask-gradient diagnostic

The diagnostic places unit masks on MLP features entering `down_proj` and differentiates mean answer-token NLL, including EOS, with respect to those masks. For a small mask decrease, the first-order loss change is the negative mask gradient times the decrease. The proposed ranking uses the positive part of the negative, equally weighted forget-family mean gradient, subtracts the maximum retain-family mean absolute gradient, then clips at zero. Absolute values must precede the retain-family average to avoid cancellation. This estimates local sensitivity; unit-to-zero ablation is a finite intervention whose effects and interactions require separate measurement. English answers to Chinese inputs reduce the confounding effect of Chinese output suppression but do not establish Chinese comprehension loss by themselves.

[LAHIS, version 2](https://arxiv.org/html/2511.07498v2) uses multiplicative attention-head masks and first-order loss sensitivity. Its negative-gradient-conditioned importance score and attention-head granularity differ from this proposed signed-family MLP ranking; this is an adaptation, not a reproduction. [CRANE, version 1](https://arxiv.org/html/2601.04664v1) instead uses conservation-based layer-wise relevance propagation, language-conditioned relevance distributions and intervention validation. Ordinary autograd mask derivatives do not implement its LRP procedure or establish functional necessity.

[Multilingual Unlearning in LLMs, version 1](https://arxiv.org/html/2606.03291v1) reports late-layer representational changes and steering-based recovery in its studied settings. The authors explicitly limit causal interpretation of hidden-state similarity. These results motivate recovery tests and caution about output suppression; they do not establish that this pilot shares the same mechanism or has erased information.

A frozen-weight gradient diagnostic still retains activations needed for backpropagation. Absence of parameter gradients is not proof of compliance with the memory ceiling. Implementation checks must cover graph disposal per example, hook and `requires_grad` restoration on failure, unchanged source hashes, and numerical finite-difference agreement of mask gradients. No optimizer step is part of this diagnostic.

## Single-fact localization and recovery limits

[Mechanistic Unlearning (Guo et al., ICML 2025)](https://proceedings.mlr.press/v267/guo25k.html) distinguishes components that enrich a subject representation with factual information from components that extract that information into an answer. Its [methods and evaluation](https://arxiv.org/html/2410.12949v2) compare fact-lookup localization with output-tracing approaches, including causal tracing and attribution patching. Localization does not universally improve robustness: disrupting an answer-extraction pathway can suppress familiar outputs while leaving alternative recovery routes. The reported benefits concern particular mechanisms, models, datasets and attacks, not a general deletion guarantee.

The current single-fact pilot plants information through a restricted parameter set and tests edits with frozen recall and collateral checks. Knowing which parameters were trained is useful provenance; it does not independently demonstrate that they form a factual lookup circuit. A future localization study could distinguish mediated effects on intermediate subject representations from direct effects on answer logits, then test generalization across prompt formats and compare parameter-matched localization controls. This is a proposed diagnostic, not a claim about the current implementation or an alteration of the active experiment.

[Do Unlearning Methods Remove Information from Language Model Weights? (Deeb and Roger)](https://arxiv.org/html/2410.08827v3) evaluates recovery by retraining on one subset of forgotten facts and testing another subset whose answers cannot be inferred from the retraining subset. Paraphrases of one fact are not independent facts: teaching its answer through one wording leaks it to the others. The pilot's bounded retain-only benign relearning is therefore a limited recovery attempt, not a reproduction of RTT. A future RTT-style study needs independent facts and a reference demonstrating that its retraining subset does not teach the withheld answers. The [released evaluator](https://github.com/aghyad-deeb/unlearning_evaluation/blob/main/finetune_corpus.py) separates training, forget-validation and retain-validation inputs.

Deeb and Roger also caution that fine-tuning-planted information can yield more optimistic unlearning results than pretrained information. Passing this planted-fact contract would establish reduced recoverability under its tests, not pretrained-knowledge erasure. Failed recovery attempts, including RTT, do not prove information absence. Original checkpoints and backups retain the information; exact restoration demonstrates reversibility, not deletion.
