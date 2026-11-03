# Research basis and limits

LacunAI treats model-directed weight editing as a hypothesis to test. It does not present behavioral suppression as reliable information deletion.

- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) demonstrates behavioral editing, including a reasoning-loop intervention. It motivates an editor inspecting diagnostic evidence and proposing weight-space interventions; it does not establish dependable unlearning.
- [Released experiment artifacts, pinned revision](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f) include transcripts, edit manifests, reconstruction code and evaluations. The complete original agent harness is not supplied. LacunAI does not claim to reproduce that harness.
- [OpenUnlearning](https://arxiv.org/abs/2506.12618v2) and its [pinned implementation](https://github.com/locuslab/open-unlearning/tree/2d98027677f3468d52b6baa3fc4f2b9f2dc08820) provide established comparison methods. The pilot uses the GradDiff NLL objective: minimize retain loss minus forget loss, with declared equal coefficients. This is a comparison baseline, not a deletion certificate.
- [Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning](https://arxiv.org/html/2406.13356v4) motivates testing recovery after bounded, related benign training and checking that the relearning data does not itself contain the withheld targets.
- [Eight Methods to Evaluate Robust Unlearning in LLMs](https://arxiv.org/html/2402.16835v1) and [Prompt Attacks Reveal Superficial Knowledge Removal in Unlearning Methods](https://arxiv.org/html/2506.10236v2) motivate varied recovery attempts and collateral-capability measurements.

The [fast-weight programmer research note](fast-weight-programmers.md) examines the original controller-and-memory architecture, its delta update rule, and a bounded future hypothesis. It distinguishes learned online memory writes from the current externally directed checkpoint edits.

## Pilot evaluation boundaries

Chinese reading, writing and both translation directions are separate skills. Reading permits English or numeric answers so that suppressing Chinese-character output cannot masquerade as lost comprehension. Alternate prompts, mixed-language requests, receptive likelihoods, neighboring facts and non-Chinese tasks provide additional observations.

The frozen scorer leaves outputs outside its accepted expressions ambiguous. Those cases cannot be counted as established forgetting. Raw output examples must accompany apparent score changes; semantically plausible wording changes can fall outside a narrow accepted-expression set.

A true reference pretrained without Chinese is unavailable. Retain-only continuation is a control, not a counterfactual model that never learned Chinese. The initial broad qualification contract was not satisfied, so the pilot remains ineligible for an overall pass even while demonstrated skills can be measured.

Historical evaluation outputs were visible to the current-session editor. A separate small audit was generated after the edit specification was committed; it uses unseen combinations of the same templates and lexicon. It is descriptive and does not replace the original acceptance contract.

No finite collection of failed extraction attempts establishes information absence. Retained original checkpoints and backups still contain the original capabilities.
