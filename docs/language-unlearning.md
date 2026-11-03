# Chinese-literacy experiments

The current language experiment targets Chinese reading, writing, and translation
in a local Qwen2.5-0.5B-Instruct checkpoint. It is separate from the planted-fact
acceptance experiment. Failure of one run is evidence for the next experiment,
not a claim that the language has been removed.

`erase.language` measures post-gate MLP activity across every layer using matched
Chinese/English development text and non-Chinese background prompts. Numerical
code ranks positively contrasted features and emits a source-bound row selection.
The editor explicitly chooses a candidate from the declared fraction ladder. The
operator zeros the corresponding up-projection rows after writing durable tensor
preimages, checks finite values and per-tensor/model change bounds, then verifies
that other parameters are unchanged. An exact in-memory restoration follows each
development screen. Persisted candidates use the existing checkpoint writer.

This magnitude-ratio heuristic is **not LAPE**. The released
[LAPE implementation](https://github.com/RUCAIBox/Language-Specific-Neurons)
measures positive gate-activation probabilities separately by language and selects
globally by entropy. Its
[ACL paper](https://aclanthology.org/2024.acl-long.309/) demonstrates language
impairment and steering, not information absence. Shared language features make
collateral damage a measured risk, particularly for Chinese and Japanese.

`verify.language` preserves direct, paraphrased, traditional-character, pinyin,
indirect, and supplementary recovery observations. It accepts ordinary Chinese
short forms that the historical scorer missed. Missing, truncated, or unresolved
outputs cannot become proof of forgetting. Conservative target recovery and
stricter retained-answer correctness have separate rules. Comparisons recompute
scores from raw evidence, require baseline-demonstrated ability, and expose paired
retain losses. Non-Chinese token-weighted perplexity is checked separately by the
iteration script. Fresh held-out commitments remain outside the editor's
development data; historical examples are explicitly unblinded.

`scripts/chinese_iteration.py` provides diagnosis, development baseline, candidate
screening, checkpoint persistence, and independent evaluation phases. It defaults
to a plan unless `--apply` is present. Runtime settings and source/data/code hashes
are saved for every phase. An explicitly authorized open-ended run uses
`RuntimeConfig(max_seconds=None)` and no overall watchdog deadline. CPU, memory,
disk, workload identity, and pressure protections still apply. Existing frozen
historical batches retain their original deadlines.

Passing a finite contract means **reduced recoverability under those tests**.
It does not prove that all Chinese information is absent. Originals, preimages,
and backups retain the original information. No checkpoint is promoted by this
experiment.
