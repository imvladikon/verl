# GLM-5.2 full-model quality execution gate

The result is **PENDING** until a full-model adapter improves the held-out
Russian, Markdown, and accidental-Han targets without a semantic regression.
The 9B surgery results prove the training path, not language quality.

## Current gate

| gate | status | evidence or missing artifact |
|---|---|---|
| Immutable BF16 base | PASS | `zai-org/GLM-5.2@cf457fa734ab149ffef225f80893eb38c6ff5cdc` and locked `config.json` hash |
| Teacher-free quality mixture | PASS | 2,728 rows, three no-truncation buckets, mixture SHA-256 `094a0385dcc27d647b92d2d4d40ad4ec7ae1bbeab8de878915efaed88bc824e7` |
| Full-width surgery LoRA | PASS | finite backward, adapter export/reload, and MLA+`lm_head` ablation on the 9B fixture |
| Tensor/expert sharding gates | PASS | TP2 evidence `80ce91da…958`; EP2 evidence `a6a739c9…e13` |
| Full TP8/EP32 config | PASS, runtime pending | 64-H200 planning profile; this is not a memory or training pass |
| Full base held-out predictions | PENDING | raw JSONL with prompt and decoding-contract hashes |
| Full MLA-only adapter | PENDING | H200 training checkpoint and run evidence |
| Full MLA+`lm_head` adapter | PENDING | run only if the MLA-only result does not close the quality gate |
| Paired held-out comparison | PENDING | `compare_quality_outputs.py` must report PASS for all three targets |

## Sequential experiment order

1. Generate and preserve the full-model base outputs once.
2. Train only the MLA-only adapter on the locked 2,728-row mixture with seed 52.
3. Generate paired outputs and run the quality comparator.
4. Train MLA+`lm_head` only if MLA-only is insufficient. Use the identical
   rows, seed, sequence buckets, 33 optimizer updates, rank, alpha, and
   decoding contract.
5. Select an adapter only on validation. Report the untouched test split once.

The launchers intentionally use different acknowledgement strings and run
directories, so one profile cannot overwrite or masquerade as the other.

## Config-only commands

MLA-only:

```bash
CONFIG_ONLY=1 MODEL_PATH=/path/to/config-only-snapshot \
MIXTURE_DIR=/path/to/quality-mixture-targeted-wikipedia-2728 \
examples/glm52_lora/run_full_sft_locked_mixture_megatron.sh \
  > resolved-mla-only.yaml

python examples/glm52_lora/verify_full_sft_config.py \
  resolved-mla-only.yaml --expected-max-length 768 --expected-steps 33 \
  --expected-train-file-count 3 --expected-val-file-count 3 \
  --expected-lora-profile mla-only
```

MLA plus output head:

```bash
CONFIG_ONLY=1 MODEL_PATH=/path/to/config-only-snapshot \
MIXTURE_DIR=/path/to/quality-mixture-targeted-wikipedia-2728 \
examples/glm52_lora/run_full_sft_locked_mixture_mla_lm_head_megatron.sh \
  > resolved-mla-lm-head.yaml

python examples/glm52_lora/verify_full_sft_config.py \
  resolved-mla-lm-head.yaml --expected-max-length 768 --expected-steps 33 \
  --expected-train-file-count 3 --expected-val-file-count 3 \
  --expected-lora-profile mla-lm-head
```

Expected global trainable counts are 106,149,888 for MLA-only and 108,726,272
for MLA+`lm_head`. At TP8 their local counts are 29,552,640 and 29,874,688.

## Quality decision

Run `compare_quality_outputs.py` on paired rows with identical IDs, prompt
hashes, decoding-contract hashes, and full semantic-score coverage. A result
is accepted only when Russian semantic quality is non-inferior and both the
required-Markdown validity and accidental-Han gates improve. Missing semantic
scores, a defect not reproduced in the base, or a confidence interval crossing
the decision boundary remains `PENDING`, never `PASS`.
