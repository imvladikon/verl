# GLM-5.2 LoRA validation

This directory validates LoRA SFT and GRPO on our own full-width GLM-5.2
surgery pair. The checkpoints are test fixtures, not chat or benchmark models.

- BF16 trainer anchor:
  [imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy](https://huggingface.co/imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy)
- mixed E4M3/BF16 rollout twin:
  `imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy-FP8`
- pair ID: `glm52-9b-2db222dcbd5d236a`

Both checkpoints keep the original width and MLA/DSA geometry. They use the
same ten donor layers and the same 16 experts in every retained MoE layer.

## Sequence

1. Create the deterministic SFT and GRPO smoke data:

   ```bash
   python examples/glm52_lora/prepare_smoke_data.py /path/to/data
   ```

2. Run two BF16 Megatron Bridge SFT steps on an audited free GPU:

   ```bash
   GPU_ID=5 \
   TRAIN_FILE=/path/to/data/sft.parquet \
   examples/glm52_lora/run_surgery_sft_megatron.sh
   ```

3. Verify the final adapter. The verifier requires the HF export, Megatron
   dist checkpoint, exact rank-16 MLA topology, finite BF16 tensors, and a
   nonzero update in every LoRA-B tensor:

   ```bash
   python examples/glm52_lora/verify_surgery_adapter.py \
     runs/glm52-lora-surgery-sft-megatron/global_step_2
   ```

   Then reload the exported HF adapter with the same resolved CUDA runtime and
   prove that it produces finite, nonzero logit changes:

   ```bash
   GPU_ID=5 \
   ADAPTER_PATH=/path/to/global_step_2/model/huggingface/adapter \
   examples/glm52_lora/run_verify_adapter_reload.sh
   ```

4. Pass the Megatron adapter checkpoint to the FP8 rollout run:

   ```bash
   GPU_ID=5 \
   TRAIN_FILE=/path/to/data/rl.parquet \
   SFT_ADAPTER_CKPT=/path/to/global_step_2/model/dist_ckpt \
   examples/glm52_lora/run_surgery_grpo_megatron_sglang.sh
   ```

5. Verify the final PPO actor adapter and its Megatron dist checkpoint:

   ```bash
   python examples/glm52_lora/verify_surgery_adapter.py \
     runs/glm52-lora-surgery-grpo-megatron-sglang/global_step_2/actor
   ```

The first profile adapts only the five MLA projections. `lm_head`, the DSA
indexer, shared experts, and routed experts are separate ablations after the
MLA-only save/reload, sharding, hot-sync, and finite-gradient gates pass.

`setup_vm_env.sh` creates a new checkout and venv and refuses to overwrite an
existing environment. It installs all four GitHub forks with normal dependency
resolution, checks out Megatron Bridge at the recorded exact revision, and
installs the matching CUDA-13 Transformer Engine and NVIDIA ModelOpt version
recorded by Bridge (needed by adapter-only checkpoint filtering). Bridge is
imported directly from that exact source checkout because its published
all-model metadata pins an older Transformers and adds unrelated diffusion
dependencies. No dependency is installed with `--no-deps`; existing venvs and
Ray clusters are untouched.

Before downloading model shards, verify that the Hub config selects the GLM
bridge and preserves the surgery model's full-width MLA/DSA/MoE contract:

```bash
python examples/glm52_lora/verify_bridge_contract.py
```

## Quality data and evaluation

The six-row smoke files above prove integration only. They are not a Russian
quality corpus. Curated full-model data uses the schema demonstrated by
`quality_dataset.example.jsonl` and is validated before conversion:

```bash
python examples/glm52_lora/build_quality_dataset.py \
  curated_quality.jsonl runs/glm52-quality-data
```

Build the pinned public-source review queue separately:

```bash
python examples/glm52_lora/build_quality_review_queue.py \
  runs/glm52-quality-review
```

Its rows are deliberately marked `pending`, and the training builder rejects
them until a named reviewer changes the status to `accepted`. Source locks,
selection policy and explicit exclusions are documented in
[`QUALITY_SOURCES.md`](QUALITY_SOURCES.md).

Generate the separate teacher-free targeted set for the three failure modes:

```bash
python examples/glm52_lora/generate_targeted_quality_data.py \
  runs/glm52-targeted-quality
```

It contains 720 deterministic transformations of facts already present in the
prompt: 128 natural-Russian rewrites, 384 Markdown renderings, 32 additional
Markdown/code controls, 128 accidental-Han removals, 48 Han scope controls,
and 32 intentional-Chinese retention controls. Semantic groups, not prompt
wordings, determine the 540/90/90 train/validation/test split. This set targets
specific behavior; it is not a substitute for reviewed general-Russian data.

Measure the generated rows with the exact checkpoint tokenizer and chat
template before choosing a sequence length:

```bash
python examples/glm52_lora/audit_quality_tokens.py \
  runs/glm52-targeted-quality/targeted_quality.jsonl \
  /path/to/GLM-5.2-tokenizer \
  --output runs/glm52-targeted-quality/token_audit.json
```

The pinned surgery tokenizer audit of `targeted-template-v1` measured 79,674
full-chat tokens. Full-chat length was p50 105, p95 166, p99 180, and maximum
187 tokens, so sequence length 256 covers this targeted set without
truncation. Re-run the audit after adding reviewed public-source rows; those
lengths are not covered by this measurement.

The builder rejects unreviewed rows, missing provenance, duplicate/leaking
prompts across splits, accidental Han in Russian targets, structurally broken
Markdown, and Russian targets without Cyrillic. It writes SFT parquet files,
held-out contracts, and a separately named `rl_constraint_smoke.parquet`. The
latter is deliberately not a production RL dataset because it has no
semantic-quality reward.

Evaluate generated base and adapter outputs independently, then compare the
same held-out IDs:

```bash
python examples/glm52_lora/evaluate_quality_outputs.py \
  adapter_predictions.jsonl --details adapter_details.jsonl
```

Each prediction carries its contract and, when available, exact generated
token count and a separately produced semantic score. The evaluator reports
Markdown structural validity, conditional accidental-Han rates, Han per 1,000
tokens, Russian-script diagnostics, semantic-score coverage, and never
silently substitutes a character estimate for missing token counts.

`quality_reward.py` is only the deterministic constraint component. It masks
code, URLs and link destinations, permits Han only under an explicit Chinese,
Japanese, global, or blockquote contract, and parses CommonMark plus tables.
Combine it with an independent semantic/task reward after the SFT adapter has
already passed held-out evaluation.

For the first full-model quality experiment compare rank-16 MLA-only against
rank-16 MLA plus `lm_head` with identical data, seed, tokens and updates. Do
not begin with routed-expert LoRA. The surgery checkpoint cannot decide this
quality comparison; it only qualified the engineering path.

The `lm_head` engineering ablation is reproducible with:

```bash
GPU_ID=5 \
TRAIN_FILE=/path/to/sft_train.parquet \
examples/glm52_lora/run_surgery_sft_mla_lm_head_megatron.sh

python examples/glm52_lora/verify_surgery_adapter.py \
  runs/glm52-lora-surgery-sft-mla-lm-head/global_step_2 \
  --include-lm-head
```

On the 9B surgery model it exported and reloaded 102 finite BF16 tensors and
16,185,344 trainable parameters. Peak CUDA allocated/reserved was
17.753/18.053 GiB, versus 17.726/17.973 GiB for MLA-only. The first-batch loss
was identical; the second was 12.992048 versus 12.982867 for MLA-only. This
proves the head's training/export/reload path but gives no quality reason to
prefer it; that choice remains a controlled full-model held-out ablation.

## Full-model SFT qualification profile

`run_full_sft_megatron.sh` is the fail-closed 753B qualification profile. It
locks the immutable GLM-5.2 config and targeted train/validation hashes, five
MLA targets at rank 16 / alpha 32, sequence length 256, and the source-derived
64-H200 topology (TP8, EP32, DP8). Resolve it without a model download or GPU:

```bash
CONFIG_ONLY=1 \
MODEL_PATH=/path/to/config-only-snapshot \
TRAIN_FILE=/path/to/sft_train.parquet \
VAL_FILE=/path/to/sft_validation.parquet \
examples/glm52_lora/run_full_sft_megatron.sh > resolved.yaml

python examples/glm52_lora/verify_full_sft_config.py resolved.yaml
```

The validator reports `CONFIG-PASS/RUNTIME-PENDING`; this is not a claim that
the 753B checkpoint has trained. A real launch additionally requires the exact
snapshot sentinel, eight exclusive H200s on every node, an explicit operator
acknowledgement, and the SHA-256 from a prior TP2 adapter save/reload gate.

Keep activation recomputation disabled for this short-sequence BSHD profile.
An empirical 9B run with uniform one-layer full recomputation failed because
backward recomputed DSA skip layer 10 before source layer 7 and the cross-layer
top-k holder was empty. With recomputation disabled, the same two batches
matched the prior THD losses and gradient norms exactly (`14.749032 / 221.216156`
and `12.982867 / 20.208773`), while peak CUDA allocated/reserved rose from
16.878/17.180 GiB to 17.726/17.973 GiB. Re-evaluate a DSA-aware recomputation
schedule before increasing sequence length rather than enabling per-layer full
recomputation.
