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

   The one-GPU gate starts a private Ray instance under a short socket path,
   disables its dashboard frontend, and defaults to 8 CPUs, a 4-GiB object
   store, and an 1800-second SGLang watchdog for first-request JIT. Override
   these with `RAY_TMPDIR`, `RAY_NUM_CPUS`,
   `RAY_OBJECT_STORE_MEMORY_BYTES`, and `SGLANG_WATCHDOG_SECONDS`; the script
   validates every value before starting Ray.

5. Verify the final PPO actor adapter and its Megatron dist checkpoint:

   ```bash
   python examples/glm52_lora/verify_surgery_adapter.py \
     runs/glm52-lora-surgery-grpo-megatron-sglang/global_step_2/actor
   ```

   Preserve and validate the step metrics, GPU/RSS telemetry, log hashes, and
   the same actor checkpoint as one fail-closed record:

   ```bash
   python examples/glm52_lora/verify_surgery_grpo_run.py \
     runs/glm52-lora-surgery-grpo-megatron-sglang \
     --log /path/to/grpo.log --expected-steps 2
   ```

   For a run resumed from step 2 through step 3, add
   `--expected-steps 3 --first-step 3 --resumed-from-step 2`. The verifier
   requires the log to start at `2/3`, rather than silently restarting.

## TP2 save/resume gate

Use two independently audited free 80-GiB GPUs to exercise tensor and sequence
parallelism plus the distributed optimizer. The gate trains steps 1-2, saves
adapter/optimizer/RNG/dataloader state, starts a fresh TP2 process, resumes
exactly at step 2, trains step 3, exports the adapter, and reloads it through
Transformers on one GPU:

```bash
TP_GATE_ACK=GLM52_TP2_MLA_R16 GPU_IDS=5,7 \
MODEL_PATH=/path/to/immutable/GLM-5.2-9B-LoRA-Surgery-Dummy \
TRAIN_FILE=/path/to/locked/sft_train.parquet \
examples/glm52_lora/run_surgery_sft_tp2_gate.sh
```

`verify_tp2_sft_config.py` rejects topology or resume-path drift before each
training phase. `verify_tp2_resume_run.py` rejects missing rank-local loads,
wrong step numbers, restarted dataloaders, and non-finite losses or gradients.
The qualified run observed token counts `95, 91, 149`, finite losses
`14.76285, 13.00762, 12.62799`, finite gradient norms
`243.89970, 19.79066, 9.55080`, and peak sampled memory of 17,786/17,014 MiB on
the two GPUs. The final 13,608,960-parameter adapter produced a finite nonzero
logit delta after a fresh reload. Its complete evidence-manifest root is
`80ce91da59c5615618b03c14fb74163374c7bb8e529c699ab0a661cfcd0ee958`.

## EP2 routed-expert gate

The complementary two-GPU gate sets `TP=1`, `EP=2`, `DP=2`, and
expert-DP=1. Each rank owns eight of the surgery model's 16 routed experts;
MLA-only LoRA stays replicated and its optimizer is distributed across DP:

```bash
EP_GATE_ACK=GLM52_EP2_MLA_R16 GPU_IDS=5,7 \
MODEL_PATH=/path/to/immutable/GLM-5.2-9B-LoRA-Surgery-Dummy \
TRAIN_FILE=/path/to/locked/sft_train.parquet \
examples/glm52_lora/run_surgery_sft_ep2_gate.sh
```

The gate saves separate `data_0.pt` and `data_1.pt`, resumes both DP ranks in
a fresh process, and requires the locked global-token sequence `186, 288,
214`. The qualified run observed finite losses
`14.51996, 13.63048, 12.50402`, finite gradient norms
`218.45088, 34.82770, 11.59513`, and sampled peaks of 19,210/19,056 MiB. Its
final adapter reloaded with a finite nonzero logit delta. Evidence-manifest
root: `a6a739c9e8a8031e89506da1f582b0255b5513823d5ace17b4fe5f723aa0ee13`.
This validates routed-expert sharding without expert LoRA; it does not by
itself validate the full model's EP32 topology.

The first profile adapts only the five MLA projections. `lm_head`, the DSA
indexer, shared experts, and routed experts are separate ablations after the
MLA-only save/reload, sharding, hot-sync, and finite-gradient gates pass.

`setup_vm_env.sh` creates a new checkout and venv and refuses to overwrite an
existing environment. It installs all four GitHub forks with normal dependency
resolution, checks out Megatron Bridge at the recorded base revision, and
cherry-picks the exact upstream GLM-5 FP8 dequantization change. It verifies
the two-file diff hash before installation, then installs the matching CUDA-13
Transformer Engine and NVIDIA ModelOpt version recorded by Bridge (needed by
adapter-only checkpoint filtering). Bridge is imported directly from that
source checkout because its published
all-model metadata pins an older Transformers and adds unrelated diffusion
dependencies. No dependency is installed with `--no-deps`; existing venvs and
Ray clusters are untouched.

Before downloading model shards, verify that the Hub config selects the GLM
bridge and preserves the surgery model's full-width MLA/DSA/MoE contract:

```bash
python examples/glm52_lora/verify_bridge_contract.py
```

## Quality data and evaluation

The small-checkpoint decision boundary is recorded in
[`QUALITY_ORACLE.md`](QUALITY_ORACLE.md). Our 9B surgery pair is the primary
engineering fixture, while only the pinned full `zai-org/GLM-5.2` checkpoint
can support a Russian/Markdown/accidental-Han quality claim. Public 0.8B test
and DSpark draft checkpoints are not quality proxies.

The production trainer choice, independent full-model evidence, exact LoRA
parameter counts, and AutoModel/Bridge/Slime/Axolotl tradeoffs are recorded in
[`LORA_FRAMEWORKS.md`](LORA_FRAMEWORKS.md).

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

Build a second teacher-free set from authentic Russian prose without a model
teacher. The sampler locks the dataset revision, keeps source URLs and text
hashes, and materializes only three accepted sentences per article:

```bash
python examples/glm52_lora/sample_wikipedia_ru.py \
  runs/glm52-wikipedia/source_sample.jsonl \
  --max-articles 512 --max-source-rows 20000
python examples/glm52_lora/build_teacher_free_russian_corruptions.py \
  runs/glm52-wikipedia/source_sample.jsonl \
  runs/glm52-wikipedia/artifacts
python examples/glm52_lora/audit_quality_tokens.py \
  runs/glm52-wikipedia/artifacts/teacher_free_rows.jsonl \
  /path/to/immutable/GLM-5.2-tokenizer \
  --output runs/glm52-wikipedia/artifacts/token_audit.json
```

The pinned 64-article engineering sample produced 244 rows from 61 unique
articles after removing three duplicate-prompt groups. Its exact surgery
tokenizer audit measured full-chat p95 472, p99 546, and maximum 556 tokens.
Use the dedicated no-truncation bucket instead of the old 256-token smoke
profile:

```bash
GPU_ID=5 \
MODEL_PATH=/path/to/immutable/GLM-5.2-9B-LoRA-Surgery-Dummy \
TRAIN_FILE=runs/glm52-wikipedia/artifacts/sft_train.parquet \
MAX_LENGTH=640 REQUIRED_MAX_TOKENS=556 \
examples/glm52_lora/run_surgery_sft_teacher_free_megatron.sh
```

Re-run the token audit for every larger materialization and set
`REQUIRED_MAX_TOKENS` to its measured maximum. The generated `NOTICE.md` and
`ATTRIBUTION.jsonl` are part of the artifact; production use remains gated on
license and content review.

The default 512-article materialization was also audited independently. It
accepted 502 article groups and emitted 2,008 rows (1,588/244/176 by split).
The builder writes two disjoint sequence buckets automatically:

- `corrections/`: 1,506 rows, full-chat maximum 357; qualify with `seq=384`;
- `markdown/`: 502 rows, full-chat maximum 706; qualify with `seq=768`.

For example, run the buckets separately so short correction examples do not
pay the Markdown activation-memory cost:

```bash
TRAIN_FILE=runs/glm52-wikipedia/artifacts/corrections/sft_train.parquet \
MAX_LENGTH=384 REQUIRED_MAX_TOKENS=357 GPU_ID=5 MODEL_PATH=/path/to/model \
examples/glm52_lora/run_surgery_sft_teacher_free_megatron.sh

TRAIN_FILE=runs/glm52-wikipedia/artifacts/markdown/sft_train.parquet \
MAX_LENGTH=768 REQUIRED_MAX_TOKENS=706 GPU_ID=5 MODEL_PATH=/path/to/model \
examples/glm52_lora/run_surgery_sft_teacher_free_megatron.sh
```

The buckets preserve article-level splits and attribution. Audit both bucket
JSONL files after every rematerialization; the 357/706 bounds belong only to
the pinned 512-article candidate.

Both bucket profiles passed two steps on the 9B surgery checkpoint. The
384-token correction run measured loss `15.461869 -> 13.652021`, gradient norm
`261.420868 -> 30.508768`, and peak CUDA allocated/reserved
`17.934/18.223 GiB`. The 768-token Markdown run measured loss
`15.846344 -> 13.241395`, gradient norm `263.868103 -> 21.209194`, and peak
CUDA allocated/reserved `20.328/21.117 GiB`. Each exported 100 finite adapter
tensors with all 50 LoRA-B tensors nonzero. Matching full-model TP8/EP32
profiles resolve as `CONFIG-PASS/RUNTIME-PENDING` at both sequence lengths.

Build the locked ASAP mixture from the 720 project-authored rows and the 2,008
authentic-text rows with the exact surgery tokenizer:

```bash
python examples/glm52_lora/build_token_bucket_mixture.py \
  runs/glm52-quality-mixture /path/to/immutable/GLM-5.2-tokenizer \
  --input targeted-template-v2 /path/to/targeted_quality.jsonl \
    2f6072525e971fa5798473049078c0209b51fd799a0d4d95781901527700a938 \
  --input wikipedia-corruption-v1 /path/to/teacher_free_rows.jsonl \
    5841e7a00dd6109269d9a04d92ccfad26b207ab82b6570b17371b6d04f9a0078 \
  --bucket 256 --bucket 384 --bucket 768 \
  --tokenizer-json-sha256 \
    19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d \
  --tokenizer-config-sha256 \
    98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc
```

The exact output has 2,728 rows: 2,184 in `seq256`, 259 in `seq384`,
and 285 in `seq768`. Observed maxima are exactly 256, 384, and 706. A second
build was byte-identical, including every train Parquet. The builder rejects
source/tokenizer hash drift, duplicate IDs or prompts, unaccepted rows, empty
buckets, and any sequence above the largest bucket. The three matching
full-model configs all validate as `CONFIG-PASS/RUNTIME-PENDING`.

Before spending a full-model allocation, run the exact three-file input as one
optimizer stream on the surgery checkpoint:

```bash
MIXTURE_DIR=/path/to/quality-mixture-targeted-wikipedia-2728 \
GPU_ID=5 \
examples/glm52_lora/run_surgery_sft_locked_mixture_megatron.sh
```

This performs 33 optimizer updates, matching one full-model epoch's update
count at global batch 64. The one-GPU surgery gate deliberately uses batch
size one, so it proves mixed-length input handling and optimizer stability,
not full-batch numerical parity or language quality. A single concatenated
dataset keeps one optimizer; chaining three adapter-only jobs would reset the
optimizer, while ordinary checkpoint resume would incorrectly restore the
previous bucket's dataloader state.

The `targeted-template-v2` gate completed all 33 updates on the 9B surgery
checkpoint. Loss was finite from `14.3883` at step 1 through `8.6188` at step
33; every gradient norm was finite. Peak CUDA allocated/reserved was
`21.218/21.809 GiB` and the external sampler observed `25,254 MiB`. Export
contained 100 finite adapter tensors with all 50 LoRA-B tensors nonzero; reload
changed logits by L2 `3594.38`. The evidence root is
`d62fd1a61489d3790004c654193fa6a7c6664740a6ef056f7972c4eeb742fcf2`.
This remains an engineering gate, not evidence that the full model improved.

Measure the generated rows with the exact checkpoint tokenizer and chat
template before choosing a sequence length:

```bash
python examples/glm52_lora/audit_quality_tokens.py \
  runs/glm52-targeted-quality/targeted_quality.jsonl \
  /path/to/GLM-5.2-tokenizer \
  --output runs/glm52-targeted-quality/token_audit.json
```

The pinned surgery tokenizer audit of `targeted-template-v2` measured 79,466
full-chat tokens. Full-chat length was p50 105, p95 171, p99 185, and maximum
191 tokens, so sequence length 256 covers this targeted set without
truncation. Re-run the audit after adding reviewed public-source rows; those
lengths are not covered by this measurement.

The builder rejects unreviewed rows, missing provenance, duplicate/leaking
prompts across splits, accidental Han in Russian targets, structurally broken
Markdown, and Russian targets without Cyrillic. It writes SFT parquet files,
held-out contracts, and a separately named `rl_constraint_smoke.parquet`. The
latter is deliberately not a production RL dataset because it has no
semantic-quality reward.

Generate the paired outputs through one exact SGLang runtime. First write a
JSON object containing the actual server arguments. It must include
`model_path`, `served_base_model`, `endpoint`, `tp_size`, the physical
`gpu_ids`, `max_model_len`, strict LoRA fields, and the adapter's exact target
modules. Put any required CUDA library directories in `ld_library_paths`.
Optional kernel choices such as `attention_backend` and the three DSA backends
belong in the same object; there is no unrecorded command-line escape hatch.
Build the manifest from real immutable trainer and inference snapshots,
the verified adapter, and the clean SGLang checkout that will actually be
imported:

```bash
python examples/glm52_lora/build_quality_sglang_runtime.py \
  --trainer-model-path /snapshots/GLM-5.2/REVISION \
  --model-path /snapshots/GLM-5.2-FP8/REVISION \
  --adapter-path /runs/mla-only/adapter \
  --adapter-verification /runs/mla-only/adapter_verification.json \
  --adapter-name glm52-quality-mla-r16 --profile mla-only \
  --sglang-checkout /src/sglang --server-args server-args.json \
  --server-instance-id glm52-quality-validation-1 \
  --output runtime-validation.json
```

The builder verifies both model revisions from their snapshot directory name
or revision sentinel, hashes both configs and weight indexes, checks the full
78-layer contract, and binds the adapter serialization to its verification
record. Launch exactly that manifest; `PYTHONPATH`, `CUDA_VISIBLE_DEVICES`, and
any `LD_LIBRARY_PATH` entries must match the bound checkout and runtime:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=/src/sglang/python \
python examples/glm52_lora/launch_quality_sglang_server.py \
  --runtime-manifest runtime-validation.json
```

Then generate base and adapter rows with the same endpoint, runtime manifest,
decoding contract, and server instance:

```bash
for VARIANT in base adapter; do
  python examples/glm52_lora/generate_full_quality_outputs_sglang.py \
    /path/to/seq256/eval_contracts.jsonl \
    /path/to/seq384/eval_contracts.jsonl \
    /path/to/seq768/eval_contracts.jsonl \
    --split validation --variant "$VARIANT" \
    --runtime-manifest runtime-validation.json \
    --endpoint http://127.0.0.1:30000/v1/chat/completions \
    --output "$VARIANT-validation.jsonl" \
    --manifest "$VARIANT-validation-manifest.json"
done
```

For a surgery-checkpoint plumbing test, pass the exact acknowledgement
`nonofficial-checkpoint-output-is-not-quality-evidence` to the builder,
launcher, and generator. Those rows are permanently marked
`quality_claim_allowed=false` and cannot pass the comparator or blinded
review gate.

The surgery plumbing test was run on one A100 with the BF16 9B fixture,
adapter SHA-256 `721f134b…c63e7`, and SGLang `9377c437…55c2`. The model loaded
16.89 GB of weights and the configured server occupied 61,736 MiB including
its static pools. Strict LoRA loading completed. On an identical deterministic
request the base and adapter both selected EOS, as expected from the dummy,
but its first-token log-prob changed from `-0.456743` to `-4.043040`. This
proves that the adapter affected the SGLang forward path; the production
generator correctly rejected the empty dummy completion as quality output.

Evaluate generated base and adapter outputs independently, then compare the
same held-out IDs:

```bash
python examples/glm52_lora/evaluate_quality_outputs.py \
  adapter_predictions.jsonl --details adapter_details.jsonl
python examples/glm52_lora/compare_quality_outputs.py \
  base_predictions.jsonl adapter_predictions.jsonl \
  --details paired_details.jsonl
```

Produce semantic scores through a blinded paired review rather than inferring
them from Cyrillic ratio or the structural checks. Generate a secret key
outside the repository, prepare an unlabeled A/B packet, and give unchanged
copies to at least two reviewers:

```bash
openssl rand -hex 32 > /secure/path/glm52-validation-review.key
python examples/glm52_lora/build_blind_quality_review.py prepare \
  /path/to/seq256/eval_contracts.jsonl \
  /path/to/seq384/eval_contracts.jsonl \
  /path/to/seq768/eval_contracts.jsonl \
  --split validation \
  --base base-validation.jsonl --adapter adapter-validation.jsonl \
  --blinding-key-file /secure/path/glm52-validation-review.key \
  --packet validation-review.jsonl --manifest validation-review-manifest.json
```

Each reviewer fills only `review` in their copy. The tool rejects changed
prompts, contracts, completions, hashes, incomplete ratings, and duplicate
reviewer identities. The key is never written to an artifact; the manifest
stores only its SHA-256 commitment. Adjudicate the completed copies and feed
the scored rows to the existing comparator:

```bash
python examples/glm52_lora/build_blind_quality_review.py adjudicate \
  /path/to/seq256/eval_contracts.jsonl \
  /path/to/seq384/eval_contracts.jsonl \
  /path/to/seq768/eval_contracts.jsonl \
  --split validation \
  --base base-validation.jsonl --adapter adapter-validation.jsonl \
  --blinding-key-file /secure/path/glm52-validation-review.key \
  --review reviewer-1.jsonl --review reviewer-2.jsonl \
  --base-output base-validation-scored.jsonl \
  --adapter-output adapter-validation-scored.jsonl \
  --manifest validation-adjudication.json

python examples/glm52_lora/compare_quality_outputs.py \
  base-validation-scored.jsonl adapter-validation-scored.jsonl \
  --details validation-paired-details.jsonl
```

Use validation to choose MLA-only versus the conditional `lm_head` ablation.
Generate and review the test outputs once, only after selection, with a new
key. Do not publish or commit either key.

Each prediction carries its contract and, when available, exact generated
token count and a separately produced semantic score. The evaluator reports
Markdown structural validity, conditional accidental-Han rates, Han per 1,000
tokens, Russian-script diagnostics, semantic-score coverage, and never
silently substitutes a character estimate for missing token counts.

The paired comparator additionally requires identical IDs, contracts, prompt
hashes, decoding-contract hashes, the same exact runtime-manifest hash, and
paired semantic-score coverage. It programmatically rejects unproven or
surgery-checkpoint output. It uses a deterministic paired bootstrap and
remains `PENDING` unless all three target failures are reproduced in the base
outputs and improve without Russian semantic regression. When non-Russian
retention rows are present, their semantic scores also require full paired
coverage and non-inferiority; accidental-Han cleanup cannot pass by destroying
legitimate Chinese ability.

`quality_reward.py` is only the deterministic constraint component. It masks
code, URLs and link destinations, permits Han only under an explicit Chinese,
Japanese, global, or blockquote contract, and parses CommonMark plus tables.
Combine it with an independent semantic/task reward after the SFT adapter has
already passed held-out evaluation.

For the first full-model quality experiment compare rank-16 MLA-only against
rank-16 MLA plus `lm_head` with identical data, seed, tokens and updates. Do
not begin with routed-expert LoRA. The surgery checkpoint cannot decide this
quality comparison; it only qualified the engineering path.

The executable status matrix and sequential decision gate are in
[`QUALITY_EXECUTION_GATE.md`](QUALITY_EXECUTION_GATE.md). The locked production
wrappers differ only in adapter surface, acknowledgement, output directory,
and experiment name; both pin seed 52 and the same 33-update data stream.
The same gate documents the bounded hosted full-model baseline preflight and
its explicit limitation: provider routing cannot prove an exact Hub revision.

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
locks the immutable GLM-5.2 config and targeted train/validation hashes plus
five MLA targets at rank 16 / alpha 32. The bounded smoke defaults to sequence
length 256; the locked quality mixture uses 768. The launcher defaults to the
original W64/TP8/EP32 plan. The currently qualified family accepts explicit
W/EP choices with TP8, ETP1, PP1, CP1, eight GPUs per node, and
EP8/16/32/128. The header- and capacity-qualified first official-FP8
seq768 candidate is W32/TP8/EP32 on measured 141-GiB devices; it is still
runtime-pending. W128/TP8/EP128 remains a valid process grid, but no longer
passes the 80-GiB seq768 memory gate.
Every real launch must pass the memory planner using the smallest capacity
reported by `nvidia-smi` across the allocated visible GPUs. Resolve a config
without a model download or GPU:

```bash
CONFIG_ONLY=1 \
MODEL_PATH=/path/to/config-only-snapshot \
TRAIN_FILE=/path/to/sft_train.parquet \
VAL_FILE=/path/to/sft_validation.parquet \
examples/glm52_lora/run_full_sft_megatron.sh > resolved.yaml

python examples/glm52_lora/verify_full_sft_config.py resolved.yaml
```

The validator includes a config-only parameter and static-memory calculation.
It can also be run directly against the 1.5-TB checkpoint's small
`config.json`; it never opens a weight shard:

```bash
python examples/glm52_lora/estimate_full_sft_memory.py \
  /path/to/GLM-5.2/config.json \
  --tp 8 --ep 32 --etp 1 --lora-rank 16 --sequence-length 768 \
  --expect-policy-parameters 743377000704
```

Compare candidate process grids before reserving hardware:

```bash
python examples/glm52_lora/plan_full_sft_topologies.py \
  /path/to/GLM-5.2/config.json \
  --candidate 8:8:8:1 --candidate 16:8:16:1 \
  --candidate 32:8:32:1 --candidate 64:8:32:1 \
  --candidate 128:8:128:1 \
  --device-capacity-gib 141 --sequence-length 768
```

The syntax is `WORLD:TP:EP:ETP[:PP[:CP]]`. A real launcher adds
`--require-candidate`, so `MARGINAL` and every `REJECT-*` result stop before
checkpoint scanning or `torchrun`. Capacity is measured, never inferred from
a pool or accelerator name. The planner can compare broader process grids,
but the real launcher rejects grids outside the empirically resolved family.

The analytic split is 724,775,731,200 routed-expert parameters,
1,573,443,840 TP-replicated parameters, and 17,027,825,664 TP-sharded
parameters. It exactly reproduces both the official 743,377,000,704-policy
parameter count and the per-rank numel observed in our TP2 and EP2 surgery
runs. At TP8/EP32/ETP1 each rank holds 26,351,163,648 base parameters
(`49.083 GiB` BF16) and 29,552,640 rank-16 adapter parameters. A conservative
18-byte adapter-state estimate makes the static total `49.578 GiB` per GPU.
This uses Megatron's documented BF16/FP32-gradient DP1 bound; distributed
optimizer sharding only reduces it when dense DP is greater than one.

For W32/TP8/EP32, scaling the worst measured 9B locked-mixture allocation
(`21.218 GiB`, stable from step 3 through step 33) by policy layers and tokens
projects `85.983 GiB` PyTorch allocated. The estimator reports a deliberately
padded `112.186 GiB` planning envelope
(static state plus 1.5x the projected non-static residual and 8 GiB for
non-PyTorch CUDA and communication workspaces). At seq768, the corresponding
envelopes are `238.748 GiB` for
W8/TP8/EP8 and `154.373 GiB` for W16/TP8/EP16. Consequently W8 is a candidate
at measured 270 GiB, W16 is rejected at 141 GiB, and W32 is a candidate at
141 GiB. These are analytic dispositions, not runtime proofs: conversion
staging, allocator behavior, collectives, and the real 78-layer DSA schedule
can differ. They authorize only a guarded first qualification attempt.

For the available 80-GiB allocation, W128/TP8/EP128 has 17.938 GiB of static
BF16 model and rank-16 MLA adapter state per GPU. The measured-surgery
projection is 54.343 GiB and the padded planning envelope is 80.545 GiB. It is
therefore `REJECT-ENVELOPE` at seq768 even before the separate 8-GiB minimum
post-envelope headroom is enforced. Use this grid only after a shorter
sequence, DSA-safe recomputation, context parallelism, or another memory
reduction has its own runtime qualification.

The official FP8 profile is selected explicitly. The source identity is a
configuration lock; an infrastructure-specific delivery recipe must
separately bind it to an immutable object-store node and revision:

```bash
CHECKPOINT_PROFILE=official-fp8-dequant \
MODEL_SOURCE_IDENTITY=glm52-official-fp8-d1539d36-e0fe7f28 \
NNODES=16 TP_SIZE=8 EP_SIZE=128 \
MODEL_PATH=/path/to/materialized/official/GLM-5.2-FP8 \
TRAIN_FILE=/path/to/sft_train.parquet \
VAL_FILE=/path/to/sft_validation.parquet \
examples/glm52_lora/run_full_sft_megatron.sh
```

For a config-only render of that topology, add `CONFIG_ONLY=1` and validate
the result with `verify_full_sft_config.py --expected-nnodes 16
--expected-ep 128`.

For the pinned authentic-Russian sample, keep the same fail-closed full-model
profile but replace the dataset locks and audited sequence bound:

```bash
CONFIG_ONLY=1 MAX_LENGTH=640 REQUIRED_MAX_TOKENS=556 \
EXPECTED_TRAIN_SHA256=534170e62865ebbab0e65439ad8c60785dc242b6c3a56c8c9ad9e19afaf9971b \
EXPECTED_VAL_SHA256=14041be9480c77edb7e14019b32e2caf76e065323cec83ea6deca02fa597620c \
MODEL_PATH=/path/to/config-only-snapshot \
TRAIN_FILE=/path/to/wikipedia/sft_train.parquet \
VAL_FILE=/path/to/wikipedia/sft_validation.parquet \
examples/glm52_lora/run_full_sft_megatron.sh > resolved-wikipedia.yaml

python examples/glm52_lora/verify_full_sft_config.py \
  resolved-wikipedia.yaml --expected-max-length 640
```

The validator reports `CONFIG-PASS/RUNTIME-PENDING`; this is not a claim that
the 753B checkpoint has trained. A real launch additionally requires the exact
snapshot sentinel, exactly the requested number of idle visible GPUs on every
node, a measured-capacity planner `CANDIDATE`, a topology-specific operator
acknowledgement, and the exact evidence roots from the passed TP2 adapter,
EP2 expert-routing, and combined TP2xEP2 save/reload gates.

Before `torchrun`, every node also runs `audit_full_checkpoint_loading.py`.
For the BF16 profile it reads only the 7.12 MiB of safetensors JSON headers,
never tensor payloads, and requires the exact 282-shard index. That checkpoint
contains 58,368
separate routed-expert tensors: each expert projection is 24 MiB rather than a
single stacked terabyte-scale tensor. Of those, 768 belong to the disabled MTP
layer 78, leaving 57,600 expert tensors in the policy import. The largest
source tensor is the 1.773 GiB embedding or language-model head; a 5.0 GiB
shard is not materialized as one tensor.

At TP8/EP32/ETP1/PP1, the pinned Bridge importer reads HF tensors before its
rank-local TP/ETP scatter. The conservative logical traffic is therefore
active MTP-disabled import is `34.648 GiB non-expert × 64 + 1,350 GiB expert ×
2 = 4.802 TiB` across the job, or 76.835 GiB per rank on average. Scanning the
otherwise unused MTP layer gives a 4.871 TiB whole-checkpoint upper bound.
Shared filesystem page cache may lower physical backing-store traffic, but the
launch gate does not assume it. Each node writes its own
`full-hf-load-audit-nodeN.json`; use those files to separate slow streaming
from a deadlock during the first full qualification.

The `official-fp8-dequant` profile instead locks 141 shards, 59,044 E4M3
weights, their 59,044 FP32 inverse scales, every 128x128 scale shape, and the
BF16 exclusion list. Its 703.723-GiB source expands tensor-by-tensor to the
same 1,486,754,078,208-byte policy representation as the BF16 checkpoint. At
W128/TP8/EP128, the corresponding logical source traffic is 3.061 TiB. The
launcher also requires the exact upstream Bridge FP8-import diff on top of
the qualified Bridge base before it can reach `torchrun`.

The production candidate consumes all three locked train buckets in one
optimizer stream and all three disjoint validation buckets at the final step.
The three test buckets are hash-locked but never used for training or model
selection:

```bash
CONFIG_ONLY=1 \
MODEL_PATH=/path/to/config-only-snapshot \
MIXTURE_DIR=/path/to/quality-mixture-targeted-wikipedia-2728 \
examples/glm52_lora/run_full_sft_locked_mixture_megatron.sh \
  > resolved-locked-mixture.yaml

python examples/glm52_lora/verify_full_sft_config.py \
  resolved-locked-mixture.yaml --expected-lora-profile mla-only \
  --expected-max-length 768 --expected-steps 33 \
  --expected-train-file-count 3 --expected-val-file-count 3
```

Resolve the otherwise identical MLA+`lm_head` ablation with:

```bash
CONFIG_ONLY=1 \
MODEL_PATH=/path/to/config-only-snapshot \
MIXTURE_DIR=/path/to/quality-mixture-targeted-wikipedia-2728 \
examples/glm52_lora/run_full_sft_locked_mixture_mla_lm_head_megatron.sh \
  > resolved-locked-mixture-mla-lm-head.yaml

python examples/glm52_lora/verify_full_sft_config.py \
  resolved-locked-mixture-mla-lm-head.yaml \
  --expected-lora-profile mla-lm-head \
  --expected-max-length 768 --expected-steps 33 \
  --expected-train-file-count 3 --expected-val-file-count 3
```

The two global trainable counts are 106,149,888 and 108,726,272. At TP8,
MLA+`lm_head` adds 322,048 local parameters over the 29,552,640-parameter
MLA-only adapter. The second full-model run is conditional: evaluate MLA-only
first rather than launching both experiments together.

This wrapper remains fail-closed on a topology-specific acknowledgement such
as `GLM52_FULL_W32_TP8_EP32_MLA_R16`, the exact three sharding evidence roots,
idle-device checks, and a measured-capacity planner pass.
`CONFIG-PASS/RUNTIME-PENDING` is not permission or evidence for a full-model
launch.

Keep activation recomputation disabled for this short-sequence BSHD profile.
An empirical 9B run with uniform one-layer full recomputation failed because
backward recomputed DSA skip layer 10 before source layer 7 and the cross-layer
top-k holder was empty. With recomputation disabled, the same two batches
matched the prior THD losses and gradient norms exactly (`14.749032 / 221.216156`
and `12.982867 / 20.208773`), while peak CUDA allocated/reserved rose from
16.878/17.180 GiB to 17.726/17.973 GiB. Re-evaluate a DSA-aware recomputation
schedule before increasing sequence length rather than enabling per-layer full
recomputation.
