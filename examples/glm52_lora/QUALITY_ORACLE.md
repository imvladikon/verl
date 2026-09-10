# GLM-5.2 quality-oracle audit

The full GLM-5.2 checkpoint is the only valid oracle for claims about Russian
quality, Markdown validity, or accidental Han characters. Smaller checkpoints
remain useful for systems tests, but their output quality cannot substitute
for a held-out base-versus-adapter comparison on the full model.

The dataset side is similarly fail-closed. V2 and v3 are historical invalid
split-leak artifacts; `mixture_targeted_wikipedia_v4_2240` is the only clean
candidate. A clean split audit does not itself authorize a production claim:
the overall result remains `PENDING` until the exact trainer and inference
weight-shard manifests are trusted, local full-read receipts verify those
shards, and the paired runtime/output artifacts below are preserved.

This audit was performed with Hugging Face Hub metadata and model cards on
2026-09-02. Revisions are pinned so later repository changes cannot silently
change the conclusion.

## Candidate audit

| checkpoint | pinned revision | what it is | quality use |
|---|---|---|---|
| [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2/tree/cf457fa734ab149ffef225f80893eb38c6ff5cdc) | `cf457fa734ab149ffef225f80893eb38c6ff5cdc` | Official BF16 `glm_moe_dsa` checkpoint; Hub storage is 1,506,689,458,421 bytes. | Required base checkpoint and quality oracle. |
| [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8/tree/f33c6dc501ee5a2c7e35155653b1b1abbc320951) | `f33c6dc501ee5a2c7e35155653b1b1abbc320951` | Official mixed FP8/BF16 inference checkpoint; Hub storage is 761,025,363,709 bytes. | Valid rollout twin when paired with the BF16 trainer checkpoint; not a smaller quality proxy. |
| [`imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy`](https://huggingface.co/imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy/tree/cc2b0f160092e9965d67792bc11fb16a57847ee5) | `cc2b0f160092e9965d67792bc11fb16a57847ee5` | Our 8.763B full-width BF16 surgery fixture. | Primary SFT/LoRA/gradient/memory/checkpoint test; no language-quality claim. |
| [`imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy-FP8`](https://huggingface.co/imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy-FP8/tree/5eedf18a056d10b37452528c930487cc48dbd63a) | `5eedf18a056d10b37452528c930487cc48dbd63a` | Mixed E4M3/BF16 twin of the same surgery plan. | Primary SGLang rollout and adapter-hot-sync fixture; no language-quality claim. |
| [`inference-optimization/GLM-5.2-0.8B-A0.8B`](https://huggingface.co/inference-optimization/GLM-5.2-0.8B-A0.8B/tree/210c4dc28de31c9bd84777ded04e40df5174ded2) | `210c4dc28de31c9bd84777ded04e40df5174ded2` | A 6-layer FP32 test model with width 2048, 8 experts/top-2, reduced MLA/DSA ranks, and toy copypasta fine-tuning. | Cheap API and architecture smoke only. Its Russian/Markdown/Han behavior is unrelated to the official model. |
| [`AlayaNeW/GLM-5.2-DSpark`](https://huggingface.co/AlayaNeW/GLM-5.2-DSpark/tree/605ab990a0e196235af77d883947cc354bd60267) | `605ab990a0e196235af77d883947cc354bd60267` | Five-layer `qwen3_dspark` speculative-decoding draft that consumes hidden states from a full GLM-5.2 target. Its card explicitly says it is not a standalone LLM. | Throughput testing only; it cannot produce an independent quality baseline. |
| [`RedHatAI/GLM-5.2-speculator.dspark`](https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark/tree/cc714308fc4ad68b667a6a71bbf2a344c2ef903b) | `cc714308fc4ad68b667a6a71bbf2a344c2ef903b` | Preliminary `DSparkDraftModel` for a full RedHat GLM-5.2 verifier. | Throughput testing only; it cannot produce an independent quality baseline. |

## Proof boundary

The surgery pair has already proved the following on the real GLM-5.2 width
and MLA/DSA/MoE tensor geometry:

- finite BF16 LoRA backward and nonzero updates;
- adapter-only Megatron checkpoint export and HF reload;
- native mixed-FP8 SGLang rollout and adapter hot sync;
- two-step GRPO with finite gradients;
- measured single-A100 CUDA and CPU memory.

Those results establish engineering compatibility, not Russian fluency. The
shortened ten-layer network is deliberately not a chat or benchmark model.

A production quality claim requires all of the following on the same pinned
full checkpoint, clean v4 held-out IDs, and shared pair-runtime contract:

1. Generate the base outputs before training.
2. Train the adapter on train partitions only.
3. Build distinct base and adapter runtime manifests. Their runtime modes and
   complete hashes differ, but their independently derived
   `pair_runtime_contract_sha256` values must be identical.
4. Generate adapter outputs with identical prompts, chat template, decoding
   parameters, seed, and token limits; preserve both generation-output
   manifests.
5. Report Russian semantic scores separately from deterministic structural
   metrics. Never use Cyrillic character ratio as a semantic-quality proxy.
6. Compare Markdown validity and accidental-Han rates with confidence
   intervals and preserve every raw output for audit.
7. Require paired semantic non-inferiority on every non-Russian retention row,
   including legitimate Chinese requests.
8. Reject the adapter if semantic quality regresses even when the two
   formatting constraints improve.

The official base checkpoint is available through several Hub inference
providers, but the conversational provider route cannot pin or attest a Hub
commit. Provider output is therefore only a preflight. The final baseline and
adapter outputs must come from the exact full base revision. They use separate
base/adapter runtime manifests under one matching pair-runtime contract, with
the adapter artifact, trusted shard manifests, local verification receipts,
generation-output manifests, prepared-review manifest, and adjudication
manifest all recorded separately.
