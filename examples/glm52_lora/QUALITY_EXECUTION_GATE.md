# GLM-5.2 full-model quality execution gate

The result is **PENDING** until a full-model adapter improves the held-out
Russian, Markdown, and accidental-Han targets without a semantic regression.
The 9B surgery results prove the training path, not language quality.

## Current gate

| gate | status | evidence or missing artifact |
|---|---|---|
| Immutable BF16 base | PASS | `zai-org/GLM-5.2@cf457fa734ab149ffef225f80893eb38c6ff5cdc` and locked `config.json` hash |
| Teacher-free quality mixture | PASS | `targeted-template-v2`, 2,728 rows, three no-truncation buckets, mixture SHA-256 `9a961a52595df23e8f5c110c780297d9470a5a6c1e36d831346c67254a26318f` |
| Full-width surgery LoRA | PASS | finite backward, adapter export/reload, and MLA+`lm_head` ablation on the 9B fixture |
| Tensor/expert sharding gates | PASS | TP2 `80ce91da…958`; EP2 `a6a739c9…e13`; combined TP2xEP2 `dbf6d87a…711` |
| Full topology configs | PASS, runtime pending | W8/EP8, W16/EP16, and W32/EP32 Hydra resolution at TP8; capacity dispositions are analytic, not training passes |
| Full HF checkpoint load contract | PASS, metadata only | exact 282-shard headers; separate 24 MiB expert tensors, 1.773 GiB max source tensor, 4.802 TiB MTP-disabled logical reads (4.871 TiB whole-checkpoint upper bound) |
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

The launchers intentionally use different topology-specific acknowledgement
strings and run directories, so one profile cannot overwrite or masquerade as
the other. Before checkpoint scanning, a real launch queries every allocated
visible GPU with `nvidia-smi` and requires the selected topology to be a
planner `CANDIDATE`; pool names are not treated as hardware evidence.

## Full topology gate

Use `plan_full_sft_topologies.py` on the immutable 3.7-KiB config before making
an allocation. For the locked seq768 mixture, current calibrated envelopes
are:

| topology | dense DP | expert DP | envelope/GPU | example disposition |
|---|---:|---:|---:|---|
| W8 / TP8 / EP8 / ETP1 | 1 | 1 | 238.990 GiB | candidate at measured 270 GiB; static reject at 141 GiB |
| W16 / TP8 / EP16 / ETP1 | 2 | 1 | 154.615 GiB | envelope reject at 141 GiB |
| W32 / TP8 / EP32 / ETP1 | 4 | 1 | 112.428 GiB | candidate at measured 141 GiB |
| W64 / TP8 / EP32 / ETP1 | 8 | 2 | 112.428 GiB | candidate at measured 141 GiB; twice the expert source reads of W32 |
| W128 / TP8 / EP128 / ETP1 | 16 | 1 | 80.787 GiB | envelope reject at 80 GiB |

The planner adds a separate 8-GiB minimum headroom requirement after the
padded envelope. A `CANDIDATE` is permission only for a guarded first runtime
qualification; it is not evidence that Bridge import, collectives, DSA, or
backward fit the full checkpoint.

## Exact SGLang generation contract

`build_quality_sglang_runtime.py` hashes the real immutable BF16 trainer
snapshot, the exact BF16 or official FP8 inference snapshot, the verified
adapter, a clean SGLang Git revision, and every accepted server argument.
`launch_quality_sglang_server.py` refuses a different checkout,
`PYTHONPATH`, physical GPU list, or unrecorded server option.
`generate_full_quality_outputs_sglang.py` requires complete non-truncated
responses and stamps every row with the common runtime-manifest hash.

The trainer launch independently pins Megatron Bridge at `d0c6228a` and scans
all local safetensors headers before starting workers. That revision performs
lazy per-tensor reads, but the read happens before TP/ETP scatter; the recorded
4.802 TiB figure is logical aggregate traffic for the MTP-disabled policy and
deliberately assumes no page cache. Including the unused MTP layer raises the
upper bound to 4.871 TiB. This proves bounded source-tensor memory, not storage
throughput or a successful full-model load.

Generate base and adapter validation rows against the same live server
instance. Both the blinded-review tool and the comparator verify that common
hash and `quality_claim_allowed=true`. Surgery runs require the explicit
nonofficial acknowledgement, remain useful for endpoint and LoRA-load tests,
and are rejected as quality evidence in code.

## Hosted baseline preflight

`generate_full_quality_baseline_hf.py` can cheaply test the live full-model
behavior before reserving the H200 training allocation. It disables thinking
explicitly: otherwise GLM-5.2 can consume a short output budget entirely as
hidden reasoning and return an empty visible completion. The bundled 12-ID
preflight covers four Markdown, four Han-cleanup, and four Russian-editing
contracts.

The script is resumable and requires an exact dynamic billing acknowledgement:

```bash
python examples/glm52_lora/generate_full_quality_baseline_hf.py \
  /path/to/seq256/eval_contracts.jsonl \
  /path/to/seq384/eval_contracts.jsonl \
  /path/to/seq768/eval_contracts.jsonl \
  --ids-file examples/glm52_lora/quality_preflight_ids.txt \
  --output base-preflight.jsonl --max-examples 12 --max-tokens 256 \
  --billing-ack 'max_examples=12,max_tokens=256' \
  --unverified-revision-ack hosted-provider-revision-is-not-hf-pinned
```

This is only `HOSTED-PREFLIGHT/PROVIDER-REVISION-UNVERIFIED`. The HF
conversational provider route does not accept a Hub commit revision, so it
cannot replace generation from the exact full checkpoint during the final
quality gate.

The first bounded attempt completed 9 of 12 requests before the account's
included inference credit was exhausted. It removed accidental Han in all four
cleanup cases and returned valid Russian, but it also exposed a v1 evaluation
bug: a correct table-only answer was rejected because the hidden contract
required an unrequested heading. `targeted-template-v2` removes that heading
from the target and contract. Resume now verifies the exact prompt, request
messages, and quality contract, so stale v1 rows fail closed instead of being
silently reused. The partial hosted run is diagnostic evidence only.

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

For a non-default topology, pass the same expected dimensions to the config
validator. For example, W32/TP8/EP32 on four eight-GPU nodes uses:

```bash
python examples/glm52_lora/verify_full_sft_config.py resolved-mla-only.yaml \
  --expected-nnodes 4 --expected-gpus-per-node 8 \
  --expected-tp 8 --expected-ep 32 --expected-etp 1 \
  --expected-max-length 768 --expected-steps 33 \
  --expected-train-file-count 3 --expected-val-file-count 3 \
  --expected-lora-profile mla-only
```

## Quality decision

Run `build_blind_quality_review.py` first on paired rows from the exact full
checkpoint and the same hashed SGLang runtime. It deterministically hides
base/adapter identity with an HMAC key,
requires complete ratings from at least two distinct reviewers, rejects any
changed review item, and adds auditable semantic scores to both prediction
files. Keep the key outside version control; only its SHA-256 commitment is
recorded.

Run `compare_quality_outputs.py` on the scored rows with identical IDs, prompt
hashes, request-message hashes, decoding-contract hashes, and full semantic
coverage. A result is accepted only when Russian semantic quality is
non-inferior, both required-Markdown validity and accidental-Han improve, and
the non-Russian retention slice is semantically non-inferior. Missing Russian
or retention semantic scores, a defect not reproduced in the base, or a
confidence interval crossing the decision boundary remains `PENDING`, never `PASS`.
Choose the adapter on validation, then generate and review the untouched test
split once with a separate blinding key.
