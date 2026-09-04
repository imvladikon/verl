# GLM-5.2 full-model quality execution gate

The result is **PENDING** until exact trusted trainer and inference shard
manifests are supplied, every local shard is verified by a full-read receipt,
and a full-model adapter improves the held-out Russian, Markdown, and
accidental-Han targets without a semantic regression under the bound runtime
contract. The 9B surgery results prove the training path, not language quality.

## Current gate

| gate | status | evidence or missing artifact |
|---|---|---|
| Immutable BF16 base | PASS | `zai-org/GLM-5.2@cf457fa734ab149ffef225f80893eb38c6ff5cdc` and locked `config.json` hash |
| Teacher-free quality mixture | SPLIT-ISOLATION PASS, content review pending | `mixture_targeted_wikipedia_v4_2240`: `targeted-template-v4` + `wikipedia-corruption-v3`, 2,240 rows (1,812 train / 244 validation / 184 test), mixture SHA-256 `34f0d92ad9b46f0289f26c7aec8cee1b4bdae76310bceda3a8bb36a71d211442` |
| Full-width surgery LoRA | PASS | finite backward, adapter export/reload, and MLA+`lm_head` ablation on the 9B fixture |
| Tensor/expert sharding gates | PASS | TP2 `80ce91da…958`; EP2 `a6a739c9…e13`; combined TP2xEP2 `dbf6d87a…711` |
| Full topology configs | PASS, runtime pending | W8/EP8, W16/EP16, and W32/EP32 Hydra resolution at TP8; capacity dispositions are analytic, not training passes |
| Full HF checkpoint load contract | PASS, metadata only | exact 282-shard headers; separate 24 MiB expert tensors, 1.773 GiB max source tensor, 4.802 TiB MTP-disabled logical reads (4.871 TiB whole-checkpoint upper bound) |
| Trusted full-checkpoint shards | PENDING | reviewed trainer and inference shard manifests plus successful local full-read receipts |
| Full base held-out predictions | PENDING | base runtime manifest, generation-output manifest, and raw JSONL with prompt and decoding-contract hashes |
| Full MLA-only adapter | PENDING | H200 training checkpoint, adapter verification binding its exact trainer base and shard manifest, adapter runtime manifest, generation-output manifest, and run evidence |
| Full MLA+`lm_head` adapter | PENDING | run only if the MLA-only result does not close the quality gate |
| Paired held-out comparison | PENDING | `compare_quality_outputs.py` must report PASS for all three targets |

The v2 and v3 mixtures are historical invalid split-leak artifacts. Their
training, memory, export, and reload evidence remains systems evidence only;
neither artifact may be used for model selection or held-out evaluation. V4 is
the only clean candidate, but its data audit does not make the production gate
pass without the shard and runtime proof above.

## Sequential experiment order

1. Supply and review the exact trainer and inference shard manifests, then
   complete the local full-read verification for both snapshots.
2. Build and preserve the full-model base runtime manifest and outputs once.
3. Train only the MLA-only adapter on the locked 2,240-row v4 mixture with seed
   52.
4. Build a separate adapter runtime manifest, generate paired outputs, and run
   the blinded review and comparator.
5. Train MLA+`lm_head` only if MLA-only is insufficient. Use the identical
   rows, seed, sequence buckets, explicitly locked optimizer-update budget,
   rank, alpha, and decoding contract.
6. Select an adapter only on validation. Report the untouched test split once.

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
| W8 / TP8 / EP8 / ETP1 | 1 | 1 | 238.749 GiB | candidate at measured 270 GiB; static reject at 141 GiB |
| W16 / TP8 / EP16 / ETP1 | 2 | 1 | 154.374 GiB | envelope reject at 141 GiB |
| W32 / TP8 / EP32 / ETP1 | 4 | 1 | 112.186 GiB | candidate at measured 141 GiB |
| W64 / TP8 / EP32 / ETP1 | 8 | 2 | 112.186 GiB | candidate at measured 141 GiB; twice the expert source reads of W32 |
| W128 / TP8 / EP128 / ETP1 | 16 | 1 | 80.546 GiB | envelope reject at 80 GiB |

The planner adds a separate 8-GiB minimum headroom requirement after the
padded envelope. A `CANDIDATE` is permission only for a guarded first runtime
qualification; it is not evidence that Bridge import, collectives, DSA, or
backward fit the full checkpoint.

## Exact SGLang generation contract

`build_quality_sglang_runtime.py` hashes the real immutable BF16 trainer
snapshot, the exact BF16 or official FP8 inference snapshot, the verified
adapter, a clean SGLang Git revision, and every accepted server argument.
Production manifests must bind both `--trainer-weight-shard-manifest` and
`--inference-weight-shard-manifest`; `--verify-weight-shards` performs the
one-time full read and `--weight-verification-cache-dir` preserves the reusable
receipts. A runtime left in `OFFICIAL-QUALITY-PENDING-SHARD-IDENTITY` is not
quality evidence.
`launch_quality_sglang_server.py` refuses a different checkout,
`PYTHONPATH`, physical GPU list, or unrecorded server option.
`generate_full_quality_outputs_sglang.py` requires complete non-truncated
responses and stamps every row with its variant's runtime-manifest hash.

The trainer launch independently pins Megatron Bridge at `d0c6228a` and scans
all local safetensors headers before starting workers. That revision performs
lazy per-tensor reads, but the read happens before TP/ETP scatter; the recorded
4.802 TiB figure is logical aggregate traffic for the MTP-disabled policy and
deliberately assumes no page cache. Including the unused MTP layer raises the
upper bound to 4.871 TiB. This proves bounded source-tensor memory, not storage
throughput or a successful full-model load.

Base and adapter generation use separate runtime manifests and separate server
modes. Their complete manifest hashes are expected to differ, while their
independently derived `pair_runtime_contract_sha256` values must match. That one
pair contract binds the trainer/inference shard-manifest hashes, SGLang tree,
runtime scripts, environment, decoding-relevant server semantics, and model
artifact identities. Preserve both runtime manifests and both
generation-output manifests. The blinded-review tool validates all four
artifacts and the shared pair contract; the comparator then validates the
prepared and adjudication manifests. Surgery runs require the explicit
nonofficial acknowledgement, remain useful for endpoint and LoRA-load tests,
and are rejected as quality evidence in code.

## Hosted baseline preflight

`generate_full_quality_baseline_hf.py` can cheaply test the live full-model
behavior before reserving the H200 training allocation. It disables thinking
explicitly: otherwise GLM-5.2 can consume a short output budget entirely as
hidden reasoning and return an empty visible completion. The bundled 12-ID
preflight covers four Markdown, four Han-cleanup, and four Russian-editing
contracts.

From the repository root, the script is resumable and requires an exact
dynamic billing acknowledgement:

```bash
GLM52_QUALITY_ROOT=/path/to/staged/glm52/lora/quality
python examples/glm52_lora/generate_full_quality_baseline_hf.py \
  "${GLM52_QUALITY_ROOT}/mixture_targeted_wikipedia_v4_2240/seq256/eval_contracts.jsonl" \
  "${GLM52_QUALITY_ROOT}/mixture_targeted_wikipedia_v4_2240/seq384/eval_contracts.jsonl" \
  "${GLM52_QUALITY_ROOT}/mixture_targeted_wikipedia_v4_2240/seq768/eval_contracts.jsonl" \
  --ids-file /secure/path/reviewed-v4-quality-preflight-ids.txt \
  --output base-preflight.jsonl --max-examples 12 --max-tokens 256 \
  --billing-ack 'max_examples=12,max_tokens=256' \
  --unverified-revision-ack hosted-provider-revision-is-not-hf-pinned
```

The checked-in `quality_preflight_ids.txt` belongs to the retired pre-v4
contracts and must not be reused; create and review a new v4 ID list before
running this diagnostic preflight.

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
silently reused. The partial hosted run is diagnostic evidence only. The v2
mixture was retired after repeated targets were found across splits; the later
exhaustive audit also invalidated v3 for reused targeted templates and
near-duplicate Wikipedia fragments across splits. The original v2 finding is
recorded in
[`split_isolation_incident_2026-09-04.json`](http://vladigur.vla.yp-c.yandex.net:3020/root/tasks/-/blob/main/glm52/lora/quality/split_isolation_incident_2026-09-04.json),
and the v3 failure is recorded in
[`mixture_targeted_wikipedia_v3_2716/split_isolation_audit.json`](http://vladigur.vla.yp-c.yandex.net:3020/root/tasks/-/blob/main/glm52/lora/quality/mixture_targeted_wikipedia_v3_2716/split_isolation_audit.json).

## Config-only status

The historical `run_full_sft_locked_mixture*.sh` launchers fail immediately and
must not be pointed at v4. The current
`run_full_sft_clean_v4_megatron.sh` launcher verifies every staged source/view
artifact, checks the exact 28×64 budget, and delegates to
`run_full_sft_megatron.sh`. A config-only resolution still requires a staged
clean-v4 tree and immutable model config because those identities are part of
the gate:

```bash
CONFIG_ONLY=1 \
GLM52_QUALITY_ROOT=/path/to/staged/glm52/lora/quality \
MODEL_PATH=/path/to/immutable/GLM-5.2 \
examples/glm52_lora/run_full_sft_clean_v4_megatron.sh \
  > resolved-clean-v4-mla-only.yaml
```

The Yandex profile still names historical v3 YT tables. Do not launch that path
for quality training until its executable data locks and published YT manifests
are updated in a separately reviewed code change.

The topology and trainable-count results remain systems evidence: expected
global trainable counts are 106,149,888 for MLA-only and 108,726,272 for
MLA+`lm_head`; at TP8 their local counts are 29,552,640 and 29,874,688. A new
v4 config-only resolution must use the exact clean-v4 training view, explicitly
lock its batch/update budget, and then pass
`examples/glm52_lora/verify_full_sft_config.py` with the matching topology and
file-count expectations before allocation.

## Quality decision

Run `build_blind_quality_review.py` first on paired rows from the exact full
checkpoint. Pass all three v4 `eval_contracts.jsonl` files, both generation
output manifests, and the separate base and adapter runtime manifests. The
runtime manifests must have different modes but the same derived pair-runtime
contract. The tool deterministically hides base/adapter identity with an HMAC key,
requires complete ratings from at least two distinct reviewers, rejects any
changed review item, and adds auditable semantic scores to both prediction
files. Keep the key outside version control; only its SHA-256 commitment is
recorded.

During adjudication pass the prepared-review manifest again, along with the
same contracts, raw outputs, generation-output manifests, and runtime
manifests. Run `compare_quality_outputs.py` on the scored rows with both
`--adjudication-manifest` and `--prepared-review-manifest`. It requires
identical IDs, prompt hashes, request-message hashes, decoding-contract hashes,
and full semantic coverage. A result is accepted only when Russian semantic quality is
non-inferior, both required-Markdown validity and accidental-Han improve, and
the non-Russian retention slice is semantically non-inferior. Missing Russian
or retention semantic scores, a defect not reproduced in the base, or a
confidence interval crossing the decision boundary remains `PENDING`, never `PASS`.
Choose the adapter on validation, then generate and review the untouched test
split once with a separate blinding key.
