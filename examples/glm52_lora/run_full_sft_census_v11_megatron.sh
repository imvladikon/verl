#!/usr/bin/env bash
set -euo pipefail

# Exact independently censused 9x64 GLM-5.2 quality view. Only train and
# validation are passed to VERL; test is hash-checked and kept out of training.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${VERL_REPO_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}
quality_root=${GLM52_QUALITY_ROOT:-${script_dir}/quality}
view_root=${CENSUS_V11_VIEW_DIR:-${quality_root}/mixture_targeted_wikipedia_v11_train_576}
model_path=${MODEL_PATH:?Set MODEL_PATH to the immutable full GLM-5.2 snapshot}
lora_profile=${LORA_PROFILE:-mla-only}
python_bin=${PYTHON_BIN:-python3}
run_dir=${RUN_DIR:-${repo_root}/runs/glm52-full-census-v11-${lora_profile}}

case "${lora_profile}" in
  mla-only)
    locked_bridge_targets='[linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj]'
    ;;
  mla-lm-head)
    locked_bridge_targets='[linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj,output_layer]'
    ;;
  *)
    echo "unknown LORA_PROFILE: ${lora_profile}" >&2
    exit 2
    ;;
esac

declare -Ar expected_sha256=(
  [manifest.json]=1662481ddf1175fca7344fd6fea39e881ab3bc700642375316e681e26980c3eb
  [selection.json]=f940e845d962de53017f51f3fb8b874acd3bd7fcb4f9bb803c5a3754b55881ad
  [sft_train.parquet]=4e9b834ec1661762a66902b4ebe93855eb10496641ce3b62b55eba82b9efa244
  [sft_validation.parquet]=e7360419c0c22019e6dc7e73bde4dd481afaa1737c72d58cb838e437d9d02996
  [sft_test.parquet]=e548d8a039711bd64e7302b14c1c86d9e53d2b6c57cb0e49d5e9f27fe0584529
  [train_rows.jsonl]=fa6c257c78d2b2c10f8d2a5d0cf9456a88f6e1e1d4e90189a0d7ec4237938658
  [validation_rows.jsonl]=2bb6f3725ee64f330d88b703324f28c6d5b8a3bb6bb6345bb9ed0a9c8948cb2c
  [test_rows.jsonl]=1545a27b46cdb37e51ce81d28b3558ce6be2584e8da5eeb1030eb59302462a89
)

for relative_file in "${!expected_sha256[@]}"; do
  artifact=${view_root}/${relative_file}
  if [[ ! -f "${artifact}" ]]; then
    echo "census-v11 view artifact not found: ${artifact}" >&2
    exit 3
  fi
  actual=$(sha256sum "${artifact}" | cut -d' ' -f1)
  if [[ "${actual}" != "${expected_sha256[${relative_file}]}" ]]; then
    echo "census-v11 view SHA-256 mismatch: ${relative_file} expected=${expected_sha256[${relative_file}]} actual=${actual}" >&2
    exit 3
  fi
done

mkdir -p "${run_dir}"
token_audit=${run_dir}/census-v11-token-audit.json
"${python_bin}" "${script_dir}/audit_quality_tokens.py" \
  "${view_root}/train_rows.jsonl" \
  "${model_path}" \
  --output "${token_audit}" >/dev/null
"${python_bin}" "${script_dir}/verify_census_v11_token_audit.py" \
  "${token_audit}" >"${run_dir}/census-v11-token-audit-verification.json"

export MODEL_PATH="${model_path}"
export TRAIN_FILE="${view_root}/sft_train.parquet"
export VAL_FILE="${view_root}/sft_validation.parquet"
export RUN_DIR="${run_dir}"
export EXPECTED_TRAIN_SHA256="${expected_sha256[sft_train.parquet]}"
export EXPECTED_VAL_SHA256="${expected_sha256[sft_validation.parquet]}"
export QUALIFICATION_PROFILE=census-v11-quality-576
export STEPS=9
export GLOBAL_BATCH_SIZE=64
export MAX_LENGTH=576
export REQUIRED_MAX_TOKENS=548
export NNODES=4
export GPUS_PER_NODE=8
export TP_SIZE=8
export EP_SIZE=32
export ETP_SIZE=1
export PP_SIZE=1
export CP_SIZE=1
export SEED=52

exec "${script_dir}/run_full_sft_megatron.sh" \
  "$@" \
  "data.train_files=${view_root}/sft_train.parquet" \
  "data.val_files=${view_root}/sft_validation.parquet" \
  data.train_batch_size=64 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=576 \
  data.max_token_len_per_gpu=576 \
  data.truncation=error \
  data.train_max_samples=576 \
  data.val_max_samples=160 \
  model.lora.type=lora \
  model.lora.merge=false \
  model.lora.rank=16 \
  model.lora.alpha=32 \
  model.lora.dropout=0.0 \
  model.lora.dtype=bfloat16 \
  "model.lora.target_modules=${locked_bridge_targets}" \
  engine.tensor_model_parallel_size=8 \
  engine.pipeline_model_parallel_size=1 \
  engine.expert_model_parallel_size=32 \
  engine.expert_tensor_parallel_size=1 \
  engine.context_parallel_size=1 \
  engine.seed=52 \
  engine.sequence_parallel=true \
  engine.pad_bshd_to_minibatch_max=false \
  engine.override_transformer_config.dsa_kernel_backend=none \
  engine.override_transformer_config.moe_router_dtype=fp32 \
  engine.override_transformer_config.recompute_granularity=null \
  engine.override_transformer_config.recompute_method=null \
  engine.override_transformer_config.recompute_num_layers=null \
  optim.lr=1e-4 \
  optim.weight_decay=0.0 \
  trainer.seed=52 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=9 \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  trainer.save_freq=3 \
  trainer.test_freq=9 \
  trainer.max_ckpt_to_keep=3 \
  "trainer.experiment_name=full-sft-census-v11-${lora_profile}-r16"
