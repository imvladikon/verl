#!/usr/bin/env bash
set -euo pipefail

# Exact 28x64 clean-v4 formatting/script-repair view. This launcher does not
# claim broad Russian-language quality. It never passes the untouched test
# parquet to VERL and never wraps, duplicates, or resamples the 1,792 rows.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${VERL_REPO_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}
quality_root=${GLM52_QUALITY_ROOT:-${script_dir}/quality}
source_root=${CLEAN_V4_SOURCE_DIR:-${quality_root}/mixture_targeted_wikipedia_v4_2240}
view_root=${CLEAN_V4_VIEW_DIR:-${quality_root}/mixture_targeted_wikipedia_v4_train_1792}
model_path=${MODEL_PATH:?Set MODEL_PATH to the immutable full GLM-5.2 snapshot}
lora_profile=${LORA_PROFILE:-mla-only}
python_bin=${PYTHON_BIN:-python3}
run_dir=${RUN_DIR:-${repo_root}/runs/glm52-full-clean-v4-format-script-${lora_profile}}

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
  [manifest.json]=389e9574d42b234419f1fb9f4b9ed8c2771aaba40800d84e12e22ae019bca69c
  [omitted_train_ids.json]=044dead88e78d5c58443639905fad6a1c274be9701de037889f377a177bddfd8
  [sft_train.parquet]=0f89b1a2b6de76231b8ba419579cbb1bb0355a823048ffe1b1b8a1bdebe47dda
  [sft_validation.parquet]=d2004cc631d867bcb59eeceb5dcdc8556a86c319eae2b6067da9194d56a59aed
  [sft_test.parquet]=4fe3e5c4f5862a74418d91221be292f5357daf4e03f2f3a395e0a89332946394
  [train_rows.jsonl]=3131c44ae33051ee8a8fc1ae91e9c8aa94a2750fbe89e0307bb3bd3ff46a262e
  [validation_rows.jsonl]=9b257a52e6271e31089cb057b938e6a415828e99efe97c836ae87d12a7f7bc2c
  [test_rows.jsonl]=148a76ce000b7fda4205859bfa64f9c7e05958367d03064b7de1eb46963b3e96
)

for relative_file in "${!expected_sha256[@]}"; do
  artifact=${view_root}/${relative_file}
  if [[ ! -f "${artifact}" ]]; then
    echo "clean-v4 view artifact not found: ${artifact}" >&2
    exit 3
  fi
  actual=$(sha256sum "${artifact}" | cut -d' ' -f1)
  if [[ "${actual}" != "${expected_sha256[${relative_file}]}" ]]; then
    echo "clean-v4 view SHA-256 mismatch: ${relative_file} expected=${expected_sha256[${relative_file}]} actual=${actual}" >&2
    exit 3
  fi
done

"${python_bin}" "${script_dir}/build_clean_v4_training_view.py" \
  check "${source_root}" "${view_root}" >/dev/null

export MODEL_PATH="${model_path}"
export TRAIN_FILE="${view_root}/sft_train.parquet"
export VAL_FILE="${view_root}/sft_validation.parquet"
export RUN_DIR="${run_dir}"
export EXPECTED_TRAIN_SHA256="${expected_sha256[sft_train.parquet]}"
export EXPECTED_VAL_SHA256="${expected_sha256[sft_validation.parquet]}"
export QUALIFICATION_PROFILE=clean-v4-format-script-1792
export STEPS=28
export GLOBAL_BATCH_SIZE=64
export MAX_LENGTH=768
export REQUIRED_MAX_TOKENS=706
export NNODES=4
export GPUS_PER_NODE=8
export TP_SIZE=8
export EP_SIZE=32
export ETP_SIZE=1
export PP_SIZE=1
export CP_SIZE=1
export SEED=52

# With a one-example micro-batch, preserve that example's actual TP-aligned
# sequence length. The default BSHD mini-batch-max mode would make all 16 local
# micro-batches pay for the longest sample on that DP rank.
exec "${script_dir}/run_full_sft_megatron.sh" \
  "$@" \
  "data.train_files=${view_root}/sft_train.parquet" \
  "data.val_files=${view_root}/sft_validation.parquet" \
  data.train_batch_size=64 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=768 \
  data.max_token_len_per_gpu=768 \
  data.truncation=error \
  data.train_max_samples=1792 \
  data.val_max_samples=244 \
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
  trainer.total_training_steps=28 \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  trainer.save_freq=7 \
  trainer.test_freq=28 \
  trainer.max_ckpt_to_keep=4 \
  "trainer.experiment_name=full-sft-clean-v4-format-script-${lora_profile}-r16"
