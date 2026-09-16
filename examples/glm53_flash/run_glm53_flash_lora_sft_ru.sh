#!/usr/bin/env bash
# LoRA SFT for GLM-5.3-Flash on Russian reasoning data prepared by
# examples/glm53_flash/prepare_russian_reasoning.py.
#
# Unlike run_glm53_flash_9b_lora_sft_smoke.sh this is meant to train: it saves checkpoints, keeps a
# validation split, and refuses to truncate. Two settings are load-bearing for this model family:
#
#   data.tokenize_full_conversation=true  GLM's template is contextual (it prepends a reasoning
#       effort turn and opens <think> itself), so per-turn tokenization does not reassemble into the
#       canonical chat. Full-conversation tokenization derives the assistant spans by prefix
#       differencing, which is what makes the loss mask cover the answer and nothing else.
#   data.truncation=error  the prepare script already dropped anything longer than the budget, so a
#       truncation here means the data and the budget disagree and the run should stop, not silently
#       train on half a reasoning trace.
#
# Usage (single node):
#   MODEL_PATH=/path/to/checkpoint TRAIN_FILE=/data/train.parquet VAL_FILE=/data/val.parquet \
#   NPROC=8 ./examples/glm53_flash/run_glm53_flash_lora_sft_ru.sh
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model_path=${MODEL_PATH:?Set MODEL_PATH to the GLM-5.3-Flash checkpoint}
train_file=${TRAIN_FILE:?Set TRAIN_FILE to the prepared messages parquet}
val_file=${VAL_FILE:-null}
nproc=${NPROC:-8}
nnodes=${NNODES:-1}
engine=${ENGINE:-fsdp}
rank=${LORA_RANK:-32}
alpha=${LORA_ALPHA:-64}
lr=${LR:-1e-4}
max_length=${MAX_LENGTH:-8192}
train_batch_size=${TRAIN_BATCH_SIZE:-64}
micro_batch_size=${MICRO_BATCH_SIZE:-1}
max_token_len=${MAX_TOKEN_LEN_PER_GPU:-16384}
epochs=${EPOCHS:-1}
steps=${TOTAL_STEPS:--1}
save_freq=${SAVE_FREQ:-200}
test_freq=${TEST_FREQ:-100}
# GLM-5.3-Flash has no Flash-Attention-2 path in its modeling code, and its grouped_mm experts
# reject the unaligned pointers LoRA hands them; transformers reads the private key, so overriding
# "experts_implementation" (as older scripts do) silently does nothing.
attn_impl=${ATTN_IMPL:-eager}
experts_impl=${EXPERTS_IMPL:-eager}
remove_padding=${REMOVE_PADDING:-false}
run_id=${GLM53_RUN_ID:-lora_sft_ru_$(date -u +%Y%m%dT%H%M%SZ)}
output_dir=${OUTPUT_DIR:-"${repo_root}/outputs/glm53_flash/${run_id}"}

if (( rank <= 0 || alpha <= 0 )); then
  echo "LORA_RANK and LORA_ALPHA must be positive" >&2
  exit 2
fi

if [[ ${NNODES:-1} -gt 1 ]]; then
  launcher=(torchrun --nnodes="${nnodes}" --nproc_per_node="${nproc}"
    --node_rank="${NODE_RANK:?Set NODE_RANK for multi-node}"
    --master_addr="${MASTER_ADDR:?Set MASTER_ADDR}" --master_port="${MASTER_PORT:-29500}")
else
  launcher=(torchrun --standalone --nnodes=1 --nproc_per_node="${nproc}")
fi

if [[ ${VERL_USE_UV:-1} == 1 ]]; then
  runner=(uv run --frozen --extra glm "${launcher[@]}")
else
  runner=("${launcher[@]}")
fi

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${output_dir}"

"${runner[@]}" -m verl.trainer.sft_trainer \
  data.train_files="${train_file}" \
  data.val_files="${val_file}" \
  data.train_batch_size="${train_batch_size}" \
  data.micro_batch_size_per_gpu="${micro_batch_size}" \
  data.max_token_len_per_gpu="${max_token_len}" \
  data.use_dynamic_bsz=true \
  data.max_length="${max_length}" \
  data.truncation=error \
  data.tokenize_full_conversation=true \
  data.append_stop_token=true \
  data.messages_key=messages \
  data.num_workers=4 \
  model.path="${model_path}" \
  +model.override_config.attn_implementation="${attn_impl}" \
  +model.override_config._experts_implementation="${experts_impl}" \
  model.use_remove_padding="${remove_padding}" \
  model.enable_gradient_checkpointing=true \
  model.lora_rank="${rank}" \
  model.lora_alpha="${alpha}" \
  model.target_modules=all-linear \
  engine="${engine}" \
  engine.param_offload=true \
  engine.optimizer_offload=true \
  engine.model_dtype=bf16 \
  engine.dtype=bfloat16 \
  engine.use_torch_compile=false \
  engine.use_orig_params=false \
  optim.optimizer_impl=torch.optim \
  optim.optimizer=AdamW \
  optim.lr="${lr}" \
  optim.weight_decay=0.01 \
  optim.lr_warmup_steps_ratio=0.03 \
  optim.lr_scheduler_type=cosine \
  optim.min_lr_ratio=0.1 \
  optim.clip_grad=1.0 \
  checkpoint.save_contents='["model","optimizer","extra"]' \
  checkpoint.save_lora_only=true \
  trainer.total_epochs="${epochs}" \
  trainer.total_training_steps="${steps}" \
  trainer.n_gpus_per_node="${nproc}" \
  trainer.nnodes="${nnodes}" \
  trainer.save_freq="${save_freq}" \
  trainer.test_freq="${test_freq}" \
  trainer.logger='["console"]' \
  trainer.project_name=glm53_flash_ru \
  trainer.experiment_name="${run_id}" \
  trainer.default_local_dir="${output_dir}" \
  "$@"
