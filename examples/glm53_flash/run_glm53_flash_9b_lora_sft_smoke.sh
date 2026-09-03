#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model_path=${MODEL_PATH:?Set MODEL_PATH to the local 9B surgery checkpoint}
train_file=${TRAIN_FILE:?Set TRAIN_FILE to a small messages parquet}
steps=${GLM53_SFT_STEPS:-2}
rank=${LORA_RANK:-4}
alpha=${LORA_ALPHA:-8}
run_id=${GLM53_RUN_ID:-fsdp_lora_sft_9b_$(date -u +%Y%m%dT%H%M%SZ)}
output_dir=${OUTPUT_DIR:-"${repo_root}/outputs/glm53_flash/${run_id}"}

if (( steps < 2 )); then
  echo "GLM53_SFT_STEPS must be at least 2 to test optimizer-state reuse" >&2
  exit 2
fi
if (( rank <= 0 || alpha <= 0 )); then
  echo "LORA_RANK and LORA_ALPHA must be positive" >&2
  exit 2
fi

if [[ ${VERL_USE_UV:-1} == 1 ]]; then
  runner=(uv run --frozen --extra glm53-flash torchrun --standalone --nnodes=1 --nproc_per_node=1)
else
  runner=(torchrun --standalone --nnodes=1 --nproc_per_node=1)
fi

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${output_dir}"

"${runner[@]}" -m verl.trainer.sft_trainer \
  data.train_files="${train_file}" \
  data.val_files=null \
  data.train_batch_size=2 \
  data.micro_batch_size_per_gpu=1 \
  data.max_token_len_per_gpu=128 \
  data.use_dynamic_bsz=false \
  data.max_length=128 \
  data.truncation=right \
  data.num_workers=0 \
  data.ignore_input_ids_mismatch=true \
  model.path="${model_path}" \
  +model.override_config.attn_implementation=eager \
  +model.override_config.experts_implementation=eager \
  model.use_remove_padding=false \
  model.enable_gradient_checkpointing=true \
  model.freeze_vision_tower=true \
  model.lora_rank="${rank}" \
  model.lora_alpha="${alpha}" \
  model.target_modules=all-linear \
  engine=fsdp \
  engine.param_offload=true \
  engine.optimizer_offload=true \
  engine.model_dtype=bf16 \
  engine.dtype=bfloat16 \
  engine.use_orig_params=false \
  engine.use_torch_compile=false \
  optim.optimizer_impl=torch.optim \
  optim.optimizer=AdamW \
  optim.lr=0.0001 \
  checkpoint.save_contents='["model"]' \
  checkpoint.load_contents='["model"]' \
  +checkpoint.save_lora_only=true \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${steps}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.balance_batch=false \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.logger='["console"]' \
  trainer.project_name=glm53_flash \
  trainer.experiment_name="${run_id}" \
  trainer.default_local_dir="${output_dir}" \
  "$@"

echo "GLM-5.3-Flash 9B two-step LoRA SFT smoke passed: ${output_dir}"
