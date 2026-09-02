#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

python_bin=${PYTHON_BIN:-python3}
model_path=${MODEL_PATH:?Set MODEL_PATH to a local GLM-5.2 checkpoint}
train_file=${TRAIN_FILE:?Set TRAIN_FILE to sft.parquet}
run_dir=${RUN_DIR:-${repo_root}/runs/glm52-lora-tiny-sft}
rank=${LORA_RANK:-32}
alpha=${LORA_ALPHA:-64}
target_modules='["q_a_proj","q_b_proj","kv_a_proj_with_mqa","kv_b_proj","o_proj","lm_head"]'

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=1 \
  -m verl.trainer.sft_trainer \
  "data.train_files=${train_file}" \
  data.val_files=null \
  data.train_batch_size=1 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=128 \
  data.max_token_len_per_gpu=128 \
  data.use_dynamic_bsz=false \
  data.pad_mode=no_padding \
  data.truncation=error \
  data.messages_key=messages \
  data.enable_thinking_key=enable_thinking \
  data.enable_thinking_default=false \
  data.ignore_input_ids_mismatch=false \
  data.tokenize_full_conversation=true \
  data.num_workers=0 \
  optim.lr=2e-4 \
  optim.weight_decay=0.0 \
  engine=fsdp \
  engine.strategy=fsdp \
  engine.model_dtype=bfloat16 \
  engine.dtype=bfloat16 \
  engine.use_torch_compile=false \
  model.path="${model_path}" \
  +model.override_config.attn_implementation=sdpa \
  model.enable_gradient_checkpointing=false \
  model.use_remove_padding=false \
  model.lora_rank="${rank}" \
  model.lora_alpha="${alpha}" \
  "model.target_modules=${target_modules}" \
  trainer.default_local_dir="${run_dir}" \
  trainer.project_name=glm52-lora-contract \
  trainer.experiment_name=tiny-sft \
  trainer.total_epochs=1 \
  trainer.total_training_steps=2 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.logger='["console"]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  'checkpoint.save_contents=[model,hf_model]' \
  checkpoint.save_lora_only=true
