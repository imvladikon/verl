#!/usr/bin/env bash
set -euo pipefail

# One-GPU GLM-5.2 surgery-dummy LoRA SFT gate using Megatron Bridge.  It is
# intentionally small, but exercises the same GLM MoE DSA provider and adapter
# checkpoint format used by the RL actor.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

python_bin=${PYTHON_BIN:-python3}
model_path=${MODEL_PATH:-imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy}
train_file=${TRAIN_FILE:?Set TRAIN_FILE to the GLM-5.2 sft.parquet}
run_dir=${RUN_DIR:-${repo_root}/runs/glm52-lora-surgery-sft-megatron}
gpu_id=${GPU_ID:-}
rank=${LORA_RANK:-16}
alpha=${LORA_ALPHA:-32}
steps=${STEPS:-2}
config_only=${CONFIG_ONLY:-0}

source "${script_dir}/stack_env.sh"

if (( rank <= 0 || alpha <= 0 || steps < 2 )); then
  echo "LORA_RANK/LORA_ALPHA must be positive and STEPS must be at least 2" >&2
  exit 2
fi

if [[ "${config_only}" != 1 ]]; then
  : "${gpu_id:?Set GPU_ID to an audited free physical GPU index}"
  gpu_used_mib=$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ ! "${gpu_used_mib}" =~ ^[0-9]+$ ]] || (( gpu_used_mib > 256 )); then
    echo "refusing to use GPU ${gpu_id}: memory.used=${gpu_used_mib:-unknown} MiB" >&2
    exit 3
  fi
fi

mkdir -p "${run_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export MAX_JOBS=${MAX_JOBS:-2}

metrics_file="${run_dir}/gpu.csv"
sampler_pid=
if [[ "${config_only}" != 1 ]]; then
  nvidia-smi -i "${gpu_id}" \
    --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits -lms 1000 >"${metrics_file}" 2>&1 &
  sampler_pid=$!
fi
cleanup_sampler() {
  if [[ -n "${sampler_pid}" ]]; then
    kill "${sampler_pid}" 2>/dev/null || true
    wait "${sampler_pid}" 2>/dev/null || true
  fi
}
trap cleanup_sampler EXIT INT TERM

bridge_targets='[linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj]'
hydra_args=()
if [[ "${config_only}" == 1 ]]; then
  hydra_args+=(--cfg job)
fi

/usr/bin/time -v -o "${run_dir}/time.txt" \
  "${python_bin}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=1 \
  -m verl.trainer.sft_trainer \
  "${hydra_args[@]}" \
  engine=megatron \
  optim=megatron \
  "data.train_files=${train_file}" \
  data.val_files=null \
  data.train_batch_size=1 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=256 \
  data.max_token_len_per_gpu=256 \
  data.use_dynamic_bsz=false \
  data.pad_mode=no_padding \
  data.truncation=error \
  data.messages_key=messages \
  data.enable_thinking_key=enable_thinking \
  data.enable_thinking_default=false \
  data.tokenize_full_conversation=true \
  data.ignore_input_ids_mismatch=false \
  data.num_workers=0 \
  "model.path=${model_path}" \
  model.trust_remote_code=false \
  model.lora.type=lora \
  model.lora.merge=false \
  model.lora.rank="${rank}" \
  model.lora.alpha="${alpha}" \
  model.lora.dropout=0.0 \
  model.lora.dtype=bfloat16 \
  "model.lora.target_modules=${bridge_targets}" \
  'model.lora.exclude_modules=[]' \
  model.lora.adapter_path=null \
  engine.use_mbridge=true \
  engine.vanilla_mbridge=false \
  engine.dtype=bfloat16 \
  engine.tensor_model_parallel_size=1 \
  engine.pipeline_model_parallel_size=1 \
  engine.expert_model_parallel_size=1 \
  engine.expert_tensor_parallel_size=1 \
  engine.context_parallel_size=1 \
  engine.sequence_parallel=false \
  engine.param_offload=false \
  engine.optimizer_offload=false \
  +engine.override_transformer_config.dsa_kernel_backend=none \
  +engine.override_transformer_config.moe_router_dtype=fp32 \
  +engine.override_transformer_config.recompute_granularity=full \
  +engine.override_transformer_config.recompute_method=uniform \
  +engine.override_transformer_config.recompute_num_layers=1 \
  optim.lr=2e-4 \
  optim.weight_decay=0.0 \
  trainer.default_local_dir="${run_dir}" \
  trainer.project_name=glm52-lora-contract \
  trainer.experiment_name=surgery-sft-megatron \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${steps}" \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.max_ckpt_to_keep=2 \
  trainer.resume_mode=disable \
  'trainer.logger=["console"]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  'checkpoint.save_contents=[model,optimizer,extra]' \
  checkpoint.save_lora_only=true \
  "$@"
