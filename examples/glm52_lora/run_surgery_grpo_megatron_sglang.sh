#!/usr/bin/env bash
set -euo pipefail

# Two-step GLM-5.2 LoRA GRPO gate.  The Megatron actor imports a BF16 view of
# the mixed FP8/BF16 surgery checkpoint; SGLang keeps the native FP8 rollout
# representation and receives adapter-only updates after the initial base sync.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

python_bin=${PYTHON_BIN:-python3}
model_path=${MODEL_PATH:-imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy-FP8}
adapter_ckpt=${SFT_ADAPTER_CKPT:?Set SFT_ADAPTER_CKPT to global_step_N/model/dist_ckpt}
train_file=${TRAIN_FILE:?Set TRAIN_FILE to the GLM-5.2 rl.parquet}
run_dir=${RUN_DIR:-${repo_root}/runs/glm52-lora-surgery-grpo-megatron-sglang}
gpu_id=${GPU_ID:-}
rank=${LORA_RANK:-16}
alpha=${LORA_ALPHA:-32}
steps=${STEPS:-2}
rollout_memory=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}
ray_tmpdir=${RAY_TMPDIR:-/tmp/glm52-lora-ray-${UID}-$$}
config_only=${CONFIG_ONLY:-0}

source "${script_dir}/stack_env.sh"

if (( rank <= 0 || alpha <= 0 || steps < 2 )); then
  echo "LORA_RANK/LORA_ALPHA must be positive and STEPS must be at least 2" >&2
  exit 2
fi
if [[ "${config_only}" != 1 && ! -d "${adapter_ckpt}" ]]; then
  echo "adapter checkpoint directory not found: ${adapter_ckpt}" >&2
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

mkdir -p "${run_dir}" "${ray_tmpdir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR="${ray_tmpdir}"
export MAX_JOBS=${MAX_JOBS:-2}
unset RAY_ADDRESS
ulimit -n 65535 2>/dev/null || true

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
sglang_targets='[q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj]'
hydra_args=()
if [[ "${config_only}" == 1 ]]; then
  hydra_args+=(--cfg job)
fi

/usr/bin/time -v -o "${run_dir}/time.txt" \
  "${python_bin}" -m verl.trainer.main_ppo \
  "${hydra_args[@]}" \
  model_engine=megatron \
  trainer.v1.sampler.sync_refill_failed_groups=true \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  "data.train_files=${train_file}" \
  "data.val_files=${train_file}" \
  data.train_batch_size=2 \
  data.gen_batch_size=2 \
  data.max_prompt_length=192 \
  data.max_response_length=32 \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts=false \
  data.truncation=error \
  "actor_rollout_ref.model.path=${model_path}" \
  actor_rollout_ref.model.trust_remote_code=false \
  "actor_rollout_ref.model.target_modules=${sglang_targets}" \
  actor_rollout_ref.model.lora.type=lora \
  actor_rollout_ref.model.lora.merge=false \
  actor_rollout_ref.model.lora.rank="${rank}" \
  actor_rollout_ref.model.lora.alpha="${alpha}" \
  actor_rollout_ref.model.lora.dropout=0.0 \
  actor_rollout_ref.model.lora.dtype=bfloat16 \
  "actor_rollout_ref.model.lora.target_modules=${bridge_targets}" \
  'actor_rollout_ref.model.lora.exclude_modules=[]' \
  "actor_rollout_ref.model.lora.adapter_path=${adapter_ckpt}" \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.megatron.use_mbridge=true \
  actor_rollout_ref.actor.megatron.vanilla_mbridge=false \
  actor_rollout_ref.actor.megatron.dtype=bfloat16 \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
  actor_rollout_ref.actor.megatron.context_parallel_size=1 \
  actor_rollout_ref.actor.megatron.sequence_parallel=false \
  actor_rollout_ref.actor.megatron.param_offload=true \
  actor_rollout_ref.actor.megatron.optimizer_offload=true \
  +actor_rollout_ref.actor.megatron.override_transformer_config.dsa_kernel_backend=none \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.quantization=fp8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_memory}" \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.max_model_len=256 \
  actor_rollout_ref.rollout.max_num_batched_tokens=256 \
  actor_rollout_ref.rollout.max_num_seqs=2 \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.enable_prefix_caching=false \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.ignore_eos=true \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=256 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.max_total_tokens=384 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.kv_cache_dtype=bfloat16 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.dsa_prefill_backend=torch \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.dsa_decode_backend=torch \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.dsa_topk_backend=torch \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_runner_backend=triton \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_weights_cpu_backup=false \
  "reward.custom_reward_function.path=${script_dir}/reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  reward.num_workers=1 \
  transfer_queue.backend.SimpleStorage.num_data_storage_units=1 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${steps}" \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=1 \
  'trainer.logger=["console"]' \
  trainer.project_name=glm52-lora-contract \
  trainer.experiment_name=surgery-grpo-megatron-sglang \
  trainer.default_local_dir="${run_dir}" \
  trainer.resume_mode=disable \
  ray_kwargs.ray_init.runtime_env.py_executable=null \
  "$@"
