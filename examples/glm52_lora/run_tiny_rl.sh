#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

python_bin=${PYTHON_BIN:-python3}
model_path=${MODEL_PATH:?Set MODEL_PATH to a local GLM-5.2 checkpoint}
train_file=${TRAIN_FILE:?Set TRAIN_FILE to rl.parquet}
run_dir=${RUN_DIR:-${repo_root}/runs/glm52-lora-tiny-rl}
rank=${LORA_RANK:-32}
alpha=${LORA_ALPHA:-64}
steps=${STEPS:-2}
rollout_memory=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}
ray_tmpdir=${RAY_TMPDIR:-/tmp/g52-lora-ray-${UID}-$$}
target_modules='["q_a_proj","q_b_proj","kv_a_proj_with_mqa","kv_b_proj","o_proj","lm_head"]'

if (( steps < 2 )); then
  echo "STEPS must be at least 2 to verify adapter update reuse" >&2
  exit 2
fi

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR="${ray_tmpdir}"
export MAX_JOBS="${MAX_JOBS:-2}"
python_dir=$(cd -- "$(dirname -- "${python_bin}")" && pwd)
export PATH="${python_dir}:${PATH}"

# SGLang starts fresh Python subprocesses.  CUDA wheels keep libcudart under
# site-packages/nvidia/cu*/lib, which is not necessarily visible to the dynamic
# loader in those subprocesses even though torch.cuda works in the parent.
cuda_runtime_lib=$("${python_bin}" - <<'PY'
from pathlib import Path
import site

for site_dir in site.getsitepackages():
    for candidate in sorted(Path(site_dir).glob("nvidia/cu*/lib"), reverse=True):
        if any(candidate.glob("libcudart.so*")):
            print(candidate)
            raise SystemExit
PY
)
if [[ -n "${cuda_runtime_lib}" ]]; then
  cuda_root=$(dirname -- "${cuda_runtime_lib}")
  export LD_LIBRARY_PATH="${cuda_runtime_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  if [[ -x "${cuda_root}/bin/nvcc" ]]; then
    export CUDA_HOME="${cuda_root}"
    export PATH="${cuda_root}/bin:${PATH}"
  fi
fi

unset RAY_ADDRESS
ulimit -n 65535 2>/dev/null || true
mkdir -p "${run_dir}" "${ray_tmpdir}"

"${python_bin}" -m verl.trainer.main_ppo \
  trainer.v1.sampler.sync_refill_failed_groups=true \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="${train_file}" \
  data.val_files="${train_file}" \
  data.train_batch_size=2 \
  data.gen_batch_size=2 \
  data.max_prompt_length=128 \
  data.max_response_length=16 \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts=false \
  data.truncation=error \
  actor_rollout_ref.model.path="${model_path}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  +actor_rollout_ref.model.override_config.experts_implementation=eager \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.lora_rank="${rank}" \
  actor_rollout_ref.model.lora_alpha="${alpha}" \
  "actor_rollout_ref.model.target_modules=${target_modules}" \
  actor_rollout_ref.model.lora.merge=false \
  actor_rollout_ref.actor.optim.lr=2e-4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_memory}" \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.max_model_len=192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=192 \
  actor_rollout_ref.rollout.max_num_seqs=2 \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.enable_prefix_caching=false \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.ignore_eos=true \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=192 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.max_total_tokens=256 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.kv_cache_dtype=bfloat16 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.dsa_prefill_backend=torch \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.dsa_decode_backend=torch \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.dsa_topk_backend=torch \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_runner_backend=triton \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_weights_cpu_backup=false \
  reward.custom_reward_function.path="${script_dir}/reward.py" \
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
  trainer.save_freq=-1 \
  'trainer.logger=["console"]' \
  trainer.project_name=glm52-lora-contract \
  trainer.experiment_name=tiny-rl \
  trainer.default_local_dir="${run_dir}" \
  trainer.resume_mode=disable \
  ray_kwargs.ray_init.runtime_env.py_executable=null
