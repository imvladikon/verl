#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model_path=${MODEL_PATH:?Set MODEL_PATH to a local tiny GLM-5.3-Flash checkpoint}
steps=${GLM53_GRPO_STEPS:-2}
run_id=${GLM53_RUN_ID:-fsdp_grpo_$(date -u +%Y%m%dT%H%M%SZ)}
output_dir=${OUTPUT_DIR:-"${repo_root}/outputs/glm53_flash/${run_id}"}
data_dir=${DATA_DIR:-"${output_dir}/data"}

if (( steps < 2 )); then
  echo "GLM53_GRPO_STEPS must be at least 2 for the inter-step update gate" >&2
  exit 2
fi
if [[ -e "${output_dir}/global_step_1" ]]; then
  echo "Refusing to reuse an existing checkpoint directory: ${output_dir}" >&2
  exit 2
fi

if [[ ${VERL_USE_UV:-1} == 1 ]]; then
  runner=(uv run --frozen --extra glm53-flash python)
else
  runner=(python3)
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

# TileLang invokes the system NVCC for the production DSA kernels. CUDA 12
# rejects GCC newer than 12, so select an installed compatible host compiler
# before Ray snapshots the worker environment. TILELANG_CXX remains an
# explicit override for clusters whose compiler toolchain lives elsewhere.
if [[ -n ${TILELANG_CXX:-} ]]; then
  export CXX=${TILELANG_CXX}
elif command -v nvcc >/dev/null 2>&1; then
  nvcc_release=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | tail -1)
  current_cxx=${CXX:-$(command -v g++ || true)}
  current_cxx_major=$(
    if [[ -n ${current_cxx} ]]; then
      "${current_cxx}" -dumpversion 2>/dev/null | cut -d. -f1
    fi
  )
  if [[ ${nvcc_release:-0} == 12 && ${current_cxx_major:-0} -gt 12 ]]; then
    compatible_cxx=
    for candidate in g++-12 g++-11; do
      if command -v "${candidate}" >/dev/null 2>&1; then
        compatible_cxx=$(command -v "${candidate}")
        break
      fi
    done
    if [[ -z ${compatible_cxx} ]]; then
      echo "CUDA 12 needs GCC <= 12 for TileLang; install g++-12 or set TILELANG_CXX" >&2
      exit 2
    fi
    export CXX=${compatible_cxx}
  fi
fi

cuda_runtime_lib=$("${runner[@]}" -c '
from pathlib import Path
import site
matches = [p for root in site.getsitepackages() for p in Path(root).glob("nvidia/cu13/lib") if p.is_dir()]
print(matches[0] if matches else "")
')
if [[ -n "${cuda_runtime_lib}" ]]; then
  export LD_LIBRARY_PATH="${cuda_runtime_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

mkdir -p "${output_dir}" "${data_dir}"
"${runner[@]}" "${repo_root}/examples/glm53_flash/prepare_tiny_data.py" \
  --output-dir "${data_dir}"
"${runner[@]}" "${repo_root}/examples/glm53_flash/verify_provenance.py" \
  --profile flash \
  --output "${output_dir}/provenance.json"

"${runner[@]}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="${data_dir}/rl.parquet" \
  data.val_files="${data_dir}/rl.parquet" \
  data.train_batch_size=2 \
  data.gen_batch_size=2 \
  data.max_prompt_length=64 \
  data.max_response_length=4 \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts=false \
  data.truncation=error \
  actor_rollout_ref.model.path="${model_path}" \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  +actor_rollout_ref.model.override_config._experts_implementation=batched_mm \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.model.enable_gradient_checkpointing=false \
  actor_rollout_ref.model.freeze_vision_tower=true \
  actor_rollout_ref.actor.optim.lr=0.001 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.max_model_len=128 \
  actor_rollout_ref.rollout.max_num_batched_tokens=128 \
  actor_rollout_ref.rollout.max_num_seqs=1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.enable_prefix_caching=false \
  actor_rollout_ref.rollout.load_format=dummy \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.ignore_eos=true \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=128 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.max_total_tokens=192 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.max_mamba_cache_size=10 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.kv_cache_dtype=bfloat16 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=dsa \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.dsa_prefill_backend=tilelang \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.dsa_decode_backend=tilelang \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.dsa_topk_backend=torch \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.linear_attn_backend=triton \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_runner_backend=triton \
  reward.custom_reward_function.path="${repo_root}/examples/glm53_flash/reward.py" \
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
  trainer.project_name=glm53_flash \
  trainer.experiment_name="${run_id}" \
  trainer.default_local_dir="${output_dir}" \
  trainer.resume_mode=disable \
  ray_kwargs.ray_init.runtime_env.py_executable=null \
  "$@"

"${runner[@]}" "${repo_root}/examples/glm53_flash/verify_checkpoints.py" rl \
  --base-model "${model_path}" \
  --checkpoint-one "${output_dir}/global_step_1/actor/model_world_size_1_rank_0.pt" \
  --checkpoint-two "${output_dir}/global_step_2/actor/model_world_size_1_rank_0.pt" \
  --output "${output_dir}/checkpoint_verification.json"

echo "GLM-5.3-Flash FSDP/SGLang GRPO lifecycle passed: ${output_dir}"
