#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model_path=${MODEL_PATH:?Set MODEL_PATH to the local 9B surgery checkpoint}
steps=${GLM53_GRPO_STEPS:-2}
run_id=${GLM53_RUN_ID:-fsdp_grpo_9b_$(date -u +%Y%m%dT%H%M%SZ)}
output_dir=${OUTPUT_DIR:-"${repo_root}/outputs/glm53_flash/${run_id}"}
data_dir=${DATA_DIR:-"${output_dir}/data"}
rollout_memory=${GLM53_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
optimizer_impl=${GLM53_OPTIMIZER_IMPL:-torchao.optim}
optimizer_name=${GLM53_OPTIMIZER:-AdamW8bit}
# Ray appends a long session/socket suffix. Keep the default short enough for
# Linux's 107-byte AF_UNIX path limit and unique on a shared host.
ray_tmpdir=${GLM53_RAY_TMPDIR:-"/tmp/g53-ray-${UID}-$$"}

if (( steps < 2 )); then
  echo "GLM53_GRPO_STEPS must be at least 2 to test optimizer-state reuse" >&2
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
export RAY_TMPDIR="${ray_tmpdir}"
unset RAY_ADDRESS
ulimit -n 65535 2>/dev/null || true

mkdir -p "${output_dir}" "${data_dir}" "${ray_tmpdir}"
"${runner[@]}" "${repo_root}/examples/glm53_flash/prepare_tiny_data.py" \
  --output-dir "${data_dir}"
"${runner[@]}" "${repo_root}/examples/glm53_flash/verify_provenance.py" \
  --profile flash \
  --output "${output_dir}/provenance.json"

MODEL_PATH="${model_path}" GLM53_OPTIMIZER_IMPL="${optimizer_impl}" "${runner[@]}" - <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path

config_path = Path(os.environ["MODEL_PATH"]) / "config.json"
config = json.loads(config_path.read_text())
if config.get("model_type") != "glm5_next":
    raise SystemExit(f"Expected model_type=glm5_next in {config_path}")
quantization = config.get("quantization_config") or {}
if quantization.get("dequantize") is not True:
    raise SystemExit(
        "The A100 actor needs quantization_config.dequantize=true in a local "
        "copy of the 9B surgery checkpoint"
    )
if (
    os.environ["GLM53_OPTIMIZER_IMPL"] == "torchao.optim"
    and importlib.metadata.version("torchao") != "0.18.0"
):
    raise SystemExit("This smoke was qualified with torchao==0.18.0")
PY

optimizer_args=(
  actor_rollout_ref.actor.optim.optimizer_impl="${optimizer_impl}"
  actor_rollout_ref.actor.optim.optimizer="${optimizer_name}"
)
if [[ ${optimizer_impl} == torchao.optim ]]; then
  optimizer_args+=(+actor_rollout_ref.actor.optim.override_optimizer_config.bf16_stochastic_round=true)
fi

"${runner[@]}" -m verl.trainer.main_ppo \
  trainer.v1.sampler.sync_refill_failed_groups=true \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="${data_dir}/rl.parquet" \
  data.val_files="${data_dir}/rl.parquet" \
  data.train_batch_size=2 \
  data.gen_batch_size=2 \
  data.max_prompt_length=64 \
  data.max_response_length=16 \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts=false \
  data.truncation=error \
  actor_rollout_ref.model.path="${model_path}" \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  +actor_rollout_ref.model.override_config.experts_implementation=eager \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.freeze_vision_tower=true \
  actor_rollout_ref.actor.optim.lr=0.000001 \
  "${optimizer_args[@]}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=false \
  actor_rollout_ref.rollout.quantization=fp8 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_memory}" \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.max_model_len=128 \
  actor_rollout_ref.rollout.max_num_batched_tokens=128 \
  actor_rollout_ref.rollout.max_num_seqs=1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.enable_prefix_caching=false \
  actor_rollout_ref.rollout.load_format=dummy \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.ignore_eos=true \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=128 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.max_total_tokens=192 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.kv_cache_dtype=bfloat16 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=dsa \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.dsa_prefill_backend=torch \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.dsa_decode_backend=torch \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.dsa_topk_backend=torch \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.linear_attn_backend=triton \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_runner_backend=triton \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_weights_cpu_backup=false \
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
  trainer.save_freq=-1 \
  'trainer.logger=["console"]' \
  trainer.project_name=glm53_flash \
  trainer.experiment_name="${run_id}" \
  trainer.default_local_dir="${output_dir}" \
  trainer.resume_mode=disable \
  ray_kwargs.ray_init.runtime_env.py_executable=null \
  "$@"

echo "GLM-5.3-Flash 9B two-step FSDP/SGLang smoke passed: ${output_dir}"
