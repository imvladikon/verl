#!/usr/bin/env bash
set -euo pipefail

# Two-GPU TP2 adapter-only save/reload gate for the GLM-5.2 surgery model.
# CONFIG_ONLY=1 resolves the job without CUDA. A real run requires two distinct
# independently idle GPUs and an explicit operator acknowledgement.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

python_bin=${PYTHON_BIN:-python3}
model_path=${MODEL_PATH:?Set MODEL_PATH to the immutable BF16 surgery snapshot}
train_file=${TRAIN_FILE:?Set TRAIN_FILE to targeted-quality sft_train.parquet}
run_dir=${RUN_DIR:-${repo_root}/runs/glm52-lora-surgery-sft-tp2-gate}
gpu_list=${GPU_IDS:-}
rank=${LORA_RANK:-16}
alpha=${LORA_ALPHA:-32}
steps=${STEPS:-2}
config_only=${CONFIG_ONLY:-0}

expected_model_config_sha256=c15c0d218bf368b6a08e5d15138fca910292353946d7cfcb847be3325ddb53da
expected_model_index_sha256=2c863b67b3ddece85fdfbc94584078328e56b5837d640f70077c481d3fbfb561
expected_model_revision=cc2b0f160092e9965d67792bc11fb16a57847ee5
expected_train_sha256=c2b970b938c171ce4db805d5274a4d8f3771d40307e20f56c7f4fcfd9832fe6c

for integer_setting in rank alpha steps; do
  integer_value=${!integer_setting}
  if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
    echo "${integer_setting} must be a non-negative integer, got: ${integer_value}" >&2
    exit 2
  fi
done
if [[ "${config_only}" != 0 && "${config_only}" != 1 ]]; then
  echo "CONFIG_ONLY must be 0 or 1" >&2
  exit 2
fi

if (( rank != 16 || alpha != 32 || steps != 2 )); then
  echo "TP2 gate is locked to rank 16 / alpha 32 / two steps" >&2
  exit 2
fi

require_sha256() {
  local source_file=$1
  local expected=$2
  local label=$3
  if [[ ! -f "${source_file}" ]]; then
    echo "${label} not found: ${source_file}" >&2
    exit 3
  fi
  local actual
  actual=$(sha256sum "${source_file}" | cut -d' ' -f1)
  if [[ "${actual}" != "${expected}" ]]; then
    echo "${label} SHA-256 mismatch: expected=${expected} actual=${actual}" >&2
    exit 3
  fi
}

require_sha256 "${model_path}/config.json" "${expected_model_config_sha256}" "model config"
require_sha256 \
  "${model_path}/model.safetensors.index.json" \
  "${expected_model_index_sha256}" \
  "model index"
require_sha256 "${train_file}" "${expected_train_sha256}" "train parquet"

model_path=$(realpath -e "${model_path}")
train_file=$(realpath -e "${train_file}")
run_dir=$(realpath -m "${run_dir}")
if [[ "$(basename -- "${model_path}")" != "${expected_model_revision}" ]]; then
  sentinel=${model_path}/.glm52_snapshot_revision
  if [[ ! -f "${sentinel}" ]] || [[ "$(<"${sentinel}")" != "${expected_model_revision}" ]]; then
    echo "model path is not pinned to surgery revision ${expected_model_revision}" >&2
    exit 3
  fi
fi

gpu_ids=()
if [[ "${config_only}" != 1 ]]; then
  if [[ "${TP_GATE_ACK:-}" != GLM52_TP2_MLA_R16 ]]; then
    echo "set TP_GATE_ACK=GLM52_TP2_MLA_R16 after auditing the allocation" >&2
    exit 4
  fi
  IFS=',' read -r -a gpu_ids <<<"${gpu_list}"
  if (( ${#gpu_ids[@]} != 2 )) || [[ "${gpu_ids[0]}" == "${gpu_ids[1]}" ]]; then
    echo "GPU_IDS must contain two distinct physical GPU indices" >&2
    exit 4
  fi
  for gpu_id in "${gpu_ids[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
      echo "invalid physical GPU index: ${gpu_id}" >&2
      exit 4
    fi
    gpu_row=$(nvidia-smi -i "${gpu_id}" \
      --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)
    gpu_used=$(cut -d, -f2 <<<"${gpu_row}" | tr -d ' ')
    gpu_total=$(cut -d, -f3 <<<"${gpu_row}" | tr -d ' ')
    if [[ ! "${gpu_used}" =~ ^[0-9]+$ ]] || (( gpu_used > 256 )); then
      echo "refusing occupied GPU ${gpu_id}: ${gpu_row}" >&2
      exit 4
    fi
    if [[ ! "${gpu_total}" =~ ^[0-9]+$ ]] || (( gpu_total < 80000 )); then
      echo "refusing GPU below the qualified 80-GiB class: ${gpu_row}" >&2
      exit 4
    fi
  done
fi

if [[ "${config_only}" != 1 && -e "${run_dir}" ]]; then
  echo "refusing existing TP2 run directory: ${run_dir}" >&2
  exit 4
fi

source "${script_dir}/stack_env.sh"

mkdir -p "${run_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_list}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export MAX_JOBS=${MAX_JOBS:-2}

targets='[linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj]'
job_args=(
  engine=megatron
  optim=megatron
  "data.train_files=${train_file}"
  data.val_files=null
  data.train_batch_size=1
  data.micro_batch_size_per_gpu=1
  data.max_length=256
  data.max_token_len_per_gpu=256
  data.use_dynamic_bsz=false
  data.pad_mode=no_padding
  data.truncation=error
  data.messages_key=messages
  data.enable_thinking_key=enable_thinking
  data.enable_thinking_default=false
  data.tokenize_full_conversation=true
  data.ignore_input_ids_mismatch=false
  data.num_workers=0
  "model.path=${model_path}"
  model.trust_remote_code=false
  model.use_remove_padding=false
  model.lora.type=lora
  model.lora.merge=false
  "model.lora.rank=${rank}"
  "model.lora.alpha=${alpha}"
  model.lora.dropout=0.0
  model.lora.dtype=bfloat16
  "model.lora.target_modules=${targets}"
  'model.lora.exclude_modules=[]'
  model.lora.adapter_path=null
  engine.use_mbridge=true
  engine.vanilla_mbridge=false
  engine.dtype=bfloat16
  engine.tensor_model_parallel_size=2
  engine.pipeline_model_parallel_size=1
  engine.expert_model_parallel_size=1
  engine.expert_tensor_parallel_size=1
  engine.context_parallel_size=1
  engine.sequence_parallel=true
  engine.param_offload=false
  engine.optimizer_offload=false
  engine.use_distributed_optimizer=true
  engine.pad_bshd_to_minibatch_max=true
  +engine.override_transformer_config.dsa_kernel_backend=none
  +engine.override_transformer_config.moe_router_dtype=fp32
  engine.override_transformer_config.recompute_granularity=null
  engine.override_transformer_config.recompute_method=null
  engine.override_transformer_config.recompute_num_layers=null
  optim.lr=2e-4
  optim.weight_decay=0.0
  "trainer.default_local_dir=${run_dir}"
  trainer.project_name=glm52-lora-contract
  trainer.experiment_name=surgery-sft-tp2-gate
  trainer.total_epochs=1
  trainer.total_training_steps=2
  trainer.save_freq=1
  trainer.test_freq=-1
  trainer.max_ckpt_to_keep=2
  trainer.resume_mode=disable
  trainer.resume_from_path=null
  'trainer.logger=["console"]'
  trainer.nnodes=1
  trainer.n_gpus_per_node=2
  'checkpoint.save_contents=[model,optimizer,extra]'
  checkpoint.save_lora_only=true
)

if [[ "${config_only}" == 1 ]]; then
  exec "${python_bin}" -m verl.trainer.sft_trainer --cfg job "${job_args[@]}" "$@"
fi

CUDA_VISIBLE_DEVICES= "${python_bin}" -m verl.trainer.sft_trainer \
  --cfg job "${job_args[@]}" "$@" >"${run_dir}/resolved.yaml"
"${python_bin}" "${script_dir}/verify_tp2_sft_config.py" \
  "${run_dir}/resolved.yaml" \
  --expected-model-path "${model_path}" \
  --expected-train-file "${train_file}" \
  --expected-run-dir "${run_dir}" \
  >"${run_dir}/config_verification.json"

sampler_pids=()
for gpu_id in "${gpu_ids[@]}"; do
  nvidia-smi -i "${gpu_id}" \
    --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits -lms 1000 >"${run_dir}/gpu_${gpu_id}.csv" 2>&1 &
  sampler_pids+=("$!")
done
cleanup_samplers() {
  local sampler_pid
  for sampler_pid in "${sampler_pids[@]}"; do
    kill "${sampler_pid}" 2>/dev/null || true
    wait "${sampler_pid}" 2>/dev/null || true
  done
  sampler_pids=()
}
trap cleanup_samplers EXIT INT TERM

/usr/bin/time -v -o "${run_dir}/time.txt" \
  "${python_bin}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  -m verl.trainer.sft_trainer \
  "${job_args[@]}" \
  "$@" \
  2>&1 | tee "${run_dir}/run.log"

checkpoint_step2=${run_dir}/global_step_2
resume_dir=${run_dir}/resumed
if [[ ! -d "${checkpoint_step2}" ]]; then
  echo "missing initial TP2 checkpoint: ${checkpoint_step2}" >&2
  exit 5
fi
if [[ -e "${resume_dir}" ]]; then
  echo "refusing existing resume directory: ${resume_dir}" >&2
  exit 5
fi

"${python_bin}" "${script_dir}/verify_surgery_adapter.py" \
  "${checkpoint_step2}" | tee "${run_dir}/adapter_step2_verification.json"

resume_args=(
  "trainer.default_local_dir=${resume_dir}"
  trainer.experiment_name=surgery-sft-tp2-resume-gate
  trainer.total_training_steps=3
  trainer.max_ckpt_to_keep=1
  trainer.resume_mode=resume_path
  "trainer.resume_from_path=${checkpoint_step2}"
)

CUDA_VISIBLE_DEVICES= "${python_bin}" -m verl.trainer.sft_trainer \
  --cfg job "${job_args[@]}" "${resume_args[@]}" "$@" \
  >"${run_dir}/resume_resolved.yaml"
"${python_bin}" "${script_dir}/verify_tp2_sft_config.py" \
  "${run_dir}/resume_resolved.yaml" \
  --expected-model-path "${model_path}" \
  --expected-train-file "${train_file}" \
  --expected-run-dir "${resume_dir}" \
  --expected-steps 3 \
  --expected-resume-from-path "${checkpoint_step2}" \
  >"${run_dir}/resume_config_verification.json"

/usr/bin/time -v -o "${run_dir}/resume.time" \
  "${python_bin}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  -m verl.trainer.sft_trainer \
  "${job_args[@]}" \
  "${resume_args[@]}" \
  "$@" \
  2>&1 | tee "${run_dir}/resume.log"

checkpoint_step3=${resume_dir}/global_step_3
if [[ ! -d "${checkpoint_step3}" ]]; then
  echo "missing resumed TP2 checkpoint: ${checkpoint_step3}" >&2
  exit 5
fi
"${python_bin}" "${script_dir}/verify_surgery_adapter.py" \
  "${checkpoint_step3}" | tee "${run_dir}/adapter_verification.json"
"${python_bin}" "${script_dir}/verify_tp2_resume_run.py" \
  "${run_dir}" >"${run_dir}/runtime_verification.json"

GPU_ID=${gpu_ids[0]} \
PYTHON_BIN="${python_bin}" \
MODEL_PATH="${model_path}" \
ADAPTER_PATH="${checkpoint_step3}/model/huggingface/adapter" \
/usr/bin/time -v -o "${run_dir}/reload.time" \
  "${script_dir}/run_verify_adapter_reload.sh" \
  >"${run_dir}/reload.json" 2>"${run_dir}/reload.stderr"

cleanup_samplers
trap - EXIT INT TERM

(
  cd -- "${run_dir}"
  find global_step_2 -type f -print0 | sort -z | xargs -0 sha256sum \
    >step2_checkpoint.sha256
  find resumed/global_step_3 -type f -print0 | sort -z | xargs -0 sha256sum \
    >step3_checkpoint.sha256
  sha256sum \
    adapter_step2_verification.json \
    adapter_verification.json \
    config_verification.json \
    resume_config_verification.json \
    reload.json \
    reload.stderr \
    reload.time \
    resume.log \
    resume.time \
    resume_resolved.yaml \
    resolved.yaml \
    runtime_verification.json \
    run.log \
    step2_checkpoint.sha256 \
    step3_checkpoint.sha256 \
    time.txt \
    "gpu_${gpu_ids[0]}.csv" \
    "gpu_${gpu_ids[1]}.csv" \
    resumed/global_step_3/model/huggingface/adapter/adapter_model.safetensors \
    >tp2_gate_inputs.sha256
  sha256sum tp2_gate_inputs.sha256 | cut -d' ' -f1 >tp2_gate.sha256
)

printf 'TP2_GATE_PASS sha256=%s\n' "$(<"${run_dir}/tp2_gate.sha256")"
