#!/usr/bin/env bash
set -euo pipefail

# Full GLM-5.2 quality-SFT qualification profile. CONFIG_ONLY=1 resolves the
# Hydra job without touching a GPU. A real launch is deliberately blocked
# until the immutable full checkpoint, 64 exclusive H200s, and a prior TP2
# adapter checkpoint/reload gate are all supplied.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${VERL_REPO_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}

python_bin=${PYTHON_BIN:-python3}
model_path=${MODEL_PATH:?Set MODEL_PATH to an immutable local GLM-5.2 snapshot}
train_file=${TRAIN_FILE:?Set TRAIN_FILE to the targeted-quality train parquet}
val_file=${VAL_FILE:?Set VAL_FILE to the targeted-quality validation parquet}
run_dir=${RUN_DIR:-${repo_root}/runs/glm52-full-quality-sft-megatron}
rank=${LORA_RANK:-16}
alpha=${LORA_ALPHA:-32}
steps=${STEPS:-8}
max_length=${MAX_LENGTH:-256}
required_max_tokens=${REQUIRED_MAX_TOKENS:-187}
qualification_profile=${QUALIFICATION_PROFILE:-bounded}
nnodes=${NNODES:-8}
gpus_per_node=${GPUS_PER_NODE:-8}
node_rank=${NODE_RANK:-0}
master_addr=${MASTER_ADDR:-}
master_port=${MASTER_PORT:-29500}
config_only=${CONFIG_ONLY:-0}

expected_model_revision=cf457fa734ab149ffef225f80893eb38c6ff5cdc
expected_config_sha256=185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a
expected_train_sha256=${EXPECTED_TRAIN_SHA256:-c2b970b938c171ce4db805d5274a4d8f3771d40307e20f56c7f4fcfd9832fe6c}
expected_val_sha256=${EXPECTED_VAL_SHA256:-df60c803f1988843bef46c8438084810afd61b6dcf278b371beaf1b3f1212c87}

for integer_setting in rank alpha steps max_length required_max_tokens nnodes gpus_per_node node_rank master_port; do
  integer_value=${!integer_setting}
  if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
    echo "${integer_setting} must be a non-negative integer, got: ${integer_value}" >&2
    exit 2
  fi
done

if (( rank != 16 || alpha != 32 )); then
  echo "full-model qualification is locked to rank 16 / alpha 32" >&2
  exit 2
fi
case "${qualification_profile}" in
  bounded)
    if (( steps < 2 || steps > 8 )); then
      echo "STEPS must stay in the bounded qualification range [2,8]" >&2
      exit 2
    fi
    ;;
  locked-quality-mixture-2728)
    if (( steps != 33 || max_length != 768 || required_max_tokens != 706 )); then
      echo "locked-quality-mixture-2728 requires STEPS=33, MAX_LENGTH=768, REQUIRED_MAX_TOKENS=706" >&2
      exit 2
    fi
    ;;
  *)
    echo "unknown QUALIFICATION_PROFILE: ${qualification_profile}" >&2
    exit 2
    ;;
esac
if (( max_length < required_max_tokens )); then
  echo "MAX_LENGTH=${max_length} truncates the audited ${required_max_tokens}-token example" >&2
  exit 2
fi
if (( nnodes != 8 || gpus_per_node != 8 || nnodes * gpus_per_node != 64 )); then
  echo "profile is locked to the source-qualified 8 nodes x 8 GPUs" >&2
  exit 2
fi
if (( node_rank < 0 || node_rank >= nnodes )); then
  echo "NODE_RANK must be in [0,$((nnodes - 1))]" >&2
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

require_sha256 "${model_path}/config.json" "${expected_config_sha256}" "model config"
require_sha256 "${train_file}" "${expected_train_sha256}" "train parquet"
require_sha256 "${val_file}" "${expected_val_sha256}" "validation parquet"

if [[ "${config_only}" != 1 ]]; then
  if [[ "${FULL_MODEL_ACK:-}" != GLM52_64H200_MLA_R16 ]]; then
    echo "set FULL_MODEL_ACK=GLM52_64H200_MLA_R16 after auditing the allocation" >&2
    exit 4
  fi
  if [[ ! "${TP_ADAPTER_GATE_SHA:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "set TP_ADAPTER_GATE_SHA to the 64-hex result of the passing TP2 save/reload gate" >&2
    exit 4
  fi
  if [[ ! -f "${model_path}/model.safetensors.index.json" ]]; then
    echo "full checkpoint index is missing: ${model_path}/model.safetensors.index.json" >&2
    exit 4
  fi
  if [[ ! -f "${model_path}/.glm52_snapshot_revision" ]] ||
     [[ "$(<"${model_path}/.glm52_snapshot_revision")" != "${expected_model_revision}" ]]; then
    echo "missing or wrong immutable snapshot revision sentinel" >&2
    exit 4
  fi
  : "${master_addr:?Set MASTER_ADDR for the 8-node torchrun job}"

  mapfile -t gpu_rows < <(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits)
  if (( ${#gpu_rows[@]} != gpus_per_node )); then
    echo "expected ${gpus_per_node} visible GPUs, found ${#gpu_rows[@]}" >&2
    exit 4
  fi
  for gpu_row in "${gpu_rows[@]}"; do
    gpu_name=${gpu_row%,*}
    gpu_used=${gpu_row##*,}
    gpu_used=${gpu_used// /}
    if [[ "${gpu_name}" != *H200* ]]; then
      echo "refusing non-H200 device: ${gpu_name}" >&2
      exit 4
    fi
    if [[ ! "${gpu_used}" =~ ^[0-9]+$ ]] || (( gpu_used > 256 )); then
      echo "refusing occupied GPU before launch: ${gpu_row}" >&2
      exit 4
    fi
  done
fi

source "${repo_root}/examples/glm52_lora/stack_env.sh"

mkdir -p "${run_dir}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export MAX_JOBS=${MAX_JOBS:-4}

bridge_targets='[linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj]'
job_args=(
  engine=megatron
  optim=megatron
  "data.train_files=${train_file}"
  "data.val_files=${val_file}"
  data.train_batch_size=64
  data.micro_batch_size_per_gpu=1
  "data.max_length=${max_length}"
  "data.max_token_len_per_gpu=${max_length}"
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
  "model.lora.target_modules=${bridge_targets}"
  'model.lora.exclude_modules=[]'
  model.lora.adapter_path=null
  engine.use_mbridge=true
  engine.vanilla_mbridge=false
  engine.dtype=bfloat16
  engine.tensor_model_parallel_size=8
  engine.pipeline_model_parallel_size=1
  engine.expert_model_parallel_size=32
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
  optim.lr=1e-4
  optim.weight_decay=0.0
  "trainer.default_local_dir=${run_dir}"
  trainer.project_name=glm52-quality
  trainer.experiment_name=full-sft-mla-r16
  trainer.total_epochs=1
  "trainer.total_training_steps=${steps}"
  trainer.save_freq=4
  trainer.test_freq=2
  trainer.max_ckpt_to_keep=2
  trainer.resume_mode=disable
  'trainer.logger=["console"]'
  "trainer.nnodes=${nnodes}"
  "trainer.n_gpus_per_node=${gpus_per_node}"
  'checkpoint.save_contents=[model,optimizer,extra]'
  checkpoint.save_lora_only=true
)

if [[ "${config_only}" == 1 ]]; then
  exec "${python_bin}" -m verl.trainer.sft_trainer --cfg job "${job_args[@]}" "$@"
fi

exec "${python_bin}" -m torch.distributed.run \
  --nnodes="${nnodes}" \
  --nproc-per-node="${gpus_per_node}" \
  --node-rank="${node_rank}" \
  --master-addr="${master_addr}" \
  --master-port="${master_port}" \
  -m verl.trainer.sft_trainer \
  "${job_args[@]}" \
  "$@"
