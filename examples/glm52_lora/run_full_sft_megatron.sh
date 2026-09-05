#!/usr/bin/env bash
set -euo pipefail

# Full GLM-5.2 quality-SFT qualification profile. CONFIG_ONLY=1 resolves the
# Hydra job without touching a GPU. A real launch is deliberately blocked
# until the immutable full checkpoint, a topology that passes the measured
# memory envelope, and the exact validated sharding-gate evidence are supplied.

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
checkpoint_profile=${CHECKPOINT_PROFILE:-official-bf16}
lora_profile=${LORA_PROFILE:-mla-only}
seed=${SEED:-52}
nnodes=${NNODES:-8}
gpus_per_node=${GPUS_PER_NODE:-8}
tp_size=${TP_SIZE:-8}
ep_size=${EP_SIZE:-32}
etp_size=${ETP_SIZE:-1}
pp_size=${PP_SIZE:-1}
cp_size=${CP_SIZE:-1}
global_batch_size=${GLOBAL_BATCH_SIZE:-64}
minimum_additional_headroom_gib=${MINIMUM_ADDITIONAL_HEADROOM_GIB:-8}
node_rank=${NODE_RANK:-0}
master_addr=${MASTER_ADDR:-}
master_port=${MASTER_PORT:-29500}
config_only=${CONFIG_ONLY:-0}

expected_bridge_revision=d0c6228a2a832f566dd44a3a179b3136613c11b7
expected_bridge_fp8_patch_sha256=d5764f406994684392cb78bc2977b6ca90a30680c448022742023d9c1298c590
expected_tp_adapter_gate_sha256=80ce91da59c5615618b03c14fb74163374c7bb8e529c699ab0a661cfcd0ee958
expected_ep_routing_gate_sha256=a6a739c9e8a8031e89506da1f582b0255b5513823d5ace17b4fe5f723aa0ee13
expected_tp_ep_gate_sha256=dbf6d87a6ffdb2065a5a6bb066558d92a07aff8f63e7a0192ff257da2ebca711
expected_train_sha256=${EXPECTED_TRAIN_SHA256:-c2b970b938c171ce4db805d5274a4d8f3771d40307e20f56c7f4fcfd9832fe6c}
expected_val_sha256=${EXPECTED_VAL_SHA256:-df60c803f1988843bef46c8438084810afd61b6dcf278b371beaf1b3f1212c87}

case "${checkpoint_profile}" in
  official-bf16)
    expected_model_revision=cf457fa734ab149ffef225f80893eb38c6ff5cdc
    expected_config_sha256=185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a
    expected_index_sha256=5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e
    profile_ack_suffix=
    official_audit_profile=bf16
    ;;
  official-fp8-dequant)
    expected_config_sha256=d1539d36be7546a1d827fe9cf74c55874695652efb6a5aaa3e60cde1c76ba819
    expected_index_sha256=e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf
    expected_model_source_identity='glm52-official-fp8-d1539d36-e0fe7f28'
    profile_ack_suffix=_FP8_DEQUANT
    official_audit_profile=fp8-dequant
    ;;
  *)
    echo "unknown CHECKPOINT_PROFILE: ${checkpoint_profile}" >&2
    exit 2
    ;;
esac

for integer_setting in rank alpha steps max_length required_max_tokens nnodes gpus_per_node tp_size ep_size etp_size pp_size cp_size global_batch_size node_rank master_port seed; do
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

world_size=$((nnodes * gpus_per_node))
if (( world_size % (tp_size * pp_size * cp_size) != 0 )); then
  echo "world size ${world_size} is not divisible by TP*PP*CP" >&2
  exit 2
fi
dense_dp=$((world_size / (tp_size * pp_size * cp_size)))
if (( global_batch_size % dense_dp != 0 )); then
  echo "GLOBAL_BATCH_SIZE=${global_batch_size} is not divisible by dense DP=${dense_dp}" >&2
  exit 2
fi
if (( world_size % (etp_size * ep_size * pp_size) != 0 )); then
  echo "world size ${world_size} is not divisible by ETP*EP*PP" >&2
  exit 2
fi
if (( 256 % ep_size != 0 )); then
  echo "256 routed experts are not divisible by EP=${ep_size}" >&2
  exit 2
fi
if (( gpus_per_node != 8 || tp_size != 8 || etp_size != 1 || pp_size != 1 || cp_size != 1 )); then
  echo "qualified full-model family requires 8 GPUs/node, TP8, ETP1, PP1, and CP1" >&2
  exit 2
fi
if (( ep_size != 8 && ep_size != 16 && ep_size != 32 && ep_size != 128 )); then
  echo "qualified full-model family requires EP in {8,16,32,128}" >&2
  exit 2
fi
if (( tp_size > 1 )); then
  sequence_parallel=true
else
  sequence_parallel=false
fi

case "${lora_profile}" in
  mla-only)
    planner_profile_args=()
    bridge_targets='[linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj]'
    experiment_name=full-sft-mla-r16
    required_ack=GLM52_FULL_W${world_size}_TP${tp_size}_EP${ep_size}_MLA_R16${profile_ack_suffix}
    ;;
  mla-lm-head)
    planner_profile_args=(--include-output-layer)
    bridge_targets='[linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj,output_layer]'
    experiment_name=full-sft-mla-lm-head-r16
    required_ack=GLM52_FULL_W${world_size}_TP${tp_size}_EP${ep_size}_MLA_LM_HEAD_R16${profile_ack_suffix}
    ;;
  *)
    echo "unknown LORA_PROFILE: ${lora_profile}" >&2
    exit 2
    ;;
esac
case "${qualification_profile}" in
  bounded)
    if (( steps < 2 || steps > 8 )); then
      echo "STEPS must stay in the bounded qualification range [2,8]" >&2
      exit 2
    fi
    ;;
  locked-quality-mixture-v2-2728)
    if (( steps != 33 || max_length != 768 || required_max_tokens != 706 )); then
      echo "locked-quality-mixture-v2-2728 requires STEPS=33, MAX_LENGTH=768, REQUIRED_MAX_TOKENS=706" >&2
      exit 2
    fi
    ;;
  clean-v4-format-script-1792)
    if (( steps != 28 || global_batch_size != 64 || max_length != 768 || required_max_tokens != 706 )); then
      echo "clean-v4-format-script-1792 requires STEPS=28, GLOBAL_BATCH_SIZE=64, MAX_LENGTH=768, REQUIRED_MAX_TOKENS=706" >&2
      exit 2
    fi
    ;;
  census-v11-quality-576)
    if (( steps != 9 || global_batch_size != 64 || max_length != 576 || required_max_tokens != 548 )); then
      echo "census-v11-quality-576 requires STEPS=9, GLOBAL_BATCH_SIZE=64, MAX_LENGTH=576, REQUIRED_MAX_TOKENS=548" >&2
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
  if [[ "${FULL_MODEL_ACK:-}" != "${required_ack}" ]]; then
    echo "set FULL_MODEL_ACK=${required_ack} after auditing the allocation" >&2
    exit 4
  fi
  if [[ "${TP_ADAPTER_GATE_SHA:-}" != "${expected_tp_adapter_gate_sha256}" ]]; then
    echo "TP_ADAPTER_GATE_SHA must match the validated TP2 evidence root" >&2
    exit 4
  fi
  if [[ "${EP_ROUTING_GATE_SHA:-}" != "${expected_ep_routing_gate_sha256}" ]]; then
    echo "EP_ROUTING_GATE_SHA must match the validated EP2 evidence root" >&2
    exit 4
  fi
  if [[ "${TP_EP_GATE_SHA:-}" != "${expected_tp_ep_gate_sha256}" ]]; then
    echo "TP_EP_GATE_SHA must match the validated combined TP2xEP2 evidence root" >&2
    exit 4
  fi
  if [[ ! -f "${model_path}/model.safetensors.index.json" ]]; then
    echo "full checkpoint index is missing: ${model_path}/model.safetensors.index.json" >&2
    exit 4
  fi
  if [[ "${checkpoint_profile}" == official-bf16 ]]; then
    if [[ ! -f "${model_path}/.glm52_snapshot_revision" ]] ||
       [[ "$(<"${model_path}/.glm52_snapshot_revision")" != "${expected_model_revision}" ]]; then
      echo "missing or wrong immutable snapshot revision sentinel" >&2
      exit 4
    fi
  elif [[ "${MODEL_SOURCE_IDENTITY:-}" != "${expected_model_source_identity}" ]]; then
    echo "set MODEL_SOURCE_IDENTITY=${expected_model_source_identity} for the audited YT checkpoint" >&2
    exit 4
  fi
  : "${master_addr:?Set MASTER_ADDR for the multinode torchrun job}"

  mapfile -t gpu_rows < <(
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits
  )
  if (( ${#gpu_rows[@]} != gpus_per_node )); then
    echo "expected ${gpus_per_node} visible GPUs, found ${#gpu_rows[@]}" >&2
    exit 4
  fi
  min_gpu_total_mib=0
  for gpu_row in "${gpu_rows[@]}"; do
    IFS=, read -r gpu_name gpu_used gpu_total <<<"${gpu_row}"
    gpu_used=${gpu_used// /}
    gpu_total=${gpu_total// /}
    if [[ ! "${gpu_used}" =~ ^[0-9]+$ ]] || (( gpu_used > 256 )); then
      echo "refusing occupied GPU before launch: ${gpu_row}" >&2
      exit 4
    fi
    if [[ ! "${gpu_total}" =~ ^[0-9]+$ ]] || (( gpu_total <= 0 )); then
      echo "invalid GPU capacity: ${gpu_row}" >&2
      exit 4
    fi
    if (( min_gpu_total_mib == 0 || gpu_total < min_gpu_total_mib )); then
      min_gpu_total_mib=${gpu_total}
    fi
  done
fi

source "${repo_root}/examples/glm52_lora/stack_env.sh"

mkdir -p "${run_dir}"
if [[ "${config_only}" != 1 ]]; then
  require_sha256 \
    "${model_path}/model.safetensors.index.json" \
    "${expected_index_sha256}" \
    "model index"
  bridge_head=$(git -C "${megatron_bridge_root}" rev-parse HEAD)
  if [[ -n "$(git -C "${megatron_bridge_root}" status --porcelain)" ]]; then
    echo "Megatron Bridge checkout must be clean for the full-model import" >&2
    exit 4
  fi
  bridge_audit_args=(--bridge-revision "${bridge_head}")
  if [[ "${checkpoint_profile}" == official-bf16 ]]; then
    if [[ "${bridge_head}" != "${expected_bridge_revision}" ]]; then
      echo "Megatron Bridge revision mismatch: expected=${expected_bridge_revision} actual=${bridge_head}" >&2
      exit 4
    fi
  else
    if ! git -C "${megatron_bridge_root}" merge-base --is-ancestor \
      "${expected_bridge_revision}" HEAD; then
      echo "Megatron Bridge does not descend from ${expected_bridge_revision}" >&2
      exit 4
    fi
    if [[ "$(git -C "${megatron_bridge_root}" rev-list --count "${expected_bridge_revision}..HEAD")" != 1 ]]; then
      echo "Megatron Bridge FP8 import overlay must contain exactly one commit" >&2
      exit 4
    fi
    expected_bridge_files=$'src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py\ntests/unit_tests/models/glm_moe_dsa/test_glm5_bridge.py'
    actual_bridge_files=$(git -C "${megatron_bridge_root}" diff --name-only \
      "${expected_bridge_revision}..HEAD")
    if [[ "${actual_bridge_files}" != "${expected_bridge_files}" ]]; then
      echo "Megatron Bridge FP8 import overlay changed unexpected files" >&2
      exit 4
    fi
    bridge_patch_sha256=$(
      git -C "${megatron_bridge_root}" diff \
        "${expected_bridge_revision}..HEAD" -- \
        src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py \
        tests/unit_tests/models/glm_moe_dsa/test_glm5_bridge.py |
        sha256sum | cut -d' ' -f1
    )
    if [[ "${bridge_patch_sha256}" != "${expected_bridge_fp8_patch_sha256}" ]]; then
      echo "Megatron Bridge FP8 import patch drift: expected=${expected_bridge_fp8_patch_sha256} actual=${bridge_patch_sha256}" >&2
      exit 4
    fi
    bridge_audit_args+=(
      --bridge-base-revision "${expected_bridge_revision}"
      --bridge-patch-sha256 "${bridge_patch_sha256}"
    )
  fi
  min_gpu_capacity_gib=$(awk -v capacity_mib="${min_gpu_total_mib}" \
    'BEGIN { printf "%.6f", capacity_mib / 1024 }')
  "${python_bin}" "${repo_root}/examples/glm52_lora/plan_full_sft_topologies.py" \
    "${model_path}/config.json" \
    --candidate "${world_size}:${tp_size}:${ep_size}:${etp_size}:${pp_size}:${cp_size}" \
    --device-capacity-gib "${min_gpu_capacity_gib}" \
    --sequence-length "${max_length}" \
    --lora-rank "${rank}" \
    "${planner_profile_args[@]}" \
    --minimum-additional-headroom-gib "${minimum_additional_headroom_gib}" \
    --require-candidate \
    > "${run_dir}/full-topology-plan-node${node_rank}.json"
  "${python_bin}" "${repo_root}/examples/glm52_lora/audit_full_checkpoint_loading.py" \
    "${model_path}" \
    --world-size "${world_size}" \
    --tp "${tp_size}" --ep "${ep_size}" --etp "${etp_size}" \
    --pp "${pp_size}" --cp "${cp_size}" \
    "${bridge_audit_args[@]}" \
    --official-profile "${official_audit_profile}" \
    > "${run_dir}/full-hf-load-audit-node${node_rank}.json"
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export MAX_JOBS=${MAX_JOBS:-4}

job_args=(
  engine=megatron
  optim=megatron
  "data.train_files=${train_file}"
  "data.val_files=${val_file}"
  "data.train_batch_size=${global_batch_size}"
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
  "engine.tensor_model_parallel_size=${tp_size}"
  "engine.pipeline_model_parallel_size=${pp_size}"
  "engine.expert_model_parallel_size=${ep_size}"
  "engine.expert_tensor_parallel_size=${etp_size}"
  "engine.context_parallel_size=${cp_size}"
  "engine.seed=${seed}"
  "engine.sequence_parallel=${sequence_parallel}"
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
  "trainer.experiment_name=${experiment_name}"
  "trainer.seed=${seed}"
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
