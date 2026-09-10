#!/usr/bin/env bash
set -euo pipefail

# HISTORICAL / DO NOT LAUNCH. This stability run used the invalidated v2
# quality mixture. Its prior output remains systems evidence only; it is not
# valid training or quality evidence.

echo "HISTORICAL-INVALID-DATA: locked quality mixture v2 must not be launched" >&2
exit 2

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

python_bin=${PYTHON_BIN:-python3}
mixture_dir=${MIXTURE_DIR:?Set MIXTURE_DIR to the locked 2,728-row artifact}
model_path=${MODEL_PATH:-imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy}
run_dir=${RUN_DIR:-${repo_root}/runs/glm52-lora-surgery-locked-mixture-v2}
gpu_id=${GPU_ID:-}
config_only=${CONFIG_ONLY:-0}

expected_manifest_sha256=8453969b0a1e56fd876bef39ce8095ed45644e1e5c5f44217dc0eec869c419ed
expected_seq256_manifest_sha256=ef4e1b801b9ec49dfc5c737b89a9dd19014871158a83f88cb4413ab016f5fc27
expected_seq384_manifest_sha256=8ee60b837211a0f8375426d60e9a26f4197071df0fc96c8e5d351b6e4168382e
expected_seq768_manifest_sha256=28c3aed75965d17cedd2899fec5e428f9e3aa6cea2d4b21a64d6088c28b89c51
expected_seq256_train_sha256=360d64bb9f8d84748f13ff113dd736b75c6365b29155b4c2ea0cb4b4602bf819
expected_seq384_train_sha256=4d3eb1440dff7c5fde198130f4afc14c2990da685680599aff888b8e21e1dbb0
expected_seq768_train_sha256=ed6398563751b5e80351aca9cd3802011b0473eeb807ae3a4250b3338004f24b

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

manifest=${mixture_dir}/manifest.json
seq256_manifest=${mixture_dir}/seq256/manifest.json
seq384_manifest=${mixture_dir}/seq384/manifest.json
seq768_manifest=${mixture_dir}/seq768/manifest.json
seq256_train=${mixture_dir}/seq256/sft_train.parquet
seq384_train=${mixture_dir}/seq384/sft_train.parquet
seq768_train=${mixture_dir}/seq768/sft_train.parquet

require_sha256 "${manifest}" "${expected_manifest_sha256}" "mixture manifest"
require_sha256 "${seq256_manifest}" "${expected_seq256_manifest_sha256}" "seq256 manifest"
require_sha256 "${seq384_manifest}" "${expected_seq384_manifest_sha256}" "seq384 manifest"
require_sha256 "${seq768_manifest}" "${expected_seq768_manifest_sha256}" "seq768 manifest"
require_sha256 "${seq256_train}" "${expected_seq256_train_sha256}" "seq256 train parquet"
require_sha256 "${seq384_train}" "${expected_seq384_train_sha256}" "seq384 train parquet"
require_sha256 "${seq768_train}" "${expected_seq768_train_sha256}" "seq768 train parquet"

for source_path in "${seq256_train}" "${seq384_train}" "${seq768_train}"; do
  if [[ "${source_path}" == *"["* || "${source_path}" == *"]"* ||
        "${source_path}" == *","* || "${source_path}" == *" "* ]]; then
    echo "Hydra list-unsafe path: ${source_path}" >&2
    exit 3
  fi
done

if [[ "${config_only}" != 1 && -e "${run_dir}" ]]; then
  echo "refusing existing run directory: ${run_dir}" >&2
  exit 4
fi

train_files="[${seq256_train},${seq384_train},${seq768_train}]"
export PYTHON_BIN="${python_bin}"
export MODEL_PATH="${model_path}"
export TRAIN_FILE="${seq256_train}"
export RUN_DIR="${run_dir}"
export GPU_ID="${gpu_id}"
export CONFIG_ONLY="${config_only}"
export STEPS=33

"${script_dir}/run_surgery_sft_megatron.sh" \
  "data.train_files=${train_files}" \
  data.max_length=768 \
  data.max_token_len_per_gpu=768 \
  model.use_remove_padding=false \
  engine.pad_bshd_to_minibatch_max=true \
  engine.override_transformer_config.recompute_granularity=null \
  engine.override_transformer_config.recompute_method=null \
  engine.override_transformer_config.recompute_num_layers=null \
  optim.lr=1e-4 \
  trainer.save_freq=11 \
  trainer.max_ckpt_to_keep=3 \
  trainer.experiment_name=surgery-sft-locked-mixture-v2-33-updates

if [[ "${config_only}" == 1 ]]; then
  exit 0
fi

checkpoint=${run_dir}/global_step_33
"${python_bin}" "${script_dir}/verify_surgery_adapter.py" "${checkpoint}" \
  | tee "${run_dir}/adapter_verification.json"

GPU_ID="${gpu_id}" \
PYTHON_BIN="${python_bin}" \
MODEL_PATH="${model_path}" \
ADAPTER_PATH="${checkpoint}/model/huggingface/adapter" \
  /usr/bin/time -v -o "${run_dir}/reload.time" \
  "${script_dir}/run_verify_adapter_reload.sh" \
  >"${run_dir}/reload.json" 2>"${run_dir}/reload.stderr"

(
  cd -- "${run_dir}"
  sha256sum \
    "${manifest}" \
    "${seq256_manifest}" \
    "${seq384_manifest}" \
    "${seq768_manifest}" \
    "${seq256_train}" \
    "${seq384_train}" \
    "${seq768_train}" \
    adapter_verification.json \
    reload.json \
    reload.stderr \
    reload.time \
    time.txt \
    gpu.csv \
    global_step_33/model/huggingface/adapter/adapter_model.safetensors \
    >locked_mixture_gate_inputs.sha256
  sha256sum locked_mixture_gate_inputs.sha256 \
    | cut -d' ' -f1 >locked_mixture_gate.sha256
)

printf 'LOCKED_MIXTURE_GATE_PASS sha256=%s\n' \
  "$(<"${run_dir}/locked_mixture_gate.sha256")"
