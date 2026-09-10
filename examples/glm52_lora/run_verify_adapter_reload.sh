#!/usr/bin/env bash
set -euo pipefail

# Reload an exported HF LoRA adapter on an audited GPU using the same resolved
# CUDA/Megatron environment as the SFT and GRPO qualification scripts.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

python_bin=${PYTHON_BIN:-python3}
model_path=${MODEL_PATH:-imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy}
adapter_path=${ADAPTER_PATH:?Set ADAPTER_PATH to the exported HF adapter directory}
gpu_id=${GPU_ID:-}
max_used_mib=${MAX_USED_MIB:-256}

source "${script_dir}/stack_env.sh"

: "${gpu_id:?Set GPU_ID to an audited free physical GPU index}"
if [[ ! "${max_used_mib}" =~ ^[0-9]+$ ]]; then
  echo "MAX_USED_MIB must be a nonnegative integer" >&2
  exit 2
fi
if [[ ! -f "${adapter_path}/adapter_config.json" ]]; then
  echo "adapter_config.json not found: ${adapter_path}" >&2
  exit 2
fi

gpu_used_mib=$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [[ ! "${gpu_used_mib}" =~ ^[0-9]+$ ]] || (( gpu_used_mib > max_used_mib )); then
  echo "refusing to use GPU ${gpu_id}: memory.used=${gpu_used_mib:-unknown} MiB" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TOKENIZERS_PARALLELISM=false

exec "${python_bin}" "${script_dir}/verify_adapter_reload.py" \
  --model "${model_path}" \
  --adapter "${adapter_path}" \
  --device cuda:0
