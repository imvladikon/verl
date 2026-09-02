#!/usr/bin/env bash
set -euo pipefail

# One-GPU MLA-LoRA SFT gate for the authentic Russian corruption corpus.  The
# Markdown family is deliberately kept in a separate, longer sequence bucket:
# the pinned 64-article audit has a 556-token maximum, so seq=256 would truncate
# valid targets and invalidate both the loss and the memory measurement.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

max_length=${MAX_LENGTH:-640}
required_max_tokens=${REQUIRED_MAX_TOKENS:-556}
export RUN_DIR=${RUN_DIR:-${repo_root}/runs/glm52-lora-surgery-sft-teacher-free}

if [[ ! "${max_length}" =~ ^[0-9]+$ ]] || [[ ! "${required_max_tokens}" =~ ^[0-9]+$ ]]; then
  echo "MAX_LENGTH and REQUIRED_MAX_TOKENS must be positive integers" >&2
  exit 2
fi
if (( max_length < required_max_tokens )); then
  echo "MAX_LENGTH=${max_length} truncates the audited ${required_max_tokens}-token example" >&2
  exit 2
fi

exec "${script_dir}/run_surgery_sft_megatron.sh" \
  "data.max_length=${max_length}" \
  "data.max_token_len_per_gpu=${max_length}" \
  model.use_remove_padding=false \
  engine.pad_bshd_to_minibatch_max=true \
  engine.override_transformer_config.recompute_granularity=null \
  engine.override_transformer_config.recompute_method=null \
  engine.override_transformer_config.recompute_num_layers=null \
  "$@"
