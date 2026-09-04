#!/usr/bin/env bash
set -euo pipefail

# Controlled output-head ablation for the same exact clean-v4 28x64 view.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${VERL_REPO_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}

export LORA_PROFILE=mla-lm-head
export RUN_DIR=${RUN_DIR:-${repo_root}/runs/glm52-full-clean-v4-format-script-mla-lm-head}

exec "${script_dir}/run_full_sft_clean_v4_megatron.sh" "$@"
