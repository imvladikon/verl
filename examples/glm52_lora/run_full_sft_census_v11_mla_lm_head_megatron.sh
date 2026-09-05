#!/usr/bin/env bash
set -euo pipefail

# Conditional output-head ablation on the same exact censused v11 view.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${VERL_REPO_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}

export LORA_PROFILE=mla-lm-head
export RUN_DIR=${RUN_DIR:-${repo_root}/runs/glm52-full-census-v11-mla-lm-head}

exec "${script_dir}/run_full_sft_census_v11_megatron.sh" "$@"
