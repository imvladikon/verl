#!/usr/bin/env bash
set -euo pipefail

# Exact-data counterpart to run_full_sft_locked_mixture_megatron.sh. It changes
# only the adapter surface and run directory, keeping data, seed, token budget,
# optimizer updates and topology identical to the MLA-only qualification.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${VERL_REPO_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}

export LORA_PROFILE=mla-lm-head
export RUN_DIR=${RUN_DIR:-${repo_root}/runs/glm52-full-locked-quality-mixture-mla-lm-head}

exec "${script_dir}/run_full_sft_locked_mixture_megatron.sh" "$@"
