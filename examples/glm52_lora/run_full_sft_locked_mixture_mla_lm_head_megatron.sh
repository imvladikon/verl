#!/usr/bin/env bash
set -euo pipefail

# HISTORICAL / DO NOT LAUNCH. The delegated v2 data launcher fails closed and
# this wrapper is retained only to interpret its old configuration evidence.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${VERL_REPO_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}

export LORA_PROFILE=mla-lm-head
export RUN_DIR=${RUN_DIR:-${repo_root}/runs/glm52-full-locked-quality-mixture-mla-lm-head}

exec "${script_dir}/run_full_sft_locked_mixture_megatron.sh" "$@"
