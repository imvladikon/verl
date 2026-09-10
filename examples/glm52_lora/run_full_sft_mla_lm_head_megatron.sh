#!/usr/bin/env bash
set -euo pipefail

# Full-model rank-16 ablation: the five qualified MLA projections plus the
# untied output layer. The base runner owns all topology, data and evidence
# locks; this wrapper changes only the adapter surface and output directory.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${VERL_REPO_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}

export LORA_PROFILE=mla-lm-head
export RUN_DIR=${RUN_DIR:-${repo_root}/runs/glm52-full-quality-sft-megatron-mla-lm-head}

exec "${script_dir}/run_full_sft_megatron.sh" "$@"
