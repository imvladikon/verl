#!/usr/bin/env bash
set -euo pipefail

# Rank-16 ablation: the five qualified MLA projections plus the untied output
# layer. The base runner owns GPU auditing, metrics, data locks and checkpoint
# export; arguments here deliberately override only the target set and the
# BSHD/DSA-safe activation policy.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

export RUN_DIR=${RUN_DIR:-${repo_root}/runs/glm52-lora-surgery-sft-mla-lm-head}

targets='[linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj,output_layer]'

exec "${script_dir}/run_surgery_sft_megatron.sh" \
  "model.lora.target_modules=${targets}" \
  model.use_remove_padding=false \
  engine.pad_bshd_to_minibatch_max=true \
  engine.override_transformer_config.recompute_granularity=null \
  engine.override_transformer_config.recompute_method=null \
  engine.override_transformer_config.recompute_num_layers=null \
  "$@"
