#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
rank=${LORA_RANK:-4}
alpha=${LORA_ALPHA:-8}
steps=${GLM53_GRPO_STEPS:-2}

if (( rank <= 0 || alpha <= 0 )); then
  echo "LORA_RANK and LORA_ALPHA must be positive" >&2
  exit 2
fi

if [[ -n ${TRANSFORMERS_SOURCE:-} ]]; then
  transformers_source=${TRANSFORMERS_SOURCE%/}
  modeling_file=${transformers_source}/src/transformers/models/glm5_next/modeling_glm5_next.py
  if [[ ! -f ${modeling_file} ]]; then
    echo "TRANSFORMERS_SOURCE does not contain the GLM5-Next implementation" >&2
    exit 2
  fi
  if [[ -n ${TRANSFORMERS_COMMIT:-} ]]; then
    actual_commit=$(git -C "${transformers_source}" rev-parse HEAD)
    if [[ ${actual_commit} != "${TRANSFORMERS_COMMIT}" ]]; then
      echo "Transformers source is ${actual_commit}, expected ${TRANSFORMERS_COMMIT}" >&2
      exit 2
    fi
  fi
  export PYTHONPATH="${transformers_source}/src${PYTHONPATH:+:${PYTHONPATH}}"
fi

export GLM53_OPTIMIZER_IMPL=torch.optim
export GLM53_OPTIMIZER=AdamW

"${repo_root}/examples/glm53_flash/run_glm53_flash_9b_smoke.sh" \
  actor_rollout_ref.model.lora_rank="${rank}" \
  actor_rollout_ref.model.lora_alpha="${alpha}" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=true \
  actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
  actor_rollout_ref.actor.checkpoint.load_contents='["model"]' \
  +actor_rollout_ref.actor.checkpoint.save_lora_only=true \
  trainer.save_freq="${steps}" \
  "$@"

echo "GLM-5.3-Flash 9B two-step LoRA RL smoke passed"
