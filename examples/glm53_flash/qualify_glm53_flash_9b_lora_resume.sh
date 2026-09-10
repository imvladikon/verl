#!/usr/bin/env bash
# Qualify that a Flash LoRA run can actually be continued, not just started.
#
# The two smoke scripts next to this one deliberately do not test that: the SFT
# one runs with save_freq=-1, resume_mode=disable and saves only ["model"], and
# the RL wrapper also saves the model alone. Nothing there exercises the
# optimizer state, the RNG or the data cursor, so a green smoke says the run
# starts, not that it survives a restart.
#
# This script runs the same configuration three times:
#   reference  steps 1..N in one process
#   part one   steps 1..K, saving model + optimizer + extra
#   part two   a NEW process resuming from K and finishing at N
# and then compares the adapter of the reference against the resumed one. They
# have to match: a resume that silently restarts the optimizer or reshuffles the
# data produces a different adapter while reporting success.
#
# Data settings are strict here on purpose. The smoke allows right truncation
# and ignores input-id mismatches, which is fine for a two-step start-up check
# and wrong for anything that claims the pipeline is correct.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model_path=${MODEL_PATH:?Set MODEL_PATH to the local 9B surgery checkpoint}
train_file=${TRAIN_FILE:?Set TRAIN_FILE to a small messages parquet}
total=${GLM53_TOTAL_STEPS:-4}
half=${GLM53_RESUME_AT:-2}
rank=${LORA_RANK:-4}
alpha=${LORA_ALPHA:-8}
run_id=${GLM53_RUN_ID:-lora_resume_9b_$(date -u +%Y%m%dT%H%M%SZ)}
root=${OUTPUT_DIR:-"${repo_root}/outputs/glm53_flash/${run_id}"}

if (( half < 1 || half >= total )); then
  echo "GLM53_RESUME_AT must satisfy 1 <= RESUME_AT < TOTAL_STEPS" >&2
  exit 2
fi

if [[ ${VERL_USE_UV:-1} == 1 ]]; then
  runner=(uv run --frozen --extra glm53-flash torchrun --standalone --nnodes=1 --nproc_per_node=1)
else
  runner=(torchrun --standalone --nnodes=1 --nproc_per_node=1)
fi

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${root}"

train() {   # <output dir> <total steps> <save freq> <resume mode>
  local out=$1 steps=$2 save_freq=$3 resume=$4
  mkdir -p "${out}"
  "${runner[@]}" -m verl.trainer.sft_trainer \
    data.train_files="${train_file}" \
    data.val_files=null \
    data.train_batch_size=2 \
    data.micro_batch_size_per_gpu=1 \
    data.max_token_len_per_gpu=128 \
    data.use_dynamic_bsz=false \
    data.max_length=128 \
    data.truncation=error \
    data.num_workers=0 \
    data.ignore_input_ids_mismatch=false \
    model.path="${model_path}" \
    +model.override_config.attn_implementation=eager \
    +model.override_config.experts_implementation=eager \
    model.use_remove_padding=false \
    model.enable_gradient_checkpointing=true \
    model.freeze_vision_tower=true \
    model.lora_rank="${rank}" \
    model.lora_alpha="${alpha}" \
    model.target_modules=all-linear \
    engine=fsdp \
    engine.param_offload=true \
    engine.optimizer_offload=true \
    engine.model_dtype=bf16 \
    engine.dtype=bfloat16 \
    engine.use_orig_params=false \
    engine.use_torch_compile=false \
    optim.optimizer_impl=torch.optim \
    optim.optimizer=AdamW \
    optim.lr=0.0001 \
    checkpoint.save_contents='["model","optimizer","extra"]' \
    checkpoint.load_contents='["model","optimizer","extra"]' \
    +checkpoint.save_lora_only=true \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${steps}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.balance_batch=false \
    trainer.save_freq="${save_freq}" \
    trainer.test_freq=-1 \
    trainer.resume_mode="${resume}" \
    trainer.logger='["console"]' \
    trainer.project_name=glm53_flash \
    trainer.experiment_name="$(basename "${out}")" \
    trainer.default_local_dir="${out}"
}

echo "== reference: ${total} steps in one process"
train "${root}/reference" "${total}" "${total}" disable

echo "== part one: ${half} steps, saving optimizer and extra"
train "${root}/resumed" "${half}" "${half}" disable

echo "== part two: new process, resuming to ${total}"
train "${root}/resumed" "${total}" "${total}" auto

echo "== comparing adapters"
python3 - "${root}/reference" "${root}/resumed" "${total}" <<'PY'
import pathlib, sys
import torch

reference, resumed, total = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]


def adapter(root):
    hits = sorted(root.rglob("adapter_model.safetensors")) + sorted(root.rglob("adapter_model.bin"))
    if not hits:
        raise SystemExit(f"no adapter written under {root}")
    latest = max(hits, key=lambda p: p.stat().st_mtime)
    if latest.suffix == ".safetensors":
        from safetensors.torch import load_file
        return latest, load_file(str(latest))
    return latest, torch.load(latest, map_location="cpu")


ref_path, ref = adapter(reference)
res_path, res = adapter(resumed)
print(f"reference {ref_path}")
print(f"resumed   {res_path}")

if set(ref) != set(res):
    raise SystemExit(f"adapter keys differ: only in reference {sorted(set(ref) - set(res))}, "
                     f"only in resumed {sorted(set(res) - set(ref))}")

worst, worst_key = 0.0, None
for key in sorted(ref):
    a, b = ref[key].float(), res[key].float()
    if a.shape != b.shape:
        raise SystemExit(f"{key}: shapes differ, {tuple(a.shape)} vs {tuple(b.shape)}")
    delta = (a - b).abs().max().item()
    if delta > worst:
        worst, worst_key = delta, key

# BF16 training is not bit-reproducible across a restart, but a resume that
# dropped the optimizer moments or reshuffled the data moves the adapter far
# more than accumulation noise.
tolerance = 1e-3
print(f"largest difference {worst:.3e} at {worst_key} (tolerance {tolerance:g})")
if worst > tolerance:
    raise SystemExit("resume does not reproduce the reference adapter")
print("resume qualification passed")
PY
