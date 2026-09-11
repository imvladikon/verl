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
# A fourth run is the negative control: the same split with only the model
# saved, so Adam really is restarted. Without it the tolerance proves nothing,
# because a comparison that cannot fail cannot qualify anything.
#
# The comparison reads the payload the trainer actually writes for an
# adapter-only save -- model_world_size_*_rank_*.pt with its format marker --
# at exactly the expected final step. adapter_model.safetensors belongs to the
# hf_model branch, which gathers the full base model, the very peak an
# adapter-only save exists to avoid. Empty, non-finite and mismatched payloads
# are rejected rather than silently compared: with `delta > worst` alone, NaN
# passes, because every comparison against NaN is false.
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

train() {   # <output dir> <total steps> <save freq> <resume mode> [save contents] [load contents]
  local out=$1 steps=$2 save_freq=$3 resume=$4
  local save_contents=${5:-'["model","optimizer","extra"]'}
  local load_contents=${6:-'["model","optimizer","extra"]'}
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
    checkpoint.save_contents="${save_contents}" \
    checkpoint.load_contents="${load_contents}" \
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

echo "== negative control: resume with the optimizer deliberately dropped"
# A resume that restarts Adam must not look like a good one. Save the model
# alone at step K, resume from it, and require the comparison to fail.
train "${root}/no_optimizer" "${half}" "${half}" disable '["model"]' '["model"]'
train "${root}/no_optimizer" "${total}" "${total}" auto '["model"]' '["model"]'

echo "== comparing adapters"
python3 - "${root}/reference" "${root}/resumed" "${root}/no_optimizer" "${total}" <<'COMPARE'
import pathlib
import sys

import torch

reference, resumed, control, total = (
    pathlib.Path(sys.argv[1]),
    pathlib.Path(sys.argv[2]),
    pathlib.Path(sys.argv[3]),
    int(sys.argv[4]),
)

# The trainer writes the adapter-only payload as a torch file per rank, with a
# format marker inside. adapter_model.safetensors belongs to the hf_model save
# branch, which gathers the full base model -- exactly the peak an adapter-only
# save exists to avoid -- so it is not what this qualification may look at.
FORMAT_KEY = "__verl_lora_checkpoint_format__"
EXPECTED_FORMAT = "peft_adapter_v1"


def adapter(root, step):
    """The adapter saved at exactly `step`, verified to be a usable payload."""
    directory = root / f"global_step_{step}"
    if not directory.is_dir():
        raise SystemExit(f"no checkpoint for step {step} under {root}")
    shards = sorted(directory.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        raise SystemExit(f"no adapter shard written in {directory}")

    merged = {}
    for shard in shards:
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        marker = payload.pop(FORMAT_KEY, None)
        if marker != EXPECTED_FORMAT:
            raise SystemExit(f"{shard} is not an adapter-only checkpoint (marker {marker!r})")
        for key, value in payload.items():
            if not torch.is_tensor(value):
                raise SystemExit(f"{shard}: {key} is {type(value).__name__}, not a tensor")
            merged[f"{shard.name}:{key}"] = value
    if not merged:
        raise SystemExit(f"{directory}: adapter is empty")
    for key, value in merged.items():
        if value.numel() == 0:
            raise SystemExit(f"{directory}: {key} has no elements")
        if not torch.isfinite(value.float()).all():
            raise SystemExit(f"{directory}: {key} is not finite")
    return directory, merged


def compare(left, right):
    """Largest deviation, or a description of why they are not comparable."""
    if set(left) != set(right):
        only_left = sorted(set(left) - set(right))
        only_right = sorted(set(right) - set(left))
        return None, f"adapter keys differ: only in first {only_left}, only in second {only_right}"
    worst, worst_key = 0.0, None
    for key in sorted(left):
        a, b = left[key].float(), right[key].float()
        if a.shape != b.shape:
            return None, f"{key}: shapes differ, {tuple(a.shape)} vs {tuple(b.shape)}"
        delta = (a - b).abs().max().item()
        # NaN fails every comparison, so it has to be rejected explicitly:
        # `delta > worst` is False for NaN and would leave worst at zero.
        if delta != delta:
            return None, f"{key}: difference is NaN"
        if delta > worst:
            worst, worst_key = delta, key
    return (worst, worst_key), None


reference_path, reference_state = adapter(reference, total)
resumed_path, resumed_state = adapter(resumed, total)
print(f"reference {reference_path}")
print(f"resumed   {resumed_path}")
print(f"tensors   {len(reference_state)}")

# BF16 training is not bit-reproducible across a restart, but a resume that
# dropped the optimizer moments or reshuffled the data moves the adapter far
# more than accumulation noise.
tolerance = 1e-3
result, problem = compare(reference_state, resumed_state)
if problem:
    raise SystemExit(f"resume does not reproduce the reference adapter: {problem}")
worst, worst_key = result
print(f"largest difference {worst:.3e} at {worst_key} (tolerance {tolerance:g})")
if worst > tolerance:
    raise SystemExit("resume does not reproduce the reference adapter")

# The tolerance only means something if a resume that really lost its optimizer
# state lands outside it. Without this the check cannot tell a correct resume
# from a run that silently restarted Adam.
_, control_state = adapter(control, total)
control_result, control_problem = compare(reference_state, control_state)
if control_problem:
    print(f"negative control differs structurally: {control_problem}")
else:
    control_worst, control_key = control_result
    print(f"negative control difference {control_worst:.3e} at {control_key}")
    if control_worst <= tolerance:
        raise SystemExit(
            "a resume without optimizer state reproduced the reference too: "
            "this comparison cannot qualify anything"
        )

print("resume qualification passed")
COMPARE
