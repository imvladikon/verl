#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model_path=${MODEL_PATH:?Set MODEL_PATH to a local tiny GLM-5.3-Flash checkpoint}
run_id=${GLM53_RUN_ID:-automodel_sft_$(date -u +%Y%m%dT%H%M%SZ)}
output_dir=${OUTPUT_DIR:-"${repo_root}/outputs/glm53_flash/${run_id}"}
data_dir=${DATA_DIR:-"${output_dir}/data"}

if [[ -e "${output_dir}/global_step_1" ]]; then
  echo "Refusing to reuse an existing checkpoint directory: ${output_dir}" >&2
  exit 2
fi

if [[ ${VERL_USE_UV:-1} == 1 ]]; then
  runner=(uv run --frozen --extra glm53-automodel python)
else
  runner=(python3)
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${output_dir}" "${data_dir}"
"${runner[@]}" "${repo_root}/examples/glm53_flash/prepare_tiny_data.py" \
  --output-dir "${data_dir}"
"${runner[@]}" "${repo_root}/examples/glm53_flash/verify_provenance.py" \
  --profile automodel \
  --output "${output_dir}/provenance.json"

"${runner[@]}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  -m verl.trainer.sft_trainer \
  model.path="${model_path}" \
  model.freeze_vision_tower=true \
  model.enable_gradient_checkpointing=false \
  model.use_remove_padding=true \
  engine=automodel \
  engine.distributed_strategy=fsdp2 \
  engine.model_dtype=bf16 \
  engine.attn_implementation=sdpa \
  engine.enable_compile=false \
  engine.backend_config.attn=sdpa \
  engine.backend_config.linear=torch \
  engine.backend_config.rms_norm=torch_fp32 \
  engine.backend_config.experts=torch_mm \
  engine.backend_config.dispatcher=torch \
  engine.backend_config.rope_fusion=false \
  engine.backend_config.enable_hf_state_dict_adapter=true \
  optim=automodel \
  optim.lr=0.001 \
  data.train_files="${data_dir}/sft.parquet" \
  data.val_files=null \
  data.train_batch_size=1 \
  data.micro_batch_size_per_gpu=1 \
  data.use_dynamic_bsz=false \
  data.max_length=128 \
  data.max_token_len_per_gpu=128 \
  data.pad_mode=no_padding \
  data.truncation=right \
  data.ignore_input_ids_mismatch=true \
  data.num_workers=0 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  'trainer.logger=["console"]' \
  trainer.resume_mode=disable \
  trainer.default_local_dir="${output_dir}" \
  'checkpoint.save_contents=[model]'

"${runner[@]}" "${repo_root}/examples/glm53_flash/verify_checkpoints.py" sft \
  --base-model "${model_path}" \
  --checkpoint "${output_dir}/global_step_1/model/consolidated" \
  --output "${output_dir}/checkpoint_verification.json"

echo "GLM-5.3-Flash AutoModel SFT lifecycle passed: ${output_dir}"
