#!/usr/bin/env bash
set -euo pipefail

# Full GLM-5.2 LoRA SFT profile for the exact 2,728-row quality mixture.
# The three train buckets form one optimizer stream. Validation uses the three
# disjoint validation buckets; test buckets are hash-locked but never trained
# or selected on.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

mixture_dir=${MIXTURE_DIR:?Set MIXTURE_DIR to the locked 2,728-row artifact}
model_path=${MODEL_PATH:?Set MODEL_PATH to the immutable full GLM-5.2 snapshot}
run_dir=${RUN_DIR:-${repo_root}/runs/glm52-full-locked-quality-mixture}

declare -Ar expected_sha256=(
  [manifest.json]=e0e471b74d2b56cbbb681f665899d355f2b30505ea93a996cab2cd9187d54aef
  [seq256/manifest.json]=d9d6f073d302a8e34ad20b0088298ef1da72a1d6a430d94d2a3df2cb8ec77780
  [seq384/manifest.json]=8ee60b837211a0f8375426d60e9a26f4197071df0fc96c8e5d351b6e4168382e
  [seq768/manifest.json]=28c3aed75965d17cedd2899fec5e428f9e3aa6cea2d4b21a64d6088c28b89c51
  [seq256/sft_train.parquet]=cb25249a0b43ccc45c1c984a942a6f2b0f6741569af868f6c948e707ae57bf93
  [seq384/sft_train.parquet]=4d3eb1440dff7c5fde198130f4afc14c2990da685680599aff888b8e21e1dbb0
  [seq768/sft_train.parquet]=ed6398563751b5e80351aca9cd3802011b0473eeb807ae3a4250b3338004f24b
  [seq256/sft_validation.parquet]=32af28ffab5dc6c3a465bdc4edd1f19820b50fb12bbc68d7cabf63a4602428c2
  [seq384/sft_validation.parquet]=2ace3c2ffe41f99c9a5f75bfb9219a030bcb603f24645bddffa6fc47c3ac8ded
  [seq768/sft_validation.parquet]=daf360c04c5205b0c6afdc7e1052689ef603786e6c5db3bb5d6bb6e0002df616
  [seq256/sft_test.parquet]=f9564e7bbf51d10aa91ac4fbe996ca09f7094000ec3818613a0ee999903b1008
  [seq384/sft_test.parquet]=1705623ddb7592a00101199e0e252c542bc67a8ce4a38ad27f21d5f1bacaa679
  [seq768/sft_test.parquet]=a7254bcdbda8e07f0bfeeaae3142096b5a9e23b612682243a432317306068301
)

for relative_file in "${!expected_sha256[@]}"; do
  source_file=${mixture_dir}/${relative_file}
  if [[ ! -f "${source_file}" ]]; then
    echo "locked input not found: ${source_file}" >&2
    exit 3
  fi
  actual=$(sha256sum "${source_file}" | cut -d' ' -f1)
  if [[ "${actual}" != "${expected_sha256[${relative_file}]}" ]]; then
    echo "locked input SHA-256 mismatch: ${relative_file} expected=${expected_sha256[${relative_file}]} actual=${actual}" >&2
    exit 3
  fi
done

train_files=(
  "${mixture_dir}/seq256/sft_train.parquet"
  "${mixture_dir}/seq384/sft_train.parquet"
  "${mixture_dir}/seq768/sft_train.parquet"
)
val_files=(
  "${mixture_dir}/seq256/sft_validation.parquet"
  "${mixture_dir}/seq384/sft_validation.parquet"
  "${mixture_dir}/seq768/sft_validation.parquet"
)

for source_file in "${train_files[@]}" "${val_files[@]}"; do
  if [[ "${source_file}" == *"["* || "${source_file}" == *"]"* ||
        "${source_file}" == *","* || "${source_file}" == *" "* ]]; then
    echo "Hydra list-unsafe path: ${source_file}" >&2
    exit 3
  fi
done

train_list=$(IFS=,; printf '[%s]' "${train_files[*]}")
val_list=$(IFS=,; printf '[%s]' "${val_files[*]}")

export MODEL_PATH="${model_path}"
export TRAIN_FILE="${train_files[0]}"
export VAL_FILE="${val_files[0]}"
export RUN_DIR="${run_dir}"
export EXPECTED_TRAIN_SHA256="${expected_sha256[seq256/sft_train.parquet]}"
export EXPECTED_VAL_SHA256="${expected_sha256[seq256/sft_validation.parquet]}"
export QUALIFICATION_PROFILE=locked-quality-mixture-2728
export STEPS=33
export MAX_LENGTH=768
export REQUIRED_MAX_TOKENS=706

exec "${script_dir}/run_full_sft_megatron.sh" \
  "data.train_files=${train_list}" \
  "data.val_files=${val_list}" \
  trainer.save_freq=11 \
  trainer.test_freq=33 \
  trainer.max_ckpt_to_keep=3 \
  trainer.experiment_name=full-sft-locked-mixture-mla-r16
