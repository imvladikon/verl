#!/usr/bin/env bash
set -euo pipefail

# HISTORICAL / DO NOT LAUNCH. This 2,728-row v2 mixture predates the exhaustive
# split-isolation audit and is retained only as systems/configuration evidence.
# Use run_full_sft_clean_v4_megatron.sh with the checked-in clean-v4 view.

echo "HISTORICAL-INVALID-DATA: locked quality mixture v2 must not be launched" >&2
exit 2

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

mixture_dir=${MIXTURE_DIR:?Set MIXTURE_DIR to the locked 2,728-row artifact}
model_path=${MODEL_PATH:?Set MODEL_PATH to the immutable full GLM-5.2 snapshot}
lora_profile=${LORA_PROFILE:-mla-only}
run_dir=${RUN_DIR:-${repo_root}/runs/glm52-full-locked-quality-mixture-${lora_profile}}

declare -Ar expected_sha256=(
  [manifest.json]=8453969b0a1e56fd876bef39ce8095ed45644e1e5c5f44217dc0eec869c419ed
  [seq256/manifest.json]=ef4e1b801b9ec49dfc5c737b89a9dd19014871158a83f88cb4413ab016f5fc27
  [seq384/manifest.json]=8ee60b837211a0f8375426d60e9a26f4197071df0fc96c8e5d351b6e4168382e
  [seq768/manifest.json]=28c3aed75965d17cedd2899fec5e428f9e3aa6cea2d4b21a64d6088c28b89c51
  [seq256/sft_train.parquet]=360d64bb9f8d84748f13ff113dd736b75c6365b29155b4c2ea0cb4b4602bf819
  [seq384/sft_train.parquet]=4d3eb1440dff7c5fde198130f4afc14c2990da685680599aff888b8e21e1dbb0
  [seq768/sft_train.parquet]=ed6398563751b5e80351aca9cd3802011b0473eeb807ae3a4250b3338004f24b
  [seq256/sft_validation.parquet]=e662b197830a4f232fd75117396dfda62dfe60cb75e4eef9c1597b8a73fc155a
  [seq384/sft_validation.parquet]=2ace3c2ffe41f99c9a5f75bfb9219a030bcb603f24645bddffa6fc47c3ac8ded
  [seq768/sft_validation.parquet]=daf360c04c5205b0c6afdc7e1052689ef603786e6c5db3bb5d6bb6e0002df616
  [seq256/sft_test.parquet]=79b526958f549c4eeda0521606c54f247d896cabb88190e44e138ed8483c6b2c
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
export QUALIFICATION_PROFILE=locked-quality-mixture-v2-2728
export STEPS=33
export MAX_LENGTH=768
export REQUIRED_MAX_TOKENS=706

exec "${script_dir}/run_full_sft_megatron.sh" \
  "data.train_files=${train_list}" \
  "data.val_files=${val_list}" \
  trainer.save_freq=11 \
  trainer.test_freq=33 \
  trainer.max_ckpt_to_keep=3 \
  "trainer.experiment_name=full-sft-locked-mixture-${lora_profile}-r16"
