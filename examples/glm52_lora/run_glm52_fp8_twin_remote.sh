#!/usr/bin/env bash
set -euo pipefail

# CPU-only, bounded-memory builder for the FP8 rollout half of our GLM-5.2
# 8.76B surgery pair. The default performs metadata download and a dry run.
# Real transfer and Hub publication require separate explicit switches.

readonly SOURCE_REPO="zai-org/GLM-5.2-FP8"
readonly SOURCE_REVISION="f33c6dc501ee5a2c7e35155653b1b1abbc320951"
readonly TARGET_REPO="imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy-FP8"
readonly TARGET_REVISION="5eedf18a056d10b37452528c930487cc48dbd63a"
readonly EXPECTED_PLAN_SHA256="5e0152c0d8dcbc7e0fdb236e4b264ab4a7e997fa6421e20970b4a274c5883181"
readonly EXPECTED_USER="imvladikon"

tool_dir="${GLM52_TOOL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
build_root="${GLM52_BUILD_ROOT:-$HOME/glm52-lora-surgery-build}"
metadata_dir="$build_root/source-metadata"
plan_dir="$build_root/published-plan"
model_dir="$build_root/GLM-5.2-9B-LoRA-Surgery-Dummy-FP8"
plan_file="$plan_dir/surgery_plan.json"
python_bin="${GLM52_PYTHON:-python3}"
hf_bin="${GLM52_HF:-hf}"

for command_name in "$python_bin" "$hf_bin" nice ionice; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "missing command: $command_name" >&2
    exit 2
  fi
done
"$python_bin" -c 'import numpy' >/dev/null

builder="$tool_dir/build_glm52_surgery_dummy.py"
verifier="$tool_dir/verify_glm52_surgery_pair.py"
for required_file in "$builder" "$verifier"; do
  if [[ ! -f "$required_file" ]]; then
    echo "missing checked-in surgery tool: $required_file" >&2
    exit 2
  fi
done

mkdir -p "$metadata_dir" "$plan_dir" "$build_root"
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false

echo "Downloading pinned metadata only from $SOURCE_REPO@$SOURCE_REVISION"
"$hf_bin" download "$SOURCE_REPO" \
  config.json model.safetensors.index.json .gitattributes \
  chat_template.jinja generation_config.json tokenizer.json \
  tokenizer_config.json \
  --type model \
  --revision "$SOURCE_REVISION" \
  --local-dir "$metadata_dir"

echo "Downloading the immutable published surgery plan from $TARGET_REPO@$TARGET_REVISION"
"$hf_bin" download "$TARGET_REPO" surgery_plan.json \
  --type model \
  --revision "$TARGET_REVISION" \
  --local-dir "$plan_dir"

actual_plan_sha256="$($python_bin - "$plan_file" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"
if [[ "$actual_plan_sha256" != "$EXPECTED_PLAN_SHA256" ]]; then
  echo "published surgery plan hash drift: $actual_plan_sha256" >&2
  exit 3
fi

build_args=(
  "$python_bin"
  "$builder"
  --plan "$plan_file"
  --metadata-dir "$metadata_dir"
  --output "$model_dir"
  --chunk-mib "${GLM52_CHUNK_MIB:-8}"
  --max-shard-size-gib "${GLM52_MAX_SHARD_GIB:-1.5}"
)
if [[ "${EXECUTE:-0}" == "1" ]]; then
  build_args+=(--execute)
fi

echo "Build mode: EXECUTE=${EXECUTE:-0}; output=$model_dir"
nice -n 10 ionice -c2 -n7 "${build_args[@]}"

if [[ "${EXECUTE:-0}" != "1" ]]; then
  echo "Dry run complete. Set EXECUTE=1 for the resumable weight transfer."
  exit 0
fi

nice -n 10 ionice -c2 -n7 \
  "$python_bin" "$verifier" \
  --single-plan "$plan_file" \
  --single-precision fp8-rollout \
  --single-model "$model_dir"

if [[ "${PUBLISH:-0}" != "1" ]]; then
  echo "Verified build complete. Set PUBLISH=1 together with EXECUTE=1 to upload."
  exit 0
fi

hub_identity="$(
  "$hf_bin" auth whoami --format json |
    "$python_bin" -c 'import json, sys; print(json.load(sys.stdin).get("user", ""))'
)"
if [[ "$hub_identity" != "$EXPECTED_USER" ]]; then
  echo "refusing upload: expected Hub user $EXPECTED_USER, got ${hub_identity:-none}" >&2
  exit 3
fi

nice -n 15 ionice -c3 \
  "$hf_bin" upload "$TARGET_REPO" "$model_dir" . \
  --type model \
  --revision main \
  --no-private \
  --commit-message "Rebuild the verified GLM-5.2 FP8 surgery dummy"

echo "Published https://huggingface.co/$TARGET_REPO/tree/main"
