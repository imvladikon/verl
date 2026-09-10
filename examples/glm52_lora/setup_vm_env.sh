#!/usr/bin/env bash
set -euo pipefail

# Create an isolated GLM-5.2 validation checkout and venv.  Existing projects,
# venvs, Ray clusters, and GPU processes are never modified.

root=${GLM52_VM_ROOT:-$HOME/glm52lora}
branch=${GLM_BRANCH:-glm-5.2}
python_seed=${PYTHON_SEED:-python3.12}
phase=${GLM52_SETUP_PHASE:-all}
bridge_revision=${MEGATRON_BRIDGE_REVISION:-d0c6228a2a832f566dd44a3a179b3136613c11b7}
bridge_fp8_import_revision=${MEGATRON_BRIDGE_FP8_IMPORT_REVISION:-44c871b4eab107028933ea1a3aaa42dacc260c1c}
bridge_fp8_import_patch_sha256=${MEGATRON_BRIDGE_FP8_IMPORT_PATCH_SHA256:-d5764f406994684392cb78bc2977b6ca90a30680c448022742023d9c1298c590}
apply_bridge_fp8_import=${APPLY_MEGATRON_BRIDGE_FP8_IMPORT:-1}
transformer_engine_version=${TRANSFORMER_ENGINE_VERSION:-2.18.0}
modelopt_version=${MODELOPT_VERSION:-0.46.0rc1}

if [[ "${phase}" != "all" && "${phase}" != "clone" && "${phase}" != "install" ]]; then
  echo "GLM52_SETUP_PHASE must be all, clone, or install" >&2
  exit 2
fi

if [[ "${phase}" == "all" || "${phase}" == "clone" ]]; then
  if [[ -e "${root}/src" || -e "${root}/.venv" ]]; then
    echo "refusing to overwrite existing environment below ${root}" >&2
    exit 2
  fi

  mkdir -p "${root}/src"
  git clone --filter=blob:none --single-branch --branch "${branch}" \
    https://github.com/imvladikon/verl.git "${root}/src/verl"
  git clone --filter=blob:none --single-branch --branch "${branch}" \
    https://github.com/imvladikon/sglang.git "${root}/src/sglang"
  git clone --filter=blob:none --single-branch --branch "${branch}" \
    https://github.com/imvladikon/Megatron-LM.git "${root}/src/Megatron-LM"
  git clone --filter=blob:none --single-branch --branch "${branch}" \
    https://github.com/imvladikon/slime.git "${root}/src/slime"
  git clone --filter=blob:none https://github.com/NVIDIA-NeMo/Megatron-Bridge.git \
    "${root}/src/Megatron-Bridge"
  git -C "${root}/src/Megatron-Bridge" checkout "${bridge_revision}"
  if [[ "${apply_bridge_fp8_import}" == 1 ]]; then
    git -C "${root}/src/Megatron-Bridge" \
      -c user.name='Hersh Godse' -c user.email='hersh.godse@gmail.com' \
      cherry-pick "${bridge_fp8_import_revision}"
    actual_patch_sha256=$(
      git -C "${root}/src/Megatron-Bridge" diff \
        "${bridge_revision}..HEAD" -- \
        src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py \
        tests/unit_tests/models/glm_moe_dsa/test_glm5_bridge.py |
        sha256sum | cut -d' ' -f1
    )
    if [[ "${actual_patch_sha256}" != "${bridge_fp8_import_patch_sha256}" ]]; then
      echo "Megatron Bridge FP8 import patch drift: expected=${bridge_fp8_import_patch_sha256} actual=${actual_patch_sha256}" >&2
      exit 2
    fi
  elif [[ "${apply_bridge_fp8_import}" != 0 ]]; then
    echo "APPLY_MEGATRON_BRIDGE_FP8_IMPORT must be 0 or 1" >&2
    exit 2
  fi
fi

if [[ "${phase}" == "clone" ]]; then
  echo "Clone phase complete below ${root}/src; run with GLM52_SETUP_PHASE=install after reviewing the resolved sources."
  exit 0
fi

for repository in verl sglang Megatron-LM slime Megatron-Bridge; do
  if [[ ! -d "${root}/src/${repository}/.git" ]]; then
    echo "missing cloned repository: ${root}/src/${repository}" >&2
    exit 2
  fi
done
bridge_root=${root}/src/Megatron-Bridge
if [[ -n "$(git -C "${bridge_root}" status --porcelain)" ]]; then
  echo "Megatron Bridge checkout must be clean" >&2
  exit 2
fi
if [[ "${apply_bridge_fp8_import}" == 1 ]]; then
  if ! git -C "${bridge_root}" merge-base --is-ancestor "${bridge_revision}" HEAD; then
    echo "Megatron Bridge does not descend from ${bridge_revision}" >&2
    exit 2
  fi
  if [[ "$(git -C "${bridge_root}" rev-list --count "${bridge_revision}..HEAD")" != 1 ]]; then
    echo "Megatron Bridge FP8 import overlay must contain exactly one commit" >&2
    exit 2
  fi
  actual_patch_sha256=$(
    git -C "${bridge_root}" diff "${bridge_revision}..HEAD" -- \
      src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py \
      tests/unit_tests/models/glm_moe_dsa/test_glm5_bridge.py |
      sha256sum | cut -d' ' -f1
  )
  if [[ "${actual_patch_sha256}" != "${bridge_fp8_import_patch_sha256}" ]]; then
    echo "Megatron Bridge FP8 import patch drift: expected=${bridge_fp8_import_patch_sha256} actual=${actual_patch_sha256}" >&2
    exit 2
  fi
elif [[ "$(git -C "${bridge_root}" rev-parse HEAD)" != "${bridge_revision}" ]]; then
  echo "Megatron Bridge revision drift without FP8 import overlay" >&2
  exit 2
fi
if [[ -e "${root}/.venv" ]]; then
  echo "refusing to overwrite existing venv: ${root}/.venv" >&2
  exit 2
fi

"${python_seed}" -m venv "${root}/.venv"
python_bin="${root}/.venv/bin/python"
pip_bin="${root}/.venv/bin/pip"
"${pip_bin}" config --site set global.index-url https://pypi.yandex-team.ru/simple/
"${pip_bin}" config --site set global.extra-index-url https://pypi.org/simple/
"${pip_bin}" install --upgrade pip setuptools wheel
"${pip_bin}" install torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu130

export PATH="${HOME}/.cargo/bin:${root}/.venv/bin:${PATH}"
export MATURIN_PEP517_ARGS="--features vendored-openssl"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.18
export MAX_JOBS=${MAX_JOBS:-4}

# Resolve the four local GitHub checkouts together.  Dependency resolution is
# intentionally enabled; this is not a --no-deps overlay installation.
"${pip_bin}" install \
  "${root}/src/verl" \
  "${root}/src/sglang/python" \
  "${root}/src/Megatron-LM" \
  "${root}/src/slime"
"${pip_bin}" install --no-build-isolation \
  "transformer-engine[pytorch,core_cu13]==${transformer_engine_version}"
"${pip_bin}" install --extra-index-url https://pypi.nvidia.com \
  "nvidia-modelopt==${modelopt_version}"
"${pip_bin}" check

purelib=$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
nvidia_lib_dirs=()
shopt -s nullglob
for candidate in "${purelib}"/nvidia/*/lib; do
  if [[ -d "${candidate}" ]]; then
    nvidia_lib_dirs+=("${candidate}")
  fi
done
shopt -u nullglob
old_ifs=${IFS}
IFS=:
export LD_LIBRARY_PATH="${nvidia_lib_dirs[*]}"
IFS=${old_ifs}
unset LD_PRELOAD
export PYTHONPATH="${root}/src/Megatron-Bridge/src:${root}/src/Megatron-LM:${root}/src/verl"

"${python_bin}" - <<'PY'
import importlib.metadata
import torch
import transformers
import transformer_engine
import megatron.bridge

print("torch", torch.__version__)
print("transformers", transformers.__version__)
for distribution in ("verl", "sglang", "megatron-core", "slime"):
    print(distribution, importlib.metadata.version(distribution))
print("transformer-engine", transformer_engine.__version__)
print("megatron-bridge", megatron.bridge.__version__)
print("modelopt", importlib.metadata.version("nvidia-modelopt"))
from transformers import GlmMoeDsaForCausalLM  # noqa: F401

print("glm_moe_dsa import: OK")
PY

"${python_bin}" "${root}/src/verl/examples/glm52_lora/verify_bridge_contract.py"

for repository in verl sglang Megatron-LM slime Megatron-Bridge; do
  printf '%s %s\n' \
    "${repository}" \
    "$(git -C "${root}/src/${repository}" rev-parse HEAD)"
done | tee "${root}/resolved_revisions.txt"
