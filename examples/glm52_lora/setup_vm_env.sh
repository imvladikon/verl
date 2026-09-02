#!/usr/bin/env bash
set -euo pipefail

# Create an isolated GLM-5.2 validation checkout and venv.  Existing projects,
# venvs, Ray clusters, and GPU processes are never modified.

root=${GLM52_VM_ROOT:-$HOME/glm52lora}
branch=${GLM_BRANCH:-glm-5.3-flash}
python_seed=${PYTHON_SEED:-python3.12}
phase=${GLM52_SETUP_PHASE:-all}
bridge_revision=${MEGATRON_BRIDGE_REVISION:-d0c6228a2a832f566dd44a3a179b3136613c11b7}
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
fi

if [[ "${phase}" == "clone" ]]; then
  echo "Clone phase complete below ${root}/src; apply the bounded validation overlay, then run with GLM52_SETUP_PHASE=install."
  exit 0
fi

for repository in verl sglang Megatron-LM slime Megatron-Bridge; do
  if [[ ! -d "${root}/src/${repository}/.git" ]]; then
    echo "missing cloned repository: ${root}/src/${repository}" >&2
    exit 2
  fi
done
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
