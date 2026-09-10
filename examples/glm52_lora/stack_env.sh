#!/usr/bin/env bash

# Shared runtime environment for the isolated GLM-5.2 validation stack.
# The caller must define repo_root and python_bin before sourcing this file.

: "${repo_root:?repo_root must be set before sourcing stack_env.sh}"
: "${python_bin:?python_bin must be set before sourcing stack_env.sh}"

stack_src=$(cd -- "${repo_root}/.." && pwd)
megatron_bridge_root=${MEGATRON_BRIDGE_ROOT:-${stack_src}/Megatron-Bridge}
megatron_lm_root=${MEGATRON_LM_ROOT:-${stack_src}/Megatron-LM}

if [[ ! -d "${megatron_bridge_root}/src/megatron/bridge" ]]; then
  echo "Megatron Bridge source not found: ${megatron_bridge_root}" >&2
  return 2
fi
if [[ ! -d "${megatron_lm_root}/megatron/core" ]]; then
  echo "Megatron-LM source not found: ${megatron_lm_root}" >&2
  return 2
fi

purelib=$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
nvidia_lib_dirs=()
shopt -s nullglob
for candidate in "${purelib}"/nvidia/*/lib; do
  if [[ -d "${candidate}" ]]; then
    nvidia_lib_dirs+=("${candidate}")
  fi
done
shopt -u nullglob
if (( ${#nvidia_lib_dirs[@]} == 0 )); then
  echo "no venv NVIDIA library directories found below ${purelib}" >&2
  return 2
fi

# Login shells on shared VMs may expose an older system NCCL.  Use only the
# CUDA libraries resolved with this venv; the dynamic loader still searches
# its normal system paths for libc and other non-CUDA libraries.
old_ifs=${IFS}
IFS=:
export LD_LIBRARY_PATH="${nvidia_lib_dirs[*]}"
IFS=${old_ifs}
unset LD_PRELOAD

python_dir=$(cd -- "$(dirname -- "${python_bin}")" && pwd)
export PATH="${python_dir}:${PATH}"
export PYTHONPATH="${megatron_bridge_root}/src:${megatron_lm_root}:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
