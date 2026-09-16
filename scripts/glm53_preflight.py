# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preflight checks for GLM-5.3-Flash runs on the glm-5.x forks.

Every check here exists because that exact thing bit a run: a stale image whose
sglang predates a fix, dense attention silently replacing DSA, an indexer kernel
that cannot serve the model's head count, a checkpoint missing config keys,
``ninja`` outside ``PATH`` so Megatron's zero-copy checkpoint path fails to build.

Run it on the job's own node, with the job's own interpreter, before training::

    python scripts/glm53_preflight.py                       # environment + fork fixes
    python scripts/glm53_preflight.py --kernels             # also run the GPU kernels
    python scripts/glm53_preflight.py --model /path/to/hf   # also check a checkpoint

Exit code is 1 when a check FAILs, so it can gate a launcher.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Callable

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"

_MARK = {PASS: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]", SKIP: "[skip]"}


@dataclass
class Result:
    name: str
    status: str
    detail: str
    hint: str = ""


@dataclass
class Registry:
    checks: list[tuple[str, str, Callable[[argparse.Namespace], Result]]] = field(default_factory=list)

    def add(self, group: str, name: str):
        def wrapper(fn):
            self.checks.append((group, name, fn))
            return fn

        return wrapper


registry = Registry()


# Why a lookup came back empty, keyed by what was looked up. A check that swallows an
# exception must still be able to say what went wrong.
_ERRORS: dict[str, str] = {}


def _describe_error(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def _why(key: str) -> str:
    """The recorded failure for a lookup, ready to append to a detail line."""
    reason = _ERRORS.get(key)
    return f" ({reason})" if reason else ""


def _version(package: str) -> str | None:
    try:
        import importlib.metadata as metadata

        return metadata.version(package)
    except Exception as error:
        _ERRORS[f"version:{package}"] = _describe_error(error)
        return None


def _module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as error:
        _ERRORS[name] = _describe_error(error)
        return None


def _source_of(obj) -> str:
    try:
        return inspect.getsource(obj)
    except Exception as error:
        _ERRORS[f"source:{obj!r}"] = _describe_error(error)
        return ""


def _git_commit(module_name: str) -> str:
    """Short commit of the checkout a module is imported from, when it is a git checkout."""
    import subprocess

    module = _module(module_name)
    if module is None:
        return f"not imported{_why(module_name)}"
    if not getattr(module, "__file__", None):
        return "no file"
    directory = os.path.dirname(module.__file__)
    try:
        out = subprocess.run(
            ["git", "-C", directory, "rev-parse", "--short=9", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "not-a-checkout"
    except Exception as error:
        return _describe_error(error)


# --------------------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------------------


@registry.add("runtime", "torch + devices")
def _check_torch(args) -> Result:
    torch = _module("torch")
    if torch is None:
        return Result("torch + devices", FAIL, f"torch is not importable{_why('torch')}")
    if not torch.cuda.is_available():
        return Result("torch + devices", WARN, f"torch {torch.__version__}, no CUDA device visible")
    names = {torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
    major, minor = torch.cuda.get_device_capability()
    detail = (
        f"torch {torch.__version__} (cuda {torch.version.cuda}), "
        f"{torch.cuda.device_count()}x {'/'.join(sorted(names))}, SM{major}{minor}"
    )
    if major < 9:
        return Result(
            "torch + devices",
            WARN,
            detail,
            "fa3, DeepGEMM MQA logits and the TileLang FP8 GEMM need SM90; DSA falls back to torch here",
        )
    return Result("torch + devices", PASS, detail)


@registry.add("runtime", "sglang stack")
def _check_sglang_stack(args) -> Result:
    sglang = _module("sglang")
    if sglang is None:
        return Result("sglang stack", FAIL, f"sglang is not importable{_why('sglang')}")
    kernel_module = _module("sgl_kernel")
    kernel_version = _version("sgl-kernel") or getattr(kernel_module, "__version__", None)
    parts = [
        f"sglang {getattr(sglang, '__version__', '?')} @ {_git_commit('sglang')}",
        f"sgl-kernel {kernel_version or 'missing'}",
        f"sgl-deep-gemm {_version('sgl-deep-gemm') or 'missing'}",
        f"tilelang {_version('tilelang') or 'missing'}",
        f"flash-linear-attention {_version('flash-linear-attention') or 'missing'}",
    ]
    detail = ", ".join(parts)
    skip_check = os.environ.get("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "")
    if kernel_version is not None and kernel_version < "0.4.7" and skip_check not in ("1", "true", "True"):
        return Result(
            "sglang stack",
            FAIL,
            detail,
            "sglang requires sgl-kernel >= 0.4.7; install it or set SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 "
            "(only for probes, never for a real run)",
        )
    return Result("sglang stack", PASS, detail)


@registry.add("runtime", "training stack")
def _check_training_stack(args) -> Result:
    mcore = _module("megatron.core")
    bridge = _module("megatron.bridge")
    verl = _module("verl")
    detail = (
        f"megatron.core {getattr(mcore, '__version__', 'missing')} @ {_git_commit('megatron.core')}, "
        f"megatron.bridge {getattr(bridge, '__version__', 'missing')} @ {_git_commit('megatron.bridge')}, "
        f"verl @ {_git_commit('verl')}, "
        f"transformers {_version('transformers') or 'missing'}"
    )
    if mcore is None or verl is None:
        return Result("training stack", FAIL, detail, "megatron.core and verl must be importable")
    return Result("training stack", PASS, detail)


@registry.add("runtime", "ninja on PATH")
def _check_ninja(args) -> Result:
    """Megatron's zero-copy CheckpointWithoutOutput builds a C++ extension with load_inline."""
    binary = shutil.which("ninja")
    if binary:
        return Result("ninja on PATH", PASS, binary)
    installed = importlib.util.find_spec("ninja") is not None
    hint = (
        "the ninja package is installed but its binary is not on PATH: call the venv's bin first "
        "(PATH=<venv>/bin:$PATH), otherwise CheckpointWithoutOutput fails to compile share_storage"
        if installed
        else "install ninja; without it Megatron's zero-copy recompute path cannot build"
    )
    return Result("ninja on PATH", FAIL, "not found", hint)


@registry.add("runtime", "CUDA_VISIBLE_DEVICES format")
def _check_cvd(args) -> Result:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not value:
        return Result("CUDA_VISIBLE_DEVICES format", SKIP, "unset")
    if value.startswith("GPU-"):
        return Result(
            "CUDA_VISIBLE_DEVICES format",
            WARN,
            value.split(",")[0][:20] + "...",
            "sglang's test utilities parse this as an integer; pass device indices when running its tests",
        )
    return Result("CUDA_VISIBLE_DEVICES format", PASS, value)


# --------------------------------------------------------------------------------------
# fork fixes: catches a stale image or a partial overlay
# --------------------------------------------------------------------------------------


@registry.add("fork fixes", "sglang refuses dense attention for DSA")
def _check_dense_guard(args) -> Result:
    module = _module("sglang.srt.arg_groups.model_overrides.deepseek_v2")
    if module is None:
        return Result(
            "sglang refuses dense attention for DSA",
            FAIL,
            f"override module not importable{_why('sglang.srt.arg_groups.model_overrides.deepseek_v2')}",
        )
    if not hasattr(module, "_check_dense_attention_for_dsa"):
        return Result(
            "sglang refuses dense attention for DSA",
            FAIL,
            "guard missing",
            "this sglang predates the fix: an explicit attention_backend silently runs dense MLA and "
            "ignores every --dsa-* option",
        )
    return Result("sglang refuses dense attention for DSA", PASS, "guard present")


@registry.add("fork fixes", "sglang DSA backend resolution")
def _check_dsa_resolution(args) -> Result:
    module = _module("sglang.srt.arg_groups.overrides")
    if module is None:
        return Result(
            "sglang DSA backend resolution",
            FAIL,
            f"overrides module not importable{_why('sglang.srt.arg_groups.overrides')}",
        )
    missing = [
        name
        for name in (
            "_resolve_kpool_sparse_prefill",
            "_check_explicit_torch_dsa_backends",
            "_check_compact_kpool_indexer",
        )
        if not hasattr(module, name)
    ]
    if missing:
        return Result(
            "sglang DSA backend resolution",
            FAIL,
            f"missing {', '.join(missing)}",
            "KPool prefill would be swapped at runtime and a torch DSA path could be selected silently",
        )
    return Result("sglang DSA backend resolution", PASS, "kpool + torch + compact-indexer guards present")


@registry.add("fork fixes", "TileLang indexer head padding")
def _check_tilelang_padding(args) -> Result:
    module = _module("sglang.kernels.ops.attention.dsa.tilelang_kernel")
    if module is None:
        return Result(
            "TileLang indexer head padding",
            SKIP,
            f"tilelang kernels not importable{_why('sglang.kernels.ops.attention.dsa.tilelang_kernel')}",
        )
    source = _source_of(getattr(module, "tilelang_fp8_paged_mqa_logits", None))
    if "padded_heads" not in source:
        return Result(
            "TileLang indexer head padding",
            FAIL,
            "padding missing",
            "CUDA graph capture fails for indexer head counts that are not a multiple of 8 "
            "(Flash-8B has 4): 'N must be divisible by 8'",
        )
    return Result("TileLang indexer head padding", PASS, "heads padded to a multiple of 8")


@registry.add("fork fixes", "verl leaves DSA attention to sglang")
def _check_verl_dsa(args) -> Result:
    module = _module("verl.workers.rollout.sglang_rollout.async_sglang_server")
    if module is None:
        return Result(
            "verl leaves DSA attention to sglang",
            FAIL,
            f"async server module not importable{_why('verl.workers.rollout.sglang_rollout.async_sglang_server')}",
        )
    missing = [name for name in ("uses_dsa_attention", "describe_sglang_backends") if not hasattr(module, name)]
    if missing:
        return Result(
            "verl leaves DSA attention to sglang",
            FAIL,
            f"missing {', '.join(missing)}",
            "this verl fills attention_backend with fa3/flashinfer, so a DSA model runs dense attention",
        )
    return Result("verl leaves DSA attention to sglang", PASS, "DSA models keep the sparse backend")


@registry.add("fork fixes", "rollout.server config node")
def _check_rollout_server_config(args) -> Result:
    try:
        from verl.workers.config import RolloutConfig, ServerConfig
    except Exception as error:  # pragma: no cover - depends on the installed verl
        return Result("rollout.server config node", FAIL, f"import failed: {_describe_error(error)}")
    config = RolloutConfig.__dataclass_fields__.get("server")
    if config is None or not hasattr(ServerConfig, "generation_timeout"):
        return Result("rollout.server config node", FAIL, "server config missing")
    return Result(
        "rollout.server config node",
        PASS,
        "present",
        "override it only together with _target_=verl.workers.config.ServerConfig, or config.server "
        "becomes a plain dict and generation fails on the first request",
    )


@registry.add("fork fixes", "megatron raw-MLP layernorm recompute")
def _check_raw_mlp_recompute(args) -> Result:
    module = _module("megatron.core.transformer.transformer_layer")
    if module is None:
        return Result(
            "megatron raw-MLP layernorm recompute",
            SKIP,
            f"transformer_layer not importable{_why('megatron.core.transformer.transformer_layer')}",
        )
    source = _source_of(getattr(module.TransformerLayer, "_forward_mlp_output_with_bias", None))
    if "discard_output_and_register_recompute" not in source:
        return Result(
            "megatron raw-MLP layernorm recompute",
            WARN,
            "discard missing",
            'recompute_modules=["layernorm"] is a no-op on mHC layers: the norm activation stays resident',
        )
    return Result("megatron raw-MLP layernorm recompute", PASS, "discard registered on the raw path")


@registry.add("fork fixes", "megatron recompute with frozen inputs")
def _check_frozen_input_recompute(args) -> Result:
    module = _module("megatron.core.tensor_parallel.random")
    if module is None:
        return Result(
            "megatron recompute with frozen inputs",
            SKIP,
            f"random module not importable{_why('megatron.core.tensor_parallel.random')}",
        )
    source = _source_of(module)
    if "_tensor_args_without_grad" not in source:
        return Result(
            "megatron recompute with frozen inputs",
            FAIL,
            "fix missing",
            "under LoRA a checkpoint whose inputs are all frozen loses its saved tensors and backward "
            "raises StopIteration",
        )
    return Result("megatron recompute with frozen inputs", PASS, "frozen-input checkpoints keep their args")


@registry.add("fork fixes", "fla autotune determinism")
def _check_fla_autotune(args) -> Result:
    module = _module("megatron.core.ssm.fla_autotune")
    if module is None:
        return Result(
            "fla autotune determinism",
            FAIL,
            f"megatron.core.ssm.fla_autotune missing{_why('megatron.core.ssm.fla_autotune')}",
            "without it the Triton autotuner picks different block sizes per run and results drift",
        )
    requested = module.fla_autotune_pinning_requested()
    pinned = os.environ.get("MCORE_FLA_FIXED_AUTOTUNE_META", "<unset>")
    detail = f"MCORE_FLA_FIXED_AUTOTUNE_META={pinned}, pinning={requested}"
    if not requested:
        return Result(
            "fla autotune determinism",
            WARN,
            detail,
            "set MCORE_FLA_FIXED_AUTOTUNE_META=1 (or enable torch deterministic mode) for run-to-run "
            "reproducibility of KDA layers",
        )
    return Result("fla autotune determinism", PASS, detail)


# --------------------------------------------------------------------------------------
# runtime environment consistency
# --------------------------------------------------------------------------------------


@registry.add("env", "DSA env flags")
def _check_dsa_env(args) -> Result:
    dense = os.environ.get("SGLANG_DSA_ALLOW_DENSE_ATTENTION", "")
    torch_fallback = os.environ.get("SGLANG_DSA_ALLOW_TORCH_FALLBACK", "")
    fuse_topk = os.environ.get("SGLANG_DSA_FUSE_TOPK", "<default>")
    detail = (
        f"ALLOW_DENSE_ATTENTION={dense or '<unset>'}, ALLOW_TORCH_FALLBACK={torch_fallback or '<unset>'}, "
        f"FUSE_TOPK={fuse_topk}"
    )
    enabled = {"1", "true", "True"}
    if dense in enabled:
        return Result(
            "DSA env flags",
            WARN,
            detail,
            "dense attention is explicitly allowed: the indexer and every --dsa-* option are ignored",
        )
    if torch_fallback in enabled:
        return Result(
            "DSA env flags",
            WARN,
            detail,
            "the eager torch DSA reference is allowed: correct but far slower than the fused kernels",
        )
    return Result("DSA env flags", PASS, detail)


@registry.add("env", "CUDA graph memory saver")
def _check_memory_saver(args) -> Result:
    value = os.environ.get("SGLANG_MEMORY_SAVER_CUDA_GRAPH", "<unset>")
    return Result(
        "CUDA graph memory saver",
        PASS if value != "<unset>" else WARN,
        f"SGLANG_MEMORY_SAVER_CUDA_GRAPH={value}",
        ""
        if value != "<unset>"
        else "with CUDA graphs on, graph pools are not released while the "
        "rollout sleeps; set it to 1 if a colocated actor hits OOM on wake-up",
    )


# --------------------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------------------


@registry.add("model", "checkpoint config")
def _check_model_config(args) -> Result:
    if not args.model:
        return Result("checkpoint config", SKIP, "pass --model to check a checkpoint")
    import glob

    config_path = os.path.join(args.model, "config.json")
    if not os.path.exists(config_path):
        return Result("checkpoint config", FAIL, f"{config_path} not found")
    with open(config_path) as handle:
        config = json.load(handle)
    text = config.get("text_config", config)
    keys = ("index_topk", "index_kpool", "index_head_dim", "index_kpool_compress", "index_kpool_always_select_tail")
    values = {key: text.get(key, "<absent>") for key in keys}
    detail = f"{config.get('architectures')} {values}"
    missing = [key for key, value in values.items() if value == "<absent>"]
    if text.get("index_topk") is None:
        return Result("checkpoint config", PASS, detail + " (not a DSA model)")
    if missing:
        has_weights = any(
            "index_kpool_compress" in name
            for path in glob.glob(os.path.join(args.model, "*.index.json"))
            for name in json.load(open(path)).get("weight_map", {})
        )
        return Result(
            "checkpoint config",
            FAIL,
            detail,
            f"missing {', '.join(missing)}: Megatron-Bridge refuses such a config, and sglang's indexer "
            f"raises AttributeError"
            + (" (the weights do carry kpool compression, so the keys should be true)" if has_weights else ""),
        )
    if values["index_head_dim"] != 128:
        return Result(
            "checkpoint config",
            WARN,
            detail,
            "only the torch reference indexer supports this head dim: all four dsa_* backends must be torch",
        )
    return Result("checkpoint config", PASS, detail)


@registry.add("model", "multimodal processor")
def _check_processor(args) -> Result:
    if not args.model:
        return Result("multimodal processor", SKIP, "pass --model to check a checkpoint")
    present = [
        name
        for name in ("processor_config.json", "preprocessor_config.json")
        if os.path.exists(os.path.join(args.model, name))
    ]
    if present:
        return Result("multimodal processor", PASS, ", ".join(present))
    return Result(
        "multimodal processor",
        WARN,
        "absent",
        "GLM-5.3 configs look multimodal, so AutoProcessor raises: pass enable_multimodal=False to the engine",
    )


# --------------------------------------------------------------------------------------
# kernels (opt-in, needs a GPU)
# --------------------------------------------------------------------------------------


@registry.add("kernels", "load_inline (C++ extension build)")
def _check_load_inline(args) -> Result:
    if not args.kernels:
        return Result("load_inline (C++ extension build)", SKIP, "pass --kernels")
    torch = _module("torch")
    if torch is None:
        return Result("load_inline (C++ extension build)", FAIL, f"torch missing{_why('torch')}")
    try:
        from torch.utils.cpp_extension import load_inline

        module = load_inline(
            name="glm53_preflight_probe",
            cpp_sources="int probe() { return 7; }",
            functions=["probe"],
            verbose=False,
        )
        return Result("load_inline (C++ extension build)", PASS, f"built, probe()={module.probe()}")
    except Exception as error:
        return Result(
            "load_inline (C++ extension build)",
            FAIL,
            _describe_error(error)[:400],
            "Megatron's zero-copy CheckpointWithoutOutput compiles share_storage this way at first use",
        )


@registry.add("kernels", "DSA indexer MQA logits")
def _check_indexer_kernels(args) -> Result:
    if not args.kernels:
        return Result("DSA indexer MQA logits", SKIP, "pass --kernels")
    torch = _module("torch")
    if torch is None or not torch.cuda.is_available():
        return Result("DSA indexer MQA logits", SKIP, "no CUDA device")
    major = torch.cuda.get_device_capability()[0]
    findings = []
    deep_gemm = _module("deep_gemm")
    findings.append(
        "deep_gemm.fp8_paged_mqa_logits "
        + ("ok" if deep_gemm is not None and hasattr(deep_gemm, "fp8_paged_mqa_logits") else "missing")
    )
    if major != 9:
        return Result(
            "DSA indexer MQA logits",
            SKIP,
            f"SM{major}x: the fused indexer kernels are Hopper-only ({', '.join(findings)})",
        )

    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_fp8_paged_mqa_logits
    from sglang.srt.layers.attention.dsa.torch_dsa_fallback import fp8_paged_mqa_logits_torch_dsa

    page_size, head_dim, pages, heads = 64, 128, 4, 4
    raw = torch.zeros((pages, page_size * (head_dim + 4)), dtype=torch.uint8, device="cuda")
    keys = torch.randn(pages, page_size, head_dim, device="cuda").clamp(-2, 2).to(torch.float8_e4m3fn)
    scales = torch.rand(pages, page_size, device="cuda") * 0.2 + 0.05
    raw[:, : page_size * head_dim] = keys.reshape(pages, -1).view(torch.uint8)
    raw[:, page_size * head_dim :] = scales.contiguous().view(torch.uint8)
    kv = raw.view(pages, page_size, 1, head_dim + 4)
    query = torch.randn(2, 1, heads, head_dim, device="cuda").clamp(-2, 2).to(torch.float8_e4m3fn)
    weights = torch.randn(2, heads, device="cuda")
    seq_lens = torch.tensor([100, 64], dtype=torch.int32, device="cuda")
    page_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda")
    max_len = 2 * page_size
    try:
        actual = tilelang_fp8_paged_mqa_logits(
            query, kv, weights, seq_lens, page_table, None, max_len, clean_logits=False
        )
        expected = fp8_paged_mqa_logits_torch_dsa(
            query,
            kv,
            weights,
            seq_lens,
            page_table,
            None,
            max_len,
            kv_chunk_tokens=4096,
            clean_logits=False,
        )
        error = max(
            (actual[row, :length] - expected[row, :length]).abs().max().item()
            for row, length in enumerate(seq_lens.tolist())
        )
        scale = expected.abs().max().item()
        findings.append(f"tilelang heads={heads} max_abs_err={error:.2e} (scale {scale:.2f})")
        if error > 5e-3 * max(scale, 1.0):
            return Result("DSA indexer MQA logits", FAIL, ", ".join(findings), "TileLang disagrees with the reference")
        return Result("DSA indexer MQA logits", PASS, ", ".join(findings))
    except Exception as error:
        return Result(
            "DSA indexer MQA logits",
            FAIL,
            f"{', '.join(findings)}; tilelang raised {_describe_error(error)[:400]}",
            "an indexer head count that is not a multiple of 8 needs the padding fix",
        )


@registry.add("kernels", "raw-MLP layernorm recompute behaviour")
def _check_recompute_behaviour(args) -> Result:
    if not args.kernels:
        return Result("raw-MLP layernorm recompute behaviour", SKIP, "pass --kernels")
    torch = _module("torch")
    if torch is None or not torch.cuda.is_available():
        return Result("raw-MLP layernorm recompute behaviour", SKIP, "no CUDA device")
    try:
        from megatron.core import parallel_state
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core.transformer.transformer_layer import TransformerLayer

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="gloo",
                init_method="tcp://127.0.0.1:%d" % (29500 + os.getpid() % 1000),  # noqa: UP031
                world_size=1,
                rank=0,
            )
        parallel_state.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            recompute_granularity="selective",
            recompute_modules=["layernorm"],
        )
        layer = TransformerLayer(config, get_gpt_layer_local_spec().submodules, layer_number=1).cuda()
        layer.train()
        hidden_states = torch.randn(8, 2, config.hidden_size, device="cuda", requires_grad=True)
        (mlp_output, _bias), _residual = layer._forward_mlp_output_with_bias(hidden_states)
        resident = layer.pre_mlp_norm_checkpoint.outputs[0].untyped_storage().size()
        mlp_output.float().sum().backward()
        parallel_state.destroy_model_parallel()
        if resident != 0:
            return Result(
                "raw-MLP layernorm recompute behaviour",
                WARN,
                f"{resident} bytes still resident",
                'recompute_modules=["layernorm"] saves nothing on mHC layers in this build',
            )
        return Result("raw-MLP layernorm recompute behaviour", PASS, "norm activation discarded and recomputed")
    except Exception as error:
        return Result(
            "raw-MLP layernorm recompute behaviour",
            WARN,
            f"probe failed: {_describe_error(error)[:400]}",
        )


# --------------------------------------------------------------------------------------
# resolved backends, as the rollout will see them
# --------------------------------------------------------------------------------------


@registry.add("rollout", "resolved sglang backends")
def _check_resolved_backends(args) -> Result:
    if not args.model:
        return Result("resolved sglang backends", SKIP, "pass --model to resolve backends")
    try:
        from sglang.srt.server_args import ServerArgs

        from verl.workers.rollout.sglang_rollout.async_sglang_server import (
            describe_sglang_backends,
        )
    except Exception as error:
        return Result("resolved sglang backends", FAIL, f"import failed: {_describe_error(error)}")
    try:
        kwargs = dict(model_path=args.model, tp_size=1, dtype="bfloat16", skip_tokenizer_init=True)
        if args.attention_backend:
            kwargs["attention_backend"] = args.attention_backend
        server_args = ServerArgs(**kwargs)
        # Backends are decided by the resolution pipeline, which the engine runs at launch.
        server_args.resolve_once()
        detail = describe_sglang_backends(server_args)
        if "attention_backend=dsa" not in detail and "index_topk" in _model_text_config(args.model):
            return Result(
                "resolved sglang backends",
                FAIL,
                detail,
                "a DSA model resolved to a non-DSA attention backend: the rollout would run dense attention",
            )
        return Result("resolved sglang backends", PASS, detail)
    except Exception as error:
        return Result(
            "resolved sglang backends",
            FAIL,
            _describe_error(error)[:400],
            "the server args this run would use are rejected; fix them before launching",
        )


def _model_text_config(model_path: str) -> str:
    try:
        with open(os.path.join(model_path, "config.json")) as handle:
            return handle.read()
    except Exception:
        return ""


# --------------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> list[Result]:
    results = []
    current_group = None
    for group, name, check in registry.checks:
        if args.group and group not in args.group:
            continue
        try:
            result = check(args)
        except Exception as error:  # a check must never take the launcher down
            result = Result(name, FAIL, f"check raised {_describe_error(error)}")
        results.append(result)
        if not args.json:
            if group != current_group:
                print(f"\n== {group}")
                current_group = group
            print(f"{_MARK[result.status]} {result.name}: {result.detail}")
            if result.hint and result.status != PASS:
                print(f"       -> {result.hint}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="HF checkpoint directory to check and to resolve backends for")
    parser.add_argument("--kernels", action="store_true", help="run the GPU kernel probes (slower)")
    parser.add_argument("--attention-backend", help="attention backend the run would pass to sglang")
    parser.add_argument("--group", action="append", help="only run these groups (repeatable)")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    args = parser.parse_args()

    results = run(args)
    failures = [r for r in results if r.status == FAIL]
    warnings = [r for r in results if r.status == WARN]
    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
    else:
        print(
            f"\n{len(results)} checks: {len(results) - len(failures) - len(warnings)} ok, "
            f"{len(warnings)} warnings, {len(failures)} failed"
        )
        for result in failures:
            print(f"  FAIL {result.name}: {result.detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
