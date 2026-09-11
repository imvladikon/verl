# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Utility functions for the Automodel engine integration."""

import logging

import torch
import torch.distributed

from verl.utils.device import get_device_id, get_torch_device

logger = logging.getLogger(__name__)

GLM53_FLASH_MODEL_TYPE = "glm5_next"
GLM53_FLASH_ARCHITECTURE = "Glm5NextForConditionalGeneration"


def is_glm53_flash_config(hf_config) -> bool:
    """Match only the released GLM-5.3-Flash config contract."""
    model_type = getattr(hf_config, "model_type", None)
    architectures = getattr(hf_config, "architectures", None) or []
    has_flash_marker = model_type == GLM53_FLASH_MODEL_TYPE or GLM53_FLASH_ARCHITECTURE in architectures
    if has_flash_marker and not (model_type == GLM53_FLASH_MODEL_TYPE and architectures == [GLM53_FLASH_ARCHITECTURE]):
        raise ValueError(
            "Inconsistent GLM-5.3-Flash config: expected model_type='glm5_next' and exactly "
            f"architectures=['{GLM53_FLASH_ARCHITECTURE}'], got {model_type=!r}, {architectures=!r}"
        )
    return has_flash_marker


def get_dp_rank(device_mesh, include_cp=False):
    """Get data-parallel rank from device mesh."""
    if device_mesh is None:
        return 0
    if include_cp and "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1:
        return device_mesh.get_local_rank("dp_cp")
    return device_mesh.get_local_rank("dp")


def get_tp_rank(device_mesh):
    """Get tensor-parallel rank from device mesh."""
    if device_mesh is None or "tp" not in device_mesh.mesh_dim_names or device_mesh["tp"].size() == 1:
        return 0
    return device_mesh.get_local_rank("tp")


def get_pp_rank(device_mesh):
    """Get pipeline-parallel rank from device mesh."""
    if device_mesh is None or "pp" not in device_mesh.mesh_dim_names or device_mesh["pp"].size() == 1:
        return 0
    return device_mesh.get_local_rank("pp")


def get_dp_group_size(device_mesh, include_cp=False):
    """Get data-parallel group size from device mesh."""
    if device_mesh is None:
        return torch.distributed.get_world_size()
    if include_cp and "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1:
        return device_mesh["dp_cp"].size()
    if "dp" in device_mesh.mesh_dim_names:
        return device_mesh["dp"].size()
    return torch.distributed.get_world_size()


def maybe_fully_shard_optimizer(model, optimizer, distributed_config):
    """Call fully_shard_optimizer for MegatronFSDP strategy."""
    from nemo_automodel.components.distributed.config import MegatronFSDPConfig

    if isinstance(distributed_config, MegatronFSDPConfig) and torch.distributed.get_world_size() > 1:
        from megatron_fsdp.fully_shard import fully_shard_optimizer

        fully_shard_optimizer(model, optimizer)


def build_distributed_setup_from_engine_config(engine_config, world_size):
    """Build AutoModel's current single-object distributed contract.

    Args:
        engine_config: AutomodelEngineConfig instance.
        world_size: Total number of processes in the job.

    Returns:
        A resolved ``DistributedSetup`` containing policy and meshes.
    """
    from nemo_automodel.components.distributed.config import (
        DDPConfig,
        DistributedSetup,
        FSDP2Config,
        MegatronFSDPConfig,
        MoEParallelizerConfig,
    )
    from nemo_automodel.components.distributed.mesh import ParallelismSizes

    strategy = engine_config.distributed_strategy

    if strategy == "fsdp2":
        from torch.distributed.fsdp import MixedPrecisionPolicy

        from verl.utils.torch_dtypes import PrecisionType

        mp_policy = MixedPrecisionPolicy(
            param_dtype=PrecisionType.to_dtype(engine_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(engine_config.mp_reduce_dtype),
            output_dtype=PrecisionType.to_dtype(engine_config.mp_output_dtype),
            cast_forward_inputs=True,
        )

        distributed_config = FSDP2Config(
            sequence_parallel=engine_config.sequence_parallel,
            mp_policy=mp_policy,
            activation_checkpointing=engine_config.activation_checkpointing,
            defer_fsdp_grad_sync=engine_config.defer_fsdp_grad_sync,
            patch_is_packed_sequence=False,
            enable_compile=engine_config.enable_compile,
        )

    elif strategy == "megatron_fsdp":
        distributed_config = MegatronFSDPConfig(
            activation_checkpointing=engine_config.activation_checkpointing,
        )

    elif strategy == "ddp":
        distributed_config = DDPConfig(
            activation_checkpointing=engine_config.activation_checkpointing,
        )

    else:
        raise ValueError(f"Unsupported distributed_strategy: {strategy}")

    moe_parallel_config = None
    if engine_config.ep_size > 1:
        moe_kwargs = dict(engine_config.moe_config or {})
        if isinstance(distributed_config, FSDP2Config):
            moe_kwargs.setdefault("mp_policy", distributed_config.mp_policy)
        moe_parallel_config = MoEParallelizerConfig(**moe_kwargs)

    return DistributedSetup.build(
        strategy=distributed_config,
        parallelism_sizes=ParallelismSizes(
            tp_size=engine_config.tp_size,
            pp_size=engine_config.pp_size,
            cp_size=engine_config.cp_size,
            ep_size=engine_config.ep_size,
            dp_replicate_size=engine_config.dp_replicate_size,
        ),
        moe_parallel_config=moe_parallel_config,
        activation_checkpointing=engine_config.activation_checkpointing,
        world_size=world_size,
    )


def _validate_glm53_flash_engine(engine_config, backend_kwargs: dict) -> None:
    """Reject unqualified parallelism and state-dict settings for Flash."""
    if engine_config.tp_size != 1 or engine_config.pp_size != 1:
        raise ValueError("GLM-5.3-Flash AutoModel supports TP=1 and PP=1; use CP/EP for scale-out")
    if not backend_kwargs.get("enable_hf_state_dict_adapter", False):
        raise ValueError("GLM-5.3-Flash requires backend_config.enable_hf_state_dict_adapter=true")


def build_automodel_model(model_config, engine_config, distributed_setup):
    """Build a model through AutoModel's current ``DistributedSetup`` API."""
    from nemo_automodel._transformers import NeMoAutoModelForCausalLM, NeMoAutoModelForImageTextToText

    hf_config = model_config.hf_config
    is_glm53_flash = is_glm53_flash_config(hf_config)
    backend_kwargs = dict(engine_config.backend_config or {})
    if is_glm53_flash:
        _validate_glm53_flash_engine(engine_config, backend_kwargs)

    kwargs = {
        "attn_implementation": engine_config.attn_implementation,
        "distributed_setup": distributed_setup,
        "has_packed_sequence": bool(model_config.use_remove_padding),
        "trust_remote_code": model_config.trust_remote_code,
    }

    if engine_config.enable_fp8:
        from nemo_automodel.components.quantization.fp8 import FP8Config

        kwargs["fp8_config"] = FP8Config()

    if engine_config.enable_compile:
        from nemo_automodel.components.utils.compile_utils import CompileConfig

        kwargs["compile_config"] = CompileConfig()

    model_path = model_config.local_path or model_config.path

    # Qwen/Llama with ep_size<=1: use HF implementation.
    from transformers import AutoConfig

    _cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
    _arch = (getattr(_cfg, "architectures", None) or [""])[0].lower()
    if not is_glm53_flash and engine_config.ep_size <= 1 and ("qwen" in _arch or "llama" in _arch):
        kwargs["force_hf"] = True

    if backend_kwargs and not kwargs.get("force_hf", False):
        from nemo_automodel.components.models.common import BackendConfig

        kwargs["backend"] = BackendConfig(**backend_kwargs)

    from verl.utils.torch_dtypes import PrecisionType

    kwargs["torch_dtype"] = PrecisionType.to_dtype(engine_config.model_dtype)

    auto_model_cls = NeMoAutoModelForCausalLM
    if is_glm53_flash:
        auto_model_cls = NeMoAutoModelForImageTextToText
        kwargs.update(
            {
                # AutoModel's custom-model loader resolves its native config
                # from the checkpoint and deep-merges this nested override.
                # Passing a finished ``config=`` object currently duplicates
                # the constructor's positional config argument.
                "text_config": {
                    "num_nextn_predict_layers": 0,
                    "output_hidden_states": True,
                },
                "freeze_config": {
                    "freeze_vision_tower": bool(getattr(model_config, "freeze_vision_tower", False)),
                    "freeze_audio_tower": True,
                    "freeze_language_model": False,
                },
                "use_liger_kernel": False,
                "use_sdpa_patching": False,
            }
        )
        if kwargs["torch_dtype"] is not torch.bfloat16:
            logger.warning(
                "GLM-5.3-Flash production training was qualified with BF16 checkpoint dequantization; got %s",
                kwargs["torch_dtype"],
            )

    return auto_model_cls.from_pretrained(pretrained_model_name_or_path=model_path, **kwargs)


@torch.no_grad()
def offload_automodel_model_to_cpu(model, empty_cache=True):
    """Offload an FSDP2-wrapped model to CPU (reshard, move to CPU, optional cache clear)."""
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

    for module in model.modules():
        state = _get_module_fsdp_state(module)
        if state is None:
            continue
        fsdp_param_group = state._fsdp_param_group

        if fsdp_param_group is None:
            continue

        fsdp_param_group._training_state = TrainingState.IDLE

    model.reshard()
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def load_automodel_model_to_gpu(model):
    """Load model back to GPU."""
    device = get_device_id()
    model.to(device, non_blocking=True)


@torch.no_grad()
def offload_automodel_optimizer(optimizer):
    """Offload optimizer state to CPU."""
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_automodel_optimizer(optimizer, device_id):
    """Load optimizer state back to GPU."""
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)
