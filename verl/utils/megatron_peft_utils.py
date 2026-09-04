# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Utilities for PEFT (Parameter-Efficient Fine-Tuning) of Megatron in VERL."""


def _is_adapter_parameter(name: str) -> bool:
    normalized = name.lower()
    return "lora" in normalized or "adapter" in normalized


def _unwrapped_chunks(model):
    from verl.utils.megatron_utils import unwrap_model

    unwrapped = unwrap_model(model)
    return unwrapped if isinstance(unwrapped, list | tuple) else [unwrapped]


def summarize_peft_parameters(model) -> dict[str, int | list[str]]:
    """Summarize the post-wrap trainable set without double-counting shared parameters."""
    total_params = 0
    trainable_params = 0
    adapter_params = 0
    trainable_tensors = 0
    adapter_tensors = 0
    unexpected_trainable: list[str] = []
    seen: set[int] = set()

    for chunk_id, chunk in enumerate(_unwrapped_chunks(model)):
        for name, param in chunk.named_parameters():
            identity = id(param)
            if identity in seen:
                continue
            seen.add(identity)

            numel = param.numel()
            total_params += numel
            if not param.requires_grad:
                continue

            trainable_params += numel
            trainable_tensors += 1
            if _is_adapter_parameter(name):
                adapter_params += numel
                adapter_tensors += 1
            else:
                unexpected_trainable.append(f"chunk={chunk_id}:{name}")

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_tensors": trainable_tensors,
        "adapter_parameters": adapter_params,
        "adapter_tensors": adapter_tensors,
        "unexpected_trainable": unexpected_trainable,
    }


def validate_peft_trainable_parameters(model) -> dict[str, int | list[str]]:
    """Fail if distributed wrapping thawed the base model or removed all adapters."""
    summary = summarize_peft_parameters(model)
    unexpected = summary["unexpected_trainable"]
    if unexpected:
        raise RuntimeError(
            "PEFT invariant failed after distributed model construction: "
            f"non-adapter parameters are trainable: {unexpected[:8]}"
        )
    if summary["adapter_parameters"] == 0 or summary["adapter_tensors"] == 0:
        raise RuntimeError(
            "PEFT invariant failed after distributed model construction: no trainable adapter parameters remain"
        )
    if summary["trainable_parameters"] != summary["adapter_parameters"]:
        raise RuntimeError(
            "PEFT invariant failed after distributed model construction: "
            "the trainable parameter count differs from the adapter count"
        )
    return summary


def count_adapter_parameters(model):
    """Count the number of trainable adapter parameters.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (adapter_params, total_params, percentage)
    """
    summary = summarize_peft_parameters(model)
    adapter_params = summary["adapter_parameters"]
    total_params = summary["total_parameters"]
    percentage = 100 * adapter_params / total_params if total_params > 0 else 0

    return adapter_params, total_params, percentage


def print_adapter_info(model):
    """Print information about adapter parameters in the model."""
    summary = summarize_peft_parameters(model)
    adapter_params = summary["adapter_parameters"]
    total_params = summary["total_parameters"]
    percentage = 100 * adapter_params / total_params if total_params > 0 else 0

    print(f"\n{'=' * 60}")
    print("PEFT Adapter Information:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Adapter parameters:   {adapter_params:,}")
    print(f"  Trainable parameters: {summary['trainable_parameters']:,}")
    print(f"  Trainable percentage: {percentage:.2f}%")
    print(f"{'=' * 60}\n")


def build_peft_config_for_vllm(lora_config: dict) -> dict:
    """Build the ``peft_config`` every rollout backend receives, from megatron's LoRA config.

    Args:
        lora_config: Megatron lora configuration dictionary.

    Returns:
        A dict accepted by both vLLM's PEFTHelper.from_dict() and SGLang's adapter loader.
    """
    from peft import PeftType, TaskType

    return {
        "task_type": TaskType.CAUSAL_LM,
        "peft_type": PeftType.LORA,
        "r": lora_config.get("rank", 0),
        "lora_alpha": lora_config.get("alpha", 32),
        # vLLM doesn't really use target_modules to determine which modules
        # to apply LoRA to, so we set "all-linear" as a placeholder.
        "target_modules": "all-linear",
        "bias": "none",
        "lora_dropout": lora_config.get("dropout", 0.0),
    }


__all__ = [
    "count_adapter_parameters",
    "print_adapter_info",
    "summarize_peft_parameters",
    "validate_peft_trainable_parameters",
    "build_peft_config_for_vllm",
]
