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
"""PEFT configuration of Megatron for verl."""

import json
import os


def _adapter_path_of(model_config) -> str | None:
    """Where the adapter being resumed lives, from whichever block carries it."""
    lora_cfg = getattr(model_config, "lora", None) or {}
    return lora_cfg.get("adapter_path", None) or getattr(model_config, "lora_adapter_path", None)


def check_adapter_matches_config(model_config, rank: int, alpha: int) -> None:
    """Refuse an adapter whose own rank or alpha differs from the one this run will apply.

    LoRA's contribution is scaled by ``alpha / rank`` taken from the *config*, not from the adapter,
    so resuming an adapter trained at alpha 8 into a run configured for alpha 32 quietly multiplies
    every adapter output by four. Nothing downstream notices: the shapes match, the load reports the
    expected number of keys, and the run trains on a silently rescaled policy.

    ``verl/utils/model.py`` has read ``r`` back from ``adapter_config.json`` since before this, but
    nothing ever called it, and it never looked at ``lora_alpha`` at all.
    """
    adapter_path = _adapter_path_of(model_config)
    if not adapter_path:
        return
    config_path = os.path.join(os.path.expanduser(str(adapter_path)), "adapter_config.json")
    if not os.path.exists(config_path):
        return  # a bare weights directory carries no claim to check against
    with open(config_path, encoding="utf-8") as handle:
        saved = json.load(handle)

    mismatched = {
        name: (theirs, ours)
        for name, theirs, ours in (
            ("rank", saved.get("r"), rank),
            ("alpha", saved.get("lora_alpha"), alpha),
        )
        if theirs is not None and int(theirs) != int(ours)
    }
    if mismatched:
        detail = ", ".join(
            f"{name}: adapter has {theirs}, this run applies {ours}" for name, (theirs, ours) in mismatched.items()
        )
        raise ValueError(
            f"{config_path} does not match this run's LoRA config ({detail}). LoRA is scaled by "
            "alpha/rank from the config, so loading it anyway rescales every adapter output "
            "instead of failing. Fix the config to match the adapter, or point at another adapter."
        )


def get_peft_cls(model_config, bridge, provider, dtype=None):
    """Create a Megatron-Bridge PEFT object from ``model_config.lora``."""
    if not hasattr(model_config, "lora"):
        return None

    lora_cfg = model_config.lora
    if lora_cfg.get("rank", 0) <= 0:
        return None

    assert bridge is not None and provider is not None, "LoRA/PEFT only supported via Megatron-Bridge"

    check_adapter_matches_config(
        model_config, rank=int(lora_cfg.get("rank", 0)), alpha=int(lora_cfg.get("alpha", 0) or 0)
    )

    from megatron.bridge.peft.utils import create_peft

    peft_cls = create_peft(lora_cfg, dtype=dtype)
    print(
        f"Enabling {lora_cfg.get('type', 'lora').upper()} with rank={lora_cfg.get('rank')}, "
        f"alpha={lora_cfg.get('alpha')}, dropout={lora_cfg.get('dropout')}"
    )
    return peft_cls


__all__ = [
    "check_adapter_matches_config",
    "get_peft_cls",
]
