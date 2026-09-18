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
import logging
import os

logger = logging.getLogger(__name__)

# What the adapter says about itself, and where a checkpoint keeps it. `adapter_path` points at the
# weights the run loads, which on the Megatron path is the torch_dist shard directory -- the claim
# lives beside it, not in it, so looking only inside makes the check a no-op exactly where it is
# needed. Both files spell the fields the same way (`r`, `lora_alpha`), so one reader covers them.
_CLAIM_LOCATIONS = (
    ("adapter_config.json",),  # an HF adapter directory, given directly
    ("..", "huggingface", "adapter", "adapter_config.json"),  # given <ckpt>/model/dist_ckpt
    ("model", "huggingface", "adapter", "adapter_config.json"),  # given the checkpoint root
    ("lora_train_meta.json",),  # the checkpoint root's own record
    ("..", "..", "lora_train_meta.json"),  # again from <ckpt>/model/dist_ckpt
    ("..", "lora_train_meta.json"),  # from <ckpt>/model
)


def _adapter_path_of(model_config) -> str | None:
    """Where the adapter being resumed lives, from whichever block carries it."""
    lora_cfg = getattr(model_config, "lora", None) or {}
    return lora_cfg.get("adapter_path", None) or getattr(model_config, "lora_adapter_path", None)


def adapter_claim(adapter_path: str) -> tuple[str, dict] | None:
    """The first record of what this adapter was trained with, and where it was found."""
    base = os.path.expanduser(str(adapter_path))
    for parts in _CLAIM_LOCATIONS:
        candidate = os.path.normpath(os.path.join(base, *parts))
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, encoding="utf-8") as handle:
                return candidate, json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Cannot read the LoRA record at %s: %s", candidate, error)
    return None


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
    found = adapter_claim(adapter_path)
    if found is None:
        # Worth saying out loud: the run is resuming an adapter that records nothing about how it
        # was trained, so a rescaling mismatch here stays undetectable.
        logger.warning(
            "No adapter_config.json or lora_train_meta.json found for %s, so this run's LoRA "
            "rank/alpha cannot be checked against the adapter's own.",
            adapter_path,
        )
        return
    config_path, saved = found

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
