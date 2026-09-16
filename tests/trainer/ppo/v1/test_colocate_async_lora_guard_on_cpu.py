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
"""Async rollout with a separately served LoRA adapter deadlocks the weight sync."""

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_colocate_async import reject_unmerged_lora_adapter


def _config(**model_fields):
    return OmegaConf.create({"actor_rollout_ref": {"model": {"lora_rank": 0, "lora": {}, **model_fields}}})


def test_unmerged_adapter_is_rejected_with_the_reason():
    with pytest.raises(ValueError, match="lora.merge=true"):
        reject_unmerged_lora_adapter(_config(lora_rank=16))
    with pytest.raises(ValueError, match="rank=8"):
        reject_unmerged_lora_adapter(_config(lora={"rank": 8, "merge": False}))


def test_merged_lora_and_full_finetuning_are_allowed():
    reject_unmerged_lora_adapter(_config(lora_rank=16, lora={"merge": True}))
    reject_unmerged_lora_adapter(_config(lora={"rank": 8, "merge": True}))
    reject_unmerged_lora_adapter(_config())
