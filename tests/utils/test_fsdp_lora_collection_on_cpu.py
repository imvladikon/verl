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

from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn
from torch.distributed.fsdp import ShardingStrategy

from verl.utils.fsdp_utils import (
    collect_lora_params,
    layered_load_lora_params,
    layered_summon_lora_params,
    restore_fsdp1_no_shard_frozen_param_views,
)


def test_layered_summon_restores_global_name_before_peft_filtering():
    root = nn.Module()
    root._fsdp_wrapped_module = nn.Module()
    root._fsdp_wrapped_module.base_model = nn.Module()
    root._fsdp_wrapped_module.base_model.model = nn.Module()
    root._fsdp_wrapped_module.base_model.model.proj = nn.Module()
    root._fsdp_wrapped_module.base_model.model.proj.lora_A = nn.ModuleDict({"default": nn.Linear(3, 2, bias=False)})
    leaf = root._fsdp_wrapped_module.base_model.model.proj.lora_A["default"]
    leaf._is_root = False

    def fake_peft_state_dict(_model, state_dict=None):
        return {name.replace(".default", ""): value for name, value in state_dict.items() if "lora_" in name}

    with (
        patch("verl.utils.fsdp_utils.fsdp_version", side_effect=lambda module: int(module is leaf)),
        patch("verl.utils.fsdp_utils.FSDP.summon_full_params", return_value=nullcontext()),
        patch("verl.utils.fsdp_utils.get_peft_model_state_dict", side_effect=fake_peft_state_dict),
    ):
        params = layered_summon_lora_params(root)

    assert list(params) == ["base_model.model.proj.lora_A.weight"]
    assert params["base_model.model.proj.lora_A.weight"].device.type == "cpu"
    assert leaf._is_root is False


def test_no_shard_collects_adapter_without_summoning_full_params():
    module = nn.Module()
    module._fsdp_wrapped_module = nn.Module()
    module.sharding_strategy = ShardingStrategy.NO_SHARD
    module._use_orig_params = True
    adapter = torch.ones(2, 3)

    with (
        patch("verl.utils.fsdp_utils.fsdp_version", return_value=1),
        patch(
            "verl.utils.fsdp_utils.get_peft_model_state_dict",
            return_value={"base_model.model.proj.lora_A.weight": adapter},
        ),
        patch("verl.utils.fsdp_utils.layered_summon_lora_params") as layered,
        patch("verl.utils.fsdp_utils.FSDP.summon_full_params") as summon,
    ):
        params = collect_lora_params(module, layered_summon=True, base_sync_done=True)

    assert list(params) == ["base_model.model.proj.lora_A.weight"]
    assert params["base_model.model.proj.lora_A.weight"].device.type == "cpu"
    layered.assert_not_called()
    summon.assert_not_called()


def test_layered_collection_can_forbid_full_model_fallback():
    module = nn.Module()
    module._fsdp_wrapped_module = nn.Module()
    module.sharding_strategy = ShardingStrategy.FULL_SHARD
    module._use_orig_params = False

    with (
        patch("verl.utils.fsdp_utils.fsdp_version", return_value=1),
        patch("verl.utils.fsdp_utils.layered_summon_lora_params", return_value=OrderedDict()),
        patch("verl.utils.fsdp_utils.FSDP.summon_full_params") as summon,
    ):
        params = collect_lora_params(
            module,
            layered_summon=True,
            base_sync_done=True,
            allow_full_summon_fallback=False,
        )

    assert not params
    summon.assert_not_called()


def test_no_shard_loads_peft_adapter_without_model_state_dict():
    peft_model = nn.Module()
    adapter = nn.Parameter(torch.zeros(2, 3))
    peft_model.register_parameter("adapter", adapter)
    module = SimpleNamespace(
        _fsdp_wrapped_module=peft_model,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        _use_orig_params=True,
    )

    with (
        patch("verl.utils.fsdp_utils.fsdp_version", return_value=1),
        patch(
            "verl.utils.fsdp_utils.get_peft_model_state_dict",
            return_value={"proj.lora_A.weight": adapter},
        ),
    ):
        layered_load_lora_params(module, {"proj.lora_A.weight": torch.ones(2, 3)})

    torch.testing.assert_close(adapter, torch.ones(2, 3))


def test_no_shard_restores_frozen_parameter_views_after_backward():
    owner = nn.Module()
    saved = nn.Parameter(torch.ones(2, 3), requires_grad=False)
    owner._parameters["weight"] = saved.detach().view_as(saved)
    handle = SimpleNamespace(
        uses_sharded_strategy=False,
        _use_orig_params=True,
        flat_param=SimpleNamespace(
            requires_grad=False,
            _params=[saved],
            _param_infos=[SimpleNamespace(module=owner, param_name="weight")],
        ),
        _use_sharded_views=Mock(side_effect=lambda: owner._parameters.__setitem__("weight", saved)),
    )
    module = SimpleNamespace(_use_orig_params=True, _all_handles=[handle])

    with (
        patch("verl.utils.fsdp_utils.fsdp_version", return_value=1),
        patch("verl.utils.fsdp_utils._lazy_init"),
    ):
        restore_fsdp1_no_shard_frozen_param_views(module)

    handle._use_sharded_views.assert_called_once_with()
    assert owner._parameters["weight"] is saved


def test_no_shard_fails_if_frozen_parameter_objects_were_already_lost():
    owner = nn.Module()
    temporary_view = torch.ones(2, 3)
    owner._parameters["weight"] = temporary_view
    handle = SimpleNamespace(
        uses_sharded_strategy=False,
        _use_orig_params=True,
        flat_param=SimpleNamespace(
            requires_grad=False,
            _params=[temporary_view],
            _param_infos=[SimpleNamespace(module=owner, param_name="weight")],
        ),
        _use_sharded_views=Mock(),
    )
    module = SimpleNamespace(_use_orig_params=True, _all_handles=[handle])

    with (
        patch("verl.utils.fsdp_utils.fsdp_version", return_value=1),
        patch("verl.utils.fsdp_utils._lazy_init"),
        pytest.raises(RuntimeError, match="lost the saved nn.Parameter objects"),
    ):
        restore_fsdp1_no_shard_frozen_param_views(module)

    handle._use_sharded_views.assert_not_called()
