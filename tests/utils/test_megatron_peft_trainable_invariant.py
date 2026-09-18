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

import sys
from types import SimpleNamespace

import pytest

from verl.utils.megatron_peft_utils import (
    count_adapter_parameters,
    freeze_peft_router_expert_bias,
    summarize_peft_parameters,
    validate_peft_trainable_parameters,
)


class FakeParameter:
    def __init__(self, size: int, *, requires_grad: bool):
        self._size = size
        self.requires_grad = requires_grad

    def numel(self):
        return self._size


class FakeModule:
    def __init__(self, parameters):
        self._parameters = parameters

    def named_parameters(self):
        return iter(self._parameters)


@pytest.fixture(autouse=True)
def fake_unwrap_model(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "verl.utils.megatron_utils",
        SimpleNamespace(unwrap_model=lambda model: model),
    )


def test_validate_peft_trainable_parameters_covers_all_chunks_and_shared_parameters():
    shared_base = FakeParameter(100, requires_grad=False)
    chunks = [
        FakeModule(
            [
                ("decoder.layers.0.weight", shared_base),
                ("decoder.layers.0.linear.lora_a", FakeParameter(7, requires_grad=True)),
            ]
        ),
        FakeModule(
            [
                ("decoder.layers.0.weight", shared_base),
                ("decoder.layers.1.linear.lora_b", FakeParameter(11, requires_grad=True)),
            ]
        ),
    ]

    summary = validate_peft_trainable_parameters(chunks)

    assert summary == {
        "total_parameters": 118,
        "trainable_parameters": 18,
        "trainable_tensors": 2,
        "adapter_parameters": 18,
        "adapter_tensors": 2,
        "unexpected_trainable": [],
    }
    assert count_adapter_parameters(chunks) == (18, 118, 100 * 18 / 118)


def test_validate_peft_trainable_parameters_rejects_thawed_backbone():
    module = FakeModule(
        [
            ("decoder.layers.0.weight", FakeParameter(100, requires_grad=True)),
            ("decoder.layers.0.linear.lora_a", FakeParameter(7, requires_grad=True)),
        ]
    )

    with pytest.raises(RuntimeError, match="non-adapter parameters are trainable"):
        validate_peft_trainable_parameters(module)


def test_validate_peft_trainable_parameters_rejects_missing_adapter():
    module = FakeModule([("decoder.layers.0.weight", FakeParameter(100, requires_grad=False))])

    with pytest.raises(RuntimeError, match="no trainable adapter parameters remain"):
        validate_peft_trainable_parameters(module)


def test_summarize_peft_parameters_reports_unexpected_trainable_names():
    module = FakeModule([("decoder.layers.4.experts.weight", FakeParameter(37, requires_grad=True))])

    assert summarize_peft_parameters(module)["unexpected_trainable"] == ["chunk=0:decoder.layers.4.experts.weight"]


class FakeModuleTree:
    def __init__(self, *children):
        self.children = children

    def modules(self):
        return iter(self.children)


def test_freeze_router_bias_preserves_values_and_counts_shared_routers_once():
    bias = [0.25, -0.5]
    tokens = [3, 1]
    router = SimpleNamespace(expert_bias=bias, frozen_expert_bias=False, local_tokens_per_expert=tokens)
    chunks = [FakeModuleTree(router), FakeModuleTree(router)]

    assert freeze_peft_router_expert_bias(chunks) == 1
    assert router.frozen_expert_bias is True
    assert router.expert_bias is bias
    assert router.local_tokens_per_expert is tokens
    assert bias == [0.25, -0.5]
    assert freeze_peft_router_expert_bias(chunks) == 1


def test_freeze_router_bias_is_noop_for_dense_or_bias_disabled_models():
    disabled = SimpleNamespace(expert_bias=None, frozen_expert_bias=False)
    assert freeze_peft_router_expert_bias(FakeModuleTree(SimpleNamespace(), disabled)) == 0
    assert disabled.frozen_expert_bias is False


def test_freeze_router_bias_rejects_missing_native_update_guard_before_any_mutation():
    supported = SimpleNamespace(expert_bias=[1.0], frozen_expert_bias=False)
    unsupported = SimpleNamespace(expert_bias=[2.0])
    with pytest.raises(RuntimeError, match="frozen_expert_bias update guard"):
        freeze_peft_router_expert_bias(FakeModuleTree(supported, unsupported))
    assert supported.frozen_expert_bias is False


def test_router_freeze_stops_native_mcore_update_without_disabling_bias(tmp_path):
    torch = pytest.importorskip("torch")
    finalizer = pytest.importorskip("megatron.core.distributed.finalize_model_grads")
    if torch.distributed.is_initialized():
        pytest.skip("This single-rank CPU control owns its process group")

    router = torch.nn.Module()
    router.register_buffer("expert_bias", torch.tensor([0.25, -0.5]))
    router.register_buffer("local_tokens_per_expert", torch.tensor([3.0, 1.0]))
    router.register_parameter("weight", torch.nn.Parameter(torch.ones(2, 2), requires_grad=False))
    router.frozen_expert_bias = False
    router.enable_expert_bias = True
    config = SimpleNamespace(moe_router_bias_update_rate=0.001)
    before = router.expert_bias.clone()

    torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        # Frozen parameters alone do not prevent the native update.
        finalizer._update_router_expert_bias([router], config, torch.distributed.group.WORLD)
        assert torch.equal(router.expert_bias, before + torch.tensor([-0.001, 0.001]))
        router.expert_bias.copy_(before)

        assert freeze_peft_router_expert_bias(router) == 1
        finalizer._update_router_expert_bias([router], config, torch.distributed.group.WORLD)
        assert torch.equal(router.expert_bias, before)
        assert router.enable_expert_bias is True
        assert config.moe_router_bias_update_rate == 0.001
        assert router.weight.requires_grad is False
    finally:
        torch.distributed.destroy_process_group()
