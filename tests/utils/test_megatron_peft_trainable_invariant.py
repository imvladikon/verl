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
