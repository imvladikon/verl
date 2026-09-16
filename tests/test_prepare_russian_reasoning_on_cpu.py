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
"""The SFT target must speak GLM-5.3's reasoning format, not the dataset's."""

import importlib.util
import pathlib
import sys

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "examples" / "glm53_flash" / "prepare_russian_reasoning.py"


def _load():
    spec = importlib.util.spec_from_file_location("prepare_russian_reasoning", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_russian_reasoning"] = module
    spec.loader.exec_module(module)
    return module


prepare = _load()


def _row(assistant, user="Сколько будет 2+2?"):
    return {
        "system": "Ты полезный ассистент. <Thought> ... </Thought> <output> ... </output>",
        "conversation": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
    }


def test_the_target_closes_the_thinking_block_the_template_opened():
    messages = prepare.build_messages(_row("<Thought>Считаю.</Thought>\n<output>4</output>"))
    assert [m["role"] for m in messages] == ["user", "assistant"]
    # The template renders "<|assistant|><think>", so the target must not open it again.
    assert not messages[-1]["content"].startswith("<think>")
    assert messages[-1]["content"] == "Считаю.</think>4"


def test_the_datasets_markup_does_not_survive():
    messages = prepare.build_messages(_row("<Thought>a</Thought><output>b</output>"))
    target = messages[-1]["content"]
    assert "<Thought>" not in target and "<output>" not in target


def test_the_datasets_system_prompt_is_dropped_by_default():
    messages = prepare.build_messages(_row("<Thought>a</Thought><output>b</output>"))
    assert all(m["role"] != "system" for m in messages)
    with_system = prepare.build_messages(_row("<Thought>a</Thought><output>b</output>"), system="S")
    assert with_system[0] == {"role": "system", "content": "S"}


def test_rows_without_the_markup_are_rejected():
    assert prepare.build_messages(_row("просто ответ без разметки")) is None
    assert prepare.build_messages(_row("<Thought>only reasoning</Thought>")) is None
    assert prepare.build_messages(_row("<Thought></Thought><output></output>")) is None
    assert prepare.build_messages({"conversation": []}) is None
