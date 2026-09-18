# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import pytest

from verl.workers.rollout.sglang_rollout.async_sglang_server import _normalize_sglang_launch_result


def test_current_engine_launch_layout_uses_scheduler_init_result():
    tokenizer_manager = object()
    template_manager = object()
    port_args = object()
    scheduler_info = {"max_req_input_len": 128}
    scheduler_init_result = SimpleNamespace(scheduler_infos=[scheduler_info])
    watchdog = object()

    normalized = _normalize_sglang_launch_result(
        (tokenizer_manager, template_manager, port_args, scheduler_init_result, watchdog)
    )

    assert normalized[:3] == (tokenizer_manager, template_manager, scheduler_info)
    assert normalized[3] == (port_args, scheduler_init_result, watchdog)


def test_legacy_launch_layout_keeps_direct_scheduler_info():
    scheduler_info = {"max_req_input_len": 64}

    normalized = _normalize_sglang_launch_result(("tokenizer", "template", scheduler_info, "watchdog"))

    assert normalized[:3] == ("tokenizer", "template", scheduler_info)
    assert normalized[3] == (scheduler_info, "watchdog")


def test_invalid_launch_layout_fails_loudly():
    with pytest.raises(RuntimeError, match="expected at least 3"):
        _normalize_sglang_launch_result(("tokenizer", "template"))
