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
"""A stalled replica must be judged by generation output, not by any scheduler message."""

import asyncio
import types

from verl.workers.rollout.sglang_rollout import async_sglang_server


def _server(tokenizer_manager):
    server = async_sglang_server.SGLangHttpServer.__new__(async_sglang_server.SGLangHttpServer)
    server.tokenizer_manager = tokenizer_manager
    server.replica_rank = 0
    server.node_rank = 0
    server.config = types.SimpleNamespace(server=types.SimpleNamespace(generation_timeout=7200, timeout=60))
    return server


def _captured_timestamp_source(monkeypatch, tokenizer_manager, *, generation=True):
    captured = {}

    async def fake_await(response, **kwargs):
        captured.update(kwargs)
        return "done"

    monkeypatch.setattr(async_sglang_server, "await_scheduler_response", fake_await)

    async def run():
        return await _server(tokenizer_manager)._await_scheduler_response(
            asyncio.sleep(0), "generate", generation=generation
        )

    assert asyncio.run(run()) == "done"
    return captured


def test_generation_progress_is_read_from_the_generation_timestamp(monkeypatch):
    manager = types.SimpleNamespace(last_generation_tstamp=111.0, last_receive_tstamp=999.0)
    captured = _captured_timestamp_source(monkeypatch, manager)
    # 999.0 is the control-traffic clock a hung scheduler keeps advancing; reading it never times out.
    assert captured["last_scheduler_output"]() == 111.0
    assert captured["timeout"] == 7200


def test_a_build_without_the_generation_timestamp_falls_back_and_warns(monkeypatch):
    manager = types.SimpleNamespace(last_receive_tstamp=999.0)
    captured = _captured_timestamp_source(monkeypatch, manager)
    assert captured["last_scheduler_output"]() == 999.0


def test_control_rpcs_keep_their_own_deadline(monkeypatch):
    manager = types.SimpleNamespace(last_generation_tstamp=111.0, last_receive_tstamp=999.0)
    captured = _captured_timestamp_source(monkeypatch, manager, generation=False)
    assert captured["last_scheduler_output"] is None
    assert captured["timeout"] == 60
