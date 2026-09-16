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
"""Rollout sampling options the agent loop does not build must still reach SGLang."""

from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")

from verl.workers.rollout.sglang_rollout.async_sglang_server import (  # noqa: E402
    apply_rollout_sampling_defaults,
)


def _config(**fields):
    defaults = dict(ignore_eos=False, min_new_tokens=None, repetition_penalty=1.0)
    defaults.update(fields)
    return SimpleNamespace(**defaults)


def test_ignore_eos_reaches_sglang():
    params = {"temperature": 1.0, "max_new_tokens": 64}
    apply_rollout_sampling_defaults(params, _config(ignore_eos=True))
    assert params["ignore_eos"] is True


def test_request_values_win_over_the_rollout_config():
    params = {"ignore_eos": False, "max_new_tokens": 64}
    apply_rollout_sampling_defaults(params, _config(ignore_eos=True))
    assert params["ignore_eos"] is False


def test_min_new_tokens_is_clamped_to_the_request_budget():
    params = {"max_new_tokens": 16}
    apply_rollout_sampling_defaults(params, _config(min_new_tokens=64))
    assert params["min_new_tokens"] == 16


def test_absent_options_are_not_invented():
    params = {"max_new_tokens": 16}
    apply_rollout_sampling_defaults(params, SimpleNamespace())
    assert set(params) == {"max_new_tokens"}
