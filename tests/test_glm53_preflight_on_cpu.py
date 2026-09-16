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
"""The preflight script must survive a broken environment and say what broke."""

import argparse
import importlib.util
import json
import pathlib
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "glm53_preflight.py"


def _load():
    spec = importlib.util.spec_from_file_location("glm53_preflight", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["glm53_preflight"] = module
    spec.loader.exec_module(module)
    return module


preflight = _load()


def _args(**kw):
    defaults = dict(model=None, kernels=False, attention_backend=None, group=None, json=True, allow_fail=None)
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_a_missing_module_reports_why_it_is_missing():
    assert preflight._module("verl_module_that_does_not_exist") is None
    reason = preflight._why("verl_module_that_does_not_exist")
    assert "ModuleNotFoundError" in reason, reason


def test_a_raising_check_is_reported_with_the_exception_text(monkeypatch):
    def explode(_args):
        raise RuntimeError("kernel image is invalid")

    monkeypatch.setattr(preflight, "registry", preflight.Registry())
    preflight.registry.add("group", "exploding check")(explode)

    (result,) = preflight.run(_args())
    assert result.status == preflight.FAIL
    assert "RuntimeError: kernel image is invalid" in result.detail


def test_every_check_returns_a_result_without_a_model_or_a_gpu():
    results = preflight.run(_args())
    assert results, "no checks ran"
    assert {r.status for r in results} <= {preflight.PASS, preflight.WARN, preflight.FAIL, preflight.SKIP}
    for result in results:
        assert result.name and result.detail, result
    # Checks that need an argument must skip rather than fail on a bare run.
    by_name = {r.name: r for r in results}
    assert by_name["checkpoint config"].status == preflight.SKIP
    assert by_name["DSA indexer MQA logits"].status == preflight.SKIP


def test_json_output_is_machine_readable(capsys):
    monkey = _args(json=True)
    results = preflight.run(monkey)
    payload = json.dumps([r.__dict__ for r in results])
    assert json.loads(payload)[0]["status"] in {
        preflight.PASS,
        preflight.WARN,
        preflight.FAIL,
        preflight.SKIP,
    }


@pytest.mark.parametrize("group", ["runtime", "fork fixes", "env"])
def test_groups_can_be_selected(group):
    results = preflight.run(_args(group=[group]))
    assert results, f"group {group} ran no checks"


def test_allow_fail_downgrades_a_failure_without_hiding_it(monkeypatch):
    def failing(_args):
        return preflight.Result("megatron raw-MLP recompute", preflight.FAIL, "fix missing")

    monkeypatch.setattr(preflight, "registry", preflight.Registry())
    preflight.registry.add("fork fixes", "megatron raw-MLP recompute")(failing)

    (gated,) = preflight.run(_args())
    assert gated.status == preflight.FAIL

    (allowed,) = preflight.run(_args(allow_fail=["megatron"]))
    assert allowed.status == preflight.ALLOWED
    assert allowed.detail == "fix missing"

    (unrelated,) = preflight.run(_args(allow_fail=["sglang"]))
    assert unrelated.status == preflight.FAIL
