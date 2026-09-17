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
"""A rank that hangs before its first collective has to leave a stack behind."""

import subprocess
import sys
import textwrap

from verl.utils.stack_dump import install_hang_dump


def test_disabled_unless_asked(monkeypatch):
    monkeypatch.delenv("VERL_HANG_DUMP_SECONDS", raising=False)
    assert install_hang_dump() is None


def test_a_malformed_interval_does_not_break_startup(monkeypatch):
    # A bad value must not stop training; the dump is a diagnostic, not a feature.
    monkeypatch.setenv("VERL_HANG_DUMP_SECONDS", "не число")
    assert install_hang_dump() is None


def test_the_environment_supplies_the_interval(monkeypatch, tmp_path):
    monkeypatch.setenv("VERL_HANG_DUMP_SECONDS", "900")
    with open(tmp_path / "out", "w") as stream:
        assert install_hang_dump(stream=stream) == 900.0
    import faulthandler

    faulthandler.cancel_dump_traceback_later()


def test_a_hung_process_prints_its_stack():
    """The point of the tool: a process stuck in a wait still reports where."""
    program = textwrap.dedent(
        """
        import sys, threading
        sys.path.insert(0, %r)
        from verl.utils.stack_dump import install_hang_dump
        install_hang_dump(0.2)
        def wait_forever():
            threading.Event().wait(5)
        wait_forever()
        """
    ) % str(__file__).rsplit("/tests/", 1)[0]
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=30)
    assert "wait_forever" in result.stderr, result.stderr[-500:]
