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
"""A rank that stops making progress must say where, and a healthy one must stay quiet."""

import subprocess
import sys
import textwrap

from verl.utils.stack_dump import disarm, install_hang_dump

_REPO = str(__file__).rsplit("/tests/", 1)[0]


def test_disabled_unless_asked(monkeypatch):
    monkeypatch.delenv("VERL_HANG_DUMP_SECONDS", raising=False)
    assert install_hang_dump() is None


def test_a_malformed_interval_does_not_break_startup(monkeypatch):
    # A bad value must not stop training; the dump is a diagnostic, not a feature.
    monkeypatch.setenv("VERL_HANG_DUMP_SECONDS", "не число")
    assert install_hang_dump() is None


def test_each_rank_writes_its_own_file(monkeypatch, tmp_path):
    monkeypatch.setenv("VERL_HANG_DUMP_SECONDS", "900")
    monkeypatch.setenv("RANK", "16")
    try:
        assert install_hang_dump(directory=str(tmp_path)) == 900.0
        assert (tmp_path / "faulthandler_rank16.log").exists()
    finally:
        disarm()


def _run(body: str, timeout: int = 40) -> str:
    program = textwrap.dedent(
        f"""
        import sys, threading, time
        sys.path.insert(0, {_REPO!r})
        from verl.utils.stack_dump import install_hang_dump, heartbeat, disarm
        install_hang_dump(0.3)
        def the_step():
            time.sleep(0.1)
        def the_hang():
            threading.Event().wait(3)
        {body}
        """
    )
    return subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=timeout).stderr


def test_a_step_that_never_finishes_is_reported():
    assert "the_hang" in _run("the_hang()")


def test_a_run_that_keeps_finishing_steps_stays_quiet():
    """The defect this replaces: a timer fired on schedule and buried the real dump in noise."""
    body = "for _ in range(12):\n    the_step()\n    heartbeat()\ndisarm()"
    stderr = _run(body)
    assert "the_step" not in stderr, stderr[-400:]
    assert "Timeout" not in stderr, stderr[-400:]
