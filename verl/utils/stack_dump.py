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
"""Make a rank that stops before its first collective say where it is standing.

NCCL's watchdog only knows about collectives in flight, so a rank that hangs *before* entering one
leaves no record at all: the flight recorder shows its peers waiting and nothing about it. That is
the shape of the failure that costs a multi-node job its whole allocation, because the one process
holding everyone else up is the one piece of evidence missing.

``faulthandler`` answers it from inside the process: every thread's Python stack, printed to stderr
on a timer, which for these jobs is the log that is already collected.
"""

from __future__ import annotations

import faulthandler
import os
import sys

_ENV = "VERL_HANG_DUMP_SECONDS"


def install_hang_dump(seconds: float | None = None, stream=None) -> float | None:
    """Print every thread's stack every ``seconds`` until the process exits.

    Args:
        seconds: interval; falls back to ``VERL_HANG_DUMP_SECONDS``. Zero or unset disables it.
        stream: where to write; defaults to stderr, which the job log captures.

    Returns:
        The interval in force, or None when the dump is not installed.
    """
    if seconds is None:
        raw = os.environ.get(_ENV, "")
        try:
            seconds = float(raw) if raw else 0.0
        except ValueError:
            seconds = 0.0
    if not seconds or seconds <= 0:
        return None

    target = stream if stream is not None else sys.stderr
    # enable() covers the crash case; dump_traceback_later(repeat=True) covers the hang, which is
    # the one NCCL cannot describe.
    faulthandler.enable(file=target)
    faulthandler.dump_traceback_later(seconds, repeat=True, file=target, exit=False)
    return seconds
