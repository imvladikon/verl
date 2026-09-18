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
"""Make a rank that stops making progress say where it is standing.

NCCL's watchdog only knows about collectives in flight, so a rank that hangs *before* entering one
leaves no record at all: the flight recorder shows its peers waiting and nothing about the process
actually holding them up. That is the shape of failure that costs a multi-node job its allocation,
because the one process to look at is the one with no evidence.

The dump is driven by a heartbeat, not by a timer: :func:`heartbeat` is called when a step
finishes, and it pushes the deadline out. A dump therefore means "this rank did not finish a step
within the interval", not "the interval elapsed" -- the difference between a signal and a stream of
stacks from a healthy run.

Output goes to one file per rank when a directory is given, because stderr from many ranks is
aggregated by the launcher and a faulthandler dump can be interleaved away or dropped.
"""

from __future__ import annotations

import faulthandler
import os
import sys

_INTERVAL_ENV = "VERL_HANG_DUMP_SECONDS"
_DIRECTORY_ENV = "VERL_HANG_DUMP_DIR"

_interval: float | None = None
_stream = None  # kept open for the process lifetime: faulthandler writes to the descriptor


def _resolve_interval(seconds: float | None) -> float:
    if seconds is not None:
        return seconds
    raw = os.environ.get(_INTERVAL_ENV, "")
    try:
        return float(raw) if raw else 0.0
    except ValueError:
        return 0.0


def _open_stream(directory: str | None, rank: int | None):
    """One file per rank, or stderr when no directory is configured."""
    directory = directory if directory is not None else os.environ.get(_DIRECTORY_ENV, "")
    if not directory:
        return sys.stderr
    if rank is None:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)
    os.makedirs(directory, exist_ok=True)
    # Line buffered: a dump has to survive the process it is describing.
    return open(os.path.join(directory, f"faulthandler_rank{rank}.log"), "a", buffering=1)  # noqa: SIM115


def install_hang_dump(
    seconds: float | None = None, directory: str | None = None, rank: int | None = None, stream=None
) -> float | None:
    """Arm the dump. Returns the interval in force, or None when it stays off.

    Args:
        seconds: interval; falls back to ``VERL_HANG_DUMP_SECONDS``. Zero or unset disables it.
        directory: where per-rank files go; falls back to ``VERL_HANG_DUMP_DIR``. Without it the
            dump goes to stderr, which a launcher may aggregate lossily.
        rank: this process's rank, for the file name. Defaults to ``RANK``.
        stream: an already-open destination, which wins over ``directory``.
    """
    global _interval, _stream
    interval = _resolve_interval(seconds)
    if interval <= 0:
        return None

    _stream = stream if stream is not None else _open_stream(directory, rank)
    _interval = interval
    # enable() covers a crash; the heartbeat below covers a hang, which is what NCCL cannot describe.
    faulthandler.enable(file=_stream)
    faulthandler.dump_traceback_later(interval, repeat=True, file=_stream, exit=False)
    return interval


def heartbeat() -> None:
    """Report progress: push the dump deadline out by one interval.

    Call it where a unit of work completes. Without this the timer fires on schedule and prints
    stacks of a perfectly healthy run, which buries the one dump that matters.
    """
    if _interval is None:
        return
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(_interval, repeat=True, file=_stream, exit=False)


def disarm() -> None:
    """Stop dumping, for a shutdown path that is allowed to take its time."""
    global _interval
    if _interval is None:
        return
    faulthandler.cancel_dump_traceback_later()
    _interval = None
