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

"""Bound direct tokenizer-manager RPCs when a scheduler stops replying."""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

_T = TypeVar("_T")


def _consume_exception(task: asyncio.Future) -> None:
    if not task.cancelled():
        task.exception()


async def await_scheduler_response(
    response: Awaitable[_T],
    *,
    timeout: float,
    description: str,
    process_failure: Callable[[], str | None],
    last_scheduler_output: Callable[[], float] | None = None,
) -> _T:
    """Wait for a response, scheduler exit, or a finite deadline.

    A tokenizer HTTP process can stay healthy after its scheduler dies. Checking
    HTTP health therefore cannot replace checking the owned subprocess handles.
    The deadline also covers versions which do not expose those handles.

    Without ``last_scheduler_output`` the deadline counts from the call. With it
    (a ``time.time()`` timestamp of the scheduler's latest output on this replica)
    the deadline counts from the later of the call and that output, so requests
    queued behind a busy but live scheduler never time out, while a scheduler that
    stops producing any output still fails after ``timeout`` seconds.
    """
    task = asyncio.ensure_future(response)
    try:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("SGLang response timeout must be a positive finite number")
        started = time.time()
        while True:
            if task.done():
                return task.result()
            failure = process_failure()
            if failure is not None:
                raise RuntimeError(f"SGLang {description} failed: {failure}. Check the scheduler log for the cause.")
            last_output = last_scheduler_output() if last_scheduler_output is not None else started
            remaining = max(started, last_output) + timeout - time.time()
            if remaining <= 0:
                waited = "no scheduler output" if last_scheduler_output is not None else "no scheduler response"
                raise TimeoutError(
                    f"SGLang {description} received {waited} within {timeout:g}s. "
                    "The HTTP process may still be alive; check the scheduler log for a crash or CUDA OOM."
                )
            await asyncio.wait({task}, timeout=min(1.0, remaining))
    finally:
        if not task.done():
            task.cancel()
            # wait_for() can itself wait forever if coroutine cleanup suppresses
            # cancellation. Bound cleanup too, while consuming eventual errors.
            task.add_done_callback(_consume_exception)
            await asyncio.wait({task}, timeout=1.0)
