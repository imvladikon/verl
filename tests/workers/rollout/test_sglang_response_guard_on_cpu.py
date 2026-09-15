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

import asyncio
import time

import pytest

from verl.workers.rollout.sglang_rollout.response_guard import await_scheduler_response


def test_response_and_scheduler_errors_are_preserved():
    async def run():
        async def response():
            return {"token_ids": [1, 2], "finish_reason": "length"}

        result = await await_scheduler_response(
            response(), timeout=1, description="generate", process_failure=lambda: None
        )
        assert result == {"token_ids": [1, 2], "finish_reason": "length"}

        failure = RuntimeError("scheduler rejected request")

        async def rejected():
            raise failure

        with pytest.raises(RuntimeError) as caught:
            await await_scheduler_response(rejected(), timeout=1, description="generate", process_failure=lambda: None)
        assert caught.value is failure

    asyncio.run(run())


def test_dead_scheduler_fails_without_waiting_for_request_deadline():
    async def run():
        response = asyncio.get_running_loop().create_future()
        with pytest.raises(RuntimeError, match=r"replica=2.*scheduler_0.*pid=123.*code -9"):
            await asyncio.wait_for(
                await_scheduler_response(
                    response,
                    timeout=1800,
                    description="replica=2 resume_memory_occupation tags=['kv_cache']",
                    process_failure=lambda: "scheduler_0 (pid=123) exited with code -9",
                ),
                timeout=0.5,
            )
        assert response.cancelled()

    asyncio.run(run())


def test_death_after_request_started_is_detected():
    async def run():
        started = asyncio.Event()
        cleaned_up = asyncio.Event()

        async def response():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned_up.set()

        def process_failure():
            return "scheduler_0 exited with code 1" if started.is_set() else None

        with pytest.raises(RuntimeError, match="scheduler_0 exited"):
            await asyncio.wait_for(
                await_scheduler_response(
                    response(), timeout=1800, description="generate", process_failure=process_failure
                ),
                timeout=3,
            )
        assert cleaned_up.is_set()

    asyncio.run(run())


def test_deadline_covers_missing_process_handles_and_cancels_request():
    async def run():
        response = asyncio.get_running_loop().create_future()
        with pytest.raises(TimeoutError, match=r"replica=3.*request_id=abc.*0.01s.*HTTP.*CUDA OOM"):
            await await_scheduler_response(
                response,
                timeout=0.01,
                description="replica=3 generate request_id=abc",
                process_failure=lambda: None,
            )
        assert response.cancelled()

    asyncio.run(run())


def test_caller_cancellation_reaches_request():
    async def run():
        response = asyncio.get_running_loop().create_future()
        guard = asyncio.create_task(
            await_scheduler_response(response, timeout=1800, description="generate", process_failure=lambda: None)
        )
        await asyncio.sleep(0)
        guard.cancel()
        with pytest.raises(asyncio.CancelledError):
            await guard
        assert response.cancelled()

    asyncio.run(run())


def test_cancellation_suppressing_cleanup_cannot_hide_timeout():
    async def run():
        release_cleanup = asyncio.Event()
        tasks = []

        async def response():
            tasks.append(asyncio.current_task())
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release_cleanup.wait()
                raise RuntimeError("late cleanup error") from None

        loop = asyncio.get_running_loop()
        unhandled = []
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        try:
            with pytest.raises(TimeoutError, match="no scheduler response"):
                await asyncio.wait_for(
                    await_scheduler_response(
                        response(), timeout=0.01, description="generate", process_failure=lambda: None
                    ),
                    timeout=2,
                )
        finally:
            release_cleanup.set()
            await asyncio.wait(tasks, timeout=1)
        assert not unhandled

    asyncio.run(run())


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_rejects_and_cancels_request(timeout):
    async def run():
        response = asyncio.get_running_loop().create_future()
        with pytest.raises(ValueError, match="positive finite"):
            await await_scheduler_response(
                response, timeout=timeout, description="generate", process_failure=lambda: None
            )
        assert response.cancelled()

    asyncio.run(run())


def test_queued_generation_survives_while_scheduler_keeps_producing_output():
    async def run():
        last_output = [time.time()]

        async def scheduler_progress():
            for _ in range(10):
                await asyncio.sleep(0.05)
                last_output[0] = time.time()

        async def queued_response():
            await asyncio.sleep(0.5)  # longer than the timeout, but the scheduler keeps emitting output
            return {"token_ids": [7]}

        progress = asyncio.create_task(scheduler_progress())
        result = await await_scheduler_response(
            queued_response(),
            timeout=0.2,
            description="generate",
            process_failure=lambda: None,
            last_scheduler_output=lambda: last_output[0],
        )
        await progress
        assert result == {"token_ids": [7]}

    asyncio.run(run())


def test_silent_scheduler_times_out_generation():
    async def run():
        stale_output = time.time() - 3600
        response = asyncio.get_running_loop().create_future()
        started = time.time()
        with pytest.raises(TimeoutError, match="no scheduler output within 0.2s"):
            await await_scheduler_response(
                response,
                timeout=0.2,
                description="generate",
                process_failure=lambda: None,
                last_scheduler_output=lambda: stale_output,
            )
        # The deadline counts from the call when the last output is older than the request.
        assert 0.15 <= time.time() - started < 1.5
        assert response.cancelled()

    asyncio.run(run())


def test_control_rpc_deadline_counts_from_the_call():
    async def run():
        response = asyncio.get_running_loop().create_future()
        with pytest.raises(TimeoutError, match="no scheduler response within 0.2s"):
            await await_scheduler_response(
                response, timeout=0.2, description="flush_cache", process_failure=lambda: None
            )

    asyncio.run(run())
