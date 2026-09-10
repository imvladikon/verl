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

import asyncio

from verl.trainer.ppo.v1.agent_loop_tq import _settle_session_tasks


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())


def test_terminal_prompt_tag_distinguishes_permanent_actor_death():
    import ray
    from verl.trainer.ppo.v1.agent_loop_tq import _terminal_prompt_tag

    assert _terminal_prompt_tag([]) == {"status": "finished"}
    for error in [ValueError("bad sample"), TimeoutError("temporary"), asyncio.CancelledError()]:
        assert _terminal_prompt_tag([error]) == {"status": "failure"}
    assert _terminal_prompt_tag([ray.exceptions.ActorDiedError()]) == {
        "status": "failure", "fatal_error": "rollout_actor_died"
    }


def test_dead_actor_is_published_only_after_sibling_sessions_settle(monkeypatch):
    import ray
    from types import SimpleNamespace
    import verl.trainer.ppo.v1.agent_loop_tq as module

    async def run():
        settled = asyncio.Event()
        tags = []

        async def put(**kwargs):
            tag = kwargs["tag"]
            if tag["status"] != "running":
                assert settled.is_set(), "Fatal status must not race sibling trajectory writes"
            tags.append(tag)

        async def agent(*args, session_id, **kwargs):
            if session_id == 0:
                raise ray.exceptions.ActorDiedError()
            await asyncio.sleep(0.01)
            settled.set()

        monkeypatch.setattr(module.tq, "async_kv_put", put)
        rollout = SimpleNamespace(n=2, val_kwargs=SimpleNamespace(n=2))
        worker = SimpleNamespace(config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout)),
                                 _run_agent_loop=agent)
        method = module.AgentLoopWorkerTQ.__ray_metadata__.modified_class._run_prompt
        await method(worker, {"uid": "dead-actor-case"}, {}, {"validate": False})
        assert tags == [{"status": "running"}, {"status": "failure", "fatal_error": "rollout_actor_died"}]

    asyncio.run(run())
