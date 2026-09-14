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

"""Opt-in regression with a tiny GLM and a real scheduler subprocess.

VERL_SGLANG_GLM_TEST_MODEL must point to a local, small GLM checkpoint.
Run in an isolated process on an otherwise idle GPU. The injected exception is
raised without allocating excess memory. SIGQUIT is deliberately recorded so
the tokenizer process survives, reproducing the orphaned-server failure mode.
"""

import asyncio
import os
import signal
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest


def _raise_on_resume(self, request):
    import torch

    raise torch.OutOfMemoryError("injected scheduler resume(kv_cache) failure")


def _scheduler_with_resume_failure(*args, **kwargs):
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager

    SchedulerWeightUpdaterManager.resume_memory_occupation = _raise_on_resume
    run_scheduler_process(*args, **kwargs)
    # The scheduler catches/logs the injected exception and returns. CUDA
    # teardown threads may outlive that event loop; explicitly terminate this
    # test worker to reproduce the dead-process branch of the orphaned server.
    os._exit(1)


def _free_ports():
    with socket.socket() as http, socket.socket() as nccl:
        http.bind(("127.0.0.1", 0))
        nccl.bind(("127.0.0.1", 0))
        return http.getsockname()[1], nccl.getsockname()[1]


@pytest.mark.skipif(not os.environ.get("VERL_SGLANG_GLM_TEST_MODEL"), reason="Requires an explicit tiny GLM checkpoint")
def test_scheduler_exception_does_not_leave_verl_waiting(monkeypatch):
    import torch
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput

    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer

    assert torch.cuda.is_available()
    model = Path(os.environ["VERL_SGLANG_GLM_TEST_MODEL"])
    assert model.is_dir()
    monkeypatch.setattr(Engine, "run_scheduler_process_func", staticmethod(_scheduler_with_resume_failure))
    port, nccl_port = _free_ports()
    engine = None
    previous_handler = signal.getsignal(signal.SIGQUIT)
    signals = []
    try:
        engine = Engine(
            model_path=str(model),
            host="127.0.0.1",
            port=port,
            nccl_port=nccl_port,
            tp_size=1,
            dtype="bfloat16",
            context_length=512,
            max_running_requests=1,
            mem_fraction_static=0.3,
            skip_tokenizer_init=True,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            enable_multimodal=False,
            attention_backend="triton",
            dsa_prefill_backend="torch",
            dsa_decode_backend="torch",
            dsa_paged_mqa_logits_backend="torch",
            dsa_topk_backend="torch",
            moe_runner_backend="triton",
            linear_attn_backend="triton",
            linear_attn_prefill_backend="triton",
            linear_attn_decode_backend="triton",
            log_level="warning",
        )
        manager = engine.tokenizer_manager
        processes = manager._subprocess_watchdog._processes
        assert all(process.exitcode is None for process in processes)
        server = SGLangHttpServer.__new__(SGLangHttpServer)
        server.tokenizer_manager = manager
        server.node_rank = 0
        server.replica_rank = 7
        server.config = SimpleNamespace(
            free_cache_engine=True, server=SimpleNamespace(timeout=30.0, generation_timeout=30.0)
        )

        async def exercise():
            # The first RPC creates the tokenizer loop and replaces SIGQUIT
            # handling. Install our orphaned-server simulation after that step.
            manager.auto_create_handle_loop()
            asyncio.get_running_loop().add_signal_handler(signal.SIGQUIT, signals.append, signal.SIGQUIT)
            # The original direct await receives no error response even after
            # the scheduler logs the injected OOM and exits.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    manager.resume_memory_occupation(ResumeMemoryOccupationReqInput(tags=["kv_cache"]), None),
                    timeout=5,
                )
            assert signals, "Scheduler must have raised the injected exception"
            assert any(process.exitcode is not None for process in processes)

            # Exercise the production VERL entry point and real Process handles.
            # A normal exit code is also fatal while a request is pending.
            with pytest.raises(RuntimeError, match=r"replica=7.*resume_memory_occupation.*pid=.*exited with code"):
                await asyncio.wait_for(server.resume_kv_cache(), timeout=3)

        engine.loop.run_until_complete(exercise())
    finally:
        if engine is not None:
            engine.shutdown()
            engine.loop.remove_signal_handler(signal.SIGQUIT)
        signal.signal(signal.SIGQUIT, previous_handler)
