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

"""Two CPU/gloo ranks verify VERL's registration of Bridge's EP adapter sync.

python -m torch.distributed.run --standalone --nproc-per-node=2 -m pytest -q \
    tests/utils/test_megatron_expert_adapter_grad_sync_on_cpu.py

The DDP buffer normalization and fused buffer write are simulated; the adapter
hook, VERL registration, Bridge finalize wrapper and EP collective are real.
An actual fused CUDA backward still requires the distributed model test.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

core_distributed = pytest.importorskip("megatron.core.distributed")
peft = pytest.importorskip("megatron.bridge.peft.utils")


@pytest.fixture(scope="module")
def ep_group():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("requires two CPU/gloo ranks")
    assert not dist.is_initialized()
    dist.init_process_group("gloo")
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fused_accumulation", [False, True])
def test_registered_finalize_synchronizes_shared_expert_adapter(ep_group, fused_accumulation, monkeypatch):
    from verl.utils.megatron_utils import register_megatron_training_hooks

    param = torch.nn.Parameter(torch.ones(2, 3))
    model = torch.nn.ParameterList([param])
    model.config = SimpleNamespace()
    model.ddp_config = SimpleNamespace()
    optimizer = SimpleNamespace(config=SimpleNamespace(), scale_loss=lambda loss: loss)
    peft.mark_expert_parallel_replicated(param, ep_group=ep_group)

    def finish_ddp(_model, *_args, **_kwargs):
        # EP2 has expert-DP=1, but its expert buffers are scaled by the full
        # data-parallel world (2). The missing EP sum is the regression.
        param.main_grad.div_(dist.get_world_size(ep_group))

    monkeypatch.setattr(core_distributed, "finalize_model_grads", finish_ddp)
    monkeypatch.setattr(peft, "finalize_model_grads", finish_ddp)
    register_megatron_training_hooks([model], optimizer)
    value = float(dist.get_rank(ep_group) + 1)
    if fused_accumulation:
        # MCore's fused kernel writes the real gradient here and gives
        # autograd a dummy. Eager hooks cannot reduce main_grad.
        param.main_grad = torch.full_like(param, value)
        param.grad_added_to_main_grad = True
        (param * 0).sum().backward()
    else:
        (param * value).sum().backward()
        param.main_grad = param.grad.clone()

    model.config.finalize_model_grads_func([model], pg_collection=SimpleNamespace(ep=ep_group))
    torch.testing.assert_close(param.main_grad, torch.full_like(param, 1.5), atol=0, rtol=0)
