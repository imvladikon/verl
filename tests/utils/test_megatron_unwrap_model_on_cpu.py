from types import SimpleNamespace

import torch.nn as nn

from verl.utils.megatron_utils import (
    ALL_MODULE_WRAPPER_CLASSNAMES,
    DISTRIBUTED_MODULE_WRAPPER_CLASSES,
    register_megatron_training_hooks,
    unwrap_model,
)


def test_default_unwrap_types_are_valid_for_isinstance():
    assert ALL_MODULE_WRAPPER_CLASSNAMES
    assert all(isinstance(wrapper, type) for wrapper in ALL_MODULE_WRAPPER_CLASSNAMES)
    assert all(isinstance(wrapper, type) for wrapper in DISTRIBUTED_MODULE_WRAPPER_CLASSES)
    module = nn.Linear(2, 2)
    assert unwrap_model(module) is module


def test_new_megatron_fsdp_concrete_wrappers_are_registered():
    from megatron.core.distributed.fsdp import mcore_fsdp_adapter

    for name in ("FullyShardedDataParallelV1", "FullyShardedDataParallelV2"):
        wrapper = getattr(mcore_fsdp_adapter, name, None)
        if isinstance(wrapper, type):
            assert wrapper in ALL_MODULE_WRAPPER_CLASSNAMES


def test_training_hook_accepts_new_fsdp_factory_api_for_plain_model():
    class PlainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(no_sync_func=None)
            self.ddp_config = SimpleNamespace(
                overlap_grad_reduce=True,
                align_param_gather=False,
            )

    optimizer = SimpleNamespace(
        scale_loss=lambda loss: loss,
        config=SimpleNamespace(overlap_param_gather=False),
    )
    model = PlainModel()
    register_megatron_training_hooks([model], optimizer)
    assert model.config.no_sync_func is None
    assert callable(model.config.finalize_model_grads_func)
