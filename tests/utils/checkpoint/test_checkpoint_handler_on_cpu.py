from types import SimpleNamespace

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.config import CheckpointConfig
from verl.utils.checkpoint.checkpoint_handler import CheckpointHandler
from verl.workers.config import AutomodelCheckpointConfig


def test_checkpoint_handler_honors_extra_content_controls():
    handler = object.__new__(CheckpointHandler)
    handler.engine = SimpleNamespace(
        checkpoint_config=CheckpointConfig(
            save_contents=["model"],
            load_contents=["model", "extra"],
        )
    )

    assert not handler._content_enabled("extra", "save")
    assert handler._content_enabled("extra", "load")


def test_checkpoint_handler_preserves_legacy_extra_default_without_config():
    handler = object.__new__(CheckpointHandler)
    handler.engine = SimpleNamespace()

    assert handler._content_enabled("extra", "save")
    assert handler._content_enabled("extra", "load")


def test_checkpoint_handler_uses_explicit_config_with_ray_like_engine():
    handler = object.__new__(CheckpointHandler)
    handler.engine = SimpleNamespace()
    handler.checkpoint_config = CheckpointConfig(
        save_contents=["model"],
        load_contents=["model", "extra"],
    )

    assert not handler._content_enabled("extra", "save")
    assert handler._content_enabled("extra", "load")


def test_automodel_checkpoint_config_is_hydra_instantiable():
    config = OmegaConf.create(
        {
            "_target_": "verl.workers.config.AutomodelCheckpointConfig",
            "save_contents": ["model", "optimizer", "extra"],
            "load_contents": ["model", "optimizer", "extra"],
            "save_consolidated": False,
            "strict_rng_state": True,
        }
    )

    checkpoint = instantiate(config)

    assert isinstance(checkpoint, AutomodelCheckpointConfig)
    assert checkpoint.save_consolidated is False
    assert checkpoint.strict_rng_state is True


def _checkpoint_handler_for_dataloader(dataloader, resume_global_step):
    handler = object.__new__(CheckpointHandler)
    handler.train_dataloader = dataloader
    handler.resume_global_step = resume_global_step
    handler.dp_rank = 0
    handler.rank = 0
    return handler


def test_checkpoint_handler_starts_fresh_after_epoch_boundary(tmp_path):
    dataset = TensorDataset(torch.arange(4))
    saved_dataloader = StatefulDataLoader(dataset, batch_size=4, shuffle=False)
    list(saved_dataloader)
    torch.save(saved_dataloader.state_dict(), tmp_path / "data_0.pt")

    resumed_dataloader = StatefulDataLoader(dataset, batch_size=4, shuffle=False)
    handler = _checkpoint_handler_for_dataloader(resumed_dataloader, resume_global_step=1)
    handler._load_dataloader_state(str(tmp_path))

    (batch,) = next(iter(resumed_dataloader))
    assert batch.tolist() == [0, 1, 2, 3]


def test_checkpoint_handler_restores_mid_epoch_cursor(tmp_path):
    dataset = TensorDataset(torch.arange(4))
    saved_dataloader = StatefulDataLoader(dataset, batch_size=2, shuffle=False)
    next(iter(saved_dataloader))
    torch.save(saved_dataloader.state_dict(), tmp_path / "data_0.pt")

    resumed_dataloader = StatefulDataLoader(dataset, batch_size=2, shuffle=False)
    handler = _checkpoint_handler_for_dataloader(resumed_dataloader, resume_global_step=1)
    handler._load_dataloader_state(str(tmp_path))

    (batch,) = next(iter(resumed_dataloader))
    assert batch.tolist() == [2, 3]
