# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Backend-specific extensions of :class:`verl.trainer.config.CheckpointConfig`.

The base :class:`CheckpointConfig` lives in ``verl/trainer/config/config.py`` and
carries only fields that every backend understands (``save_contents``,
``load_contents``, ``async_save``). Anything that is meaningful only to one
training backend (e.g. the AutoModel consolidation options) goes into a subclass here,
mirroring how ``ActorConfig`` / ``McoreActorConfig`` are split between
``verl/trainer/config`` and ``verl/workers/config``.
"""

from dataclasses import dataclass

from verl.trainer.config import CheckpointConfig

__all__ = ["AutomodelCheckpointConfig"]


@dataclass
class AutomodelCheckpointConfig(CheckpointConfig):
    """Checkpoint options specific to the AutoModel backend.

    ``save_consolidated`` defaults to the historical VERL behavior. Large
    distributed restart checkpoints can disable it explicitly, while small
    lifecycle tests can retain an HF export for downstream rollout stages.
    ``strict_rng_state`` lets new deterministic-resume qualifications fail
    closed without making pre-existing checkpoints unloadable by default.
    """

    save_consolidated: bool = True
    strict_rng_state: bool = False

    def __post_init__(self) -> None:
        allowed = {"model", "hf_model", "optimizer", "extra"}
        for attribute in ("save_contents", "load_contents"):
            contents = getattr(self, attribute)
            if not contents:
                raise ValueError(f"AutoModel checkpoint {attribute} must not be empty")
            unknown = set(contents) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown AutoModel checkpoint {attribute}: {sorted(unknown)}; allowed values are {sorted(allowed)}"
                )
        if "hf_model" in self.save_contents and not self.save_consolidated:
            raise ValueError("AutoModel save_contents includes 'hf_model', but save_consolidated is false")
