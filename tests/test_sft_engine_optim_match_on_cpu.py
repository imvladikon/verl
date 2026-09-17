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
"""Selecting an engine must not leave another engine's optimizer in place."""

import types

import pytest

from verl.trainer.sft_trainer import SFTTrainer


def _check(engine_target, optim_target):
    """The guard reads two keys, so a plain mapping stands in for the composed config."""
    trainer = types.SimpleNamespace(
        config={"engine": {"_target_": engine_target}, "optim": {"_target_": optim_target}}
    )
    SFTTrainer._check_engine_optimizer_match(trainer)


def test_matching_engine_and_optimizer_pass():
    _check("verl.workers.config.McoreEngineConfig", "verl.workers.config.McoreOptimizerConfig")
    _check("verl.workers.config.FSDPEngineConfig", "verl.workers.config.FSDPOptimizerConfig")


def test_engine_megatron_with_the_default_fsdp_optimizer_is_refused():
    # `engine=megatron` alone leaves optim at its own default; this is that command line.
    with pytest.raises(ValueError, match="optim=megatron"):
        _check("verl.workers.config.McoreEngineConfig", "verl.workers.config.FSDPOptimizerConfig")


def test_an_unknown_target_is_not_second_guessed():
    # Nothing to compare against, so stay out of the way rather than refuse a valid run.
    _check("", "verl.workers.config.FSDPOptimizerConfig")
    _check("verl.workers.config.McoreEngineConfig", "")
