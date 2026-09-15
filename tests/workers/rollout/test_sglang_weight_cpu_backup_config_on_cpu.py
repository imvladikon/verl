# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.sglang_rollout.async_sglang_server import (
    _set_default_weights_cpu_backup,
)


def test_explicit_weight_cpu_backup_override_is_not_overwritten():
    args = {"enable_weights_cpu_backup": False}
    _set_default_weights_cpu_backup(args, rollout_mode=RolloutMode.HYBRID, lora_rank=0)
    assert args["enable_weights_cpu_backup"] is False


def test_hybrid_weight_cpu_backup_defaults_to_enabled():
    args = {}
    _set_default_weights_cpu_backup(args, rollout_mode=RolloutMode.HYBRID, lora_rank=0)
    assert args["enable_weights_cpu_backup"] is True


def test_standalone_weight_cpu_backup_defaults_to_disabled():
    args = {}
    _set_default_weights_cpu_backup(args, rollout_mode=RolloutMode.STANDALONE, lora_rank=0)
    assert args["enable_weights_cpu_backup"] is False
