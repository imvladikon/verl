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

import os

import pytest
from hydra import compose, initialize_config_dir

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import RolloutConfig, ServerConfig

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../verl/trainer/config/rollout"))


@pytest.mark.parametrize("overrides", [[], ["server.timeout=900", "server.generation_timeout=3600"]])
def test_server_section_is_overridable_and_instantiated(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="rollout", overrides=["name=sglang", *overrides])

    server = omega_conf_to_dataclass(cfg.server)
    assert isinstance(server, ServerConfig)
    if overrides:
        assert (server.timeout, server.generation_timeout) == (900.0, 3600.0)
    else:
        assert server == ServerConfig()


def test_rollout_config_keeps_server_dataclass():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="rollout", overrides=["name=sglang", "server.timeout=900"])

    # Same path as the workers: instantiate through the node's _target_. A nested node without its own
    # _target_ would stay a plain dict here and break config.server.timeout at the first scheduler RPC.
    rollout = omega_conf_to_dataclass(cfg)
    assert isinstance(rollout, RolloutConfig)
    assert isinstance(rollout.server, ServerConfig)
    assert rollout.server.timeout == 900.0
