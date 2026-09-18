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
"""Asking for validation samples on a console-only run must not fail quietly."""

from verl.utils.tracking import ValidationGenerationsLogger

_SAMPLES = [("prompt", "generation", 1.0)]


def test_a_console_only_run_renders_nowhere_and_says_so_in_the_return():
    """Every renderer here is a table; console is not one, which is the whole trap."""
    assert ValidationGenerationsLogger().log(["console"], _SAMPLES, step=1) == 0


def test_no_backends_at_all_is_also_zero():
    assert ValidationGenerationsLogger().log([], _SAMPLES, step=1) == 0


def test_a_backend_that_can_render_is_counted(monkeypatch):
    seen = []
    instance = ValidationGenerationsLogger()
    monkeypatch.setattr(
        instance, "log_generations_to_tensorboard", lambda samples, step: seen.append((samples, step))
    )
    assert instance.log(["console", "tensorboard"], _SAMPLES, step=7) == 1
    assert seen == [(_SAMPLES, 7)]


def test_every_configured_backend_is_counted_not_just_the_first(monkeypatch):
    instance = ValidationGenerationsLogger()
    for name in ("log_generations_to_tensorboard", "log_generations_to_mlflow"):
        monkeypatch.setattr(instance, name, lambda samples, step: None)
    assert instance.log(["tensorboard", "mlflow", "console"], _SAMPLES, step=1) == 2
