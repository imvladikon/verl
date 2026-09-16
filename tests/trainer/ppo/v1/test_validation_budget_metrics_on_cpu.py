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
"""A validation score must not move just because the policy thinks longer or shorter."""

from verl.trainer.ppo.v1.trainer_base import validation_budget_metrics


def test_shorter_thinking_alone_moves_the_blended_score_but_not_the_split():
    # Same per-answer accuracy, fewer answers cut off at the 100-token budget.
    before = validation_budget_metrics([1.0, -1.0, -1.0, -1.0], [40, 60, 100, 100], budget=100)
    after = validation_budget_metrics([1.0, 1.0, -1.0, -1.0], [40, 50, 60, 100], budget=100)

    assert before["val-aux/truncated_ratio"] == 0.5
    assert after["val-aux/truncated_ratio"] == 0.25
    # Accuracy among answers that actually finished is unchanged; only completion moved.
    assert before["val-aux/score_among_untruncated"] == 0.0
    assert after["val-aux/score_among_untruncated"] == 1.0 / 3


def test_lengths_are_reported():
    metrics = validation_budget_metrics([1.0, -1.0], [10, 200], budget=200)
    assert metrics["val-aux/response_length/mean"] == 105
    assert metrics["val-aux/response_length/max"] == 200
    assert metrics["val-aux/truncated_ratio"] == 0.5
    assert metrics["val-aux/score_among_untruncated"] == 1.0


def test_degenerate_inputs_produce_no_metrics():
    assert validation_budget_metrics([], [], budget=100) == {}
    assert validation_budget_metrics([1.0], [10], budget=0) == {}
    assert validation_budget_metrics([1.0, -1.0], [10], budget=100) == {}


def test_all_truncated_reports_no_accuracy():
    metrics = validation_budget_metrics([-1.0, -1.0], [100, 120], budget=100)
    assert metrics["val-aux/truncated_ratio"] == 1.0
    assert "val-aux/score_among_untruncated" not in metrics
