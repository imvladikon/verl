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


def test_accuracy_survives_the_truncated_rows_that_sink_the_aggregate():
    """Accuracy, not score, is what a checkpoint is compared against.

    Without it a model that got longer and a model that got worse look the same.
    """
    metrics = validation_budget_metrics(
        [-1.0, -1.0, 1.0, 1.0], [100, 100, 40, 50], budget=100, accuracies=[0.0, 0.0, 1.0, 1.0]
    )
    assert metrics["val-aux/truncated_ratio"] == 0.5
    assert metrics["val-aux/acc_among_untruncated"] == 1.0


def test_a_source_that_truncates_is_reported_apart_from_one_that_does_not():
    metrics = validation_budget_metrics(
        [-1.0, -1.0, 1.0, -1.0],
        [100, 100, 10, 20],
        budget=100,
        data_sources=["tmath", "tmath", "mera", "mera"],
        accuracies=[0.0, 0.0, 1.0, 0.0],
    )
    assert metrics["val-aux/tmath/truncated_ratio"] == 1.0
    assert metrics["val-aux/mera/truncated_ratio"] == 0.0
    assert metrics["val-aux/mera/acc_among_untruncated"] == 0.5
    # tmath truncated everything, so it has no finished slice to report.
    assert "val-aux/tmath/acc_among_untruncated" not in metrics


def test_the_smoke_run_numbers_reproduce():
    """Built from the RL smoke: 37.4% truncated at 4096, mean length 2448.

    There the aggregate accuracy read 0.26 while the answers that finished scored 0.43, next to a
    base of 0.42 -- the collapse was the budget, not the adapter.
    """
    budget, finished_len, truncated, total = 4096, 1463, 284, 759
    lengths = [budget] * truncated + [finished_len] * (total - truncated)
    right = round((total - truncated) * 0.426)
    accuracies = [0.0] * truncated + [1.0] * right + [0.0] * (total - truncated - right)
    scores = [-1.0] * truncated + [1.0] * right + [-1.0] * (total - truncated - right)

    metrics = validation_budget_metrics(scores, lengths, budget, accuracies=accuracies)
    assert abs(metrics["val-aux/truncated_ratio"] - 0.374) < 0.001
    assert abs(metrics["val-aux/response_length/mean"] - 2448) < 2
    assert abs(sum(accuracies) / total - 0.26) < 0.01
    assert abs(metrics["val-aux/acc_among_untruncated"] - 0.426) < 0.001


def test_missing_accuracies_leave_the_old_metrics_intact():
    """The reward may not report acc; the budget split must still work."""
    metrics = validation_budget_metrics([1.0, -1.0], [10, 100], budget=100)
    assert metrics["val-aux/score_among_untruncated"] == 1.0
    assert not any("acc_among_untruncated" in key for key in metrics)


def test_mismatched_lengths_are_ignored_rather_than_paired_wrongly():
    metrics = validation_budget_metrics(
        [1.0, -1.0], [10, 20], budget=100, accuracies=[1.0], data_sources=["a"]
    )
    assert not any("acc_among_untruncated" in key for key in metrics)
    assert not any(key.startswith("val-aux/a/") for key in metrics)
