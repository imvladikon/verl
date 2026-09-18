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
"""Dropped groups split by which end they collapsed at: all solved, or none solved."""

from verl.trainer.ppo.v1.replay_buffer import summarize_dapo_filtered_rewards


def test_the_two_ends_are_counted_apart():
    """The whole point: 40 groups nobody solved and 42 everybody solved are different problems."""
    summary = summarize_dapo_filtered_rewards({-1.0: 40, 1.0: 42})
    assert summary == {
        "filter_groups/dropped_groups": 82,
        "filter_groups/dropped_all_positive": 42,
        "filter_groups/dropped_all_negative": 40,
        "filter_groups/dropped_all_zero": 0,
    }


def test_zero_is_its_own_bucket_not_folded_into_a_sign():
    summary = summarize_dapo_filtered_rewards({0.0: 7})
    assert summary["filter_groups/dropped_all_zero"] == 7
    assert summary["filter_groups/dropped_all_positive"] == 0
    assert summary["filter_groups/dropped_all_negative"] == 0


def test_values_are_bucketed_by_sign_not_by_being_one():
    """A reward in [0, 1] collapses at fractional values, not at -1/+1."""
    summary = summarize_dapo_filtered_rewards({0.25: 3, 0.75: 4, -0.5: 2})
    assert summary["filter_groups/dropped_all_positive"] == 7
    assert summary["filter_groups/dropped_all_negative"] == 2


def test_string_keys_survive_a_round_trip_through_a_tracker():
    """Counter keys can arrive as strings once they have been through json."""
    summary = summarize_dapo_filtered_rewards({"-1.0": 5, "1.0": 6})
    assert summary["filter_groups/dropped_all_negative"] == 5
    assert summary["filter_groups/dropped_all_positive"] == 6


def test_nothing_dropped_reports_zeros_rather_than_nothing():
    summary = summarize_dapo_filtered_rewards({})
    assert summary["filter_groups/dropped_groups"] == 0
    assert set(summary) == {
        "filter_groups/dropped_groups",
        "filter_groups/dropped_all_positive",
        "filter_groups/dropped_all_negative",
        "filter_groups/dropped_all_zero",
    }
