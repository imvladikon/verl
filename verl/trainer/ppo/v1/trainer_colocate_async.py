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
import logging

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

logger = logging.getLogger(__name__)


def reject_unmerged_lora_adapter(config) -> None:
    """Async rollout cannot unload a LoRA adapter while partial rollout keeps requests alive.

    After on_sample_end aborts, the client immediately resubmits the aborted requests, so they sit
    in the paused engine holding the adapter; the unload_lora_adapter inside the weight sync then
    times out. Merged LoRA does not hit this: the rollout receives full weights and no adapter is
    unloaded.
    """
    model = config.actor_rollout_ref.model
    lora = model.get("lora", {}) or {}
    rank = max(int(model.get("lora_rank", 0) or 0), int(lora.get("rank", 0) or 0))
    if rank <= 0 or bool(lora.get("merge", False)):
        return
    raise ValueError(
        "trainer_mode=colocate_async requires actor_rollout_ref.model.lora.merge=true when LoRA is "
        f"enabled (rank={rank}). Serving the adapter separately deadlocks the weight sync: the "
        "partial-rollout retries hold the adapter in the paused engine and unload_lora_adapter times "
        "out. Use the merged path, or run trainer_mode=sync."
    )


def warn_about_stale_prefix_cache(config) -> None:
    """Partial rollout keeps the queue busy, so the post-sync cache flush never runs.

    SGLang flushes only when the scheduler is fully idle; under partial rollout the aborted
    requests are resubmitted immediately, so after every weight update the radix cache still holds
    KV computed with the previous weights, and an unrelated request sharing a prefix reuses it.
    """
    engine_kwargs = (config.actor_rollout_ref.rollout.get("engine_kwargs", {}) or {}).get("sglang", {}) or {}
    if engine_kwargs.get("disable_radix_cache", False):
        return
    logger.warning(
        "colocate_async keeps the radix cache enabled: after each weight update it still holds KV "
        "computed with the previous weights (the flush needs an idle scheduler, and partial-rollout "
        "retries keep it busy), so requests sharing a prefix can reuse stale entries. Set "
        "engine_kwargs.sglang.disable_radix_cache=true to rule that out, at the cost of recomputing "
        "shared prefixes."
    )


@register_trainer("colocate_async")
class PPOTrainerColocateAsync(PPOTrainer):
    """Asynchronous PPO trainer
    1. Trainer and rollout are colocated.
    2. Partial rollout is enabled.
    """

    def get_llm_client(self):
        """Get the LLM server client for rollout generation."""
        return self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def on_init_end(self):
        reject_unmerged_lora_adapter(self.config)
        warn_about_stale_prefix_cache(self.config)
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_train_begin(self):
        self._add_async_warmup_batches(self.config.trainer.v1.colocate_async.num_warmup_batches)

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)
            # resume generation
            self.checkpoint_manager.resume_generation_replicas()

    def on_sample_end(self):
        # abort all unfinished requests and pause generation
        self.checkpoint_manager.abort_replicas()
        # sleep all replicas to discard weights and kv cache
        self.checkpoint_manager.sleep_replicas()
        if self.curr_step_profile:
            self._stop_rollout_profiling()
