from typing import Callable, Iterator
from omegaconf import DictConfig
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import deque, namedtuple
from operator import itemgetter
import logging

from torchfly.training import TrainerLoop
from torchfly.training.callbacks import Callback, CallbackHandler, Events
from torchfly.training.callbacks import Checkpoint
from torchfly.common import move_to_device
from . import TextRLReplayBuffer, TextRLRewardFunc, TextRLLogHandler

logger = logging.getLogger(__name__)

# pylint:disable=no-member


class TextRLTrainerLoop(TrainerLoop):
    """
    On Policy Text RL Trainer
    """
    def __init__(self, config, reward_func, decoder, collate_fn, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.reward_func = reward_func
        self.decoder = decoder
        self.collate_fn = collate_fn

        self.pad_token_id = self.decoder._tokenizer.pad_token_id
        self.ppo_buffer_size = self.config.text_ppo.ppo_buffer_size
        self.sample_batch_size = self.config.text_ppo.sample_batch_size
        self.ppo_mini_batch_size = self.config.text_ppo.ppo_mini_batch_size
        self.mix_human_demo_ratio = self.config.text_ppo.mix_human_demo_init_ratio
        self.mix_human_demo_ratio_decay = self.config.text_ppo.mix_human_demo_ratio_decay
        self.ppo_epsilon = self.config.text_ppo.ppo_epsilon
        self.recompute_log_probs = self.config.text_ppo.recompute_log_probs

        self.replay_buffer = TextRLReplayBuffer(max_buffer_size=self.ppo_buffer_size)

        # Configuration Check
        if self.gradient_accumulation_steps != 1:
            raise ValueError("Please set gradient accumulation steps to 1!")

        if self.training_in_epoch:
            raise ValueError(
                "Does not support epoch training! Please set config.training.total_num.num_steps bigger than 0."
            )
            
        if self.collate_fn is None:
            # Maybe it is defined in the train_dataloader
            self.collate_fn = self.train_dataloader.dataset.collate_fn

    def configure_optimizers(self):
        # The model will update multiple times in each update step
        # So we need to adjust the scheduler
        update_steps_multiplier = self.ppo_buffer_size // self.ppo_mini_batch_size 
        return self.model.configure_optimizers(self.total_num_update_steps * update_steps_multiplier)
        

    def configure_callbacks(self):
        # Callback
        # by default set up LogHandler and Checkpointer
        self.checkpoint_callback = Checkpoint(self.config)
        self.add_callback(self.checkpoint_callback)

        if self.rank == 0:
            self.log_handler = TextRLLogHandler(self.config)
            self.add_callback(self.log_handler)

    def train_epoch(self):
        self.optimizer = self.optimizers[0]
        self.scheduler = self.schedulers[0]

        self.local_step_count = 0

        iter_train_dataloader = iter(self.train_dataloader)

        while self.global_step_count < self.total_num_update_steps:
            self.callback_handler.fire_event(Events.BATCH_BEGIN)

            # Collect samples
            buffer_count = 0
            while buffer_count < self.ppo_buffer_size:
                try:
                    batch = next(iter_train_dataloader)
                except StopIteration:
                    iter_train_dataloader = iter(self.train_dataloader)
                    batch = next(iter_train_dataloader)

                self.collect_samples(batch)
                buffer_count += len(batch)

            # Train all samples in the buffer
            for mini_batch in self.replay_buffer.iterate_sample(self.ppo_mini_batch_size):

                # (state, action, action_log_prob, reward, normalized_reward)
                states, actions, action_log_probs, rewards, normalized_rewards = zip(*mini_batch)

                for i in range(len(states)):
                    states[i]["target_token_ids"] = actions[i]

                ppo_batch = self.collate_fn(states)
                ppo_batch["normalized_rewards"] = torch.LongTensor(normalized_rewards)
                ppo_batch["old_log_probs"] = torch.FloatTensor(action_log_probs)

                ppo_batch = move_to_device(ppo_batch, self.device)

                self.tmp_vars["log_dict"] = self.train_step(ppo_batch)

                self.callback_handler.fire_event(Events.STEP_BEGIN)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.callback_handler.fire_event(Events.STEP_END)

            self.callback_handler.fire_event(Events.BATCH_END)
            self.replay_buffer.clear()
            self.global_step_count += 1
            self.local_step_count += 1

    def train_step(self, batch):
        results = self.model(batch)

        # old log probilities
        log_probs = results["log_probs"]
        old_log_probs = batch["old_log_probs"]

        # advantages
        batch_rewards = batch["normalized_rewards"]
        advantages = batch_rewards

        # self-imitation
        # advantages = advantages.clamp(min=0)

        # Policy Loss
        ## shape: (batch)
        ratio = (log_probs - old_log_probs).exp()
        ## shape: (batch)
        policy_loss1 = -advantages * ratio
        ## shape: (batch)
        policy_loss2 = -advantages * ratio.clamp(1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon)
        ## shape: (batch)
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()

        loss = policy_loss + results["loss"]

        # Backward Loss
        self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
        self.loss_backward(loss)
        self.callback_handler.fire_event(Events.BACKWARD_END)

        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > self.ppo_epsilon).float().mean()
            approx_kl = (log_probs - old_log_probs).pow(2).mean()

        log_dict = {}
        log_dict["loss"] = loss.item()
        log_dict["policy_loss"] = policy_loss.item()
        log_dict["clip_frac"] = clip_frac.item()
        log_dict["approx_kl"] = approx_kl.item()

        return log_dict

    @torch.no_grad()
    def collect_samples(self, batch):
        """Generate samples, collect rewards, and update replay buffer"""

        num_human_demos = int(len(batch) * self.mix_human_demo_ratio)
        # num_generations = int(len(batch) * (1 - self.mix_human_demo_ratio))

        self.mix_human_demo_ratio *= self.mix_human_demo_ratio_decay

        actual_sample_size = min(num_human_demos, self.sample_batch_size)
        for i in range(0, num_human_demos, actual_sample_size):
            # Update Buffer for Human Demos
            human_demos_batch = batch[i:i + actual_sample_size]

            # collect human demos log probs
            human_demos_batch_collated = self.collate_fn(human_demos_batch)
            human_demos_batch_collated = move_to_device(human_demos_batch_collated, self.device)
            log_probs = self.model.compute_log_probs(human_demos_batch_collated)["log_probs"]
            human_log_probs = log_probs.tolist()

            human_tokens = [item["target_token_ids"] for item in human_demos_batch]

            if self.config.text_ppo.constant_human_demo_reward:
                rewards = np.ones((len(human_demos_batch))) * 2.0

                self.replay_buffer.update_batch(
                    states=human_demos_batch,
                    actions=human_tokens,
                    action_log_probs=human_log_probs,
                    rewards=rewards,
                    normalize_reward=False
                )
            else:
                results = {}
                results["tokens"] = human_tokens
                rewards = self.reward_func(human_demos_batch_collated, results)
                self.replay_buffer.update_batch(
                    states=human_demos_batch, actions=human_tokens, action_log_probs=human_log_probs, rewards=rewards
                )

        # Update Buffer for Generations
        actual_sample_size = min(len(batch) - num_human_demos, self.sample_batch_size)
        for i in range(num_human_demos, len(batch), actual_sample_size):
            sample_batch = batch[i:i + actual_sample_size]
            sample_batch_collated = self.collate_fn(sample_batch)
            sample_batch_collated = move_to_device(sample_batch_collated, self.device)
            results = self.decoder_generate(sample_batch_collated)

            # TODO: Consider num_return_sequences
            results["tokens"] = [item[0] for item in results["tokens"]]

            if self.recompute_log_probs:
                for i in range(len(batch)):
                    sample_batch_collated["target_token_ids"] = pad_sequence(
                        results["tokens"], batch_first=True, padding_value=self.pad_token_id
                    ).to(self.device)

                log_probs = self.model.compute_log_probs(sample_batch_collated)["log_probs"]
                results["log_probs"] = log_probs.tolist()
            else:
                results["log_probs"] = [item[0] for item in results["log_probs"]]

            rewards = self.reward_func(sample_batch_collated, results)

            self.replay_buffer.update_batch(
                states=sample_batch, actions=results["tokens"], action_log_probs=results["log_probs"], rewards=rewards
            )

    def decoder_generate(self, batch):
        # force it not to use beam search
        results = self.decoder.generate(input_ids=batch["source_token_ids"])
        return results
