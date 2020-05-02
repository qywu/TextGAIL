from typing import Callable, Iterator
from omegaconf import DictConfig
import torch
import numpy as np
from collections import deque, namedtuple
import logging

from torchfly.training import TrainerLoop
from torchfly.training.callbacks import Callback, CallbackHandler, Events
from torchfly.training.callbacks import Checkpoint
from torchfly.common import move_to_device

logger = logging.getLogger(__name__)

TextSample = namedtuple('TextSample', ['state', 'action', 'action_log_prob', 'reward', 'normalized_reward'])


class TextRewardFunc:
    """Assign a reward to a batch of sequences
    """
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        rewards = self.calc_reward(*args, **kwargs)
        return rewards

    def calc_reward(self, is_human_demo: bool = False, *args, **kwargs):
        raise NotImplementedError("Please override this function before use")


class TextRLExperienceBuffer:
    """
    We need to store (state, action, action_log_probs, reward, and normalized_reward)
    All rewards are normalized with running mean and std (Important for RL)
    We use momentum so that the running stats only depends on the recent data
    """
    def __init__(self, max_buffer_size=512, momentum=0.90):
        self.max_buffer_size = max_buffer_size
        #self.buffer = [deque(maxlen=self.max_buffer_size)]
        self.buffer = []
        self.momentum = momentum
        self.reward_mean = 0.0
        self.reward_mean_sq = 0.0
        self.reward_std = 1.0

    def update_batch(self, states, actions, action_log_probs, rewards, normalize_reward=True):
        if normalize_reward:
            batch_momentum = self.momentum**len(rewards)
            self.reward_mean = self.reward_mean * batch_momentum + np.mean(rewards) * (1 - batch_momentum)
            self.reward_mean_sq = self.reward_mean_sq * batch_momentum + np.mean(rewards**2) * (1 - batch_momentum)
            self.reward_std = (self.reward_mean_sq - self.reward_mean**2)**0.5
            normalized_rewards = (rewards - self.reward_mean) / self.reward_std
        else:
            normalized_rewards = rewards

        self.buffer.extend(zip(states, actions, action_log_probs, rewards, normalized_rewards))

    def update(self, state, action, action_log_prob, reward, normalize_reward=True):
        if normalize_reward:
            self.reward_mean = self.reward_mean * self.momentum + reward * (1 - self.momentum)
            self.reward_mean_sq = self.reward_mean_sq * self.momentum + (reward**2) * (1 - self.momentum)
            self.reward_std = (self.reward_mean_sq - self.reward_mean**2)**0.5
            normalized_reward = (reward - self.reward_mean) / self.reward_std
        else:
            normalize_reward = reward

        self.buffer.append((state, action, action_log_prob, reward, normalized_reward))

    def __getitem__(self, index):
        return self.buffer[index]

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []

    def iterate_sample(self, mini_batch_size, shuffle=False) -> Iterator:
        """
        A mini batch iterator
        """
        indices = np.arange(len(self.buffer))
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(self.buffer), mini_batch_size):
            sampled_indices = indices[i:i + mini_batch_size]
            # get sampled batch
            yield self.buffer[sampled_indices]


class TextRLTrainerLoop(TrainerLoop):
    """
    On Policy Text RL Trainer
    """
    def __init__(self, config, reward_func, decoder, collator, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.replay_buffer = TextRLExperienceBuffer(max_buffer_size=512)
        self.reward_func = reward_func
        self.decoder = decoder
        self.collator = collator

        self.ppo_buffer_size = self.config.text_ppo.ppo_buffer_size
        self.sample_batch_size = self.config.text_ppo.sample_batch_size
        self.ppo_mini_batch_size = self.config.text_ppo.ppo_mini_batch_size
        self.mix_human_demo_ratio = self.config.text_ppo.mix_human_demo_init_ratio
        self.mix_human_demo_ratio_decay = self.config.text_ppo.mix_human_demo_ratio_decay
        self.ppo_epsilon = self.config.text_ppo.ppo_epsilon

    def configure_callbacks(self):
        # Callback
        # by default set up LogHandler and Checkpointer
        self.checkpoint_callback = Checkpoint(self.config)
        self.add_callback(self.checkpoint_callback)

    def train_epoch(self):
        self.optimizer = self.optimizers[0]
        self.scheduler = self.schedulers[0]

        self.local_step_count = 0

        iter_train_dataloader = iter(self.train_dataloader)

        while self.global_step_count < self.total_num_update_steps:
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
            for batch in self.replay_buffer.iterate_sample(self.ppo_mini_batch_size):
                pass

            # for sampled_indices, batch in buffer_loader(
            #             range(len(buffer)),
            #             mini_batch_size=self.ppo_mini_batch_size,

            self.replay_buffer.clear()

        for batch in self.train_dataloader:
            self.callback_handler.fire_event(Events.BATCH_BEGIN)

            self.collect_samples(batch)

            self.global_step_count += 1
            self.local_step_count += 1

    def train_step(self, batch):
        results = self.model(batch)

        # old log probilities
        log_probs = batch["log_probs"]
        old_log_probs = batch["old_log_probs"]

        # advantages
        batch_rewards = batch["normalized_rewards"].to(self.device)
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
        self.loss_backward(loss)

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

        for i in range(0, num_human_demos, self.sample_batch_size):
            # Update Buffer for Human Demos
            human_demos_batch = batch[i:i + self.sample_batch_size]

            human_log_probs = [torch.zeros((item["source_token_ids"].shape[0])) for item in human_demos_batch]
            human_tokens = [item["source_token_ids"] for item in human_demos_batch]

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
                results["tokens"] = [item["target_token_ids"] for item in human_demos_batch]
                rewards = self.reward_func(human_demos_batch, results, is_human_demo=True)
                self.replay_buffer.update_batch(
                    states=human_demos_batch, actions=human_tokens, action_log_probs=human_log_probs, rewards=rewards
                )
        # Update Buffer for Generations
        for i in range(num_human_demos, len(batch), self.sample_batch_size):
            sample_batch = batch[i:i + self.sample_batch_size]
            sample_batch_collated = self.collator.sample_collate(sample_batch)
            sample_batch_collated = move_to_device(sample_batch_collated, self.device)
            results = self.decoder_generate(sample_batch_collated)

            # TODO: Conside num_return_sequences
            results["tokens"] = [item[0] for item in results["tokens"]]
            results["log_probs"] = [item[0] for item in results["log_probs"]]
            rewards = self.reward_func(sample_batch, results, is_human_demo=False)

            self.replay_buffer.update_batch(
                states=sample_batch, actions=results["tokens"], action_log_probs=results["log_probs"], rewards=rewards
            )

    def decoder_generate(self, batch):
        # force it not to use beam search
        results = self.decoder.generate(input_ids=batch["source_token_ids"], num_beams=1)
        return results
