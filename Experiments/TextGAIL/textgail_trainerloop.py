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
from torchfly.text.rl import TextRLReplayBuffer, TextRLLogHandler

logger = logging.getLogger(__name__)

# pylint:disable=no-member


class TextGAILTrainerLoop(TrainerLoop):
    """
    On Policy Text RL Trainer
    """
    def __init__(self, config, reward_func, decoder_helper, collate_fn=None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.reward_func = reward_func
        self.decoder_helper = decoder_helper
        self.collate_fn = collate_fn

        self.pad_token_id = self.decoder_helper._tokenizer.pad_token_id

        self.ppo_epoch = self.config.text_gail.ppo_epoch
        self.ppo_buffer_size = self.config.text_gail.ppo_buffer_size
        self.sample_batch_size = self.config.text_gail.sample_batch_size
        self.ppo_mini_batch_size = self.config.text_gail.ppo_mini_batch_size
        self.mix_human_demo_ratio = self.config.text_gail.mix_human_demo_init_ratio
        self.mix_human_demo_ratio_warmup_steps = self.config.text_gail.mix_human_demo_ratio_warmup_steps
        self.ppo_epsilon = self.config.text_gail.ppo_epsilon
        self.recompute_log_probs = self.config.text_gail.recompute_log_probs
        self.discriminator_pretrain_steps = self.config.text_gail.discriminator_pretrain_steps
        self.constant_human_demo_reward = self.config.text_gail.constant_human_demo_reward
        self.done_discriminator_pretrain = False

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
        update_steps_multiplier = self.config.text_gail.ppo_buffer_size // self.config.text_gail.ppo_mini_batch_size
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
        self.D_optimizer = self.optimizers[0]
        self.G_optimizer = self.optimizers[1]
        self.D_scheduler = self.schedulers[0]
        self.G_scheduler = self.schedulers[1]

        prop_decay = self.mix_human_demo_ratio / self.total_num_update_steps

        # total counts
        self.local_step_count = 0
        self.generator_step_count = 0
        self.discriminator_step_count = 0

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
                    self.epochs_trained += 1
                    logger.info(f"{self.epochs_trained} has finished!")
                    batch = next(iter_train_dataloader)

                self.collect_samples(batch)
                buffer_count += len(batch)

            # Discriminator Warmup
            if not self.done_discriminator_pretrain:
                for mini_batch in self.replay_buffer.iterate_sample(self.ppo_mini_batch_size):
                    # (state, action, action_log_prob, reward, normalized_reward)
                    states, actions, action_log_probs, rewards, normalized_rewards = zip(*mini_batch)
                    self.tmp_vars["log_dict"] = self.train_discriminator_step(states, actions)

                    if (self.discriminator_step_count + 1) % self.gradient_accumulation_steps == 0:
                        # self.callback_handler.fire_event(Events.STEP_BEGIN)
                        self.D_optimizer.step()
                        self.D_scheduler.step()
                        self.D_optimizer.zero_grad()
                        # self.callback_handler.fire_event(Events.STEP_END)

                        self.discriminator_step_count += 1

                    if self.discriminator_step_count >= self.discriminator_pretrain_steps:
                        self.done_discriminator_pretrain = True
                        break
            else:
                "Generator Training"
                for _ in range(self.ppo_epoch):
                    # Train the Generator
                    for mini_batch in self.replay_buffer.iterate_sample(self.ppo_mini_batch_size):

                        # (state, action, action_log_prob, reward, normalized_reward)
                        states, actions, action_log_probs, rewards, normalized_rewards = zip(*mini_batch)

                        ppo_batch = self.collate_fn(states)

                        ppo_batch["target_token_ids"] = pad_sequence(
                            actions, batch_first=True, padding_value=self.pad_token_id
                        )
                        ppo_batch["normalized_rewards"] = torch.LongTensor(normalized_rewards)
                        ppo_batch["old_log_probs"] = torch.FloatTensor(action_log_probs)

                        ppo_batch = move_to_device(ppo_batch, self.device)

                        self.tmp_vars["log_dict"] = self.train_generator_step(ppo_batch)

                        if (self.generator_step_count + 1) % self.gradient_accumulation_steps == 0:
                            # self.callback_handler.fire_event(Events.STEP_BEGIN)
                            self.G_optimizer.step()
                            self.G_scheduler.step()
                            self.G_optimizer.zero_grad()
                            # self.callback_handler.fire_event(Events.STEP_END)

                            self.generator_step_count += 1

                "Discriminator Training"
                for mini_batch in self.replay_buffer.iterate_sample(self.ppo_mini_batch_size):
                    states, actions, action_log_probs, rewards, normalized_rewards = zip(*mini_batch)
                    log_dict = self.train_discriminator_step(states, actions)
                    self.tmp_vars["log_dict"].update(log_dict)

                    if (self.discriminator_step_count + 1) % self.gradient_accumulation_steps == 0:
                        # self.callback_handler.fire_event(Events.STEP_BEGIN)
                        self.D_optimizer.step()
                        self.D_scheduler.step()
                        self.D_optimizer.zero_grad()
                        # self.callback_handler.fire_event(Events.STEP_END)

                        self.discriminator_step_count += 1

            # update human mix_human_ratio
            self.mix_human_demo_ratio -= prop_decay 
            self.tmp_vars["log_dict"]["mix_human_demo_ratio"] = self.mix_human_demo_ratio

            self.callback_handler.fire_event(Events.BATCH_END)
            self.replay_buffer.clear()

            # Only rank 0 can run the validation dataset
            if self.rank == 0:
                if (self.global_step_count + 1) % self.validation_steps_interval == 0:
                    if not self.validation_dataloader is None:
                        self.model.eval()
                        # BEGIN
                        self.callback_handler.fire_event(Events.VALIDATE_BEGIN)

                        self.tmp_vars["validate_metrics"] = self.validate()

                        self.callback_handler.fire_event(Events.VALIDATE_END)
                        self.model.train()
            
            self.global_step_count += 1
            self.local_step_count += 1

    def train_discriminator_step(self, states, actions):
        results = self.reward_func.get_loss(states, actions)

        self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
        loss = results["loss"] / self.gradient_accumulation_steps
        self.loss_backward(loss)
        self.callback_handler.fire_event(Events.BACKWARD_END)
        # return the results
        log_dict = {"discriminator/loss": loss.item() * self.gradient_accumulation_steps}

        return log_dict

    def train_generator_step(self, batch):
        results = self.model.generator(batch)

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
        loss = loss / self.gradient_accumulation_steps
        self.loss_backward(loss)
        self.callback_handler.fire_event(Events.BACKWARD_END)

        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > self.ppo_epsilon).float().mean()
            approx_kl = (log_probs - old_log_probs).pow(2).mean()

        log_dict = {}
        log_dict["generator/loss"] = loss.item() * self.gradient_accumulation_steps
        log_dict["generator/policy_loss"] = policy_loss.item()
        log_dict["generator/clip_frac"] = clip_frac.item()
        log_dict["generator/approx_kl"] = approx_kl.item()

        return log_dict

    @torch.no_grad()
    def collect_samples(self, batch):
        """Generate samples, collect rewards, and update replay buffer"""

        num_human_demos = int(len(batch) * self.mix_human_demo_ratio)
        # num_generations = int(len(batch) * (1 - self.mix_human_demo_ratio))

        actual_sample_size = min(num_human_demos, self.sample_batch_size)
        if actual_sample_size > 0:
            for i in range(0, num_human_demos, actual_sample_size):
                # Update Buffer for Human Demos
                human_demos_batch = batch[i:i + actual_sample_size]

                # collect human demos log probs
                human_demos_batch_collated = self.collate_fn(human_demos_batch)
                human_demos_batch_collated = move_to_device(human_demos_batch_collated, self.device)
                log_probs = self.model.generator.compute_log_probs(human_demos_batch_collated)["log_probs"]
                human_log_probs = log_probs.tolist()

                human_tokens = [item["target_token_ids"] for item in human_demos_batch]

                if self.constant_human_demo_reward:
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
                    rewards = self.reward_func.get_reward(human_demos_batch, results)
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

            # recompute the log probs for better precision
            if self.recompute_log_probs:
                temp_target_token_ids = sample_batch_collated["target_token_ids"]
                sample_batch_collated["target_token_ids"] = pad_sequence(
                    results["tokens"], batch_first=True, padding_value=self.pad_token_id
                ).to(self.device)

                log_probs = self.model.generator.compute_log_probs(sample_batch_collated)["log_probs"]
                results["log_probs"] = log_probs.tolist()
                # we switch back the original target_token_ids
                sample_batch_collated["target_token_ids"] = temp_target_token_ids
            else:
                results["log_probs"] = [item[0] for item in results["log_probs"]]

            rewards = self.reward_func.get_reward(sample_batch, results)

            self.replay_buffer.update_batch(
                states=sample_batch, actions=results["tokens"], action_log_probs=results["log_probs"], rewards=rewards
            )

    def decoder_generate(self, batch):
        # force it not to use beam search
        results = self.decoder_helper.generate(input_ids=batch["source_token_ids"])
        return results

    def validate(self):
        # Validation
        self.model.eval()
        # No gradient is needed for validation
        with torch.no_grad():
            for batch in self.validation_dataloader:
                batch = self.collate_fn(batch)
                # send to cuda device
                batch = move_to_device(batch, self.device)

                if self.distributed_training:
                    self.model.module.predict(batch)
                else:
                    self.model.predict(batch)
        #END
        # get metrics
        if self.distributed_training:
            metrics = self.model.module.get_metrics(reset=True)
        else:
            metrics = self.model.get_metrics(reset=True)
        return metrics
