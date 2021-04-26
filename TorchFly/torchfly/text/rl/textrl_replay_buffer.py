from typing import Callable, Iterator
from omegaconf import DictConfig
import numpy as np
from operator import itemgetter
from collections import namedtuple

TextRLSample = namedtuple('TextRLSample', ['state', 'action', 'action_log_prob', 'reward', 'normalized_reward'])


class TextRLReplayBuffer:
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
            self.reward_std = np.abs(self.reward_mean_sq - self.reward_mean**2)**0.5
            normalized_rewards = (rewards - self.reward_mean) / (self.reward_std + 1e-5)
            normalized_rewards = np.clip(normalized_rewards, -2.0, 2.0)
        else:
            normalized_rewards = rewards

        self.buffer.extend(zip(states, actions, action_log_probs, rewards, normalized_rewards))

    def update(self, state, action, action_log_prob, reward, normalize_reward=True):
        if normalize_reward:
            self.reward_mean = self.reward_mean * self.momentum + reward * (1 - self.momentum)
            self.reward_mean_sq = self.reward_mean_sq * self.momentum + (reward**2) * (1 - self.momentum)
            self.reward_std = np.abs(self.reward_mean_sq - self.reward_mean**2)**0.5
            normalized_reward = (reward - self.reward_mean) / (self.reward_std + 1e-5)
            normalized_reward = np.clip(normalized_reward, -2.0, 2.0)
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
            yield itemgetter(*sampled_indices)(self.buffer)
