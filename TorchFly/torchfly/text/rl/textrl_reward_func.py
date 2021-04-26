from typing import Callable
import numpy as np


class TextRLRewardFunc:
    """Assign a reward to a batch of sequences
    """
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs) -> np.array:
        rewards = self.get_reward(*args, **kwargs)
        return rewards

    def get_reward(self, *args, **kwargs) -> np.array:
        raise NotImplementedError("Please override this function before use")