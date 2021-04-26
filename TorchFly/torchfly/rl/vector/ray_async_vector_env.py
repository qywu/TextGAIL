import os
import ray
import numpy as np
from omegaconf import OmegaConf

from typing import Callable, List
from enum import Enum

import atexit
import logging

from torchfly.rl.vector.vector_env import VectorEnv

# pylint:disable=no-member

logger = logging.getLogger(__name__)


class AsyncState(Enum):
    DEFAULT = 'default'
    WAITING_RESET = 'reset'
    WAITING_STEP = 'step'


class RayAsyncVectorEnv(VectorEnv):
    """Vectorized environment that runs multiple environments in parallel. It
    uses `multiprocessing` processes, and pipes for communication.
    """
    def __init__(
        self,
        env_funcs,
        queue,
        observation_space=None,
        action_space=None,
        in_series: int = 1,
    ):
        """
        Args:
            env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
            in_series: number of environments to run in series in a single process
                (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        super().__init__(num_envs=len(env_funcs), observation_space=observation_space, action_space=action_space)
        self.closed = False
        self.in_series = in_series

        num_envs = len(env_funcs)

        assert num_envs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.num_subproc = num_envs // in_series
        all_env_funcs = np.array_split(env_funcs, self.num_subproc)
        all_ranks = np.arange(num_envs)
        all_ranks = np.array_split(all_ranks, self.num_subproc)

        # multiprocessing
        self.workers = [SeriesWorker.remote(ranks, env_funcs, queue) for env_funcs, ranks in zip(all_env_funcs, all_ranks)]
        self.ray_obj_ids = None

        self._state = AsyncState.DEFAULT

        atexit.register(self.__del__)

    def reset_async(self):
        self._assert_is_running()

        if self._state != AsyncState.DEFAULT:
            self.ray_obj_ids = None
            logger.warn("Flushing due to reset.")
            # raise AssertionError('Calling `reset_async` without any prior ')

        self.ray_obj_ids = [worker.reset.remote() for worker in self.workers]

        # waiting state
        self._state = AsyncState.WAITING_RESET

    def reset_wait(self, timeout=None):
        """
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise AssertionError(
                'Calling `reset_wait` without any prior '
                'call to `reset_async`.', AsyncState.WAITING_RESET.value
            )
        results = ray.get(self.ray_obj_ids)
        self._state = AsyncState.DEFAULT
        return results

    def step_async(self, actions=None) -> None:
        self._assert_is_running()

        if actions is None:
            actions = [None for _ in range(self.in_series)]
            self.ray_obj_ids = [worker.step.remote(actions) for worker in self.workers]
        else:
            actions = np.array_split(actions, self.num_subproc)
            self.ray_obj_ids = [worker.step.remote(action) for worker, action in zip(self.workers, actions)]

        self._state = AsyncState.WAITING_STEP

    def step_wait(self):
        results = ray.get(self.ray_obj_ids)
        results = _flatten_list(results)
        observations, infos, dones = zip(*results)
        self._state = AsyncState.DEFAULT
        return observations, infos, dones

    def seed(self, seeds=None):
        self._assert_is_running()

        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]

        elif isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]

        seeds = np.array_split(seeds, self.num_subproc)

        assert len(seeds) == self.num_envs

        if self._state != AsyncState.DEFAULT:
            raise AssertionError(
                'Calling `seed` while waiting '
                'for a pending call to `{0}` to complete.'.format(self._state.value), self._state.value
            )

        self.ray_obj_ids = [worker.seed.remote(seed) for worker, seed in zip(self.workers, seeds)]

    def close_extras(self, timeout=None, terminate=False):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `close` times out. If `None`,
            the call to `close` never times out. If the call to `close` times
            out, then all processes are terminated.
        terminate : bool (default: `False`)
            If `True`, then the `close` operation is forced and all processes
            are terminated.
        """
        del self.workers

    def _assert_is_running(self):
        if self.closed:
            raise AssertionError(
                'Trying to operate on `{0}`, after a '
                'call to `close()`.'.format(type(self).__name__)
            )

    def __del__(self):
        if not self.closed:
            self.close()


@ray.remote
class SeriesWorker:
    def __init__(self, ranks, env_funcs, queue):
        self.envs = [env_fn_wrapper(rank, queue) for env_fn_wrapper, rank in zip(env_funcs, ranks)]
        atexit.register(self.__del__)

    def step(self, actions):
        return [env.step(action) for env, action in zip(self.envs, actions)]

    def reset(self):
        return [env.reset() for env in self.envs]  

    def seed(self):
        for env in self.envs:
            env.seed()

    def get_spaces_spec(self):
        return (self.envs[0].observation_space, self.envs[0].action_space, self.envs[0].spec)

    def __del__(self):
        if hasattr(self, "envs"):
            for env in self.envs:
                env.close()


def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]
