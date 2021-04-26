import os
os.environ['MKL_THREADING_LAYER'] = "GNU"  # a temporary workaround

from enum import Enum
import numpy as np
from torch import multiprocessing as mp
from omegaconf import OmegaConf
from pyarrow import plasma
from typing import Callable, List
import atexit
import logging

from .vector_env import VectorEnv
from .utils import CloudpickleWrapper

logger = logging.getLogger(__name__)


class AsyncState(Enum):
    DEFAULT = 'default'
    WAITING_RESET = 'reset'
    WAITING_STEP = 'step'


class AsyncVectorEnv(VectorEnv):
    """Vectorized environment that runs multiple environments in parallel. It
    uses `multiprocessing` processes, and pipes for communication.
    """
    def __init__(
        self,
        env_funcs,
        observation_space=None,
        action_space=None,
        context: str = 'spawn',
        in_series: int = 1,
        plasma_config: OmegaConf = None
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
        env_funcs = np.array_split(env_funcs, self.num_subproc)
        ranks = np.arange(num_envs)
        ranks = np.array_split(ranks, self.num_subproc)

        # multiprocessing
        ctx = mp.get_context(context)

        self.manager_pipes, self.worker_pipes = zip(*[ctx.Pipe() for _ in range(self.num_subproc)])

        self.processes = [
            ctx.Process(
                target=worker, args=(seg_ranks, worker_pipe, manager_pipe, CloudpickleWrapper(env_func), plasma_config)
            ) for (worker_pipe, manager_pipe, env_func,
                   seg_ranks) in zip(self.worker_pipes, self.manager_pipes, env_funcs, ranks)
        ]

        for process in self.processes:
            process.daemon = True  # if the main process crashes, we should not cause things to hang
            process.start()

        for pipe in self.worker_pipes:
            pipe.close()

        self._state = AsyncState.DEFAULT

        atexit.register(self.__del__)

    def reset_async(self):
        self._assert_is_running()

        if self._state != AsyncState.DEFAULT:
            self.flush_pipe()
            logger.warn("Flushing the Pipe due to reset.")
            # raise AssertionError('Calling `reset_async` without any prior ')

        for pipe in self.manager_pipes:
            pipe.send(('reset', None))
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
        results = [pipe.recv() for pipe in self.manager_pipes]
        self._state = AsyncState.DEFAULT

    def step_async(self, actions=None) -> None:
        self._assert_is_running()
        if actions is None:
            for pipe in self.manager_pipes:
                action = [None for _ in range(self.in_series)]
                pipe.send(('step', action))
        else:
            actions = np.array_split(actions, self.num_subproc)
            for pipe, action in zip(self.manager_pipes, actions):
                pipe.send(('step', action))
        self._state = AsyncState.WAITING_STEP

    def step_wait(self):
        results = [pipe.recv() for pipe in self.manager_pipes]
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

        for pipe, seed in zip(self.manager_pipes, seeds):
            pipe.send(('seed', seed))

    def flush_pipe(self):
        if self._state == AsyncState.WAITING_RESET or self._state == AsyncState.WAITING_STEP:
            [pipe.recv() for pipe in self.manager_pipes]
            self._state = AsyncState.DEFAULT

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
        try:
            if terminate:
                for process in self.processes:
                    if process.is_alive():
                        process.terminate()
            else:
                for pipe in self.manager_pipes:
                    if (pipe is not None) and (not pipe.closed):
                        pipe.send(('close', None))
                for pipe in self.manager_pipes:
                    if (pipe is not None) and (not pipe.closed):
                        pipe.recv()

            for pipe in self.manager_pipes:
                if pipe is not None:
                    pipe.close()

            for process in self.processes:
                process.join()
        except Exception as e:
            print(f"{type(e)} has occured!")

    def _assert_is_running(self):
        if self.closed:
            raise AssertionError(
                'Trying to operate on `{0}`, after a '
                'call to `close()`.'.format(type(self).__name__)
            )

    def __del__(self):
        if not self.closed:
            self.close()


def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]


def worker(
    ranks: int,
    pipe: List[mp.Pipe],
    parent_pipe: List[mp.Pipe],
    env_fn_wrappers: List[Callable],
    plasma_config: OmegaConf = None
):
    """
    """
    import torch
    torch.set_num_threads(1)
    # use plasma object in-store
    if plasma_config:
        plasma_client = plasma.connect(f"/tmp/torchfly/plasma/{plasma_config.plasma_store_name}/plasma.sock")

    def step_env(env, action):
        observation, info, done = env.step(action)

        if plasma_config:
            observation = plasma_client.put(observation)

        return observation, info, done

    parent_pipe.close()
    envs = [env_fn_wrapper(rank) for env_fn_wrapper, rank in zip(env_fn_wrappers.x, ranks)]

    try:
        while True:
            command, data = pipe.recv()
            if command == 'step':
                pipe.send([step_env(env, action) for env, action in zip(envs, data)])
            elif command == 'reset':
                pipe.send([env.reset() for env in envs])
            elif command == 'seed':
                [env.seed() for env in envs]
            elif command == 'close':
                pipe.close()
                break
            elif command == 'get_spaces_spec':
                pipe.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space, envs[0].spec)))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    except BrokenPipeError:
        print(f"{ranks} having BrokenPipeError! Closing the environments.")
    finally:
        for env in envs:
            env.close()
