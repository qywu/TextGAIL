from omegaconf import OmegaConf
from pyarrow import plasma
import logging

from ..rl.vector.async_vector_env import AsyncState
from .plasma import GlobalPlasmaManager

logger = logging.getLogger(__name__)


class FlyDataLoader:
    def __init__(self, flydata_config: OmegaConf, env, collate_func=None):
        self.config = flydata_config
        self.batch_size = flydata_config.dataloader.batch_size
        self.env = env
        self.collate_func = collate_func
        self.drop_last = self.config.dataloader.drop_last
        self.plasma_client = None

        if self.config.plasma:
            self.plasma_client = GlobalPlasmaManager(
                plasma_store_name=self.config.plasma.plasma_store_name,
                use_mem_percent=self.config.plasma.use_mem_percent
            ).client

        self.started = False

    def __iter__(self):
        self.env.reset()
        self.env.step_async()
        self.started = True
        return self

    def __next__(self):
        if not self.started:
            self.env.reset()
            self.env.step_async()
            self.started = True

        observations, infos, dones = self.env.step_wait()

        # all env has ended
        if self.drop_last and any(dones):
            raise StopIteration
        elif all(dones):
            raise StopIteration
        else:
            pass

        self.env.step_async()

        if self.plasma_client is not None:
            observations = self.plasma_client.get(observations)

        if self.collate_func is not None:
            return self.collate_func(observations, infos, dones)
        else:
            return observations, infos, dones

    def __del__(self):
        self.env.close()

        # clean plasma store
        if self.plasma_client:
            existing_objects = self.plasma_client.list()
            self.plasma_client.delete(list(existing_objects.keys()))