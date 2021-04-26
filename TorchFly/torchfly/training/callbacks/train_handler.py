import os
import sys
import time
import torch
import random
import numpy as np
import logging
from apex import amp
from apex.parallel import DistributedDataParallel, Reducer
# from torch.nn.parallel import DistributedDataParallel
from omegaconf import DictConfig
from typing import Any, Dict

from .events import Events
from .callback import Callback, handle_event
from ...common import move_to_device

logger = logging.getLogger(__name__)
Trainer = Any


def set_random_seed(random_seed):
    # Reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


@Callback.register("train_handler")
class TrainHandler(Callback):
    """
    Handles the distributed learning
    This callback must be included in the basic trainer
    """
    @handle_event(Events.INITIALIZE)
    def init_distributed(self, trainer: Trainer):
        pass

    @handle_event(Events.TRAIN_BEGIN, priority=199)
    def configure_distributed(self, trainer: Trainer):
        # Disable Profiling
        if not self.config.training.profiling:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)

        # Reproducibility
        if self.config.training.random_seed:
            set_random_seed(trainer.rank + self.config.training.random_seed)

        # Setup Distributed Training properties
        if self.config.training.num_gpus_per_node > 1:
            # must assume that trainer has rank property
            trainer.master = trainer.rank == 0
            torch.cuda.set_device(trainer.rank)
            trainer.device = torch.device("cuda", trainer.rank)
        else:
            assert trainer.rank == 0
            trainer.master == True
            trainer.device = torch.device("cuda")

        # Initialize Distributed Training
        if self.config.training.num_gpus_per_node > 1:
            # Init distributed
            # TODO: multi-node multi-gpu training
            torch.distributed.init_process_group(
                backend="nccl", rank=trainer.rank, world_size=self.config.training.num_gpus_per_node
            )

    @handle_event(Events.TRAIN_BEGIN, priority=190)
    def configure_optimizer(self, trainer: Trainer):
        # Optimizer
        trainer.optimizer = trainer.configure_optimizer()

    @handle_event(Events.TRAIN_BEGIN, priority=175)
    def configure_dataloader(self, trainer: Trainer):
        # Initialize Dataloader
        # DataLoader
        if trainer.train_loader is None:
            if trainer.train_loader_fn is None:
                logger.error("Please specify either `train_loader` or `train_loader_fn`!")
                raise NotImplementedError
            trainer.train_loader = trainer.train_loader_fn(self.config)

    @handle_event(Events.TRAIN_BEGIN, priority=165)
    def configure_variables(self, trainer: Trainer):
        if self.config.training.total_num_epochs > 0:
            try:
                num_training_batches = len(trainer.train_loader)
                trainer.no_epoch_training = False
                trainer.num_training_batches = num_training_batches
            except TypeError:
                # connot set the number of total_num_epoch
                # because it is impossible to know
                logger.error("Cannot get the length of train dtrainer.model")
                raise NotImplementedError("Please specify the `total_num_epochs` or `total_num_update_steps`!")
            else:
                pass
                # trainer.total_num_epochs = trainer.total_num_steps // num_training_batches

        trainer.total_num_steps = trainer.total_num_update_steps * self.config.training.gradient_accumulation_steps

        # Setup validation interval
        if self.config.training.validation_steps_interval is None or \
            self.config.training.validation_steps_interval < 0:
            # validation for every epoch
            if not trainer.no_epoch_training:
                self.config.training.validation_steps_interval = num_training_batches - 1

    # @handle_event(Events.TRAIN_BEGIN, priority=180)
    # def configure_ray(self, trainer: Trainer):
    #     # Ray Initialize
    #     if trainer.master:
    #         # close existing logging
    #         if not ray.is_initialized():
    #             logger.info(ray.init())

    @handle_event(Events.TRAIN_BEGIN, priority=150)
    def setup_model(self, trainer: Trainer):

        trainer.model = move_to_device(trainer.model, trainer.device)

        # FP16
        if self.config.training.fp16:                
            trainer.model, trainer.optimizer = amp.initialize(
                trainer.model, trainer.optimizer, opt_level=self.config.training.fp16_opt_level
            )

        if self.config.training.num_gpus_per_node > 1:
            # Distributed training (should be after apex fp16 initialization)
            trainer.model = DistributedDataParallel(trainer.model, delay_allreduce=True)
            # trainer.model = torch.nn.parallel.DistributedDataParallel(
            #     trainer.model, device_ids=[trainer.rank], output_device=trainer.rank, find_unused_parameters=True
            # )

    @handle_event(Events.TRAIN_BEGIN, priority=140)
    def configure_scheduler(self, trainer: Trainer):
        # Scheduler
        trainer.scheduler = trainer.configure_scheduler()