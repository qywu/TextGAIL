from typing import Any, Dict
import os
import sys
import time
import torch
import logging
from omegaconf import DictConfig
# from apex import amp

from ..checkpointer import Checkpointer
from .events import Events
from .callback import Callback, handle_event

logger = logging.getLogger("checkpointer")

__all__ = ["Checkpoint"]
Trainer = Any


@Callback.register("checkpoint")
class Checkpoint(Callback):
    """
    Callback that handles Checkpointing
    """
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        self.checkpoint_dir = self.config.training.checkpointing.directory
        if self.checkpoint_dir is None:
            self.checkpoint_dir = "Checkpoints"

        self.rank = int(os.environ.get("RANK", 0))
        self.fix_amp_bug = False

        if self.config.training.checkpointing.async_save is None:
            self.config.training.checkpointing.async_save = False

        # Initialize Checkpointer
        self.checkpointer = Checkpointer(
            sync_every_save=True,
            async_save=self.config.training.checkpointing.async_save,
            num_checkpoints_to_keep=self.config.training.checkpointing.num_checkpoints_to_keep,
            keep_checkpoint_every_num_seconds=(self.config.training.checkpointing.keep_checkpoint_every_num_seconds),
            storage_dir=self.checkpoint_dir
        )

        # checkpointed states contain two parts: model and training progress
        self.restored_states = None

        # setup the timer
        if self.rank == 0:
            # only rank 0 can save files
            self.last_save_time = time.time()

    @handle_event(Events.INITIALIZE, priority=199)
    def setup_checkpointer(self, trainer: Trainer):
        # Checkpoint in epochs or steps
        if self.config.training.checkpointing.steps_interval < 0 and self.config.training.checkpointing.seconds_interval < 0:
            self.checkpoint_in_epoch = True
        else:
            self.checkpoint_in_epoch = False

        # Checkpoint in seconds or steps
        if self.config.training.checkpointing.steps_interval > 0 and self.config.training.checkpointing.seconds_interval > 0:
            raise ValueError(
                "Either `checkpointing.steps_interval` or `checkpointing.seconds_interval` can be set greater than 0!"
            )
        elif self.config.training.checkpointing.steps_interval < 0 and self.config.training.checkpointing.seconds_interval > 0:
            self.checkpoint_in_seconds = True
        elif self.config.training.checkpointing.steps_interval > 0 and self.config.training.checkpointing.seconds_interval < 0:
            self.checkpoint_in_seconds = False
        else:
            self.checkpoint_in_seconds = False

        # Search for the latest checkpoint
        if self.config.training.resume.resume:
            logger.info("Try to restore the latest checkpoint")
            self.restored_states = self.checkpointer.restore_latest_checkpoint()
            if self.restored_states:
                file_path = self.restored_states[2]
                print(f"RANK {self.rank} has found checkpoint at {file_path}!")
                self.checkpointer.load_state_dict(self.restored_states[1]["checkpointer_state_dict"])
            else:
                logger.warn("Fail to find any checkpoint! Start new training!")
        else:
            self.restored_states = None

    @handle_event(Events.TRAIN_BEGIN, priority=170)
    def load_checkpoint(self, trainer: Trainer):
        # Load the model
        # Resume the training
        if self.restored_states and self.config.training.resume.resume:
            # Model State
            if self.config.training.resume.resume_model:
                trainer.set_model_state(self.restored_states[0])
            # Everything Else
            trainer.set_trainer_state(self.restored_states[1])

            file_path = self.restored_states[2]
            print(f"RANK {self.rank} has loaded checkpoint at {file_path}!")

    @handle_event(Events.BATCH_END)
    def save_checkpoint(self, trainer: Trainer):
        # Checkpointing
        if self.rank == 0:
            if not self.checkpoint_in_epoch:
                if self.checkpoint_in_seconds:
                    current_time = time.time()
                    # the elapsed time is longer than the seconds
                    if (current_time - self.last_save_time) > self.config.training.checkpointing.seconds_interval:
                        self._save_trainer_state(trainer)
                        self.last_save_time = current_time
                else:
                    if (trainer.global_step_count + 1) % self.config.training.checkpointing.steps_interval == 0:
                        self._save_trainer_state(trainer)

    @handle_event(Events.EPOCH_END)
    def save_checkpoint_epoch(self, trainer: Trainer):
        # Checkpointing
        if self.rank == 0:
            if self.checkpoint_in_epoch:
                self._save_trainer_state(trainer)

    def _save_trainer_state(self, trainer: Trainer):

        trainer_state_dict = trainer.get_trainer_state()
        self.checkpointer.save_checkpoint(
            "iter_" + str(trainer.global_step_count), trainer.get_model_state(), trainer_state_dict
        )
        logger.info(f"Saved Checkpoint for Epoch {trainer.epochs_trained + 1} Iteration {trainer.global_step_count}!")

    def state_dict(self):
        return self.checkpointer.state_dict()

    def load_state_dict(self, state_dict):
        self.checkpointer.load_state_dict(state_dict)