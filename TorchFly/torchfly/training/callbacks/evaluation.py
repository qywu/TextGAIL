from typing import Any, Dict
import os
import sys
import math
import time
import torch
import logging
from omegaconf import DictConfig

from .events import Events
from .callback import Callback, handle_event

logger = logging.getLogger(__name__)

Trainer = Any


@Callback.register("evaluation")
class Evaluation(Callback):
    """
    Callback that handles Checkpointing
    """
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.last_save_time = time.time()
        self.started = False

    @handle_event(Events.INITIALIZE, priority=199)
    def setup_evaluation(self, trainer: Trainer):
        # Checkpoint in seconds or steps
        if self.config.training.evaluation.steps_interval > 0:
            self.evaluation_in_seconds = False
        else:
            if (
                not hasattr(self.config.training.evaluation, "seconds_interval")
            ) or self.config.training.evaluation.seconds_interval < 0:
                self.evaluation_in_seconds = False
            else:
                self.evaluation_in_seconds = True

        # evaluation steps interval
        if self.config.training.evaluation.after_num_steps is None:
            self.evaluation_after_num_steps = 0
        else:
            self.evaluation_after_num_steps = self.config.training.evaluation.after_num_steps

        if self.config.training.evaluation.steps_interval < 0 and self.config.training.evaluation.seconds_interval < 0:
            self.evaluate_in_epoch = trainer.training_in_epoch
            if self.evaluate_in_epoch == False:
                raise ValueError(
                    "Please set either `config.training.evaluation.steps_interval` or `config.training.evaluation.seconds_interval`"
                )
        else:
            self.evaluate_in_epoch = False

    @handle_event(Events.TRAIN_BEGIN)
    def on_train_begin(self, trainer):
        # Start validation at the begining
        if not self.started:
            self._evaluation(trainer)
            self.started = True

    @handle_event(Events.BATCH_END)
    def on_batch_end(self, trainer: Trainer):
        if not self.evaluate_in_epoch:
            # Check evaluation
            if trainer.global_step_count > self.evaluation_after_num_steps:
                if self.evaluation_in_seconds:
                    current_time = time.time()
                    # the elapsed time is longer than the seconds
                    if (current_time - self.last_save_time) > self.config.training.evaluation.seconds_interval:
                        self._evaluation(trainer)
                        self.last_save_time = current_time
                else:
                    if (trainer.global_step_count + 1) % self.config.training.evaluation.steps_interval == 0:
                        self._evaluation(trainer)

    @handle_event(Events.EPOCH_END)
    def on_epoch_end(self, trainer: Trainer):
        if self.evaluate_in_epoch:
            self._evaluation(trainer)

    def _evaluation(self, trainer):
        if trainer.validation_dataloader is not None:
            trainer.validate()

        if trainer.test_dataloader is not None:
            trainer.test()

    def state_dict(self):
        state_dict = {"started": self.started}
        return state_dict

    def load_state_dict(self, state_dict):
        self.started = state_dict["started"]