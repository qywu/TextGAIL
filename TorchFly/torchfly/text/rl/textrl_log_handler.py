from typing import Any, Dict
import os
import sys
import time
import tqdm
import datetime
import collections
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
import logging
import atexit

from torchfly.training.callbacks import Callback, Events, handle_event

logger = logging.getLogger("torchfly.training.logger")
Trainer = Any


def get_rank():
    """
    We use environment variables to pass the rank info
    Returns:
        rank: rank in the multi-node system 
        local_rank: local rank on a node
    """
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        rank = 0
        local_rank = 0
        os.environ["RANK"] = str(0)
        os.environ["LOCAL_RANK"] = str(0)
    else:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

    return rank, local_rank


class TextRLLogHandler(Callback):
    """
    Callback that handles all Tensorboard logging.
    """
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.rank, _ = get_rank()
        self.cumulative_time = 0.0

        self.history_log_dict = {}
        self.smooth_coef = 0.95

        # Log in seconds or steps
        if config.training.logging.steps_interval > 0:
            self.log_in_seconds = False
        else:
            if config.training.logging.seconds_interval < 0:
                raise NotImplementedError("config.training.logging.seconds_interval must be larger than 0")
            self.log_in_seconds = True

        # Train in epochs or steps
        if config.training.total_num.epochs > 0:
            self.training_in_epoch = True
        else:
            if config.training.total_num.update_steps < 0:
                raise NotImplementedError("config.training.total_num.updated_steps must be larger than 0")
            self.training_in_epoch = False

        if self.rank > 0:
            logging.shutdown()
            logger.handlers.clear()

        # correctly handles exception
        atexit.register(self.__del__)

    @handle_event(Events.INITIALIZE, priority=100)
    def report_init_config(self, trainer: Trainer):
        logger.info(OmegaConf.to_yaml(self.config))

    @handle_event(Events.TRAIN_BEGIN, priority=155)
    def setup_timer(self, trainer: Trainer):

        if self.rank == 0:
            self.last_log_time = time.time()
            self.epoch_start_time = time.time()
            # Info the start
            logger.info("Training Starts!")

            if trainer.global_step_count > 0:
                self.resume_training = True
                logger.info("Resume the training!")
            else:
                self.resume_training = False

        self.last_log_global_step = 0

    @handle_event(Events.TRAIN_BEGIN, priority=140)
    def setup_tensorboard(self, trainer: Trainer):
        # Setup tensorboard
        if self.rank == 0:
            log_dir = os.path.join(os.getcwd(), "tensorboard")
            os.makedirs(log_dir, exist_ok=True)
            self.tensorboard = SummaryWriter(log_dir=log_dir, purge_step=trainer.global_step_count)

    @handle_event(Events.STEP_BEGIN)
    def debug_step(self, trainer: Trainer):
        if self.rank == 0:
            # This is only for debug purposes
            log_string = ""
            for key, value in trainer.tmp_vars["log_dict"].items():
                log_string += f"- {key}: {value}"

            logger.debug(log_string)

    @handle_event(Events.BATCH_END)
    def on_batch_end(self, trainer: Trainer):
        if self.rank == 0:
            if len(trainer.replay_buffer) > 0:
                trainer.tmp_vars["log_dict"]["mean_reward"] = trainer.replay_buffer.reward_mean

            if self.resume_training:
                self.log(trainer, trainer.tmp_vars["log_dict"])
                self.resume_training = False
            elif self.log_in_seconds:
                current_time = time.time()
                iter_elapsed_time = current_time - self.last_log_time

                if iter_elapsed_time > self.config.training.logging.seconds_interval:
                    self.log(trainer, trainer.tmp_vars["log_dict"])
            else:
                if (trainer.global_step_count + 1) % self.config.training.logging.steps_interval == 0:
                    self.log(trainer, trainer.tmp_vars["log_dict"])

    @handle_event(Events.VALIDATE_BEGIN)
    def info_valid_begin(self, trainer: Trainer):
        if self.rank == 0:
            logger.info(f"Steps {trainer.global_step_count}: Validation Begins:")

    @handle_event(Events.VALIDATE_END)
    def show_metrics(self, trainer: Trainer):
        if self.rank == 0:
            for metric_name, value in trainer.tmp_vars["validate_metrics"].items():
                metric_name = metric_name[0].upper() + metric_name[1:]
                if not self.training_in_epoch:
                    logger.info(f"Steps {trainer.global_step_count}: Validation {metric_name} {value:4.4f}")
                else:
                    logger.info(f"Epoch {trainer.epochs_trained + 1}: Validation {metric_name} {value:4.4f}")

                # tensorboard
                if isinstance(value, float):
                    self.tensorboard.add_scalar("validate/" + metric_name, value, global_step=trainer.global_step_count)

    def log(self, trainer: Trainer, log_dict: Dict[str, float]):
        """
        Args:
            trainer: Trainer class
            log_dict: Dict
        """
        updated_steps = trainer.global_step_count
        if trainer.epoch_num_batches is not None:
            if not self.training_in_epoch:
                percent = 100. * trainer.global_step_count / trainer.total_num_update_steps
            else:
                percent = 100. * trainer.epoch_step_count / (trainer.epoch_num_batches // trainer.gradient_accumulation_batches)

        iter_elapsed_time = time.time() - self.last_log_time
        elapsed_steps = trainer.global_step_count - self.last_log_global_step

        speed = elapsed_steps * trainer.ppo_buffer_size * self.config.training.num_gpus_per_node / iter_elapsed_time
        self.cumulative_time += iter_elapsed_time

        if not self.training_in_epoch:
            log_string = (f"Train Steps - {updated_steps:<6} - " f"[{percent:7.4f}%] - " f"Speed: {speed:4.1f} - ")
        else:
            log_string = (
                f"Train Epoch: [{trainer.epochs_trained + 1}/{self.config.training.total_num.epochs}] - "
                f"Steps: {updated_steps:<6} - "
                f"[{percent:7.4f}%] - "
                f"Speed: {speed:4.1f} - "
            )

        # Smooth values in the log_dict
        for key, value in log_dict.items():
            if key in self.history_log_dict:
                if not isinstance(self.history_log_dict[key], float):
                    logger.error(f"{key} is not a float. Only float can be logged!")
                    raise NotImplementedError(f"{key} is not a float. Only float can be logged!")
                self.history_log_dict[key
                                     ] = self.history_log_dict[key] * self.smooth_coef + value * (1 - self.smooth_coef)
            else:
                self.history_log_dict[key] = value

            self.tensorboard.add_scalar(f"train/{key.lower()}", value, trainer.global_step_count + 1)

        # Add to Logging
        for key, value in self.history_log_dict.items():
            if key[0] != '_':
                log_string = log_string + f"{key[0].upper() + key[1:]}: {value:7.4f} - "

        log_string = log_string.strip(" - ")

        logger.info(log_string)

        self.tensorboard.add_scalar("train/speed", speed, trainer.global_step_count + 1)
        self.tensorboard.add_scalar("train/cumulative_time", self.cumulative_time, trainer.global_step_count + 1)

        self.last_log_time = time.time()
        self.last_log_global_step = trainer.global_step_count

    def __del__(self):
        if self.rank == 0:
            logging.shutdown()
            logger.handlers.clear()

            if self.tensorboard:
                self.tensorboard.close()

    def state_dict(self):
        state_dict = {"cumulative_time": self.cumulative_time, "history_log_dict": self.history_log_dict}
        return state_dict

    def load_state_dict(self, state_dict):
        self.cumulative_time = state_dict["cumulative_time"]
        self.history_log_dict = state_dict["history_log_dict"]